## Summary

Adds `fla/ops/decay_max_scan/`, a new operator implementing an input-dependent
decay running-max scan with a strictly O(1) recurrent state:
`e_t = max(x_t, g_t * e_{t-1})` with `g_t = exp(-softplus(x_t * v_d + b_d))`.
Unlike cummax (monotone, saturates) and matmul-based linear attention, the
state can decrease — the gate implements selective forgetting of the running
peak while keeping exact retrieval of the decay-weighted maximum. Design
discussion: #1705.

One complete operator: `naive.py` sequential reference (varlen via
`cu_seqlens`), `chunk.py` dispatch + vectorized torch evaluation via an
associative `(max, *)` semiring doubling scan (no `cumprod` division — the
cumprod/cummax formulation underflows fp32 for adversarial `g` and poisons the
cummax through `x / pref`; derivation in the op `README.md`),
`triton_backend.py` Triton kernels (block-local reverse scan producing
per-chunk carries, carry-composition forward sweep writing outputs + the
argmax-owner chain, reverse-BPTT backward over per-position coefficients
recomputed in torch — no coefficient tensors saved for backward), and
`fused_recurrent.py` single-step decode.

## Test plan

`tests/ops/test_decay_max_scan.py` follows the `test_gla`/`test_kda` naming
and `assert_close` patterns: forward / final-state / gradients (`dx`, `dv`,
`db`) vs `naive.py` across dense shapes incl. non-power-of-2 lengths (63, 200,
500, 1000) and fp16; varlen packing via `cu_seqlens` vs per-sequence naive;
`fused_recurrent` single-step decode equivalence over 50 steps. **8 passed** on
Windows (torch 2.6.0+cu124 / triton 3.2.0) and on Linux WSL2 (torch
2.11.0+cu128 / triton 3.6.0 — exceeding the torch >= 2.7, triton >= 3.3
prerequisites), both `FLA_CI_ENV=0` strict mode, run in-repo against this
branch. Hardware: RTX 4090 D 24GB.

Dependent tests: `python scripts/find_dependent_tests.py` on the changed files
returns this PR's new test file plus `tests/benchmarks/test_graph_benchmark.py`
(via the registry import) — the latter passes 15/15 with the new registry entry.
`pre-commit` ruff check on the added files: all checks passed.

End-to-end evidence: the operator was trained at 101M params / 488M tokens as
one channel of a hybrid architecture (alongside chunked gated delta rule +
short conv); under an equal-budget protocol it outperforms a same-size
Transformer at all tested lengths (64-2048, up to -75% PPL) and Gated DeltaNet
on 5/7 lengths while keeping the O(1) state. Protocol and logs: #1705 and
https://github.com/dfytensor/OpenAshAnyWhere (REPORT_v35.md).

## Benchmark / NCU

Registered as `chunk_decay_max_scan` in `benchmarks/ops/registry.py`. Same-
hardware spot numbers from the attached standalone benchmark
(`bench_decay_max_scan.py`, RTX 4090 D):

| Workload (B, T, D, H; fp32; dense) | Throughput | Triton | torch semiring | naive |
| --- | --- | --- | --- | --- |
| (8, 2048, 96, 8) | Mtok/s | 21.5 | 1.4 | 0.2 |
| (32, 384, 96, 8) | Mtok/s | 17.9 | 3.8 | 0.8 |

varlen (packed 4x512 in T=2048, single row) routes to the torch path and
measures 12.4 ms for 16384 tokens; the unified-driver `benchmarks/ops/run`
sweep (dense + varlen) will be run on this branch once the registry entry is
confirmed compatible with the driver.

## Breaking changes

None (new operator only).

## Checklist

- [x] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [x] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [x] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.

### Large PR justification

One complete operator implementation — reference, chunked evaluation, Triton
backend, fused decode, tests, benchmark registration, and docs are cohesive
and were reviewed together per the "Adding a New Operator" checklist; splitting
the reference/tests from the kernels would leave the PR untestable mid-stack.
Review order: `naive.py` (semantics), `chunk.py` (semiring scan), 
`triton_backend.py` (kernels), `tests/`, then registry/docs.

## Known gaps (honest)

- Triton kernel address arithmetic is int32; the int64 sweep per kernel
  conventions is pending (tested shapes stay well inside int32).
- varlen (`cu_seqlens`) routes to the torch path; a Triton varlen variant is
  follow-up work.
- Hopper/Blackwell untested (development hardware is Ada).
