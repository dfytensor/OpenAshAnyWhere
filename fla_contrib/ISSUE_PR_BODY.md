## Summary

Adds `fla/ops/decay_max_scan/`, a new operator implementing an input-dependent
decay running-max scan with a strictly O(1) recurrent state:
`e_t = max(x_t, g_t * e_{t-1})` with `g_t = exp(-softplus(x_t * v_d + b_d))`.
Unlike cummax (monotone, saturates) and matmul-based linear attention, the
state can decrease — the gate implements selective forgetting of the running
peak while keeping exact retrieval of the decay-weighted maximum. Design
discussion: #1705.

Contents (one complete operator: reference + chunked + Triton backend + tests +
registration + docs):

- `naive.py` — sequential pure-torch reference, varlen via `cu_seqlens`
- `chunk.py` — dispatch + vectorized torch evaluation via an associative
  `(max, *)` semiring Hillis-Steele doubling scan. No `cumprod` division: the
  cumprod/cummax formulation underflows fp32 for adversarial `g` (g ~ 0.5 over
  128 steps) and poisons the cummax through `x / pref` — derivation in the op
  `README.md`
- `triton_backend.py` — Triton kernels: block-local reverse scan producing
  per-chunk carries `(g, m, owner)`, carry-composition forward sweep writing
  outputs + argmax-owner chain, and a reverse-BPTT backward over per-position
  coefficients recomputed in torch (no coefficient tensors saved for backward);
  `fused_recurrent`-style O(1) single-step semantics across train/decode
- `fused_recurrent.py` — single-step decode update

## Test plan

`tests/ops/test_decay_max_scan.py` (follows the `test_gla`/`test_kda` naming and
`assert_close` patterns):

- forward / final-state / gradients (`dx`, `dv`, `db`) vs `naive.py` across
  dense shapes incl. non-power-of-2 lengths (63, 200, 500, 1000) and fp16
- varlen packing via `cu_seqlens` vs per-sequence naive
- `fused_recurrent` single-step decode equivalence over 50 steps

Results: **8 passed** on Windows (torch 2.6.0+cu124 / triton 3.2.0) and on
Linux WSL2 (torch 2.11.0+cu128 / triton 3.6.0 — exceeding the torch >= 2.7,
triton >= 3.3 prerequisites), both with `FLA_CI_ENV=0` strict mode, run
in-repo against this branch. Hardware: RTX 4090 D 24GB.

End-to-end evidence: the operator was trained at 101M params / 488M tokens as
one channel of a hybrid architecture (alongside chunked gated delta rule +
short conv); under an equal-budget protocol it outperforms a same-size
Transformer at all tested lengths (64-2048, up to -75% PPL) and Gated DeltaNet
on 5/7 lengths while keeping the O(1) state. Protocol and logs: #1705 and
https://github.com/dfytensor/OpenAshAnyWhere (REPORT_v35.md).

## Benchmark / NCU

Registered as `chunk_decay_max_scan` in `benchmarks/ops/registry.py`. The
op-level `benchmarks/ops/run` sweep will be run on the PR branch once the
registry registration is confirmed compatible with the unified driver; same-
hardware spot numbers from the attached standalone benchmark
(`bench_decay_max_scan.py`, RTX 4090 D):

| Workload (B, T, D, H; fp32; dense) | Throughput | Triton | torch semiring | naive |
| --- | --- | --- | --- | --- |
| (8, 2048, 96, 8) | Mtok/s | 21.5 | 1.4 | 0.2 |
| (32, 384, 96, 8) | Mtok/s | 17.9 | 3.8 | 0.8 |

## Breaking changes

None (new operator only).

## Known gaps (honest)

- Triton kernel address arithmetic is int32; the int64 sweep per kernel
  conventions is pending (tested shapes stay well inside int32).
- varlen (`cu_seqlens`) routes to the torch path; a Triton varlen variant is
  follow-up work.
- Hopper/Blackwell untested (development hardware is Ada).
- Large-PR justification: one complete operator implementation — reference,
  chunked evaluation, Triton backend, fused decode, tests, benchmark
  registration, and docs are cohesive and were reviewed together per the
  "Adding a New Operator" checklist.
