# [Ops] New operator: decay_max_scan — input-dependent decay running-max scan with O(1) state

## Proposal

Add `fla/ops/decay_max_scan/`, a linear-complexity scan operator implementing an
**input-dependent decay running maximum** with a strictly O(1) recurrent state:

```
g_t = exp(-softplus(x_t * v_d + b_d))      # per-lane decay in (0, 1), driven by the input
e_t = max(x_t, g_t * e_{t-1})              # state: one [B, D, H] tensor
```

Unlike cummax (monotone, saturates) and unlike matmul-based linear attention,
the state **can decrease** — the gate `g_t < 1` implements selective forgetting
of the running peak while keeping exact retrieval of the decay-weighted
maximum. The pair (max, *) forms an associative semiring, so the op admits the
standard chunked/parallel treatment (block carries composed as
`compose((g1,m1),(g2,m2)) = (g1*g2, max(m2, g2*m1))`), a fused O(1) decoding
kernel, and a reverse-BPTT backward.

## Why this fits FLA

- Same contract shape as existing ops: tensor in → (output, final_state) out,
  `initial_state`/`final_state` support, differentiable, decode kernel.
- Trained end-to-end at 101M params / 488M tokens as one channel of a hybrid
  architecture (alongside chunked gated delta rule + short conv): under an
  equal-budget protocol it outperforms a same-size Transformer baseline at all
  tested lengths (64-2048, up to -75% PPL) and Gated DeltaNet on 5/7 lengths
  (-5.4% PPL @384), while keeping the O(1) state. Full protocol, logs, and
  training curves: https://github.com/dfytensor/OpenAshAnyWhere (REPORT_v35.md).

## Package contents (this drop-in tree)

- `naive.py` — sequential pure-torch reference (varlen via `cu_seqlens`)
- `chunk.py` — vectorized torch implementation: associative (max, *) semiring
  doubling scan (no `cumprod` division; the cumprod/cummax formulation underflows
  fp32 for adversarial g and poisons the cummax) + Triton dispatch
- `triton_backend.py` — Triton kernels: block-local reverse scan producing per-chunk
  carries, carry-composition forward sweep writing output + argmax-owner chain, and
  a reverse-BPTT backward over coefficients recomputed in torch (no saved coefficient
  tensors); O(1) state across training and decoding
- `fused_recurrent.py` — single-step decode update
- `tests/ops/test_decay_max_scan.py` — forward/state/gradient checks vs `naive.py`
  (dense, varlen, fp16, non-power-of-2 lengths) and decode equivalence — 8 passed
- `benchmarks/bench_decay_max_scan.py` — measured on RTX 3090/Windows (488M-token
  training runs were on the same stack): Triton 18-21 Mtok/s, torch semiring
  0.8-3.7, sequential naive 0.13-0.5

## Known gaps before merge

1. Address arithmetic inside the Triton kernels is int32; needs the int64 sweep
   per kernel conventions (shapes up to 96*2048*8 stay well inside int32).
2. varlen (`cu_seqlens`) currently routes to the torch path; a Triton varlen
   variant is follow-up work.
3. Kernels were developed/validated on Windows/Triton (where a "loop-exit
   accumulator store" miscompilation was observed and avoided by per-step
   stores); re-validation on Linux/Hopper is required.
4. Model integration (HF `PreTrainedModel` + cache class) is a separate
   follow-up PR.
