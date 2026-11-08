# decay_max_scan

Input-dependent decay running-max scan with an O(1) recurrent state.

## Recurrence

Per lane (b, d, h), with input `x_t` and per-channel decay parameters `v_d, b_d`:

```
g_t = exp(-softplus(x_t * v_d + b_d))    # input-dependent decay, g_t in (0, 1)
e_t = max(x_t, g_t * e_{t-1}),           # e_0 = -1e30 (cold start)
```

`e_t` is the decay-weighted running maximum: `e_t = max_{s<=t} x_s * prod_{r=s+1..t} g_r`.
Cold start uses `-1e30` instead of `-inf` so the `d * state` gradient term stays finite.

## Chunked evaluation (chunk.py)

`max(x_s * prod_{r=s+1..t} g_r)` over `s <= t` is a scan under the associative
semiring `(max, *)` with pairs `(m, d)` and

```
combine((m_l, d_l), (m_r, d_r)) = (max(m_r, d_r * m_l), d_r * d_l)
```

so a Hillis-Steele doubling scan (log2(CHUNK) vectorized steps) evaluates each
chunk without `cumprod` division: the naive `cumprod/cummax` formulation
underflows when `prod(g)` reaches fp32 subnormals (e.g. g ~ 0.5 over 128 steps),
which poisons the cummax through `x / pref`. The doubling form only clamps the
decay product at `1e-30`, where the decayed history term is negligible for
bounded inputs.

## Gradients

Where the input branch wins (`x_t > g_t * e_{t-1}`), gradients flow into `x_t`;
otherwise they flow to `e_{t-1}` scaled by `g_t`, plus the decay-parameter
terms through `g_t = exp(-softplus(x_t * v_d + b_d))` (sigmoid factor). The
chunked scan is differentiable end to end; `tests/ops/test_decay_max_scan.py`
checks outputs, final states, and `dx/dv/db` gradients against `naive.py`.

## Decoding

`fused_recurrent_decay_max_scan` performs the single-token update; the state is
one `[B, D, H]` tensor — strictly O(1) per lane.
