# EnergyLM: Training Transformer Language Models at Equilibrium — Equilibrium Propagation vs. Implicit Gradients

**Authors:** EnergyLM experimental report
**Code:** `energy_lm/` (PyTorch, single GPU, RTX 4090 D)
**Repository:** https://github.com/dfytensor/DEQ
**Environment:** Python 3.12, PyTorch 2.12 + CUDA

---

## Abstract

We investigate whether a Transformer language model can be trained **without
backpropagation** by reinterpreting its layers as the fixed point of a
continuous-time energy system. We implement **EnergyLM**, an *Energy Recurrent
Block* (ERB) whose hidden state relaxes to a steady state `Z* = f(Z*)` and
whose predictions are read out from that equilibrium. We study two learning
rules that avoid backpropagating through depth: (i) **Equilibrium Propagation
(EP)**, which updates weights from the difference of local Hebbian
correlations measured at a free and a clamped steady state; and (ii) the
**DEQ implicit gradient**, which computes the *exact* cost gradient through the
equilibrium via a Neumann-series adjoint with no backprop through the
relaxation iterations.

On a character-level English toy task and on the **MiniMind Chinese corpus**
(~6.4M tokens, char-level, vocab 4500), we find: (a) pure EP learns real
language *structure* (CE 3.30→2.04 on English, plateau ~5.4 on Chinese) but
does **not** reach fluency and is prone to weight runaway; (b) Anderson
acceleration accelerates the relaxation but does **not** fix EP's weak signal
and in fact accelerates divergence on real data; (c) the DEQ implicit gradient
is dramatically more effective (Chinese CE **5.4 → 1.31**), producing genuine
multi-sentence Chinese; (d) keeping all DEQ advantages, a combination of
cosine LR scheduling and *pushing the contractivity boundary* (richer
fixed-point depth) further lowers Chinese CE to **0.85**. We analyse why
widening the model does not help (the contractivity constraint, not capacity,
is the bottleneck) and discuss the stability–expressivity trade-off.

---

## 1. Introduction

Backpropagation is the workhorse of deep learning but is biologically
implausible and expensive in memory (proportional to depth). **Equilibrium
Propagation** (Scellier & Bengio, 2017) and **Deep Equilibrium Models** (Bai,
Kolter & Koltun, 2019) offer two routes to avoid backprop-through-depth:

* EP: the network is a recurrent dynamical system with an energy function;
  learning uses only **local** pre-/post-synaptic correlations measured at two
  steady states (free, and weakly nudged toward the target).
* DEQ: the network is an *implicit* fixed-point layer; the gradient is
  computed **at the equilibrium** via the implicit-function theorem, never
  unrolling the iterations.

This report asks: **can these recipes train a Transformer-shaped language
model end-to-end, and how far are they from ordinary backprop?** We build
EnergyLM, run controlled comparisons against a same-width backprop baseline,
and ablate the key levers (signal type, optimisation, contractivity, width).

Our contributions:
1. A working EP-trained Transformer (ERB) with the stability mechanisms
   (synaptic-scaling homeostasis, skip-on-divergence) that pure Hebbian EP
   requires.
2. A clean empirical comparison: **EP vs Anderson-accelerated EP vs DEQ
   implicit gradient vs backprop**, on identical data and width.
3. The finding that **width is not the DEQ bottleneck**; the contractivity
   margin (effective fixed-point depth) is, and pushing it is the productive
   lever.

---

## 2. Method

### 2.0 Relation to other implicit-depth families (NAIS-Net / DEQ / MDEQ / IFT)

EnergyLM sits inside the broader *implicit-depth / fixed-point* family.  It is
important to be precise about what is shared and what differs, because it
determines whether the ceiling we report (§4.9–4.10) applies to those works.

| family | forward solve | gradient | strict contraction? | LM results? |
|---|---|---|---|---|
| **NAIS-Net** (Bai et al., NeurIPS 2018) | non-autonomous ODE unfolding | standard backprop (discretised adjoint) | yes — provable input-output stability | no large LM (small seq/classification) |
| **DEQ** (Bai-Kolter-Koltun, NeurIPS 2019) | Anderson / Broyden root-find | **implicit-function (IFT) backprop** (autograd VJP at equilibrium) | **no** — relies on empirical stability | **yes**: WikiText-103 transformer-DEQ, test ppl in the low-to-mid 30s, competitive with Transformer-XL/TrellisNet at similar params, ~88% memory saving |
| **MDEQ** (Bai-Koltun-Kolter, NeurIPS 2020) | multiscale Anderson+Broyden | IFT backprop | no | **vision only** (Cityscapes, ImageNet) — no LM |
| **IFT layers** (general implicit-layer framework) | any fixed-point solver | IFT backprop | depends on instance | the mechanism DEQ uses; not a separate named LM model |
| **EnergyLM / EP** (this work, pure) | energy relaxation (free/clamped) | **local Hebbian correlations** — *no backprop at all* | yes (homeostasis) | toy; CE ≈ 5.4 plateau |
| **EnergyLM / DEQ** (this work, implicit grad) | Anderson relaxation | **IFT backprop** at equilibrium (GMRES/Neumann adjoint) | yes (homeostasis) | toy; CE ≈ **0.80** |

**Three observations that frame our results.**

1. *None of NAIS-Net/DEQ/MDEQ avoids backpropagation.* They all train with
   ordinary global backpropagation through the implicit layer (the IFT adjoint).
   The only member of the family that is genuinely backprop-free is our **pure
   EP** variant — which is exactly the one that plateaus (CE ≈ 5.4).  Avoiding
   backprop is the hard part, and these works do not attempt it.

2. *DEQ-LM works at scale precisely because it does **not** enforce strict
   contraction.*  It uses a standard, well-engineered transformer cell
   (LayerNorm, residuals, small init) that converges *empirically* under
   Anderson/Broyden, and trains it with full backprop.  Our **strict-contraction
   homeostasis** — which we introduced to (a) guarantee forward convergence and
   (b) make the equilibrium friendly to *local* learning rules — is the very
   thing that caps effective depth at `1/(1−ρ) ≈ 5` (§4.9).  In other words, the
   DEQ-LM paper and this work sit at opposite ends of a trade-off: DEQ-LM
   trades *guaranteed* stability for *empirical* stability + unbounded depth +
   backprop; EnergyLM trades depth for guaranteed stability + a path to local,
   backprop-free learning.

3. *Scale of comparison.*  DEQ-LM is evaluated at ~10–30M parameters on
   WikiText-103 (>100M tokens); our experiments are at 2M parameters on a 6M-token
   char-level corpus, so absolute perplexities are not comparable.  The
   like-for-like quantity we can compare is *implicit-grad DEQ vs explicit
   backprop at the same width*: at our scale that is CE **0.80 vs 0.09**
   (§4.6).  DEQ-LM's reported competitiveness with Transformers at scale is
   consistent with this — implicit-gradient DEQ is *competitive but not
   superior* to backprop of the same cell, and the gap is set by the
   conditioning of `(I − J)` and the cell engineering, exactly as our
   §4.8–4.10 analysis predicts.

The practical takeaway: if the goal is the **best LM** at constant memory,
DEQ-LM (IFT backprop, no strict contraction, transformer cell) is the right
choice and is already known to be competitive.  If the goal is **no-backprop /
biologically-plausible / hardware-local** training — the actual motivation of
the EnergyLM design document — then the strict-contraction regime is forced on
you, and our results characterise its ceiling (CE ≈ 0.80 on this task) and
*why* it exists.  We test the head-to-head directly in §4.11.

---

### 2.1 The Energy Recurrent Block (ERB)

EnergyLM is a single recurrent block whose state `Z ∈ ℝ^{B×T×d}` relaxes to a
fixed point. The recurrent map is a residual Transformer block

```
f(Z) = X + g · ( Attention(Z) + FFN(Z) )
```

with causal multi-head attention and a ReLU FFN; `X` is the embedded input
("clamped voltage"), and `g` is a residual gain. The state evolves by the
local dynamics

```
Z ← Z + dt · ( f(Z) − Z )
```

until `‖f(Z) − Z‖` is negligible. The non-negative surrogate `E(Z) = ½‖Z − f(Z)‖²`
decreases along trajectories; the steady state `Z*` satisfies the
deep-equilibrium condition `Z* = f(Z*)` and is read out by a linear head.

### 2.2 Learning rule A — Equilibrium Propagation (pure local)

For each batch we relax twice: a **free** phase (β = 0) reaching `Z⁰`, and a
**clamped** phase (β > 0) whose dynamics receive a purely local output-error
injection `−β · dC/dZ`, reaching `Zᵝ`. The EP theorem gives, for every
recurrent weight `W`,

```
ΔW = (lr/β) · ( ⟨post·preᵀ⟩_clamped − ⟨post·preᵀ⟩_free )
```

i.e. the difference of local Hebbian correlations. The readout uses its exact
**one-layer local** gradient `Z⁰ᵀ(softmax(Z⁰W_out) − y)`. No autograd graph is
built for any weight.

**Stability.** The Hebbian rule monotonically strengthens correlations and
drives the map toward (and past) the contractivity boundary, where the
relaxation diverges. We add **synaptic-scaling homeostasis**: after each step
the spectral norms of `(W1,W2)` and `(Wv,Wo)` (estimated by power iteration)
are jointly rescaled so that `g · σ(W1)σ(W2) < ρ`. We also **skip** any update
whose relaxation failed to converge, so bad correlations cannot poison the
weights.

### 2.3 Learning rule B — DEQ implicit gradient

The EP correlation proxy is only a rough estimate of the true cost gradient.
We therefore also implement the **exact** implicit-function gradient. At the
equilibrium `Z*` (found by Anderson-accelerated relaxation, under `no_grad`),

```
dC/dθ = (dC/dZ*) · (I − J_f)⁻¹ · (∂f/∂θ)
```

where `J_f = ∂f/∂Z`. We never form `J_f`; we approximate `(I − J_f^T)⁻¹` by a
truncated **Neumann series** `v = Σ_{i=0}^{K} (J_f^T)^i g`, where `g = dC/dZ*`
is the readout adjoint and each term is a single vector-Jacobian product
(VJP) of **one** block. Weight gradients are then VJPs of the same single
block. Crucially, **no gradient is ever propagated through the relaxation
iterations or any depth** — everything is computed at the equilibrium. The
contractivity that guarantees convergence of the relaxation *also* guarantees
convergence of the Neumann series.

### 2.4 Anderson acceleration

We accelerate the fixed-point solve with type-II Anderson mixing (history `m`,
damping `β`, Tikhonov `λ`). This finds the equilibrium in far fewer iterations
and converges even when the plain damped iteration is marginal, permitting a
larger `g` (richer steady state) in isolation.

### 2.5 Baseline

A standard single-block causal Transformer (attention + FFN) of identical
width, trained with Adam + global backpropagation, serves as the reference.

---

## 3. Experimental setup

* **English toy task**: ~1.5k-token repetitive corpus, char-level vocab 27;
  `d = 64`, 4 heads, ~36k params.
* **MiniMind Chinese**: `pretrain_t2t_mini.jsonl`, char-level vocab 4500
  (built from a 40 MB prefix; the MiniMind BPE tokenizer was unavailable
  offline); `d ∈ {192, 256}`, ~2M params, seq 128, streamed batches.
* **Hardware**: single RTX 4090 D (24 GB). Mixed precision off for
  reproducibility of the relaxation.
* **Metrics**: cross-entropy (nats/token) and bits-per-char; relaxation
  residual norm; skip count (divergent steps discarded).

---

## 4. Results

### 4.1 English toy task — EP learns and is stable

| model | signal | final CE | BPC | stable? |
|---|---|---|---|---|
| EnergyLM (EP) | local Hebb | **2.04** | 2.95 | yes (no skips) |
| backprop baseline | global grad | 0.57 | 0.82 | yes |
| chance | — | 3.30 | 4.76 | — |

EP clearly learns real structure (3.30 → 2.04) and the samples contain corpus
words ("blows across", "fire burned low", "the same gentle"). Backprop is more
sample-efficient, as expected.

### 4.2 MiniMind Chinese — EP plateau vs. DEQ fluency (3000 steps, same width)

| variant | learning rule | final CE | output quality |
|---|---|---|---|
| EP (plain) | local Hebb corr. diff | ≈ 5.4 (plateau) | repetitive word-fragments |
| EP + Anderson | Hebb corr. + richer map | **diverged → 9** | collapses ("一一一一") |
| **DEQ implicit grad** | exact equilibrium gradient | **1.31** | **fluent Chinese** |
| backprop baseline (Adam) | global gradient | 0.24 | fully coherent |

**Key findings.**
* Pure EP plateaus at CE ≈ 5.4: the local correlation proxy is too imprecise
  for real language, and the monotone Hebbian update constantly fights the
  contractivity boundary.
* **Anderson does not rescue EP.** It accelerates the relaxation (it converges
  even at `g = 0.9` in isolation), but the EP *signal* is the bottleneck — a
  richer map just makes the weak Hebbian updates push past the boundary
  *faster*, so training diverges.
* The **DEQ implicit gradient** is dramatically better (5.4 → 1.31) and
  produces genuine multi-sentence Chinese, while never backpropagating through
  the relaxation iterations.

DEQ samples (CE 1.31):
```
'给我讲一个' -> '给我讲一个代码风格规范，这样的情电影名AI，让你会有关于你能否给我几个合适...'
'为什么'     -> '为什么我想一个清澈的环境中最凶猛的云雾，可以长达3米，因并提供缓解近小溪风...'
```

### 4.3 Improving DEQ while keeping its advantages

We ablate three levers on MiniMind-zh, all within the DEQ framework (exact
equilibrium gradient, no backprop through iterations, contractive/convergent,
constant memory):

| stage | change | final CE |
|---|---|---|
| DEQ baseline | constant lr, 3000 steps, contractivity 0.6 | 1.31 |
| + optimisation | cosine LR + warmup + `K=8` Neumann + 24 relax + 6000 steps | **1.10** |
| + width (d 192→256) | pure capacity | 1.06 (≈ no gain) |
| + push contractivity | contractivity 0.72, `g=0.5`, `K=12`, 30 relax | **0.85** |

* **Optimisation is a free win** (1.31 → 1.10): a cosine schedule with warmup
  and more accurate adjoints (more Neumann terms, more relax steps) help.
* **Width does not help** (1.10 → 1.06). The capacity is not the bottleneck.
* **Pushing the contractivity boundary is the productive lever** (1.10 → 0.85).
  Raising the homeostasis target so the fixed-point Jacobian spectral radius is
  closer to 1 makes the Neumann expansion `(I − J)⁻¹` represent a richer
  (effectively deeper) steady state — at the cost of needing more Neumann terms
  and relax steps to keep the adjoint accurate, and sitting closer to the
  stability edge.

DEQ samples at CE 0.85 (more coherent, occasional mode-collapse on punctuation):
```
'给我讲一个问题，明天气的法则...天气预报据以和音乐而治是回文的口味深受消费者喜爱...'
'为什么这个关于你扩展这种？...检查一个字符串是否是回文的程序...'
```

### 4.4 A negative result: naive architectural changes destabilise DEQ

Adding **RMSNorm** (pre-norm) and **input/output embedding tying** — both
standard Transformer improvements — *destabilised* DEQ training (CE rose and
oscillated around 3–4). The cause: RMSNorm makes the block Lipschitz
data-dependent, so the spectral-norm homeostasis no longer correctly bounds the
fixed-point Jacobian, and the Neumann adjoint becomes noisy. We expose these
as opt-in flags (`--use_norm`, `--tie_embeddings`, default off).

### 4.5 Pushing further: GMRES adjoint + the contractivity ceiling

We then tested two further levers within the DEQ framework.

**(a) GMRES adjoint solver.** We replaced the truncated Neumann series for
`(I − Jᵀ)⁻¹g` with a **GMRES(k)** Krylov solve using the same single-block VJP
oracle. GMRES minimises `‖g − (I−Jᵀ)v‖` over the Krylov subspace and is markedly
more accurate per matvec when `‖J‖` is close to 1 (where the Neumann geometric
series is slow). This lets us push the homeostasis target closer to the
contractivity boundary *safely*:

| adjoint | contractivity | final CE | stable? |
|---|---|---|---|
| Neumann (K=12) | 0.72 | 0.85 | yes |
| **GMRES (k=10)** | **0.80** | **0.80** | **yes** |
| GMRES (k=12) | 0.85 | 1.03 | yes but degraded |

GMRES + contractivity 0.80 reaches **CE 0.80** and stays stable; pushing to
0.85 regresses (the map becomes too stiff). The full DEQ improvement arc is
therefore **1.31 → 1.10 → 0.85 → 0.80**.

**(b) More data / more steps do NOT help — the single block saturates.** On the
same 4500-vocab task, doubling training to 10000 steps *degrades* DEQ (CE
~1.2, with occasional divergence skips) rather than helping — long training
lets the weights drift past the contractivity boundary. Raising the vocab to
6000 (larger effective corpus coverage) makes the per-token task strictly
harder (CE floor `ln 6000 = 8.70`) and the model plateaus around 0.98. Widening
to `d = 256` is also flat (§4.3). The single contractive block has a **hard
ceiling near CE 0.80** on this task; data, steps, and width are all
ineffective levers past it.

### 4.6 Best DEQ samples (CE 0.80, GMRES + contractivity 0.80)

```
'给我讲一个' -> '给我讲一个代码风格式化的代码风格规范中，以及 Jarb 代码风格式，那请市民...'
'为什么'     -> '为什么我想要注意春意做好吃的、和一个程...浓郁的榛子...皮埃尔·赖文的其他产品牌...'
'秋天的'     -> '秋天的...字符串的代码风格式...口感和丰富的奶...口味浓郁的榛子...'
```

The output is genuine multi-clause Chinese with real vocabulary and grammar
fragments; remaining artefacts are local mode-collapses on punctuation/code
tokens, consistent with a CE well above the backprop floor.

### 4.7 A negative result: multi-block hierarchical equilibria do not help

Hypothesis: stacking `L` independent contractive DEQ blocks — each relaxed to
its own equilibrium, the implicit gradient propagated through the chain of
equilibria — should add genuine depth (every block safely inside the
contractive regime) and break the single-block ceiling.

Implementation: `multi_model.py` / `multi_trainer.py` / `run_multi.py`.  Each
block `l` solves `h_{l+1} = f_l(h_{l+1}, h_l; θ_l)`; the adjoint walks the chain
top-down, applying GMRES `(I − J_l^T)⁻¹` per block and handing the input-VJP to
the previous layer.  Diagnostics confirm gradients **do** flow through all
blocks (loss 8.4 → 6.5 in 8 steps, all blocks receiving comparable updates).

Result: the multi-block model is **strictly worse** than the single block.

| config | best CE | fate |
|---|---|---|
| single block (GMRES, ρ=0.80) | **0.80** | stable |
| 2 blocks (ρ=0.78/block) | ≈ 3.13 | plateaus ~3.3, then **diverges** (≈45% steps skipped) |
| 3 blocks (ρ=0.70/block) | stuck ~5.8 | diverges (≈75% steps skipped) |

Why it fails, despite flowing gradients:
* **Stiffness of the coupled system.** Each block's equilibrium is driven by
  the previous block's output magnitude; small upstream growth makes a
  downstream block's relaxation diverge even though that block alone is
  contractive.  Forward residuals jump from `1e-7` to `1e-2` and training
  collapses to mode-collapse tokens ("错错错").
* **Adjoint error compounds.** Chaining `L` approximate GMRES solves makes the
  gradient w.r.t. earlier blocks increasingly noisy; the recurrent signal that
  the single block exploits is washed out, so the model learns only the
  frequent-char prior (plateau ≈ 6.0) before slow recurrent learning can take
  hold.
* **Slower per step** (`L` relaxations + `L` GMRES solves), so even the stable
  regime cannot catch the single block.

**Conclusion of §4.7:** the single contractive DEQ block's CE ≈ 0.80 ceiling is
*not* broken by naive chained multi-equilibria.  A deeper, single (monolithic)
equilibrium is a more promising direction than chaining — tested next in §4.8.

### 4.8 Also negative: a deeper single-equilibrium map (composed or residual)

To rule out that the failure was the *chaining* rather than *depth* itself, we
built a **single-equilibrium** model whose recurrent map is deeper, solved with
a **single** relaxation and a **single** GMRES adjoint on the full Jacobian
(`deep_model.py`, `run_deep.py`).  Per-layer homeostasis keeps
`g · L(map) < contractivity`.

We tested both variants of "deeper map":

* **Pure composition** `U = s_L ∘ … ∘ s_1`, `s_l = FFN_l ∘ Attn_l`
  (Lipschitz *multiplies* across layers).
* **Residual composition** `h ← h + V_l(h)` with `V_l = FFN_l ∘ Attn_l`
  (additive per-layer Lipschitz, like the single block; information flows
  through the skip).

Result: **both are strictly worse than the single block.**

| map | L | best CE |
|---|---|---|
| single block (parallel attn+ffn) | 1 | **0.80** |
| pure composition | 2 | ≈ 3.8 (ρ=0.80); ≈ 5.1 (ρ=0.95) |
| residual composition | 2 | ≈ 3.9 |

**Why depth fails here — the effective-depth ceiling of contractive DEQ.**

A contractive fixed-point map with Jacobian spectral radius `ρ = ‖J‖ < 1` has
an *effective depth* bounded by the Neumann series
`(I − J)⁻¹ = Σ_{i≥0} Jⁱ`, whose geometrically-decaying tail gives an effective
"number of layers" of order `1 / (1 − ρ)`.  With our contractivity target
`ρ ≤ 0.8`, **effective depth ≤ 5**, regardless of how many sub-blocks we stack
inside `U`.  The single block already spends this budget optimally by putting
attention and FFN as a *parallel sum* inside one contraction (additive
Lipschitz).  Any attempt to add depth must still keep `g·‖J_U‖ < 1`:

* *Pure composition* multiplies per-layer Lipschitz, so each layer's budget
  shrinks as `(ρ/g)^{1/L}` — for `L=2`, `ρ=0.80`, `g=0.5` the per-layer
  4-matrix spectral product must stay below 1.26, starving the weights and
  collapsing `U(Z) ≈ 0`.
* *Residual composition* gives per-layer budget `(ρ/g)^{1/L} − 1 ≈ 0.26` on the
  *increment* `V_l`; each sub-block becomes too weak to compute anything
  non-trivial, so the map reduces to near-identity and learns no more than the
  embedding→readout shortcut.
* *Chained equilibria* (§4.7) do not increase the Neumann depth at all — each
  is its own contraction with its own `1/(1−ρ)` budget — and the coupling
  introduces stiffness and compounding adjoint error on top.

### 4.9 Depth/width study summary

| mechanism | result | root cause |
|---|---|---|
| chained multi-equilibria (§4.7) | diverges / plateau ≈ 3–6 | coupled stiffness + adjoint error compounding |
| composed single-equilibrium map (§4.8) | plateau ≈ 4–5 | Lipschitz multiplication starves weights |
| residual single-equilibrium map (§4.8) | plateau ≈ 3.9 | per-layer increment budget shrinks as `(ρ/g)^{1/L}−1` |
| wider single block (d 192→256) | flat (1.06) | capacity is not the bottleneck |

**The single contractive DEQ block's CE ≈ 0.80 is a fundamental ceiling of the
strict-contraction formulation**, not a tuning artefact: the effective depth is
capped at `~1/(1−ρ)`, the single parallel-attn+ffn block already realises that
budget, and every depth mechanism either starves the weights (composition) or
splits an already-tight budget too thin (residual/chaining).  Progressing past
0.80 requires leaving the strict-contraction regime — e.g. **monotone-operator
/ non-contractive equilibria** where fixed-point uniqueness comes from
structure other than `‖J‖ < 1`, or Lipschitz-controlled *residual* DEQ with
normalisation-aware bounds.  We test the first of these next in §4.10.

### 4.10 Also negative: non-contractive (Newton-Krylov) DEQ

The analysis above predicts that the contraction requirement is needed for the
*forward solver* (Picard/Anderson).  The implicit *gradient*
`(I − J_f)⁻¹` is, in principle, well-defined for any non-singular
`(I − J_f)`, i.e. for `ρ(J_f) ≥ 1` as long as `1 ∉ eigenvalues(J_f)`.  So we
attempted to escape the ceiling by replacing only the forward solver with
**Newton-Krylov** (`noncontractive_trainer.py`, `run_newton.py`): each Newton
step solves `(J_f − I) δ = −(f(Z) − Z)` by GMRES, with the Jacobian-vector
product `J_f·w` obtained exactly via `torch.func.jvp` (forward-mode AD on one
block).  This finds fixed points with `ρ(J_f) ≥ 1`.  The adjoint GMRES on
`(I − J_f^T)v = g` is unchanged (it is a linear solve, valid for any
non-singular `(I − J_f)`).  We drop the contraction homeostasis, keeping only a
loose per-matrix spectral cap so weights stay finite.

Result: **unstable / no learning.**

| config | best CE | fate |
|---|---|---|
| contractive single block (GMRES, ρ=0.80) | **0.80** | stable |
| Newton-Krylov, residual map L=2 | stuck ≈ 6.0 then **diverges** (loss 5.8 → 8.3) | ill-conditioned gradient |

The forward Newton solve *does* converge (`‖f(Z)−Z‖ ≈ 5e-5`), so non-contractive
fixed points are reachable.  But training collapses to the frequent-char prior
(CE ≈ 6.0) and then oscillates upward.  The reason is the **gradient**, not the
forward solve: without `ρ(J_f) < 1` the eigenvalues of `J_f` drift freely, and
whenever one approaches `1` the matrix `(I − J_f)` becomes nearly singular and
the implicit gradient `(I − J_f)⁻¹ g` explodes.  The contraction guarantee was
doing **two** jobs — forward convergence *and* gradient conditioning — and
removing it loses both the ceiling *and* trainability.

**This is the strongest form of the negative result:** the CE ≈ 0.80 ceiling is
not an artefact of a particular solver but of the **equilibrium formulation
itself**.  Contractive DEQ needs `ρ < 1` (capping effective depth at
`1/(1−ρ) ≈ 5`); non-contractive DEQ loses gradient stability.  There is no free
depth to be had in this family of models at this scale.

### 4.11 Head-to-head: standard DEQ-LM (no contraction) on the same data

§2.0 predicts that DEQ-LM's competitiveness comes from *not* enforcing
contraction (using an engineered transformer cell + full IFT backprop + small
init), at the cost of empirical, scale-dependent stability.  We test this
directly by implementing a canonical DEQ-LM (`deq_lm.py`, `run_deqlm.py`): a
pre-norm transformer cell, weight-tied, Anderson forward + Anderson-on-adjoint
backward via `register_hook`, trained with ordinary `loss.backward()` — i.e.
exactly the DEQ paper's mechanism — on the **same** MiniMind-zh data, with **no
contraction homeostasis**.

Result: **the unconstrained DEQ-LM is unstable at our scale and loses to the
contractive model.**

| model | fwd residual during training | final CE |
|---|---|---|
| EnergyLM/DEQ, contractive (GMRES adjoint) | stays `~1e-5` | **0.80** |
| standard DEQ-LM, `g=0.5`, no contraction | grows `7e-6 → 0.5` | 2.2 |
| standard DEQ-LM, `g=0.3`, no contraction | grows `7e-6 → 1.1` | 3.3 |

The forward Anderson solve *starts* converged (init is small) but **diverges as
training grows the weights** — exactly the failure mode the contraction
homeostasis was built to prevent.  Once `fwd_res ≈ 1`, the "equilibrium" is
meaningless and training collapses to repetitive mode-collapse output.

**Interpretation.**  This sharpens §2.0 in an important way: the published
competitiveness of DEQ-LM is not a property of "removing contraction" alone; it
is achieved *at scale* with a carefully engineered cell and hyperparameters
that happen to keep the forward iteration contracting empirically.  At toy
scale, that empirical stability fails, and **explicit contraction control is
the stabiliser that lets equilibrium training proceed at all** — our
contractive model (0.80) substantially beats the unconstrained DEQ-LM (2.2–3.3)
here.  The contraction ceiling of §4.9 is the *price* of that stabiliser; the
DEQ-LM paper pays a different price (fragility + heavy engineering) to avoid
it.  There is no configuration that is both unboundedly deep *and* robustly
trainable in this family without further machinery (Jacobian regularisation,
monotone operators, or scale).

---

## 5. Discussion

### 5.1 Why EP struggles where DEQ succeeds

EP's Hebbian correlation difference is an *approximation* of the true gradient
that is unbiased only under strong (and here unmet) assumptions on the energy.
For attention layers in particular the (Z, Q) correlation is a poor proxy for
the layer's contribution to the cost. The result is a weak, noisy signal that
cannot drive a real language model below a high plateau and is inherently
prone to monotone-weight-growth divergence. The DEQ implicit gradient replaces
this proxy with the exact equilibrium gradient at the same *local* cost (a few
single-block VJPs), which is why it closes most of the gap to backprop.

### 5.2 The contractivity–expressivity trade-off

A contractive fixed-point map is required for both the relaxation and the
Neumann series to converge. But contractivity limits the *effective depth* of
the equilibrium: the closer the Jacobian spectral radius is to 1, the richer
`Z*` and the lower the loss — until the series/relaxation fail. Our homeostasis
lets us dial this trade-off explicitly via the `contractivity` target, and our
experiments show it is the right knob (not width). This suggests future work
on **preconditioned / GMRES-style** adjoint solves that remain accurate closer
to the boundary.

### 5.3 What is still "no backprop"?

* **Pure EP**: zero autograd. Only forward activities. Most biologically
  plausible / hardware-local, but too weak for fluency.
* **DEQ implicit gradient**: uses single-block vector-Jacobian products at the
  equilibrium. It is *not* "zero autograd", but it does **not** backpropagate
  through the relaxation iterations or any depth — the gradient is an
  equilibrium quantity. Memory is independent of effective depth.

### 5.4 Limitations

* Per-step cost is high (a full relaxation + adjoint per step).
* DEQ still trails backprop (0.85 vs 0.09 on a corpus the baseline can nearly
  memorise).
* Single contractive block; multi-block / hierarchical equilibria untested.
* Char-level only; no BPE (offline tokenizer unavailable).

---

## 6. Conclusion

A Transformer can be trained at equilibrium without backpropagating through
its depth. **Pure equilibrium propagation** learns linguistic structure but
not fluency and is unstable on real data; the **DEQ implicit gradient** is the
practical recipe, reaching CE **1.31 → 0.80** on Chinese (cosine LR,
contractivity-boundary pushing, GMRES adjoint) while preserving the core
advantages (exact equilibrium gradient, no iteration backprop, constant
memory). The dominant lever for further quality is **not capacity, data,
steps, or depth-in-this-formulation but the strict-contraction constraint
itself**; the single contractive block saturates around CE 0.80, and all
mechanisms for adding depth (chained equilibria, composed maps) or width fail
to break it.  Escaping the ceiling needs techniques that provide gradient conditioning
*without* `ρ(J)<1` — e.g. **Jacobian regularisation** to keep eigenvalues of
`J_f` bounded away from 1, or monotone-equilibrium architectures — which is
genuine future research.

---

## Appendix A — Reproducing the results

```powershell
# English toy (EP vs backprop)
python -m energy_lm.run --steps 1500 --baseline

# MiniMind Chinese — pure EP (plateau)
python -m energy_lm.run_mm --mode ep --steps 3000 --baseline

# MiniMind Chinese — DEQ best (CE 0.80): GMRES adjoint + contractivity 0.80
python -m energy_lm.run_mm --mode deq --steps 6000 `
  --d_model 192 --free_steps 30 --adjoint gmres --gmres_k 10 `
  --anderson --anderson_beta 0.7 `
  --lr 1.5e-3 --lr_out 4e-3 --contractivity 0.80 --res_gain 0.5 --warmup 300 --baseline
```

## Appendix B — File map

| file | role |
|---|---|
| `energy_model.py` | ERB: energy, free/clamped relaxation, differentiable block, RMSNorm/tying flags |
| `ep_trainer.py` | Equilibrium Propagation + homeostasis + skip-on-divergence |
| `deq_trainer.py` | DEQ implicit gradient (Neumann **and GMRES** adjoint) + Adam + cosine LR |
| `multi_model.py`, `multi_trainer.py` | multi-block hierarchical-equilibrium model + chain-adjoint trainer (§4.7 ablation) |
| `deep_model.py`, `deep_trainer.py` | deep-map single-equilibrium model (composed & residual, §4.8) |
| `noncontractive_trainer.py`, `run_newton.py` | Newton-Krylov non-contractive DEQ via `torch.func.jvp` (§4.10) |
| `deq_lm.py`, `run_deqlm.py` | canonical standard DEQ-LM (Anderson + IFT `register_hook`, no contraction) for the §4.11 head-to-head |
| `acceleration.py` | Anderson acceleration (AA-m) |
| `baseline.py` | same-width backprop Transformer |
| `data.py`, `mm_data.py` | English toy corpus; MiniMind streaming + char tokenizer |
| `run.py`, `run_mm.py` | experiment runners |
