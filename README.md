# LongRange-PFN (proof of concept)

PoC of a **long-range PFN**: instead of running the PFN on the full context
$D_{\mathrm{tot}}$, a distant part $D'$ is compressed into a **latent
summary** $z = \mathrm{rep}(D')$ and the model predicts from the remaining
local context $D$ conditioned on $z$. The goal is to approximate the
prediction that would be obtained with the full context.

Built on top of the original PFN repository ([automl/PFNs](https://github.com/automl/PFNs),
cloned in `PFNs/`), reusing `TableTransformer`, `PerFeatureLayer`, the
encoders, and `FullSupportBarDistribution`.

## The idea (from the paper)

Case 1 — everything fits in context ($D_{\mathrm{tot}} = D'$):

```
z ← ∅ (learned null token)
r ← g(x, D', ∅)          →   q_r ≈ p(y | x, D')
```

Case 2 — long context ($D_{\mathrm{tot}} = D' \cup D$):

```
z ← rep(D')               (latent summary of D')
r ← g(x, D, z)            →   q_r ≈ p(y | x, D, D') = ∫ p(y|x,ρ) p(ρ|D,D') dρ
```

Training minimizes the KL divergence between the full-context teacher
(frozen) and the student predictive:

```
L = E_{D_tot, x}  KL( p_teacher(y | x, D' ∪ D)  ‖  q_student(y | x, D, z(D')) )
```

Because both predictives are bar distributions over the same buckets, the KL
reduces to the categorical KL between bucket probabilities.

## Implementation

| Component | File | Description |
|---|---|---|
| Teacher | [models.py](long_range_pfn/models.py) `build_teacher` | Original `TableTransformer` (classic PFN architecture, 1 token per sample), trained with bar-distribution NLL on the GP prior (`pfns.priors.fast_gp`) with random context size. |
| Summarizer `rep(D')` | `Summarizer` | Embeds the $(x,y)$ pairs in $D'$ with the repo encoders and appends $k{=}8$ learned latent query tokens as pseudo-test positions: at each layer they attend to $D'$ tokens (standard PFN mask). The final query outputs are $z \in \mathbb{R}^{k \times e}$. $D'{=}\emptyset$ uses a learned null summary. |
| Student `g(x, D, z)` | `LongRangePFN` | Same backbone as the teacher (initialized from its weights). The $k$ summary tokens are prepended as extra context items: sequence `[z₁..z_k | D | x_test]` with `single_eval_pos = k + |D|`. Every test point attends to $z$ as if they were real samples. |
| Distillation | [train_student.py](long_range_pfn/train_student.py) | KL(teacher‖student) with split $|D| \sim U\{5..25\}$, $|D'| = 80 - |D|$, and with probability 0.1 Case 1 ($|D'|=0$, null $z$). |
| **Recursive summary** | `Summarizer.forward(prev_z=...)` | Streaming update $z_t = \mathrm{rep}(C_t, z_{t-1})$: previous summary tokens enter as extra context inside the summarizer. Lets the model consume context **larger than the teacher window** with $O(\text{chunk})$ memory. Trained by sampling 1–3 chunks per step. |

## New contribution (beyond the baseline setup)

**Recursive** summarization turns the summarizer into a bounded-state posterior
accumulator — a learned analogue of the Bayesian filtering update
$p(\rho \mid C_{1:t}) \propto p(C_t \mid \rho)\, p(\rho \mid C_{1:t-1})$.
Final results (see [paper/main.tex](paper/main.tex)):

- **Information certificate**: the student's NLL sits 0.03–1.30 nats
  *below* the exact GP posterior restricted to $D$, at **every** $|D|$ tested
  — a level unreachable by any method that only sees $D$ (Proposition 1),
  proving that posterior information flows through $z$. Without $z$ (ablation),
  the gain vanishes.
- **Truncation gap**: closed by 97% at $|D|=10$; the student tracks the
  full-context teacher across the sweep.
- **Streaming beyond the window**: with $|D_{\mathrm{tot}}| = 320$ ($4\times$ the
  training window) in 5 recursive chunks, the student **beats** the recency
  baseline from $|D_{\mathrm{tot}}|=160$, recovering ~39% of the remaining
  headroom to the exact posterior, with constant memory.
- **Training phase transition** (not grokking): the ability to read $z$ forms
  abruptly between steps 6k and 10k, synchronized across regimes — online
  regime, no train/validation gap (corr 0.98). Short runs stall on the plateau,
  which explained the weak early results.
- **Real limitation**: multi-chunk summarization costs ~14× more KL than
  single-shot ($T{=}5$: 0.28 vs 0.02 nats), and the cost persists after
  fixing the model-selection criterion — compounded error across recursive
  steps, quantified in the paper.

## How to run

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps -e ./PFNs && .venv/bin/pip install gpytorch einops
.venv/bin/python -m long_range_pfn.train_teacher   # ~6 min (MPS)
.venv/bin/python -m long_range_pfn.train_student   # ~12 min (MPS)
.venv/bin/python -m long_range_pfn.evaluate        # metrics + figures in results/
```

## Evaluation

`evaluate.py` compares, on fresh prior draws with $|D_{\mathrm{tot}}| = 80$ and 40
test points, varying $|D| \in \{5, 10, 20, 40\}$:

- **Exact GP (full / D-only)** — analytic posterior of the generating GP (gold standard, via gpytorch);
- **teacher-full** — $p(y \mid x, D_{\mathrm{tot}})$, the reference the student aims to match;
- **teacher-local** — $p(y \mid x, D)$, what you get by simply truncating the context;
- **LongRange-PFN** — $q(y \mid x, D, z(D'))$;
- **student without z** — ablation with null summary (isolates the contribution of $z$).

Metrics: test-target NLL and KL to teacher-full. Outputs in `results/`:
`metrics.json`, `sweep.png` (curves), and `qualitative.png` (1-D predictive
with mean and 90% credible interval).
