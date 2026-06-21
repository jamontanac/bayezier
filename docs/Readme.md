# Bayesian k-NN — Implementation Details

In this document i try to explain *why* each piece of the implementation is designed the way it is and why i decided to do it in that way.

**Related docs:**
- [[Steps_for_sampling_parameters|Step-by-step sampler walkthrough]] — the Gibbs and MH mechanics with numerical examples
- [[Technical_details_sampling_parameters|Sampling parameters reference]] — how to choose `n_iters`, `burn_in`, `thinning`, `proposal_width`, `beta_sigma`
- **Pseudocode:** [[Precomputation|Phase 1 (precomputation)]], [[Training|Phase 2 (MCMC)]], [[Inference|Phase 3 (prediction)]]

---

## Pseudocode Reference

[[Precomputation|### Phase 1 — Precomputation (runs once before MCMC)]]
> Builds `Count_Tensor`: for every training point, precomputes class-counts at every candidate k. Runs once; the MCMC loop later indexes into this tensor instead of re-sorting neighbors every iteration.

[[Training|### Phase 2 — Metropolis-within-Gibbs (sampling engine)]]
> MCMC sampler: alternates a **Gibbs step** over k (exhaustive evaluation of all candidates) with a **Metropolis-Hastings step** over β (Gaussian proposal with accept/reject). Produces the posterior chain of (k, β) draws.

[[Inference|### Phase 3 — Posterior predictive (post-inference)]]
> For each saved (k, β) draw in the chain, computes softmax probabilities over classes for test points, then Monte Carlo averages them into a final prediction.


---

## Why Metropolis-within-Gibbs Works for This Model

To understand why the sampler is structured the way it is, we need to look closely at how$k$ and $\beta$ behave under their conditional distributions.

The goal of the MCMC sampler is to draw samples from the joint posterior $p(k, \beta \mid \text{Data})$. Because tracking the complete joint space simultaneously is analytically intractable, we break the problem into a sequential coordinate-wise update: fix $\beta$ to update $k$, then fix $k$ to update $\beta$.

All the Steps are described [[Steps_for_sampling_parameters|here]] and the technical details explaining each of the parameters are in [[Technical_details_sampling_parameters|here]]

---

## MCMC as a Replacement for Numerical Integration

### The calculus problem we bypass

Equation 3 in Holmes & Adams (2002) states that the true predictive probability for a
new point $x$ belonging to class $c$ requires integrating over the full parameter space:

$$
p(c \mid x, \text{Data})
= \sum_{k=1}^{k_{\max}} \int_{0}^{\infty}
    \underbrace{p(c \mid x, k, \beta)}_{\text{PNN prediction for one configuration}}
    \cdot
    \overbrace{p(k, \beta \mid \text{Data})}^{\text{the complex posterior density}}
  \, d\beta
$$

Evaluating this directly would require setting up a numerical integration grid (Simpson's rule, Gaussian quadrature) for every candidate $k$ and every test point.

### The Monte Carlo approximation

MCMC completely sidesteps the integral by exploiting the **Monte Carlo identity**:

$$
\hat{p}(c \mid x, \text{Data})
\approx \frac{1}{S} \sum_{s=1}^{S} p\!\left(c \mid x,\, k^{(s)},\, \beta^{(s)}\right)
$$

where $\bigl(k^{(s)}, \beta^{(s)}\bigr)$ are the parameter draws collected by the sampler during training. Instead of solving a calculus problem, we replace the integral with a plain average over saved history — basic array addition and division.

### Concept mapping: math to code

| Concept | Mathematical form | Implementation |
|---|---|---|
| Posterior draws | $(k^{(s)}, \beta^{(s)})$ | `chain['k_idx']`, `chain['beta']` |
| Summation engine | $\sum_{s=1}^{S}$ | `for d in range(n_draws)` |
| Single draw prediction | $p(c \mid x, k^{(s)}, \beta^{(s)})$ | local `probs` vector per draw |
| Monte Carlo average | $\frac{1}{S}\sum(\cdots)$ | `accum_probs / n_draws` |

### Why this saves time

The MCMC loop in `fit_bayesknn` acts as an **importance filter** on the parameter space. It spends its iterations walking through $(k, \beta)$ pairs, discovering which configurations are actually plausible given the training data.

If a configuration like $k = 19, \beta = 0.2$ is highly implausible, the chain will simply never visit it. When `predict_bayesknn` runs, CPU cycles are spent only at the specific, high-probability parameter coordinates that the sampler saved — not uniformly across all of $\{1, \dots, k_{\max}\} \times [0, \infty)$. This turns an otherwise intractable integral into a loop over a few hundred saved draws.

---

## Rust CLI

All commands are run from the `rust/` workspace root using Pixi as the entrypoint. Use `pixi run cargo run -p pnn-cli --`.

### CLI argument synopsis

```bash
pixi run cargo run -p pnn-cli -- \
  --train <path> --test <path> --out <path> \
  [--dataset <name>] [--implementation <str>] \
  [--k <int>] [--k-values <int,int,...>] [--k-range <start,end>] \
  [--method hybrid|joint-mh] \
  [--n-samples <int>] [--burn-in <int>] [--thinning <int>] [--seed <int>] \
  [--beta-step <float>] [--beta-sigma <float>] \
  [--diagnose <path>]
```

### Required flags

| Flag | Type | Description |
|------|------|-------------|
| `--train <path>` | path | CSV file with labelled training data. Must have a header; last column must be named `label` with integer class ids starting at 0. |
| `--test <path>` | path | CSV file with test data. Same format as train. Labels are used only to compute `misclassification_cost`. |
| `--out <path>` | path | Where to write the output JSON. Parent directories are created automatically. |

### K-candidate flags (mutually exclusive — highest priority wins)

| Flag | Example | Description |
|------|---------|-------------|
| `--k-range <start,end>` | `--k-range 1,20` | Candidate set is every integer from `start` to `end` inclusive. Recommended for most experiments. |
| `--k-values <int,...>` | `--k-values 1,3,5,7` | Explicit comma-separated list of k candidates. Useful when you want a non-contiguous or sparse set. |
| `--k <int>` | `--k 5` | Single k candidate (degenerates to a one-point Gibbs step; beta is still inferred). Default: `3`. |

Precedence: `--k-values` > `--k-range` > `--k`.

Values larger than `n_train - 1` are silently clamped to `n_train - 1`.

### MCMC tuning flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--method <hybrid\|joint-mh>` | enum | `hybrid` | Sampler update rule. `hybrid` = Gibbs over k + MH over beta. `joint-mh` = one MH proposal over `(k, beta)` accepted/rejected jointly. |
| `--n-samples <int>` | positive integer | `1000` | Number of draws to keep after burn-in (the posterior chain length). More samples = lower Monte Carlo variance. |
| `--burn-in <int>` | non-negative integer | `500` | Iterations discarded before recording begins. Should be long enough for the chain to forget its starting state. Rule of thumb: 20–30 % of total iterations. |
| `--thinning <int>` | integer `>= 1` | `1` | Keep one draw every `thinning` post-burn-in iterations. Total iterations become `burn_in + n_samples * thinning`. |
| `--beta-step <float>` | positive float | `0.3` | Standard deviation of the Gaussian MH proposal for β. Controls how far the chain tries to jump each iteration. Target acceptance rate: 20–50 % (Hybrid) or 5–25 % (JointMh). Increase if acceptance rate is too high; decrease if too low. |
| `--beta-sigma <float>` | positive float | `5.0` | Scale of the half-normal prior on β. Encodes prior belief about interaction strength. Larger values allow β to be bigger; smaller values pull β toward zero regardless of the data. |
| `--seed <int>` | unsigned 64-bit integer | random | Fixed RNG seed for reproducible runs. Omit for non-deterministic output. |

### Metadata flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset <name>` | string | `"unknown"` | Label written into the output JSON for identification. |
| `--implementation <str>` | string | `"rust"` | Implementation tag in the output JSON. |

### Diagnostics flag

| Flag | Type | Description |
|------|------|-------------|
| `--diagnose <path>` | path | Write a second JSON file with MCMC health diagnostics: trace plots, autocorrelation, effective sample size (ESS), MH acceptance rate, and burn-in traces. Does not affect the main `--out` output. |

### Flag-specific options and validation

| Flag | Valid options / constraints |
|------|-----------------------------|
| `--method` | `hybrid` or `joint-mh` only. Any other value is rejected. |
| `--thinning` | Must be an integer `>= 1`. `0` is rejected. |
| `--k` | Must be integer `>= 1`. |
| `--k-values` | Comma-separated integers, each `>= 1`. |
| `--k-range` | `start,end` with `start >= 1` and `end >= start`. |
| `--n-samples` | Must be integer `>= 1`. |
| `--burn-in` | Must be a non-negative integer. |
| `--seed` | Must be a non-negative integer (`u64`). |

---

### Output JSON fields

The `--out` file contains:

```json
{
  "implementation": "rust",
  "dataset": "pima",
  "predictions": [
    { "index": 0, "probabilities": [0.72, 0.28], "predicted_class": 0 },
    ...
  ],
  "k_posterior": [7, 9, 7, 12, ...],
  "beta_posterior": [1.83, 1.91, 1.78, ...],
  "misclassification_cost": 0.211,
  "runtime_ms": 48.3
}
```

- `k_posterior` / `beta_posterior` — the full posterior chain (length = `n_samples`). Use these to inspect what the model learned about k and β.
- `misclassification_cost` — fraction of test points misclassified (0 = perfect, 1 = all wrong).

---

### Example commands

#### Minimal run (toy dataset, sanity check)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/sample_train.csv \
  --test  ../data/sample_test.csv \
  --out   /tmp/sample_out.json \
  --dataset sample \
  --k-range 1,3 \
  --n-samples 200 \
  --burn-in 50 \
  --seed 42
```

5 training points, 3 test points, 2 classes. Runs in milliseconds. Good for verifying the pipeline works before running larger experiments.

---

#### Pima diabetes (200 train / 332 test, 7 features, 2 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/pima_train.csv \
  --test  ../data/pima_test.csv \
  --out   results/pima.json \
  --dataset pima \
  --k-range 1,15 \
  --n-samples 2000 \
  --burn-in 500 \
  --seed 42
```

Binary classification of diabetes onset. A good first real dataset: medium-sized, 2 classes, clean features.

---

#### Synth (250 train / 1000 test, 2 features, 2 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/synth_train.csv \
  --test  ../data/synth_test.csv \
  --out   results/synth.json \
  --dataset synth \
  --k-range 1,20 \
  --n-samples 2000 \
  --burn-in 500 \
  --seed 42
```

Synthetic 2D dataset from Ripley (1996). The large test set (1000 points) gives a stable misclassification estimate. Useful for checking that the posterior concentrates on sensible k values.

---

#### Forest glass (171 train / 43 test, 9 features, 6 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/fglass_train.csv \
  --test  ../data/fglass_test.csv \
  --out   results/fglass.json \
  --dataset fglass \
  --k-range 1,10 \
  --n-samples 2000 \
  --burn-in 500 \
  --seed 42
```

Multi-class problem (6 glass types). Keep `k-range` small relative to training set size (171 points). Expect higher misclassification cost than binary problems.

---

#### Crabs (160 train / 40 test, 6 features, 2 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/crabs_train.csv \
  --test  ../data/crabs_test.csv \
  --out   results/crabs.json \
  --dataset crabs \
  --k-range 1,20 \
  --n-samples 2000 \
  --burn-in 500 \
  --seed 42
```

Classify crab species (Blue vs Orange) from morphological measurements. Well-separated classes — expect low misclassification cost and the chain concentrating on small k.

---

#### Viruses (48 train / 13 test, 17 features, 6 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/viruses_train.csv \
  --test  ../data/viruses_test.csv \
  --out   results/viruses.json \
  --dataset viruses \
  --k-range 1,5 \
  --n-samples 1000 \
  --burn-in 300 \
  --seed 42
```

Small dataset (48 training points) — keep k-range small (well below 48). High-dimensional features (17) relative to training set size.

---

#### Cushing's syndrome (21 train / 6 test, 2 features, 4 classes)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/cushings_train.csv \
  --test  ../data/cushings_test.csv \
  --out   results/cushings.json \
  --dataset cushings \
  --k-range 1,5 \
  --n-samples 1000 \
  --burn-in 200 \
  --seed 42
```

Very small dataset (21 training points). Use a narrow k-range. Treat results with caution given the limited test set size (6 points).

---

### Using `--diagnose` to check chain health

Add `--diagnose <path>` to any run to get a second JSON with MCMC diagnostics:

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/pima_train.csv \
  --test  ../data/pima_test.csv \
  --out   results/pima.json \
  --diagnose results/pima_diag.json \
  --dataset pima \
  --k-range 1,15 \
  --n-samples 2000 \
  --burn-in 500 \
  --seed 42
```

The diagnostics file contains:

```json
{
  "config": { "method": "Hybrid", "k_candidates": {"start": 1, "end": 15}, ... },
  "mh_acceptance": { "n_accepted": 980, "n_proposed": 2000, "rate": 0.49 },
  "beta":  { "trace": [...], "mean": 1.9, "std": 0.4, "acf": [...], "ess": 280 },
  "k":     { "trace": [...], "frequencies": {"7": 800, "9": 1200}, "acf": [...], "ess": 310 },
  "burn_in": { "beta_trace": [...] }
}
```

**What to look for:**

| Diagnostic | Healthy | Problem | Fix |
|---|---|---|---|
| `mh_acceptance.rate` | 0.20 – 0.50 (Hybrid) / 0.05 – 0.25 (JointMh) | Below 0.10: proposals too wide, chain stuck. Above 0.70: proposals too narrow, chain crawls. | Decrease or increase `--beta-step` respectively. |
| `beta.ess` | > 100 for n_samples=1000 | Very low (< 20): successive draws are nearly identical. | Increase `--beta-step` to make bolder moves, or increase `--thinning` to space draws further apart. |
| `beta.acf` | Drops to ~0 within 20 lags | Slow decay: β moves are tiny relative to the posterior width. | Increase `--beta-step`. |
| `beta.mean` / `beta.max` | β > 0 with some spread | β pinned near 0 across the whole chain: prior too restrictive. | Increase `--beta-sigma` to widen the half-normal prior. |
| `k.frequencies` | Spread across several candidates | Single k dominates (>90% of draws): either the data genuinely prefers one k, or the range is too narrow. | Try a wider `--k-range` to give the Gibbs step more room. |
| `burn_in.beta_trace` | Stabilises well before the end of burn-in | Still drifting or jumping at the end: chain has not found the posterior yet. | Increase `--burn-in`. |

### Worked tuning loop (`--beta-step`)

Use this practical loop when the chain mixes poorly.

#### 1) First run (diagnose only)

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/pima_train.csv \
  --test  ../data/pima_test.csv \
  --out   results/pima.json \
  --diagnose results/pima_diag_step1.json \
  --dataset pima \
  --k-range 1,15 \
  --n-samples 2000 \
  --burn-in 500 \
  --thinning 1 \
  --beta-step 0.3 \
  --beta-sigma 5.0 \
  --seed 42
```

Read `results/pima_diag_step1.json` and check:
- `mh_acceptance.rate`
- `beta.ess`
- `beta.acf`

#### 2) Retune `--beta-step`, then rerun

If acceptance is too low (< 0.20 for Hybrid), decrease `--beta-step`.
If acceptance is too high (> 0.50 for Hybrid), increase `--beta-step`.

Example retry with smaller proposal width:

```bash
pixi run cargo run -p pnn-cli -- \
  --train ../data/pima_train.csv \
  --test  ../data/pima_test.csv \
  --out   results/pima_tuned.json \
  --diagnose results/pima_diag_step2.json \
  --dataset pima \
  --k-range 1,15 \
  --n-samples 2000 \
  --burn-in 500 \
  --thinning 1 \
  --beta-step 0.15 \
  --beta-sigma 5.0 \
  --seed 42
```

Compare `pima_diag_step1.json` vs `pima_diag_step2.json`:
- acceptance rate should move toward the target range
- `beta.ess` should usually improve
- `beta.acf` should decay faster

Notes:
- For `--method joint-mh`, target acceptance is lower (roughly 5–25%).
- Change `--beta-sigma` only when you want to alter the prior strength on β, not proposal behavior.
