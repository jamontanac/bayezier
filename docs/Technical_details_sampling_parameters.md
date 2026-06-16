# Sampling Parameters

The MCMC sampler is controlled by a `config` object with several knobs. Here i want to document what each knob does, why it exists, and how to think about choosing values.

**Related docs:**
- [[Steps_for_sampling_parameters|Step-by-step sampler walkthrough]] — see how these parameters are used in the Gibbs + MH loop
- [[Training|Phase 2 pseudocode]] — the MCMC loop that reads this config
- [[Readme|Main readme]] — overall implementation overview

---

## `n_iters` — Total MCMC iterations

**What it is:** The total number of Gibbs+MH cycles the sampler runs.

**Why it exists:** MCMC needs many iterations to converge to the stationary distribution and then collect enough samples for a reliable Monte Carlo estimate.

**How to choose:**
- Too few → chain hasn't converged; samples are biased toward the initial state.
- Too many → wasted compute after the chain has already explored the posterior.
- Typical range: a few thousand to tens of thousands, depending on dataset size and how correlated successive draws are.

**Relation to other parameters:** Only the post-burn-in and post-thinning iterations end up in the chain. If `n_iters = 10_000`, `burn_in = 2_000`, and `thinning = 5`, you get `(10_000 - 2_000) / 5 = 1_600` saved draws.

---

## `burn_in` — Warm-up iterations to discard

**What it is:** The number of initial iterations thrown away before the chain starts recording.

**Why it exists:** The chain starts at an arbitrary initial state (currently `β = 1.0`, `k` at the median candidate). Early iterations are biased toward this initialization, not the true posterior. Discarding them lets the chain "forget" where it started.

**How to choose:**
- Look at trace plots: discard up to the point where the chain appears to have stabilized.
- Rule of thumb: 10–20 % of `n_iters`.
- Aggressive burn-in is cheap because the full iteration still runs; only the recording step is skipped.

---

## `thinning` — Spacing between saved draws

**What it is:** Only every `thinning`-th iteration after burn-in is appended to the chain.

**Why it exists:** Consecutive MCMC draws are **autocorrelated** — each iteration starts from the previous one, so neighboring samples are similar. Thinning reduces correlation between saved draws (at the cost of discarding information).

**How to choose:**
- `thinning = 1` → save every iteration (maximum information, highest autocorrelation).
- Higher thinning → less autocorrelation, fewer total samples.
- Common values: 1–10. Check autocorrelation plots to decide.
- For this model, k can change by at most one candidate per Gibbs step, and β moves by the proposal width. If the chain mixes well, thinning of 5–10 is often sufficient.

**Trade-off:** Throwing away samples reduces the effective sample size (ESS). If storage is not a concern and autocorrelation is low, `thinning = 1` is fine — just be aware that your effective sample size is lower than the raw count.

---

## `proposal_width` — MH proposal standard deviation (σ)

**What it is:** The standard deviation of the Gaussian used to propose new β values: `β* ~ N(β, σ²)`.

**Why it exists:** The MH step for β needs a proposal distribution to explore the continuous space. A Gaussian centered on the current value is the simplest symmetric choice.

**How to choose:**
- This is the **most sensitive parameter** in the sampler.
- Too small → chain moves in tiny increments, explores slowly (high acceptance, but high autocorrelation).
- Too large → most proposals land in low-density regions and are rejected; chain gets stuck (low acceptance, doesn't move).
- Target acceptance rate: 20–50 %.
- Start with a small value (e.g. 0.1–0.5) and increase if the acceptance rate is above 50 %; decrease if below 20 %.

**Practical note:** Many implementations use an adaptive scheme that tunes this during burn-in. The current pseudocode uses a fixed value for simplicity.

---

## `beta_sigma` — Prior scale for β (half-normal)

**What it is:** The scale parameter σ of the half-normal prior on β:

```
log p(β) = -ln(σ) - 0.5·(β/σ)² - ln(√(2/π))   for β > 0
```

**Why it exists:** The prior encodes what values of β are plausible before seeing any data. The half-normal constrains β to be positive (no negative interaction strengths) and penalizes very large β values.

**How to choose:**
- `beta_sigma` controls how strongly large β values are penalized: a larger σ means a weaker penalty (fatter tail).
- If you expect strong class-separation signals (tight clusters), β may need to be large → use a larger σ (e.g. 5–10).
- If the data is noisy with overlapping classes, β should stay small → use a smaller σ (e.g. 1–3).
- Can also be treated as a hierarchical parameter and inferred, but in this implementation it is fixed.

---

## Summary table

| Parameter | Role | Typical range | Sensitivity |
|---|---|---|---|
| `n_iters` | Total iterations | 2_000 – 50_000 | Low (more is safer) |
| `burn_in` | Discarded warm-up | 10–20 % of n_iters | Low |
| `thinning` | Save spacing | 1 – 10 | Low–Medium |
| `proposal_width` | MH step size | 0.1 – 2.0 | **High** |
| `beta_sigma` | Prior scale | 1.0 – 10.0 | Medium |

