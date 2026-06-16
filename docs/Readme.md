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

Evaluating this directly would require setting up a numerical integration grid
(Simpson's rule, Gaussian quadrature) for every candidate $k$ and every test point —
computationally prohibitive.

### The Monte Carlo approximation

MCMC completely sidesteps the integral by exploiting the **Monte Carlo identity**:

$$
\hat{p}(c \mid x, \text{Data})
\approx \frac{1}{S} \sum_{s=1}^{S} p\!\left(c \mid x,\, k^{(s)},\, \beta^{(s)}\right)
$$

where $\bigl(k^{(s)}, \beta^{(s)}\bigr)$ are the parameter draws collected by the
sampler during training.
Instead of solving a calculus problem, we replace the integral with a plain average
over saved history — basic array addition and division.

### Concept mapping: math to code

| Concept | Mathematical form | Implementation |
|---|---|---|
| Posterior draws | $(k^{(s)}, \beta^{(s)})$ | `chain['k_idx']`, `chain['beta']` |
| Summation engine | $\sum_{s=1}^{S}$ | `for d in range(n_draws)` |
| Single draw prediction | $p(c \mid x, k^{(s)}, \beta^{(s)})$ | local `probs` vector per draw |
| Monte Carlo average | $\frac{1}{S}\sum(\cdots)$ | `accum_probs / n_draws` |

### Why this saves time

The MCMC loop in `fit_bayesknn` acts as an **importance filter** on the parameter space.
It spends its iterations walking through $(k, \beta)$ pairs, discovering which
configurations are actually plausible given the training data.

If a configuration like $k = 19, \beta = 0.2$ is highly implausible, the chain will
simply never visit it.
When `predict_bayesknn` runs, CPU cycles are spent only at the specific, high-probability
parameter coordinates that the sampler saved — not uniformly across all of
$\{1, \dots, k_{\max}\} \times [0, \infty)$.
This turns an otherwise intractable integral into a loop over a few hundred saved draws.
