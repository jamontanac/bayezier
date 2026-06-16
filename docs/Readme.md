# Bayesian k-NN — Implementation Details

In this document i try to explain *why* each piece of the implementation is designed the way it is and why i decided to do it in that way.

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

To understand why the sampler is structured the way it is, we need to look closely at how
$k$ and $\beta$ behave under their conditional distributions.

The goal of the MCMC sampler is to draw samples from the joint posterior
$p(k, \beta \mid \text{Data})$.
Because tracking the complete joint space simultaneously is analytically intractable,
we break the problem into a sequential coordinate-wise update:
fix $\beta$ to update $k$, then fix $k$ to update $\beta$.

---

### Step 1 — Gibbs Step for $k$

#### Why Gibbs works here

A Gibbs step requires sampling directly from a parameter's full conditional distribution
$p(k \mid \beta, \text{Data})$.
Usually Gibbs is reserved for conjugate priors where the math resolves into a known
closed-form distribution (Gamma, Normal, etc.).

$k$ is uniquely suited for a Gibbs step because:

1. It is entirely **discrete** — you cannot have 3.5 neighbors.
2. It is **finitely bounded** by $k_{\max}$ (e.g. $k \in \{1, 2, \dots, 30\}$).

Because the universe of valid choices for $k$ is small and finite, we do not need to
guess or propose. We exhaustively evaluate the exact posterior probability for every
valid $k$ simultaneously, build a precise discrete probability distribution, and sample
from it directly.

#### The mathematical engine

For each candidate value $\kappa$ in our array of choices we calculate its
unnormalized log-posterior score:

$$
\log p(k = \kappa \mid \beta, \text{Data})
= \sum_{i=1}^{n} \left[
    \frac{\beta}{\kappa}\, n_{i,\kappa,y_i}
    - \log \sum_{m=1}^{M} \exp\left(\frac{\beta}{\kappa}\, n_{i,\kappa,m}\right)
  \right]
+ \log p(k = \kappa)
$$

Where:

- $n_{i,\kappa,y_i}$ — count of the $\kappa$ nearest neighbors of point $i$ that share
  point $i$'s true class label $y_i$.
- $M$ — total number of classes.
- $p(k = \kappa)$ — prior over neighborhood size (uniform, so $\log p(k)$ is a constant
  and cancels in the normalization step).

#### The numerical underflow trap — Log-Sum-Exp

If you evaluate the log-score for all candidates and immediately exponentiate to recover
probabilities, the code will produce `NaN` or `0.0`.
Because the log-likelihood sums accumulate over hundreds of training points, raw
log-scores look like `[-1420.4, -1422.1, -1455.8]`.
Computing $e^{-1420.4}$ underflows to exactly `0.0` in IEEE 754 floating point.

To convert these safely to a valid discrete probability distribution, we apply the
**Log-Sum-Exp / Softmax trick**:

1. Find the maximum log-score: $C = \max_\kappa(\log p_\kappa)$.
2. Subtract $C$ before exponentiating:

$$
P(k = \kappa \mid \beta, \text{Data})
= \frac{
    \exp\bigl(\log p_\kappa - C\bigr)
  }{
    \displaystyle\sum_{\kappa'} \exp\bigl(\log p_{\kappa'} - C\bigr)
  }
$$

Pulling out the largest exponent forces the best-scoring candidate to evaluate to
$\exp(0) = 1$, and all others scale proportionally below it without underflowing.
The normalized vector is fed into a categorical sampler to select the next state for $k$.

---

### Step 2 — Metropolis-Hastings Step for $\beta$

#### Why MH instead of Gibbs

Unlike $k$, the interaction strength $\beta$ is a **continuous** variable on
$[0, \infty)$.
There are infinitely many values between any two points, so we cannot enumerate a finite
grid and compute exact probabilities.
Additionally, $\beta$ appears inside the denominator of the pseudo-likelihood in a way
that cannot be isolated analytically.

Because we cannot compute the conditional distribution of $\beta$ directly, we must
explore it with a guided random walk: Metropolis-Hastings.

#### The proposal mechanism

At iteration $t$, we propose a new candidate $\beta^*$ by drawing from a symmetric
Gaussian centered on the current state:

$$
\beta^* \sim \mathcal{N}\!\left(\beta^{(t)},\, \sigma^2\right)
$$

where $\sigma$ is the tuning width (proposal standard deviation).

- If $\sigma$ is **too large**: proposals jump far into low-probability regions; the chain
  rejects almost everything and gets stuck.
- If $\sigma$ is **too small**: the chain accepts frequently but moves in tiny steps,
  exploring the distribution very slowly.

A well-tuned chain targets an acceptance rate between 20 % and 50 %.

#### The acceptance/rejection filter

Once $\beta^*$ is generated, we measure how much the data prefers this new value over
the current one by computing the log acceptance ratio:

$$
\log r
= \log p\left(\beta^* \mid k^{(t+1)}, \text{Data}\right)
- \log p\left(\beta^{(t)} \mid k^{(t+1)}, \text{Data}\right)
$$

The update decision follows a three-branch rule:

1. **Boundary rejection**: if $\beta^* \le 0$, the half-normal prior assigns density
   zero ($\log 0 = -\infty$). The proposal is rejected immediately without further
   evaluation.
2. **Uphill move** ($\log r > 0$): the proposed state has higher posterior density.
   We always accept: $\beta^{(t+1)} = \beta^*$.
3. **Downhill move** ($\log r < 0$): the proposed state is worse, but we do not discard
   it automatically. We draw $u \sim \mathcal{U}(0,1)$ and accept if $\log u < \log r$.
   This probabilistic acceptance of worse states is what prevents the chain from
   collapsing to a local mode and preserves global exploration.

---

### Step 3 — The Update Cycle and Why Order Matters

The two mechanisms combine into a precise sequential dependency within each iteration:

```text
State at start of iteration:     [ k^(t),   β^(t) ]
                                         │
                                         ▼
  1. Evaluate log-posterior for all k candidates using fixed β^(t)
  2. Softmax-normalize → sample new k index
                                         │
                                         ▼
Intermediate state:              [ k^(t+1), β^(t) ]
                                         │
                                         ▼
  3. Propose β* centered around β^(t)
  4. Evaluate log-posterior at β* using the newly updated k^(t+1)
  5. Run accept/reject coin flip
                                         │
                                         ▼
State at end of iteration:       [ k^(t+1), β^(t+1) ]
```

The MH step for $\beta$ uses $k^{(t+1)}$ — the value just sampled by the Gibbs step —
not the stale $k^{(t)}$ from the beginning of the loop.

This ordering creates a **tight feedback loop**: if the Gibbs step discovers that a
smaller neighborhood ($k = 3$) fits the local data clusters better, the subsequent MH
step immediately tests whether $\beta$ needs to scale upward to compensate for the
tighter neighborhood constraints.
The two parameters pull each other toward the high-density region of the joint posterior
rather than drifting independently.

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
