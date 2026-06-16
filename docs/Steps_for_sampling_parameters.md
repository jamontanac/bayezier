

# Steps for Sampling Parameters

This document walks through the mechanics of the Metropolis-within-Gibbs sampler step by step. It explains *how* and *why* each update works, provides concrete numerical examples, and covers common issues you might see when running the sampler.

**Related docs:**
- [[Technical_details_sampling_parameters|Sampling parameters reference]] — how to choose `n_iters`, `burn_in`, `thinning`, `proposal_width`, `beta_sigma`
- [[Training|Pseudocode — Phase 2 (MCMC loop)]] — the code that implements these steps
- [[Readme|Main readme]] — overall implementation overview

---

# Step 1 — Gibbs Step for $k$

## Why Gibbs works here

A Gibbs step requires sampling directly from a parameter's full conditional distribution $p(k \mid \beta, \text{Data})$.
Usually Gibbs is reserved for conjugate priors where the math resolves into a known closed-form distribution (Gamma, Normal, etc.).
$k$ is uniquely suited for a Gibbs step because:

1. It is entirely **discrete** — you cannot have 3.5 neighbors.
2. It is **finitely bounded** by $k_{\max}$ (e.g. $k \in \{1, 2, \dots, 30\}$).

Because the universe of valid choices for $k$ is small and finite, we do not need to guess or propose. We exhaustively evaluate the exact posterior probability for every valid $k$ simultaneously, build a precise discrete probability distribution, and sample from it directly.

## The mathematical engine

For each candidate value $\kappa$ in our array of choices we calculate its unnormalized log-posterior score:

$$
\log p(k = \kappa \mid \beta, \text{Data})
= \sum_{i=1}^{n} \left[
    \frac{\beta}{\kappa}\, n_{i,\kappa,y_i}
    - \log \sum_{m=1}^{M} \exp\left(\frac{\beta}{\kappa}\, n_{i,\kappa,m}\right)
  \right] + \log p(k = \kappa)
$$

Where:

- $n_{i,\kappa,y_i}$ — count of the $\kappa$ nearest neighbors of point $i$ that share point $i$'s true class label $y_i$.
- $M$ — total number of classes.
- $p(k = \kappa)$ — prior over neighborhood size (uniform, so $\log p(k)$ is a constant
  and cancels in the normalization step).

## The numerical underflow trap — Log-Sum-Exp

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

## Numerical example — Gibbs softmax

Using the same small dataset from the pseudocode example (4 points, 2 classes, `k = [1, 3]`):

```
Count_Tensor for point i=0:
  k=1: [1, 0]   ← 1 neighbor, class 0
  k=3: [1, 2]   ← 3 neighbors: 1 of class 0, 2 of class 1
```

Suppose we fix $\beta = 1.0$ and compute the log-likelihood contribution for each candidate k:

**For k=1:**
```
logits = β · counts / k = 1.0 · [1, 0] / 1 = [1.0, 0.0]

log_p_i for point i=0 (true class = 0):
  = logits[0] - log_sum_exp(logits)
  = 1.0 - log(exp(1.0) + exp(0.0))
  = 1.0 - log(2.718 + 1.0)
  = 1.0 - log(3.718)
  = 1.0 - 1.314
  = -0.314
```

**For k=3:**
```
logits = β · counts / k = 1.0 · [1, 2] / 3 = [0.333, 0.667]

log_p_i for point i=0 (true class = 0):
  = logits[0] - log_sum_exp(logits)
  = 0.333 - log(exp(0.333) + exp(0.667))
  = 0.333 - log(1.395 + 1.948)
  = 0.333 - log(3.343)
  = 0.333 - 1.207
  = -0.874
```

That was just one point. Now we sum over **all 4 training points** using the full Count_Tensor from Details.md:

```
log_likelihood(k=1) = (-0.314) + (-0.314) + (-0.314) + (-0.314) = -1.256
log_likelihood(k=3) = (-0.874) + (-0.540) + (-0.540) + (-0.874) = -2.828
```

Then add the uniform prior (same for every candidate, but included for correctness):

```
log_prior_k = -ln(n_candidates) = -ln(2) ≈ -0.693
```

The final log-posterior scores (which the pseudocode stores as `log_w`):

```
log_w(k=1) = -1.256 + (-0.693) = -1.949
log_w(k=3) = -2.828 + (-0.693) = -3.521
```

```
log_w = [-1.949, -3.521]
```

**Softmax normalization (log-sum-exp trick):**
```
C = max(log_w) = -1.949

P(k=1) = exp(-1.949 - (-1.949)) / [exp(0) + exp(-3.521 - (-1.949))]
       = exp(0) / [1 + exp(-1.572)]
       = 1 / (1 + 0.208)
       ≈ 0.828

P(k=3) = exp(-1.572) / 1.208
       = 0.208 / 1.208
       ≈ 0.172
```

The sampler draws from `Categorical([0.828, 0.172])` — k=1 is more likely at this β.

> **Note:** With only 4 training points, the log-scores are around -2. With hundreds of points they would be more like -1400, making the log-sum-exp trick essential. The arithmetic is identical either way — the numbers just shift down.

---

# Step 2 — Metropolis-Hastings Step for $\beta$

## Why MH instead of Gibbs

Unlike $k$, the interaction strength $\beta$ is a **continuous** variable on $[0, \infty)$. There are infinitely many values between any two points, so we cannot enumerate a finite grid and compute exact probabilities. Additionally, $\beta$ appears inside the denominator of the pseudo-likelihood in a way that cannot be isolated analytically.

Because we cannot compute the conditional distribution of $\beta$ directly, we must explore it with a guided random walk: Metropolis-Hastings.

## The proposal mechanism

At iteration $t$, we propose a new candidate $\beta^*$ by drawing from a symmetric Gaussian centered on the current state:

$$ \beta^* \sim \mathcal{N}\!\left(\beta^{(t)},\, \sigma^2\right) $$

where $\sigma$ is the tuning width (proposal standard deviation).

- If $\sigma$ is **too large**: proposals jump far into low-probability regions; the chain rejects almost everything and gets stuck.
- If $\sigma$ is **too small**: the chain accepts frequently but moves in tiny steps, exploring the distribution very slowly.

A well-tuned chain targets an acceptance rate between 20 % and 50 %.

## The acceptance/rejection filter

Once $\beta^*$ is generated, we measure how much the data prefers this new value over the current one by computing the log acceptance ratio:

$$ \log r = \log p\left(\beta^* \mid k^{(t+1)}, \text{Data}\right) - \log p\left(\beta^{(t)} \mid k^{(t+1)}, \text{Data}\right) $$
The update decision follows a three-branch rule:

1. **Boundary rejection**: if $\beta^* \le 0$, the half-normal prior assigns density zero ($\log 0 = -\infty$). The proposal is rejected immediately without further evaluation.
2. **Uphill move** ($\log r > 0$): the proposed state has higher posterior density. We always accept: $\beta^{(t+1)} = \beta^*$.
3. **Downhill move** ($\log r < 0$): the proposed state is worse, but we do not discard it automatically. We draw $u \sim \mathcal{U}(0,1)$ and accept if $\log u < \log r$. This probabilistic acceptance of worse states is what prevents the chain from collapsing to a local mode and preserves global exploration.

## Numerical example — MH accept/reject

Assume we are mid-chain with current state:

```
k = 3,  β = 1.0,  proposal_width = 0.5
```

**Step A — Propose:**
```
β* ~ Normal(1.0, 0.5) → suppose we draw β* = 1.37
```

**Step B — Evaluate log-likelihood at current β and proposed β* (using k=3):**

For a single point i=0 with Count_Tensor[0, k=3] = [1, 2]:

```
At β = 1.0:
  logits = [1.0·1/3, 1.0·2/3] = [0.333, 0.667]
  log_p_i = logits[0] - log_sum_exp = 0.333 - 1.207 = -0.874

At β = 1.37:
  logits = [1.37·1/3, 1.37·2/3] = [0.457, 0.913]
  log_p_i = 0.457 - log(exp(0.457) + exp(0.913))
          = 0.457 - log(1.579 + 2.492)
          = 0.457 - log(4.071)
          = 0.457 - 1.404
          = -0.947
```

Summing over all points and adding the half-normal prior $(\sigma=3)$:

```
log_joint(β=1.0,  k=3) = -1422.1 + prior(1.0)  = -1422.1 + (-1.72) = -1423.82
log_joint(β=1.37, k=3) = -1425.4 + prior(1.37) = -1425.4 + (-1.89) = -1427.29
```

**Step C — Acceptance ratio:**
```
log r = log_joint(β*) - log_joint(β)
      = -1427.29 - (-1423.82)
      = -3.47
```

**Step D — Decision:**
- `log r < 0` → downhill move. We draw `u ~ Uniform(0,1)`.
- Suppose `u = 0.03` → `log(0.03) = -3.51`.
- Is `log(u) < log r`? Is `-3.51 < -3.47`? **Yes** → accept β = 1.37 despite being worse.

If instead `u = 0.01` → `log(0.01) = -4.61`. Is `-4.61 < -3.47`? **No** → reject, stay at β = 1.0.

---

# Step 3 — The Update Cycle and Why Order Matters

The two mechanisms combine into a precise sequential dependency within each iteration:

```
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

The MH step for $\beta$ uses $k^{(t+1)}$ — the value just sampled by the Gibbs step — not the stale $k^{(t)}$ from the beginning of the loop.

This ordering creates a **tight feedback loop**: if the Gibbs step discovers that a smaller neighborhood ($k = 3$) fits the local data clusters better, the subsequent MH step immediately tests whether $\beta$ needs to scale upward to compensate for the tighter neighborhood constraints.
The two parameters pull each other toward the high-density region of the joint posterior rather than drifting independently.

---

# Troubleshooting

## Chain never moves from initial values

If `k` and `β` remain at their starting values for hundreds of iterations, check:

- **`proposal_width` is too small** — the MH step proposes tiny β changes, so the chain moves in imperceptible steps. Try increasing it.
- **One k candidate dominates completely** — the softmax assigns >0.99 probability to a single k value. This can happen if the data is very clean and one neighborhood size fits far better. Check the softmax probabilities in the trace.

## β gets stuck near 0

If `β` hovers near 0 and never moves upward:

- **`beta_sigma` is too small** — the half-normal prior is too tight, penalizing any moderate β. Try increasing it.
- **Data is extremely noisy** — the model genuinely prefers β near 0 (flat predictions). This is valid but worth checking if the chain can't escape.
- **Numerical issue** — ensure `Count_Tensor` has non-zero counts for each class at each candidate k. All-zero rows produce logits of 0 which can cause the likelihood to misbehave.

## k never switches

If the chain stays on a single k value for the entire run:

- **Check the softmax probabilities** — if one candidate gets >0.99 probability at every iteration, the data strongly prefers that value. This is fine if the chain is genuinely concentrated.
- **Try more k candidates** — if only k=1 and k=30 are candidates, the chain may have nowhere intermediate to go.
- **Try fewer k candidates** — too many dense candidates can spread probability thin and make the chain indecisive.

## Acceptance rate is < 10 % or > 70 %

Both extremes indicate poor mixing. From [[Technical_details_sampling_parameters|the sampling parameters doc]]:

- **Too low** → `proposal_width` is too large. Proposals land in low-density regions and get rejected. Decrease it.
- **Too high** → `proposal_width` is too small. The chain accepts almost everything but moves in tiny increments. Increase it.

Target acceptance rate: **20–50 %**.

## Slow mixing (high autocorrelation)

If consecutive draws are nearly identical:

- Increase `proposal_width` (within reason) to make β explore faster.
- Ensure `thinning` is high enough — if you keep `thinning = 1`, neighboring samples will be correlated by construction.
- Check that the Gibbs step is actually exploring different k values. If k changes rarely, the chain can get trapped in one k-region.

## The two chains (k and β) disagree

Occasionally the Gibbs step will prefer k=1 while the MH step pushes β upward (which would favor larger k). This tug-of-war is **normal** — the joint posterior may have a ridge where both (k=small, β=large) and (k=large, β=small) have similar density. If the chain oscillates between these modes, it is correctly exploring a multimodal posterior. Longer runs and more thinning help capture both modes.
