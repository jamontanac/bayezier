# JointMh Sampler

The `JointMh` sampler is the alternative to the default [[Pseudo_code/Training|Hybrid]] sampler. Instead of separating the update into a Gibbs step for k and a Metropolis-Hastings step for β, it proposes new values for **both parameters simultaneously** and accepts or rejects the joint proposal atomically.

**Related docs:**
- [[Pseudo_code/Training|Phase 2 pseudocode]] — both sampler variants side by side
- [[Steps_for_sampling_parameters|Sampler walkthrough]] — detailed explanation of the Hybrid sampler's Gibbs + MH mechanics
- [[Technical_details_sampling_parameters|Sampling parameters]] — how to choose `proposal_width`, `beta_sigma`, etc.

---

## The proposal

At each iteration the sampler draws:

$$
k^* \sim \text{Uniform}\{0,\, 1,\, \dots,\, n_{\text{candidates}} - 1\}
$$

$$
\beta^* = \beta^{(t)} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\, \sigma^2)
$$

where $\sigma$ is `proposal_width` (`beta_step` in `SamplerConfig`).

`k*` is drawn independently of the current state — it does not prefer nearby candidates. `β*` is a Gaussian perturbation centered on the current β, identical to the proposal in the Hybrid sampler.

---

## Why the proposal ratio cancels

The Metropolis-Hastings acceptance probability is:

$$
\alpha = \min\!\left(1,\; \frac{p(k^*, \beta^* \mid \text{Data})}{p(k, \beta \mid \text{Data})} \cdot \frac{q(k^*, \beta^* \to k, \beta)}{q(k, \beta \to k^*, \beta^*)}\right)
$$

The proposal ratio is the second fraction. It decomposes as:

$$
\frac{q(k^*, \beta^* \to k, \beta)}{q(k, \beta \to k^*, \beta^*)}
= \underbrace{\frac{1/n_{\text{candidates}}}{1/n_{\text{candidates}}}}_{\text{k proposal}} \cdot \underbrace{\frac{\mathcal{N}(\beta^* - \beta;\, 0, \sigma^2)}{\mathcal{N}(\beta - \beta^*;\, 0, \sigma^2)}}_{\text{β proposal}}
= 1 \cdot 1 = 1
$$

Both ratios are 1:
- The k proposal is independent uniform in both directions ($1/n$ forward, $1/n$ backward).
- The Gaussian proposal is symmetric ($f(x - \mu) = f(\mu - x)$).

This means the acceptance probability reduces to just the posterior ratio:

$$
\alpha = \min\!\left(1,\; \frac{p(k^*, \beta^* \mid \text{Data})}{p(k, \beta \mid \text{Data})}\right)
$$

No correction term is needed.

---

## Acceptance / rejection rule

```
If beta_prop <= 0:
    Reject immediately (half-normal prior assigns zero density).

Else:
    log_alpha = log_joint(k_prop, beta_prop) - log_joint(k_index, beta)
    u ~ Uniform(0, 1)
    If log(u) < log_alpha:
        Accept: k_index = k_prop, beta = beta_prop
    Else:
        Reject: keep current k_index and beta
```

The `log(u) < log_alpha` test is equivalent to `u < exp(log_alpha)` = `u < min(1, posterior_ratio)`, but avoids exponentiation of large numbers.

---

## Tradeoffs vs Hybrid

| | Hybrid | JointMh |
|---|---|---|
| k update | Exact Gibbs (exhaustive softmax) | Single MH proposal (independent uniform) |
| β update | MH (Gaussian) | MH (Gaussian) — same |
| k and β coupled? | No — updated sequentially | Yes — accepted or rejected together |
| Acceptance rate | Higher (k always moves optimally) | Lower (joint proposal must be "uphill" in both dimensions) |
| k exploration | Concentrates quickly on high-mass values | Can jump across the k space in one step |
| Implementation complexity | Higher (two distinct update mechanisms) | Lower (one MH step) |

### When JointMh can outperform Hybrid

The Hybrid sampler's Gibbs step concentrates k toward its current maximum very quickly. This is efficient when the posterior has a single dominant k, but can cause the chain to get *stuck* when:
- The posterior is **bimodal in k** — there are two k values with similar probability separated by a trough (e.g. k=1 and k=15 are both good, k=7 is not). The Gibbs step will always jump to the local mode of the full conditional; it never crosses the trough.
- **k and β are strongly correlated** — a good (k, β) pair lives far from the current state in both dimensions simultaneously. Hybrid updates them one at a time and may not easily reach the joint mode.

JointMh proposes both at once, which means it can jump directly to (k=15, β=3.2) even from (k=1, β=0.5) in a single step. The tradeoff is a lower acceptance rate.

### When JointMh underperforms

- **Many k candidates** — the independent uniform proposal wastes $\frac{n-1}{n}$ proposals on suboptimal k values. With 20 candidates, only 1 in 20 proposals will hit the correct k even in an ideal scenario. Hybrid always picks the right k.
- **Tight β posterior** — if the data strongly constrains β to a narrow region, the Gaussian proposal for β will be rejected most of the time regardless of k. This poor β acceptance drags down the joint acceptance rate.

### Practical recommendation

Use `Hybrid` (the default) unless you have evidence of:
- A bimodal posterior over k (rare in practice), or
- Strong cross-parameter correlation (can be diagnosed from trace plots showing k and β oscillating together).

---

## Diagnostics

To check whether JointMh is mixing properly:

1. **Acceptance rate** — log how often the joint proposal is accepted. Target: 5–25 % (lower than Hybrid because the proposal space is 2D).
2. **k trace** — check whether the chain visits more than one k value. If k never changes, the proposal width for β may be too large (causing all β proposals to be rejected, which also rejects k).
3. **β trace** — should look like a diffuse random walk. If β is constant, β proposals are failing; decrease `proposal_width`.
4. **Comparison with Hybrid** — run both and compare the posterior means of k and β. Large disagreements indicate mixing problems in one of the chains.
