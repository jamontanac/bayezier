
### Phase 3 — Posterior predictive (post-inference)

**Purpose:** Predict class probabilities for unseen test points using the posterior chain from Phase 2.
**Input:**  `X_test`, the posterior `chain` (list of (β, k) draws), plus `X_train`, `y_train`.
**Output:** `predictions` — shape `[n_test, n_classes]`, Monte Carlo average over all saved draws.
**How:** For each (β, k) in the chain, compute k-NN class counts for each test point, apply the PNN softmax, then average across draws.

**Related docs:**
- [[Training|Phase 2 (MCMC)]] — produces the posterior chain consumed here
- [[Steps_for_sampling_parameters|Sampler walkthrough]] — explains the Monte Carlo averaging that powers this step

```
Function Predict(X_test, chain, X_train, y_train):
    Test_Dist = Pairwise_Euclidean_Distances(X_test, X_train)
    Sorted_Test_Neighbors = sort each row of Test_Dist

    accumulated = zeros([n_test, n_classes])

    For each (beta, k) in chain:
        For each test point i:
            counts  = class counts of Sorted_Test_Neighbors[i][0..k]
            logits  = (beta * counts) / k
            probs   = softmax_stable(logits)
            accumulated[i] += probs

    Return accumulated / len(chain)     # Monte Carlo average
```
