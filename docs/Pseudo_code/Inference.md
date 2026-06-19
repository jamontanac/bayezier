
### Phase 3 — Posterior predictive (post-inference)

**Purpose:** Predict class probabilities for unseen test points using the posterior chain from Phase 2.
**Input:**  `X_test`, the posterior `chain` (list of (β, k) draws), plus `X_train`, `y_train`.
**Output:** `predictions` — shape `[n_test, n_classes]`, Monte Carlo average over all saved draws.
**How:** For each (β, k) in the chain, compute k-NN class counts for each test point, apply the PNN softmax, then average across draws.

**Related docs:**
- [[Training|Phase 2 (MCMC)]] — produces the posterior chain consumed here
- [[Steps_for_sampling_parameters|Sampler walkthrough]] — explains the Monte Carlo averaging that powers this step

---

## High-level pseudocode

```
Function Predict(X_test, chain, X_train, y_train):
    Test_Dist = Pairwise_Euclidean_Distances(X_test, X_train)
    Sorted_Test_Neighbors = sort each row of Test_Dist by distance (index as tie-break)

    accumulated = zeros([n_test, n_classes])

    For each (beta, k) in chain:
        For each test point i:
            counts  = class counts of Sorted_Test_Neighbors[i][0..k]
            logits  = (beta * counts) / k
            probs   = softmax_stable(logits)
            accumulated[i] += probs

    Return accumulated / len(chain)     # Monte Carlo average
```

---

## Implementation breakdown

The three building blocks needed before the main prediction loop can be assembled.

### 1. `batch_sorted_neighbors(x_test, x_train, k_max) -> Vec<Vec<usize>>`

**What it does:** For every test point, returns the indices of the `k_max` nearest training points in ascending distance order.

**Why precompute up to `k_max`:** Each draw in the chain may use a different `k`. Rather than re-sorting for every draw (which would be O(chain_len × n_test × n_train log n_train)), we sort once to depth `k_max` and then read a prefix of length `k` for each draw.

**Algorithm:**

```
Function batch_sorted_neighbors(X_test, X_train, k_max):
    result = empty list

    For each test point i:
        # Compute squared Euclidean distances to all training points.
        dists = [(j, sq_euclidean(X_test[i], X_train[j])) for j in 0..n_train]

        # Sort (distance ASC, index ASC) — same tie-break rule as knn.rs and precompute.rs.
        Sort dists by (distance, index) ascending

        # Keep only the k_max closest neighbor indices.
        result[i] = [j for (j, _) in dists[0..k_max]]

    Return result    # shape [n_test, k_max]
```

**Output shape:** `Vec<Vec<usize>>` where `result[i][rank]` is the training-point index of the (rank+1)-th nearest neighbor of test point `i`.

**Notes:**
- Use squared distances (no `sqrt`) for consistency with `knn.rs` and `precompute.rs`.
- The tie-break rule (index ASC) must match `precompute.rs::build_count_tensor` to ensure train-time and test-time neighbor ordering is consistent.
- `k_max` is the largest k in the posterior chain — `chain.iter().map(|d| d.k).max()`. All draws can then index into the same pre-sorted list.

---

### 2. `softmax_stable(logits: &[f64]) -> Vec<f64>`

**What it does:** Converts a vector of unnormalized log-scores into a valid probability distribution that sums to 1.

**Why the naive approach fails:** `[exp(x) / sum(exp(x))]` overflows for logits above ~700 and underflows to all-zeros for logits below ~−700. In this model, logits are `β * counts / k`. When `β` is large (say 10) and `k = 1`, a count of 1 gives logit 10, which is fine, but if counts grow (large k, many neighbors), logits can become large.

**Algorithm (log-sum-exp trick):**

```
Function softmax_stable(logits):
    C = max(logits)                         # shift constant
    shifted = [x - C for x in logits]      # largest becomes 0 → exp(0) = 1
    exps = [exp(x) for x in shifted]
    total = sum(exps)
    Return [e / total for e in exps]        # sums to 1 by construction
```

**Properties to test:**
- Output sums to 1.0 (within floating-point tolerance).
- For uniform logits (`[c, c, c]`), output is `[1/n, 1/n, 1/n]`.
- For `logits = [-inf, 0, -inf]`, output is `[0, 1, 0]` (numerical underflow collapses dominated classes to 0).

---

### 3. `extract_class_counts(sorted_neighbors_i, k, y_train, n_classes) -> Vec<usize>`

**What it does:** Counts how many of the `k` nearest neighbors belong to each class.

```
Function extract_class_counts(neighbor_indices, k, y_train, n_classes):
    counts = zero array, length n_classes
    For rank in 0..k:
        counts[y_train[neighbor_indices[rank]]] += 1
    Return counts
```

This is a simple prefix scan over the pre-sorted neighbor list. No sorting is needed here — the heavy work was done in `batch_sorted_neighbors`.

---

### 4. `predict_proba(x_test, x_train, y_train, chain, n_classes) -> Vec<Vec<f64>>`

Assembles the three helpers above into the Monte Carlo averaging loop.

```
Function predict_proba(X_test, X_train, y_train, chain, n_classes):
    n_test   = len(X_test)
    n_draws  = len(chain)
    k_max    = max(d.k for d in chain)

    # O(n_test × n_train × k_max): sort once, read prefix many times.
    sorted_neighbors = batch_sorted_neighbors(X_test, X_train, k_max)

    accumulated = zeros([n_test, n_classes])

    For each draw d in chain:
        k    = d.k
        beta = d.beta
        For each test point i in 0..n_test:
            counts = extract_class_counts(sorted_neighbors[i], k, y_train, n_classes)
            logits = [(beta * c) / k for c in counts]
            probs  = softmax_stable(logits)
            accumulated[i] += probs

    Return accumulated / n_draws
```

**Why divide at the end, not incrementally:** Dividing once at the end reduces floating-point rounding error compared to accumulating normalized fractions.

**Empty chain:** If `chain` is empty, `n_draws = 0` and division would panic. Callers should validate that `chain` is non-empty before calling `predict_proba`.

---

### 5. `argmax(probs: &[f64]) -> usize`

Converts a probability vector (output of `predict_proba`) into a hard class prediction.

```
Function argmax(probs):
    Return index of the maximum value in probs
```

**Tie-breaking:** If two classes have exactly equal probability, the index of the first maximum is returned (consistent with Rust's `Iterator::position` behavior).

**Note:** This function currently lives in `pnn-cli/main.rs` as a private helper. It moves to `pnn-core` (alongside `predict.rs`) in Step 5 of the build plan so the CLI can import it directly.

---

## Full complexity analysis

| Step | Time complexity | Notes |
|---|---|---|
| `batch_sorted_neighbors` | O(n_test × n_train log n_train) | Done once |
| Per-draw inner loop | O(n_test × k) | k ≤ k_max ≤ n_train |
| Total prediction | O(n_test × n_train log n_train + S × n_test × k_max) | S = chain length |

For typical usage (S = 500–2000, k_max ≤ 30), the sort dominates for large test sets; the chain loop dominates for large S.
