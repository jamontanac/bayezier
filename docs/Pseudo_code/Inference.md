
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
    # Distance metric: SQUARED Euclidean — consistent with Phase 1 and knn.rs.
    Sorted_Test_Neighbors = batch_sorted_neighbors(X_test, X_train, k_max=max(d.k for d in chain))

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

### 1. `batch_sorted_neighbors(x_test, x_train, k_max) -> Vec<Vec<usize>>`

**What it does:** For every test point, returns the indices of the `k_max` nearest training points in ascending distance order.

**Why precompute up to `k_max`:** Each draw in the chain may use a different `k`. Rather than re-sorting for every draw (which would be O(chain_len × n_test × n_train log n_train)), we sort once to depth `k_max` and then read a prefix of length `k` for each draw.

**Algorithm:**

```
Function batch_sorted_neighbors(X_test, X_train, k_max):
    result = empty list

    For each test point i:
        # Squared Euclidean distances — no sqrt, consistent with knn.rs and precompute.rs.
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
- `k_max` is derived inside `predict_proba` as `chain.iter().map(|d| d.k).max()`. All draws then index into the same pre-sorted list.

---

### 2. `softmax_stable(logits: &[f64]) -> Vec<f64>`

**What it does:** Converts a vector of unnormalized log-scores into a valid probability distribution that sums to 1.

**Why the naive approach fails:** `[exp(x) / sum(exp(x))]` overflows for logits above ~700 and underflows to all-zeros for logits below ~−700. In this model, logits are `β * counts / k`. When `β` is large and `k` is small, individual logits can easily exceed safe exponentiation range.

**Algorithm (log-sum-exp trick):**

```
Function softmax_stable(logits):
    If logits is empty: Return []          # returns Vec::new(), no panic
    C = max(logits)                        # shift constant; largest logit becomes 0
    exps  = [exp(x - C) for x in logits]  # exp(0) = 1 for the dominant class
    total = sum(exps)
    Return [e / total for e in exps]       # sums to 1 by construction
```

**Edge cases:**
- **Empty input** — returns an empty `Vec<f64>` without panicking. This cannot arise inside `predict_proba` because `n_classes > 0` is asserted before calling softmax.
- **Uniform logits** (`[c, c, c]`) — output is `[1/n, 1/n, 1/n]` exactly (shift to 0, all exp = 1, divide by n).
- **One large logit** — dominated classes underflow to near 0; the dominant class gets probability ≈ 1.

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

This is a simple prefix scan over the pre-sorted neighbor list. No sorting is needed here — the heavy work was done once in `batch_sorted_neighbors`.

---

### 4. `predict_proba(x_test, x_train, y_train, chain, n_classes) -> Vec<Vec<f64>>`

Assembles the three helpers above into the Monte Carlo averaging loop.

```
Function predict_proba(X_test, X_train, y_train, chain, n_classes):
    # Preconditions (panics if violated):
    Assert chain is non-empty
    Assert n_classes > 0

    n_test  = len(X_test)
    k_max   = max(d.k for d in chain)

    # O(n_test × n_train log n_train): sort once, read prefix many times.
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

    Return accumulated / len(chain)    # divide once at the end
```

**Why divide at the end, not incrementally:** Dividing once at the end reduces floating-point rounding error compared to normalising after each draw.

**Preconditions:**
- `chain` must be non-empty — the function panics with a clear assertion message if it is. This is a programming error, not a recoverable runtime condition; the sampler always returns a chain of the requested size.
- `n_classes` must be > 0 — also asserted. Comes from `PnnModel::n_classes`, which is validated > 0 at model-construction time.

**Empty `x_test`:** Returns an empty `Vec` without panicking — no test points, no predictions.

---

### 5. `argmax(probs: &[f64]) -> usize`

Converts a probability vector (output of `predict_proba`) into a hard class prediction.

```
Function argmax(probs):
    Return index of the maximum value in probs
```

**Tie-breaking:** If two classes have exactly equal probability, the index of the first maximum is returned.

**Empty input:** Returns 0 (the fold initial value). Cannot arise after a valid `predict_proba` call since `n_classes > 0`.

**Location:** `pnn-core/predict.rs`, re-exported from `pnn-core` at the crate root. The CLI uses this export rather than maintaining its own copy.

---

## Numerical worked example

Using the 4-point, 2-class training set from the Phase 1 worked example:

```
X_train = [[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]]
y_train = [0, 0, 1, 1]
n_classes = 2
```

Single test point: `X_test = [[0.5, 0.5]]`

Suppose the sampler returned this 2-draw chain:

```
chain = [
    PosteriorDraw { k=1, beta=2.0 },
    PosteriorDraw { k=2, beta=1.0 },
]
```

**Step 1 — `batch_sorted_neighbors` (k_max = 2)**

Squared distances from `[0.5, 0.5]` to each training point:

| j | train point | sq. dist |
|---|-------------|----------|
| 0 | [0.0, 0.0]  | 0.50     |
| 1 | [0.1, 0.1]  | 0.32     |
| 2 | [10, 10]    | 180.50   |
| 3 | [10.1,10.1] | 184.82   |

Sorted ascending: `[1, 0, 2, 3]`. Taking k_max=2: `sorted[0] = [1, 0]`.

**Step 2 — Draw 1: k=1, beta=2.0**

```
counts = extract_class_counts([1, 0], k=1, y_train) = [1, 0]
            # only neighbor j=1, class 0

logits = [2.0 * 1 / 1, 2.0 * 0 / 1] = [2.0, 0.0]

softmax([2.0, 0.0]):
    max = 2.0
    exps = [exp(0), exp(-2)] = [1.0, 0.135]
    total = 1.135
    probs = [0.881, 0.119]
```

**Step 3 — Draw 2: k=2, beta=1.0**

```
counts = extract_class_counts([1, 0], k=2, y_train) = [2, 0]
            # neighbors j=1 (class 0) and j=0 (class 0)

logits = [1.0 * 2 / 2, 1.0 * 0 / 2] = [1.0, 0.0]

softmax([1.0, 0.0]):
    max = 1.0
    exps = [1.0, exp(-1)] = [1.0, 0.368]
    total = 1.368
    probs = [0.731, 0.269]
```

**Step 4 — Accumulate and average**

```
accumulated[0] = [0.881, 0.119] + [0.731, 0.269] = [1.612, 0.388]

predictions[0] = [1.612 / 2, 0.388 / 2] = [0.806, 0.194]
```

Hard prediction: `argmax([0.806, 0.194]) = 0` → class 0. Correct, since the test point
`[0.5, 0.5]` is near the class-0 cluster.

---

## Full complexity analysis

| Step | Time complexity | Notes |
|---|---|---|
| `batch_sorted_neighbors` | O(n_test × n_train log n_train) | Full sort per test point; done once |
| Per-draw inner loop | O(n_test × k) | k ≤ k_max ≤ n_train |
| Total prediction | O(n_test × n_train log n_train + S × n_test × k_max) | S = chain length |

For typical usage (S = 500–2000, k_max ≤ 30), the sort dominates for large test sets; the chain loop dominates for large S. A partial sort (heap to depth k_max) would improve the sort term to O(n_test × n_train log k_max), but the current implementation uses a full sort for simplicity.
