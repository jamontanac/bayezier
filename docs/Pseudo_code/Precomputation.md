
### Phase 1 — Precomputation (runs once before MCMC)

**Purpose:** For every training point i, precompute the class-count vector for every candidate k value.
**Input:**  `X_train` (features), `y_train` (labels), `k_candidate_values` (sorted), `n_classes`
**Output:** `Count_Tensor` — shape `[n_train, n_candidates, n_classes]` — the k-NN class counts at each candidate k for each point.
**Why separate:** The counts depend only on fixed training data, not on β or the current k. Precomputing avoids O(n² log n) per iteration.

```
# Preconditions (must be validated by the caller):
#   k_candidate_values is sorted ascending
#   max(k_candidate_values) <= len(X_train) - 1
#
# Distance metric: SQUARED Euclidean throughout (avoids sqrt, preserves ordering).
# Count_Tensor cells are integer counts (usize), not floats.

Function Prepare_Static_Structures(X_train, y_train, k_candidate_values_SORTED, n_classes):
    n_train      = len(X_train)
    n_candidates = len(k_candidate_values_SORTED)
    max_k        = k_candidate_values_SORTED[last]

    Count_Tensor = zero integer array [n_train, n_candidates, n_classes]

    For each i in 0..n_train:
        # Exclude self before sorting so the sorted list has exactly n_train-1 entries
        Dists = [(j, squared_euclidean(X_train[i], X_train[j])) for j != i]

        # Tie-break rule: sort by (distance ASC, index ASC)
        Sort Dists by (distance, index) ascending

        # ── Step A: accumulate class counts for every possible k (1..max_k) ──
        # Running_Counts_by_k[k-1, :] = class counts for the k nearest neighbors
        Running_Counts_by_k = empty integer array [max_k, n_classes]
        Running_Counts      = integer zero array, length n_classes

        For rank in 0..max_k:                          # rank = 0, 1, ..., max_k-1
            (neighbor_j, _) = Dists[rank]
            Running_Counts[y_train[neighbor_j]] += 1    # add this neighbor's class

            # Save a snapshot after including the (rank+1)-th closest neighbor
            Running_Counts_by_k[rank, :] = COPY of Running_Counts

        # ── Step B: pick out only the k values listed in k_candidate_values ──
        ki = 0   # index into k_candidate_values_SORTED
        For current_k in 1..max_k:
            If ki < n_candidates AND k_candidate_values_SORTED[ki] == current_k:
                # Running_Counts_by_k[current_k-1] holds counts for exactly current_k neighbors
                Count_Tensor[i, ki, :] = Running_Counts_by_k[current_k - 1, :]
                ki += 1

    Return Count_Tensor
```
