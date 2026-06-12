use crate::math::{cmp_dist_then_idx, sq_euclidean};
use crate::types::{CountTensor, DataMatrix, PnnModel};

/// Squared Euclidean distance matrix between every row of `a` and every row of `b`.
/// `result[i][j]` = squared distance from `a[i]` to `b[j]`.
/// Uses squared distance (no `sqrt`) for consistency with `knn.rs`.
pub fn pairwise_sq_distances(a: &DataMatrix, b: &DataMatrix) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row_a| b.iter().map(|row_b| sq_euclidean(row_a, row_b)).collect())
        .collect()
}

/// Build the static count tensor required by the MCMC sampler.
///
/// `result[i][ki][c]` = number of training point `i`'s neighbors of class `c`
/// when using `model.k_values[ki]` nearest neighbors (self excluded).
///
/// Preconditions guaranteed by `PnnModel::new`:
/// - `k_values` is sorted ascending
/// - all `k < n_train` (after self-exclusion, `n_train - 1` neighbors exist)
/// - labels are in `0..n_classes`
pub fn build_count_tensor(model: &PnnModel) -> CountTensor {
    let n_train = model.n_train();
    let n_candidates = model.k_values.len();
    let n_classes = model.n_classes;
    let max_k = model.k_max();

    let mut tensor: CountTensor =
        vec![vec![vec![0usize; n_classes]; n_candidates]; n_train];

    for i in 0..n_train {
        // Compute squared distances to all points except self.
        let mut dists: Vec<(usize, f64)> = (0..n_train)
            .filter(|&j| j != i)
            .map(|j| (j, sq_euclidean(&model.x_train[i], &model.x_train[j])))
            .collect();

        // Sort (distance ASC, index ASC) — same tie-break rule as knn.rs::k_nearest.
        dists.sort_by(cmp_dist_then_idx);

        let mut running_counts = vec![0usize; n_classes];
        let mut ki = 0usize; // advances O(1) per snapshot

        for rank in 0..max_k {
            let (neighbor_j, _) = dists[rank];
            running_counts[model.y_train[neighbor_j]] += 1;

            let current_k = rank + 1;
            if ki < n_candidates && model.k_values[ki] == current_k {
                tensor[i][ki] = running_counts.clone();
                ki += 1;
            }
        }
    }

    tensor
}
