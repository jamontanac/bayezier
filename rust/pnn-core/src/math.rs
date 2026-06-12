use std::cmp::Ordering;

/// Squared Euclidean distance. Uses an explicit delta*delta loop — benchmarks
/// in this repo showed this form outperforms `.powi(2)` on the relevant workloads.
pub(crate) fn sq_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        let delta = a[i] - b[i];
        sum += delta * delta;
    }
    sum
}

/// Ordering for `(index, distance)` pairs: distance ascending, index ascending on ties.
/// Used by `knn.rs` and `precompute.rs` to guarantee consistent neighbor ordering.
pub(crate) fn cmp_dist_then_idx(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}
