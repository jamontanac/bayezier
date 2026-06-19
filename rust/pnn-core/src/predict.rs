use crate::math::{cmp_dist_then_idx, sq_euclidean};
use crate::types::{DataMatrix, Labels, PosteriorDraw};

// ── Softmax ───────────────────────────────────────────────────────────────────

/// Numerically stable softmax: subtracts max(logits) before exp to prevent
/// overflow when any logit is large.
pub fn softmax_stable(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}

// ── Argmax ────────────────────────────────────────────────────────────────────

/// Index of the largest element. Returns 0 for an empty slice.
/// On ties, returns the first (lowest) index.
pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v { (i, v) } else { (best_i, best_v) }
        })
        .0
}

// ── Neighbor sorting ──────────────────────────────────────────────────────────

/// For each test point, returns the indices of its `k_max` nearest training
/// points in ascending-distance order.
///
/// Tie-break: index ascending — consistent with `knn.rs` and `precompute.rs`.
pub fn batch_sorted_neighbors(
    x_test: &DataMatrix,
    x_train: &DataMatrix,
    k_max: usize,
) -> Vec<Vec<usize>> {
    x_test
        .iter()
        .map(|test_point| {
            let mut dists: Vec<(usize, f64)> = x_train
                .iter()
                .enumerate()
                .map(|(j, train_point)| (j, sq_euclidean(test_point, train_point)))
                .collect();
            dists.sort_by(cmp_dist_then_idx);
            dists.iter().take(k_max).map(|(j, _)| *j).collect()
        })
        .collect()
}

// ── Class count extraction ────────────────────────────────────────────────────

fn extract_class_counts(
    sorted_neighbors: &[usize],
    k: usize,
    y_train: &Labels,
    n_classes: usize,
) -> Vec<usize> {
    let mut counts = vec![0usize; n_classes];
    for &j in sorted_neighbors.iter().take(k) {
        counts[y_train[j]] += 1;
    }
    counts
}

// ── Posterior predictive ──────────────────────────────────────────────────────

/// Monte Carlo average of the PNN posterior predictive.
///
/// Sorts test-point neighbors once up to `k_max`, then for each (β, k) draw
/// in `chain` accumulates the softmax probability vector. Returns the mean
/// over all draws — shape `[n_test][n_classes]`.
///
/// Panics if `chain` is empty or `n_classes` is 0.
pub fn predict_proba(
    x_test: &DataMatrix,
    x_train: &DataMatrix,
    y_train: &Labels,
    chain: &[PosteriorDraw],
    n_classes: usize,
) -> Vec<Vec<f64>> {
    assert!(!chain.is_empty(), "chain must be non-empty");
    assert!(n_classes > 0, "n_classes must be > 0");

    let n_test = x_test.len();
    let k_max = chain.iter().map(|d| d.k).max().unwrap(); // safe: chain non-empty

    // Sort neighbors once; every draw reads a prefix of length d.k.
    let sorted = batch_sorted_neighbors(x_test, x_train, k_max);

    let mut accumulated = vec![vec![0.0_f64; n_classes]; n_test];

    for draw in chain {
        let beta = draw.beta;
        let k = draw.k;
        let k_f = k as f64;
        for i in 0..n_test {
            let counts = extract_class_counts(&sorted[i], k, y_train, n_classes);
            let logits: Vec<f64> = counts.iter().map(|&c| beta * c as f64 / k_f).collect();
            let probs = softmax_stable(&logits);
            for (a, p) in accumulated[i].iter_mut().zip(probs) {
                *a += p;
            }
        }
    }

    let n_draws = chain.len() as f64;
    for row in &mut accumulated {
        for p in row.iter_mut() {
            *p /= n_draws;
        }
    }
    accumulated
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // --- softmax_stable ---------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        for logits in [
            vec![1.0, 2.0, 3.0],
            vec![-100.0, -100.0],
            vec![0.0],
            vec![1000.0, 999.0, 998.0],
        ] {
            let probs = softmax_stable(&logits);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "logits {logits:?}: sum={sum}");
        }
    }

    #[test]
    fn softmax_uniform_logits_gives_uniform_probs() {
        let probs = softmax_stable(&[2.0, 2.0, 2.0]);
        for &p in &probs {
            assert!(approx_eq(p, 1.0 / 3.0));
        }
    }

    #[test]
    fn softmax_single_element_returns_one() {
        assert!(approx_eq(softmax_stable(&[42.0])[0], 1.0));
    }

    #[test]
    fn softmax_large_value_dominates() {
        let probs = softmax_stable(&[100.0, 0.0]);
        assert!(probs[0] > 0.999);
    }

    #[test]
    fn softmax_stable_with_large_negatives_no_nan() {
        // Would underflow to NaN without the max-shift trick.
        let probs = softmax_stable(&[-1000.0, -1001.0]);
        assert!(probs.iter().all(|p| p.is_finite()));
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < TOL);
    }

    // --- argmax -----------------------------------------------------------------

    #[test]
    fn argmax_returns_index_of_maximum() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.9, 0.1, 0.3]), 0);
        assert_eq!(argmax(&[0.1, 0.3, 0.9]), 2);
    }

    #[test]
    fn argmax_tie_returns_first_index() {
        assert_eq!(argmax(&[0.5, 0.5]), 0);
        assert_eq!(argmax(&[0.3, 0.3, 0.3]), 0);
    }

    #[test]
    fn argmax_single_element() {
        assert_eq!(argmax(&[0.7]), 0);
    }

    // --- batch_sorted_neighbors -------------------------------------------------
    //
    // Hand-checked against the worked example from precompute_test.rs:
    //   X = [[1,2],[5,3],[2,1]]
    //   Query [1,2]: dist to [5,3]=17, dist to [2,1]=2   → order: [2, 1]
    //   Query [5,3]: dist to [1,2]=17, dist to [2,1]=13  → order: [2, 0]
    //   Query [2,1]: dist to [1,2]=2,  dist to [5,3]=13  → order: [0, 1]

    #[test]
    fn batch_sorted_neighbors_hand_checked() {
        let x = vec![vec![1.0, 2.0], vec![5.0, 3.0], vec![2.0, 1.0]];
        let result = batch_sorted_neighbors(&x, &x, 2);
        // Each point's 2 nearest (self would be rank 0, so excluded only if this
        // were training; here x_test == x_train, so self IS included).
        // i=0: self(d=0) < [2,1](d=2) < [5,3](d=17)
        assert_eq!(result[0], vec![0, 2]);
        // i=1: self(d=0) < [2,1](d=13) < [1,2](d=17)
        assert_eq!(result[1], vec![1, 2]);
        // i=2: self(d=0) < [1,2](d=2) < [5,3](d=13)
        assert_eq!(result[2], vec![2, 0]);
    }

    #[test]
    fn batch_sorted_neighbors_k_max_limits_output_length() {
        let x_test = vec![vec![0.0]];
        let x_train = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = batch_sorted_neighbors(&x_test, &x_train, 2);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn batch_sorted_neighbors_tie_break_by_lower_index() {
        // x_test[0] = [1.0] is equidistant from [0.0] and [2.0] (both d=1).
        // Index 0 < index 1, so [0.0] should come first.
        let x_test = vec![vec![1.0]];
        let x_train = vec![vec![0.0], vec![2.0]];
        let result = batch_sorted_neighbors(&x_test, &x_train, 2);
        assert_eq!(result[0], vec![0, 1]);
    }

    // --- predict_proba ----------------------------------------------------------

    fn toy_train() -> (DataMatrix, Labels) {
        (
            vec![
                vec![0.0, 0.0],
                vec![0.1, 0.1],
                vec![10.0, 10.0],
                vec![10.1, 10.1],
            ],
            vec![0, 0, 1, 1],
        )
    }

    #[test]
    fn predict_proba_sums_to_one_per_point() {
        let (x_train, y_train) = toy_train();
        let x_test = vec![vec![0.5, 0.5], vec![9.5, 9.5]];
        let chain = vec![
            PosteriorDraw { k_index: 0, k: 1, beta: 1.0 },
            PosteriorDraw { k_index: 1, k: 2, beta: 2.0 },
        ];
        let probs = predict_proba(&x_test, &x_train, &y_train, &chain, 2);
        for (i, row) in probs.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "test point {i}: sum={sum}");
        }
    }

    #[test]
    fn predict_proba_beta_zero_gives_uniform() {
        // With beta=0, all logits are 0 → softmax → uniform over n_classes.
        let (x_train, y_train) = toy_train();
        let x_test = vec![vec![0.5, 0.5]];
        let chain = vec![PosteriorDraw { k_index: 0, k: 1, beta: 0.0 }];
        let probs = predict_proba(&x_test, &x_train, &y_train, &chain, 2);
        assert!(approx_eq(probs[0][0], 0.5));
        assert!(approx_eq(probs[0][1], 0.5));
    }

    #[test]
    fn predict_proba_concentrated_on_correct_class() {
        // Test point near class 0; high beta → probability mass on class 0.
        let (x_train, y_train) = toy_train();
        let x_test = vec![vec![0.5, 0.5]];
        let chain = vec![PosteriorDraw { k_index: 0, k: 1, beta: 10.0 }];
        let probs = predict_proba(&x_test, &x_train, &y_train, &chain, 2);
        assert!(probs[0][0] > 0.99, "expected class 0 prob > 0.99, got {}", probs[0][0]);
    }

    #[test]
    fn predict_proba_is_deterministic() {
        let (x_train, y_train) = toy_train();
        let x_test = vec![vec![0.5, 0.5], vec![9.5, 9.5]];
        let chain = vec![
            PosteriorDraw { k_index: 0, k: 1, beta: 2.0 },
            PosteriorDraw { k_index: 1, k: 2, beta: 3.0 },
        ];
        let a = predict_proba(&x_test, &x_train, &y_train, &chain, 2);
        let b = predict_proba(&x_test, &x_train, &y_train, &chain, 2);
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (pa, pb) in ra.iter().zip(rb.iter()) {
                assert!(approx_eq(*pa, *pb));
            }
        }
    }
}
