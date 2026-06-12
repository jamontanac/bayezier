use crate::types::{CountTensor, Labels};

// ── Numerical utilities ───────────────────────────────────────────────────────

/// Numerically stable log-sum-exp: ln(Σ exp(lw_i)).
/// Returns `NEG_INFINITY` for an empty slice or when all weights are `-inf`.
pub fn log_sum_exp(log_weights: &[f64]) -> f64 {
    if log_weights.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + log_weights.iter().map(|&lw| (lw - max).exp()).sum::<f64>().ln()
}

// ── Priors ────────────────────────────────────────────────────────────────────

/// Log prior for k: discrete uniform over `n_candidates` candidate values.
pub fn log_prior_k(n_candidates: usize) -> f64 {
    -(n_candidates as f64).ln()
}

/// Log prior for beta: half-normal with scale `sigma`.
/// Returns `NEG_INFINITY` for `beta <= 0`.
pub fn log_prior_beta(beta: f64, sigma: f64) -> f64 {
    if beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // Half-normal log-PDF: ln( sqrt(2/pi)/sigma ) - beta^2 / (2*sigma^2)
    0.5 * (2.0 / std::f64::consts::PI).ln() - sigma.ln() - 0.5 * (beta / sigma).powi(2)
}

// ── Likelihood ────────────────────────────────────────────────────────────────

/// Pseudo-log-likelihood summed over all training points at candidate `k_index`.
///
/// For each training point `i`:
///   logits_c = beta * count_tensor[i][k_index][c] / k
///   contribution = logits[true_class] - log_sum_exp(logits)
pub fn log_likelihood(
    count_tensor: &CountTensor,
    y_train: &Labels,
    k_index: usize,
    k_values: &[usize],
    beta: f64,
) -> f64 {
    let k = k_values[k_index] as f64;
    y_train
        .iter()
        .enumerate()
        .map(|(i, &true_class)| {
            let counts = &count_tensor[i][k_index];
            let logits: Vec<f64> = counts.iter().map(|&c| beta * c as f64 / k).collect();
            logits[true_class] - log_sum_exp(&logits)
        })
        .sum()
}

// ── Joint log-posterior ───────────────────────────────────────────────────────

/// Log joint posterior: log_likelihood + log_prior_k + log_prior_beta.
/// This is the function evaluated at every MCMC step.
pub fn log_joint(
    count_tensor: &CountTensor,
    y_train: &Labels,
    k_index: usize,
    k_values: &[usize],
    beta: f64,
    beta_sigma: f64,
) -> f64 {
    log_likelihood(count_tensor, y_train, k_index, k_values, beta)
        + log_prior_k(k_values.len())
        + log_prior_beta(beta, beta_sigma)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // --- log_sum_exp ---

    #[test]
    fn log_sum_exp_empty_returns_neg_infinity() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_single_element_is_identity() {
        assert!(approx_eq(log_sum_exp(&[3.7]), 3.7));
        assert!(approx_eq(log_sum_exp(&[-100.0]), -100.0));
    }

    #[test]
    fn log_sum_exp_equal_weights_is_log_n_plus_weight() {
        // log_sum_exp([0, 0]) = ln(2)
        assert!(approx_eq(
            log_sum_exp(&[0.0, 0.0]),
            std::f64::consts::LN_2
        ));
        // log_sum_exp([a, a, a]) = a + ln(3)
        let a = -5.0_f64;
        assert!(approx_eq(
            log_sum_exp(&[a, a, a]),
            a + 3.0_f64.ln()
        ));
    }

    #[test]
    fn log_sum_exp_is_symmetric() {
        let a = log_sum_exp(&[1.0, 3.0, -2.0]);
        let b = log_sum_exp(&[-2.0, 1.0, 3.0]);
        assert!(approx_eq(a, b));
    }

    #[test]
    fn log_sum_exp_numerically_stable_with_large_negatives() {
        // Would underflow to 0 without the max shift, giving ln(0) = -inf
        let result = log_sum_exp(&[-1000.0, -1000.0]);
        let expected = -1000.0 + std::f64::consts::LN_2;
        assert!(approx_eq(result, expected));
    }

    #[test]
    fn log_sum_exp_all_neg_infinity_returns_neg_infinity() {
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    // --- log_prior_k ---

    #[test]
    fn log_prior_k_single_candidate_is_zero() {
        // Uniform over 1 value → probability 1 → log 0
        assert!(approx_eq(log_prior_k(1), 0.0));
    }

    #[test]
    fn log_prior_k_four_candidates() {
        assert!(approx_eq(log_prior_k(4), -4.0_f64.ln()));
    }

    #[test]
    fn log_prior_k_is_non_positive() {
        for n in 1..=20 {
            assert!(log_prior_k(n) <= 0.0);
        }
    }

    // --- log_prior_beta ---

    #[test]
    fn log_prior_beta_non_positive_returns_neg_infinity() {
        assert_eq!(log_prior_beta(0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(log_prior_beta(-1.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(log_prior_beta(-100.0, 5.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_prior_beta_positive_is_finite_negative() {
        let lp = log_prior_beta(1.0, 5.0);
        assert!(lp.is_finite() && lp < 0.0);
    }

    #[test]
    fn log_prior_beta_hand_checked_value() {
        // 0.5*ln(2/pi) - ln(5) - 0.5*(1/5)^2
        let expected = 0.5 * (2.0 / std::f64::consts::PI).ln() - 5.0_f64.ln() - 0.5 * 0.04;
        assert!(approx_eq(log_prior_beta(1.0, 5.0), expected));
    }

    #[test]
    fn log_prior_beta_decreasing_away_from_zero() {
        // Half-normal mode is at 0; density strictly decreases for beta > 0
        let lp1 = log_prior_beta(0.5, 5.0);
        let lp2 = log_prior_beta(2.0, 5.0);
        let lp3 = log_prior_beta(10.0, 5.0);
        assert!(lp1 > lp2 && lp2 > lp3);
    }

    // --- log_likelihood ---

    // Hand-built count tensor from the worked example in pnn_build_plan.md:
    //   X_train = [[1,2],[5,3],[2,1]], y_train = [0,1,0], n_classes=2, k_values=[1,2]
    //
    //   Count_Tensor[0] = [[1,0], [1,1]]
    //   Count_Tensor[1] = [[1,0], [2,0]]
    //   Count_Tensor[2] = [[1,0], [1,1]]
    fn worked_example() -> (CountTensor, Vec<usize>, Vec<usize>) {
        let count_tensor = vec![
            vec![vec![1, 0], vec![1, 1]],
            vec![vec![1, 0], vec![2, 0]],
            vec![vec![1, 0], vec![1, 1]],
        ];
        let y_train = vec![0usize, 1, 0];
        let k_values = vec![1usize, 2];
        (count_tensor, y_train, k_values)
    }

    #[test]
    fn log_likelihood_k1_beta1_hand_checked() {
        let (ct, y, kv) = worked_example();
        // k=1, beta=1: logits = [1.0, 0.0] for every point
        // lse([1.0, 0.0]) = 1.0 + ln(1 + exp(-1))
        let lse = 1.0 + (1.0 + (-1.0_f64).exp()).ln();
        // point 0 (true=0): 1.0 - lse
        // point 1 (true=1): 0.0 - lse
        // point 2 (true=0): 1.0 - lse
        let expected = (1.0 - lse) + (0.0 - lse) + (1.0 - lse);
        let result = log_likelihood(&ct, &y, 0, &kv, 1.0);
        assert!((result - expected).abs() < TOL, "got {result}, expected {expected}");
    }

    #[test]
    fn log_likelihood_beta_zero_gives_uniform() {
        let (ct, y, kv) = worked_example();
        // beta=0 → all logits=0 → log_p_i = -ln(n_classes) = -ln(2) for every point
        let expected = -(y.len() as f64) * 2.0_f64.ln();
        let result = log_likelihood(&ct, &y, 0, &kv, 0.0);
        assert!((result - expected).abs() < TOL, "got {result}, expected {expected}");
    }

    #[test]
    fn log_likelihood_is_negative() {
        let (ct, y, kv) = worked_example();
        for &beta in &[0.0, 0.5, 1.0, 5.0] {
            for ki in 0..kv.len() {
                assert!(log_likelihood(&ct, &y, ki, &kv, beta) <= 0.0);
            }
        }
    }

    // --- log_joint ---

    #[test]
    fn log_joint_beta_non_positive_returns_neg_infinity() {
        let (ct, y, kv) = worked_example();
        assert_eq!(log_joint(&ct, &y, 0, &kv, 0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(log_joint(&ct, &y, 0, &kv, -1.0, 5.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_joint_is_negative_for_valid_params() {
        let (ct, y, kv) = worked_example();
        let lj = log_joint(&ct, &y, 0, &kv, 1.0, 5.0);
        assert!(lj.is_finite() && lj < 0.0);
    }

    #[test]
    fn log_joint_decomposes_correctly() {
        let (ct, y, kv) = worked_example();
        let (beta, sigma, ki) = (1.0, 5.0, 0usize);
        let combined = log_joint(&ct, &y, ki, &kv, beta, sigma);
        let manual = log_likelihood(&ct, &y, ki, &kv, beta)
            + log_prior_k(kv.len())
            + log_prior_beta(beta, sigma);
        assert!(approx_eq(combined, manual));
    }
}
