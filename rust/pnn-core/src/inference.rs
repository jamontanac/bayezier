use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::model::{log_joint, log_sum_exp};
use crate::precompute::build_count_tensor;
use crate::types::{CountTensor, InferenceMethod, PnnModel, PosteriorDraw, SamplerConfig};

// ── Internal helpers ──────────────────────────────────────────────────────────

fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Draw an index from the categorical distribution defined by `log_weights`.
/// Applies the log-sum-exp trick so large-negative weights don't underflow.
fn sample_categorical(log_weights: &[f64], rng: &mut StdRng) -> usize {
    let lse = log_sum_exp(log_weights);
    let u: f64 = rng.r#gen();
    let mut cumsum = 0.0_f64;
    for (i, &lw) in log_weights.iter().enumerate() {
        cumsum += (lw - lse).exp();
        if u <= cumsum {
            return i;
        }
    }
    log_weights.len() - 1 // absorbs floating-point rounding at the tail
}

// ── Hybrid sampler ────────────────────────────────────────────────────────────
//
// Each iteration:
//   1. Gibbs step — enumerate all k candidates, softmax-normalize log-joints,
//      sample k exactly from its full conditional (no accept/reject needed).
//   2. MH step  — propose β* ~ N(β, σ²); accept/reject via log-ratio.
//
// The MH step uses the k just sampled by the Gibbs step (k^(t+1), not k^(t)).
// This tight coupling means a new k immediately informs the β proposal.

#[allow(unused_assignments)] // initial k_index is overwritten by the Gibbs step on iteration 0
fn run_hybrid(
    model: &PnnModel,
    count_tensor: &CountTensor,
    config: &SamplerConfig,
) -> Vec<PosteriorDraw> {
    debug_assert!(config.thinning > 0, "thinning must be >= 1");

    let mut rng = make_rng(config.seed);
    let n_candidates = model.k_values.len();
    let n_iters = config.burn_in + config.n_samples * config.thinning;
    let normal = Normal::new(0.0_f64, config.beta_step).expect("beta_step must be > 0");

    let mut beta = 1.0_f64;
    let mut k_index = n_candidates / 2; // start at median candidate
    let mut chain = Vec::with_capacity(config.n_samples);

    for step in 0..n_iters {
        // --- Gibbs step: sample k from its exact full conditional ---
        let log_w: Vec<f64> = (0..n_candidates)
            .map(|ki| {
                log_joint(
                    count_tensor,
                    &model.y_train,
                    ki,
                    &model.k_values,
                    beta,
                    config.beta_sigma,
                )
            })
            .collect();
        k_index = sample_categorical(&log_w, &mut rng);

        // --- MH step: propose β* = β + N(0, σ²) ---
        let beta_prop = beta + normal.sample(&mut rng);
        if beta_prop > 0.0 {
            // log acceptance ratio: log p(β*|k,data) - log p(β|k,data)
            // Gaussian proposal is symmetric so proposal ratio cancels.
            let log_alpha = log_joint(
                count_tensor,
                &model.y_train,
                k_index,
                &model.k_values,
                beta_prop,
                config.beta_sigma,
            ) - log_joint(
                count_tensor,
                &model.y_train,
                k_index,
                &model.k_values,
                beta,
                config.beta_sigma,
            );
            // log(U) < log_alpha  ⟺  U < exp(log_alpha)  ⟺  accept
            if rng.r#gen::<f64>().ln() < log_alpha {
                beta = beta_prop;
            }
        }
        // beta_prop <= 0: half-normal prior gives -∞, so we always reject.

        // --- Record post-burn-in, every `thinning` steps ---
        if step >= config.burn_in && (step - config.burn_in) % config.thinning == 0 {
            chain.push(PosteriorDraw {
                k_index,
                k: model.k_values[k_index],
                beta,
            });
        }
    }

    chain
}

// ── JointMh sampler ───────────────────────────────────────────────────────────
//
// Each iteration proposes (k*, β*) jointly and accepts or rejects both atomically.
//
//   k* ~ Uniform{0, …, n_candidates − 1}   (independent of current k)
//   β* ~ N(β, σ²)
//
// Because the k proposal is independent-uniform, q(k→k') = q(k'→k) = 1/n.
// Because the β proposal is a symmetric Gaussian, q(β→β') = q(β'→β).
// Both proposal ratios cancel, leaving the standard posterior ratio as the
// Metropolis acceptance probability.
//
// Tradeoffs vs Hybrid:
//   + Can jump across the k space in a single step.
//   − Lower acceptance rate: the joint proposal must be "uphill" in both
//     dimensions simultaneously.
//   − If β happens to be a bad proposal, k is also rejected even if k* alone
//     would have been a good choice.

fn run_joint_mh(
    model: &PnnModel,
    count_tensor: &CountTensor,
    config: &SamplerConfig,
) -> Vec<PosteriorDraw> {
    debug_assert!(config.thinning > 0, "thinning must be >= 1");

    let mut rng = make_rng(config.seed);
    let n_candidates = model.k_values.len();
    let n_iters = config.burn_in + config.n_samples * config.thinning;
    let normal = Normal::new(0.0_f64, config.beta_step).expect("beta_step must be > 0");

    let mut beta = 1.0_f64;
    let mut k_index = n_candidates / 2;
    let mut chain = Vec::with_capacity(config.n_samples);

    for step in 0..n_iters {
        // --- Joint proposal ---
        let k_prop = rng.gen_range(0..n_candidates);
        let beta_prop = beta + normal.sample(&mut rng);

        if beta_prop > 0.0 {
            let log_alpha = log_joint(
                count_tensor,
                &model.y_train,
                k_prop,
                &model.k_values,
                beta_prop,
                config.beta_sigma,
            ) - log_joint(
                count_tensor,
                &model.y_train,
                k_index,
                &model.k_values,
                beta,
                config.beta_sigma,
            );
            if rng.r#gen::<f64>().ln() < log_alpha {
                k_index = k_prop;
                beta = beta_prop;
            }
        }

        if step >= config.burn_in && (step - config.burn_in) % config.thinning == 0 {
            chain.push(PosteriorDraw {
                k_index,
                k: model.k_values[k_index],
                beta,
            });
        }
    }

    chain
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn sample_posterior(model: &PnnModel, config: &SamplerConfig) -> Vec<PosteriorDraw> {
    let count_tensor = build_count_tensor(model);
    match config.method {
        InferenceMethod::Hybrid => run_hybrid(model, &count_tensor, config),
        InferenceMethod::JointMh => run_joint_mh(model, &count_tensor, config),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InferenceMethod, PnnModel, SamplerConfig};

    // 4-point, 2-class dataset with clear spatial separation.
    // Classes 0 and 1 cluster near (0,0) and (10,10) respectively.
    fn toy_model() -> PnnModel {
        PnnModel::new(
            vec![
                vec![0.0, 0.0],
                vec![0.1, 0.1],
                vec![10.0, 10.0],
                vec![10.1, 10.1],
            ],
            vec![0, 0, 1, 1],
            2,
            vec![1, 2, 3],
        )
        .unwrap()
    }

    fn seeded(method: InferenceMethod) -> SamplerConfig {
        SamplerConfig {
            method,
            n_samples: 50,
            burn_in: 10,
            thinning: 1,
            beta_step: 0.3,
            beta_sigma: 5.0,
            seed: Some(42),
        }
    }

    // --- chain length ------------------------------------------------------------

    #[test]
    fn hybrid_chain_length_equals_n_samples() {
        assert_eq!(sample_posterior(&toy_model(), &seeded(InferenceMethod::Hybrid)).len(), 50);
    }

    #[test]
    fn joint_mh_chain_length_equals_n_samples() {
        assert_eq!(
            sample_posterior(&toy_model(), &seeded(InferenceMethod::JointMh)).len(),
            50
        );
    }

    #[test]
    fn chain_length_respects_thinning() {
        let config = SamplerConfig {
            n_samples: 7,
            burn_in: 5,
            thinning: 4,
            seed: Some(1),
            ..seeded(InferenceMethod::Hybrid)
        };
        assert_eq!(sample_posterior(&toy_model(), &config).len(), 7);
    }

    #[test]
    fn zero_n_samples_returns_empty_chain() {
        let config = SamplerConfig { n_samples: 0, ..seeded(InferenceMethod::Hybrid) };
        assert!(sample_posterior(&toy_model(), &config).is_empty());
    }

    // --- draw validity ----------------------------------------------------------

    #[test]
    fn all_beta_values_are_positive() {
        for method in [InferenceMethod::Hybrid, InferenceMethod::JointMh] {
            let chain = sample_posterior(&toy_model(), &seeded(method));
            assert!(chain.iter().all(|d| d.beta > 0.0), "{method:?}: beta <= 0 found");
        }
    }

    #[test]
    fn all_k_index_and_k_are_consistent() {
        let model = toy_model();
        for method in [InferenceMethod::Hybrid, InferenceMethod::JointMh] {
            let chain = sample_posterior(&model, &seeded(method));
            for draw in &chain {
                assert!(
                    draw.k_index < model.k_values.len(),
                    "{method:?}: k_index {} out of bounds",
                    draw.k_index
                );
                assert_eq!(
                    draw.k,
                    model.k_values[draw.k_index],
                    "{method:?}: k mismatch at k_index {}",
                    draw.k_index
                );
            }
        }
    }

    // --- reproducibility --------------------------------------------------------

    #[test]
    fn hybrid_fixed_seed_is_reproducible() {
        let model = toy_model();
        let config = seeded(InferenceMethod::Hybrid);
        let a = sample_posterior(&model, &config);
        let b = sample_posterior(&model, &config);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.k_index, y.k_index);
            assert!((x.beta - y.beta).abs() < 1e-15);
        }
    }

    #[test]
    fn joint_mh_fixed_seed_is_reproducible() {
        let model = toy_model();
        let config = seeded(InferenceMethod::JointMh);
        let a = sample_posterior(&model, &config);
        let b = sample_posterior(&model, &config);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.k_index, y.k_index);
            assert!((x.beta - y.beta).abs() < 1e-15);
        }
    }

    // --- chain moves ------------------------------------------------------------

    #[test]
    fn beta_is_not_constant_over_chain() {
        // With proposal_width=0.3 and a non-trivial dataset the MH step must
        // accept at least some proposals over 200 recorded draws.
        let config = SamplerConfig {
            n_samples: 200,
            burn_in: 100,
            ..seeded(InferenceMethod::Hybrid)
        };
        let chain = sample_posterior(&toy_model(), &config);
        let first = chain[0].beta;
        assert!(
            chain.iter().any(|d| (d.beta - first).abs() > 1e-9),
            "beta never moved from {first}; proposal may be too narrow"
        );
    }
}
