use std::collections::BTreeMap;

use serde::Serialize;

use crate::types::{InferenceMethod, SamplerConfig, SamplerResult};

// ── ACF / ESS ─────────────────────────────────────────────────────────────────

/// Autocorrelation at lags 0..=max_lag.
///
/// Returns `vec![1.0]` when the chain has fewer than 2 elements or zero variance
/// (a constant chain has undefined autocorrelation; 1.0 at lag 0 is the safe sentinel).
pub fn compute_acf(values: &[f64], max_lag: usize) -> Vec<f64> {
    let n = values.len();
    if n < 2 {
        return vec![1.0];
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

    if var == 0.0 {
        return vec![1.0];
    }

    let effective_max = max_lag.min(n - 1);
    (0..=effective_max)
        .map(|lag| {
            let cov = values[..n - lag]
                .iter()
                .zip(&values[lag..])
                .map(|(&a, &b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / n as f64;
            cov / var
        })
        .collect()
}

/// Effective sample size using Geyer's truncated positive-pairs estimator.
///
/// `n` is the actual chain length (not `acf.len()` — the ACF only covers lags
/// 0..=max_lag which is typically far shorter than the chain). Passing the wrong
/// `n` caps ESS at max_lag instead of n_samples.
///
/// Forms consecutive pairs `P_s = rho_{2s} + rho_{2s+1}` and stops at the first
/// non-positive pair, preventing noise in the ACF tail from inflating the estimate.
/// Result is capped at `n` and floored at 1.0.
pub fn compute_ess(acf: &[f64], n: usize) -> f64 {
    if acf.is_empty() || n == 0 {
        return 1.0;
    }
    let n_acf = acf.len(); // number of lags available (max_lag + 1)

    let mut pair_sum = 0.0_f64;
    let mut s = 0usize;
    loop {
        let lag_even = 2 * s;
        let lag_odd = 2 * s + 1;
        if lag_odd >= n_acf {
            break;
        }
        let pair = acf[lag_even] + acf[lag_odd];
        if pair <= 0.0 {
            break;
        }
        pair_sum += pair;
        s += 1;
    }

    // ESS = n / (-1 + 2 * pair_sum).  The -1 accounts for lag 0 (rho_0 = 1).
    let denom = (2.0 * pair_sum - 1.0).max(1.0);
    (n as f64 / denom).clamp(1.0, n as f64)
}

// ── Output structs ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct KCandidateRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize)]
pub struct DiagnosticsConfig {
    pub method: String,
    pub n_samples: usize,
    pub burn_in: usize,
    pub thinning: usize,
    pub beta_step: f64,
    pub beta_sigma: f64,
    pub k_candidates: KCandidateRange,
    pub total_iterations: usize,
}

#[derive(Debug, Serialize)]
pub struct MhAcceptance {
    pub n_accepted: usize,
    pub n_proposed: usize,
    pub rate: f64,
}

#[derive(Debug, Serialize)]
pub struct BetaDiagnostics {
    pub trace: Vec<f64>,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub acf: Vec<f64>,
    pub ess: f64,
}

#[derive(Debug, Serialize)]
pub struct KDiagnostics {
    pub trace: Vec<usize>,
    pub frequencies: BTreeMap<usize, usize>,
    pub acf: Vec<f64>,
    pub ess: f64,
}

/// Burn-in traces for convergence inspection.
///
/// `k_trace` is present only for `JointMh` — the Hybrid sampler's Gibbs step
/// has no accept/reject for k, so its burn-in trace is uninformative.
/// `#[serde(skip_serializing_if)]` omits the field entirely from the JSON for Hybrid.
#[derive(Debug, Serialize)]
pub struct BurnInDiagnostics {
    pub beta_trace: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k_trace: Option<Vec<usize>>,
}

#[derive(Debug, Serialize)]
pub struct DiagnosticsOutput {
    pub config: DiagnosticsConfig,
    pub mh_acceptance: MhAcceptance,
    pub beta: BetaDiagnostics,
    pub k: KDiagnostics,
    pub burn_in: BurnInDiagnostics,
}

// ── Assembly ──────────────────────────────────────────────────────────────────

pub fn build_diagnostics(
    result: &SamplerResult,
    sampler_config: &SamplerConfig,
    k_candidates: &[usize],
) -> DiagnosticsOutput {
    let chain = &result.chain;
    let n = chain.len();
    let max_lag = (sampler_config.n_samples / 4).min(50);

    // ── config ──
    let k_start = k_candidates.iter().copied().min().unwrap_or(0);
    let k_end = k_candidates.iter().copied().max().unwrap_or(0);
    let config = DiagnosticsConfig {
        method: format!("{:?}", sampler_config.method),
        n_samples: sampler_config.n_samples,
        burn_in: sampler_config.burn_in,
        thinning: sampler_config.thinning,
        beta_step: sampler_config.beta_step,
        beta_sigma: sampler_config.beta_sigma,
        k_candidates: KCandidateRange { start: k_start, end: k_end },
        total_iterations: result.total_iters,
    };

    // ── acceptance ──
    let n_proposed = n; // one proposal per post-burn-in iteration
    let rate = if n_proposed > 0 { result.n_accepted as f64 / n_proposed as f64 } else { 0.0 };
    let mh_acceptance = MhAcceptance { n_accepted: result.n_accepted, n_proposed, rate };

    // ── beta ──
    let beta_trace: Vec<f64> = chain.iter().map(|d| d.beta).collect();
    let beta_mean = beta_trace.iter().sum::<f64>() / n.max(1) as f64;
    let beta_var =
        beta_trace.iter().map(|&b| (b - beta_mean).powi(2)).sum::<f64>() / n.max(1) as f64;
    let beta_std = beta_var.sqrt();
    let beta_min = beta_trace.iter().cloned().fold(f64::INFINITY, f64::min);
    let beta_max = beta_trace.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let beta_acf = compute_acf(&beta_trace, max_lag);
    let beta_ess = compute_ess(&beta_acf, n);
    let beta = BetaDiagnostics {
        trace: beta_trace,
        mean: beta_mean,
        std: beta_std,
        min: if beta_min.is_finite() { beta_min } else { 0.0 },
        max: if beta_max.is_finite() { beta_max } else { 0.0 },
        acf: beta_acf,
        ess: beta_ess,
    };

    // ── k ──
    let k_trace: Vec<usize> = chain.iter().map(|d| d.k).collect();
    let mut frequencies: BTreeMap<usize, usize> = BTreeMap::new();
    for &k in &k_trace {
        *frequencies.entry(k).or_insert(0) += 1;
    }
    let k_as_float: Vec<f64> = k_trace.iter().map(|&k| k as f64).collect();
    let k_acf = compute_acf(&k_as_float, max_lag);
    let k_ess = compute_ess(&k_acf, n);
    let k = KDiagnostics { trace: k_trace, frequencies, acf: k_acf, ess: k_ess };

    // ── burn-in ──
    let burn_in_beta: Vec<f64> = result.burn_in_chain.iter().map(|d| d.beta).collect();
    let burn_in_k_opt = match sampler_config.method {
        InferenceMethod::JointMh => {
            Some(result.burn_in_chain.iter().map(|d| d.k).collect())
        }
        InferenceMethod::Hybrid => None,
    };
    let burn_in = BurnInDiagnostics { beta_trace: burn_in_beta, k_trace: burn_in_k_opt };

    DiagnosticsOutput { config, mh_acceptance, beta, k, burn_in }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // ── compute_acf ──────────────────────────────────────────────────────────

    #[test]
    fn acf_lag0_is_one() {
        let acf = compute_acf(&[1.0, 2.0, 3.0, 4.0, 5.0], 4);
        assert!((acf[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn acf_perfectly_correlated_chain_all_ones() {
        // Constant chain → we return [1.0] sentinel.
        let acf = compute_acf(&[3.0, 3.0, 3.0, 3.0], 3);
        assert_eq!(acf, vec![1.0]);
    }

    #[test]
    fn acf_iid_chain_lags_near_zero() {
        // AR(0) chain: alternating +1/-1. ACF at lag 1 should be -1.
        let v: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let acf = compute_acf(&v, 2);
        assert!(acf[1] < -0.9, "lag-1 ACF should be near -1, got {}", acf[1]);
    }

    #[test]
    fn acf_length_equals_max_lag_plus_one() {
        let acf = compute_acf(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(acf.len(), 4); // lags 0, 1, 2, 3
    }

    #[test]
    fn acf_max_lag_clamped_to_n_minus_one() {
        // Chain of length 4; asking for max_lag=10 should clamp to 3.
        let acf = compute_acf(&[1.0, 2.0, 3.0, 4.0], 10);
        assert_eq!(acf.len(), 4); // lags 0..3
    }

    #[test]
    fn acf_empty_chain_returns_sentinel() {
        assert_eq!(compute_acf(&[], 5), vec![1.0]);
    }

    #[test]
    fn acf_single_element_returns_sentinel() {
        assert_eq!(compute_acf(&[7.0], 5), vec![1.0]);
    }

    #[test]
    fn acf_values_in_range() {
        let v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let acf = compute_acf(&v, 10);
        for (lag, &r) in acf.iter().enumerate() {
            assert!(
                r >= -1.0 - TOL && r <= 1.0 + TOL,
                "lag {lag}: ACF = {r} out of [-1, 1]"
            );
        }
    }

    // ── compute_ess ──────────────────────────────────────────────────────────

    #[test]
    fn ess_iid_chain_near_n() {
        // IID chain: ACF = [1, 0, 0, 0]. Chain length n=1000 (much larger than acf.len()).
        // pair 0: 1.0 + 0.0 = 1.0 > 0, pair 1: 0.0 + 0.0 = 0.0 → stop.
        // ESS = 1000 / (2*1.0 - 1) = 1000.0
        let acf = vec![1.0, 0.0, 0.0, 0.0];
        let ess = compute_ess(&acf, 1000);
        assert!((ess - 1000.0).abs() < TOL, "expected ess=1000.0, got {ess}");
    }

    #[test]
    fn ess_scales_with_n_not_acf_len() {
        // Same ACF, two different chain lengths → ESS scales proportionally.
        let acf = vec![1.0, 0.5, 0.0, 0.0];
        // pair 0: 1.0 + 0.5 = 1.5 > 0, pair 1: 0.0 + 0.0 = 0.0 → stop.
        // denom = 2*1.5 - 1 = 2.0
        let ess_500 = compute_ess(&acf, 500);
        let ess_2000 = compute_ess(&acf, 2000);
        assert!((ess_500 - 250.0).abs() < TOL, "expected 250, got {ess_500}");
        assert!((ess_2000 - 1000.0).abs() < TOL, "expected 1000, got {ess_2000}");
    }

    #[test]
    fn ess_correlated_chain_less_than_n() {
        // High autocorrelation → ESS < n.
        let acf = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let n = 1000usize;
        let ess = compute_ess(&acf, n);
        assert!(ess < n as f64, "correlated chain ESS should be < n={n}, got {ess}");
    }

    #[test]
    fn ess_never_below_one() {
        let acf = vec![1.0]; // degenerate one-element ACF
        assert!(compute_ess(&acf, 100) >= 1.0);
    }

    #[test]
    fn ess_never_above_n() {
        let acf = vec![1.0, 0.01, 0.01, 0.01];
        let n = 500usize;
        let ess = compute_ess(&acf, n);
        assert!(ess <= n as f64 + TOL, "ESS {ess} exceeded n={n}");
    }

    #[test]
    fn ess_empty_acf_returns_one() {
        assert!((compute_ess(&[], 1000) - 1.0).abs() < TOL);
    }

    #[test]
    fn ess_zero_n_returns_one() {
        assert!((compute_ess(&[1.0, 0.5], 0) - 1.0).abs() < TOL);
    }

    // ── build_diagnostics ────────────────────────────────────────────────────

    use crate::inference::sample_posterior;
    use crate::types::{PnnModel, SamplerConfig};

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

    fn seeded_config(method: InferenceMethod) -> SamplerConfig {
        SamplerConfig {
            method,
            n_samples: 60,
            burn_in: 20,
            thinning: 1,
            beta_step: 0.3,
            beta_sigma: 5.0,
            seed: Some(7),
        }
    }

    #[test]
    fn build_diagnostics_beta_trace_length_equals_n_samples() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert_eq!(diag.beta.trace.len(), 60);
    }

    #[test]
    fn build_diagnostics_k_trace_length_equals_n_samples() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert_eq!(diag.k.trace.len(), 60);
    }

    #[test]
    fn build_diagnostics_k_frequencies_sum_to_n_samples() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        let total: usize = diag.k.frequencies.values().sum();
        assert_eq!(total, 60);
    }

    #[test]
    fn build_diagnostics_acceptance_rate_in_range() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert!(diag.mh_acceptance.rate >= 0.0 && diag.mh_acceptance.rate <= 1.0);
        assert!(diag.mh_acceptance.n_accepted <= diag.mh_acceptance.n_proposed);
    }

    #[test]
    fn build_diagnostics_burn_in_beta_trace_length_equals_burn_in() {
        for method in [InferenceMethod::Hybrid, InferenceMethod::JointMh] {
            let model = toy_model();
            let cfg = seeded_config(method);
            let result = sample_posterior(&model, &cfg);
            let diag = build_diagnostics(&result, &cfg, &model.k_values);
            assert_eq!(
                diag.burn_in.beta_trace.len(),
                20,
                "{method:?}: expected burn_in beta_trace length 20"
            );
        }
    }

    #[test]
    fn hybrid_burn_in_has_no_k_trace() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert!(diag.burn_in.k_trace.is_none(), "Hybrid burn_in.k_trace should be None");
    }

    #[test]
    fn joint_mh_burn_in_has_k_trace_of_correct_length() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::JointMh);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        let k_trace = diag.burn_in.k_trace.expect("JointMh burn_in.k_trace should be Some");
        assert_eq!(k_trace.len(), 20);
    }

    #[test]
    fn build_diagnostics_config_k_candidates_reflects_min_max() {
        let model = toy_model(); // k_values = [1, 2, 3]
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert_eq!(diag.config.k_candidates.start, 1);
        assert_eq!(diag.config.k_candidates.end, 3);
    }

    #[test]
    fn build_diagnostics_beta_acf_lag0_is_one() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert!((diag.beta.acf[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn build_diagnostics_ess_in_valid_range() {
        let model = toy_model();
        let cfg = seeded_config(InferenceMethod::Hybrid);
        let result = sample_posterior(&model, &cfg);
        let diag = build_diagnostics(&result, &cfg, &model.k_values);
        assert!(diag.beta.ess >= 1.0 && diag.beta.ess <= cfg.n_samples as f64 + TOL);
        assert!(diag.k.ess >= 1.0 && diag.k.ess <= cfg.n_samples as f64 + TOL);
    }
}
