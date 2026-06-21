pub type DataMatrix = Vec<Vec<f64>>;
pub type Labels = Vec<usize>;
/// `CountTensor[i][ki][c]` — class-c neighbor count for training point `i` at the `ki`-th
/// candidate k value. Built once before MCMC; never mutated during sampling.
pub type CountTensor = Vec<Vec<Vec<usize>>>;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyTrainingData,
    EmptyLabels,
    LengthMismatch { n_samples: usize, n_labels: usize },
    EmptyFeatureVector { row_index: usize },
    InconsistentFeatureDimensions { expected: usize, found: usize, row_index: usize },
    InvalidClassCount { n_classes: usize },
    EmptyKValues,
    InvalidKValue { k: usize, n_train: usize },
    LabelOutOfRange { label: usize, n_classes: usize, row_index: usize },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTrainingData => write!(f, "training data must not be empty"),
            Self::EmptyLabels => write!(f, "labels must not be empty"),
            Self::LengthMismatch { n_samples, n_labels } => write!(
                f,
                "training sample count ({n_samples}) does not match label count ({n_labels})"
            ),
            Self::EmptyFeatureVector { row_index } => {
                write!(f, "feature vector at row {row_index} must not be empty")
            }
            Self::InconsistentFeatureDimensions { expected, found, row_index } => write!(
                f,
                "feature vector at row {row_index} has {found} columns, expected {expected}"
            ),
            Self::InvalidClassCount { n_classes } => {
                write!(f, "n_classes must be > 0, got {n_classes}")
            }
            Self::EmptyKValues => write!(f, "k_values must not be empty"),
            Self::InvalidKValue { k, n_train } => {
                write!(f, "k must be in 1..={n_train}, got {k}")
            }
            Self::LabelOutOfRange { label, n_classes, row_index } => write!(
                f,
                "label value {label} at row {row_index} exceeds class range 0..{}",
                n_classes.saturating_sub(1)
            ),
        }
    }
}

impl std::error::Error for ModelError {}

// ── Sampler configuration ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMethod {
    /// Gibbs step for k + log-scale MH for beta. Default; mixes well in practice.
    Hybrid,
    /// Joint MH proposal over (k, beta). Paper-style; useful for reference runs.
    JointMh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    pub method: InferenceMethod,
    pub n_samples: usize,
    pub burn_in: usize,
    /// Keep one draw every `thinning` iterations after burn-in.
    pub thinning: usize,
    /// Proposal standard deviation for the MH step on beta.
    pub beta_step: f64,
    /// Scale parameter of the half-normal prior on beta.
    pub beta_sigma: f64,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            method: InferenceMethod::Hybrid,
            n_samples: 1000,
            burn_in: 500,
            thinning: 1,
            beta_step: 0.3,
            beta_sigma: 5.0,
            seed: None,
        }
    }
}

// ── Posterior draw ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosteriorDraw {
    /// Position in `k_values` that was sampled.
    pub k_index: usize,
    /// Actual k value (`model.k_values[k_index]`).
    pub k: usize,
    pub beta: f64,
}

// ── Sampler result ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplerResult {
    /// Post-burn-in, post-thinning draws — the posterior chain.
    pub chain: Vec<PosteriorDraw>,
    /// Every burn-in draw at full resolution (no thinning), in iteration order.
    pub burn_in_chain: Vec<PosteriorDraw>,
    /// Proposals accepted during the post-burn-in phase.
    /// Hybrid: β MH accepts. JointMh: joint (k*, β*) accepts.
    pub n_accepted: usize,
    /// Total Gibbs+MH cycles run: burn_in + n_samples * thinning.
    pub total_iters: usize,
}

// ── Model ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PnnModel {
    pub x_train: DataMatrix,
    pub y_train: Labels,
    pub n_classes: usize,
    /// Sorted, deduplicated candidate k values (e.g. `[1, 3, 5, 7]`).
    pub k_values: Vec<usize>,
}

impl PnnModel {
    pub fn new(
        x_train: DataMatrix,
        y_train: Labels,
        n_classes: usize,
        mut k_values: Vec<usize>,
    ) -> Result<Self, ModelError> {
        if x_train.is_empty() {
            return Err(ModelError::EmptyTrainingData);
        }
        if y_train.is_empty() {
            return Err(ModelError::EmptyLabels);
        }
        if x_train.len() != y_train.len() {
            return Err(ModelError::LengthMismatch {
                n_samples: x_train.len(),
                n_labels: y_train.len(),
            });
        }

        let expected_features = x_train[0].len();
        if expected_features == 0 {
            return Err(ModelError::EmptyFeatureVector { row_index: 0 });
        }
        for (row_index, row) in x_train.iter().enumerate() {
            if row.is_empty() {
                return Err(ModelError::EmptyFeatureVector { row_index });
            }
            if row.len() != expected_features {
                return Err(ModelError::InconsistentFeatureDimensions {
                    expected: expected_features,
                    found: row.len(),
                    row_index,
                });
            }
        }

        if n_classes == 0 {
            return Err(ModelError::InvalidClassCount { n_classes });
        }

        if k_values.is_empty() {
            return Err(ModelError::EmptyKValues);
        }
        k_values.sort_unstable();
        k_values.dedup();

        let n_train = x_train.len();
        for &k in &k_values {
            // After self-exclusion there are only n_train-1 neighbors, so k must be < n_train.
            if k == 0 || k >= n_train {
                return Err(ModelError::InvalidKValue { k, n_train });
            }
        }

        for (row_index, &label) in y_train.iter().enumerate() {
            if label >= n_classes {
                return Err(ModelError::LabelOutOfRange { label, n_classes, row_index });
            }
        }

        Ok(Self { x_train, y_train, n_classes, k_values })
    }

    /// Largest candidate k. Safe because `k_values` is validated non-empty.
    pub fn k_max(&self) -> usize {
        *self.k_values.last().expect("k_values non-empty by construction")
    }

    pub fn n_train(&self) -> usize {
        self.x_train.len()
    }

    pub fn n_features(&self) -> usize {
        self.x_train.first().map_or(0, Vec::len)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn three_point_model() -> PnnModel {
        PnnModel::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
            vec![1, 2],
        )
        .unwrap()
    }

    // ── Happy paths ───────────────────────────────────────────────────────────

    #[test]
    fn new_stores_all_fields_correctly() {
        let m = three_point_model();
        assert_eq!(m.n_train(), 3);
        assert_eq!(m.n_features(), 2);
        assert_eq!(m.n_classes, 2);
        assert_eq!(m.k_values, vec![1, 2]);
        assert_eq!(m.k_max(), 2);
    }

    #[test]
    fn new_sorts_k_values_ascending() {
        let m = PnnModel::new(
            vec![vec![0.0]; 5],
            vec![0, 1, 0, 1, 0],
            2,
            vec![3, 1, 2],
        )
        .unwrap();
        assert_eq!(m.k_values, vec![1, 2, 3]);
    }

    #[test]
    fn new_deduplicates_k_values() {
        let m = PnnModel::new(
            vec![vec![0.0]; 5],
            vec![0, 1, 0, 1, 0],
            2,
            vec![1, 1, 2, 2, 3],
        )
        .unwrap();
        assert_eq!(m.k_values, vec![1, 2, 3]);
    }

    #[test]
    fn k_max_equals_largest_candidate() {
        let m = PnnModel::new(
            vec![vec![0.0]; 5],
            vec![0, 0, 1, 1, 0],
            2,
            vec![1, 4], // k=4 < n_train=5: valid
        )
        .unwrap();
        assert_eq!(m.k_max(), 4);
    }

    #[test]
    fn k_equals_n_train_minus_one_is_valid() {
        // Maximum valid k after self-exclusion is n_train - 1.
        let n_train = 4;
        let result = PnnModel::new(
            vec![vec![0.0]; n_train],
            vec![0, 1, 0, 1],
            2,
            vec![n_train - 1],
        );
        assert!(result.is_ok(), "k = n_train-1 should be valid");
    }

    // ── ModelError variants ───────────────────────────────────────────────────

    #[test]
    fn error_empty_training_data() {
        let err = PnnModel::new(vec![], vec![], 2, vec![1]).unwrap_err();
        assert_eq!(err, ModelError::EmptyTrainingData);
    }

    #[test]
    fn error_empty_labels() {
        // y_train empty is checked before length-mismatch.
        let err = PnnModel::new(vec![vec![1.0], vec![2.0]], vec![], 2, vec![1]).unwrap_err();
        assert_eq!(err, ModelError::EmptyLabels);
    }

    #[test]
    fn error_length_mismatch() {
        let err = PnnModel::new(
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 1],
            2,
            vec![1],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::LengthMismatch { n_samples: 3, n_labels: 2 });
    }

    #[test]
    fn error_empty_feature_vector_first_row() {
        let err = PnnModel::new(
            vec![vec![], vec![1.0]],
            vec![0, 1],
            2,
            vec![1],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::EmptyFeatureVector { row_index: 0 });
    }

    #[test]
    fn error_empty_feature_vector_later_row() {
        let err = PnnModel::new(
            vec![vec![1.0], vec![]],
            vec![0, 1],
            2,
            vec![1],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::EmptyFeatureVector { row_index: 1 });
    }

    #[test]
    fn error_inconsistent_feature_dimensions() {
        let err = PnnModel::new(
            vec![vec![1.0, 2.0], vec![3.0]], // row 1 has 1 feature, expected 2
            vec![0, 1],
            2,
            vec![1],
        )
        .unwrap_err();
        assert_eq!(
            err,
            ModelError::InconsistentFeatureDimensions { expected: 2, found: 1, row_index: 1 }
        );
    }

    #[test]
    fn error_invalid_class_count_zero() {
        let err = PnnModel::new(
            vec![vec![1.0], vec![2.0]],
            vec![0, 0],
            0, // n_classes = 0
            vec![1],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::InvalidClassCount { n_classes: 0 });
    }

    #[test]
    fn error_empty_k_values() {
        let err = PnnModel::new(
            vec![vec![1.0], vec![2.0]],
            vec![0, 1],
            2,
            vec![],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::EmptyKValues);
    }

    #[test]
    fn error_k_value_zero() {
        let err = PnnModel::new(
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 1, 0],
            2,
            vec![0],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::InvalidKValue { k: 0, n_train: 3 });
    }

    #[test]
    fn error_k_equals_n_train() {
        // After self-exclusion only n_train-1 neighbors exist; k = n_train is one too many.
        let err = PnnModel::new(
            vec![vec![0.0]; 3],
            vec![0, 1, 0],
            2,
            vec![3], // k=3 == n_train=3
        )
        .unwrap_err();
        assert_eq!(err, ModelError::InvalidKValue { k: 3, n_train: 3 });
    }

    #[test]
    fn error_k_exceeds_n_train() {
        let err = PnnModel::new(
            vec![vec![0.0]; 3],
            vec![0, 1, 0],
            2,
            vec![5], // k=5 > n_train=3
        )
        .unwrap_err();
        assert_eq!(err, ModelError::InvalidKValue { k: 5, n_train: 3 });
    }

    #[test]
    fn error_label_out_of_range() {
        let err = PnnModel::new(
            vec![vec![0.0]; 3],
            vec![0, 1, 2], // label 2 >= n_classes=2
            2,
            vec![1, 2],
        )
        .unwrap_err();
        assert_eq!(err, ModelError::LabelOutOfRange { label: 2, n_classes: 2, row_index: 2 });
    }

    // ── ModelError::Display ───────────────────────────────────────────────────

    #[test]
    fn display_empty_training_data() {
        assert!(ModelError::EmptyTrainingData.to_string().contains("empty"));
    }

    #[test]
    fn display_length_mismatch_shows_both_counts() {
        let msg = ModelError::LengthMismatch { n_samples: 5, n_labels: 3 }.to_string();
        assert!(msg.contains('5') && msg.contains('3'));
    }

    #[test]
    fn display_invalid_k_value_shows_k_and_n_train() {
        let msg = ModelError::InvalidKValue { k: 7, n_train: 4 }.to_string();
        assert!(msg.contains('7') && msg.contains('4'));
    }

    #[test]
    fn display_label_out_of_range_shows_label() {
        let msg = ModelError::LabelOutOfRange { label: 9, n_classes: 3, row_index: 0 }.to_string();
        assert!(msg.contains('9'));
    }

    // ── SamplerConfig::default ────────────────────────────────────────────────

    #[test]
    fn sampler_config_default_uses_hybrid_method() {
        assert_eq!(SamplerConfig::default().method, InferenceMethod::Hybrid);
    }

    #[test]
    fn sampler_config_default_has_positive_parameters() {
        let cfg = SamplerConfig::default();
        assert!(cfg.n_samples > 0);
        assert!(cfg.thinning >= 1);
        assert!(cfg.beta_step > 0.0);
        assert!(cfg.beta_sigma > 0.0);
    }

    #[test]
    fn sampler_config_default_seed_is_none() {
        assert_eq!(SamplerConfig::default().seed, None);
    }
}
