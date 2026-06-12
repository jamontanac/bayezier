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
