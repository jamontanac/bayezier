pub type DataMatrix = Vec<Vec<f64>>;
pub type Labels = Vec<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyTrainingData,
    EmptyLabels,
    LengthMismatch { n_samples: usize, n_labels: usize },
    EmptyFeatureVector { row_index: usize },
    InconsistentFeatureDimensions {
        expected: usize,
        found: usize,
        row_index: usize,
    },
    InvalidClassCount { n_classes: usize },
    InvalidKMax { k_max: usize, n_train: usize },
    LabelOutOfRange {
        label: usize,
        n_classes: usize,
        row_index: usize,
    },
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
            Self::InconsistentFeatureDimensions {
                expected,
                found,
                row_index,
            } => write!(
                f,
                "feature vector at row {row_index} has {found} columns, expected {expected}"
            ),
            Self::InvalidClassCount { n_classes } => {
                write!(f, "n_classes must be > 0, got {n_classes}")
            }
            Self::InvalidKMax { k_max, n_train } => {
                write!(f, "k_max must be in 1..={n_train}, got {k_max}")
            }
            Self::LabelOutOfRange {
                label,
                n_classes,
                row_index,
            } => write!(
                f,
                "label value {label} at row {row_index} exceeds class range 0..{}",
                n_classes.saturating_sub(1)
            ),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelParams {
    pub k: usize,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct PnnModel {
    pub x_train: DataMatrix,
    pub y_train: Labels,
    pub n_classes: usize,
    pub k_max: usize,
}

impl PnnModel {
    pub fn new(
        x_train: DataMatrix,
        y_train: Labels,
        n_classes: usize,
        k_max: usize,
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

        let n_train = x_train.len();
        if k_max == 0 || k_max > n_train {
            return Err(ModelError::InvalidKMax { k_max, n_train });
        }

        for (row_index, &label) in y_train.iter().enumerate() {
            if label >= n_classes {
                return Err(ModelError::LabelOutOfRange {
                    label,
                    n_classes,
                    row_index,
                });
            }
        }

        Ok(Self {
            x_train,
            y_train,
            n_classes,
            k_max,
        })
    }

    pub fn n_train(&self) -> usize {
        self.x_train.len()
    }

    pub fn n_features(&self) -> usize {
        self.x_train.first().map_or(0, Vec::len)
    }
}
