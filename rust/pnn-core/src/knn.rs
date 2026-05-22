use crate::types::DataMatrix;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub enum KnnError {
    EmptyData,
    EmptyQuery,
    InvalidK { k: usize, n_samples: usize },
    EmptyFeatureVector { row_index: usize },
    DataDimensionMismatch {
        expected: usize,
        found: usize,
        row_index: usize,
    },
    QueryDimensionMismatch {
        expected: usize,
        found: usize,
    },
    NonFiniteQueryValue {
        feature_index: usize,
        value: f64,
    },
    NonFiniteDataValue {
        row_index: usize,
        feature_index: usize,
        value: f64,
    },
}

impl std::fmt::Display for KnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "data must not be empty"),
            Self::EmptyQuery => write!(f, "query must not be empty"),
            Self::InvalidK { k, n_samples } => {
                write!(f, "k must be in 1..={n_samples}, got {k}")
            }
            Self::EmptyFeatureVector { row_index } => {
                write!(f, "feature vector at row {row_index} must not be empty")
            }
            Self::DataDimensionMismatch {
                expected,
                found,
                row_index,
            } => write!(
                f,
                "feature vector at row {row_index} has {found} columns, expected {expected}"
            ),
            Self::QueryDimensionMismatch { expected, found } => {
                write!(f, "query has {found} columns, expected {expected}")
            }
            Self::NonFiniteQueryValue {
                feature_index,
                value,
            } => write!(
                f,
                "query feature at index {feature_index} must be finite, got {value}"
            ),
            Self::NonFiniteDataValue {
                row_index,
                feature_index,
                value,
            } => write!(
                f,
                "data feature at row {row_index}, index {feature_index} must be finite, got {value}"
            ),
        }
    }
}

impl std::error::Error for KnnError {}

pub fn k_nearest(data: &DataMatrix, query: &[f64], k: usize) -> Result<Vec<usize>, KnnError> {
    validate_inputs(data, query, k)?;

    let mut ranked: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(idx, row)| (idx, squared_euclidean_distance(row, query)))
        .collect();

    ranked.sort_by(|(idx_a, dist_a), (idx_b, dist_b)| {
        dist_a
            .partial_cmp(dist_b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| idx_a.cmp(idx_b))
    });

    Ok(ranked
        .into_iter()
        .take(k)
        .map(|(idx, _distance)| idx)
        .collect())
}

fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum()
}

fn validate_inputs(data: &DataMatrix, query: &[f64], k: usize) -> Result<(), KnnError> {
    if data.is_empty() {
        return Err(KnnError::EmptyData);
    }

    if query.is_empty() {
        return Err(KnnError::EmptyQuery);
    }

    let n_samples = data.len();
    if k == 0 || k > n_samples {
        return Err(KnnError::InvalidK { k, n_samples });
    }

    let expected = data[0].len();
    if expected == 0 {
        return Err(KnnError::EmptyFeatureVector { row_index: 0 });
    }

    if query.len() != expected {
        return Err(KnnError::QueryDimensionMismatch {
            expected,
            found: query.len(),
        });
    }

    for (feature_index, value) in query.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(KnnError::NonFiniteQueryValue {
                feature_index,
                value,
            });
        }
    }

    for (row_index, row) in data.iter().enumerate() {
        if row.is_empty() {
            return Err(KnnError::EmptyFeatureVector { row_index });
        }

        if row.len() != expected {
            return Err(KnnError::DataDimensionMismatch {
                expected,
                found: row.len(),
                row_index,
            });
        }

        for (feature_index, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(KnnError::NonFiniteDataValue {
                    row_index,
                    feature_index,
                    value,
                });
            }
        }
    }

    Ok(())
}
