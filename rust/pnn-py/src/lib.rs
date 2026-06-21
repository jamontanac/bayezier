#![allow(unsafe_op_in_unsafe_fn)]

use csv::StringRecord;
use pnn_core::{
    InferenceMethod, ModelError, PnnModel, SamplerConfig, SamplerResult, argmax, build_diagnostics,
    predict_proba, sample_posterior,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Serialize;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
enum BindingError {
    Message(String),
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
    Model(ModelError),
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(msg) => write!(f, "{msg}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::Csv(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
            Self::Model(err) => write!(f, "{err}"),
        }
    }
}

impl Error for BindingError {}

impl From<std::io::Error> for BindingError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for BindingError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl From<serde_json::Error> for BindingError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<ModelError> for BindingError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

#[derive(Debug)]
struct RunConfig {
    train: PathBuf,
    test: PathBuf,
    output: Option<PathBuf>,
    dataset: String,
    implementation: String,
    k_values: Vec<usize>,
    method: InferenceMethod,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    beta_step: f64,
    beta_sigma: f64,
    seed: Option<u64>,
    diagnose: Option<PathBuf>,
}

#[derive(Debug)]
struct Dataset {
    features: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct Prediction {
    index: usize,
    probabilities: Vec<f64>,
    predicted_class: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    implementation: String,
    dataset: String,
    predictions: Vec<Prediction>,
    k_posterior: Vec<usize>,
    beta_posterior: Vec<f64>,
    misclassification_cost: f64,
    runtime_ms: f64,
}

fn to_py_err(err: BindingError) -> PyErr {
    match err {
        BindingError::Message(msg) => PyValueError::new_err(msg),
        BindingError::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(io_err.to_string())
        }
        BindingError::Io(io_err) => PyRuntimeError::new_err(io_err.to_string()),
        BindingError::Csv(csv_err) => PyValueError::new_err(csv_err.to_string()),
        BindingError::Json(json_err) => PyRuntimeError::new_err(json_err.to_string()),
        BindingError::Model(model_err) => PyValueError::new_err(model_err.to_string()),
    }
}

fn parse_method(raw: &str) -> Result<InferenceMethod, BindingError> {
    match raw {
        "hybrid" => Ok(InferenceMethod::Hybrid),
        "joint-mh" => Ok(InferenceMethod::JointMh),
        other => Err(BindingError::Message(format!(
            "invalid value for method: '{other}' (expected 'hybrid' or 'joint-mh')"
        ))),
    }
}

fn optional_path_from_string(
    value: Option<String>,
    field_name: &str,
) -> Result<Option<PathBuf>, BindingError> {
    match value {
        Some(raw) => {
            if raw.trim().is_empty() {
                return Err(BindingError::Message(format!(
                    "{field_name} must not be empty when provided"
                )));
            }
            Ok(Some(PathBuf::from(raw)))
        }
        None => Ok(None),
    }
}

fn resolve_k_values(
    k: Option<usize>,
    k_values_explicit: Option<Vec<usize>>,
    k_range_explicit: Option<(usize, usize)>,
) -> Result<Vec<usize>, BindingError> {
    if let Some(values) = k_values_explicit {
        if values.is_empty() {
            return Err(BindingError::Message(
                "invalid value for k_values: must supply at least one value".to_string(),
            ));
        }
        if values.iter().any(|&v| v == 0) {
            return Err(BindingError::Message(
                "invalid value in k_values: all values must be >= 1".to_string(),
            ));
        }
        return Ok(values);
    }

    if let Some((start, end)) = k_range_explicit {
        if start == 0 {
            return Err(BindingError::Message(
                "invalid value for k_range: start must be >= 1".to_string(),
            ));
        }
        if end < start {
            return Err(BindingError::Message(format!(
                "invalid value for k_range: end ({end}) must be >= start ({start})"
            )));
        }
        return Ok((start..=end).collect());
    }

    let k_single = k.unwrap_or(3);
    if k_single == 0 {
        return Err(BindingError::Message(
            "invalid value for k: must be >= 1".to_string(),
        ));
    }
    Ok(vec![k_single])
}

#[allow(clippy::too_many_arguments)]
fn build_runtime_config(
    dataset: String,
    implementation: String,
    k: Option<usize>,
    k_values: Option<Vec<usize>>,
    k_range: Option<(usize, usize)>,
    method: String,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    beta_step: f64,
    beta_sigma: f64,
    seed: Option<u64>,
    out_path: Option<String>,
    diagnose_path: Option<String>,
) -> Result<RunConfig, BindingError> {
    if n_samples == 0 {
        return Err(BindingError::Message(
            "invalid value for n_samples: must be >= 1".to_string(),
        ));
    }
    if thinning == 0 {
        return Err(BindingError::Message(
            "invalid value for thinning: must be >= 1".to_string(),
        ));
    }
    if beta_step <= 0.0 {
        return Err(BindingError::Message(
            "invalid value for beta_step: must be > 0".to_string(),
        ));
    }
    if beta_sigma <= 0.0 {
        return Err(BindingError::Message(
            "invalid value for beta_sigma: must be > 0".to_string(),
        ));
    }

    let k_values = resolve_k_values(k, k_values, k_range)?;
    let method = parse_method(&method)?;
    let output = optional_path_from_string(out_path, "out_path")?;
    let diagnose = optional_path_from_string(diagnose_path, "diagnose_path")?;

    Ok(RunConfig {
        train: PathBuf::new(),
        test: PathBuf::new(),
        output,
        dataset,
        implementation,
        k_values,
        method,
        n_samples,
        burn_in,
        thinning,
        beta_step,
        beta_sigma,
        seed,
        diagnose,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    train_path: String,
    test_path: String,
    dataset: String,
    implementation: String,
    k: Option<usize>,
    k_values: Option<Vec<usize>>,
    k_range: Option<(usize, usize)>,
    method: String,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    beta_step: f64,
    beta_sigma: f64,
    seed: Option<u64>,
    out_path: Option<String>,
    diagnose_path: Option<String>,
) -> Result<RunConfig, BindingError> {
    if train_path.trim().is_empty() {
        return Err(BindingError::Message(
            "train_path must not be empty".to_string(),
        ));
    }
    if test_path.trim().is_empty() {
        return Err(BindingError::Message(
            "test_path must not be empty".to_string(),
        ));
    }

    let mut config = build_runtime_config(
        dataset,
        implementation,
        k,
        k_values,
        k_range,
        method,
        n_samples,
        burn_in,
        thinning,
        beta_step,
        beta_sigma,
        seed,
        out_path,
        diagnose_path,
    )?;
    config.train = PathBuf::from(train_path);
    config.test = PathBuf::from(test_path);
    Ok(config)
}

fn payload_to_py(py: Python<'_>, payload: &BenchmarkOutput) -> PyResult<Py<PyAny>> {
    let payload_json =
        serde_json::to_string(payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let json_module = PyModule::import_bound(py, "json")?;
    let parsed = json_module.call_method1("loads", (payload_json,))?;
    Ok(parsed.unbind())
}

fn extract_matrix_from_py_any(
    value: &Bound<'_, PyAny>,
    field_name: &str,
) -> Result<Vec<Vec<f64>>, BindingError> {
    if let Ok(matrix) = value.extract::<Vec<Vec<f64>>>() {
        return Ok(matrix);
    }

    if let Ok(as_list) = value.call_method0("tolist") {
        if let Ok(matrix) = as_list.extract::<Vec<Vec<f64>>>() {
            return Ok(matrix);
        }
    }

    Err(BindingError::Message(format!(
        "{field_name} must be a 2D sequence of numeric values"
    )))
}

fn extract_labels_from_py_any(
    value: &Bound<'_, PyAny>,
    field_name: &str,
) -> Result<Vec<usize>, BindingError> {
    if let Ok(labels) = value.extract::<Vec<usize>>() {
        return Ok(labels);
    }

    if let Ok(as_list) = value.call_method0("tolist") {
        if let Ok(labels) = as_list.extract::<Vec<usize>>() {
            return Ok(labels);
        }
    }

    Err(BindingError::Message(format!(
        "{field_name} must be a 1D sequence of non-negative integer class labels"
    )))
}

fn run_model(
    config: &RunConfig,
    train_features: Vec<Vec<f64>>,
    train_labels: Vec<usize>,
    test_features: Vec<Vec<f64>>,
    test_labels: Option<Vec<usize>>,
) -> Result<BenchmarkOutput, BindingError> {
    let started_at = Instant::now();

    if let Some(ref labels) = test_labels {
        if labels.len() != test_features.len() {
            return Err(BindingError::Message(format!(
                "y_test length ({}) must match x_test row count ({})",
                labels.len(),
                test_features.len()
            )));
        }
    }

    let n_classes = train_labels
        .iter()
        .copied()
        .max()
        .map_or(1usize, |label| label + 1);
    let n_train = train_features.len();
    let k_max_valid = n_train.saturating_sub(1).max(1);
    let k_values: Vec<usize> = config
        .k_values
        .iter()
        .map(|&candidate| candidate.clamp(1, k_max_valid))
        .collect();

    let model = PnnModel::new(train_features, train_labels, n_classes, k_values)?;

    let sampler_config = SamplerConfig {
        method: config.method,
        n_samples: config.n_samples,
        burn_in: config.burn_in,
        thinning: config.thinning,
        beta_step: config.beta_step,
        beta_sigma: config.beta_sigma,
        seed: config.seed,
    };

    let result: SamplerResult = sample_posterior(&model, &sampler_config);
    let chain = &result.chain;

    let proba = predict_proba(
        &test_features,
        &model.x_train,
        &model.y_train,
        chain,
        n_classes,
    );

    let predictions: Vec<Prediction> = proba
        .iter()
        .enumerate()
        .map(|(index, row)| Prediction {
            index,
            probabilities: row.clone(),
            predicted_class: argmax(row),
        })
        .collect();

    let k_posterior: Vec<usize> = chain.iter().map(|draw| draw.k).collect();
    let beta_posterior: Vec<f64> = chain.iter().map(|draw| draw.beta).collect();

    let misclassification_cost = test_labels.as_ref().map_or(0.0, |labels| {
        compute_misclassification_cost(&predictions, labels)
    });
    let runtime_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let payload = BenchmarkOutput {
        implementation: config.implementation.clone(),
        dataset: config.dataset.clone(),
        predictions,
        k_posterior,
        beta_posterior,
        misclassification_cost,
        runtime_ms,
    };

    if let Some(ref out_path) = config.output {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, serde_json::to_string_pretty(&payload)?)?;
    }

    if let Some(ref diag_path) = config.diagnose {
        let diag = build_diagnostics(&result, &sampler_config, &model.k_values);
        if let Some(parent) = diag_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(diag_path, serde_json::to_string_pretty(&diag)?)?;
    }

    Ok(payload)
}

fn run_pipeline(config: &RunConfig) -> Result<BenchmarkOutput, BindingError> {
    let train = read_labeled_csv(&config.train)?;
    let test = read_labeled_csv(&config.test)?;

    if train.features.is_empty() {
        return Err(BindingError::Message(
            "training CSV has no rows".to_string(),
        ));
    }

    run_model(
        config,
        train.features,
        train.labels,
        test.features,
        Some(test.labels),
    )
}

/// Runs the Bayesian k-NN sampler directly from CSV data files.
///
/// Parameters:
///   train_path (str): Path to the training CSV file.
///   test_path (str): Path to the test CSV file.
///   dataset (str, optional): Metadata label for the dataset name. Default: "unknown".
///   implementation (str, optional): Metadata label for the implementation. Default: "rust".
///   k (int, optional): Single k candidate value. Default: None.
///   k_values (list of int, optional): Explicit list of positive k candidates. Default: None.
///   k_range (tuple of (int, int), optional): Inclusive range of k candidates. Default: None.
///   method (str, optional): Sampler method ('hybrid' or 'joint-mh'). Default: "hybrid".
///   n_samples (int, optional): Number of post-burn-in samples to collect. Default: 1000.
///   burn_in (int, optional): Number of burn-in iterations to discard. Default: 500.
///   thinning (int, optional): Keep one draw every `thinning` iterations. Default: 1.
///   beta_step (float, optional): Proposal step size for β. Default: 0.3.
///   beta_sigma (float, optional): Prior standard deviation for β. Default: 5.0.
///   seed (int, optional): Random seed for reproducibility. Default: None.
///   out_path (str, optional): Filepath to write the benchmark JSON output. Default: None.
///   diagnose_path (str, optional): Filepath to write the MCMC diagnostics JSON. Default: None.
///
/// Returns:
///   dict: Benchmark payload dictionary matching the output schema.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (
    train_path,
    test_path,
    dataset = "unknown".to_string(),
    implementation = "rust".to_string(),
    k = None,
    k_values = None,
    k_range = None,
    method = "hybrid".to_string(),
    n_samples = 1000,
    burn_in = 500,
    thinning = 1,
    beta_step = 0.3,
    beta_sigma = 5.0,
    seed = None,
    out_path = None,
    diagnose_path = None,
))]
fn run_from_csv(
    py: Python<'_>,
    train_path: String,
    test_path: String,
    dataset: String,
    implementation: String,
    k: Option<usize>,
    k_values: Option<Vec<usize>>,
    k_range: Option<(usize, usize)>,
    method: String,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    beta_step: f64,
    beta_sigma: f64,
    seed: Option<u64>,
    out_path: Option<String>,
    diagnose_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    let config = build_config(
        train_path,
        test_path,
        dataset,
        implementation,
        k,
        k_values,
        k_range,
        method,
        n_samples,
        burn_in,
        thinning,
        beta_step,
        beta_sigma,
        seed,
        out_path,
        diagnose_path,
    )
    .map_err(to_py_err)?;

    let payload = py.allow_threads(|| run_pipeline(&config)).map_err(to_py_err)?;
    payload_to_py(py, &payload)
}

/// Runs the Bayesian k-NN sampler directly on in-memory sequences or NumPy arrays.
///
/// Parameters:
///   x_train (list/array): 2D training features.
///   y_train (list/array): 1D training class labels (non-negative integers).
///   x_test (list/array): 2D test features.
///   y_test (list/array, optional): 1D test class labels. Default: None.
///   dataset (str, optional): Metadata label for the dataset name. Default: "unknown".
///   implementation (str, optional): Metadata label for the implementation. Default: "rust".
///   k (int, optional): Single k candidate value. Default: None.
///   k_values (list of int, optional): Explicit list of positive k candidates. Default: None.
///   k_range (tuple of (int, int), optional): Inclusive range of k candidates. Default: None.
///   method (str, optional): Sampler method ('hybrid' or 'joint-mh'). Default: "hybrid".
///   n_samples (int, optional): Number of post-burn-in samples to collect. Default: 1000.
///   burn_in (int, optional): Number of burn-in iterations to discard. Default: 500.
///   thinning (int, optional): Keep one draw every `thinning` iterations. Default: 1.
///   beta_step (float, optional): Proposal step size for β. Default: 0.3.
///   beta_sigma (float, optional): Prior standard deviation for β. Default: 5.0.
///   seed (int, optional): Random seed for reproducibility. Default: None.
///   out_path (str, optional): Filepath to write the benchmark JSON output. Default: None.
///   diagnose_path (str, optional): Filepath to write the MCMC diagnostics JSON. Default: None.
///
/// Returns:
///   dict: Benchmark payload dictionary matching the output schema.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (
    x_train,
    y_train,
    x_test,
    y_test = None,
    dataset = "unknown".to_string(),
    implementation = "rust".to_string(),
    k = None,
    k_values = None,
    k_range = None,
    method = "hybrid".to_string(),
    n_samples = 1000,
    burn_in = 500,
    thinning = 1,
    beta_step = 0.3,
    beta_sigma = 5.0,
    seed = None,
    out_path = None,
    diagnose_path = None,
))]
fn run_from_arrays(
    py: Python<'_>,
    x_train: &Bound<'_, PyAny>,
    y_train: &Bound<'_, PyAny>,
    x_test: &Bound<'_, PyAny>,
    y_test: Option<&Bound<'_, PyAny>>,
    dataset: String,
    implementation: String,
    k: Option<usize>,
    k_values: Option<Vec<usize>>,
    k_range: Option<(usize, usize)>,
    method: String,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    beta_step: f64,
    beta_sigma: f64,
    seed: Option<u64>,
    out_path: Option<String>,
    diagnose_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    let config = build_runtime_config(
        dataset,
        implementation,
        k,
        k_values,
        k_range,
        method,
        n_samples,
        burn_in,
        thinning,
        beta_step,
        beta_sigma,
        seed,
        out_path,
        diagnose_path,
    )
    .map_err(to_py_err)?;

    let train_features = extract_matrix_from_py_any(x_train, "x_train").map_err(to_py_err)?;
    let train_labels = extract_labels_from_py_any(y_train, "y_train").map_err(to_py_err)?;
    let test_features = extract_matrix_from_py_any(x_test, "x_test").map_err(to_py_err)?;
    let test_labels = y_test
        .map(|labels| extract_labels_from_py_any(labels, "y_test"))
        .transpose()
        .map_err(to_py_err)?;

    let payload = py
        .allow_threads(|| {
            run_model(
                &config,
                train_features,
                train_labels,
                test_features,
                test_labels,
            )
        })
        .map_err(to_py_err)?;

    payload_to_py(py, &payload)
}

#[pymodule]
fn pnn_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_from_csv, m)?)?;
    m.add_function(wrap_pyfunction!(run_from_arrays, m)?)?;
    Ok(())
}

fn read_labeled_csv(path: &Path) -> Result<Dataset, BindingError> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    validate_headers(path, &headers)?;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (row_index, row) in reader.records().enumerate() {
        let row = row?;
        let (sample_features, label) = parse_row(path, &row, row_index + 1)?;
        features.push(sample_features);
        labels.push(label);
    }

    Ok(Dataset { features, labels })
}

fn validate_headers(path: &Path, headers: &StringRecord) -> Result<(), BindingError> {
    if headers.len() < 2 {
        return Err(BindingError::Message(format!(
            "{}: expected at least one feature column plus `label`",
            path.display()
        )));
    }

    match headers.get(headers.len() - 1) {
        Some("label") => Ok(()),
        _ => Err(BindingError::Message(format!(
            "{}: last column must be named `label`",
            path.display()
        ))),
    }
}

fn parse_row(
    path: &Path,
    row: &StringRecord,
    row_number: usize,
) -> Result<(Vec<f64>, usize), BindingError> {
    if row.len() < 2 {
        return Err(BindingError::Message(format!(
            "{} row {}: expected at least one feature and one label",
            path.display(),
            row_number
        )));
    }

    let label_col = row.len() - 1;
    let mut features = Vec::with_capacity(label_col);
    for col_idx in 0..label_col {
        let value = row
            .get(col_idx)
            .ok_or_else(|| {
                BindingError::Message(format!(
                    "{} row {}: missing value at column {}",
                    path.display(),
                    row_number,
                    col_idx
                ))
            })?
            .parse::<f64>()
            .map_err(|_| {
                BindingError::Message(format!(
                    "{} row {}: non-numeric feature at column {}",
                    path.display(),
                    row_number,
                    col_idx
                ))
            })?;
        features.push(value);
    }

    let label = row
        .get(label_col)
        .ok_or_else(|| {
            BindingError::Message(format!(
                "{} row {}: missing label value",
                path.display(),
                row_number
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            BindingError::Message(format!(
                "{} row {}: label must be a non-negative integer",
                path.display(),
                row_number
            ))
        })?;

    Ok((features, label))
}

fn compute_misclassification_cost(predictions: &[Prediction], labels: &[usize]) -> f64 {
    if predictions.is_empty() || labels.is_empty() || predictions.len() != labels.len() {
        return 0.0;
    }

    let mismatches = predictions
        .iter()
        .zip(labels.iter().copied())
        .filter(|(pred, expected)| pred.predicted_class != *expected)
        .count();

    mismatches as f64 / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::{
        build_config, build_runtime_config, parse_method, resolve_k_values, run_model, run_pipeline,
    };
    use serde_json::Value;
    use std::fs;
    use tempfile::tempdir;

    fn toy_csvs(temp: &tempfile::TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
        let train = temp.path().join("train.csv");
        let test = temp.path().join("test.csv");
        fs::write(&train, "x1,x2,label\n0.0,0.0,0\n1.0,0.0,1\n0.0,1.0,1\n").expect("write train");
        fs::write(&test, "x1,x2,label\n0.1,0.1,0\n0.9,0.1,1\n").expect("write test");
        (train, test)
    }

    #[test]
    fn resolve_k_values_obeys_precedence() {
        let from_values = resolve_k_values(Some(9), Some(vec![2, 4]), Some((3, 6))).unwrap();
        assert_eq!(from_values, vec![2, 4]);

        let from_range = resolve_k_values(Some(9), None, Some((3, 6))).unwrap();
        assert_eq!(from_range, vec![3, 4, 5, 6]);

        let from_single = resolve_k_values(Some(9), None, None).unwrap();
        assert_eq!(from_single, vec![9]);

        let from_default = resolve_k_values(None, None, None).unwrap();
        assert_eq!(from_default, vec![3]);
    }

    #[test]
    fn parse_method_rejects_invalid_value() {
        let err = parse_method("bad").expect_err("invalid method should fail");
        assert!(err.to_string().contains("invalid value for method"));
    }

    #[test]
    fn build_config_rejects_invalid_sampler_values() {
        let err = build_config(
            "train.csv".to_string(),
            "test.csv".to_string(),
            "toy".to_string(),
            "rust".to_string(),
            Some(1),
            None,
            None,
            "hybrid".to_string(),
            0,
            10,
            1,
            0.3,
            5.0,
            None,
            None,
            None,
        )
        .expect_err("n_samples=0 should fail");
        assert!(err.to_string().contains("n_samples"));
    }

    #[test]
    fn run_pipeline_writes_schema_valid_json() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out").join("rust_py.json");

        let cfg = build_config(
            train.display().to_string(),
            test.display().to_string(),
            "toy".to_string(),
            "rust".to_string(),
            Some(1),
            None,
            None,
            "hybrid".to_string(),
            50,
            10,
            1,
            0.3,
            5.0,
            Some(1),
            Some(out.display().to_string()),
            None,
        )
        .expect("build config");

        let payload = run_pipeline(&cfg).expect("run pipeline");
        assert_eq!(payload.predictions.len(), 2);
        assert_eq!(payload.k_posterior.len(), 50);
        assert_eq!(payload.beta_posterior.len(), 50);

        let raw = fs::read_to_string(out).expect("read output");
        let json: Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(json["implementation"], "rust");
        assert_eq!(json["dataset"], "toy");
        assert!(json["predictions"].is_array());
        assert!(json["misclassification_cost"].is_number());
        assert!(json["runtime_ms"].is_number());
    }

    #[test]
    fn run_model_from_arrays_uses_y_test_when_provided() {
        let cfg = build_runtime_config(
            "toy".to_string(),
            "rust".to_string(),
            Some(1),
            None,
            None,
            "hybrid".to_string(),
            50,
            10,
            1,
            0.3,
            5.0,
            Some(7),
            None,
            None,
        )
        .expect("build runtime config");

        let x_train = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let y_train = vec![0usize, 1, 1];
        let x_test = vec![vec![0.1, 0.1], vec![0.9, 0.1]];
        let y_test = vec![0usize, 1];

        let payload = run_model(&cfg, x_train, y_train, x_test, Some(y_test)).expect("run model");
        assert_eq!(payload.predictions.len(), 2);
        assert_eq!(payload.k_posterior.len(), 50);
        assert!(payload.misclassification_cost >= 0.0 && payload.misclassification_cost <= 1.0);
    }

    #[test]
    fn run_model_from_arrays_without_y_test_has_zero_cost() {
        let cfg = build_runtime_config(
            "toy".to_string(),
            "rust".to_string(),
            Some(1),
            None,
            None,
            "hybrid".to_string(),
            20,
            5,
            1,
            0.3,
            5.0,
            Some(11),
            None,
            None,
        )
        .expect("build runtime config");

        let x_train = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let y_train = vec![0usize, 1, 1];
        let x_test = vec![vec![0.1, 0.1], vec![0.9, 0.1]];

        let payload = run_model(&cfg, x_train, y_train, x_test, None).expect("run model");
        assert_eq!(payload.misclassification_cost, 0.0);
    }

    #[test]
    fn run_model_rejects_mismatched_y_test_length() {
        let cfg = build_runtime_config(
            "toy".to_string(),
            "rust".to_string(),
            Some(1),
            None,
            None,
            "hybrid".to_string(),
            20,
            5,
            1,
            0.3,
            5.0,
            Some(11),
            None,
            None,
        )
        .expect("build runtime config");

        let x_train = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let y_train = vec![0usize, 1, 1];
        let x_test = vec![vec![0.1, 0.1], vec![0.9, 0.1]];
        let y_test = vec![0usize];

        let err = run_model(&cfg, x_train, y_train, x_test, Some(y_test))
            .expect_err("expected y_test length mismatch error");
        assert!(err.to_string().contains("y_test length"));
    }
}
