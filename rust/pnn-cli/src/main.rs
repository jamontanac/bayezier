use csv::StringRecord;
use pnn_core::{argmax, predict_proba, sample_posterior, ModelError, PnnModel, SamplerConfig};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
enum CliError {
    Message(String),
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(msg) => write!(f, "{msg}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::Csv(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
        }
    }
}

impl Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for CliError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<ModelError> for CliError {
    fn from(e: ModelError) -> Self {
        Self::Message(e.to_string())
    }
}

#[derive(Debug)]
struct Config {
    train: PathBuf,
    test: PathBuf,
    output: PathBuf,
    dataset: String,
    implementation: String,
    /// Candidate k values for the Gibbs step. Populated from --k-values or --k.
    k_values: Vec<usize>,
    n_samples: usize,
    burn_in: usize,
    seed: Option<u64>,
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

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    run_with_args(env::args().skip(1))
}

fn run_with_args<I>(args: I) -> Result<(), CliError>
where
    I: IntoIterator<Item = String>,
{
    let config = parse_args(args)?;
    let started_at = Instant::now();

    let train = read_labeled_csv(&config.train)?;
    let test = read_labeled_csv(&config.test)?;

    if train.features.is_empty() {
        return Err(CliError::Message("training CSV has no rows".to_string()));
    }

    let n_classes = train.labels.iter().copied().max().map_or(1, |label| label + 1);
    let n_train = train.features.len();
    // Clamp each candidate to [1, n_train-1]; PnnModel::new enforces the strict bound.
    let k_max_valid = n_train.saturating_sub(1).max(1);
    let k_values: Vec<usize> = config
        .k_values
        .iter()
        .map(|&k| k.min(k_max_valid).max(1))
        .collect();

    let model = PnnModel::new(
        train.features,
        train.labels,
        n_classes,
        k_values,
    )?;

    let sampler_config = SamplerConfig {
        n_samples: config.n_samples,
        burn_in: config.burn_in,
        seed: config.seed,
        ..SamplerConfig::default()
    };

    let chain = sample_posterior(&model, &sampler_config);

    let proba = predict_proba(
        &test.features,
        &model.x_train,
        &model.y_train,
        &chain,
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

    let k_posterior: Vec<usize> = chain.iter().map(|d| d.k).collect();
    let beta_posterior: Vec<f64> = chain.iter().map(|d| d.beta).collect();

    let misclassification_cost = compute_misclassification_cost(&predictions, &test.labels);
    let runtime_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let payload = BenchmarkOutput {
        implementation: config.implementation,
        dataset: config.dataset,
        predictions,
        k_posterior,
        beta_posterior,
        misclassification_cost,
        runtime_ms,
    };

    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&payload)?;
    fs::write(&config.output, json)?;

    Ok(())
}

fn parse_args<I>(args: I) -> Result<Config, CliError>
where
    I: IntoIterator<Item = String>,
{
    let mut train = None;
    let mut test = None;
    let mut output = None;
    let mut dataset = String::from("unknown");
    let mut implementation = String::from("rust");
    let mut k_single: Option<usize> = None;
    let mut k_values_explicit: Option<Vec<usize>> = None;
    let mut n_samples = 1000usize;
    let mut burn_in = 500usize;
    let mut seed: Option<u64> = None;

    let args_vec: Vec<String> = args.into_iter().collect();
    let mut idx = 0usize;
    while idx < args_vec.len() {
        let arg = &args_vec[idx];
        let next = args_vec.get(idx + 1).cloned();

        match arg.as_str() {
            "--train" => {
                train = Some(path_value("--train", next)?);
                idx += 2;
            }
            "--test" => {
                test = Some(path_value("--test", next)?);
                idx += 2;
            }
            "--out" => {
                output = Some(path_value("--out", next)?);
                idx += 2;
            }
            "--dataset" => {
                dataset = string_value("--dataset", next)?;
                idx += 2;
            }
            "--implementation" => {
                implementation = string_value("--implementation", next)?;
                idx += 2;
            }
            "--k" => {
                let raw = string_value("--k", next)?;
                let v = raw.parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --k: {raw} (expected positive integer)"
                    ))
                })?;
                if v == 0 {
                    return Err(CliError::Message(
                        "invalid value for --k: must be >= 1".to_string(),
                    ));
                }
                k_single = Some(v);
                idx += 2;
            }
            "--k-values" => {
                let raw = string_value("--k-values", next)?;
                let mut vals = Vec::new();
                for part in raw.split(',') {
                    let v = part.trim().parse::<usize>().map_err(|_| {
                        CliError::Message(format!(
                            "invalid value in --k-values: '{part}' (expected comma-separated positive integers)"
                        ))
                    })?;
                    if v == 0 {
                        return Err(CliError::Message(
                            "invalid value in --k-values: all values must be >= 1".to_string(),
                        ));
                    }
                    vals.push(v);
                }
                if vals.is_empty() {
                    return Err(CliError::Message(
                        "invalid value for --k-values: must supply at least one value".to_string(),
                    ));
                }
                k_values_explicit = Some(vals);
                idx += 2;
            }
            "--n-samples" => {
                let raw = string_value("--n-samples", next)?;
                n_samples = raw.parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --n-samples: {raw} (expected positive integer)"
                    ))
                })?;
                if n_samples == 0 {
                    return Err(CliError::Message(
                        "invalid value for --n-samples: must be >= 1".to_string(),
                    ));
                }
                idx += 2;
            }
            "--burn-in" => {
                let raw = string_value("--burn-in", next)?;
                burn_in = raw.parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --burn-in: {raw} (expected non-negative integer)"
                    ))
                })?;
                idx += 2;
            }
            "--seed" => {
                let raw = string_value("--seed", next)?;
                let s = raw.parse::<u64>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --seed: {raw} (expected non-negative integer)"
                    ))
                })?;
                seed = Some(s);
                idx += 2;
            }
            "--help" | "-h" => {
                return Err(CliError::Message(usage()));
            }
            _ => {
                return Err(CliError::Message(format!(
                    "unknown argument: {arg}\n{}",
                    usage()
                )));
            }
        }
    }

    let train =
        train.ok_or_else(|| CliError::Message(format!("missing required --train\n{}", usage())))?;
    let test =
        test.ok_or_else(|| CliError::Message(format!("missing required --test\n{}", usage())))?;
    let output =
        output.ok_or_else(|| CliError::Message(format!("missing required --out\n{}", usage())))?;

    // --k-values takes precedence; fall back to --k (single candidate); default to [3].
    let k_values = k_values_explicit.unwrap_or_else(|| vec![k_single.unwrap_or(3)]);

    Ok(Config { train, test, output, dataset, implementation, k_values, n_samples, burn_in, seed })
}

fn path_value(flag: &str, next: Option<String>) -> Result<PathBuf, CliError> {
    let value = string_value(flag, next)?;
    Ok(PathBuf::from(value))
}

fn string_value(flag: &str, next: Option<String>) -> Result<String, CliError> {
    next.ok_or_else(|| CliError::Message(format!("missing value for {flag}")))
}

fn usage() -> String {
    String::from(
        "Usage: pnn-cli --train <path> --test <path> --out <path> \
        [--dataset <name>] [--implementation <str>] \
        [--k <int>] [--k-values <int,int,...>] \
        [--n-samples <int>] [--burn-in <int>] [--seed <int>]",
    )
}

fn read_labeled_csv(path: &Path) -> Result<Dataset, CliError> {
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

fn validate_headers(path: &Path, headers: &StringRecord) -> Result<(), CliError> {
    if headers.len() < 2 {
        return Err(CliError::Message(format!(
            "{}: expected at least one feature column plus `label`",
            path.display()
        )));
    }

    match headers.get(headers.len() - 1) {
        Some("label") => Ok(()),
        _ => Err(CliError::Message(format!(
            "{}: last column must be named `label`",
            path.display()
        ))),
    }
}

fn parse_row(
    path: &Path,
    row: &StringRecord,
    row_number: usize,
) -> Result<(Vec<f64>, usize), CliError> {
    if row.len() < 2 {
        return Err(CliError::Message(format!(
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
                CliError::Message(format!(
                    "{} row {}: missing value at column {}",
                    path.display(),
                    row_number,
                    col_idx
                ))
            })?
            .parse::<f64>()
            .map_err(|_| {
                CliError::Message(format!(
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
            CliError::Message(format!(
                "{} row {}: missing label value",
                path.display(),
                row_number
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            CliError::Message(format!(
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
    use super::run_with_args;
    use serde_json::Value;
    use std::fs;
    use tempfile::tempdir;

    fn toy_csvs(temp: &tempfile::TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
        let train = temp.path().join("train.csv");
        let test = temp.path().join("test.csv");
        fs::write(&train, "x1,x2,label\n0.0,0.0,0\n1.0,0.0,1\n0.0,1.0,1\n")
            .expect("write train");
        fs::write(&test, "x1,x2,label\n0.1,0.1,0\n0.9,0.1,1\n").expect("write test");
        (train, test)
    }

    #[test]
    fn writes_schema_valid_json_from_csv_inputs() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out").join("rust.json");

        run_with_args(vec![
            "--train".to_string(),
            train.display().to_string(),
            "--test".to_string(),
            test.display().to_string(),
            "--out".to_string(),
            out.display().to_string(),
            "--dataset".to_string(),
            "toy".to_string(),
            "--k".to_string(),
            "1".to_string(),
            "--n-samples".to_string(),
            "50".to_string(),
            "--burn-in".to_string(),
            "10".to_string(),
            "--seed".to_string(),
            "1".to_string(),
        ])
        .expect("run_with_args");

        let raw = fs::read_to_string(out).expect("read output");
        let json: Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(json["implementation"], "rust");
        assert_eq!(json["dataset"], "toy");
        assert!(json["predictions"].is_array());
        assert_eq!(json["predictions"].as_array().unwrap().len(), 2);
        assert!(json["k_posterior"].is_array());
        assert_eq!(json["k_posterior"].as_array().unwrap().len(), 50);
        assert!(json["beta_posterior"].is_array());
        assert_eq!(json["beta_posterior"].as_array().unwrap().len(), 50);
        assert!(json["misclassification_cost"].is_number());
        assert!(json["runtime_ms"].is_number());
    }

    #[test]
    fn beta_posterior_all_positive() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");

        run_with_args(vec![
            "--train".to_string(),
            train.display().to_string(),
            "--test".to_string(),
            test.display().to_string(),
            "--out".to_string(),
            out.display().to_string(),
            "--n-samples".to_string(),
            "100".to_string(),
            "--burn-in".to_string(),
            "20".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ])
        .expect("run_with_args");

        let raw = fs::read_to_string(out).expect("read output");
        let json: Value = serde_json::from_str(&raw).expect("valid json");

        let betas = json["beta_posterior"].as_array().unwrap();
        assert_eq!(betas.len(), 100);
        for b in betas {
            assert!(b.as_f64().unwrap() > 0.0, "beta must be positive, got {b}");
        }
    }

    #[test]
    fn fails_when_csv_header_does_not_end_with_label() {
        let temp = tempdir().expect("tempdir");
        let train = temp.path().join("train.csv");
        let test = temp.path().join("test.csv");
        let out = temp.path().join("out.json");

        fs::write(&train, "x1,x2,class\n0.0,0.0,0\n").expect("write train");
        fs::write(&test, "x1,x2,label\n0.1,0.1,0\n").expect("write test");

        let err = run_with_args(vec![
            "--train".to_string(),
            train.display().to_string(),
            "--test".to_string(),
            test.display().to_string(),
            "--out".to_string(),
            out.display().to_string(),
        ])
        .expect_err("expected header validation failure");

        assert!(err.to_string().contains("last column must be named `label`"));
    }
}
