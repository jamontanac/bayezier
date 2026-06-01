use csv::StringRecord;
use pnn_core::knn::k_nearest;
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

#[derive(Debug)]
struct Config {
    train: PathBuf,
    test: PathBuf,
    output: PathBuf,
    dataset: String,
    implementation: String,
    k: usize,
    beta: f64,
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

    let n_classes = train
        .labels
        .iter()
        .copied()
        .max()
        .map_or(1, |label| label + 1);
    let effective_k = config.k.min(train.features.len());

    let predictions = build_predictions(&train, &test.features, n_classes, effective_k)?;

    let misclassification_cost = compute_misclassification_cost(&predictions, &test.labels);
    let runtime_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

    let payload = BenchmarkOutput {
        implementation: config.implementation,
        dataset: config.dataset,
        predictions,
        k_posterior: vec![effective_k],
        beta_posterior: vec![config.beta],
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
    let mut k = 3usize;
    let mut beta = 1.0_f64;

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
                k = raw.parse::<usize>().map_err(|_| {
                    CliError::Message(format!("invalid value for --k: {raw} (expected positive integer)"))
                })?;
                if k == 0 {
                    return Err(CliError::Message(
                        "invalid value for --k: must be >= 1".to_string(),
                    ));
                }
                idx += 2;
            }
            "--beta" => {
                let raw = string_value("--beta", next)?;
                beta = raw.parse::<f64>().map_err(|_| {
                    CliError::Message(format!("invalid value for --beta: {raw} (expected float)"))
                })?;
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

    let train = train.ok_or_else(|| CliError::Message(format!("missing required --train\n{}", usage())))?;
    let test = test.ok_or_else(|| CliError::Message(format!("missing required --test\n{}", usage())))?;
    let output =
        output.ok_or_else(|| CliError::Message(format!("missing required --out\n{}", usage())))?;

    Ok(Config {
        train,
        test,
        output,
        dataset,
        implementation,
        k,
        beta,
    })
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
        "Usage: pnn-cli --train <path> --test <path> --out <path> [--dataset <name>] [--implementation rust] [--k <int>] [--beta <float>]",
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

fn parse_row(path: &Path, row: &StringRecord, row_number: usize) -> Result<(Vec<f64>, usize), CliError> {
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

fn build_predictions(
    train: &Dataset,
    test_features: &[Vec<f64>],
    n_classes: usize,
    k: usize,
) -> Result<Vec<Prediction>, CliError> {
    let mut predictions = Vec::with_capacity(test_features.len());

    for (index, sample) in test_features.iter().enumerate() {
        let neighbors = k_nearest(&train.features, sample, k)
            .map_err(|err| CliError::Message(format!("prediction error at row {index}: {err}")))?;

        let mut counts = vec![0usize; n_classes];
        for neighbor_idx in neighbors {
            let class = train.labels[neighbor_idx];
            if class < n_classes {
                counts[class] += 1;
            }
        }

        let denom = k as f64;
        let probabilities: Vec<f64> = counts
            .iter()
            .map(|count| (*count as f64) / denom)
            .collect();

        let predicted_class = argmax(&probabilities);

        predictions.push(Prediction {
            index,
            probabilities,
            predicted_class,
        });
    }

    Ok(predictions)
}

fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;

    for (idx, &value) in values.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }

    best_idx
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

    #[test]
    fn writes_schema_valid_json_from_csv_inputs() {
        let temp = tempdir().expect("tempdir");
        let train = temp.path().join("train.csv");
        let test = temp.path().join("test.csv");
        let out = temp.path().join("out").join("rust.json");

        fs::write(
            &train,
            "x1,x2,label\n0.0,0.0,0\n1.0,0.0,1\n0.0,1.0,1\n",
        )
        .expect("write train");
        fs::write(&test, "x1,x2,label\n0.1,0.1,0\n0.9,0.1,1\n").expect("write test");

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
        ])
        .expect("run_with_args");

        let raw = fs::read_to_string(out).expect("read output");
        let json: Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(json["implementation"], "rust");
        assert_eq!(json["dataset"], "toy");
        assert!(json["predictions"].is_array());
        assert!(json["k_posterior"].is_array());
        assert!(json["beta_posterior"].is_array());
        assert!(json["misclassification_cost"].is_number());
        assert!(json["runtime_ms"].is_number());
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
