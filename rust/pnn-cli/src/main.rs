use csv::StringRecord;
use pnn_core::{
    argmax, build_diagnostics, predict_proba, sample_posterior, InferenceMethod, ModelError,
    PnnModel, SamplerConfig, SamplerResult,
};
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
    /// Candidate k values for the Gibbs step. Populated from --k-values, --k-range, or --k.
    k_values: Vec<usize>,
    method: InferenceMethod,
    n_samples: usize,
    burn_in: usize,
    thinning: usize,
    /// Gaussian proposal std dev for the MH step on β. Tunes acceptance rate (target 20–50%).
    beta_step: f64,
    /// Half-normal prior scale on β. Encodes prior belief about interaction strength.
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
        &test.features,
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

    if let Some(ref diag_path) = config.diagnose {
        let diag = build_diagnostics(&result, &sampler_config, &model.k_values);
        if let Some(parent) = diag_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(diag_path, serde_json::to_string_pretty(&diag)?)?;
    }

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
    let mut k_range_explicit: Option<Vec<usize>> = None;
    let mut method = InferenceMethod::Hybrid;
    let mut n_samples = 1000usize;
    let mut burn_in = 500usize;
    let mut thinning = 1usize;
    let mut beta_step = 0.3_f64;
    let mut beta_sigma = 5.0_f64;
    let mut seed: Option<u64> = None;
    let mut diagnose: Option<PathBuf> = None;

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
            "--k-range" => {
                let raw = string_value("--k-range", next)?;
                let parts: Vec<&str> = raw.splitn(2, ',').collect();
                if parts.len() != 2 {
                    return Err(CliError::Message(
                        "invalid value for --k-range: expected start,end (e.g. 2,40)".to_string(),
                    ));
                }
                let start = parts[0].trim().parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid start in --k-range: '{}' (expected positive integer)",
                        parts[0].trim()
                    ))
                })?;
                let end = parts[1].trim().parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid end in --k-range: '{}' (expected positive integer)",
                        parts[1].trim()
                    ))
                })?;
                if start == 0 {
                    return Err(CliError::Message(
                        "invalid value for --k-range: start must be >= 1".to_string(),
                    ));
                }
                if end < start {
                    return Err(CliError::Message(format!(
                        "invalid value for --k-range: end ({end}) must be >= start ({start})"
                    )));
                }
                k_range_explicit = Some((start..=end).collect());
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
            "--thinning" => {
                let raw = string_value("--thinning", next)?;
                thinning = raw.parse::<usize>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --thinning: {raw} (expected positive integer)"
                    ))
                })?;
                if thinning == 0 {
                    return Err(CliError::Message(
                        "invalid value for --thinning: must be >= 1".to_string(),
                    ));
                }
                idx += 2;
            }
            "--beta-step" => {
                let raw = string_value("--beta-step", next)?;
                beta_step = raw.parse::<f64>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --beta-step: {raw} (expected positive float)"
                    ))
                })?;
                if beta_step <= 0.0 {
                    return Err(CliError::Message(
                        "invalid value for --beta-step: must be > 0".to_string(),
                    ));
                }
                idx += 2;
            }
            "--beta-sigma" => {
                let raw = string_value("--beta-sigma", next)?;
                beta_sigma = raw.parse::<f64>().map_err(|_| {
                    CliError::Message(format!(
                        "invalid value for --beta-sigma: {raw} (expected positive float)"
                    ))
                })?;
                if beta_sigma <= 0.0 {
                    return Err(CliError::Message(
                        "invalid value for --beta-sigma: must be > 0".to_string(),
                    ));
                }
                idx += 2;
            }
            "--method" => {
                let raw = string_value("--method", next)?;
                method = match raw.as_str() {
                    "hybrid" => InferenceMethod::Hybrid,
                    "joint-mh" => InferenceMethod::JointMh,
                    other => {
                        return Err(CliError::Message(format!(
                            "invalid value for --method: '{other}' (expected 'hybrid' or 'joint-mh')"
                        )));
                    }
                };
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
            "--diagnose" => {
                diagnose = Some(path_value("--diagnose", next)?);
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

    // Precedence: --k-values > --k-range > --k > default [3].
    let k_values = k_values_explicit
        .or(k_range_explicit)
        .unwrap_or_else(|| vec![k_single.unwrap_or(3)]);

    Ok(Config { train, test, output, dataset, implementation, k_values, method, n_samples, burn_in, thinning, beta_step, beta_sigma, seed, diagnose })
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
        [--k <int>] [--k-values <int,int,...>] [--k-range <start,end>] \
        [--method hybrid|joint-mh] \
        [--n-samples <int>] [--burn-in <int>] [--thinning <int>] [--seed <int>] \
        [--beta-step <float>] [--beta-sigma <float>] \
        [--diagnose <path>]",
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

    // ── --diagnose tests ──────────────────────────────────────────────────────

    fn args_with_diagnose(
        train: &std::path::Path,
        test: &std::path::Path,
        out: &std::path::Path,
        diag: &std::path::Path,
        extra: &[(&str, &str)],
    ) -> Vec<String> {
        let mut v = vec![
            "--train".to_string(), train.display().to_string(),
            "--test".to_string(),  test.display().to_string(),
            "--out".to_string(),   out.display().to_string(),
            "--diagnose".to_string(), diag.display().to_string(),
            "--k-values".to_string(), "1,2".to_string(),
            "--n-samples".to_string(), "40".to_string(),
            "--burn-in".to_string(), "10".to_string(),
            "--seed".to_string(), "7".to_string(),
        ];
        for (flag, val) in extra {
            v.push(flag.to_string());
            v.push(val.to_string());
        }
        v
    }

    #[test]
    fn diagnose_hybrid_schema_valid() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");
        let diag = temp.path().join("diag.json");

        run_with_args(args_with_diagnose(&train, &test, &out, &diag, &[]))
            .expect("run_with_args");

        let raw = fs::read_to_string(&diag).expect("read diag");
        let d: Value = serde_json::from_str(&raw).expect("valid json");

        // config block
        assert_eq!(d["config"]["method"], "Hybrid");
        assert_eq!(d["config"]["n_samples"], 40);
        assert_eq!(d["config"]["burn_in"], 10);
        assert_eq!(d["config"]["thinning"], 1);
        assert_eq!(d["config"]["k_candidates"]["start"], 1);
        assert_eq!(d["config"]["k_candidates"]["end"], 2);
        assert_eq!(d["config"]["total_iterations"], 50); // 10 + 40*1

        // acceptance
        assert!(d["mh_acceptance"]["rate"].as_f64().unwrap() >= 0.0);
        assert!(d["mh_acceptance"]["rate"].as_f64().unwrap() <= 1.0);
        assert!(d["mh_acceptance"]["n_accepted"].as_u64().unwrap()
            <= d["mh_acceptance"]["n_proposed"].as_u64().unwrap());

        // beta section
        assert_eq!(d["beta"]["trace"].as_array().unwrap().len(), 40);
        assert!(d["beta"]["acf"][0].as_f64().unwrap() > 0.999); // lag 0 ≈ 1.0
        assert!(d["beta"]["ess"].as_f64().unwrap() >= 1.0);
        assert!(d["beta"]["ess"].as_f64().unwrap() <= 40.0 + 1e-9);

        // k section
        assert_eq!(d["k"]["trace"].as_array().unwrap().len(), 40);
        let freq_sum: u64 = d["k"]["frequencies"]
            .as_object().unwrap().values()
            .map(|v| v.as_u64().unwrap()).sum();
        assert_eq!(freq_sum, 40);

        // burn_in: Hybrid must have beta_trace but NO k_trace
        assert_eq!(d["burn_in"]["beta_trace"].as_array().unwrap().len(), 10);
        assert!(d["burn_in"]["k_trace"].is_null(), "Hybrid burn_in.k_trace must be absent");
    }

    #[test]
    fn diagnose_joint_mh_has_k_trace_in_burn_in() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");
        let diag = temp.path().join("diag.json");

        run_with_args(args_with_diagnose(
            &train, &test, &out, &diag,
            &[("--method", "joint-mh")],
        ))
        .expect("run_with_args");

        let raw = fs::read_to_string(&diag).expect("read diag");
        let d: Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(d["config"]["method"], "JointMh");
        // JointMh burn_in must have both beta_trace and k_trace
        assert_eq!(d["burn_in"]["beta_trace"].as_array().unwrap().len(), 10);
        assert_eq!(d["burn_in"]["k_trace"].as_array().unwrap().len(), 10,
            "JointMh burn_in.k_trace must be present with length = burn_in");
    }

    #[test]
    fn diagnose_thinning_reflected_in_config_and_trace_length() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");
        let diag = temp.path().join("diag.json");

        run_with_args(args_with_diagnose(
            &train, &test, &out, &diag,
            &[("--thinning", "3")],
        ))
        .expect("run_with_args");

        let raw = fs::read_to_string(&diag).expect("read diag");
        let d: Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(d["config"]["thinning"], 3);
        assert_eq!(d["config"]["total_iterations"], 10 + 40 * 3); // 130
        // trace length is still n_samples=40 (thinning only affects iteration count)
        assert_eq!(d["beta"]["trace"].as_array().unwrap().len(), 40);
    }

    #[test]
    fn diagnose_records_beta_step_and_beta_sigma_in_config() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");
        let diag = temp.path().join("diag.json");

        run_with_args(args_with_diagnose(
            &train,
            &test,
            &out,
            &diag,
            &[("--beta-step", "0.05"), ("--beta-sigma", "2.5")],
        ))
        .expect("run_with_args");

        let raw = fs::read_to_string(&diag).expect("read diag");
        let d: Value = serde_json::from_str(&raw).expect("valid json");

        assert_eq!(d["config"]["beta_step"].as_f64().unwrap(), 0.05);
        assert_eq!(d["config"]["beta_sigma"].as_f64().unwrap(), 2.5);
    }

    #[test]
    fn diagnose_acceptance_rate_responds_to_beta_step() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);

        let out_small = temp.path().join("out_small.json");
        let diag_small = temp.path().join("diag_small.json");
        run_with_args(args_with_diagnose(
            &train,
            &test,
            &out_small,
            &diag_small,
            &[
                ("--n-samples", "120"),
                ("--burn-in", "30"),
                ("--beta-step", "0.05"),
            ],
        ))
        .expect("run_with_args small beta-step");

        let out_large = temp.path().join("out_large.json");
        let diag_large = temp.path().join("diag_large.json");
        run_with_args(args_with_diagnose(
            &train,
            &test,
            &out_large,
            &diag_large,
            &[
                ("--n-samples", "120"),
                ("--burn-in", "30"),
                ("--beta-step", "2.0"),
            ],
        ))
        .expect("run_with_args large beta-step");

        let small: Value =
            serde_json::from_str(&fs::read_to_string(&diag_small).expect("read small diag"))
                .expect("valid small diag json");
        let large: Value =
            serde_json::from_str(&fs::read_to_string(&diag_large).expect("read large diag"))
                .expect("valid large diag json");

        let small_rate = small["mh_acceptance"]["rate"].as_f64().unwrap();
        let large_rate = large["mh_acceptance"]["rate"].as_f64().unwrap();

        assert!(
            small_rate > large_rate,
            "expected smaller --beta-step to increase acceptance rate, got small={small_rate}, large={large_rate}"
        );
    }

    #[test]
    fn main_output_unchanged_when_diagnose_is_set() {
        let temp = tempdir().expect("tempdir");
        let (train, test) = toy_csvs(&temp);
        let out = temp.path().join("out.json");
        let diag = temp.path().join("diag.json");

        run_with_args(args_with_diagnose(&train, &test, &out, &diag, &[]))
            .expect("run_with_args");

        // Main output must still contain the standard schema fields
        let raw = fs::read_to_string(&out).expect("read main output");
        let j: Value = serde_json::from_str(&raw).expect("valid json");
        assert!(j["predictions"].is_array());
        assert!(j["k_posterior"].is_array());
        assert!(j["beta_posterior"].is_array());
        assert!(j["misclassification_cost"].is_number());
    }

    #[test]
    fn fails_on_invalid_method_value() {
        let err = run_with_args(vec!["--method".to_string(), "foo".to_string()])
            .expect_err("expected invalid --method value");
        assert!(
            err.to_string().contains("invalid value for --method"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fails_on_zero_thinning_value() {
        let err = run_with_args(vec!["--thinning".to_string(), "0".to_string()])
            .expect_err("expected invalid --thinning value");
        assert!(
            err.to_string().contains("invalid value for --thinning: must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fails_on_zero_beta_step_value() {
        let err = run_with_args(vec!["--beta-step".to_string(), "0".to_string()])
            .expect_err("expected invalid --beta-step value");
        assert!(
            err.to_string().contains("invalid value for --beta-step: must be > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fails_on_zero_beta_sigma_value() {
        let err = run_with_args(vec!["--beta-sigma".to_string(), "0".to_string()])
            .expect_err("expected invalid --beta-sigma value");
        assert!(
            err.to_string().contains("invalid value for --beta-sigma: must be > 0"),
            "unexpected error: {err}"
        );
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
