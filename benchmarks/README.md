# Benchmark Output Contract

Both CLIs must write benchmark output using the same JSON schema.

## Required top-level fields

- `implementation` (string): `"rust"` or `"zig"`.
- `dataset` (string): dataset name, for example `"ripley"`.
- `predictions` (array): prediction objects, one per test sample.
- `k_posterior` (array of integers): sampled `k` values.
- `beta_posterior` (array of numbers): sampled `beta` values.
- `misclassification_cost` (number): scalar cost `C` on the test set.
- `runtime_ms` (number): runtime in milliseconds.

## Prediction object schema

Each element in `predictions` must include:

- `index` (integer): test-row index.
- `probabilities` (array of numbers): per-class probabilities in class-id order.
- `predicted_class` (integer): argmax class.

Rules:

- `len(predictions)` must equal number of test rows.
- `sum(probabilities)` must be approximately 1.0 for each prediction.

## Example

```json
{
  "implementation": "rust",
  "dataset": "ripley",
  "predictions": [
    { "index": 0, "probabilities": [0.72, 0.28], "predicted_class": 0 },
    { "index": 1, "probabilities": [0.31, 0.69], "predicted_class": 1 }
  ],
  "k_posterior": [5, 7, 5, 6],
  "beta_posterior": [1.24, 1.37, 1.19, 1.41],
  "misclassification_cost": 0.09,
  "runtime_ms": 142.7
}
```

## Parity rule

`benchmarks/compare.py` will assert numerical agreement between Rust and Zig:

- `max(abs(rust_probability - zig_probability)) < 1e-4`

The comparison runs on the same dataset and matching prediction indices.
