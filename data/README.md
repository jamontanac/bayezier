# Data Contract

This directory contains shared datasets used by both implementations.

## Purpose

- Rust and Zig read the same CSV files.
- Any benchmark result must be reproducible from these files.

## CSV conventions

- Encoding: UTF-8.
- Delimiter: comma (`,`).
- Header row: required.
- Each row is one sample.
- Feature columns come first.
- Label column comes last and is named `label`.

## Value conventions

- Feature values must be numeric and parse to `f64`.
- `label` is an integer class id (`0..n_classes-1`).
- Missing values are not allowed in phase 1.

## Train/test split conventions

- Use paired files per dataset when possible:
  - `<dataset>_train.csv`
  - `<dataset>_test.csv`
- Train and test files must have the same feature column names and order.

## Minimal example

```csv
x1,x2,label
0.1,1.2,0
0.4,1.5,1
```
