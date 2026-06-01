# bayezier

A dual-language exploration of Bayesian kNN: posterior distributions over k and β, built from scratch in Rust and Zig.

Bayesian k-nearest neighbour classification — smooth posterior curves over k and β, implemented in Rust and Zig.

A Rust-first, dual-language exploration of Bayesian kNN with a parity pipeline designed for Rust and Zig outputs.

Current phase:
- Rust implementation is active and executable.
- Zig is design-only for now.
- Python is the neutral comparison layer.
- Pixi is the project environment and task runner.

Based on [Holmes & Adams (2002)](https://hedibert.org/wp-content/uploads/2016/02/holmes-adams-2002.pdf).

[![Rust CI](https://github.com/jamontanac/bayezier/actions/workflows/rust.yml/badge.svg)](https://github.com/jamontanac/bayezier/actions/workflows/rust.yml)
[![Zig CI](https://github.com/jamontanac/bayezier/actions/workflows/zig.yml/badge.svg)](https://github.com/jamontanac/bayezier/actions/workflows/zig.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Repository layout

- `rust/`: Rust workspace (`pnn-core`, `pnn-cli`, `pnn-py` scaffold).
- `zig/`: Zig design placeholder for upcoming implementation.
- `data/`: Shared CSV contract and sample datasets.
- `benchmarks/`: Shared JSON contract, outputs, and comparison tool.
- `first_part_rust.md`: strict ticketed implementation plan and tracker.

## Contracts (shared by Rust and Zig)

- Input CSV contract: `data/README.md`
- Output benchmark JSON contract: `benchmarks/README.md`
- Parity tolerance target: `max(abs(rust_probability - zig_probability)) < 1e-4`

## Environment setup (Pixi)

Install Pixi, then from repo root:

```bash
pixi install
```

## Command matrix

- Rust workspace check:

```bash
pixi run rust-check
```

- Rust tests:

```bash
pixi run rust-test
```

- Generate Rust benchmark output (`benchmarks/out/rust.json`):

```bash
pixi run rust-benchmark
```

- Compare Rust output against baseline (`benchmarks/out/rust_baseline.json`):

```bash
pixi run compare
```

- End-to-end parity task (Rust benchmark + compare):

```bash
pixi run parity
```

## Rust CLI usage

Current CLI executable: `rust/pnn-cli`

Example:

```bash
cargo run -p pnn-cli --manifest-path rust/Cargo.toml -- \
  --train data/sample_train.csv \
  --test data/sample_test.csv \
  --out benchmarks/out/rust.json \
  --dataset sample \
  --implementation rust \
  --k 3 \
  --beta 1.0
```

The command writes a JSON payload compatible with `benchmarks/README.md`.

## Python comparator usage

Rust vs Zig (future main path):

```bash
python benchmarks/compare.py \
  --rust-output benchmarks/out/rust.json \
  --zig-output benchmarks/out/zig.json \
  --tolerance 1e-4
```

Rust vs baseline (current Rust-first path):

```bash
python benchmarks/compare.py \
  --rust-output benchmarks/out/rust.json \
  --baseline-output benchmarks/out/rust_baseline.json \
  --tolerance 1e-4
```

## Status and roadmap

- Done now:
  - Rust kNN core + validation foundations.
  - Rust CLI CSV->JSON contract path.
  - Rust baseline parity comparator (`benchmarks/compare.py`).
  - Pixi task orchestration.
- Next:
  1. Complete Bayesian inference modules in `rust/pnn-core`.
  2. Expand contract-focused integration tests.
  3. Finalize Zig CLI design doc and then start Zig implementation.
