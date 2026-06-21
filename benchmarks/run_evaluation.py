#!/usr/bin/env python3
"""
Run Bayesian k-NN evaluation across all standard datasets in the data/ directory,
using the pnn_py Python binding. Outputs a summary Markdown table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import time

import pnn_py

DATASETS = [
    {
        "name": "cushings",
        "train": "data/cushings_train.csv",
        "test": "data/cushings_test.csv",
        "k_range": (1, 5),
        "n_samples": 1000,
        "burn_in": 200,
        "beta_step": 0.3,
    },
    {
        "name": "viruses",
        "train": "data/viruses_train.csv",
        "test": "data/viruses_test.csv",
        "k_range": (1, 5),
        "n_samples": 1000,
        "burn_in": 300,
        "beta_step": 0.3,
    },
    {
        "name": "crabs",
        "train": "data/crabs_train.csv",
        "test": "data/crabs_test.csv",
        "k_range": (1, 20),
        "n_samples": 2000,
        "burn_in": 500,
        "beta_step": 0.3,
    },
    {
        "name": "fglass",
        "train": "data/fglass_train.csv",
        "test": "data/fglass_test.csv",
        "k_range": (1, 10),
        "n_samples": 2000,
        "burn_in": 500,
        "beta_step": 0.3,
    },
    {
        "name": "pima",
        "train": "data/pima_train.csv",
        "test": "data/pima_test.csv",
        "k_range": (1, 15),
        "n_samples": 2000,
        "burn_in": 500,
        "beta_step": 0.15,
    },
    {
        "name": "synth",
        "train": "data/synth_train.csv",
        "test": "data/synth_test.csv",
        "k_range": (1, 20),
        "n_samples": 2000,
        "burn_in": 500,
        "beta_step": 0.3,
    },
]


def load_labeled_csv(path_str: str) -> tuple[list[list[float]], list[int]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"{path}: missing header row")

        features: list[list[float]] = []
        labels: list[int] = []

        for row in reader:
            if not row:
                continue
            features.append([float(value) for value in row[:-1]])
            labels.append(int(row[-1]))

    return features, labels


def run_eval():
    print("======================================================================")
    print("Running Bayesian k-NN Evaluation across all standard datasets...")
    print("======================================================================")

    results = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"Evaluating dataset: {name} ...", end="", flush=True)

        try:
            train_x, train_y = load_labeled_csv(ds["train"])
            test_x, test_y = load_labeled_csv(ds["test"])
        except Exception as e:
            print(f" ERROR (Failed to load CSV: {e})")
            continue

        start_time = time.perf_counter()
        try:
            payload = pnn_py.run_from_arrays(
                x_train=train_x,
                y_train=train_y,
                x_test=test_x,
                y_test=test_y,
                dataset=name,
                implementation="rust-py-evaluation",
                k_range=ds["k_range"],
                n_samples=ds["n_samples"],
                burn_in=ds["burn_in"],
                beta_step=ds["beta_step"],
                seed=42,
                out_path=f"benchmarks/out/{name}_eval_rust_py.json",
                diagnose_path=f"benchmarks/out/{name}_eval_diag_rust_py.json",
            )
        except Exception as e:
            print(f" ERROR (Execution failed: {e})")
            continue

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        misclass_cost = payload["misclassification_cost"]
        accuracy = (1.0 - misclass_cost) * 100.0
        
        k_posterior = payload["k_posterior"]
        beta_posterior = payload["beta_posterior"]

        mean_k = statistics.mean(k_posterior)
        mean_beta = statistics.mean(beta_posterior)

        results.append({
            "name": name,
            "n_train": len(train_x),
            "n_test": len(test_x),
            "accuracy": accuracy,
            "mean_k": mean_k,
            "mean_beta": mean_beta,
            "runtime_ms": payload["runtime_ms"],
            "total_python_ms": duration_ms,
        })
        print(" DONE")

    # Render Markdown table
    report = []
    report.append("# Bayesian k-NN Model Evaluation Report")
    report.append("")
    report.append("This report lists accuracy, parameter averages, and computational performance across all datasets using the `pnn_py` Python bindings.")
    report.append("")
    report.append("| Dataset | Train Size | Test Size | Test Accuracy | Mean Posterior $k$ | Mean Posterior $\\beta$ | Rust Engine Time (ms) | Total Python Time (ms) |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        report.append(
            f"| {r['name'].capitalize()} | {r['n_train']} | {r['n_test']} | {r['accuracy']:.2f}% | {r['mean_k']:.2f} | {r['mean_beta']:.4f} | {r['runtime_ms']:.2f} | {r['total_python_ms']:.2f} |"
        )

    report_text = "\n".join(report)
    print("\n")
    print(report_text)
    print("\n")

    # Write report
    report_file = Path("benchmarks/evaluation_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    run_eval()
