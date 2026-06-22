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

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

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
    {
        "name": "iris",
        "train": "data/iris_train.csv",
        "test": "data/iris_test.csv",
        "k_range": (1, 10),
        "n_samples": 1000,
        "burn_in": 200,
        "beta_step": 0.3,
    },
    {
        "name": "wine",
        "train": "data/wine_train.csv",
        "test": "data/wine_test.csv",
        "k_range": (1, 10),
        "n_samples": 1000,
        "burn_in": 200,
        "beta_step": 0.3,
    },
    {
        "name": "breast_cancer",
        "train": "data/breast_cancer_train.csv",
        "test": "data/breast_cancer_test.csv",
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

        # Normalize: fit on train, apply to both splits
        scaler = StandardScaler()
        train_arr = scaler.fit_transform(np.array(train_x))
        test_arr = scaler.transform(np.array(test_x))
        train_x = train_arr.tolist()
        test_x = test_arr.tolist()

        # Select top-2 features by ANOVA F-score for plotting
        f_scores, _ = f_classif(train_arr, train_y)
        top2 = np.argsort(f_scores)[::-1][:2]
        feat_x, feat_y = int(top2[0]), int(top2[1])
        print(f"\n  Plot features: {feat_x} and {feat_y} "
              f"(F={f_scores[feat_x]:.1f}, F={f_scores[feat_y]:.1f})", end="")

        # Run both methods; hybrid drives the accuracy table and boundary plot
        METHODS = ["hybrid", "joint-mh"]
        method_payloads: dict = {}
        method_diag_paths: dict = {}

        start_time = time.perf_counter()
        for method in METHODS:
            method_tag = method.replace("-", "_")
            out_p  = f"benchmarks/out/{name}_{method_tag}_eval.json"
            diag_p = f"benchmarks/out/{name}_{method_tag}_diag.json"
            method_diag_paths[method] = diag_p
            try:
                method_payloads[method] = pnn_py.run_from_arrays(
                    x_train=train_x,
                    y_train=train_y,
                    x_test=test_x,
                    y_test=test_y,
                    dataset=name,
                    implementation=f"rust-py-{method_tag}",
                    method=method,
                    k_range=ds["k_range"],
                    n_samples=ds["n_samples"],
                    burn_in=ds["burn_in"],
                    beta_step=ds["beta_step"],
                    seed=42,
                    out_path=out_p,
                    diagnose_path=diag_p,
                )
            except Exception as e:
                print(f"\n  ERROR ({method} failed: {e})")

        if "hybrid" not in method_payloads:
            print(" SKIPPING (hybrid run failed)")
            continue

        payload = method_payloads["hybrid"]
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Classification boundary plot (hybrid only — avoids doubling the grid prediction cost)
        try:
            from plotting_results import plot_classification_results
            plot_classification_results(
                x_train=train_x,
                y_train=train_y,
                x_test=test_x,
                y_test=test_y,
                dataset_name=name,
                sampler_config={
                    "k_range": ds["k_range"],
                    "n_samples": ds["n_samples"],
                    "burn_in": ds["burn_in"],
                    "beta_step": ds["beta_step"],
                    "seed": 42,
                },
                x_feature_idx=feat_x,
                y_feature_idx=feat_y,
            )
        except Exception as plot_err:
            print(f"  [Warning] Boundary plot failed: {plot_err}")

        # MCMC diagnostics plots for each method
        try:
            from plot_diagnostics import plot_diagnostics
            output_dir = Path("benchmarks") / f"plot_results_{name}"
            for method, diag_p in method_diag_paths.items():
                plot_diagnostics(diag_p, name, method, output_dir)
        except Exception as diag_err:
            print(f"  [Warning] Diagnostics plot failed: {diag_err}")

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
