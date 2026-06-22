#!/usr/bin/env python3
"""
Generate train/test CSV splits from sklearn's built-in datasets.

Run with: pixi run -e dev generate-datasets

Outputs (in data/):
    iris_train.csv / iris_test.csv
    wine_train.csv / wine_test.csv
    breast_cancer_train.csv / breast_cancer_test.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn import datasets
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).parent
SEED = 42
TEST_FRAC = 0.2


def clean_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def write_csv(path: Path, feature_names: list[str], X, y) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["label"])
        for row, lbl in zip(X, y):
            writer.writerow([float(v) for v in row] + [int(lbl)])
    print(f"  wrote {path}  ({len(X)} rows, {len(feature_names)} features)")


def generate(name: str, data) -> None:
    print(f"{name}:")
    X, y = data.data, data.target
    feat_names = [clean_name(str(n)) for n in data.feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y
    )
    write_csv(OUT_DIR / f"{name}_train.csv", feat_names, X_train, y_train)
    write_csv(OUT_DIR / f"{name}_test.csv", feat_names, X_test, y_test)


if __name__ == "__main__":
    generate("iris", datasets.load_iris())
    generate("wine", datasets.load_wine())
    generate("breast_cancer", datasets.load_breast_cancer())
    print(f"\nDone. All CSVs written to {OUT_DIR}")
