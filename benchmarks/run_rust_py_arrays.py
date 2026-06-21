#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pnn_py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Rust Bayesian PNN via Python binding using in-memory arrays "
            "(list or NumPy)."
        )
    )
    parser.add_argument("--train", required=True, help="Path to train CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--out", help="Primary output JSON path")
    parser.add_argument(
        "--parity-out",
        help=(
            "Optional extra output JSON path for parity checks. "
            "If --out is omitted, this path is used as out_path in Rust."
        ),
    )
    parser.add_argument("--dataset", default="unknown", help="Dataset name")
    parser.add_argument("--implementation", default="rust-py-arrays", help="Implementation label")
    parser.add_argument("--k", type=int, help="Single k candidate")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        help="Explicit k candidates (space-separated)",
    )
    parser.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive k range",
    )
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=("hybrid", "joint-mh"),
        help="Inference method",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--beta-step", type=float, default=0.3)
    parser.add_argument("--beta-sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--diagnose", help="Optional diagnostics JSON path")
    parser.add_argument(
        "--use-numpy",
        action="store_true",
        help="Convert arrays to NumPy before calling run_from_arrays",
    )
    return parser.parse_args()


def load_labeled_csv(path_str: str) -> tuple[list[list[float]], list[int]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"{path}: missing header row")
        if len(headers) < 2:
            raise ValueError(
                f"{path}: expected at least one feature column plus `label`"
            )
        if headers[-1] != "label":
            raise ValueError(f"{path}: last column must be named `label`")

        features: list[list[float]] = []
        labels: list[int] = []

        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(
                    f"{path} row {row_number}: expected at least one feature and one label"
                )

            try:
                feature_row = [float(value) for value in row[:-1]]
            except ValueError as exc:
                raise ValueError(
                    f"{path} row {row_number}: non-numeric feature value"
                ) from exc

            try:
                label = int(row[-1])
            except ValueError as exc:
                raise ValueError(
                    f"{path} row {row_number}: label must be a non-negative integer"
                ) from exc

            if label < 0:
                raise ValueError(
                    f"{path} row {row_number}: label must be a non-negative integer"
                )

            features.append(feature_row)
            labels.append(label)

    return features, labels


def maybe_to_numpy(
    use_numpy: bool,
    x_train: list[list[float]],
    y_train: list[int],
    x_test: list[list[float]],
    y_test: list[int],
):
    if not use_numpy:
        return x_train, y_train, x_test, y_test

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "--use-numpy requested but numpy is not installed in this environment"
        ) from exc

    return (
        np.asarray(x_train, dtype=float),
        np.asarray(y_train, dtype=int),
        np.asarray(x_test, dtype=float),
        np.asarray(y_test, dtype=int),
    )


def write_json(path_str: str, payload: dict) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    train_x, train_y = load_labeled_csv(args.train)
    test_x, test_y = load_labeled_csv(args.test)

    x_train, y_train, x_test, y_test = maybe_to_numpy(
        args.use_numpy,
        train_x,
        train_y,
        test_x,
        test_y,
    )

    binding_out_path = args.out or args.parity_out
    payload = pnn_py.run_from_arrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        dataset=args.dataset,
        implementation=args.implementation,
        k=args.k,
        k_values=args.k_values,
        k_range=tuple(args.k_range) if args.k_range else None,
        method=args.method,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        beta_step=args.beta_step,
        beta_sigma=args.beta_sigma,
        seed=args.seed,
        out_path=binding_out_path,
        diagnose_path=args.diagnose,
    )

    if args.parity_out and args.parity_out != binding_out_path:
        write_json(args.parity_out, payload)

    print(
        json.dumps(
            {
                "implementation": payload["implementation"],
                "dataset": payload["dataset"],
                "array_mode": "numpy" if args.use_numpy else "list",
                "n_train": len(train_x),
                "n_test": len(test_x),
                "n_predictions": len(payload["predictions"]),
                "n_samples": len(payload["k_posterior"]),
                "misclassification_cost": payload["misclassification_cost"],
                "runtime_ms": payload["runtime_ms"],
                "out_path": binding_out_path,
                "parity_out": args.parity_out,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
