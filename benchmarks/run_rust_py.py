#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pnn_py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Rust Bayesian PNN via Python binding from CSV files."
    )
    parser.add_argument("--train", required=True, help="Path to train CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--dataset", default="unknown", help="Dataset name")
    parser.add_argument("--implementation", default="rust-py", help="Implementation label")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = pnn_py.run_from_csv(
        train_path=args.train,
        test_path=args.test,
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
        out_path=args.out,
        diagnose_path=args.diagnose,
    )

    print(
        json.dumps(
            {
                "implementation": payload["implementation"],
                "dataset": payload["dataset"],
                "n_predictions": len(payload["predictions"]),
                "n_samples": len(payload["k_posterior"]),
                "misclassification_cost": payload["misclassification_cost"],
                "runtime_ms": payload["runtime_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
