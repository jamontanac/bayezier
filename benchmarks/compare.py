#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL = {
    "implementation": str,
    "dataset": str,
    "predictions": list,
    "k_posterior": list,
    "beta_posterior": list,
    "misclassification_cost": (int, float),
    "runtime_ms": (int, float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rust and Zig benchmark outputs for parity."
    )
    parser.add_argument("--rust-output", required=True, help="Path to Rust JSON output")
    parser.add_argument(
        "--zig-output",
        help="Path to Zig JSON output. Optional during Rust-only phase.",
    )
    parser.add_argument(
        "--baseline-output",
        help="Path to baseline JSON output (used if --zig-output is not supplied)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute probability delta",
    )
    return parser.parse_args()


def load_json(path_str: str, label: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        fail(f"{label} file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{label} is not valid JSON ({path}): {exc}")


def validate_top_level(payload: dict, label: str) -> None:
    for key, expected_type in REQUIRED_TOP_LEVEL.items():
        if key not in payload:
            fail(f"{label}: missing required field `{key}`")
        if not isinstance(payload[key], expected_type):
            fail(
                f"{label}: field `{key}` has wrong type "
                f"(expected {expected_type}, got {type(payload[key]).__name__})"
            )


def validate_predictions(payload: dict, label: str) -> None:
    for row_index, prediction in enumerate(payload["predictions"]):
        if not isinstance(prediction, dict):
            fail(f"{label}: predictions[{row_index}] must be an object")

        for key in ("index", "probabilities", "predicted_class"):
            if key not in prediction:
                fail(f"{label}: predictions[{row_index}] missing `{key}`")

        if not isinstance(prediction["index"], int):
            fail(f"{label}: predictions[{row_index}].index must be integer")
        if not isinstance(prediction["predicted_class"], int):
            fail(f"{label}: predictions[{row_index}].predicted_class must be integer")

        probabilities = prediction["probabilities"]
        if not isinstance(probabilities, list) or not probabilities:
            fail(
                f"{label}: predictions[{row_index}].probabilities must be a non-empty array"
            )

        prob_sum = 0.0
        for p_idx, value in enumerate(probabilities):
            if not isinstance(value, (int, float)):
                fail(
                    f"{label}: predictions[{row_index}].probabilities[{p_idx}] must be numeric"
                )
            prob_sum += float(value)

        if not math.isclose(prob_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            fail(
                f"{label}: predictions[{row_index}] probability sum is {prob_sum:.8f}, expected ~1.0"
            )


def index_predictions(payload: dict) -> dict[int, dict]:
    indexed = {}
    for pred in payload["predictions"]:
        idx = pred["index"]
        if idx in indexed:
            fail(f"duplicate prediction index found: {idx}")
        indexed[idx] = pred
    return indexed


def compare_probabilities(
    rust_payload: dict, other_payload: dict, tolerance: float, other_label: str
) -> int:
    rust_preds = index_predictions(rust_payload)
    other_preds = index_predictions(other_payload)

    if set(rust_preds) != set(other_preds):
        missing_in_other = sorted(set(rust_preds) - set(other_preds))
        missing_in_rust = sorted(set(other_preds) - set(rust_preds))
        fail(
            "prediction indices do not match. "
            f"Missing in {other_label}: {missing_in_other[:5]} "
            f"Missing in rust: {missing_in_rust[:5]}"
        )

    max_delta = -1.0
    worst_index = None
    worst_class = None

    for idx in sorted(rust_preds):
        rust_probs = rust_preds[idx]["probabilities"]
        other_probs = other_preds[idx]["probabilities"]

        if len(rust_probs) != len(other_probs):
            fail(
                f"prediction index {idx} has different class count "
                f"(rust={len(rust_probs)}, {other_label}={len(other_probs)})"
            )

        for c_idx, (rp, op) in enumerate(zip(rust_probs, other_probs)):
            delta = abs(float(rp) - float(op))
            if delta > max_delta:
                max_delta = delta
                worst_index = idx
                worst_class = c_idx

    print(
        "max_probability_delta="
        f"{max_delta:.10f} at prediction_index={worst_index}, class_index={worst_class}"
    )

    if max_delta >= tolerance:
        fail(
            "parity check failed: "
            f"max delta {max_delta:.10f} exceeds tolerance {tolerance:.10f}"
        )

    return 0


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    args = parse_args()

    rust = load_json(args.rust_output, "rust")
    other_path = args.zig_output or args.baseline_output
    if not other_path:
        fail("provide --zig-output or --baseline-output")

    other_label = "zig" if args.zig_output else "baseline"
    other = load_json(other_path, other_label)

    validate_top_level(rust, "rust")
    validate_predictions(rust, "rust")

    validate_top_level(other, other_label)
    validate_predictions(other, other_label)

    if rust["dataset"] != other["dataset"]:
        fail(
            f"dataset mismatch (rust={rust['dataset']}, {other_label}={other['dataset']})"
        )

    compare_probabilities(rust, other, args.tolerance, other_label)
    print(
        "PASS: parity check succeeded "
        f"for rust vs {other_label} (tolerance={args.tolerance:.10f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
