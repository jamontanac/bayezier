#!/usr/bin/env python3
"""
Helper script to plot Bayesian k-NN classification boundaries and test points.
Uses matplotlib.
"""
from __future__ import annotations

import os
from pathlib import Path
import pnn_py

def plot_classification_results(
    x_train: list[list[float]],
    y_train: list[int],
    x_test: list[list[float]],
    y_test: list[int],
    dataset_name: str,
    sampler_config: dict,
    x_feature_idx: int = 0,
    y_feature_idx: int = 1,
) -> None:
    """
    Plots the decision boundaries and test data points colored by class.
    Saves the output as a PNG file inside a folder named 'plot_results_<dataset_name>'.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend to run without GUI/X11
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  [Warning] Matplotlib or NumPy is not installed. Skipping plotting for {dataset_name}.")
        return

    # Create target directory
    output_dir = Path("benchmarks") / f"plot_results_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays for indexing/slicing
    X_train = np.array(x_train)
    Y_train = np.array(y_train)
    X_test = np.array(x_test)
    Y_test = np.array(y_test)

    _, n_features = X_test.shape
    if n_features < 2:
        print(f"  [Warning] Dataset {dataset_name} has less than 2 features. Cannot plot 2D boundaries.")
        return

    # Choose plotting limits based on test data
    x_min, x_max = X_test[:, x_feature_idx].min(), X_test[:, x_feature_idx].max()
    y_min, y_max = X_test[:, y_feature_idx].min(), X_test[:, y_feature_idx].max()

    # Add 10% padding
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Create meshgrid
    grid_resolution = 80
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )

    # Prepare grid points for evaluation
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # If high dimensional (>2 features), fix remaining features to their mean training values
    if n_features > 2:
        mean_features = X_train.mean(axis=0)
        grid_points = np.tile(mean_features, (grid_points_2d.shape[0], 1))
        grid_points[:, x_feature_idx] = grid_points_2d[:, 0]
        grid_points[:, y_feature_idx] = grid_points_2d[:, 1]
    else:
        grid_points = grid_points_2d

    # Predict classes for grid points in a single vectorized batch
    print(f"    Generating decision boundary grid (resolution {grid_resolution}x{grid_resolution})...")
    try:
        payload = pnn_py.run_from_arrays(
            x_train=X_train.tolist(),
            y_train=Y_train.tolist(),
            x_test=grid_points.tolist(),
            y_test=None,
            dataset=dataset_name,
            implementation="rust-py-plot",
            k=sampler_config.get("k"),
            k_values=sampler_config.get("k_values"),
            k_range=sampler_config.get("k_range"),
            method=sampler_config.get("method", "hybrid"),
            n_samples=sampler_config.get("n_samples", 1000),
            burn_in=sampler_config.get("burn_in", 500),
            thinning=sampler_config.get("thinning", 1),
            beta_step=sampler_config.get("beta_step", 0.3),
            beta_sigma=sampler_config.get("beta_sigma", 5.0),
            seed=sampler_config.get("seed"),
        )
        grid_probs = [p["probabilities"] for p in payload["predictions"]]
        grid_preds = [p["predicted_class"] for p in payload["predictions"]]
    except Exception as e:
        print(f"  [Warning] Failed to generate decision boundary: {e}")
        return

    # Create figure
    plt.figure(figsize=(8, 6))

    n_classes = len(np.unique(Y_train))

    if n_classes == 2:
        # Binary classification: Plot probability of Class 1
        zz_probs = np.array([prob[1] for prob in grid_probs]).reshape(xx.shape)

        # Plot filled probability contours (Red to Blue transition)
        contour_filled = plt.contourf(xx, yy, zz_probs, levels=np.linspace(0, 1, 11), alpha=0.3, cmap="RdBu", zorder=1)
        
        # Add a colorbar for probabilities
        cbar = plt.colorbar(contour_filled)
        cbar.set_label("P(Class 1)")

        # Draw specific percentile contour lines
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        contours = plt.contour(xx, yy, zz_probs, levels=levels, colors="black", linewidths=0.6, alpha=0.8, zorder=2)
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    else:
        # Multi-class classification: Plot winning class regions & confidence contours
        zz_class = np.array(grid_preds).reshape(xx.shape)
        zz_conf = np.array([max(prob) for prob in grid_probs]).reshape(xx.shape)

        # Plot winning class regions
        plt.contourf(xx, yy, zz_class, alpha=0.25, cmap="Set3", zorder=1)

        # Plot confidence contours (e.g. 0.5, 0.6, 0.7, 0.8, 0.9)
        min_conf = 1.0 / n_classes
        levels = [l for l in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if l > min_conf]
        contours = plt.contour(xx, yy, zz_conf, levels=levels, colors="black", linewidths=0.6, alpha=0.8, zorder=2)
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Scatter plot test data points colored by true class
    scatter = plt.scatter(
        X_test[:, x_feature_idx],
        X_test[:, y_feature_idx],
        c=Y_test,
        cmap="Set1",
        edgecolors="black",
        linewidths=0.8,
        s=45,
        zorder=3
    )

    # Formats & labels
    title_suffix = " (other features set to mean)" if n_features > 2 else ""
    plt.title(f"Bayesian k-NN Classification Boundary - {dataset_name.capitalize()}{title_suffix}")
    plt.xlabel(f"Feature {x_feature_idx}")
    plt.ylabel(f"Feature {y_feature_idx}")
    
    # Legend
    legend = plt.legend(*scatter.legend_elements(), title="True Classes", loc="best")
    plt.gca().add_artist(legend)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle="--", alpha=0.4)

    # Save PNG file
    output_path = output_dir / "decision_boundary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved decision boundary plot to: {output_path}")
