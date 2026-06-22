#!/usr/bin/env python3
"""
Plot MCMC diagnostics from a diagnostics JSON produced by the Rust pnn-core engine.

Panel layout (2 rows × 3 cols):
  (0,0) Beta full trace  — burn-in (gray) + post-burnin (blue) + red divider
  (0,1) Beta ACF         — bars + ±1.96/√n confidence band
  (0,2) k posterior bar  — normalised frequency per k value
  (1,0) k full trace     — burn-in (gray, joint-mh only) + post-burnin (orange)
  (1,1) k ACF            — same style as beta ACF
  (1,2) Summary          — ESS, acceptance rate, config snapshot
"""
from __future__ import annotations

import json
from pathlib import Path


def plot_diagnostics(
    diag_path: str | Path,
    dataset_name: str,
    method: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  [Warning] matplotlib/numpy not available — skipping diagnostics for {dataset_name}/{method}.")
        return

    diag_path = Path(diag_path)
    if not diag_path.exists():
        print(f"  [Warning] Diagnostics file not found: {diag_path}")
        return

    with diag_path.open() as f:
        d = json.load(f)

    cfg = d["config"]
    n_burnin = cfg["burn_in"]
    n_samples = cfg["n_samples"]

    beta_burnin = np.array(d["burn_in"]["beta_trace"])
    beta_trace  = np.array(d["beta"]["trace"])
    beta_acf    = np.array(d["beta"]["acf"])
    beta_ess    = d["beta"]["ess"]
    beta_mean   = d["beta"]["mean"]
    beta_std    = d["beta"]["std"]

    k_trace  = np.array(d["k"]["trace"])
    k_acf    = np.array(d["k"]["acf"])
    k_ess    = d["k"]["ess"]
    k_freqs  = {int(kv): cnt for kv, cnt in d["k"]["frequencies"].items()}
    k_burnin_raw = d["burn_in"].get("k_trace")
    k_burnin = np.array(k_burnin_raw) if k_burnin_raw is not None else None

    accept_rate = d["mh_acceptance"]["rate"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"{dataset_name.capitalize()} — {method}  MCMC Diagnostics",
        fontsize=13, fontweight="bold",
    )

    # ── (0,0)  Beta trace ─────────────────────────────────────────────────
    ax = axes[0, 0]
    bi_x = np.arange(len(beta_burnin))
    po_x = np.arange(len(beta_burnin), len(beta_burnin) + len(beta_trace))
    ax.plot(bi_x, beta_burnin, color="gray",     lw=0.6, alpha=0.8, label="burn-in")
    ax.plot(po_x, beta_trace,  color="steelblue", lw=0.6, alpha=0.8, label="post-burnin")
    ax.axvline(n_burnin, color="crimson", lw=1.2, ls="--", label=f"t={n_burnin}")
    ax.axhline(beta_mean, color="navy", lw=0.8, ls=":", alpha=0.6)
    ax.set_title("β  trace")
    ax.set_xlabel("iteration")
    ax.set_ylabel("β")
    ax.legend(fontsize=7, loc="upper right")

    # ── (0,1)  Beta ACF ───────────────────────────────────────────────────
    ax = axes[0, 1]
    lags = np.arange(len(beta_acf))
    ax.bar(lags, beta_acf, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ci = 1.96 / np.sqrt(max(len(beta_trace), 1))
    ax.axhline( ci, color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-ci, color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("β  ACF")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")

    # ── (0,2)  k posterior bar ────────────────────────────────────────────
    ax = axes[0, 2]
    k_vals   = sorted(k_freqs.keys())
    total    = sum(k_freqs.values())
    k_probs  = [k_freqs[kv] / total for kv in k_vals]
    ax.bar([str(kv) for kv in k_vals], k_probs,
           color="darkorange", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_title("k  posterior distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("posterior probability")

    # ── (1,0)  k trace ────────────────────────────────────────────────────
    ax = axes[1, 0]
    if k_burnin is not None:
        kb_x = np.arange(len(k_burnin))
        kp_x = np.arange(len(k_burnin), len(k_burnin) + len(k_trace))
        ax.plot(kb_x, k_burnin, color="gray",       lw=0.5, alpha=0.7, label="burn-in")
        ax.plot(kp_x, k_trace,  color="darkorange",  lw=0.5, alpha=0.8, label="post-burnin")
        ax.axvline(n_burnin, color="crimson", lw=1.2, ls="--")
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.plot(np.arange(len(k_trace)), k_trace, color="darkorange", lw=0.5, alpha=0.8)
        ax.text(0.5, 0.97, "Gibbs step — no burn-in k trace",
                ha="center", va="top", transform=ax.transAxes, fontsize=8, color="gray")
    ax.set_title("k  trace")
    ax.set_xlabel("iteration")
    ax.set_ylabel("k")

    # ── (1,1)  k ACF ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    lags_k = np.arange(len(k_acf))
    ax.bar(lags_k, k_acf, color="darkorange", alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ci_k = 1.96 / np.sqrt(max(len(k_trace), 1))
    ax.axhline( ci_k, color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-ci_k, color="crimson", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("k  ACF")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")

    # ── (1,2)  Summary panel ──────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"method    : {cfg['method']}",
        f"n_samples : {n_samples:,}",
        f"burn_in   : {n_burnin:,}",
        f"thinning  : {cfg['thinning']}",
        f"β step    : {cfg['beta_step']}",
        f"",
        f"β  mean   = {beta_mean:.4f}",
        f"β  std    = {beta_std:.4f}",
        f"β  ESS    = {beta_ess:.1f}",
        f"",
        f"k  ESS    = {k_ess:.1f}",
        f"",
        f"MH accept = {accept_rate:.3f}",
    ]
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="silver"),
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    method_tag = method.replace("-", "_")
    out_path = output_dir / f"diagnostics_{method_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved diagnostics plot to: {out_path}")
