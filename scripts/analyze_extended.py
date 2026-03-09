#!/usr/bin/env python3
"""
Extended Analysis (100+ tasks, also works on pilot 14-task data)

Reads a metrics JSON file and produces 5 plots + summary statistics.

Outputs:
  analysis/extended_crossover.png      -- Crossover bar chart (plan vs baseline by tier)
  analysis/extended_regression.png     -- Regression: plan recall vs complexity_score
  analysis/extended_recall_dist.png    -- Histogram of plan recall values
  analysis/extended_per_repo.png       -- Bar chart: average plan recall by repo
  analysis/extended_pr_scatter.png     -- Precision-recall scatter colored by tier
  analysis/extended_summary_stats.txt  -- Summary statistics (also printed to stdout)

Usage:
  python scripts/analyze_extended.py                            # defaults to data/metrics_extended.json
  python scripts/analyze_extended.py --metrics data/metrics.json --tasks data/tasks.json  # pilot data
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "analysis"
DPI = 200

COLOR_PLAN = "#0072B2"
COLOR_BASELINE = "#D55E00"
COLOR_DELTA = "#009E73"
TIER_COLORS = {
    "localized": "#56B4E9",
    "cross-module": "#E69F00",
    "architectural": "#CC79A7",
}
TIER_ORDER = ["localized", "cross-module", "architectural"]


def load_data(metrics_path: str, tasks_path: str = None):
    """Load metrics and optionally tasks (for repo info)."""
    with open(metrics_path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["complexity_tier"] = pd.Categorical(
        df["complexity_tier"], categories=TIER_ORDER, ordered=True
    )

    # Use ground_truth_count as continuous complexity score
    df["complexity_score"] = df["ground_truth_count"]

    # Compute recall delta
    if "baseline_b_file_recall" in df.columns:
        df["recall_delta"] = df["plan_file_recall"] - df["baseline_b_file_recall"]
    else:
        df["recall_delta"] = df["plan_file_recall"]

    # Try to add repo info from tasks file
    if tasks_path and os.path.exists(tasks_path):
        with open(tasks_path) as f:
            tasks = json.load(f)
        task_repo = {t["task_id"]: t.get("repo", "unknown") for t in tasks}
        df["repo"] = df["task_id"].map(task_repo).fillna("unknown")
    else:
        # Extract repo from task_id (format: repo__repo-NNNN)
        df["repo"] = df["task_id"].apply(lambda x: x.rsplit("-", 1)[0].replace("__", "/"))

    return df


# ---------------------------------------------------------------------------
# Plot 1: Crossover bar chart by tier
# ---------------------------------------------------------------------------
def plot_crossover(df: pd.DataFrame):
    """Grouped bar chart: plan recall vs baseline recall by tier."""
    has_baseline = "baseline_b_file_recall" in df.columns and df["baseline_b_file_recall"].notna().any()

    agg = (
        df.groupby("complexity_tier", observed=False)
        .agg(
            n=("task_id", "count"),
            plan_recall_mean=("plan_file_recall", "mean"),
            plan_recall_std=("plan_file_recall", "std"),
        )
        .reset_index()
    )

    if has_baseline:
        baseline_agg = (
            df.groupby("complexity_tier", observed=False)
            .agg(baseline_recall_mean=("baseline_b_file_recall", "mean"),
                 baseline_recall_std=("baseline_b_file_recall", "std"))
            .reset_index()
        )
        agg = agg.merge(baseline_agg, on="complexity_tier")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TIER_ORDER))
    bar_width = 0.32

    plan_vals = agg["plan_recall_mean"].values
    plan_errs = agg["plan_recall_std"].values / np.sqrt(agg["n"].values)

    bars_plan = ax.bar(
        x - bar_width / 2 if has_baseline else x,
        plan_vals, bar_width, yerr=plan_errs,
        label="Plan Recall", color=COLOR_PLAN, edgecolor="white",
        linewidth=0.8, zorder=3, capsize=4,
    )

    if has_baseline:
        base_vals = agg["baseline_recall_mean"].values
        base_errs = agg["baseline_recall_std"].values / np.sqrt(agg["n"].values)

        bars_base = ax.bar(
            x + bar_width / 2, base_vals, bar_width, yerr=base_errs,
            label="Baseline (Keyword) Recall", color=COLOR_BASELINE, edgecolor="white",
            linewidth=0.8, zorder=3, capsize=4,
        )

        # Delta annotations
        for i, (pv, bv) in enumerate(zip(plan_vals, base_vals)):
            delta = pv - bv
            sign = "+" if delta >= 0 else ""
            y_pos = max(pv, bv) + 0.07
            ax.annotate(
                f"{sign}{delta:.2f}",
                xy=(x[i], y_pos), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLOR_DELTA,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f8f5",
                          edgecolor=COLOR_DELTA, alpha=0.85),
            )

    # Value labels
    for bar in bars_plan:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=COLOR_PLAN)

    if has_baseline:
        for bar in bars_base:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=COLOR_BASELINE)

    # Add sample size labels
    for i, n in enumerate(agg["n"].values):
        ax.text(x[i], -0.06, f"n={int(n)}", ha="center", fontsize=10, color="gray")

    ax.set_xlabel("Complexity Tier", fontsize=13)
    ax.set_ylabel("File Recall", fontsize=13)
    ax.set_title("Plan Recall vs. Keyword Baseline by Task Complexity",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "-\n") for t in TIER_ORDER], fontsize=12)
    ax.set_ylim(-0.1, min(1.15, plan_vals.max() + 0.25))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/extended_crossover.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Plot 2: Regression (plan recall vs complexity score)
# ---------------------------------------------------------------------------
def plot_regression(df: pd.DataFrame):
    """Scatter + regression line: plan recall vs continuous complexity score."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for tier in TIER_ORDER:
        subset = df[df["complexity_tier"] == tier]
        ax.scatter(
            subset["complexity_score"], subset["plan_file_recall"],
            label=tier, color=TIER_COLORS[tier], s=80,
            edgecolors="black", linewidths=0.6, zorder=3, alpha=0.85,
        )

    x_vals = df["complexity_score"].values.astype(float)
    y_vals = df["plan_file_recall"].values.astype(float)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(max(0, x_vals.min() - 1), x_vals.max() + 1, 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="gray", linewidth=2,
            alpha=0.7, zorder=2,
            label=f"OLS (slope={slope:.4f}, R2={r_value**2:.3f})")

    # Correlation coefficients
    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)
    spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)

    textstr = (f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f})\n"
               f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f})")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", bbox=props)

    ax.set_xlabel("Complexity Score (# Ground Truth Files)", fontsize=13)
    ax.set_ylabel("Plan File Recall", fontsize=13)
    ax.set_title("Plan Recall vs. Task Complexity (Regression Analysis)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/extended_regression.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

    # Print correlation stats
    print(f"\n  Regression Analysis:")
    print(f"    OLS slope = {slope:.4f}, intercept = {intercept:.4f}, R^2 = {r_value**2:.3f}")
    print(f"    Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"    Spearman rho = {spearman_r:.3f}, p = {spearman_p:.4f}")

    return {
        "slope": slope, "intercept": intercept, "r_squared": r_value**2,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_rho": spearman_r, "spearman_p": spearman_p,
    }


# ---------------------------------------------------------------------------
# Plot 3: Histogram of plan recall
# ---------------------------------------------------------------------------
def plot_recall_histogram(df: pd.DataFrame):
    """Distribution of plan recall values."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.arange(0, 1.15, 0.1)
    ax.hist(df["plan_file_recall"], bins=bins, color=COLOR_PLAN, edgecolor="white",
            linewidth=0.8, alpha=0.85, zorder=3, label="All tasks")

    # Overlay per-tier histograms (stacked style)
    for tier in TIER_ORDER:
        subset = df[df["complexity_tier"] == tier]
        ax.hist(subset["plan_file_recall"], bins=bins, color=TIER_COLORS[tier],
                edgecolor="white", linewidth=0.5, alpha=0.5, zorder=2, label=tier)

    mean_recall = df["plan_file_recall"].mean()
    median_recall = df["plan_file_recall"].median()
    ax.axvline(mean_recall, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_recall:.3f}")
    ax.axvline(median_recall, color="orange", linestyle=":", linewidth=1.5,
               label=f"Median = {median_recall:.3f}")

    ax.set_xlabel("Plan File Recall", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Plan File Recall",
                 fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/extended_recall_dist.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Plot 4: Per-repo average plan recall
# ---------------------------------------------------------------------------
def plot_per_repo(df: pd.DataFrame):
    """Bar chart: average plan recall by repository."""
    repo_agg = (
        df.groupby("repo")
        .agg(
            mean_recall=("plan_file_recall", "mean"),
            std_recall=("plan_file_recall", "std"),
            n=("task_id", "count"),
        )
        .reset_index()
        .sort_values("mean_recall", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(repo_agg) * 0.5)))

    # Horizontal bars
    y_pos = np.arange(len(repo_agg))
    colors = [COLOR_PLAN if r >= repo_agg["mean_recall"].median() else COLOR_BASELINE
              for r in repo_agg["mean_recall"]]

    bars = ax.barh(y_pos, repo_agg["mean_recall"], color=colors,
                   edgecolor="white", linewidth=0.8, zorder=3, alpha=0.85)

    # Add value labels and sample size
    for i, (val, n) in enumerate(zip(repo_agg["mean_recall"], repo_agg["n"])):
        ax.text(val + 0.01, i, f"{val:.2f} (n={int(n)})", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(repo_agg["repo"], fontsize=10)
    ax.set_xlabel("Mean Plan File Recall", fontsize=13)
    ax.set_title("Average Plan Recall by Repository",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, min(1.15, repo_agg["mean_recall"].max() + 0.15))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/extended_per_repo.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Plot 5: Precision-Recall scatter
# ---------------------------------------------------------------------------
def plot_pr_scatter(df: pd.DataFrame):
    """Scatter of precision vs recall, colored by tier."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for tier in TIER_ORDER:
        subset = df[df["complexity_tier"] == tier]
        ax.scatter(
            subset["plan_file_recall"], subset["plan_file_precision"],
            label=f"{tier} (n={len(subset)})",
            color=TIER_COLORS[tier], s=80,
            edgecolors="black", linewidths=0.6, zorder=3, alpha=0.85,
        )

    # Add diagonal (recall = precision) for reference
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, linewidth=1, zorder=1)

    # Add F1 iso-curves
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_range = (f1_val * r_range) / (2 * r_range - f1_val)
        valid = (p_range > 0) & (p_range <= 1)
        ax.plot(r_range[valid], p_range[valid], "-", color="lightgray",
                alpha=0.5, linewidth=0.8, zorder=0)
        # Label
        label_idx = np.argmin(np.abs(r_range[valid] - 0.95))
        if label_idx < len(r_range[valid]):
            ax.text(r_range[valid][label_idx], p_range[valid][label_idx],
                    f"F1={f1_val:.1f}", fontsize=7, color="gray", alpha=0.7)

    ax.set_xlabel("Plan File Recall", fontsize=13)
    ax.set_ylabel("Plan File Precision", fontsize=13)
    ax.set_title("Precision-Recall Tradeoff by Complexity Tier",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/extended_pr_scatter.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Summary statistics + significance tests
# ---------------------------------------------------------------------------
def compute_summary_stats(df: pd.DataFrame) -> str:
    """Compute and format summary statistics. Returns the text."""
    lines = []
    lines.append("=" * 80)
    lines.append("EXTENDED ANALYSIS: SUMMARY STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Total tasks: {len(df)}")
    lines.append("")

    # Overall stats
    lines.append("--- Overall Plan Recall ---")
    lines.append(f"  Mean:   {df['plan_file_recall'].mean():.4f}")
    lines.append(f"  Median: {df['plan_file_recall'].median():.4f}")
    lines.append(f"  Std:    {df['plan_file_recall'].std():.4f}")
    lines.append(f"  Min:    {df['plan_file_recall'].min():.4f}")
    lines.append(f"  Max:    {df['plan_file_recall'].max():.4f}")
    lines.append("")

    has_baseline = "baseline_b_file_recall" in df.columns and df["baseline_b_file_recall"].notna().any()

    if has_baseline:
        lines.append("--- Overall Baseline Recall ---")
        lines.append(f"  Mean:   {df['baseline_b_file_recall'].mean():.4f}")
        lines.append(f"  Median: {df['baseline_b_file_recall'].median():.4f}")
        lines.append(f"  Std:    {df['baseline_b_file_recall'].std():.4f}")
        lines.append("")

        lines.append("--- Overall Recall Delta (Plan - Baseline) ---")
        delta = df["recall_delta"]
        lines.append(f"  Mean delta:   {delta.mean():.4f}")
        lines.append(f"  Median delta: {delta.median():.4f}")
        lines.append(f"  Std delta:    {delta.std():.4f}")
        lines.append("")

    # Per-tier stats
    lines.append("--- Per-Tier Plan Recall ---")
    lines.append(f"  {'Tier':<16} {'n':>4}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    lines.append("  " + "-" * 70)
    for tier in TIER_ORDER:
        subset = df[df["complexity_tier"] == tier]["plan_file_recall"]
        if len(subset) == 0:
            continue
        lines.append(
            f"  {tier:<16} {len(subset):>4}  "
            f"{subset.mean():>7.4f}  {subset.median():>7.4f}  "
            f"{subset.std():>7.4f}  {subset.min():>7.4f}  {subset.max():>7.4f}"
        )
    lines.append("")

    # 95% confidence intervals and significance tests for recall delta per tier
    if has_baseline:
        lines.append("--- 95% Confidence Intervals for Recall Delta (Plan - Baseline) per Tier ---")
        lines.append(f"  {'Tier':<16} {'n':>4}  {'Mean Delta':>11}  {'95% CI':>22}  {'t-stat':>8}  {'p-value':>10}  {'Sig?':>5}")
        lines.append("  " + "-" * 85)

        for tier in TIER_ORDER:
            subset = df[df["complexity_tier"] == tier]
            if len(subset) < 2:
                lines.append(f"  {tier:<16} {len(subset):>4}  insufficient data for CI")
                continue

            deltas = subset["recall_delta"].values

            mean_delta = deltas.mean()
            n = len(deltas)
            se = deltas.std(ddof=1) / np.sqrt(n)

            # t-test: is delta significantly different from 0?
            t_stat, p_value = stats.ttest_1samp(deltas, 0.0)

            # 95% CI
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_low = mean_delta - t_crit * se
            ci_high = mean_delta + t_crit * se

            sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))

            lines.append(
                f"  {tier:<16} {n:>4}  {mean_delta:>+11.4f}  "
                f"[{ci_low:>+9.4f}, {ci_high:>+9.4f}]  "
                f"{t_stat:>8.3f}  {p_value:>10.4f}  {sig:>5}"
            )

        lines.append("")

        # Bootstrap CI for overall delta
        lines.append("--- Bootstrap 95% CI for Overall Recall Delta ---")
        all_deltas = df["recall_delta"].values
        n_bootstrap = 10000
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(all_deltas, size=len(all_deltas), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_low_boot = np.percentile(boot_means, 2.5)
        ci_high_boot = np.percentile(boot_means, 97.5)
        lines.append(f"  Overall mean delta: {all_deltas.mean():+.4f}")
        lines.append(f"  Bootstrap 95% CI:   [{ci_low_boot:+.4f}, {ci_high_boot:+.4f}]")
        lines.append(f"  (based on {n_bootstrap} bootstrap samples)")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extended analysis and visualization")
    parser.add_argument("--metrics", default="data/metrics_extended.json",
                        help="Path to metrics JSON file")
    parser.add_argument("--tasks", default=None,
                        help="Path to tasks JSON file (for repo info)")
    args = parser.parse_args()

    # Auto-detect tasks file
    if args.tasks is None:
        if "extended" in args.metrics:
            args.tasks = "data/tasks_extended.json"
        else:
            args.tasks = "data/tasks.json"

    if not os.path.exists(args.metrics):
        print(f"ERROR: {args.metrics} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from {args.metrics}...")
    df = load_data(args.metrics, args.tasks)
    print(f"  {len(df)} tasks loaded\n")

    sns.set_style("whitegrid")

    # Generate all plots
    print("Generating plots...")
    plot_crossover(df)
    regression_stats = plot_regression(df)
    plot_recall_histogram(df)
    plot_per_repo(df)
    plot_pr_scatter(df)

    # Summary statistics
    print("\nComputing summary statistics...")
    stats_text = compute_summary_stats(df)
    print("\n" + stats_text)

    # Save to file
    stats_path = f"{OUTPUT_DIR}/extended_summary_stats.txt"
    with open(stats_path, "w") as f:
        f.write(stats_text + "\n")
    print(f"\n[saved] {stats_path}")

    # Final listing
    print(f"\nAll extended analysis artifacts in {OUTPUT_DIR}/:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith("extended_"):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname:40s} {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
