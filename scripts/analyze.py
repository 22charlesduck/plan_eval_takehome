#!/usr/bin/env python3
"""
Workstream 4: Crossover Analysis & Visualization

Reads data/metrics.json and produces:
  1. analysis/metrics_table.png   -- Summary table (plan vs baseline by tier)
  2. analysis/crossover_plot.png  -- Grouped bar chart of recall by tier
  3. analysis/scatter_complexity.png -- Scatter of recall vs ground-truth file count
  4. analysis/precision_plot.png  -- Grouped bar chart of precision by tier

Also prints the summary table to stdout.
"""
import os
os.chdir("/scr/clding/plan_eval")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH = "data/metrics.json"
OUTPUT_DIR = "analysis"
DPI = 200

# Colorblind-friendly palette (Okabe-Ito inspired)
COLOR_PLAN = "#0072B2"       # blue
COLOR_BASELINE = "#D55E00"   # vermilion
COLOR_DELTA = "#009E73"      # green
TIER_COLORS = {
    "localized": "#56B4E9",      # sky blue
    "cross-module": "#E69F00",   # orange
    "architectural": "#CC79A7",  # reddish-purple
}

# Ensure deterministic tier ordering
TIER_ORDER = ["localized", "cross-module", "architectural"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(INPUT_PATH, "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
df["complexity_tier"] = pd.Categorical(
    df["complexity_tier"], categories=TIER_ORDER, ordered=True
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Summary Table
# ---------------------------------------------------------------------------
agg = (
    df.groupby("complexity_tier", observed=False)
    .agg(
        n=("task_id", "count"),
        plan_recall_mean=("plan_file_recall", "mean"),
        plan_precision_mean=("plan_file_precision", "mean"),
        baseline_recall_mean=("baseline_b_file_recall", "mean"),
        baseline_precision_mean=("baseline_b_file_precision", "mean"),
    )
    .reset_index()
)
agg["recall_delta"] = agg["plan_recall_mean"] - agg["baseline_recall_mean"]

# Pretty-print
print("\n" + "=" * 90)
print("SUMMARY TABLE: Plan vs Keyword Baseline by Complexity Tier")
print("=" * 90)
header = (
    f"{'Tier':<16} {'n':>3}  {'Plan Rec':>9} {'Plan Prec':>10}  "
    f"{'Base Rec':>9} {'Base Prec':>10}  {'Recall Δ':>9}"
)
print(header)
print("-" * 90)
for _, row in agg.iterrows():
    print(
        f"{row['complexity_tier']:<16} {int(row['n']):>3}  "
        f"{row['plan_recall_mean']:>9.3f} {row['plan_precision_mean']:>10.3f}  "
        f"{row['baseline_recall_mean']:>9.3f} {row['baseline_precision_mean']:>10.3f}  "
        f"{row['recall_delta']:>+9.3f}"
    )

# Overall row
overall = {
    "n": len(df),
    "plan_recall_mean": df["plan_file_recall"].mean(),
    "plan_precision_mean": df["plan_file_precision"].mean(),
    "baseline_recall_mean": df["baseline_b_file_recall"].mean(),
    "baseline_precision_mean": df["baseline_b_file_precision"].mean(),
}
overall["recall_delta"] = overall["plan_recall_mean"] - overall["baseline_recall_mean"]
print("-" * 90)
print(
    f"{'OVERALL':<16} {overall['n']:>3}  "
    f"{overall['plan_recall_mean']:>9.3f} {overall['plan_precision_mean']:>10.3f}  "
    f"{overall['baseline_recall_mean']:>9.3f} {overall['baseline_precision_mean']:>10.3f}  "
    f"{overall['recall_delta']:>+9.3f}"
)
print("=" * 90 + "\n")

# --- Render table as image ---
fig_table, ax_table = plt.subplots(figsize=(12, 3.2))
ax_table.axis("off")
ax_table.set_title(
    "Plan vs. Keyword Baseline — Average Metrics by Complexity Tier",
    fontsize=14,
    fontweight="bold",
    pad=18,
)

col_labels = [
    "Tier", "n", "Plan Recall", "Plan Precision",
    "Baseline Recall", "Baseline Precision", "Recall Delta"
]
cell_data = []
for _, row in agg.iterrows():
    cell_data.append([
        row["complexity_tier"],
        f"{int(row['n'])}",
        f"{row['plan_recall_mean']:.3f}",
        f"{row['plan_precision_mean']:.3f}",
        f"{row['baseline_recall_mean']:.3f}",
        f"{row['baseline_precision_mean']:.3f}",
        f"{row['recall_delta']:+.3f}",
    ])
# Overall row
cell_data.append([
    "OVERALL",
    f"{overall['n']}",
    f"{overall['plan_recall_mean']:.3f}",
    f"{overall['plan_precision_mean']:.3f}",
    f"{overall['baseline_recall_mean']:.3f}",
    f"{overall['baseline_precision_mean']:.3f}",
    f"{overall['recall_delta']:+.3f}",
])

table_obj = ax_table.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table_obj.auto_set_font_size(False)
table_obj.set_fontsize(11)
table_obj.scale(1.0, 1.6)

# Style header row
for j in range(len(col_labels)):
    cell = table_obj[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

# Color the delta column based on sign
for i in range(1, len(cell_data) + 1):
    delta_val = float(cell_data[i - 1][-1])
    cell = table_obj[i, len(col_labels) - 1]
    if delta_val > 0:
        cell.set_facecolor("#d5f5e3")
    elif delta_val < 0:
        cell.set_facecolor("#fadbd8")
    else:
        cell.set_facecolor("#fef9e7")

    # Highlight overall row
    if i == len(cell_data):
        for j in range(len(col_labels)):
            table_obj[i, j].set_text_props(fontweight="bold")
            if j != len(col_labels) - 1:
                table_obj[i, j].set_facecolor("#eaecee")

fig_table.tight_layout()
fig_table.savefig(f"{OUTPUT_DIR}/metrics_table.png", dpi=DPI, bbox_inches="tight")
plt.close(fig_table)
print(f"[saved] {OUTPUT_DIR}/metrics_table.png")

# ---------------------------------------------------------------------------
# 2. Headline Bar Chart — Crossover Plot (Recall)
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

fig1, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(TIER_ORDER))
bar_width = 0.32

plan_vals = agg["plan_recall_mean"].values
base_vals = agg["baseline_recall_mean"].values

bars_plan = ax1.bar(
    x - bar_width / 2, plan_vals, bar_width,
    label="Plan Recall", color=COLOR_PLAN, edgecolor="white", linewidth=0.8, zorder=3,
)
bars_base = ax1.bar(
    x + bar_width / 2, base_vals, bar_width,
    label="Baseline (Keyword) Recall", color=COLOR_BASELINE, edgecolor="white",
    linewidth=0.8, zorder=3,
)

# Value labels on bars
for bar in bars_plan:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2, h + 0.015,
        f"{h:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=COLOR_PLAN,
    )
for bar in bars_base:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2, h + 0.015,
        f"{h:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=COLOR_BASELINE,
    )

# Delta annotations
for i, (pv, bv) in enumerate(zip(plan_vals, base_vals)):
    delta = pv - bv
    sign = "+" if delta >= 0 else ""
    y_pos = max(pv, bv) + 0.07
    ax1.annotate(
        f"{sign}{delta:.2f}",
        xy=(x[i], y_pos),
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
        color=COLOR_DELTA,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f8f5", edgecolor=COLOR_DELTA, alpha=0.85),
    )

ax1.set_xlabel("Complexity Tier", fontsize=13)
ax1.set_ylabel("File Recall", fontsize=13)
ax1.set_title(
    "Plan Recall vs. Keyword Baseline by Task Complexity",
    fontsize=15, fontweight="bold", pad=14,
)
ax1.set_xticks(x)
ax1.set_xticklabels(
    [t.replace("-", "-\n") for t in TIER_ORDER], fontsize=12
)
ax1.set_ylim(0, min(1.15, max(max(plan_vals), max(base_vals)) + 0.18))
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax1.legend(fontsize=11, loc="upper right", framealpha=0.9)
ax1.tick_params(axis="y", labelsize=11)

# Subtle gridlines
ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

fig1.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/crossover_plot.png", dpi=DPI, bbox_inches="tight")
plt.close(fig1)
print(f"[saved] {OUTPUT_DIR}/crossover_plot.png")

# ---------------------------------------------------------------------------
# 3. Scatter Plot — Recall vs Ground-Truth File Count
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))

for tier in TIER_ORDER:
    subset = df[df["complexity_tier"] == tier]
    ax2.scatter(
        subset["ground_truth_count"],
        subset["plan_file_recall"],
        label=tier,
        color=TIER_COLORS[tier],
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        alpha=0.9,
    )

# Fit a trend line across all data
gt_counts = df["ground_truth_count"].values.astype(float)
plan_recalls = df["plan_file_recall"].values.astype(float)

# Polynomial fit (degree 1)
z = np.polyfit(gt_counts, plan_recalls, 1)
p = np.poly1d(z)
x_line = np.linspace(gt_counts.min() - 0.5, gt_counts.max() + 0.5, 100)
ax2.plot(x_line, p(x_line), "--", color="gray", linewidth=1.5, alpha=0.7, zorder=2,
         label=f"Trend (slope={z[0]:.3f})")

ax2.set_xlabel("Number of Ground-Truth Files", fontsize=13)
ax2.set_ylabel("Plan File Recall", fontsize=13)
ax2.set_title(
    "Plan File Recall vs. Task Complexity (File Count)",
    fontsize=14, fontweight="bold", pad=12,
)
ax2.set_ylim(-0.05, 1.1)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax2.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax2.grid(linestyle="--", alpha=0.3, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(labelsize=11)

fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/scatter_complexity.png", dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print(f"[saved] {OUTPUT_DIR}/scatter_complexity.png")

# ---------------------------------------------------------------------------
# 4. Precision Comparison Plot
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))

plan_prec = agg["plan_precision_mean"].values
base_prec = agg["baseline_precision_mean"].values

bars_p_plan = ax3.bar(
    x - bar_width / 2, plan_prec, bar_width,
    label="Plan Precision", color=COLOR_PLAN, edgecolor="white", linewidth=0.8, zorder=3,
)
bars_p_base = ax3.bar(
    x + bar_width / 2, base_prec, bar_width,
    label="Baseline (Keyword) Precision", color=COLOR_BASELINE, edgecolor="white",
    linewidth=0.8, zorder=3,
)

# Value labels
for bar in bars_p_plan:
    h = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2, h + 0.015,
        f"{h:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=COLOR_PLAN,
    )
for bar in bars_p_base:
    h = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2, h + 0.015,
        f"{h:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=COLOR_BASELINE,
    )

# Delta annotations
for i, (pv, bv) in enumerate(zip(plan_prec, base_prec)):
    delta = pv - bv
    sign = "+" if delta >= 0 else ""
    y_pos = max(pv, bv) + 0.07
    ax3.annotate(
        f"{sign}{delta:.2f}",
        xy=(x[i], y_pos),
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
        color=COLOR_DELTA,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f8f5", edgecolor=COLOR_DELTA, alpha=0.85),
    )

ax3.set_xlabel("Complexity Tier", fontsize=13)
ax3.set_ylabel("File Precision", fontsize=13)
ax3.set_title(
    "Plan Precision vs. Keyword Baseline by Task Complexity",
    fontsize=15, fontweight="bold", pad=14,
)
ax3.set_xticks(x)
ax3.set_xticklabels(
    [t.replace("-", "-\n") for t in TIER_ORDER], fontsize=12
)
ax3.set_ylim(0, min(1.15, max(max(plan_prec), max(base_prec)) + 0.18))
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax3.legend(fontsize=11, loc="upper right", framealpha=0.9)
ax3.tick_params(axis="y", labelsize=11)
ax3.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax3.set_axisbelow(True)

fig3.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/precision_plot.png", dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print(f"[saved] {OUTPUT_DIR}/precision_plot.png")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\nAll analysis artifacts written to analysis/:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:30s} {size_kb:6.1f} KB")
print()
