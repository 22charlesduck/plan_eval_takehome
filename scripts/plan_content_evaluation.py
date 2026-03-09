import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import math
from dotenv import load_dotenv
load_dotenv()
import anthropic

client = anthropic.Anthropic()

# Load data
with open("data/tasks.json") as f:
    tasks = json.load(f)
with open("data/plans.json") as f:
    plans = json.load(f)
with open("data/metrics.json") as f:
    metrics = json.load(f)

# Index by task_id
tasks_by_id = {t["task_id"]: t for t in tasks}
plans_by_id = {p["task_id"]: p for p in plans}
metrics_by_id = {m["task_id"]: m for m in metrics}

task_ids = [t["task_id"] for t in tasks]

EVAL_PROMPT_TEMPLATE = """You are evaluating the quality of a coding plan's implementation guidance against a reference fix (merged PR).

Note: The reference patch shows one valid approach. The plan may propose a different but equally valid strategy — score based on usefulness, not exact match.

## Task Description
{problem_statement}

## Plan's Implementation Steps
{implementation_steps}

## Plan's Assumptions
{assumptions}

## Plan's Risks
{risks}

## Reference Implementation (Merged Diff)
{ground_truth_patch}

Score on 6 dimensions (1-3 each) with confidence (1-3) and explanation. Also provide step_by_step_analysis.

Respond in JSON:
{{
  "step_coverage": {{"score": N, "confidence": N, "explanation": "..."}},
  "step_granularity": {{"score": N, "confidence": N, "explanation": "..."}},
  "approach_alignment": {{"score": N, "confidence": N, "explanation": "..."}},
  "implementation_usefulness": {{"score": N, "confidence": N, "explanation": "..."}},
  "assumptions_relevance": {{"score": N, "confidence": N, "explanation": "..."}},
  "risks_relevance": {{"score": N, "confidence": N, "explanation": "..."}},
  "step_by_step_analysis": [{{"plan_step": "...", "reference_match": "...", "assessment": "useful/partially useful/not useful/misleading"}}]
}}"""


def format_steps(plan):
    steps = plan.get("implementation_steps", [])
    lines = []
    for s in steps:
        step_num = s.get("step", "?")
        desc = s.get("description", "")
        files = s.get("files_involved", [])
        lines.append(f"{step_num}. {desc}")
        if files:
            lines.append(f"   Files: {', '.join(files)}")
    return "\n".join(lines)


def format_list(items):
    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items)
    return str(items)


def parse_json_response(text):
    """Parse JSON from response, stripping markdown fences if present."""
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        # Remove first line and last line
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def evaluate_task(task_id, max_retries=2):
    task = tasks_by_id[task_id]
    plan_entry = plans_by_id[task_id]
    plan = plan_entry["plan"]

    problem_statement = task["problem_statement"]
    patch = task["ground_truth_patch"]
    # Truncate patch to 15000 chars
    if len(patch) > 15000:
        patch = patch[:15000] + "\n\n... [TRUNCATED - patch continues] ..."

    prompt = EVAL_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement,
        implementation_steps=format_steps(plan),
        assumptions=format_list(plan.get("assumptions", [])),
        risks=format_list(plan.get("risks", [])),
        ground_truth_patch=patch,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            result = parse_json_response(text)
            # Validate expected keys
            for key in ["step_coverage", "step_granularity", "approach_alignment",
                        "implementation_usefulness", "assumptions_relevance", "risks_relevance"]:
                assert key in result, f"Missing key: {key}"
                assert "score" in result[key], f"Missing score in {key}"
            return result
        except json.JSONDecodeError as e:
            print(f"  JSON parse error on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            else:
                print(f"  FAILED to parse JSON for {task_id} after {max_retries+1} attempts")
                return None
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            else:
                print(f"  FAILED for {task_id}: {e}")
                return None


def pearsonr_manual(x, y):
    """Compute Pearson correlation coefficient and p-value manually."""
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0 or denom_y == 0:
        return float('nan'), float('nan')
    r = numerator / (denom_x * denom_y)
    # t-test for significance
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        # Approximate p-value using t-distribution (two-tailed)
        # For small n, this is rough but acceptable
        try:
            from scipy.stats import t as t_dist
            p = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
        except ImportError:
            # Very rough approximation
            p = float('nan')
    return r, p


def pearsonr(x, y):
    try:
        from scipy.stats import pearsonr as _pearsonr
        return _pearsonr(x, y)
    except ImportError:
        return pearsonr_manual(x, y)


# ---- Main execution ----

# Check for existing results to allow resuming
scores_path = "data/plan_content_scores.json"
existing_scores = {}
if os.path.exists(scores_path):
    with open(scores_path) as f:
        existing_data = json.load(f)
    existing_scores = {item["task_id"]: item["scores"] for item in existing_data if item.get("scores")}
    print(f"Found {len(existing_scores)} existing scores, will skip those tasks.")

raw_results = []

for i, task_id in enumerate(task_ids):
    if task_id in existing_scores:
        print(f"[{i+1}/14] {task_id} — using cached result")
        raw_results.append({"task_id": task_id, "scores": existing_scores[task_id]})
        continue

    print(f"[{i+1}/14] Evaluating {task_id}...")
    result = evaluate_task(task_id)
    raw_results.append({"task_id": task_id, "scores": result})

    # Save incrementally
    with open(scores_path, "w") as f:
        json.dump(raw_results, f, indent=2)

    if i < len(task_ids) - 1:
        time.sleep(2)

# Save final raw results
with open(scores_path, "w") as f:
    json.dump(raw_results, f, indent=2)
print(f"\nSaved raw scores to {scores_path}")

# ---- Build metrics summary ----
DIMENSIONS = ["step_coverage", "step_granularity", "approach_alignment",
              "implementation_usefulness", "assumptions_relevance", "risks_relevance"]

content_metrics = []
for item in raw_results:
    task_id = item["task_id"]
    scores = item["scores"]
    if scores is None:
        continue

    m = metrics_by_id[task_id]
    file_recall = m["plan_file_recall"]
    tier = m["complexity_tier"]

    dim_scores = {d: scores[d]["score"] for d in DIMENSIONS}
    content_composite = sum(dim_scores[d] for d in ["step_coverage", "step_granularity",
                                                      "approach_alignment", "implementation_usefulness"]) / 4.0

    content_metrics.append({
        "task_id": task_id,
        "complexity_tier": tier,
        "file_recall": file_recall,
        "content_composite": round(content_composite, 4),
        **{d: dim_scores[d] for d in DIMENSIONS},
        "confidence_avg": round(sum(scores[d]["confidence"] for d in DIMENSIONS) / len(DIMENSIONS), 2),
    })

with open("data/plan_content_metrics.json", "w") as f:
    json.dump(content_metrics, f, indent=2)
print(f"Saved content metrics to data/plan_content_metrics.json")

# ---- Correlations ----
recalls = [cm["file_recall"] for cm in content_metrics]
composites = [cm["content_composite"] for cm in content_metrics]

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)

r, p = pearsonr(recalls, composites)
print(f"  file_recall vs content_composite: r={r:.4f}, p={p:.4f}")

for dim in DIMENSIONS:
    vals = [cm[dim] for cm in content_metrics]
    r_d, p_d = pearsonr(recalls, vals)
    print(f"  file_recall vs {dim}: r={r_d:.4f}, p={p_d:.4f}")

# ---- Per-task table ----
print("\n" + "=" * 80)
print("PER-TASK TABLE")
print("=" * 80)
header = f"{'Task ID':45s} {'Tier':15s} {'Recall':>7s} {'Comp':>5s} {'StCov':>5s} {'StGr':>5s} {'Appr':>5s} {'Impl':>5s} {'Asmp':>5s} {'Risk':>5s} {'Conf':>5s}"
print(header)
print("-" * len(header))
for cm in content_metrics:
    print(f"{cm['task_id']:45s} {cm['complexity_tier']:15s} {cm['file_recall']:7.4f} {cm['content_composite']:5.2f} "
          f"{cm['step_coverage']:5d} {cm['step_granularity']:5d} {cm['approach_alignment']:5d} "
          f"{cm['implementation_usefulness']:5d} {cm['assumptions_relevance']:5d} {cm['risks_relevance']:5d} "
          f"{cm['confidence_avg']:5.2f}")

# ---- Tier averages ----
print("\n" + "=" * 80)
print("TIER AVERAGES")
print("=" * 80)
tiers = ["localized", "cross-module", "architectural"]
for tier in tiers:
    tier_items = [cm for cm in content_metrics if cm["complexity_tier"] == tier]
    if not tier_items:
        continue
    n = len(tier_items)
    avg_recall = sum(cm["file_recall"] for cm in tier_items) / n
    avg_composite = sum(cm["content_composite"] for cm in tier_items) / n
    avg_conf = sum(cm["confidence_avg"] for cm in tier_items) / n
    dim_avgs = {d: sum(cm[d] for cm in tier_items) / n for d in DIMENSIONS}
    print(f"  {tier:15s} (n={n}): recall={avg_recall:.4f} composite={avg_composite:.2f} conf={avg_conf:.2f}")
    for d in DIMENSIONS:
        print(f"    {d}: {dim_avgs[d]:.2f}")

# ---- Plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

# Plot 1: Bar chart of avg content composite by tier
fig, ax = plt.subplots(figsize=(8, 5))
tier_labels = []
tier_means = []
tier_stds = []
for tier in tiers:
    tier_items = [cm for cm in content_metrics if cm["complexity_tier"] == tier]
    if tier_items:
        tier_labels.append(tier)
        vals = [cm["content_composite"] for cm in tier_items]
        tier_means.append(np.mean(vals))
        tier_stds.append(np.std(vals))

colors = ["#4CAF50", "#2196F3", "#FF5722"]
bars = ax.bar(tier_labels, tier_means, yerr=tier_stds, capsize=5, color=colors[:len(tier_labels)], edgecolor="black", alpha=0.85)
ax.set_ylabel("Content Composite Score (1-3)")
ax.set_xlabel("Complexity Tier")
ax.set_title("Plan Content Quality by Complexity Tier")
ax.set_ylim(0.5, 3.5)
for bar, mean in zip(bars, tier_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{mean:.2f}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("analysis/plan_content_by_tier.png", dpi=150)
print("\nSaved analysis/plan_content_by_tier.png")
plt.close()

# Plot 2: Scatter of file_recall vs content_composite
fig, ax = plt.subplots(figsize=(9, 6))
tier_colors = {"localized": "#4CAF50", "cross-module": "#2196F3", "architectural": "#FF5722"}
for cm in content_metrics:
    color = tier_colors.get(cm["complexity_tier"], "gray")
    ax.scatter(cm["file_recall"], cm["content_composite"], c=color, s=80, edgecolors="black", zorder=3)
    short_name = cm["task_id"].split("__")[1][:20]
    ax.annotate(short_name, (cm["file_recall"], cm["content_composite"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

# Legend
for tier, color in tier_colors.items():
    ax.scatter([], [], c=color, s=80, edgecolors="black", label=tier)
ax.legend(title="Tier")

# Trendline
z = np.polyfit(recalls, composites, 1)
p_line = np.poly1d(z)
x_range = np.linspace(min(recalls) - 0.05, max(recalls) + 0.05, 100)
ax.plot(x_range, p_line(x_range), "--", color="gray", alpha=0.6)

r_val, p_val = pearsonr(recalls, composites)
ax.set_xlabel("File Recall")
ax.set_ylabel("Content Composite Score")
ax.set_title(f"File Recall vs Plan Content Quality (r={r_val:.3f}, p={p_val:.3f})")
ax.set_ylim(0.5, 3.5)
plt.tight_layout()
plt.savefig("analysis/plan_content_vs_recall.png", dpi=150)
print("Saved analysis/plan_content_vs_recall.png")
plt.close()

print("\nDone!")
