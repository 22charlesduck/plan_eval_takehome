"""
Code Generation from Plans — Extended 50-Task Evaluation

Selects 50 stratified tasks (~20 localized, ~15 cross-module, ~15 architectural)
from the extended dataset. For each task, generates implementation code from its
plan using the Anthropic API, then evaluates against ground truth.

Key question: does plan quality (file recall) predict code generation quality?
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import difflib
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import anthropic
client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192
DELAY = 1.0

RESULTS_PATH = "data/code_gen_extended_results.json"
CHECKPOINT_PATH = "data/code_gen_extended_checkpoint.json"


# ── Task Selection ──────────────────────────────────────────────────────────

def select_stratified_tasks(metrics, n_localized=20, n_cross=15, n_arch=15):
    """Select tasks with evenly-spaced recall scores within each tier."""
    tier_targets = {
        "localized": n_localized,
        "cross-module": n_cross,
        "architectural": n_arch,
    }

    selected_ids = []
    for tier, n_target in tier_targets.items():
        pool = [m for m in metrics if m["complexity_tier"] == tier]
        # Sort by plan_file_recall to get diverse spread
        pool.sort(key=lambda x: x["plan_file_recall"])
        n = min(n_target, len(pool))
        if n == 0:
            continue
        # Evenly spaced indices across the sorted pool
        if n >= len(pool):
            indices = list(range(len(pool)))
        else:
            step = (len(pool) - 1) / (n - 1) if n > 1 else 0
            indices = [round(i * step) for i in range(n)]
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
            indices = unique

        tier_ids = [pool[i]["task_id"] for i in indices]
        recalls = [pool[i]["plan_file_recall"] for i in indices]
        print(f"  {tier}: selected {len(tier_ids)} from {len(pool)} "
              f"(recall range: {min(recalls):.3f} - {max(recalls):.3f})")
        selected_ids.extend(tier_ids)

    return selected_ids


# ── Code Generation ─────────────────────────────────────────────────────────

def generate_code_from_plan(task, plan, max_retries=2):
    """Call Claude API to generate implementation code from a plan."""
    task_id = task["task_id"]

    repo_structure = task.get("repo_structure_summary", "")
    # Truncate to ~30K chars as specified
    if len(repo_structure) > 30000:
        repo_structure = repo_structure[:30000] + "\n... [truncated]"

    plan_json = json.dumps(plan["plan"], indent=2)

    prompt = f"""You are implementing a fix for a GitHub issue based on a pre-made plan.

## Issue
{task["problem_statement"]}

## Implementation Plan
{plan_json}

## Repository Structure (partial)
{repo_structure}

Generate the code changes as unified diffs for each file that needs modification.
Only modify files mentioned in the plan. Write real code, not pseudocode.

Format your response as a series of file modifications:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

Be specific and concrete. Write real, complete code changes. Focus on correctness."""

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            return {
                "task_id": task_id,
                "generated_code": raw_text,
                "model": MODEL,
            }

        except Exception as e:
            print(f"    Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"    Retrying in {2 ** (attempt + 1)}s...")

    return {
        "task_id": task_id,
        "generated_code": None,
        "error": "All retries failed",
    }


# ── Evaluation ──────────────────────────────────────────────────────────────

def extract_files_from_diff(diff_text):
    """Extract file paths from a unified diff."""
    files = set()
    if not diff_text:
        return files
    for match in re.finditer(r'(?:---|\+\+\+) [ab]/(.+?)(?:\n|$)', diff_text):
        path = match.group(1).strip()
        if path and path != '/dev/null':
            files.add(path)
    return files


def normalize_path(path):
    """Normalize a file path for matching."""
    return path.strip().lstrip('./')


def compute_file_overlap(gen_files, gt_files):
    """Compute file overlap with normalized path matching."""
    gen_normalized = {normalize_path(f) for f in gen_files}
    gt_normalized = {normalize_path(f) for f in gt_files}

    # Direct match
    overlap = gen_normalized & gt_normalized

    # Suffix matching for remaining
    unmatched_gen = gen_normalized - overlap
    unmatched_gt = gt_normalized - overlap
    for gf in list(unmatched_gen):
        for gtf in list(unmatched_gt):
            if gf.endswith(gtf) or gtf.endswith(gf):
                overlap.add(gf)
                unmatched_gt.discard(gtf)
                break

    return {
        "overlap": len(overlap),
        "gen_total": len(gen_normalized),
        "gt_total": len(gt_normalized),
        "file_recall": len(overlap) / len(gt_normalized) if gt_normalized else 0,
        "file_precision": len(overlap) / len(gen_normalized) if gen_normalized else 0,
    }


def compute_diff_similarity(generated_diff, ground_truth_patch):
    """Compute SequenceMatcher similarity on change lines."""
    def extract_change_lines(diff_text):
        lines = []
        for line in diff_text.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                lines.append(line[1:].strip())
        return lines

    gen_lines = extract_change_lines(generated_diff or "")
    gt_lines = extract_change_lines(ground_truth_patch or "")

    if not gen_lines and not gt_lines:
        return 1.0
    if not gen_lines or not gt_lines:
        return 0.0

    matcher = difflib.SequenceMatcher(None, gen_lines, gt_lines)
    return matcher.ratio()


def evaluate_generation(gen_result, task, metric):
    """Evaluate a generated code result against ground truth."""
    task_id = task["task_id"]
    if gen_result.get("generated_code") is None:
        return {
            "task_id": task_id,
            "complexity_tier": task.get("complexity_tier", "unknown"),
            "error": "No generated code",
        }

    generated = gen_result["generated_code"]
    gt_patch = task.get("ground_truth_patch", "")
    gt_files = set(task["ground_truth_files_changed"])

    gen_files = extract_files_from_diff(generated)
    file_metrics = compute_file_overlap(gen_files, gt_files)
    diff_sim = compute_diff_similarity(generated, gt_patch)

    return {
        "task_id": task_id,
        "complexity_tier": task.get("complexity_tier", "unknown"),
        "plan_file_recall": metric.get("plan_file_recall", 0),
        "gen_file_recall": file_metrics["file_recall"],
        "gen_file_precision": file_metrics["file_precision"],
        "diff_similarity": diff_sim,
        "gen_files_count": len(gen_files),
        "gt_files_count": len(gt_files),
        "gen_files": sorted(gen_files),
        "gt_files": sorted(gt_files),
    }


# ── Checkpoint / Resume ────────────────────────────────────────────────────

def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} tasks already completed")
        return data
    return []


def save_checkpoint(generations):
    """Save checkpoint after each batch."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(generations, f, indent=2)


# ── Plotting ────────────────────────────────────────────────────────────────

def create_bar_chart(evaluations):
    """Bar chart: file recall + diff similarity by tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [e for e in evaluations if "error" not in e]
    tier_data = defaultdict(list)
    for e in valid:
        tier_data[e["complexity_tier"]].append(e)

    tiers = ["localized", "cross-module", "architectural"]
    tiers = [t for t in tiers if t in tier_data]

    if not tiers:
        print("  No valid data for bar chart")
        return

    recalls = [sum(e["gen_file_recall"] for e in tier_data[t]) / len(tier_data[t]) for t in tiers]
    diff_sims = [sum(e["diff_similarity"] for e in tier_data[t]) / len(tier_data[t]) for t in tiers]
    counts = [len(tier_data[t]) for t in tiers]

    x = np.arange(len(tiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, recalls, width, label="File Recall", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, diff_sims, width, label="Diff Similarity", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Code Generation Quality by Complexity Tier (n=50)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(tiers, counts)])
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("analysis/code_gen_extended.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved analysis/code_gen_extended.png")


def create_correlation_scatter(evaluations):
    """Scatter plot: plan file recall (x) vs code gen file recall (y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [e for e in evaluations if "error" not in e]
    if not valid:
        print("  No valid data for scatter plot")
        return

    tier_colors = {
        "localized": "#4C72B0",
        "cross-module": "#DD8452",
        "architectural": "#55A868",
    }
    tier_markers = {
        "localized": "o",
        "cross-module": "s",
        "architectural": "D",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for tier in ["localized", "cross-module", "architectural"]:
        tier_evals = [e for e in valid if e["complexity_tier"] == tier]
        if not tier_evals:
            continue
        x = [e["plan_file_recall"] for e in tier_evals]
        y = [e["gen_file_recall"] for e in tier_evals]
        ax.scatter(x, y, c=tier_colors.get(tier, "gray"),
                   marker=tier_markers.get(tier, "o"),
                   s=80, alpha=0.7, label=f"{tier} (n={len(tier_evals)})",
                   edgecolors="white", linewidths=0.5)

    # Add regression line across all data
    all_x = [e["plan_file_recall"] for e in valid]
    all_y = [e["gen_file_recall"] for e in valid]

    if len(all_x) >= 2:
        coeffs = np.polyfit(all_x, all_y, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(0, max(all_x) + 0.05, 100)
        ax.plot(x_line, poly(x_line), "--", color="gray", alpha=0.7, linewidth=1.5)

        # Compute Pearson correlation
        mean_x = np.mean(all_x)
        mean_y = np.mean(all_y)
        cov = np.sum((np.array(all_x) - mean_x) * (np.array(all_y) - mean_y))
        std_x = np.sqrt(np.sum((np.array(all_x) - mean_x) ** 2))
        std_y = np.sqrt(np.sum((np.array(all_y) - mean_y) ** 2))
        r = cov / (std_x * std_y) if std_x * std_y > 0 else 0

        ax.text(0.05, 0.95, f"r = {r:.3f}\nslope = {coeffs[0]:.3f}",
                transform=ax.transAxes, fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Diagonal reference line (y=x)
    ax.plot([0, 1], [0, 1], ":", color="lightgray", alpha=0.5)

    ax.set_xlabel("Plan File Recall", fontsize=12)
    ax.set_ylabel("Code Generation File Recall", fontsize=12)
    ax.set_title("Plan Quality vs Code Generation Quality\nDoes a better plan lead to better code?",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis/code_gen_plan_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved analysis/code_gen_plan_correlation.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("Loading extended dataset...")
    with open("data/tasks_extended.json") as f:
        tasks = json.load(f)
    with open("data/plans_extended.json") as f:
        plans = json.load(f)
    with open("data/metrics_extended.json") as f:
        metrics = json.load(f)
    print(f"  {len(tasks)} tasks, {len(plans)} plans, {len(metrics)} metrics")

    # Build lookup maps
    task_map = {t["task_id"]: t for t in tasks}
    plan_map = {p["task_id"]: p for p in plans}
    metric_map = {m["task_id"]: m for m in metrics}

    # Select 50 stratified tasks
    print("\nSelecting 50 stratified tasks...")
    selected_ids = select_stratified_tasks(metrics, n_localized=20, n_cross=15, n_arch=15)
    print(f"  Total selected: {len(selected_ids)}")

    # Load checkpoint for resume support
    print("\nChecking for checkpoint...")
    checkpoint = load_checkpoint()
    completed_ids = {g["task_id"] for g in checkpoint}
    remaining_ids = [tid for tid in selected_ids if tid not in completed_ids]
    generations = list(checkpoint)

    if remaining_ids:
        print(f"\nGenerating code for {len(remaining_ids)} remaining tasks "
              f"({len(completed_ids)} already done)...")
    else:
        print(f"\nAll {len(selected_ids)} tasks already completed!")

    for i, task_id in enumerate(remaining_ids, 1):
        task = task_map.get(task_id)
        plan = plan_map.get(task_id)
        if not task or not plan:
            print(f"  [{i}/{len(remaining_ids)}] {task_id} -- SKIPPED (missing data)")
            continue

        tier = task.get("complexity_tier", "?")
        plan_recall = metric_map.get(task_id, {}).get("plan_file_recall", 0)
        print(f"  [{i}/{len(remaining_ids)}] {task_id} ({tier}, planR={plan_recall:.2f})...")

        result = generate_code_from_plan(task, plan)
        generations.append(result)

        # Save checkpoint every 10 tasks
        if i % 10 == 0:
            save_checkpoint(generations)
            print(f"    -- checkpoint saved ({len(generations)} total)")

        time.sleep(DELAY)

    # Final checkpoint save
    save_checkpoint(generations)

    # ── Evaluate all generations ────────────────────────────────────────────
    print(f"\nEvaluating {len(generations)} generations...")
    evaluations = []
    for gen in generations:
        tid = gen["task_id"]
        task = task_map.get(tid)
        metric = metric_map.get(tid, {})
        if not task:
            continue
        ev = evaluate_generation(gen, task, metric)
        evaluations.append(ev)

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(evaluations, f, indent=2)
    print(f"  Saved {len(evaluations)} evaluations to {RESULTS_PATH}")

    # ── Summary table ───────────────────────────────────────────────────────
    valid = [e for e in evaluations if "error" not in e]
    errors = [e for e in evaluations if "error" in e]

    print(f"\n{'=' * 80}")
    print("CODE GENERATION EXTENDED EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Tasks attempted: {len(evaluations)}")
    print(f"Successful generations: {len(valid)}")
    print(f"Failed generations: {len(errors)}")

    if not valid:
        print("No valid evaluations to report.")
        return

    # Overall averages
    avg_fr = sum(e["gen_file_recall"] for e in valid) / len(valid)
    avg_fp = sum(e["gen_file_precision"] for e in valid) / len(valid)
    avg_ds = sum(e["diff_similarity"] for e in valid) / len(valid)
    print(f"\nOverall (n={len(valid)}):")
    print(f"  Code Gen File Recall:    {avg_fr:.3f}")
    print(f"  Code Gen File Precision: {avg_fp:.3f}")
    print(f"  Diff Similarity:         {avg_ds:.3f}")

    # Per-tier averages
    tier_evals = defaultdict(list)
    for e in valid:
        tier_evals[e["complexity_tier"]].append(e)

    print(f"\n{'Tier':<16} {'N':>3} {'FileRecall':>11} {'FilePrecis':>11} {'DiffSim':>8} {'PlanRecall':>11}")
    print("-" * 65)
    for tier in ["localized", "cross-module", "architectural"]:
        te = tier_evals.get(tier, [])
        if not te:
            continue
        fr = sum(e["gen_file_recall"] for e in te) / len(te)
        fp = sum(e["gen_file_precision"] for e in te) / len(te)
        ds = sum(e["diff_similarity"] for e in te) / len(te)
        pr = sum(e["plan_file_recall"] for e in te) / len(te)
        print(f"{tier:<16} {len(te):>3} {fr:>11.3f} {fp:>11.3f} {ds:>8.3f} {pr:>11.3f}")

    # Detailed per-task table
    print(f"\n{'Task ID':<50} {'Tier':<14} {'PlanR':>6} {'CodeR':>6} {'DiffS':>6}")
    print("-" * 90)
    for e in sorted(valid, key=lambda x: (x["complexity_tier"], -x["gen_file_recall"])):
        print(f"{e['task_id']:<50} {e['complexity_tier']:<14} "
              f"{e['plan_file_recall']:>6.3f} {e['gen_file_recall']:>6.3f} "
              f"{e['diff_similarity']:>6.3f}")

    # ── Correlation analysis ────────────────────────────────────────────────
    import numpy as np
    all_plan_r = [e["plan_file_recall"] for e in valid]
    all_gen_r = [e["gen_file_recall"] for e in valid]

    if len(all_plan_r) >= 2:
        mean_x = np.mean(all_plan_r)
        mean_y = np.mean(all_gen_r)
        cov = np.sum((np.array(all_plan_r) - mean_x) * (np.array(all_gen_r) - mean_y))
        std_x = np.sqrt(np.sum((np.array(all_plan_r) - mean_x) ** 2))
        std_y = np.sqrt(np.sum((np.array(all_gen_r) - mean_y) ** 2))
        r = cov / (std_x * std_y) if std_x * std_y > 0 else 0
        print(f"\nCorrelation: plan_file_recall vs code_gen_file_recall")
        print(f"  Pearson r = {r:.3f}")
        if r > 0.3:
            print(f"  --> Positive correlation: better plans DO lead to better code generation")
        elif r > 0:
            print(f"  --> Weak positive correlation: plans have modest predictive value")
        elif r > -0.3:
            print(f"  --> Near-zero correlation: plan quality does not predict code quality")
        else:
            print(f"  --> Negative correlation: unexpected -- better plans lead to WORSE code?")

    # ── Plots ───────────────────────────────────────────────────────────────
    os.makedirs("analysis", exist_ok=True)
    print("\nGenerating plots...")
    create_bar_chart(evaluations)
    create_correlation_scatter(evaluations)

    # Cleanup checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("\n  Removed checkpoint file (run completed successfully)")

    print("\nDone!")


if __name__ == "__main__":
    main()
