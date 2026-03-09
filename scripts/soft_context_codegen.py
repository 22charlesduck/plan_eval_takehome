"""
Soft Context vs Hard Plan vs No Plan — Code Generation Experiment

Tests 3 conditions on 15 tasks (5 per tier):
  A) Hard plan: "Follow the plan provided"
  B) Soft plan: "A colleague sketched some initial thoughts — use if helpful"
  C) No plan: Issue + repo structure only

Key hypothesis: soft framing gets the best of both worlds — structural knowledge
from the plan without over-specification that causes the model to blindly modify
all planned files.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import anthropic
client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192
DELAY = 1.0

RESULTS_PATH = "data/soft_context_results.json"
CHECKPOINT_PATH = "data/soft_context_checkpoint.json"


# ── Task Selection ──────────────────────────────────────────────────────────

def select_tasks(metrics, tasks, plans, n_per_tier=5):
    """Select 5 tasks per tier with moderate plan recall (0.3-0.7)."""
    plan_ids = {p["task_id"] for p in plans if p.get("plan")}
    metric_map = {m["task_id"]: m for m in metrics}

    tier_targets = ["localized", "cross-module", "architectural"]
    selected_ids = []

    for tier in tier_targets:
        # Filter: has plan, correct tier, moderate recall
        pool = [
            m for m in metrics
            if m["complexity_tier"] == tier
            and m["task_id"] in plan_ids
            and 0.3 <= m["plan_file_recall"] <= 0.7
        ]

        # If not enough moderate-recall tasks, widen the range
        if len(pool) < n_per_tier:
            pool = [
                m for m in metrics
                if m["complexity_tier"] == tier
                and m["task_id"] in plan_ids
                and 0.15 <= m["plan_file_recall"] <= 0.85
            ]

        # If still not enough, take any with a plan
        if len(pool) < n_per_tier:
            pool = [
                m for m in metrics
                if m["complexity_tier"] == tier
                and m["task_id"] in plan_ids
            ]

        # Sort by distance from 0.5 recall (prefer moderate)
        pool.sort(key=lambda x: abs(x["plan_file_recall"] - 0.5))

        n = min(n_per_tier, len(pool))
        tier_ids = [pool[i]["task_id"] for i in range(n)]
        recalls = [pool[i]["plan_file_recall"] for i in range(n)]
        print(f"  {tier}: selected {n} from {len(pool)} "
              f"(recall range: {min(recalls):.3f} - {max(recalls):.3f})")
        selected_ids.extend(tier_ids)

    return selected_ids


# ── Prompt Builders ─────────────────────────────────────────────────────────

def truncate_repo_structure(repo_structure, max_chars=30000):
    if len(repo_structure) > max_chars:
        return repo_structure[:max_chars] + "\n... [truncated]"
    return repo_structure


def build_hard_plan_prompt(task, plan):
    """Condition A: Hard plan — treat plan as instructions."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))
    plan_json = json.dumps(plan["plan"], indent=2)

    return f"""Fix this issue. Follow the plan provided.

## Issue
{task["problem_statement"]}

## Plan (follow these steps)
{plan_json}

## Repository Structure
{repo_structure}

Generate ONLY a unified diff. Write real code, not pseudocode.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```"""


def build_soft_plan_prompt(task, plan):
    """Condition B: Soft plan — plan as optional context."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))
    plan_json = json.dumps(plan["plan"], indent=2)

    return f"""Fix this issue. A colleague sketched some initial thoughts below — use them if helpful, but trust your own judgment about which files to modify.

## Issue
{task["problem_statement"]}

## Colleague's Notes (for reference only — may be incomplete or partially wrong)
{plan_json}

## Repository Structure
{repo_structure}

Generate ONLY a unified diff. Write real code, not pseudocode.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```"""


def build_noplan_prompt(task):
    """Condition C: No plan — issue + repo structure only."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))

    return f"""Fix this issue.

## Issue
{task["problem_statement"]}

## Repository Structure
{repo_structure}

Generate ONLY a unified diff. Write real code, not pseudocode.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```"""


# ── API Call ────────────────────────────────────────────────────────────────

def call_api(prompt, task_id, condition, max_retries=2):
    """Call Claude API with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            return {
                "task_id": task_id,
                "condition": condition,
                "generated_code": response.content[0].text,
                "model": MODEL,
            }

        except Exception as e:
            print(f"      Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"      Retrying in {2 ** (attempt + 1)}s...")

    return {
        "task_id": task_id,
        "condition": condition,
        "generated_code": None,
        "error": "All retries failed",
    }


# ── Evaluation ──────────────────────────────────────────────────────────────

def extract_files_from_diff(diff_text):
    """Extract file paths from unified diff."""
    files = set()
    if not diff_text:
        return files
    for match in re.finditer(r'(?:---|\+\+\+) [ab]/(.+?)(?:\n|$)', diff_text):
        path = match.group(1).strip()
        if path and path != '/dev/null':
            files.add(path)
    return files


def normalize_path(path):
    return path.strip().lstrip('./')


def compute_file_overlap(gen_files, gt_files):
    gen_normalized = {normalize_path(f) for f in gen_files}
    gt_normalized = {normalize_path(f) for f in gt_files}

    overlap = gen_normalized & gt_normalized

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


def evaluate_generation(gen_result, task):
    """Evaluate a generated diff against ground truth."""
    task_id = task["task_id"]
    condition = gen_result.get("condition", "unknown")

    if gen_result.get("generated_code") is None:
        return {
            "task_id": task_id,
            "condition": condition,
            "complexity_tier": task.get("complexity_tier", "unknown"),
            "error": "No generated code",
        }

    generated = gen_result["generated_code"]
    gt_files = set(task["ground_truth_files_changed"])
    gen_files = extract_files_from_diff(generated)
    file_metrics = compute_file_overlap(gen_files, gt_files)

    return {
        "task_id": task_id,
        "condition": condition,
        "complexity_tier": task.get("complexity_tier", "unknown"),
        "gen_file_recall": file_metrics["file_recall"],
        "gen_file_precision": file_metrics["file_precision"],
        "gen_files_count": len(gen_files),
        "gt_files_count": len(gt_files),
        "gen_files": sorted(gen_files),
        "gt_files": sorted(gt_files),
    }


# ── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} generations already completed")
        return data
    return []


def save_checkpoint(generations):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(generations, f, indent=2)


# ── Plotting ────────────────────────────────────────────────────────────────

def create_comparison_chart(evaluations):
    """Grouped bar chart: hard plan vs soft plan vs no plan, by tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [e for e in evaluations if "error" not in e]
    if not valid:
        print("  No valid data for chart")
        return

    tiers = ["localized", "cross-module", "architectural"]
    conditions = ["hard_plan", "soft_plan", "no_plan"]
    condition_labels = ["Hard Plan", "Soft Plan", "No Plan"]
    condition_colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    # Compute means per tier per condition
    recall_data = {}
    precision_data = {}
    files_data = {}
    for tier in tiers:
        for cond in conditions:
            subset = [e for e in valid if e["complexity_tier"] == tier and e["condition"] == cond]
            key = (tier, cond)
            if subset:
                recall_data[key] = sum(e["gen_file_recall"] for e in subset) / len(subset)
                precision_data[key] = sum(e["gen_file_precision"] for e in subset) / len(subset)
                files_data[key] = sum(e["gen_files_count"] for e in subset) / len(subset)
            else:
                recall_data[key] = 0
                precision_data[key] = 0
                files_data[key] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(tiers))
    width = 0.25

    # Plot 1: File Recall
    ax = axes[0]
    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, condition_colors)):
        values = [recall_data[(t, cond)] for t in tiers]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("File Recall")
    ax.set_title("File Recall by Condition", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tiers, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: File Precision
    ax = axes[1]
    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, condition_colors)):
        values = [precision_data[(t, cond)] for t in tiers]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("File Precision")
    ax.set_title("File Precision by Condition", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tiers, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Number of files generated
    ax = axes[2]
    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, condition_colors)):
        values = [files_data[(t, cond)] for t in tiers]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg Files Generated")
    ax.set_title("Files Generated by Condition", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tiers, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Soft Context vs Hard Plan vs No Plan: Code Generation Quality\n(n=15 tasks, 5 per tier)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("analysis/soft_context_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved analysis/soft_context_comparison.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    with open("data/tasks_extended.json") as f:
        tasks = json.load(f)
    with open("data/plans_extended.json") as f:
        plans = json.load(f)
    with open("data/metrics_extended.json") as f:
        metrics = json.load(f)
    print(f"  {len(tasks)} tasks, {len(plans)} plans, {len(metrics)} metrics")

    task_map = {t["task_id"]: t for t in tasks}
    plan_map = {p["task_id"]: p for p in plans}

    # Select 15 tasks (5 per tier, moderate recall preferred)
    print("\nSelecting 15 tasks (5 per tier, preferring moderate plan recall)...")
    selected_ids = select_tasks(metrics, tasks, plans, n_per_tier=5)
    print(f"  Total selected: {len(selected_ids)}")

    # Load checkpoint
    print("\nChecking for checkpoint...")
    checkpoint = load_checkpoint()
    completed_keys = {(g["task_id"], g["condition"]) for g in checkpoint}
    generations = list(checkpoint)

    # Build work items: 3 conditions x 15 tasks = 45 API calls
    conditions_config = [
        ("hard_plan", "Hard Plan"),
        ("soft_plan", "Soft Plan"),
        ("no_plan", "No Plan"),
    ]

    work_items = []
    for task_id in selected_ids:
        for cond_key, cond_label in conditions_config:
            if (task_id, cond_key) not in completed_keys:
                work_items.append((task_id, cond_key, cond_label))

    total_calls = len(selected_ids) * 3
    print(f"\n{total_calls} total API calls needed, {len(work_items)} remaining")

    for i, (task_id, cond_key, cond_label) in enumerate(work_items, 1):
        task = task_map.get(task_id)
        plan = plan_map.get(task_id)
        if not task:
            print(f"  [{i}/{len(work_items)}] {task_id} [{cond_label}] -- SKIPPED (missing task)")
            continue

        tier = task.get("complexity_tier", "?")
        print(f"  [{i}/{len(work_items)}] {task_id} ({tier}) [{cond_label}]...")

        if cond_key == "hard_plan":
            if not plan:
                print(f"    SKIPPED (missing plan)")
                continue
            prompt = build_hard_plan_prompt(task, plan)
        elif cond_key == "soft_plan":
            if not plan:
                print(f"    SKIPPED (missing plan)")
                continue
            prompt = build_soft_plan_prompt(task, plan)
        else:  # no_plan
            prompt = build_noplan_prompt(task)

        result = call_api(prompt, task_id, cond_key)
        generations.append(result)

        # Checkpoint every 5 calls
        if i % 5 == 0:
            save_checkpoint(generations)
            print(f"    -- checkpoint saved ({len(generations)} total)")

        time.sleep(DELAY)

    save_checkpoint(generations)

    # ── Evaluate ────────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(generations)} generations...")
    evaluations = []
    for gen in generations:
        task = task_map.get(gen["task_id"])
        if not task:
            continue
        ev = evaluate_generation(gen, task)
        evaluations.append(ev)

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(evaluations, f, indent=2)
    print(f"  Saved {len(evaluations)} evaluations to {RESULTS_PATH}")

    # ── Summary Table ───────────────────────────────────────────────────────
    valid = [e for e in evaluations if "error" not in e]
    errors = [e for e in evaluations if "error" in e]

    print(f"\n{'=' * 120}")
    print("SOFT CONTEXT vs HARD PLAN vs NO PLAN — COMPARISON TABLE")
    print(f"{'=' * 120}")
    print(f"Tasks: {len(selected_ids)}, Conditions: 3, Total evaluations: {len(evaluations)}, Errors: {len(errors)}")

    # Build pivot: task_id -> {condition -> eval}
    pivot = defaultdict(dict)
    for e in valid:
        pivot[e["task_id"]][e["condition"]] = e

    # Print detailed comparison table
    header = (f"{'Task ID':<45} {'Tier':<14} "
              f"{'Hard R':>7} {'Soft R':>7} {'NoPln R':>7} "
              f"{'Hard P':>7} {'Soft P':>7} {'NoPln P':>7} "
              f"{'Hard F':>7} {'Soft F':>7} {'NoPln F':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    tier_order = {"localized": 0, "cross-module": 1, "architectural": 2}
    sorted_tasks = sorted(
        [(tid, pivot[tid]) for tid in selected_ids if tid in pivot],
        key=lambda x: (tier_order.get(list(x[1].values())[0]["complexity_tier"], 9), x[0])
    )

    for task_id, conds in sorted_tasks:
        tier = list(conds.values())[0]["complexity_tier"]
        hr = conds.get("hard_plan", {}).get("gen_file_recall", float("nan"))
        sr = conds.get("soft_plan", {}).get("gen_file_recall", float("nan"))
        nr = conds.get("no_plan", {}).get("gen_file_recall", float("nan"))
        hp = conds.get("hard_plan", {}).get("gen_file_precision", float("nan"))
        sp = conds.get("soft_plan", {}).get("gen_file_precision", float("nan"))
        np_ = conds.get("no_plan", {}).get("gen_file_precision", float("nan"))
        hf = conds.get("hard_plan", {}).get("gen_files_count", 0)
        sf = conds.get("soft_plan", {}).get("gen_files_count", 0)
        nf = conds.get("no_plan", {}).get("gen_files_count", 0)

        print(f"{task_id:<45} {tier:<14} "
              f"{hr:>7.3f} {sr:>7.3f} {nr:>7.3f} "
              f"{hp:>7.3f} {sp:>7.3f} {np_:>7.3f} "
              f"{hf:>7d} {sf:>7d} {nf:>7d}")

    # ── Aggregate by tier ───────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("AGGREGATE BY TIER")
    print(f"{'=' * 90}")

    agg_header = (f"{'Tier':<16} {'Cond':<12} {'N':>3} "
                  f"{'Recall':>8} {'Precision':>10} {'AvgFiles':>9}")
    print(agg_header)
    print("-" * len(agg_header))

    for tier in ["localized", "cross-module", "architectural"]:
        for cond in ["hard_plan", "soft_plan", "no_plan"]:
            subset = [e for e in valid if e["complexity_tier"] == tier and e["condition"] == cond]
            if not subset:
                continue
            avg_r = sum(e["gen_file_recall"] for e in subset) / len(subset)
            avg_p = sum(e["gen_file_precision"] for e in subset) / len(subset)
            avg_f = sum(e["gen_files_count"] for e in subset) / len(subset)
            print(f"{tier:<16} {cond:<12} {len(subset):>3} "
                  f"{avg_r:>8.3f} {avg_p:>10.3f} {avg_f:>9.1f}")

    # ── Overall by condition ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("OVERALL BY CONDITION")
    print(f"{'=' * 60}")

    for cond, label in [("hard_plan", "Hard Plan"), ("soft_plan", "Soft Plan"), ("no_plan", "No Plan")]:
        subset = [e for e in valid if e["condition"] == cond]
        if not subset:
            continue
        avg_r = sum(e["gen_file_recall"] for e in subset) / len(subset)
        avg_p = sum(e["gen_file_precision"] for e in subset) / len(subset)
        avg_f = sum(e["gen_files_count"] for e in subset) / len(subset)
        print(f"  {label:<12} (n={len(subset):>2}): "
              f"Recall={avg_r:.3f}  Precision={avg_p:.3f}  AvgFiles={avg_f:.1f}")

    # ── Win/Loss analysis ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("HEAD-TO-HEAD WINS (by file recall)")
    print(f"{'=' * 60}")

    soft_beats_hard = 0
    soft_beats_noplan = 0
    hard_beats_noplan = 0
    ties_soft_hard = 0
    ties_soft_noplan = 0
    n_compared = 0

    for task_id, conds in sorted_tasks:
        if "hard_plan" in conds and "soft_plan" in conds and "no_plan" in conds:
            n_compared += 1
            hr = conds["hard_plan"]["gen_file_recall"]
            sr = conds["soft_plan"]["gen_file_recall"]
            nr = conds["no_plan"]["gen_file_recall"]

            if sr > hr:
                soft_beats_hard += 1
            elif sr == hr:
                ties_soft_hard += 1

            if sr > nr:
                soft_beats_noplan += 1
            elif sr == nr:
                ties_soft_noplan += 1

            if hr > nr:
                hard_beats_noplan += 1

    print(f"  Tasks compared: {n_compared}")
    print(f"  Soft > Hard:   {soft_beats_hard}/{n_compared} "
          f"(ties: {ties_soft_hard})")
    print(f"  Soft > NoPlan: {soft_beats_noplan}/{n_compared} "
          f"(ties: {ties_soft_noplan})")
    print(f"  Hard > NoPlan: {hard_beats_noplan}/{n_compared}")

    # ── Key question ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("KEY QUESTION: Does soft context get the best of both worlds?")
    print(f"{'=' * 60}")

    if n_compared > 0:
        soft_all = [e for e in valid if e["condition"] == "soft_plan"]
        hard_all = [e for e in valid if e["condition"] == "hard_plan"]
        noplan_all = [e for e in valid if e["condition"] == "no_plan"]

        if soft_all and hard_all and noplan_all:
            sr_avg = sum(e["gen_file_recall"] for e in soft_all) / len(soft_all)
            hr_avg = sum(e["gen_file_recall"] for e in hard_all) / len(hard_all)
            nr_avg = sum(e["gen_file_recall"] for e in noplan_all) / len(noplan_all)
            sp_avg = sum(e["gen_file_precision"] for e in soft_all) / len(soft_all)
            hp_avg = sum(e["gen_file_precision"] for e in hard_all) / len(hard_all)
            np_avg = sum(e["gen_file_precision"] for e in noplan_all) / len(noplan_all)

            print(f"  Recall:    Hard={hr_avg:.3f}  Soft={sr_avg:.3f}  NoPlan={nr_avg:.3f}")
            print(f"  Precision: Hard={hp_avg:.3f}  Soft={sp_avg:.3f}  NoPlan={np_avg:.3f}")

            if sr_avg >= hr_avg and sr_avg >= nr_avg:
                print("  --> YES: Soft context achieves the highest recall overall.")
            elif sr_avg > hr_avg:
                print("  --> PARTIAL: Soft beats hard plan but not no-plan.")
            elif sr_avg > nr_avg:
                print("  --> PARTIAL: Soft beats no-plan but not hard plan.")
            else:
                print("  --> NO: Soft context does NOT outperform both alternatives.")

            if sp_avg >= hp_avg and sp_avg >= np_avg:
                print("  --> Soft context also has the best precision — avoids over-specification.")
            elif sp_avg > hp_avg:
                print("  --> Soft context has better precision than hard plan (less over-specification).")

    # ── Plot ────────────────────────────────────────────────────────────────
    os.makedirs("analysis", exist_ok=True)
    print("\nGenerating comparison chart...")
    create_comparison_chart(evaluations)

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  Removed checkpoint file")

    print("\nDone!")


if __name__ == "__main__":
    main()
