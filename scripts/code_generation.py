"""
Code Generation from Plans — End-to-End Evaluation

Selects a stratified subset of tasks, generates implementation code from plans,
and evaluates the generated code against ground-truth patches.

Evaluation metrics:
- File targeting accuracy: did the generated code modify the right files?
- Line-level similarity: how close is the generated diff to the ground truth?
- Semantic similarity: does the generated code address the same concerns?
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


def select_stratified_subset(tasks, plans, metrics, n_per_tier=5):
    """Select n tasks per tier, preferring tasks with diverse recall scores."""
    plan_map = {p["task_id"]: p for p in plans if p.get("plan")}
    metric_map = {m["task_id"]: m for m in metrics}

    tier_pools = defaultdict(list)
    for t in tasks:
        tid = t["task_id"]
        if tid in plan_map and tid in metric_map:
            tier_pools[t["complexity_tier"]].append({
                "task": t,
                "plan": plan_map[tid],
                "metrics": metric_map[tid],
            })

    selected = []
    for tier in ["localized", "cross-module", "architectural"]:
        pool = tier_pools.get(tier, [])
        # Sort by recall to get a diverse spread
        pool.sort(key=lambda x: x["metrics"]["plan_file_recall"])
        # Take evenly spaced items
        n = min(n_per_tier, len(pool))
        if n == 0:
            continue
        step = max(1, len(pool) // n)
        indices = list(range(0, len(pool), step))[:n]
        for i in indices:
            selected.append(pool[i])
        print(f"  {tier}: selected {len(indices)} tasks from {len(pool)} available")

    return selected


def generate_code_from_plan(task, plan, max_retries=2):
    """Generate implementation code from a plan."""
    task_id = task["task_id"]

    # Build the prompt with plan + repo structure
    repo_structure = task.get("repo_structure_summary", "")
    if len(repo_structure) > 40000:
        repo_structure = repo_structure[:40000] + "\n... [truncated]"

    plan_json = json.dumps(plan["plan"], indent=2)

    prompt = f"""You are implementing a fix for a GitHub issue based on a pre-made plan.

## Issue
{task["problem_statement"]}

## Implementation Plan
{plan_json}

## Repository Structure (partial)
{repo_structure}

## Your Task
Generate the actual code changes needed to implement this fix. For each file that needs modification, show the changes as a unified diff.

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

Be specific and concrete. Write real code, not pseudocode. If you need to create new files, show the full content.
Only modify files mentioned in the plan. Focus on correctness.
"""

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

    return {
        "task_id": task_id,
        "generated_code": None,
        "error": "All retries failed",
    }


def extract_files_from_diff(diff_text):
    """Extract file paths from a unified diff."""
    files = set()
    for match in re.finditer(r'(?:---|\+\+\+) [ab]/(.+?)(?:\n|$)', diff_text):
        path = match.group(1).strip()
        if path and path != '/dev/null':
            files.add(path)
    return files


def compute_diff_similarity(generated_diff, ground_truth_patch):
    """Compute similarity between generated diff and ground truth patch."""
    # Extract the actual changed lines (+ and - lines) from both
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

    # Use SequenceMatcher for line-level similarity
    matcher = difflib.SequenceMatcher(None, gen_lines, gt_lines)
    return matcher.ratio()


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


def evaluate_generation(gen_result, task):
    """Evaluate a generated code result against ground truth."""
    if gen_result.get("generated_code") is None:
        return {"task_id": task["task_id"], "error": "No generated code"}

    generated = gen_result["generated_code"]
    gt_patch = task.get("ground_truth_patch", "")
    gt_files = set(task["ground_truth_files_changed"])

    # Extract files from generated diff
    gen_files = extract_files_from_diff(generated)

    # File targeting
    file_metrics = compute_file_overlap(gen_files, gt_files)

    # Diff similarity
    diff_sim = compute_diff_similarity(generated, gt_patch)

    return {
        "task_id": task["task_id"],
        "complexity_tier": task.get("complexity_tier", "unknown"),
        "gen_files_count": len(gen_files),
        "gt_files_count": len(gt_files),
        "file_recall": file_metrics["file_recall"],
        "file_precision": file_metrics["file_precision"],
        "diff_similarity": diff_sim,
        "gen_files": sorted(gen_files),
        "gt_files": sorted(gt_files),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data — try pilot first, then extended
    if os.path.exists("data/metrics.json"):
        with open("data/tasks.json") as f:
            tasks = json.load(f)
        with open("data/plans.json") as f:
            plans = json.load(f)
        with open("data/metrics.json") as f:
            metrics = json.load(f)
        print(f"Loaded pilot data: {len(tasks)} tasks, {len(plans)} plans")
    else:
        print("ERROR: No metrics data found")
        return

    # Select stratified subset
    print("\nSelecting stratified subset for code generation...")
    subset = select_stratified_subset(tasks, plans, metrics, n_per_tier=5)
    print(f"Selected {len(subset)} tasks total")

    # Generate code for each
    print(f"\nGenerating code for {len(subset)} tasks...")
    generations = []
    for i, item in enumerate(subset, 1):
        task = item["task"]
        plan = item["plan"]
        print(f"  [{i}/{len(subset)}] {task['task_id']}...")
        result = generate_code_from_plan(task, plan)
        generations.append(result)
        time.sleep(DELAY)

    # Save generations
    os.makedirs("data", exist_ok=True)
    with open("data/code_generations.json", "w") as f:
        json.dump(generations, f, indent=2)
    print(f"\nSaved {len(generations)} generations to data/code_generations.json")

    # Evaluate
    print("\nEvaluating generated code...")
    evaluations = []
    task_map = {t["task_id"]: t for t in tasks}
    for gen in generations:
        if gen["task_id"] in task_map:
            ev = evaluate_generation(gen, task_map[gen["task_id"]])
            evaluations.append(ev)
            print(f"  {ev['task_id']:50s} | file_recall={ev.get('file_recall',0):.3f} "
                  f"| diff_sim={ev.get('diff_similarity',0):.3f}")

    # Save evaluations
    with open("data/code_generation_eval.json", "w") as f:
        json.dump(evaluations, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("CODE GENERATION EVALUATION SUMMARY")
    print("=" * 70)

    valid_evals = [e for e in evaluations if "error" not in e]
    if not valid_evals:
        print("No valid evaluations")
        return

    # Overall
    avg_file_recall = sum(e["file_recall"] for e in valid_evals) / len(valid_evals)
    avg_file_precision = sum(e["file_precision"] for e in valid_evals) / len(valid_evals)
    avg_diff_sim = sum(e["diff_similarity"] for e in valid_evals) / len(valid_evals)
    print(f"\nOverall (n={len(valid_evals)}):")
    print(f"  File Recall:    {avg_file_recall:.3f}")
    print(f"  File Precision: {avg_file_precision:.3f}")
    print(f"  Diff Similarity: {avg_diff_sim:.3f}")

    # Per tier
    tier_evals = defaultdict(list)
    for e in valid_evals:
        tier_evals[e["complexity_tier"]].append(e)

    print(f"\nPer Tier:")
    tier_data = {}
    for tier in ["localized", "cross-module", "architectural"]:
        te = tier_evals.get(tier, [])
        if not te:
            continue
        fr = sum(e["file_recall"] for e in te) / len(te)
        fp = sum(e["file_precision"] for e in te) / len(te)
        ds = sum(e["diff_similarity"] for e in te) / len(te)
        tier_data[tier] = {"recall": fr, "precision": fp, "diff_sim": ds, "n": len(te)}
        print(f"  {tier} (n={len(te)}): recall={fr:.3f} precision={fp:.3f} diff_sim={ds:.3f}")

    # Compare plan-only recall vs code-gen file recall
    plan_metric_map = {m["task_id"]: m for m in metrics}
    print(f"\nPlan-only vs Code-gen File Recall Comparison:")
    print(f"{'Task ID':50s} | {'Tier':14s} | PlanR | CodeR | Diff Sim")
    print("-" * 100)
    for e in valid_evals:
        pm = plan_metric_map.get(e["task_id"], {})
        plan_r = pm.get("plan_file_recall", 0)
        print(f"{e['task_id']:50s} | {e['complexity_tier']:14s} | "
              f"{plan_r:.3f} | {e['file_recall']:.3f} | {e['diff_similarity']:.3f}")

    # Plot: code gen metrics by tier
    os.makedirs("analysis", exist_ok=True)
    if tier_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tiers = [t for t in ["localized", "cross-module", "architectural"] if t in tier_data]

        for ax, metric_name, metric_key in zip(
            axes,
            ["File Recall", "File Precision", "Diff Similarity"],
            ["recall", "precision", "diff_sim"]
        ):
            values = [tier_data[t][metric_key] for t in tiers]
            colors = sns.color_palette("Set2", len(tiers))
            bars = ax.bar(range(len(tiers)), values, color=colors)
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([f"{t}\n(n={tier_data[t]['n']})" for t in tiers], fontsize=9)
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f"{val:.2f}", ha="center", fontsize=10)

        fig.suptitle("Code Generation Quality by Task Complexity", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("analysis/code_generation_eval.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved analysis/code_generation_eval.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
