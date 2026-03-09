"""
Plan vs No-Plan A/B Experiment

For a stratified subset of 10 tasks (4 localized, 3 cross-module, 3 architectural),
generate code TWO ways:
  Condition A: Issue + Plan + Repo Structure -> diff
  Condition B: Issue + Repo Structure only -> diff

Then compare which approach produces code that better matches the ground truth.
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
DELAY = 2.0  # seconds between API calls

# How many tasks per tier
TIER_COUNTS = {"localized": 4, "cross-module": 3, "architectural": 3}


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def select_tasks(tasks, plans, target_counts):
    """Select a stratified subset of tasks. Returns list of (task, plan) tuples."""
    plan_map = {p["task_id"]: p for p in plans if p.get("plan")}

    tier_pools = defaultdict(list)
    for t in tasks:
        tid = t["task_id"]
        if tid in plan_map:
            tier_pools[t["complexity_tier"]].append((t, plan_map[tid]))

    selected = []
    for tier in ["localized", "cross-module", "architectural"]:
        pool = tier_pools.get(tier, [])
        n = min(target_counts.get(tier, 0), len(pool))
        # Take the first n (they are already ordered from curation)
        selected.extend(pool[:n])
        print(f"  {tier}: selected {n} / {len(pool)} available")

    return selected


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def truncate_repo_structure(repo_structure, max_chars=30000):
    """Truncate repo structure to fit in context."""
    if len(repo_structure) > max_chars:
        return repo_structure[:max_chars] + "\n... [truncated]"
    return repo_structure


def build_plan_prompt(task, plan):
    """Condition A: Issue + Plan + Repo Structure."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))
    plan_json = json.dumps(plan["plan"], indent=2)

    return f"""You are implementing a fix for a GitHub issue. Follow the plan provided.

## Issue
{task["problem_statement"]}

## Implementation Plan
{plan_json}

## Repository Structure
{repo_structure}

Generate the code changes as unified diffs. Be specific — write real code.

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

Only modify files mentioned in the plan. Focus on correctness and completeness."""


def build_noplan_prompt(task):
    """Condition B: Issue + Repo Structure only (no plan)."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))

    return f"""You are implementing a fix for a GitHub issue. You must figure out which files to modify and what changes to make.

## Issue
{task["problem_statement"]}

## Repository Structure
{repo_structure}

Generate the code changes as unified diffs. Be specific — write real code.

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

Focus on correctness and completeness."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(prompt, task_id, label, max_retries=2):
    """Call Claude API with retries. Returns raw response text."""
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait = 2 ** (attempt + 1)
                print(f"      Retry {attempt} for {task_id} ({label}), waiting {wait}s...")
                time.sleep(wait)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            print(f"      Error on attempt {attempt + 1} for {task_id} ({label}): {e}")

    return None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def extract_files_from_diff(diff_text):
    """Extract file paths from unified diff output and other common formats."""
    if not diff_text:
        return set()
    files = set()

    # Standard unified diff: --- a/path and +++ b/path
    for match in re.finditer(r'(?:---|\+\+\+) [ab]/(.+?)(?:\n|$)', diff_text):
        path = match.group(1).strip()
        if path and path != '/dev/null':
            files.add(path)

    # <file_content path="..."> tags (some responses use this format)
    for match in re.finditer(r'<file_content\s+path="([^"]+)"', diff_text):
        path = match.group(1).strip()
        if path and not path.endswith('/'):
            files.add(path)

    # ```python or ```diff blocks with file path comments like # path/to/file.py
    # Also handle "File: path/to/file.py" or "**path/to/file.py**"
    for match in re.finditer(r'(?:^|\n)(?:File:\s*|Modified:\s*|\*\*)([a-zA-Z][\w/._-]+\.(?:py|js|ts|java|c|cpp|h|rs|go|rb))', diff_text):
        path = match.group(1).strip().rstrip('*')
        if path:
            files.add(path)

    return files


def normalize_path(path):
    """Strip leading ./ and whitespace."""
    return path.strip().lstrip('./')


def compute_file_overlap(gen_files, gt_files):
    """Compute file recall and precision with suffix matching."""
    gen_normalized = {normalize_path(f) for f in gen_files}
    gt_normalized = {normalize_path(f) for f in gt_files}

    # Direct match
    overlap = gen_normalized & gt_normalized

    # Suffix matching for remaining
    unmatched_gen = gen_normalized - overlap
    unmatched_gt = gt_normalized - overlap
    suffix_matches = 0
    for gf in list(unmatched_gen):
        for gtf in list(unmatched_gt):
            if gf.endswith(gtf) or gtf.endswith(gf):
                suffix_matches += 1
                unmatched_gt.discard(gtf)
                break

    total_overlap = len(overlap) + suffix_matches
    recall = total_overlap / len(gt_normalized) if gt_normalized else 0
    precision = total_overlap / len(gen_normalized) if gen_normalized else 0

    return recall, precision


def compute_diff_similarity(generated_diff, ground_truth_patch):
    """SequenceMatcher ratio between change lines in generated vs ground truth."""
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


def evaluate_one(task, generated_text, condition_label):
    """Evaluate a single generation against ground truth."""
    gt_files = set(task["ground_truth_files_changed"])
    gt_patch = task.get("ground_truth_patch", "")

    gen_files = extract_files_from_diff(generated_text)
    recall, precision = compute_file_overlap(gen_files, gt_files)
    diff_sim = compute_diff_similarity(generated_text, gt_patch)

    return {
        "task_id": task["task_id"],
        "complexity_tier": task["complexity_tier"],
        "condition": condition_label,
        "file_recall": round(recall, 4),
        "file_precision": round(precision, 4),
        "diff_similarity": round(diff_sim, 4),
        "gen_files": sorted(gen_files),
        "gt_files": sorted(gt_files),
        "gen_files_count": len(gen_files),
        "gt_files_count": len(gt_files),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_grouped_bar_chart(results, output_path):
    """Grouped bar chart: with-plan vs without-plan by tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate per tier + condition
    tier_cond = defaultdict(lambda: defaultdict(list))
    for r in results:
        tier_cond[r["complexity_tier"]][r["condition"]].append(r)

    tiers = ["localized", "cross-module", "architectural"]
    metrics = ["file_recall", "diff_similarity"]
    metric_labels = ["File Recall", "Diff Similarity"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(tiers))
    width = 0.35

    for ax, metric_key, metric_label in zip(axes, metrics, metric_labels):
        plan_vals = []
        noplan_vals = []
        for tier in tiers:
            plan_items = tier_cond[tier].get("with_plan", [])
            noplan_items = tier_cond[tier].get("no_plan", [])
            plan_avg = (sum(r[metric_key] for r in plan_items) / len(plan_items)) if plan_items else 0
            noplan_avg = (sum(r[metric_key] for r in noplan_items) / len(noplan_items)) if noplan_items else 0
            plan_vals.append(plan_avg)
            noplan_vals.append(noplan_avg)

        bars1 = ax.bar(x - width/2, plan_vals, width, label="With Plan", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width/2, noplan_vals, width, label="No Plan", color="#DD8452", alpha=0.85)

        # Value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={len(tier_cond[t].get('with_plan', []))})" for t in tiers], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Plan vs No-Plan Code Generation Quality by Complexity Tier",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    with open("data/tasks.json") as f:
        tasks = json.load(f)
    with open("data/plans.json") as f:
        plans = json.load(f)
    print(f"Loaded {len(tasks)} tasks, {len(plans)} plans")

    # Select stratified subset
    print("\nSelecting stratified subset...")
    subset = select_tasks(tasks, plans, TIER_COUNTS)
    print(f"Total selected: {len(subset)} tasks")
    print()

    # Run both conditions for each task
    all_results = []
    for i, (task, plan) in enumerate(subset, 1):
        tid = task["task_id"]
        tier = task["complexity_tier"]
        print(f"[{i}/{len(subset)}] {tid} ({tier})")

        # Condition A: With Plan
        print(f"    Generating WITH plan...")
        prompt_plan = build_plan_prompt(task, plan)
        response_plan = call_api(prompt_plan, tid, "with_plan")
        time.sleep(DELAY)

        # Condition B: Without Plan
        print(f"    Generating WITHOUT plan...")
        prompt_noplan = build_noplan_prompt(task)
        response_noplan = call_api(prompt_noplan, tid, "no_plan")
        time.sleep(DELAY)

        # Evaluate both
        if response_plan:
            eval_plan = evaluate_one(task, response_plan, "with_plan")
            eval_plan["raw_response"] = response_plan
            all_results.append(eval_plan)
            print(f"    WITH PLAN  -> recall={eval_plan['file_recall']:.3f}  "
                  f"precision={eval_plan['file_precision']:.3f}  "
                  f"diff_sim={eval_plan['diff_similarity']:.3f}")
        else:
            print(f"    WITH PLAN  -> FAILED")

        if response_noplan:
            eval_noplan = evaluate_one(task, response_noplan, "no_plan")
            eval_noplan["raw_response"] = response_noplan
            all_results.append(eval_noplan)
            print(f"    NO PLAN    -> recall={eval_noplan['file_recall']:.3f}  "
                  f"precision={eval_noplan['file_precision']:.3f}  "
                  f"diff_sim={eval_noplan['diff_similarity']:.3f}")
        else:
            print(f"    NO PLAN    -> FAILED")

        print()

    # Save results (without raw_response for the main file, keep a separate one)
    os.makedirs("data", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    # Full results with raw responses
    with open("data/plan_vs_noplan_results_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Clean results (no raw_response)
    clean_results = [{k: v for k, v in r.items() if k != "raw_response"} for r in all_results]
    with open("data/plan_vs_noplan_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Saved {len(clean_results)} results to data/plan_vs_noplan_results.json")

    # Print comparison table + summaries
    _print_comparison(clean_results)

    # Generate plot
    plot_results = [r for r in clean_results]  # use clean results for plot
    make_grouped_bar_chart(plot_results, "analysis/plan_vs_noplan.png")

    print("\nDone!")


def reeval():
    """Re-evaluate from saved raw responses (no API calls needed)."""
    import matplotlib
    matplotlib.use("Agg")

    with open("data/tasks.json") as f:
        tasks = json.load(f)
    task_map = {t["task_id"]: t for t in tasks}

    with open("data/plan_vs_noplan_results_full.json") as f:
        raw_results = json.load(f)
    print(f"Re-evaluating {len(raw_results)} saved responses...")

    all_results = []
    for r in raw_results:
        task = task_map[r["task_id"]]
        raw_text = r.get("raw_response", "")
        ev = evaluate_one(task, raw_text, r["condition"])
        ev["raw_response"] = raw_text
        all_results.append(ev)

    # Save updated results
    clean_results = [{k: v for k, v in r.items() if k != "raw_response"} for r in all_results]
    with open("data/plan_vs_noplan_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Saved {len(clean_results)} re-evaluated results")

    # Print comparison table + summaries (same logic as main)
    _print_comparison(clean_results)

    # Generate plot
    make_grouped_bar_chart(clean_results, "analysis/plan_vs_noplan.png")
    print("\nDone!")


def _print_comparison(clean_results):
    """Print the comparison table and summaries."""
    print("\n" + "=" * 120)
    print("PLAN vs NO-PLAN COMPARISON TABLE")
    print("=" * 120)
    header = (f"{'Task':<42s} | {'Tier':<14s} | {'Plan Recall':>11s} | {'NoPlan Recall':>13s} | "
              f"{'Plan DiffSim':>12s} | {'NoPlan DiffSim':>14s} | {'Winner':>8s}")
    print(header)
    print("-" * 120)

    task_results = defaultdict(dict)
    for r in clean_results:
        task_results[r["task_id"]][r["condition"]] = r

    plan_wins = 0
    noplan_wins = 0
    ties = 0
    tier_wins = defaultdict(lambda: {"plan": 0, "noplan": 0, "tie": 0})

    for tid in dict.fromkeys(r["task_id"] for r in clean_results):
        tr = task_results[tid]
        plan_r = tr.get("with_plan", {})
        noplan_r = tr.get("no_plan", {})

        p_recall = plan_r.get("file_recall", 0)
        n_recall = noplan_r.get("file_recall", 0)
        p_dsim = plan_r.get("diff_similarity", 0)
        n_dsim = noplan_r.get("diff_similarity", 0)
        tier = plan_r.get("complexity_tier", noplan_r.get("complexity_tier", "?"))

        p_score = (p_recall + p_dsim) / 2
        n_score = (n_recall + n_dsim) / 2

        if abs(p_score - n_score) < 0.01:
            winner = "Tie"
            ties += 1
            tier_wins[tier]["tie"] += 1
        elif p_score > n_score:
            winner = "Plan"
            plan_wins += 1
            tier_wins[tier]["plan"] += 1
        else:
            winner = "NoPlan"
            noplan_wins += 1
            tier_wins[tier]["noplan"] += 1

        print(f"{tid:<42s} | {tier:<14s} | {p_recall:>11.3f} | {n_recall:>13.3f} | "
              f"{p_dsim:>12.3f} | {n_dsim:>14.3f} | {winner:>8s}")

    print("-" * 120)

    plan_items = [r for r in clean_results if r["condition"] == "with_plan"]
    noplan_items = [r for r in clean_results if r["condition"] == "no_plan"]

    def avg(items, key):
        vals = [r[key] for r in items]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n{'OVERALL AVERAGES':^120s}")
    print(f"  With Plan:    File Recall = {avg(plan_items, 'file_recall'):.3f}   "
          f"File Precision = {avg(plan_items, 'file_precision'):.3f}   "
          f"Diff Similarity = {avg(plan_items, 'diff_similarity'):.3f}")
    print(f"  No Plan:      File Recall = {avg(noplan_items, 'file_recall'):.3f}   "
          f"File Precision = {avg(noplan_items, 'file_precision'):.3f}   "
          f"Diff Similarity = {avg(noplan_items, 'diff_similarity'):.3f}")

    print(f"\n  Wins: Plan={plan_wins}  NoPlan={noplan_wins}  Tie={ties}")

    print(f"\n{'PER-TIER BREAKDOWN':^120s}")
    for tier in ["localized", "cross-module", "architectural"]:
        t_plan = [r for r in plan_items if r["complexity_tier"] == tier]
        t_noplan = [r for r in noplan_items if r["complexity_tier"] == tier]
        tw = tier_wins[tier]

        if not t_plan and not t_noplan:
            continue

        print(f"\n  {tier.upper()} (n={len(t_plan)}):")
        print(f"    With Plan:  recall={avg(t_plan, 'file_recall'):.3f}  "
              f"precision={avg(t_plan, 'file_precision'):.3f}  "
              f"diff_sim={avg(t_plan, 'diff_similarity'):.3f}")
        print(f"    No Plan:    recall={avg(t_noplan, 'file_recall'):.3f}  "
              f"precision={avg(t_noplan, 'file_precision'):.3f}  "
              f"diff_sim={avg(t_noplan, 'diff_similarity'):.3f}")
        print(f"    Wins: Plan={tw['plan']}  NoPlan={tw['noplan']}  Tie={tw['tie']}")

    plan_recall_avg = avg(plan_items, "file_recall")
    noplan_recall_avg = avg(noplan_items, "file_recall")

    print(f"\n{'KEY FINDING':^120s}")
    if plan_recall_avg > noplan_recall_avg + 0.05:
        print("  Plans improve file recall overall.")
    elif noplan_recall_avg > plan_recall_avg + 0.05:
        print("  Plans HURT file recall overall — the model does better reasoning from scratch.")
    else:
        print("  Plans provide negligible overall benefit for file recall.")

    cm_plan = [r for r in plan_items if r["complexity_tier"] == "cross-module"]
    cm_noplan = [r for r in noplan_items if r["complexity_tier"] == "cross-module"]
    if cm_plan and cm_noplan:
        cm_plan_r = avg(cm_plan, "file_recall")
        cm_noplan_r = avg(cm_noplan, "file_recall")
        if cm_plan_r > cm_noplan_r + 0.05:
            print("  Cross-module tasks: Plans help — this mirrors the file-recall finding.")
        elif cm_noplan_r > cm_plan_r + 0.05:
            print("  Cross-module tasks: Plans hurt — model reasons better alone here too.")
        else:
            print("  Cross-module tasks: Plans and no-plan are similar.")


if __name__ == "__main__":
    import sys
    if "--reeval" in sys.argv:
        reeval()
    else:
        main()
