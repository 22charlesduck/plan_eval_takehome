"""
Experiment 12: Plan vs No-Plan Code Generation at Scale (n=40)

Extends Experiment 8 (n=10) to validate whether the "plans hurt code gen" finding holds.
Selects 40 tasks (15 localized, 15 cross-module, 10 architectural) with diversity in
plan_file_recall, then generates code with and without plans. Compares file recall,
precision, and diff similarity.
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
DELAY = 1.0  # seconds between API calls

TIER_COUNTS = {"localized": 15, "cross-module": 15, "architectural": 10}

OUTPUT_FILE = "data/plan_vs_noplan_extended.json"
PLOT_WINRATE = "analysis/plan_vs_noplan_extended.png"
PLOT_SCATTER = "analysis/plan_accuracy_vs_codegen_delta.png"
WRITEUP_FILE = "analysis/plan_vs_noplan_extended.md"


# ---------------------------------------------------------------------------
# Task selection — diverse plan_file_recall within each tier
# ---------------------------------------------------------------------------

def select_tasks(tasks, plans, metrics, target_counts):
    """Select stratified tasks with diversity in plan_file_recall."""
    plan_map = {p["task_id"]: p for p in plans if p.get("plan")}
    metric_map = {m["task_id"]: m for m in metrics}

    # Only tasks that have plan + metrics + repo_structure
    eligible = []
    for t in tasks:
        tid = t["task_id"]
        if tid in plan_map and tid in metric_map and t.get("repo_structure_summary"):
            eligible.append((t, plan_map[tid], metric_map[tid]))

    # Group by tier
    tier_pools = defaultdict(list)
    for t, p, m in eligible:
        tier_pools[t["complexity_tier"]].append((t, p, m))

    selected = []
    for tier in ["localized", "cross-module", "architectural"]:
        pool = tier_pools.get(tier, [])
        n = min(target_counts.get(tier, 0), len(pool))

        # Sort by plan_file_recall and pick evenly spaced
        pool.sort(key=lambda x: x[2]["plan_file_recall"])
        if len(pool) <= n:
            chosen = pool
        else:
            # Evenly spaced indices
            indices = [int(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
            # Deduplicate indices
            indices = sorted(set(indices))
            # If we lost some due to dedup, fill from nearest unused
            while len(indices) < n:
                for candidate in range(len(pool)):
                    if candidate not in indices:
                        indices.append(candidate)
                        indices = sorted(set(indices))
                        if len(indices) >= n:
                            break
            chosen = [pool[i] for i in indices[:n]]

        recalls = [c[2]["plan_file_recall"] for c in chosen]
        print(f"  {tier}: selected {len(chosen)} / {len(pool)} available "
              f"(recall range: {min(recalls):.2f} - {max(recalls):.2f})")
        selected.extend(chosen)

    return selected


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def truncate_repo_structure(repo_structure, max_chars=30000):
    if len(repo_structure) > max_chars:
        return repo_structure[:max_chars] + "\n... [truncated]"
    return repo_structure


def build_plan_prompt(task, plan):
    """Condition A: with plan."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))
    plan_json = json.dumps(plan["plan"], indent=2)

    return f"""You are an autonomous coding agent. Fix this GitHub issue using the provided plan.

## Issue
{task["problem_statement"]}

## Plan
{plan_json}

## Repository Structure
{repo_structure}

Generate ONLY a unified diff. No explanation."""


def build_noplan_prompt(task):
    """Condition B: without plan."""
    repo_structure = truncate_repo_structure(task.get("repo_structure_summary", ""))

    return f"""You are an autonomous coding agent. Fix this GitHub issue.

## Issue
{task["problem_statement"]}

## Repository Structure
{repo_structure}

Generate ONLY a unified diff. No explanation."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(prompt, task_id, label, max_retries=2):
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
    if not diff_text:
        return set()
    files = set()

    # Standard unified diff: --- a/path and +++ b/path
    for match in re.finditer(r'(?:---|\+\+\+) [ab]/(.+?)(?:\n|$)', diff_text):
        path = match.group(1).strip()
        if path and path != '/dev/null':
            files.add(path)

    # <file_content path="..."> tags
    for match in re.finditer(r'<file_content\s+path="([^"]+)"', diff_text):
        path = match.group(1).strip()
        if path and not path.endswith('/'):
            files.add(path)

    # "File: path/to/file.py" or "**path/to/file.py**"
    for match in re.finditer(r'(?:^|\n)(?:File:\s*|Modified:\s*|\*\*)([a-zA-Z][\w/._-]+\.(?:py|js|ts|java|c|cpp|h|rs|go|rb))', diff_text):
        path = match.group(1).strip().rstrip('*')
        if path:
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
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results():
    """Load existing results for resume support."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                data = json.load(f)
            # Build set of (task_id, condition) pairs already done
            done = set()
            for r in data:
                done.add((r["task_id"], r["condition"]))
            print(f"Loaded {len(data)} existing results ({len(done)} task-condition pairs)")
            return data, done
        except (json.JSONDecodeError, KeyError):
            pass
    return [], set()


def save_results(results):
    """Save results to disk."""
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_winrate_chart(results, metric_map, output_path):
    """Bar chart: win rate by tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group results by task
    task_results = defaultdict(dict)
    for r in results:
        task_results[r["task_id"]][r["condition"]] = r

    # Count wins per tier
    tiers = ["localized", "cross-module", "architectural"]
    tier_wins = {t: {"plan": 0, "noplan": 0, "tie": 0, "total": 0} for t in tiers}

    for tid, conds in task_results.items():
        plan_r = conds.get("with_plan", {})
        noplan_r = conds.get("no_plan", {})
        if not plan_r or not noplan_r:
            continue

        tier = plan_r.get("complexity_tier", noplan_r.get("complexity_tier", ""))
        if tier not in tier_wins:
            continue

        tier_wins[tier]["total"] += 1

        p_recall = plan_r.get("file_recall", 0)
        n_recall = noplan_r.get("file_recall", 0)

        if abs(p_recall - n_recall) < 0.001:
            tier_wins[tier]["tie"] += 1
        elif p_recall > n_recall:
            tier_wins[tier]["plan"] += 1
        else:
            tier_wins[tier]["noplan"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Win rate by tier (stacked bar)
    ax = axes[0]
    x = np.arange(len(tiers))
    width = 0.6

    plan_rates = []
    noplan_rates = []
    tie_rates = []
    for t in tiers:
        total = tier_wins[t]["total"] or 1
        plan_rates.append(tier_wins[t]["plan"] / total)
        noplan_rates.append(tier_wins[t]["noplan"] / total)
        tie_rates.append(tier_wins[t]["tie"] / total)

    bars1 = ax.bar(x, plan_rates, width, label="Plan Wins", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x, tie_rates, width, bottom=plan_rates, label="Tie", color="#CCCCCC", alpha=0.85)
    bars3 = ax.bar(x, noplan_rates, width,
                   bottom=[p + t for p, t in zip(plan_rates, tie_rates)],
                   label="No-Plan Wins", color="#DD8452", alpha=0.85)

    # Counts on bars
    for i, t in enumerate(tiers):
        tw = tier_wins[t]
        total = tw["total"]
        ax.text(i, 0.5, f"{tw['plan']}P / {tw['tie']}T / {tw['noplan']}N\n(n={total})",
                ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Tier (File Recall)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "-\n") for t in tiers])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")

    # Right: Average recall by tier, grouped bars
    ax = axes[1]
    tier_cond = defaultdict(lambda: defaultdict(list))
    for r in results:
        tier_cond[r["complexity_tier"]][r["condition"]].append(r["file_recall"])

    plan_avgs = []
    noplan_avgs = []
    for t in tiers:
        pv = tier_cond[t].get("with_plan", [])
        nv = tier_cond[t].get("no_plan", [])
        plan_avgs.append(sum(pv) / len(pv) if pv else 0)
        noplan_avgs.append(sum(nv) / len(nv) if nv else 0)

    w = 0.35
    bars1 = ax.bar(x - w / 2, plan_avgs, w, label="With Plan", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w / 2, noplan_avgs, w, label="No Plan", color="#DD8452", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Avg File Recall")
    ax.set_title("Avg Code-Gen File Recall by Tier")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "-\n") for t in tiers])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    fig.suptitle("Experiment 12: Plan vs No-Plan Code Generation (n=40)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def make_scatter_plot(results, metric_map, output_path):
    """Scatter: plan_file_recall (x) vs code_gen_delta (y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    task_results = defaultdict(dict)
    for r in results:
        task_results[r["task_id"]][r["condition"]] = r

    tier_colors = {
        "localized": "#4C72B0",
        "cross-module": "#55A868",
        "architectural": "#C44E52",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    xs, ys, colors, labels_done = [], [], [], set()
    for tid, conds in task_results.items():
        plan_r = conds.get("with_plan", {})
        noplan_r = conds.get("no_plan", {})
        if not plan_r or not noplan_r:
            continue

        m = metric_map.get(tid, {})
        plan_file_recall = m.get("plan_file_recall", 0)
        delta = plan_r.get("file_recall", 0) - noplan_r.get("file_recall", 0)
        tier = plan_r.get("complexity_tier", "")

        color = tier_colors.get(tier, "gray")
        label = tier if tier not in labels_done else None
        if label:
            labels_done.add(tier)

        ax.scatter(plan_file_recall, delta, c=color, s=80, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=label, zorder=3)
        xs.append(plan_file_recall)
        ys.append(delta)

    # Trend line
    if len(xs) >= 3:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, zorder=2)

        # Correlation
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, va="top", fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", zorder=1)
    ax.set_xlabel("Plan File Recall (from metrics_extended.json)", fontsize=12)
    ax.set_ylabel("Code-Gen Delta (plan_recall - noplan_recall)", fontsize=12)
    ax.set_title("Does Plan Accuracy Predict Code-Gen Benefit?", fontsize=14, fontweight="bold")
    ax.legend(title="Tier", loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis & writeup
# ---------------------------------------------------------------------------

def analyze_and_print(results, metric_map):
    """Print detailed comparison table and return summary dict."""
    task_results = defaultdict(dict)
    for r in results:
        task_results[r["task_id"]][r["condition"]] = r

    plan_items = [r for r in results if r["condition"] == "with_plan"]
    noplan_items = [r for r in results if r["condition"] == "no_plan"]

    def avg(items, key):
        vals = [r[key] for r in items]
        return sum(vals) / len(vals) if vals else 0

    # Overall win rate
    plan_wins, noplan_wins, ties = 0, 0, 0
    tier_wins = defaultdict(lambda: {"plan": 0, "noplan": 0, "tie": 0})

    print("\n" + "=" * 130)
    print("PLAN vs NO-PLAN COMPARISON (n=40)")
    print("=" * 130)
    header = (f"{'Task':<45s} | {'Tier':<14s} | {'PlanRecall':>10s} | "
              f"{'PlanFileR':>9s} | {'NoPlanR':>7s} | {'Delta':>6s} | "
              f"{'PlanDSim':>8s} | {'NoPlanDS':>8s} | {'Winner':>8s}")
    print(header)
    print("-" * 130)

    for tid in dict.fromkeys(r["task_id"] for r in results):
        conds = task_results[tid]
        plan_r = conds.get("with_plan", {})
        noplan_r = conds.get("no_plan", {})
        if not plan_r or not noplan_r:
            continue

        tier = plan_r.get("complexity_tier", "")
        p_rec = plan_r.get("file_recall", 0)
        n_rec = noplan_r.get("file_recall", 0)
        p_ds = plan_r.get("diff_similarity", 0)
        n_ds = noplan_r.get("diff_similarity", 0)
        plan_fr = metric_map.get(tid, {}).get("plan_file_recall", 0)
        delta = p_rec - n_rec

        if abs(p_rec - n_rec) < 0.001:
            winner = "Tie"
            ties += 1
            tier_wins[tier]["tie"] += 1
        elif p_rec > n_rec:
            winner = "Plan"
            plan_wins += 1
            tier_wins[tier]["plan"] += 1
        else:
            winner = "NoPlan"
            noplan_wins += 1
            tier_wins[tier]["noplan"] += 1

        print(f"{tid:<45s} | {tier:<14s} | {plan_fr:>10.3f} | "
              f"{p_rec:>9.3f} | {n_rec:>7.3f} | {delta:>+6.3f} | "
              f"{p_ds:>8.3f} | {n_ds:>8.3f} | {winner:>8s}")

    print("-" * 130)

    total = plan_wins + noplan_wins + ties
    print(f"\nOVERALL WIN RATE: Plan={plan_wins}/{total} ({plan_wins/total*100:.0f}%)  "
          f"NoPlan={noplan_wins}/{total} ({noplan_wins/total*100:.0f}%)  "
          f"Tie={ties}/{total} ({ties/total*100:.0f}%)")

    print(f"\nOVERALL AVERAGES:")
    print(f"  With Plan:  recall={avg(plan_items, 'file_recall'):.3f}  "
          f"precision={avg(plan_items, 'file_precision'):.3f}  "
          f"diff_sim={avg(plan_items, 'diff_similarity'):.3f}")
    print(f"  No Plan:    recall={avg(noplan_items, 'file_recall'):.3f}  "
          f"precision={avg(noplan_items, 'file_precision'):.3f}  "
          f"diff_sim={avg(noplan_items, 'diff_similarity'):.3f}")

    print(f"\nPER-TIER BREAKDOWN:")
    tiers = ["localized", "cross-module", "architectural"]
    summary = {}
    for tier in tiers:
        t_plan = [r for r in plan_items if r["complexity_tier"] == tier]
        t_noplan = [r for r in noplan_items if r["complexity_tier"] == tier]
        tw = tier_wins[tier]

        if not t_plan:
            continue

        pr = avg(t_plan, "file_recall")
        nr = avg(t_noplan, "file_recall")
        print(f"\n  {tier.upper()} (n={len(t_plan)}):")
        print(f"    With Plan:  recall={pr:.3f}  "
              f"precision={avg(t_plan, 'file_precision'):.3f}  "
              f"diff_sim={avg(t_plan, 'diff_similarity'):.3f}")
        print(f"    No Plan:    recall={nr:.3f}  "
              f"precision={avg(t_noplan, 'file_precision'):.3f}  "
              f"diff_sim={avg(t_noplan, 'diff_similarity'):.3f}")
        print(f"    Wins: Plan={tw['plan']}  NoPlan={tw['noplan']}  Tie={tw['tie']}")

        summary[tier] = {
            "n": len(t_plan),
            "plan_recall": round(pr, 3),
            "noplan_recall": round(nr, 3),
            "plan_wins": tw["plan"],
            "noplan_wins": tw["noplan"],
            "ties": tw["tie"],
        }

    summary["overall"] = {
        "n": total,
        "plan_wins": plan_wins,
        "noplan_wins": noplan_wins,
        "ties": ties,
        "plan_recall": round(avg(plan_items, "file_recall"), 3),
        "noplan_recall": round(avg(noplan_items, "file_recall"), 3),
        "plan_diff_sim": round(avg(plan_items, "diff_similarity"), 3),
        "noplan_diff_sim": round(avg(noplan_items, "diff_similarity"), 3),
    }

    return summary


def write_analysis_md(summary, output_path):
    """Write brief analysis markdown."""
    ov = summary.get("overall", {})
    lines = [
        "# Experiment 12: Plan vs No-Plan Code Generation (n=40)",
        "",
        "## Key Finding",
        "",
    ]

    if ov.get("noplan_wins", 0) > ov.get("plan_wins", 0):
        lines.append(f"**No-plan wins {ov['noplan_wins']}/{ov['n']} tasks** "
                     f"(plan wins {ov['plan_wins']}, ties {ov['ties']}). "
                     f"The n=10 finding that plans hurt code generation holds at scale.")
    elif ov.get("plan_wins", 0) > ov.get("noplan_wins", 0):
        lines.append(f"**Plan wins {ov['plan_wins']}/{ov['n']} tasks** "
                     f"(no-plan wins {ov['noplan_wins']}, ties {ov['ties']}). "
                     f"At n=40, plans show a net benefit for code generation.")
    else:
        lines.append(f"**Tie: plan wins {ov['plan_wins']}, no-plan wins {ov['noplan_wins']}** "
                     f"(ties {ov['ties']}). No clear winner at n=40.")

    lines += [
        "",
        "## Overall Averages",
        "",
        f"| Condition | File Recall | Diff Similarity |",
        f"|-----------|------------|-----------------|",
        f"| With Plan | {ov.get('plan_recall', 0):.3f} | {ov.get('plan_diff_sim', 0):.3f} |",
        f"| No Plan   | {ov.get('noplan_recall', 0):.3f} | {ov.get('noplan_diff_sim', 0):.3f} |",
        "",
        "## Win Rate by Tier",
        "",
        "| Tier | n | Plan Wins | NoPlan Wins | Ties | Plan Recall | NoPlan Recall |",
        "|------|---|-----------|-------------|------|-------------|---------------|",
    ]

    for tier in ["localized", "cross-module", "architectural"]:
        ts = summary.get(tier, {})
        if ts:
            lines.append(
                f"| {tier} | {ts['n']} | {ts['plan_wins']} | {ts['noplan_wins']} | "
                f"{ts['ties']} | {ts['plan_recall']:.3f} | {ts['noplan_recall']:.3f} |"
            )

    lines += [
        "",
        "## Plan Accuracy vs Code-Gen Benefit",
        "",
        "See `analysis/plan_accuracy_vs_codegen_delta.png` for scatter plot.",
        "The scatter shows whether plan file recall predicts whether plans help code generation.",
        "",
        "## Methodology",
        "",
        "- 40 tasks selected from extended dataset (15 localized, 15 cross-module, 10 architectural)",
        "- Tasks chosen with evenly-spaced plan_file_recall for diversity",
        f"- Model: {MODEL}, max_tokens={MAX_TOKENS}",
        "- Condition A: issue + plan + repo structure",
        "- Condition B: issue + repo structure only",
        "- Winner determined by file recall comparison",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    with open("data/tasks_extended.json") as f:
        tasks = json.load(f)
    with open("data/plans_extended.json") as f:
        plans = json.load(f)
    with open("data/metrics_extended.json") as f:
        metrics = json.load(f)
    print(f"Loaded {len(tasks)} tasks, {len(plans)} plans, {len(metrics)} metrics")

    metric_map = {m["task_id"]: m for m in metrics}

    # Select tasks
    print("\nSelecting 40 tasks with recall diversity...")
    subset = select_tasks(tasks, plans, metrics, TIER_COUNTS)
    print(f"Total selected: {len(subset)} tasks\n")

    # Resume support
    all_results, done_pairs = load_existing_results()

    # Track which tasks need work
    tasks_needing_work = []
    for t, p, m in subset:
        tid = t["task_id"]
        needs_plan = (tid, "with_plan") not in done_pairs
        needs_noplan = (tid, "no_plan") not in done_pairs
        if needs_plan or needs_noplan:
            tasks_needing_work.append((t, p, m, needs_plan, needs_noplan))

    if not tasks_needing_work:
        print("All tasks already complete! Skipping API calls.")
    else:
        print(f"Need to process {len(tasks_needing_work)} tasks "
              f"({sum(1 for _, _, _, a, b in tasks_needing_work if a) + sum(1 for _, _, _, a, b in tasks_needing_work if b)} API calls)")

    task_map_local = {t["task_id"]: t for t, _, _, _, _ in tasks_needing_work}
    task_map_local.update({t["task_id"]: t for t, _, _ in subset})

    for i, (task, plan, met, needs_plan, needs_noplan) in enumerate(tasks_needing_work, 1):
        tid = task["task_id"]
        tier = task["complexity_tier"]
        print(f"[{i}/{len(tasks_needing_work)}] {tid} ({tier})")

        if needs_plan:
            print(f"    Generating WITH plan...")
            prompt = build_plan_prompt(task, plan)
            response = call_api(prompt, tid, "with_plan")
            time.sleep(DELAY)

            if response:
                ev = evaluate_one(task, response, "with_plan")
                ev["raw_response"] = response
                all_results.append(ev)
                done_pairs.add((tid, "with_plan"))
                print(f"    WITH PLAN  -> recall={ev['file_recall']:.3f}  "
                      f"precision={ev['file_precision']:.3f}  "
                      f"diff_sim={ev['diff_similarity']:.3f}")
            else:
                print(f"    WITH PLAN  -> FAILED")

        if needs_noplan:
            print(f"    Generating WITHOUT plan...")
            prompt = build_noplan_prompt(task)
            response = call_api(prompt, tid, "no_plan")
            time.sleep(DELAY)

            if response:
                ev = evaluate_one(task, response, "no_plan")
                ev["raw_response"] = response
                all_results.append(ev)
                done_pairs.add((tid, "no_plan"))
                print(f"    NO PLAN    -> recall={ev['file_recall']:.3f}  "
                      f"precision={ev['file_precision']:.3f}  "
                      f"diff_sim={ev['diff_similarity']:.3f}")
            else:
                print(f"    NO PLAN    -> FAILED")

        # Save every 5 tasks
        if i % 5 == 0:
            save_results(all_results)
            print(f"    [Checkpoint saved: {len(all_results)} results]")

        print()

    # Final save
    save_results(all_results)
    print(f"\nSaved {len(all_results)} total results to {OUTPUT_FILE}")

    # Filter to only selected task IDs for analysis
    selected_ids = {t["task_id"] for t, _, _ in subset}
    analysis_results = [r for r in all_results if r["task_id"] in selected_ids]

    # Remove raw_response for analysis
    clean_results = [{k: v for k, v in r.items() if k != "raw_response"} for r in analysis_results]

    # Analysis
    summary = analyze_and_print(clean_results, metric_map)

    # Plots
    os.makedirs("analysis", exist_ok=True)
    make_winrate_chart(clean_results, metric_map, PLOT_WINRATE)
    make_scatter_plot(clean_results, metric_map, PLOT_SCATTER)

    # Writeup
    write_analysis_md(summary, WRITEUP_FILE)

    print("\nDone!")


def reeval():
    """Re-analyze from saved results without API calls."""
    with open("data/tasks_extended.json") as f:
        tasks = json.load(f)
    with open("data/metrics_extended.json") as f:
        metrics = json.load(f)
    metric_map = {m["task_id"]: m for m in metrics}
    task_map = {t["task_id"]: t for t in tasks}

    with open(OUTPUT_FILE) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} results")

    # Re-evaluate
    re_results = []
    for r in all_results:
        task = task_map.get(r["task_id"])
        if not task:
            continue
        raw = r.get("raw_response", "")
        ev = evaluate_one(task, raw, r["condition"])
        ev["raw_response"] = raw
        re_results.append(ev)

    save_results(re_results)

    clean = [{k: v for k, v in r.items() if k != "raw_response"} for r in re_results]
    summary = analyze_and_print(clean, metric_map)

    os.makedirs("analysis", exist_ok=True)
    make_winrate_chart(clean, metric_map, PLOT_WINRATE)
    make_scatter_plot(clean, metric_map, PLOT_SCATTER)
    write_analysis_md(summary, WRITEUP_FILE)
    print("\nDone!")


if __name__ == "__main__":
    import sys
    if "--reeval" in sys.argv:
        reeval()
    else:
        main()
