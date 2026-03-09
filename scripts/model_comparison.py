#!/usr/bin/env python3
"""
Multi-model plan generation comparison.

Generates plans with claude-haiku-4-5-20251001 and compares against existing
claude-sonnet-4-20250514 plans from data/plans.json.

Outputs:
  - data/plans_haiku.json          -- Haiku-generated plans
  - data/model_comparison_metrics.json -- Recall/precision for both models
  - analysis/model_comparison.png  -- Grouped bar chart by model and tier
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import traceback
from pathlib import PurePosixPath

from dotenv import load_dotenv
load_dotenv()

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── API client ───────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# ── Models to compare ────────────────────────────────────────────────────────

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
}

MAX_TOKENS = 4096

# ── Prompt (identical to generate_plans.py) ──────────────────────────────────

PROMPT_TEMPLATE = """\
You are a senior software engineer planning a fix for a GitHub issue. You have not yet looked at the code — you are working from the issue description and the repository structure only.

## Issue
{problem_statement}

## Repository Structure
{repo_structure_summary}

## Your Task
Produce a structured implementation plan. Be specific — name actual file paths from the repository structure above. Do NOT be vague.

Respond in exactly this JSON format:
{{
  "files_to_inspect": [
    {{"path": "path/to/file.py", "reason": "why this file is relevant"}}
  ],
  "files_to_modify": [
    {{"path": "path/to/file.py", "change_type": "edit|create|delete", "description": "what change to make"}}
  ],
  "implementation_steps": [
    {{"step": 1, "description": "concise description of what to do", "files_involved": ["path/to/file.py"]}}
  ],
  "assumptions": ["assumption that needs verification"],
  "risks": ["thing that could go wrong"],
  "validation": ["how to verify the fix works"]
}}
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = 60000) -> str:
    """Truncate repo structure if it's too large (Haiku has smaller context)."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def normalize_path(p: str) -> str:
    """Normalize a file path: strip leading ./ or /."""
    p = p.strip()
    while p.startswith("./") or p.startswith("/"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def paths_match(a: str, b: str) -> bool:
    """Check if two paths refer to the same file using suffix matching."""
    na = normalize_path(a)
    nb = normalize_path(b)
    if na == nb:
        return True
    if na.endswith("/" + nb) or nb.endswith("/" + na):
        return True
    if na.endswith(nb) or nb.endswith(na):
        longer, shorter = (na, nb) if len(na) >= len(nb) else (nb, na)
        idx = longer.find(shorter)
        if idx == 0 or (idx > 0 and longer[idx - 1] == "/"):
            return True
    return False


def match_files(plan_files: set, gt_files: set):
    """Match plan files against ground truth files using normalized path matching."""
    matched_gt = set()
    matched_plan = set()
    for gf in gt_files:
        for pf in plan_files:
            if paths_match(gf, pf):
                matched_gt.add(gf)
                matched_plan.add(pf)
    missed_gt = gt_files - matched_gt
    extra_plan = plan_files - matched_plan
    return matched_gt, matched_plan, missed_gt, extra_plan


def extract_plan_files(plan: dict) -> set:
    """Extract the union of files_to_inspect and files_to_modify paths."""
    plan_files = set()
    for entry in plan.get("files_to_inspect", []):
        plan_files.add(entry["path"])
    for entry in plan.get("files_to_modify", []):
        plan_files.add(entry["path"])
    return plan_files


# ── Plan generation ──────────────────────────────────────────────────────────

def generate_plan_for_model(task: dict, model_name: str, model_id: str,
                            index: int, total: int, max_retries: int = 1) -> dict:
    """Generate a plan for a single task using a specific model."""
    task_id = task["task_id"]
    print(f"  [{index}/{total}] {model_name}: {task_id}...")

    repo_structure = truncate_repo_structure(task["repo_structure_summary"])

    prompt = PROMPT_TEMPLATE.format(
        problem_statement=task["problem_statement"],
        repo_structure_summary=repo_structure,
    )

    raw_text = None
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"           -> Retry {attempt}/{max_retries}...")
                time.sleep(2)

            response = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            plan = parse_json_response(raw_text)

            # Basic validation
            for key in ("files_to_inspect", "files_to_modify", "implementation_steps"):
                if key not in plan:
                    raise ValueError(f"Plan missing required key: {key}")

            inspect_count = len(plan.get("files_to_inspect", []))
            modify_count = len(plan.get("files_to_modify", []))
            print(f"           -> {inspect_count} inspect, {modify_count} modify")

            return {
                "task_id": task_id,
                "model": model_id,
                "plan": plan,
                "raw_response": raw_text,
            }

        except json.JSONDecodeError as e:
            last_error = e
            print(f"           -> JSON parse error: {e}")
        except anthropic.APIError as e:
            last_error = e
            print(f"           -> API error: {e}")
        except Exception as e:
            last_error = e
            print(f"           -> Error: {e}")

    # All retries exhausted — return a failed entry instead of raising
    print(f"           -> FAILED after {max_retries + 1} attempts: {last_error}")
    return {
        "task_id": task_id,
        "model": model_id,
        "plan": None,
        "raw_response": raw_text,
        "error": str(last_error),
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # ── Load data ────────────────────────────────────────────────────────
    with open("data/tasks.json") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")

    with open("data/plans.json") as f:
        sonnet_plans_raw = json.load(f)
    sonnet_by_task = {p["task_id"]: p for p in sonnet_plans_raw}
    print(f"Loaded {len(sonnet_by_task)} existing Sonnet plans")

    # ── Generate Haiku plans ─────────────────────────────────────────────
    haiku_output_path = "data/plans_haiku.json"
    haiku_existing = {}
    if os.path.exists(haiku_output_path):
        with open(haiku_output_path) as f:
            for entry in json.load(f):
                if entry.get("plan") is not None:
                    haiku_existing[entry["task_id"]] = entry
        print(f"Found {len(haiku_existing)} existing Haiku plans (will skip)")

    haiku_plans = []
    total = len(tasks)

    print(f"\n{'='*60}")
    print("Generating Haiku plans...")
    print(f"{'='*60}")

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        if task_id in haiku_existing:
            print(f"  [{i}/{total}] haiku: {task_id} (cached)")
            haiku_plans.append(haiku_existing[task_id])
            continue

        result = generate_plan_for_model(
            task, "haiku", MODELS["haiku"], i, total
        )
        haiku_plans.append(result)

        # Rate limit: 0.5s delay between calls
        if i < total:
            time.sleep(0.5)

    # Save Haiku plans
    with open(haiku_output_path, "w") as f:
        json.dump(haiku_plans, f, indent=2)

    haiku_successes = sum(1 for p in haiku_plans if p.get("plan") is not None)
    print(f"\nHaiku: {haiku_successes}/{total} plans generated successfully")
    print(f"Saved to {haiku_output_path}")

    # ── Compute metrics for both models ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing metrics...")
    print(f"{'='*60}")

    haiku_by_task = {p["task_id"]: p for p in haiku_plans}

    all_metrics = []

    for task in tasks:
        task_id = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        total_repo_files = task["total_repo_files"]
        tier = task["complexity_tier"]

        entry = {
            "task_id": task_id,
            "complexity_tier": tier,
            "ground_truth_count": len(gt_files),
            "total_repo_files": total_repo_files,
        }

        # Sonnet metrics
        sonnet_plan_data = sonnet_by_task.get(task_id, {}).get("plan")
        if sonnet_plan_data:
            s_files = extract_plan_files(sonnet_plan_data)
            s_matched_gt, s_matched_plan, _, _ = match_files(s_files, gt_files)
            entry["sonnet_plan_files_count"] = len(s_files)
            entry["sonnet_recall"] = round(len(s_matched_gt) / len(gt_files), 4) if gt_files else 0.0
            entry["sonnet_precision"] = round(len(s_matched_plan) / len(s_files), 4) if s_files else 0.0
        else:
            entry["sonnet_plan_files_count"] = 0
            entry["sonnet_recall"] = None
            entry["sonnet_precision"] = None

        # Haiku metrics
        haiku_plan_data = haiku_by_task.get(task_id, {}).get("plan")
        if haiku_plan_data:
            h_files = extract_plan_files(haiku_plan_data)
            h_matched_gt, h_matched_plan, _, _ = match_files(h_files, gt_files)
            entry["haiku_plan_files_count"] = len(h_files)
            entry["haiku_recall"] = round(len(h_matched_gt) / len(gt_files), 4) if gt_files else 0.0
            entry["haiku_precision"] = round(len(h_matched_plan) / len(h_files), 4) if h_files else 0.0
        else:
            entry["haiku_plan_files_count"] = 0
            entry["haiku_recall"] = None
            entry["haiku_precision"] = None

        # Recall difference
        if entry["sonnet_recall"] is not None and entry["haiku_recall"] is not None:
            entry["recall_diff"] = round(entry["haiku_recall"] - entry["sonnet_recall"], 4)
        else:
            entry["recall_diff"] = None

        all_metrics.append(entry)

    # Save metrics
    with open("data/model_comparison_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to data/model_comparison_metrics.json")

    # ── Print comparison table ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("MODEL COMPARISON: FILE RECALL")
    print(f"{'='*80}")
    print(f"{'Task ID':45s} | {'Tier':15s} | {'Haiku':>6s} | {'Sonnet':>6s} | {'Diff':>6s}")
    print(f"{'-'*45}-+-{'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for m in all_metrics:
        h_str = f"{m['haiku_recall']:.2f}" if m["haiku_recall"] is not None else "  N/A"
        s_str = f"{m['sonnet_recall']:.2f}" if m["sonnet_recall"] is not None else "  N/A"
        d_str = f"{m['recall_diff']:+.2f}" if m["recall_diff"] is not None else "  N/A"
        print(f"{m['task_id']:45s} | {m['complexity_tier']:15s} | {h_str:>6s} | {s_str:>6s} | {d_str:>6s}")

    # ── Per-tier averages ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-TIER AVERAGES")
    print(f"{'='*80}")

    tier_order = ["localized", "cross-module", "architectural"]
    tier_data = {t: {"haiku_recalls": [], "sonnet_recalls": [],
                     "haiku_precisions": [], "sonnet_precisions": []}
                 for t in tier_order}

    for m in all_metrics:
        tier = m["complexity_tier"]
        if tier not in tier_data:
            continue
        if m["haiku_recall"] is not None:
            tier_data[tier]["haiku_recalls"].append(m["haiku_recall"])
        if m["sonnet_recall"] is not None:
            tier_data[tier]["sonnet_recalls"].append(m["sonnet_recall"])
        if m["haiku_precision"] is not None:
            tier_data[tier]["haiku_precisions"].append(m["haiku_precision"])
        if m["sonnet_precision"] is not None:
            tier_data[tier]["sonnet_precisions"].append(m["sonnet_precision"])

    print(f"{'Tier':15s} | {'Haiku Recall':>12s} | {'Sonnet Recall':>13s} | {'Gap':>6s} | {'Haiku Prec':>10s} | {'Sonnet Prec':>11s}")
    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*13}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}")

    tier_avg = {}
    for tier in tier_order:
        td = tier_data[tier]
        h_avg_r = np.mean(td["haiku_recalls"]) if td["haiku_recalls"] else float("nan")
        s_avg_r = np.mean(td["sonnet_recalls"]) if td["sonnet_recalls"] else float("nan")
        h_avg_p = np.mean(td["haiku_precisions"]) if td["haiku_precisions"] else float("nan")
        s_avg_p = np.mean(td["sonnet_precisions"]) if td["sonnet_precisions"] else float("nan")
        gap = h_avg_r - s_avg_r

        tier_avg[tier] = {
            "haiku_recall": h_avg_r, "sonnet_recall": s_avg_r,
            "haiku_precision": h_avg_p, "sonnet_precision": s_avg_p,
            "gap": gap,
        }

        print(f"{tier:15s} | {h_avg_r:12.3f} | {s_avg_r:13.3f} | {gap:+6.3f} | {h_avg_p:10.3f} | {s_avg_p:11.3f}")

    # Overall averages
    all_h_r = [m["haiku_recall"] for m in all_metrics if m["haiku_recall"] is not None]
    all_s_r = [m["sonnet_recall"] for m in all_metrics if m["sonnet_recall"] is not None]
    all_h_p = [m["haiku_precision"] for m in all_metrics if m["haiku_precision"] is not None]
    all_s_p = [m["sonnet_precision"] for m in all_metrics if m["sonnet_precision"] is not None]

    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*13}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}")
    print(f"{'OVERALL':15s} | {np.mean(all_h_r):12.3f} | {np.mean(all_s_r):13.3f} | "
          f"{np.mean(all_h_r) - np.mean(all_s_r):+6.3f} | "
          f"{np.mean(all_h_p):10.3f} | {np.mean(all_s_p):11.3f}")

    # ── Key insights ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")

    overall_gap = np.mean(all_h_r) - np.mean(all_s_r)
    print(f"1. Overall recall gap (Haiku - Sonnet): {overall_gap:+.3f}")

    if tier_avg:
        loc_gap = tier_avg.get("localized", {}).get("gap", 0)
        cross_gap = tier_avg.get("cross-module", {}).get("gap", 0)
        arch_gap = tier_avg.get("architectural", {}).get("gap", 0)
        print(f"2. Recall gap by tier:")
        print(f"     Localized:     {loc_gap:+.3f}")
        print(f"     Cross-module:  {cross_gap:+.3f}")
        print(f"     Architectural: {arch_gap:+.3f}")

        if abs(arch_gap) > abs(loc_gap):
            print(f"3. The recall gap WIDENS for architectural tasks ({arch_gap:+.3f} vs {loc_gap:+.3f})")
            print(f"   -> Haiku degrades MORE on complex tasks")
        else:
            print(f"3. The recall gap does NOT widen for architectural tasks")
            print(f"   -> Degradation is relatively uniform across tiers")

    # Cost-quality tradeoff (rough estimates)
    # Haiku: ~$0.80/M input, $4/M output
    # Sonnet: ~$3/M input, $15/M output
    # Approximate: Haiku is ~4x cheaper than Sonnet
    print(f"\n4. Cost-quality tradeoff (approximate):")
    print(f"   Haiku:  ~$0.80/M input, ~$4/M output")
    print(f"   Sonnet: ~$3/M input,    ~$15/M output")
    print(f"   Haiku is roughly 3-4x cheaper per token")
    if abs(overall_gap) < 0.05:
        print(f"   With a recall gap of only {overall_gap:+.3f}, Haiku offers strong cost-efficiency")
    elif overall_gap < -0.15:
        print(f"   With a recall gap of {overall_gap:+.3f}, the quality drop may not justify savings")
    else:
        print(f"   The cost-quality tradeoff depends on use case and tier")

    # ── Generate comparison chart ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating comparison chart...")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Chart 1: Recall by model and tier ────────────────────────────────
    ax = axes[0]
    x = np.arange(len(tier_order))
    width = 0.32

    haiku_vals = [tier_avg[t]["haiku_recall"] for t in tier_order]
    sonnet_vals = [tier_avg[t]["sonnet_recall"] for t in tier_order]

    bars_h = ax.bar(x - width/2, haiku_vals, width, label="Haiku 4.5",
                    color="#6baed6", edgecolor="black", linewidth=0.5)
    bars_s = ax.bar(x + width/2, sonnet_vals, width, label="Sonnet 4",
                    color="#fd8d3c", edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar in bars_h:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("File Recall", fontsize=12)
    ax.set_title("File Recall by Model and Complexity Tier", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "-\n") for t in tier_order], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, min(1.15, max(haiku_vals + sonnet_vals) + 0.15))
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # ── Chart 2: Per-task recall comparison ──────────────────────────────
    ax2 = axes[1]

    # Sort tasks by tier then by sonnet recall
    tier_rank = {"localized": 0, "cross-module": 1, "architectural": 2}
    sorted_metrics = sorted(all_metrics,
                           key=lambda m: (tier_rank.get(m["complexity_tier"], 9),
                                         -(m["sonnet_recall"] or 0)))

    task_labels = []
    h_vals = []
    s_vals = []
    colors_tier = []
    tier_colors = {"localized": "#2ca02c", "cross-module": "#ff7f0e", "architectural": "#d62728"}

    for m in sorted_metrics:
        short_id = m["task_id"].split("__")[-1] if "__" in m["task_id"] else m["task_id"]
        task_labels.append(short_id)
        h_vals.append(m["haiku_recall"] if m["haiku_recall"] is not None else 0)
        s_vals.append(m["sonnet_recall"] if m["sonnet_recall"] is not None else 0)
        colors_tier.append(tier_colors.get(m["complexity_tier"], "gray"))

    y_pos = np.arange(len(task_labels))
    bar_height = 0.35

    ax2.barh(y_pos - bar_height/2, s_vals, bar_height, label="Sonnet 4",
             color="#fd8d3c", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax2.barh(y_pos + bar_height/2, h_vals, bar_height, label="Haiku 4.5",
             color="#6baed6", edgecolor="black", linewidth=0.3, alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(task_labels, fontsize=8)
    ax2.set_xlabel("File Recall", fontsize=12)
    ax2.set_title("Per-Task Recall Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")
    ax2.set_xlim(0, 1.1)
    ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # Color-code tier labels
    for i, (label, color) in enumerate(zip(task_labels, colors_tier)):
        ax2.get_yticklabels()[i].set_color(color)

    plt.tight_layout()
    os.makedirs("analysis", exist_ok=True)
    plt.savefig("analysis/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved analysis/model_comparison.png")

    print(f"\n{'='*60}")
    print("DONE. All outputs generated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
