#!/usr/bin/env python3
"""
Opus vs Sonnet vs Haiku Model Comparison

Generates plans using claude-opus-4-6 for 8 selected tasks (stratified by tier
and Sonnet recall range), computes metrics, and produces a side-by-side comparison
with existing Sonnet and Haiku plans.

Outputs:
  data/plans_opus.json                — Opus plans (8 tasks)
  data/opus_comparison_metrics.json   — Per-task metrics for all 3 models
  analysis/opus_comparison.png        — Grouped bar chart: recall by tier
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
from pathlib import PurePosixPath

from dotenv import load_dotenv
load_dotenv()

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

client = anthropic.Anthropic()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096

# ── Selected 8 tasks ──────────────────────────────────────────────────────────
SELECTED_TASKS = [
    # Localized (3)
    "scikit-learn__scikit-learn-12682",   # Sonnet recall 1.0
    "sphinx-doc__sphinx-9461",            # Sonnet recall 1.0
    "pylint-dev__pylint-8898",            # Sonnet recall 0.333
    # Cross-module (3)
    "django__django-13841",              # Sonnet recall 1.0
    "pylint-dev__pylint-6386",           # Sonnet recall 0.75
    "sympy__sympy-16597",                # Sonnet recall 0.333
    # Architectural (2)
    "matplotlib__matplotlib-24013",      # Sonnet recall 0.333
    "django__django-10989",              # Sonnet recall 0.077
]

# ── Same prompt template as generate_plans.py ────────────────────────────────
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


# ── Utility functions (from compute_metrics.py) ─────────────────────────────
def normalize_path(p: str) -> str:
    p = p.strip()
    while p.startswith("./") or p.startswith("/"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def paths_match(a: str, b: str) -> bool:
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
    """Extract unique file paths from a plan's files_to_inspect + files_to_modify."""
    files = set()
    for entry in plan.get("files_to_inspect", []):
        files.add(entry["path"])
    for entry in plan.get("files_to_modify", []):
        files.add(entry["path"])
    return files


def compute_recall_precision(plan_files: set, gt_files: set):
    matched_gt, matched_plan, missed_gt, extra_plan = match_files(plan_files, gt_files)
    recall = len(matched_gt) / len(gt_files) if gt_files else 0.0
    precision = len(matched_plan) / len(plan_files) if plan_files else 0.0
    return round(recall, 4), round(precision, 4)


# ── Plan generation ─────────────────────────────────────────────────────────
def parse_json_response(text: str) -> dict:
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = 50000) -> str:
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def generate_opus_plan(task: dict, index: int, total: int, max_retries: int = 2) -> dict:
    task_id = task["task_id"]
    print(f"  [{index}/{total}] Generating Opus plan for {task_id}...")

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
                print(f"         -> Retry {attempt}/{max_retries}...")
                time.sleep(3)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            stop_reason = response.stop_reason

            if stop_reason == "max_tokens":
                print(f"         -> WARNING: response truncated (max_tokens)")

            plan = parse_json_response(raw_text)

            for key in ("files_to_inspect", "files_to_modify", "implementation_steps"):
                if key not in plan:
                    raise ValueError(f"Plan missing required key: {key}")

            inspect_count = len(plan.get("files_to_inspect", []))
            modify_count = len(plan.get("files_to_modify", []))
            steps_count = len(plan.get("implementation_steps", []))
            print(f"         -> {inspect_count} inspect, {modify_count} modify, {steps_count} steps")

            return {
                "task_id": task_id,
                "model": MODEL,
                "plan": plan,
                "raw_response": raw_text,
            }

        except json.JSONDecodeError as e:
            last_error = e
            print(f"         -> JSON parse error on attempt {attempt + 1}: {e}")
            if raw_text:
                print(f"         -> Raw preview: {raw_text[:200]}...")
        except anthropic.APIError as e:
            last_error = e
            print(f"         -> API error on attempt {attempt + 1}: {e}")

    raise last_error


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("OPUS vs SONNET vs HAIKU MODEL COMPARISON")
    print("=" * 70)

    # Load data
    with open("data/tasks.json") as f:
        all_tasks = json.load(f)
    with open("data/plans.json") as f:
        sonnet_plans = json.load(f)
    with open("data/plans_haiku.json") as f:
        haiku_plans = json.load(f)

    # Index by task_id
    tasks_by_id = {t["task_id"]: t for t in all_tasks}
    sonnet_by_id = {p["task_id"]: p for p in sonnet_plans}
    haiku_by_id = {p["task_id"]: p for p in haiku_plans}

    # Filter to selected tasks
    selected_tasks = [tasks_by_id[tid] for tid in SELECTED_TASKS]
    print(f"\nSelected {len(selected_tasks)} tasks for Opus comparison:\n")
    for t in selected_tasks:
        print(f"  {t['task_id']:45s}  tier={t['complexity_tier']}")

    # ── Step 1: Generate Opus plans ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("STEP 1: Generating Opus plans (8 API calls)")
    print(f"{'─' * 70}\n")

    # Check for existing opus plans to allow resume
    opus_plans_path = "data/plans_opus.json"
    existing_opus = {}
    if os.path.exists(opus_plans_path):
        with open(opus_plans_path) as f:
            existing_list = json.load(f)
        for entry in existing_list:
            if entry.get("plan") is not None:
                existing_opus[entry["task_id"]] = entry
        if existing_opus:
            print(f"  Found {len(existing_opus)} existing Opus plans — will reuse\n")

    opus_results = []
    for i, task in enumerate(selected_tasks, 1):
        tid = task["task_id"]
        if tid in existing_opus:
            print(f"  [{i}/{len(selected_tasks)}] Reusing cached Opus plan for {tid}")
            opus_results.append(existing_opus[tid])
            continue

        result = generate_opus_plan(task, i, len(selected_tasks))
        opus_results.append(result)

        # Save incrementally in case of failure
        with open(opus_plans_path, "w") as f:
            json.dump(opus_results, f, indent=2)

        # 2-second delay between calls
        if i < len(selected_tasks):
            time.sleep(2)

    # Final save
    with open(opus_plans_path, "w") as f:
        json.dump(opus_results, f, indent=2)
    print(f"\n  Saved {len(opus_results)} Opus plans to {opus_plans_path}")

    opus_by_id = {p["task_id"]: p for p in opus_results}

    # ── Step 2: Compute metrics for all 3 models ───────────────────────────
    print(f"\n{'─' * 70}")
    print("STEP 2: Computing metrics for all 3 models on 8 tasks")
    print(f"{'─' * 70}\n")

    comparison_metrics = []

    for task in selected_tasks:
        tid = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        tier = task["complexity_tier"]

        # Haiku
        haiku_plan = haiku_by_id[tid]["plan"]
        haiku_files = extract_plan_files(haiku_plan)
        haiku_recall, haiku_prec = compute_recall_precision(haiku_files, gt_files)

        # Sonnet
        sonnet_plan = sonnet_by_id[tid]["plan"]
        sonnet_files = extract_plan_files(sonnet_plan)
        sonnet_recall, sonnet_prec = compute_recall_precision(sonnet_files, gt_files)

        # Opus
        opus_plan = opus_by_id[tid]["plan"]
        opus_files = extract_plan_files(opus_plan)
        opus_recall, opus_prec = compute_recall_precision(opus_files, gt_files)

        # Determine best model
        recalls = {"Haiku": haiku_recall, "Sonnet": sonnet_recall, "Opus": opus_recall}
        best_recall = max(recalls.values())
        best_models = [m for m, r in recalls.items() if r == best_recall]
        best_model = " / ".join(best_models)

        entry = {
            "task_id": tid,
            "complexity_tier": tier,
            "ground_truth_count": len(gt_files),
            "haiku_plan_files": len(haiku_files),
            "haiku_recall": haiku_recall,
            "haiku_precision": haiku_prec,
            "sonnet_plan_files": len(sonnet_files),
            "sonnet_recall": sonnet_recall,
            "sonnet_precision": sonnet_prec,
            "opus_plan_files": len(opus_files),
            "opus_recall": opus_recall,
            "opus_precision": opus_prec,
            "best_model": best_model,
        }
        comparison_metrics.append(entry)

    with open("data/opus_comparison_metrics.json", "w") as f:
        json.dump(comparison_metrics, f, indent=2)
    print("  Saved data/opus_comparison_metrics.json\n")

    # ── Step 3: Print comparison table ──────────────────────────────────────
    print(f"{'─' * 70}")
    print("STEP 3: Full 3-Model Comparison")
    print(f"{'─' * 70}\n")

    header = (f"{'Task':<42s} {'Tier':<14s} {'GT':>3s} "
              f"{'Haiku':>7s} {'Sonnet':>7s} {'Opus':>7s}  {'Best Model':<20s}")
    print(header)
    print("─" * len(header))

    tier_recalls = {}  # tier -> {model: [recalls]}

    for m in comparison_metrics:
        short_id = m["task_id"].split("__")[-1]
        tier_short = m["complexity_tier"]
        print(f"{short_id:<42s} {tier_short:<14s} {m['ground_truth_count']:>3d} "
              f"{m['haiku_recall']:>7.3f} {m['sonnet_recall']:>7.3f} {m['opus_recall']:>7.3f}  "
              f"{m['best_model']:<20s}")

        if tier_short not in tier_recalls:
            tier_recalls[tier_short] = {"Haiku": [], "Sonnet": [], "Opus": []}
        tier_recalls[tier_short]["Haiku"].append(m["haiku_recall"])
        tier_recalls[tier_short]["Sonnet"].append(m["sonnet_recall"])
        tier_recalls[tier_short]["Opus"].append(m["opus_recall"])

    print("─" * len(header))

    # Overall averages
    all_haiku = [m["haiku_recall"] for m in comparison_metrics]
    all_sonnet = [m["sonnet_recall"] for m in comparison_metrics]
    all_opus = [m["opus_recall"] for m in comparison_metrics]

    print(f"\n{'OVERALL AVERAGE':<42s} {'':14s} {'':>3s} "
          f"{np.mean(all_haiku):>7.3f} {np.mean(all_sonnet):>7.3f} {np.mean(all_opus):>7.3f}")

    # Per-tier averages
    print(f"\n{'─' * 70}")
    print("Per-Tier Average Recall:")
    print(f"{'─' * 70}")
    tier_order = ["localized", "cross-module", "architectural"]
    for tier in tier_order:
        if tier in tier_recalls:
            h = np.mean(tier_recalls[tier]["Haiku"])
            s = np.mean(tier_recalls[tier]["Sonnet"])
            o = np.mean(tier_recalls[tier]["Opus"])
            best = max(h, s, o)
            winner = []
            if h == best: winner.append("Haiku")
            if s == best: winner.append("Sonnet")
            if o == best: winner.append("Opus")
            print(f"  {tier:<18s}  Haiku={h:.3f}  Sonnet={s:.3f}  Opus={o:.3f}  -> {' / '.join(winner)}")

    # Precision table
    print(f"\n{'─' * 70}")
    print("Precision Comparison:")
    print(f"{'─' * 70}")
    print(f"{'Task':<42s} {'Haiku':>7s} {'Sonnet':>7s} {'Opus':>7s}")
    print("─" * 63)
    for m in comparison_metrics:
        short_id = m["task_id"].split("__")[-1]
        print(f"{short_id:<42s} {m['haiku_precision']:>7.3f} {m['sonnet_precision']:>7.3f} {m['opus_precision']:>7.3f}")
    print(f"\n{'AVERAGE':<42s} {np.mean([m['haiku_precision'] for m in comparison_metrics]):>7.3f} "
          f"{np.mean([m['sonnet_precision'] for m in comparison_metrics]):>7.3f} "
          f"{np.mean([m['opus_precision'] for m in comparison_metrics]):>7.3f}")

    # ── Step 4: Create grouped bar chart ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("STEP 4: Generating visualization")
    print(f"{'─' * 70}\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    colors = {"Haiku": "#93c5fd", "Sonnet": "#6366f1", "Opus": "#f59e0b"}
    bar_width = 0.25

    for ax_idx, tier in enumerate(tier_order):
        ax = axes[ax_idx]
        if tier not in tier_recalls:
            continue

        # Get tasks in this tier
        tier_tasks = [m for m in comparison_metrics if m["complexity_tier"] == tier]
        task_labels = [m["task_id"].split("__")[-1].replace("-", "\n", 1) for m in tier_tasks]
        n = len(tier_tasks)
        x = np.arange(n)

        haiku_vals = [m["haiku_recall"] for m in tier_tasks]
        sonnet_vals = [m["sonnet_recall"] for m in tier_tasks]
        opus_vals = [m["opus_recall"] for m in tier_tasks]

        bars_h = ax.bar(x - bar_width, haiku_vals, bar_width, label="Haiku", color=colors["Haiku"], edgecolor="white", linewidth=0.5)
        bars_s = ax.bar(x, sonnet_vals, bar_width, label="Sonnet", color=colors["Sonnet"], edgecolor="white", linewidth=0.5)
        bars_o = ax.bar(x + bar_width, opus_vals, bar_width, label="Opus", color=colors["Opus"], edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bars in [bars_h, bars_s, bars_o]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=7, ha='center')
        ax.set_title(f"{tier.replace('-', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        if ax_idx == 0:
            ax.set_ylabel("File Recall", fontsize=11)
        if ax_idx == 1:
            ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=True,
                      bbox_to_anchor=(0.5, -0.15))

    fig.suptitle("Plan File Recall: Haiku vs Sonnet vs Opus\n(8 SWE-bench tasks, stratified by complexity)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig("analysis/opus_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("  Saved analysis/opus_comparison.png")

    print(f"\n{'=' * 70}")
    print("DONE. All outputs saved.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
