#!/usr/bin/env python3
"""
Prompt Sensitivity Analysis

Tests whether different prompting strategies produce different plan quality.
Compares 3 prompt variants:
  A - Minimal prompt
  B - Dependency-tracing prompt
  C - Structured prompt (existing plans from data/plans.json)

Outputs:
  data/plans_minimal.json       - Plans from variant A
  data/plans_dependency.json    - Plans from variant B
  data/prompt_sensitivity_metrics.json - Metrics for all 3 variants
  analysis/prompt_sensitivity.png      - Comparison chart
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_MINIMAL = """\
Fix this GitHub issue. What files need to change?

Issue: {problem_statement}

Repository files: {repo_structure_summary}

List the files to inspect and modify in JSON format:
{{"files_to_inspect": [{{"path": "...", "reason": "..."}}], "files_to_modify": [{{"path": "...", "change_type": "edit", "description": "..."}}]}}
"""

PROMPT_DEPENDENCY = """\
You are planning a fix for a GitHub issue. Before naming specific files, reason about the dependency chain:

1. First, identify the PRIMARY module where the bug manifests
2. Then trace UPSTREAM: what modules feed data/config into this module?
3. Then trace DOWNSTREAM: what modules depend on this module's output/behavior?
4. Consider SHARED utilities: are there helper/utils modules that multiple components use?
5. Consider SIDE EFFECTS: if you change the primary module, what tests, configs, or docs break?

## Issue
{problem_statement}

## Repository Structure
{repo_structure_summary}

Now produce your plan. Respond in JSON:
{{"files_to_inspect": [{{"path": "...", "reason": "..."}}], "files_to_modify": [{{"path": "...", "change_type": "edit", "description": "..."}}]}}
"""

# ---------------------------------------------------------------------------
# Utility functions (matching compute_metrics.py)
# ---------------------------------------------------------------------------

def normalize_path(p: str) -> str:
    """Normalize a file path: strip leading ./ or /, collapse separators."""
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


def truncate_repo_structure(repo_structure: str, max_chars: int = 80000) -> str:
    """Truncate repo structure if too large."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def extract_plan_files(plan: dict) -> set:
    """Extract the union of files_to_inspect and files_to_modify paths from a plan."""
    files = set()
    for entry in plan.get("files_to_inspect", []):
        if isinstance(entry, dict) and "path" in entry:
            files.add(entry["path"])
    for entry in plan.get("files_to_modify", []):
        if isinstance(entry, dict) and "path" in entry:
            files.add(entry["path"])
    return files


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def generate_plan_variant(task: dict, prompt_template: str, variant_name: str,
                          index: int, total: int) -> dict:
    """Generate a plan using the given prompt template."""
    task_id = task["task_id"]
    repo_structure = truncate_repo_structure(task["repo_structure_summary"])

    prompt = prompt_template.format(
        problem_statement=task["problem_statement"],
        repo_structure_summary=repo_structure,
    )

    raw_text = None
    max_retries = 1  # retry once then skip

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"       -> Retry {attempt}/{max_retries}...")
                time.sleep(2)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text

            plan = parse_json_response(raw_text)

            # Basic validation
            if "files_to_inspect" not in plan and "files_to_modify" not in plan:
                raise ValueError("Plan missing both files_to_inspect and files_to_modify")

            inspect_count = len(plan.get("files_to_inspect", []))
            modify_count = len(plan.get("files_to_modify", []))
            print(f"  [{index}/{total}] {variant_name} | {task_id}: "
                  f"{inspect_count} inspect, {modify_count} modify")

            return {
                "task_id": task_id,
                "plan": plan,
                "raw_response": raw_text,
            }

        except json.JSONDecodeError as e:
            print(f"  [{index}/{total}] {variant_name} | {task_id}: "
                  f"JSON parse error (attempt {attempt+1}): {e}")
            if raw_text:
                print(f"       -> Preview: {raw_text[:200]}...")
        except anthropic.APIError as e:
            print(f"  [{index}/{total}] {variant_name} | {task_id}: "
                  f"API error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"  [{index}/{total}] {variant_name} | {task_id}: "
                  f"Error (attempt {attempt+1}): {e}")

    # All retries exhausted — skip with warning
    print(f"  WARNING: Skipping {task_id} for variant {variant_name} after all retries")
    return {
        "task_id": task_id,
        "plan": None,
        "raw_response": raw_text,
        "error": "all retries exhausted",
    }


# ---------------------------------------------------------------------------
# Generate plans for a variant
# ---------------------------------------------------------------------------

def generate_all_plans(tasks: list, prompt_template: str, variant_name: str,
                       output_path: str) -> list:
    """Generate plans for all tasks using a given prompt variant, with caching."""
    total = len(tasks)
    print(f"\n{'='*70}")
    print(f"Generating plans for variant: {variant_name}")
    print(f"{'='*70}")

    # Load existing plans to skip already-succeeded tasks
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_list = json.load(f)
        for entry in existing_list:
            if entry.get("plan") is not None:
                existing[entry["task_id"]] = entry
        print(f"Found {len(existing)} existing plans — will skip those")

    results = []
    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        if task_id in existing:
            print(f"  [{i}/{total}] {variant_name} | {task_id}: cached")
            results.append(existing[task_id])
            continue

        result = generate_plan_variant(task, prompt_template, variant_name, i, total)
        results.append(result)

        # 0.5 second delay between API calls
        if i < total:
            time.sleep(0.5)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} plans to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_variant_metrics(tasks: list, plans: list, variant_name: str) -> list:
    """Compute file recall, precision, search reduction for a variant."""
    plan_by_task = {}
    for p in plans:
        if p.get("plan") is not None:
            plan_by_task[p["task_id"]] = p["plan"]

    metrics = []
    for task in tasks:
        task_id = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        total_repo_files = task["total_repo_files"]
        complexity_tier = task["complexity_tier"]

        plan = plan_by_task.get(task_id)
        if plan is None:
            metrics.append({
                "task_id": task_id,
                "complexity_tier": complexity_tier,
                "variant": variant_name,
                "file_recall": 0.0,
                "file_precision": 0.0,
                "search_reduction": 0.0,
                "plan_files_count": 0,
                "ground_truth_count": len(gt_files),
                "skipped": True,
            })
            continue

        plan_files = extract_plan_files(plan)
        matched_gt, matched_plan, missed_gt, extra_plan = match_files(plan_files, gt_files)

        file_recall = len(matched_gt) / len(gt_files) if gt_files else 0.0
        file_precision = len(matched_plan) / len(plan_files) if plan_files else 0.0
        search_reduction = len(plan_files) / total_repo_files if total_repo_files > 0 else 0.0

        metrics.append({
            "task_id": task_id,
            "complexity_tier": complexity_tier,
            "variant": variant_name,
            "file_recall": round(file_recall, 4),
            "file_precision": round(file_precision, 4),
            "search_reduction": round(search_reduction, 6),
            "plan_files_count": len(plan_files),
            "ground_truth_count": len(gt_files),
            "files_matched": sorted(matched_gt),
            "files_missed": sorted(missed_gt),
            "files_extra": sorted(extra_plan),
            "skipped": False,
        })

    return metrics


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def print_comparison_table(all_metrics: dict, tasks: list):
    """Print the comparison table."""
    task_info = {t["task_id"]: t for t in tasks}

    # Build lookup: (task_id, variant) -> metrics
    lookup = {}
    for variant, mlist in all_metrics.items():
        for m in mlist:
            lookup[(m["task_id"], variant)] = m

    print("\n" + "=" * 120)
    print("COMPARISON TABLE: File Recall by Prompt Variant")
    print("=" * 120)
    header = (f"{'Task ID':45s} | {'Tier':15s} | {'Minimal':>10s} | "
              f"{'Dependency':>10s} | {'Structured':>10s} | {'Best':>12s}")
    print(header)
    print("-" * 120)

    for task in tasks:
        tid = task["task_id"]
        tier = task["complexity_tier"]

        recalls = {}
        for vname, vkey in [("minimal", "A_minimal"),
                            ("dependency", "B_dependency"),
                            ("structured", "C_structured")]:
            m = lookup.get((tid, vkey))
            recalls[vname] = m["file_recall"] if m else 0.0

        best = max(recalls, key=recalls.get)
        best_label = {"minimal": "Minimal", "dependency": "Dependency",
                      "structured": "Structured"}[best]

        print(f"{tid:45s} | {tier:15s} | "
              f"{recalls['minimal']:10.4f} | "
              f"{recalls['dependency']:10.4f} | "
              f"{recalls['structured']:10.4f} | "
              f"{best_label:>12s}")

    print("-" * 120)


def print_tier_averages(all_metrics: dict):
    """Print per-tier averages for each variant."""
    tiers = ["localized", "cross-module", "architectural"]
    variants = ["A_minimal", "B_dependency", "C_structured"]
    variant_labels = {"A_minimal": "Minimal", "B_dependency": "Dependency",
                      "C_structured": "Structured"}

    print("\n" + "=" * 100)
    print("PER-TIER AVERAGES")
    print("=" * 100)

    header = f"{'Tier':15s} | {'Metric':12s}"
    for v in variants:
        header += f" | {variant_labels[v]:>12s}"
    print(header)
    print("-" * 100)

    tier_data = {}  # {(tier, variant): [recall_values]}
    precision_data = {}

    for variant, mlist in all_metrics.items():
        for m in mlist:
            tier = m["complexity_tier"]
            key = (tier, variant)
            tier_data.setdefault(key, []).append(m["file_recall"])
            precision_data.setdefault(key, []).append(m["file_precision"])

    for tier in tiers:
        # Recall row
        row = f"{tier:15s} | {'Recall':12s}"
        for v in variants:
            vals = tier_data.get((tier, v), [])
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f" | {avg:12.4f}"
        print(row)

        # Precision row
        row = f"{'':15s} | {'Precision':12s}"
        for v in variants:
            vals = precision_data.get((tier, v), [])
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f" | {avg:12.4f}"
        print(row)
        print()

    # Overall averages
    print("-" * 100)
    row = f"{'OVERALL':15s} | {'Recall':12s}"
    for v in variants:
        all_vals = [m["file_recall"] for m in all_metrics[v]]
        avg = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row += f" | {avg:12.4f}"
    print(row)

    row = f"{'':15s} | {'Precision':12s}"
    for v in variants:
        all_vals = [m["file_precision"] for m in all_metrics[v]]
        avg = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row += f" | {avg:12.4f}"
    print(row)

    return tier_data, precision_data


def create_grouped_bar_chart(all_metrics: dict, output_path: str):
    """Create grouped bar chart comparing 3 variants by tier."""
    tiers = ["localized", "cross-module", "architectural"]
    tier_labels = ["Localized\n(2-3 files)", "Cross-module\n(4-6 files)",
                   "Architectural\n(7+ files)"]
    variants = ["A_minimal", "B_dependency", "C_structured"]
    variant_labels = ["Minimal", "Dependency-tracing", "Structured (current)"]
    colors = ["#4ECDC4", "#FF6B6B", "#45B7D1"]

    # Compute per-tier recall averages
    tier_recalls = {}
    tier_precisions = {}
    for variant, mlist in all_metrics.items():
        for m in mlist:
            tier = m["complexity_tier"]
            tier_recalls.setdefault((tier, variant), []).append(m["file_recall"])
            tier_precisions.setdefault((tier, variant), []).append(m["file_precision"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Recall chart ---
    ax = axes[0]
    x = np.arange(len(tiers))
    width = 0.25

    for i, (variant, label, color) in enumerate(zip(variants, variant_labels, colors)):
        means = []
        stds = []
        for tier in tiers:
            vals = tier_recalls.get((tier, variant), [0.0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        bars = ax.bar(x + i * width, means, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Complexity Tier", fontsize=12)
    ax.set_ylabel("File Recall", fontsize=12)
    ax.set_title("File Recall by Prompt Variant and Complexity Tier", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tier_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # --- Precision chart ---
    ax = axes[1]
    for i, (variant, label, color) in enumerate(zip(variants, variant_labels, colors)):
        means = []
        for tier in tiers:
            vals = tier_precisions.get((tier, variant), [0.0])
            means.append(np.mean(vals))
        bars = ax.bar(x + i * width, means, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Complexity Tier", fontsize=12)
    ax.set_ylabel("File Precision", fontsize=12)
    ax.set_title("File Precision by Prompt Variant and Complexity Tier", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tier_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved chart to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load tasks
    with open("data/tasks.json") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks")

    # Load existing structured plans (Variant C) — do NOT re-run
    with open("data/plans.json") as f:
        structured_plans = json.load(f)
    print(f"Loaded {len(structured_plans)} existing structured plans (Variant C)")

    # Generate Variant A (Minimal) plans
    minimal_plans = generate_all_plans(
        tasks, PROMPT_MINIMAL, "A_minimal", "data/plans_minimal.json"
    )

    # Generate Variant B (Dependency-tracing) plans
    dependency_plans = generate_all_plans(
        tasks, PROMPT_DEPENDENCY, "B_dependency", "data/plans_dependency.json"
    )

    # Compute metrics for all 3 variants
    print("\n" + "=" * 70)
    print("Computing metrics for all variants...")
    print("=" * 70)

    metrics_minimal = compute_variant_metrics(tasks, minimal_plans, "A_minimal")
    metrics_dependency = compute_variant_metrics(tasks, dependency_plans, "B_dependency")
    metrics_structured = compute_variant_metrics(tasks, structured_plans, "C_structured")

    all_metrics = {
        "A_minimal": metrics_minimal,
        "B_dependency": metrics_dependency,
        "C_structured": metrics_structured,
    }

    # Save all metrics
    combined_metrics = []
    for variant, mlist in all_metrics.items():
        combined_metrics.extend(mlist)

    with open("data/prompt_sensitivity_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"Saved metrics to data/prompt_sensitivity_metrics.json")

    # Print comparison table
    print_comparison_table(all_metrics, tasks)

    # Print per-tier averages
    tier_data, precision_data = print_tier_averages(all_metrics)

    # Create grouped bar chart
    os.makedirs("analysis", exist_ok=True)
    create_grouped_bar_chart(all_metrics, "analysis/prompt_sensitivity.png")

    # Key findings summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    variants = ["A_minimal", "B_dependency", "C_structured"]
    variant_labels = {"A_minimal": "Minimal", "B_dependency": "Dependency-tracing",
                      "C_structured": "Structured"}
    tiers = ["localized", "cross-module", "architectural"]

    for tier in tiers:
        best_variant = None
        best_recall = -1
        for v in variants:
            vals = [m["file_recall"] for m in all_metrics[v]
                    if m["complexity_tier"] == tier]
            avg = sum(vals) / len(vals) if vals else 0.0
            if avg > best_recall:
                best_recall = avg
                best_variant = v
        print(f"  {tier:15s}: Best = {variant_labels[best_variant]} "
              f"(avg recall = {best_recall:.4f})")

    # Overall
    best_overall = None
    best_overall_recall = -1
    for v in variants:
        vals = [m["file_recall"] for m in all_metrics[v]]
        avg = sum(vals) / len(vals) if vals else 0.0
        if avg > best_overall_recall:
            best_overall_recall = avg
            best_overall_v = v
            best_overall_recall_val = avg

    print(f"\n  Overall best: {variant_labels[best_overall_v]} "
          f"(avg recall = {best_overall_recall_val:.4f})")

    # Specific questions
    print("\n  Q: Does dependency-tracing help on architectural tasks?")
    arch_dep = [m["file_recall"] for m in all_metrics["B_dependency"]
                if m["complexity_tier"] == "architectural"]
    arch_str = [m["file_recall"] for m in all_metrics["C_structured"]
                if m["complexity_tier"] == "architectural"]
    arch_min = [m["file_recall"] for m in all_metrics["A_minimal"]
                if m["complexity_tier"] == "architectural"]
    avg_dep = sum(arch_dep) / len(arch_dep) if arch_dep else 0.0
    avg_str = sum(arch_str) / len(arch_str) if arch_str else 0.0
    avg_min = sum(arch_min) / len(arch_min) if arch_min else 0.0
    if avg_dep > avg_str:
        print(f"     YES: Dependency ({avg_dep:.4f}) > Structured ({avg_str:.4f}) "
              f"by {avg_dep - avg_str:+.4f}")
    else:
        print(f"     NO: Dependency ({avg_dep:.4f}) <= Structured ({avg_str:.4f}) "
              f"by {avg_dep - avg_str:+.4f}")

    print("\n  Q: Does minimal prompt do well on localized tasks?")
    loc_min = [m["file_recall"] for m in all_metrics["A_minimal"]
               if m["complexity_tier"] == "localized"]
    loc_str = [m["file_recall"] for m in all_metrics["C_structured"]
               if m["complexity_tier"] == "localized"]
    avg_loc_min = sum(loc_min) / len(loc_min) if loc_min else 0.0
    avg_loc_str = sum(loc_str) / len(loc_str) if loc_str else 0.0
    if avg_loc_min >= avg_loc_str - 0.05:
        print(f"     YES: Minimal ({avg_loc_min:.4f}) is competitive with "
              f"Structured ({avg_loc_str:.4f}), diff = {avg_loc_min - avg_loc_str:+.4f}")
    else:
        print(f"     NO: Minimal ({avg_loc_min:.4f}) underperforms "
              f"Structured ({avg_loc_str:.4f}) by {avg_loc_min - avg_loc_str:+.4f}")


if __name__ == "__main__":
    main()
