#!/usr/bin/env python3
"""
Iterative Plan Refinement: Generate critique -> Revise -> Measure improvement.

Takes the 8 worst plans (recall < 0.5) and runs 2 iterations of critique->revise,
measuring whether file recall improves at each stage.

Reads: data/tasks.json, data/plans.json, data/metrics.json
Writes: data/refinement_results.json, analysis/plan_refinement.png
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


# ── Path matching (copied from compute_metrics.py) ──────────────────────────

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


def compute_recall(plan_files: set, gt_files: set) -> float:
    matched_gt, _, _, _ = match_files(plan_files, gt_files)
    return len(matched_gt) / len(gt_files) if gt_files else 0.0


def compute_precision(plan_files: set, gt_files: set) -> float:
    _, matched_plan, _, _ = match_files(plan_files, gt_files)
    return len(matched_plan) / len(plan_files) if plan_files else 0.0


def extract_plan_files(plan: dict) -> set:
    """Extract all file paths from a plan's files_to_inspect and files_to_modify."""
    files = set()
    for entry in plan.get("files_to_inspect", []):
        if isinstance(entry, dict) and "path" in entry:
            files.add(entry["path"])
        elif isinstance(entry, str):
            files.add(entry)
    for entry in plan.get("files_to_modify", []):
        if isinstance(entry, dict) and "path" in entry:
            files.add(entry["path"])
        elif isinstance(entry, str):
            files.add(entry)
    return files


# ── JSON Parsing ─────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = 30000) -> str:
    """Truncate repo structure to ~30K chars as specified."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


# ── Critique & Revise Prompts ────────────────────────────────────────────────

CRITIQUE_PROMPT = """\
You are reviewing a coding plan for a GitHub issue fix. Identify weaknesses and missing files.

## Issue
{problem_statement}

## Current Plan
{plan_json}

## Repository Structure
{repo_structure}

Your critique should:
1. Identify files the plan might be missing (name specific paths from the repository structure)
2. Flag if the scope seems too narrow or too broad
3. Suggest specific improvements

Respond in JSON:
{{"concerns": ["concern 1", "concern 2"], "missing_files": ["path/to/file.py"], "suggested_changes": ["change 1", "change 2"]}}
"""

REVISE_PROMPT = """\
Revise this plan based on the critique.

## Issue
{problem_statement}

## Original Plan
{plan_json}

## Critique
{critique_json}

## Repository Structure
{repo_structure}

Produce a revised plan in the same JSON format as the original:
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


# ── API Call Helpers ─────────────────────────────────────────────────────────

def call_critique(problem_statement: str, plan: dict, repo_structure: str, max_retries: int = 2) -> dict:
    """Call Claude to critique a plan."""
    prompt = CRITIQUE_PROMPT.format(
        problem_statement=problem_statement,
        plan_json=json.dumps(plan, indent=2),
        repo_structure=repo_structure,
    )

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"       -> Critique retry {attempt}/{max_retries}...")
                time.sleep(2)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            critique = parse_json_response(raw_text)
            return critique

        except (json.JSONDecodeError, anthropic.APIError) as e:
            print(f"       -> Critique error attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise

    raise RuntimeError("Should not reach here")


def call_revise(problem_statement: str, plan: dict, critique: dict, repo_structure: str, max_retries: int = 2) -> dict:
    """Call Claude to revise a plan based on critique."""
    prompt = REVISE_PROMPT.format(
        problem_statement=problem_statement,
        plan_json=json.dumps(plan, indent=2),
        critique_json=json.dumps(critique, indent=2),
        repo_structure=repo_structure,
    )

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"       -> Revise retry {attempt}/{max_retries}...")
                time.sleep(2)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            revised_plan = parse_json_response(raw_text)

            # Validate required keys
            for key in ("files_to_inspect", "files_to_modify", "implementation_steps"):
                if key not in revised_plan:
                    raise ValueError(f"Revised plan missing key: {key}")

            return revised_plan

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            print(f"       -> Revise error attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise

    raise RuntimeError("Should not reach here")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    with open("data/tasks.json") as f:
        tasks = json.load(f)
    task_by_id = {t["task_id"]: t for t in tasks}

    with open("data/plans.json") as f:
        plans = json.load(f)
    plan_by_id = {p["task_id"]: p["plan"] for p in plans}

    with open("data/metrics.json") as f:
        metrics = json.load(f)

    # Find the 8 worst plans (recall < 0.5)
    worst = [m for m in metrics if m["plan_file_recall"] < 0.5]
    worst.sort(key=lambda x: x["plan_file_recall"])
    # Take up to 8
    worst = worst[:8]

    print(f"Found {len(worst)} plans with recall < 0.5:")
    for m in worst:
        print(f"  {m['task_id']}: recall={m['plan_file_recall']:.3f}")
    print()

    # Run refinement loop
    results = []
    total = len(worst)

    for idx, metric_entry in enumerate(worst, 1):
        task_id = metric_entry["task_id"]
        task = task_by_id[task_id]
        original_plan = plan_by_id[task_id]
        gt_files = set(task["ground_truth_files_changed"])
        repo_structure = truncate_repo_structure(task["repo_structure_summary"])

        # Stage 0: original metrics
        original_files = extract_plan_files(original_plan)
        original_recall = compute_recall(original_files, gt_files)
        original_precision = compute_precision(original_files, gt_files)

        print(f"[{idx}/{total}] {task_id}")
        print(f"  Stage 0 (original): recall={original_recall:.3f}, precision={original_precision:.3f}, files={len(original_files)}")

        result = {
            "task_id": task_id,
            "complexity_tier": metric_entry["complexity_tier"],
            "ground_truth_count": metric_entry["ground_truth_count"],
            "stages": [
                {
                    "stage": 0,
                    "recall": round(original_recall, 4),
                    "precision": round(original_precision, 4),
                    "plan_files_count": len(original_files),
                    "plan_files": sorted(original_files),
                }
            ],
            "critiques": [],
            "revised_plans": [],
        }

        current_plan = original_plan

        for iteration in range(1, 3):  # Iterations 1 and 2
            print(f"  --- Iteration {iteration}: Critique ---")

            try:
                # Critique
                critique = call_critique(
                    task["problem_statement"],
                    current_plan,
                    repo_structure,
                )
                n_concerns = len(critique.get("concerns", []))
                n_missing = len(critique.get("missing_files", []))
                n_changes = len(critique.get("suggested_changes", []))
                print(f"    Critique: {n_concerns} concerns, {n_missing} missing files, {n_changes} suggested changes")

                result["critiques"].append(critique)

                time.sleep(1)

                # Revise
                print(f"  --- Iteration {iteration}: Revise ---")
                revised_plan = call_revise(
                    task["problem_statement"],
                    current_plan,
                    critique,
                    repo_structure,
                )

                revised_files = extract_plan_files(revised_plan)
                revised_recall = compute_recall(revised_files, gt_files)
                revised_precision = compute_precision(revised_files, gt_files)

                print(f"  Stage {iteration} (revised): recall={revised_recall:.3f}, precision={revised_precision:.3f}, files={len(revised_files)}")

                result["stages"].append({
                    "stage": iteration,
                    "recall": round(revised_recall, 4),
                    "precision": round(revised_precision, 4),
                    "plan_files_count": len(revised_files),
                    "plan_files": sorted(revised_files),
                })
                result["revised_plans"].append(revised_plan)

                current_plan = revised_plan

                time.sleep(1)

            except Exception as e:
                print(f"    ERROR in iteration {iteration}: {e}")
                traceback.print_exc()
                # If an iteration fails, carry forward the last known stage
                last_stage = result["stages"][-1]
                result["stages"].append({
                    "stage": iteration,
                    "recall": last_stage["recall"],
                    "precision": last_stage["precision"],
                    "plan_files_count": last_stage["plan_files_count"],
                    "plan_files": last_stage["plan_files"],
                    "error": str(e),
                })
                break  # Stop further iterations for this task

        # Print improvement summary for this task
        s0 = result["stages"][0]["recall"]
        s_last = result["stages"][-1]["recall"]
        delta = s_last - s0
        print(f"  => Recall change: {s0:.3f} -> {s_last:.3f} (delta={delta:+.3f})")
        print()

        results.append(result)

    # ── Save Results ─────────────────────────────────────────────────────────

    with open("data/refinement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved data/refinement_results.json with {len(results)} entries.")

    # ── Summary Table ────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print(f"{'Task':<40s} | {'Original':>8s} | {'Stage 1':>8s} | {'Stage 2':>8s} | {'Improvement':>11s}")
    print("-" * 90)

    total_improvement = 0
    improved_count = 0
    degraded_count = 0

    for r in results:
        task_id = r["task_id"]
        stages = r["stages"]
        s0 = stages[0]["recall"]
        s1 = stages[1]["recall"] if len(stages) > 1 else s0
        s2 = stages[2]["recall"] if len(stages) > 2 else s1
        delta = s2 - s0
        total_improvement += delta
        if delta > 0:
            improved_count += 1
        elif delta < 0:
            degraded_count += 1

        improvement_str = f"{delta:+.3f}"
        if delta > 0:
            improvement_str += " (+)"
        elif delta < 0:
            improvement_str += " (-)"

        print(f"{task_id:<40s} | {s0:>8.3f} | {s1:>8.3f} | {s2:>8.3f} | {improvement_str:>11s}")

    print("-" * 90)
    avg_improvement = total_improvement / len(results) if results else 0
    print(f"{'Average':<40s} | {'':>8s} | {'':>8s} | {'':>8s} | {avg_improvement:>+8.3f}")
    print(f"\nImproved: {improved_count}/{len(results)}, Degraded: {degraded_count}/{len(results)}, Unchanged: {len(results) - improved_count - degraded_count}/{len(results)}")

    # ── Plot ─────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        stages = r["stages"]
        xs = [s["stage"] for s in stages]
        ys = [s["recall"] for s in stages]
        short_id = r["task_id"].split("__")[-1]  # e.g., "django-10989"
        ax.plot(xs, ys, "o-", color=colors[i % len(colors)], linewidth=2,
                markersize=8, label=short_id, alpha=0.85)

    # Average line
    avg_by_stage = {}
    for r in results:
        for s in r["stages"]:
            stage = s["stage"]
            avg_by_stage.setdefault(stage, []).append(s["recall"])
    avg_xs = sorted(avg_by_stage.keys())
    avg_ys = [sum(avg_by_stage[s]) / len(avg_by_stage[s]) for s in avg_xs]
    ax.plot(avg_xs, avg_ys, "k--", linewidth=3, markersize=10, marker="s",
            label="Average", zorder=10)

    ax.set_xlabel("Refinement Stage", fontsize=13)
    ax.set_ylabel("File Recall", fontsize=13)
    ax.set_title("Plan Refinement: Critique-Revise Loop\n(8 worst plans, 2 iterations)", fontsize=14)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Original", "After 1st Revision", "After 2nd Revision"])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis/plan_refinement.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved analysis/plan_refinement.png")
    plt.close()


if __name__ == "__main__":
    main()
