"""
Workstream 3, Script 1: Compute file recall, precision, and search-space reduction
for each plan against ground truth.

Reads: data/tasks.json, data/plans.json
Writes: data/metrics.json (plan metrics; baseline fields added by run_baseline.py)
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
from pathlib import PurePosixPath


def normalize_path(p: str) -> str:
    """Normalize a file path: strip leading ./ or /, lowercase, collapse separators."""
    p = p.strip()
    # Remove leading ./ or /
    while p.startswith("./") or p.startswith("/"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def paths_match(a: str, b: str) -> bool:
    """
    Check if two paths refer to the same file using suffix matching.
    Returns True if:
      - They are identical after normalization
      - One is a suffix of the other (i.e., one ends with '/' + the other)
    """
    na = normalize_path(a)
    nb = normalize_path(b)
    if na == nb:
        return True
    # Suffix matching: check if one path ends with the other
    if na.endswith("/" + nb) or nb.endswith("/" + na):
        return True
    # Also check if the filename components match when one is shorter
    # e.g., "src/utils/helpers.py" vs "utils/helpers.py"
    if na.endswith(nb) or nb.endswith(na):
        # Make sure the match is on a path boundary
        longer, shorter = (na, nb) if len(na) >= len(nb) else (nb, na)
        idx = longer.find(shorter)
        if idx == 0 or (idx > 0 and longer[idx - 1] == "/"):
            return True
    return False


def match_files(plan_files: set, gt_files: set):
    """
    Match plan files against ground truth files using normalized path matching.
    Returns:
      - matched_gt: set of GT files that were matched
      - matched_plan: set of plan files that matched a GT file
      - missed_gt: GT files not matched by any plan file
      - extra_plan: plan files that didn't match any GT file
    """
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


def main():
    with open("data/tasks.json") as f:
        tasks = json.load(f)

    with open("data/plans.json") as f:
        plans = json.load(f)

    # Index plans by task_id
    plan_by_task = {p["task_id"]: p["plan"] for p in plans}

    metrics = []

    for task in tasks:
        task_id = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        total_repo_files = task["total_repo_files"]
        complexity_tier = task["complexity_tier"]

        plan = plan_by_task.get(task_id)
        if plan is None:
            print(f"WARNING: No plan found for task {task_id}, skipping")
            continue

        # Extract plan files: union of files_to_inspect + files_to_modify paths
        plan_files = set()
        for entry in plan.get("files_to_inspect", []):
            plan_files.add(entry["path"])
        for entry in plan.get("files_to_modify", []):
            plan_files.add(entry["path"])

        # Compute matches using normalized path matching
        matched_gt, matched_plan, missed_gt, extra_plan = match_files(plan_files, gt_files)

        # Metrics
        file_recall = len(matched_gt) / len(gt_files) if gt_files else 0.0
        file_precision = len(matched_plan) / len(plan_files) if plan_files else 0.0
        search_reduction = len(plan_files) / total_repo_files if total_repo_files > 0 else 0.0

        entry = {
            "task_id": task_id,
            "complexity_tier": complexity_tier,
            "total_repo_files": total_repo_files,
            "ground_truth_count": len(gt_files),
            "plan_files_count": len(plan_files),
            "plan_file_recall": round(file_recall, 4),
            "plan_file_precision": round(file_precision, 4),
            "plan_search_reduction": round(search_reduction, 6),
            "plan_files_matched": sorted(matched_gt),
            "plan_files_missed": sorted(missed_gt),
            "plan_files_extra": sorted(extra_plan),
        }

        metrics.append(entry)

        print(f"{task_id:45s}  tier={complexity_tier:15s}  "
              f"recall={file_recall:.2f}  precision={file_precision:.2f}  "
              f"reduction={search_reduction:.4f}  "
              f"matched={len(matched_gt)}/{len(gt_files)}  "
              f"plan_size={len(plan_files)}")

    # Write metrics (baseline fields will be added by run_baseline.py)
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote data/metrics.json with {len(metrics)} entries (plan metrics only).")

    # Summary stats
    avg_recall = sum(m["plan_file_recall"] for m in metrics) / len(metrics)
    avg_precision = sum(m["plan_file_precision"] for m in metrics) / len(metrics)
    avg_reduction = sum(m["plan_search_reduction"] for m in metrics) / len(metrics)
    print(f"\nAverages: recall={avg_recall:.3f}  precision={avg_precision:.3f}  "
          f"search_reduction={avg_reduction:.4f}")


if __name__ == "__main__":
    main()
