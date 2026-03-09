"""
SWE-bench Test Suite Evaluation

Takes generated diffs from plan-vs-noplan experiments and runs them through
the SWE-bench Docker-based evaluation harness. This is the gold-standard
evaluation: does the generated patch actually fix the bug?

Usage:
    python scripts/swebench_eval.py [--condition with_plan|no_plan|both] [--max-tasks N]
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import argparse
import tempfile
from collections import defaultdict
from datasets import load_dataset


def load_swebench_metadata():
    """Load SWE-bench Verified metadata for test info."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    meta = {}
    for item in ds:
        meta[item["instance_id"]] = {
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "version": item.get("version", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
            "environment_setup_commit": item.get("environment_setup_commit", ""),
        }
    # Also load full SWE-bench for tasks not in Verified
    ds_full = load_dataset("princeton-nlp/SWE-bench", split="test")
    for item in ds_full:
        if item["instance_id"] not in meta:
            meta[item["instance_id"]] = {
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "version": item.get("version", ""),
                "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
                "environment_setup_commit": item.get("environment_setup_commit", ""),
            }
    return meta


def extract_diff_from_response(raw_response):
    """Extract a unified diff from a model's raw response.

    The model may produce diffs in various formats:
    - ```diff ... ``` code blocks
    - --- a/file ... +++ b/file raw diffs
    - Mixed text and diffs

    We extract all diff-like sections and concatenate them.
    """
    import re

    if not raw_response:
        return ""

    lines = raw_response.split('\n')
    diff_lines = []
    in_diff = False
    in_code_block = False

    for line in lines:
        # Track code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                in_code_block = False
                in_diff = False
            else:
                in_code_block = True
                # Check if it's a diff code block
                if 'diff' in line.lower():
                    in_diff = True
            continue

        # Inside a diff code block, take everything
        if in_code_block and in_diff:
            diff_lines.append(line)
            continue

        # Outside code blocks, look for diff markers
        if line.startswith('diff --git '):
            in_diff = True
            diff_lines.append(line)
        elif line.startswith('--- a/') or line.startswith('+++ b/'):
            in_diff = True
            diff_lines.append(line)
        elif line.startswith('@@') and in_diff:
            diff_lines.append(line)
        elif in_diff and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
            diff_lines.append(line)
        elif in_diff and line.strip() == '':
            diff_lines.append(line)
        elif in_diff and not line.startswith('+') and not line.startswith('-') and not line.startswith(' ') and not line.startswith('@@'):
            # End of diff section
            if not in_code_block:
                in_diff = False

    return '\n'.join(diff_lines).strip()


def format_predictions(results, condition, swebench_meta):
    """Format generated diffs as SWE-bench predictions JSONL."""
    predictions = []
    skipped = 0

    for r in results:
        if r.get("condition") != condition:
            continue

        task_id = r["task_id"]
        if task_id not in swebench_meta:
            print(f"  WARNING: {task_id} not found in SWE-bench metadata, skipping")
            skipped += 1
            continue

        raw = r.get("raw_response", "")
        patch = extract_diff_from_response(raw)

        if not patch:
            print(f"  WARNING: {task_id} ({condition}) has no extractable diff")
            skipped += 1
            continue

        predictions.append({
            "instance_id": task_id,
            "model_patch": patch,
            "model_name_or_path": f"plan_eval_{condition}",
        })

    print(f"  Formatted {len(predictions)} predictions, skipped {skipped}")
    return predictions


def run_swebench_eval(predictions, dataset_name, run_id, max_workers=2, timeout=300):
    """Run SWE-bench evaluation harness."""
    from swebench.harness.run_evaluation import main as run_eval

    # Write predictions to temp JSONL file
    pred_path = os.path.join("/scr/clding/plan_eval/data", f"swebench_preds_{run_id}.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    instance_ids = [p["instance_id"] for p in predictions]

    print(f"\n  Running SWE-bench eval on {len(predictions)} instances...")
    print(f"  Predictions file: {pred_path}")
    print(f"  Run ID: {run_id}")
    print(f"  This uses Docker and may take several minutes per instance.\n")

    try:
        run_eval(
            dataset_name=dataset_name,
            split="test",
            instance_ids=instance_ids,
            predictions_path=pred_path,
            max_workers=max_workers,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout,
            namespace=None,
            rewrite_reports=False,
            modal=False,
            report_dir="/scr/clding/plan_eval/data/swebench_reports",
        )
    except Exception as e:
        print(f"  SWE-bench eval error: {e}")
        return None

    # Read results
    report_path = os.path.join(
        "/scr/clding/plan_eval/data/swebench_reports",
        run_id,
        f"{run_id}.{run_id}.json"
    )

    # Try to find the report file
    report_dir = os.path.join("/scr/clding/plan_eval/data/swebench_reports", run_id)
    if os.path.exists(report_dir):
        for fname in os.listdir(report_dir):
            if fname.endswith(".json"):
                report_path = os.path.join(report_dir, fname)
                break

    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    else:
        print(f"  Report not found at {report_path}")
        # Check what files exist
        if os.path.exists(report_dir):
            print(f"  Files in report dir: {os.listdir(report_dir)}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="both", choices=["with_plan", "no_plan", "both"])
    parser.add_argument("--max-tasks", type=int, default=5, help="Max tasks to evaluate (Docker eval is slow)")
    parser.add_argument("--source", default="data/plan_vs_noplan_extended.json", help="Source results file")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("SWE-BENCH TEST SUITE EVALUATION")
    print("=" * 60)

    # Load generated results
    print(f"\nLoading results from {args.source}...")
    with open(args.source) as f:
        results = json.load(f)
    print(f"  {len(results)} entries")

    # Load SWE-bench metadata
    print("Loading SWE-bench metadata...")
    swebench_meta = load_swebench_metadata()
    print(f"  {len(swebench_meta)} instances")

    # Determine which conditions to evaluate
    conditions = ["with_plan", "no_plan"] if args.condition == "both" else [args.condition]

    # Get unique task IDs that exist in SWE-bench
    task_ids = sorted(set(r["task_id"] for r in results if r["task_id"] in swebench_meta))
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
    print(f"\nWill evaluate {len(task_ids)} tasks: {task_ids}")

    # Filter results to selected tasks
    filtered = [r for r in results if r["task_id"] in task_ids]

    all_eval_results = {}

    for condition in conditions:
        print(f"\n{'='*40}")
        print(f"CONDITION: {condition}")
        print(f"{'='*40}")

        # Use full SWE-bench (superset of Verified)
        dataset_name = "princeton-nlp/SWE-bench"

        predictions = format_predictions(filtered, condition, swebench_meta)

        if not predictions:
            print(f"  No valid predictions for {condition}, skipping")
            continue

        run_id = f"plan_eval_{condition}"
        eval_result = run_swebench_eval(
            predictions, dataset_name, run_id,
            max_workers=args.max_workers,
            timeout=args.timeout
        )

        if eval_result:
            all_eval_results[condition] = eval_result
            print(f"\n  Results for {condition}:")
            print(f"  {json.dumps(eval_result, indent=2)[:500]}")

    # Save combined results
    output_path = "data/swebench_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Summary comparison
    if len(all_eval_results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: with_plan vs no_plan")
        print(f"{'='*60}")
        for condition, results in all_eval_results.items():
            resolved = results.get("resolved", results.get("resolved_instances", []))
            total = results.get("total", results.get("total_instances", 0))
            if isinstance(resolved, list):
                n_resolved = len(resolved)
            else:
                n_resolved = resolved
            print(f"  {condition}: {n_resolved}/{total} tests pass")


if __name__ == "__main__":
    main()
