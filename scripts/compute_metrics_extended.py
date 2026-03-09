#!/usr/bin/env python3
"""
Extended Metrics Computation (plan + baseline B in one pass)

Reads data/tasks_extended.json + data/plans_extended.json
Writes data/metrics_extended.json

Computes plan metrics AND keyword-baseline metrics together.
Works on both pilot data and extended data via CLI flags.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import argparse


# ---------------------------------------------------------------------------
# Common tokens to filter out from keyword baseline (too generic)
# ---------------------------------------------------------------------------
STOP_TOKENS = {
    "test", "tests", "self", "class", "def", "import", "from", "return",
    "error", "file", "files", "data", "type", "name", "value", "list",
    "dict", "none", "true", "false", "init", "main", "base", "core",
    "utils", "util", "helper", "helpers", "config", "settings", "setup",
    "module", "modules", "package", "model", "models", "view", "views",
    "with", "that", "this", "have", "will", "should", "would", "could",
    "when", "what", "which", "where", "there", "their", "they", "them",
    "been", "being", "some", "than", "then", "also", "just", "only",
    "other", "into", "over", "after", "before", "between", "under",
    "about", "each", "make", "like", "does", "doing", "done", "more",
    "much", "many", "most", "such", "very", "same", "even", "still",
    "back", "well", "here", "thing", "things", "work", "works", "working",
    "because", "since", "while", "through", "using", "used", "uses",
    "python", "code", "method", "function", "attribute", "object",
    "string", "number", "print", "call", "args", "kwargs", "param",
    "default", "example", "issue", "problem", "patch", "change",
    "changes", "added", "removed", "fixed", "need", "needs", "want",
    "case", "result", "expected", "actual", "version", "current",
    "new", "old", "original", "update", "updated", "instead",
    "docs", "doc", "src", "lib", "bin", "conf", "locale", "templates",
    "static", "contrib", "compat", "internal", "extern", "third_party",
    "vendor",
}


# ---------------------------------------------------------------------------
# Path normalization and matching (same as pilot)
# ---------------------------------------------------------------------------
def normalize_path(p: str) -> str:
    """Normalize a file path: strip leading ./ or /, collapse separators."""
    p = p.strip()
    while p.startswith("./") or p.startswith("/"):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def paths_match(a: str, b: str) -> bool:
    """
    Check if two paths refer to the same file using suffix matching.
    Returns True if:
      - They are identical after normalization
      - One is a suffix of the other on a path boundary
    """
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


def match_files(candidate_files: set, gt_files: set):
    """
    Match candidate files against ground truth using normalized path matching.
    Returns (matched_gt, matched_candidate, missed_gt, extra_candidate).
    """
    matched_gt = set()
    matched_candidate = set()

    for gf in gt_files:
        for cf in candidate_files:
            if paths_match(gf, cf):
                matched_gt.add(gf)
                matched_candidate.add(cf)

    missed_gt = gt_files - matched_gt
    extra_candidate = candidate_files - matched_candidate
    return matched_gt, matched_candidate, missed_gt, extra_candidate


# ---------------------------------------------------------------------------
# Keyword baseline: extract tokens from problem statement, match to files
# ---------------------------------------------------------------------------
def extract_tokens(text: str) -> set:
    """
    Extract identifier-like tokens from problem statement text.
    Looks for file paths, dot-separated paths, CamelCase, snake_case, etc.
    """
    tokens = set()

    # 1. File-path-like strings
    file_paths = re.findall(r'[\w]+(?:/[\w]+)+(?:\.[\w]+)?', text)
    for fp in file_paths:
        tokens.add(fp)
        for part in fp.split('/'):
            if len(part) > 3 and part.lower() not in STOP_TOKENS:
                tokens.add(part)

    # 2. Dot-separated identifiers
    dot_paths = re.findall(r'[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+', text)
    for dp in dot_paths:
        tokens.add(dp)
        slash_path = dp.replace('.', '/')
        tokens.add(slash_path)
        for part in dp.split('.'):
            if len(part) > 3 and part.lower() not in STOP_TOKENS:
                tokens.add(part)

    # 3. CamelCase words
    camel_words = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', text)
    for cw in camel_words:
        if len(cw) > 3 and cw.lower() not in STOP_TOKENS:
            tokens.add(cw)

    # 4. snake_case words
    snake_words = re.findall(r'[a-z]+(?:_[a-z]+)+', text)
    for sw in snake_words:
        if len(sw) > 3 and sw.lower() not in STOP_TOKENS:
            tokens.add(sw)

    # 5. General Python identifiers (code-like, longer than 5 chars)
    identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
    for ident in identifiers:
        if len(ident) > 5 and ident.lower() not in STOP_TOKENS:
            if ('_' in ident or
                    re.match(r'[A-Z][a-z]+[A-Z]', ident) or
                    ident.isupper()):
                tokens.add(ident)

    # Filter out stop tokens
    filtered = set()
    for t in tokens:
        if t.lower() not in STOP_TOKENS and len(t) > 2:
            filtered.add(t)

    return filtered


def match_token_to_files(token: str, file_paths: list) -> list:
    """Match a token against a list of file paths using substring matching."""
    matches = []
    token_lower = token.lower()
    for fp in file_paths:
        fp_lower = fp.lower()
        if token_lower in fp_lower:
            matches.append(fp)
        elif '.' in token and '/' not in token:
            slash_token = token.replace('.', '/').lower()
            if slash_token in fp_lower:
                matches.append(fp)
    return matches


def compute_baseline_files(task: dict) -> set:
    """Compute baseline B file set from keyword matching."""
    problem_statement = task["problem_statement"]
    repo_structure = task["repo_structure_summary"]

    repo_files = [line.strip() for line in repo_structure.strip().split('\n')
                  if line.strip()]

    tokens = extract_tokens(problem_statement)

    baseline_files = set()
    for token in tokens:
        matched = match_token_to_files(token, repo_files)
        # Skip tokens that match too many files (too generic)
        if len(matched) > 30:
            continue
        baseline_files.update(matched)

    return baseline_files


# ---------------------------------------------------------------------------
# Plan file extraction
# ---------------------------------------------------------------------------
def extract_plan_files(plan: dict) -> set:
    """Extract all file paths mentioned in a plan (inspect + modify)."""
    plan_files = set()
    for entry in plan.get("files_to_inspect", []):
        plan_files.add(entry["path"])
    for entry in plan.get("files_to_modify", []):
        plan_files.add(entry["path"])
    return plan_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute metrics for plans and baseline")
    parser.add_argument("--tasks", default="data/tasks_extended.json",
                        help="Path to tasks JSON file")
    parser.add_argument("--plans", default="data/plans_extended.json",
                        help="Path to plans JSON file")
    parser.add_argument("--output", default="data/metrics_extended.json",
                        help="Path to output metrics JSON file")
    args = parser.parse_args()

    # Load tasks
    if not os.path.exists(args.tasks):
        print(f"ERROR: {args.tasks} not found.")
        return
    with open(args.tasks) as f:
        tasks = json.load(f)

    # Load plans
    if not os.path.exists(args.plans):
        print(f"ERROR: {args.plans} not found.")
        return
    with open(args.plans) as f:
        plans = json.load(f)

    plan_by_task = {p["task_id"]: p.get("plan") for p in plans}

    print(f"Loaded {len(tasks)} tasks from {args.tasks}")
    print(f"Loaded {len(plans)} plans from {args.plans}")
    print()

    metrics = []
    skipped = 0

    for task in tasks:
        task_id = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        total_repo_files = task["total_repo_files"]
        complexity_tier = task["complexity_tier"]

        plan = plan_by_task.get(task_id)
        if plan is None:
            print(f"  WARNING: No plan found for {task_id}, skipping")
            skipped += 1
            continue

        # --- Plan metrics ---
        plan_files = extract_plan_files(plan)
        p_matched_gt, p_matched_plan, p_missed_gt, p_extra_plan = match_files(plan_files, gt_files)

        plan_recall = len(p_matched_gt) / len(gt_files) if gt_files else 0.0
        plan_precision = len(p_matched_plan) / len(plan_files) if plan_files else 0.0
        plan_search_reduction = len(plan_files) / total_repo_files if total_repo_files > 0 else 0.0

        # --- Baseline B metrics ---
        baseline_files = compute_baseline_files(task)
        b_matched_gt, b_matched_base, b_missed_gt, b_extra_base = match_files(baseline_files, gt_files)

        baseline_recall = len(b_matched_gt) / len(gt_files) if gt_files else 0.0
        baseline_precision = len(b_matched_base) / len(baseline_files) if baseline_files else 0.0
        baseline_search_reduction = len(baseline_files) / total_repo_files if total_repo_files > 0 else 0.0

        entry = {
            "task_id": task_id,
            "complexity_tier": complexity_tier,
            "total_repo_files": total_repo_files,
            "ground_truth_count": len(gt_files),
            # Plan metrics
            "plan_files_count": len(plan_files),
            "plan_file_recall": round(plan_recall, 4),
            "plan_file_precision": round(plan_precision, 4),
            "plan_search_reduction": round(plan_search_reduction, 6),
            "plan_files_matched": sorted(p_matched_gt),
            "plan_files_missed": sorted(p_missed_gt),
            "plan_files_extra": sorted(p_extra_plan),
            # Baseline B metrics
            "baseline_b_file_recall": round(baseline_recall, 4),
            "baseline_b_file_precision": round(baseline_precision, 4),
            "baseline_b_search_reduction": round(baseline_search_reduction, 6),
            "baseline_b_files_matched": sorted(b_matched_gt),
            "baseline_b_files_missed": sorted(b_missed_gt),
            "baseline_b_files_extra": sorted(b_extra_base),
            "baseline_b_files_count": len(baseline_files),
        }

        metrics.append(entry)

        print(f"  {task_id:45s}  tier={complexity_tier:15s}  "
              f"plan_recall={plan_recall:.2f}  base_recall={baseline_recall:.2f}  "
              f"delta={plan_recall - baseline_recall:+.2f}  "
              f"matched={len(p_matched_gt)}/{len(gt_files)}")

    # Write metrics
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {args.output} with {len(metrics)} entries ({skipped} skipped).")

    # Summary stats
    if metrics:
        avg_plan_recall = sum(m["plan_file_recall"] for m in metrics) / len(metrics)
        avg_plan_precision = sum(m["plan_file_precision"] for m in metrics) / len(metrics)
        avg_base_recall = sum(m["baseline_b_file_recall"] for m in metrics) / len(metrics)
        avg_base_precision = sum(m["baseline_b_file_precision"] for m in metrics) / len(metrics)
        print(f"\nPlan averages:     recall={avg_plan_recall:.3f}  precision={avg_plan_precision:.3f}")
        print(f"Baseline averages: recall={avg_base_recall:.3f}  precision={avg_base_precision:.3f}")
        print(f"Recall delta:      {avg_plan_recall - avg_base_recall:+.3f}")


if __name__ == "__main__":
    main()
