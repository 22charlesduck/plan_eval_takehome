#!/usr/bin/env python3
"""
Plan Critic Prototype

For each of the 14 tasks, calls the Claude API with a critic prompt that
evaluates the plan's quality. Then analyzes whether the critic correctly
identifies plans that actually failed vs succeeded.

This prototypes a pre-execution plan quality gate that Cognition could ship.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time

from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# Threshold: plans with recall below this are considered "bad"
BAD_PLAN_THRESHOLD = 0.5

CRITIC_PROMPT_TEMPLATE = """\
You are reviewing a coding plan before an agent executes it. Your job is to identify potential weaknesses, missing files, or blind spots.

## The Issue
{problem_statement}

## The Plan
{plan_json}

## Repository Structure (partial)
{repo_structure_summary}

## Your Review

Analyze this plan critically. Consider:
1. Does the plan account for ALL files that might need changes, including shared utilities, __init__.py files, and downstream dependencies?
2. Is the scope appropriate? Could this be a codebase-wide issue that the plan treats as localized?
3. Are there files in the repo structure that the plan should have included but didn't?
4. Does the plan consider test files, configuration files, and documentation that might need updates?

Respond in JSON:
{{
  "verdict": "approve|flag|reject",
  "confidence": 0.0-1.0,
  "concerns": ["specific concern about the plan"],
  "missing_files_suggested": ["path/to/file.py that the plan might be missing"],
  "scope_assessment": "too_narrow|appropriate|too_broad",
  "key_risk": "the single biggest risk with this plan"
}}
"""


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = 30000) -> str:
    """Truncate repo structure to 30K chars as specified."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def run_critic(task: dict, plan_entry: dict, index: int, total: int, max_retries: int = 2) -> dict:
    """Run the critic on a single task's plan via the Anthropic API."""
    task_id = task["task_id"]
    print(f"[{index}/{total}] Critiquing plan for {task_id}...")

    repo_structure = truncate_repo_structure(task["repo_structure_summary"])
    plan_json_str = json.dumps(plan_entry["plan"], indent=2)

    prompt = CRITIC_PROMPT_TEMPLATE.format(
        problem_statement=task["problem_statement"],
        plan_json=plan_json_str,
        repo_structure_summary=repo_structure,
    )

    raw_text = None
    last_error = None

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

            evaluation = parse_json_response(raw_text)

            # Validate required keys
            for key in ("verdict", "confidence", "concerns", "missing_files_suggested",
                        "scope_assessment", "key_risk"):
                if key not in evaluation:
                    raise ValueError(f"Critic response missing required key: {key}")

            # Normalize verdict
            evaluation["verdict"] = evaluation["verdict"].lower().strip()
            if evaluation["verdict"] not in ("approve", "flag", "reject"):
                print(f"       -> WARNING: unexpected verdict '{evaluation['verdict']}', treating as 'flag'")
                evaluation["verdict"] = "flag"

            print(f"       -> Verdict: {evaluation['verdict']}, Confidence: {evaluation['confidence']}, "
                  f"Scope: {evaluation['scope_assessment']}, "
                  f"Suggested missing: {len(evaluation['missing_files_suggested'])} files")

            return {
                "task_id": task_id,
                "evaluation": evaluation,
                "raw_response": raw_text,
            }

        except json.JSONDecodeError as e:
            last_error = e
            print(f"       -> JSON parse error on attempt {attempt + 1}: {e}")
            if raw_text:
                print(f"       -> Raw response preview: {raw_text[:200]}...")
        except anthropic.APIError as e:
            last_error = e
            print(f"       -> API error on attempt {attempt + 1}: {e}")

    # All retries exhausted — return a fallback
    print(f"       -> FINAL ERROR for {task_id}: {last_error}")
    return {
        "task_id": task_id,
        "evaluation": {
            "verdict": "flag",
            "confidence": 0.0,
            "concerns": [f"Critic failed: {last_error}"],
            "missing_files_suggested": [],
            "scope_assessment": "unknown",
            "key_risk": "Critic evaluation failed",
        },
        "raw_response": raw_text,
        "error": str(last_error),
    }


def normalize_path(path: str) -> str:
    """Normalize a file path for comparison (strip leading ./ or /)."""
    path = path.strip()
    while path.startswith("./") or path.startswith("/"):
        path = path[1:] if path.startswith("/") else path[2:]
    return path


def compute_analysis(critic_results: list, metrics: list, tasks: list) -> dict:
    """Compute critic accuracy analysis."""
    # Build lookup dicts
    metrics_by_id = {m["task_id"]: m for m in metrics}
    tasks_by_id = {t["task_id"]: t for t in tasks}

    # Classify each task
    analysis_rows = []
    for cr in critic_results:
        task_id = cr["task_id"]
        ev = cr["evaluation"]
        m = metrics_by_id[task_id]

        plan_recall = m["plan_file_recall"]
        is_bad_plan = plan_recall < BAD_PLAN_THRESHOLD
        critic_flagged = ev["verdict"] in ("flag", "reject")

        # Compute overlap of suggested missing files with actual missed files
        actual_missed = set(normalize_path(f) for f in m["plan_files_missed"])
        suggested_missing = set(normalize_path(f) for f in ev["missing_files_suggested"])
        overlap = actual_missed & suggested_missing
        overlap_count = len(overlap)
        overlap_pct = overlap_count / len(actual_missed) if actual_missed else None

        analysis_rows.append({
            "task_id": task_id,
            "complexity_tier": m["complexity_tier"],
            "plan_recall": plan_recall,
            "is_bad_plan": is_bad_plan,
            "critic_verdict": ev["verdict"],
            "critic_confidence": ev["confidence"],
            "critic_flagged": critic_flagged,
            "scope_assessment": ev["scope_assessment"],
            "concerns": ev["concerns"],
            "key_risk": ev["key_risk"],
            "actual_missed_count": len(actual_missed),
            "suggested_missing_count": len(suggested_missing),
            "missing_files_overlap_count": overlap_count,
            "missing_files_overlap_pct": overlap_pct,
            "overlap_files": sorted(overlap),
            "suggested_but_wrong": sorted(suggested_missing - actual_missed),
            "missed_but_not_suggested": sorted(actual_missed - suggested_missing),
        })

    # Aggregate statistics
    bad_plans = [r for r in analysis_rows if r["is_bad_plan"]]
    good_plans = [r for r in analysis_rows if not r["is_bad_plan"]]

    # Critic recall on bad plans: of actually bad plans, how many did critic flag?
    bad_flagged = [r for r in bad_plans if r["critic_flagged"]]
    critic_recall_on_bad = len(bad_flagged) / len(bad_plans) if bad_plans else 0.0

    # Critic false positive rate: of good plans, how many did critic wrongly flag?
    good_flagged = [r for r in good_plans if r["critic_flagged"]]
    critic_fpr = len(good_flagged) / len(good_plans) if good_plans else 0.0

    # Critic precision: when critic flags, how often is the plan actually bad?
    all_flagged = [r for r in analysis_rows if r["critic_flagged"]]
    critic_precision = len(bad_flagged) / len(all_flagged) if all_flagged else 0.0

    # Missing files overlap for flagged bad plans
    flagged_bad_with_overlap = [r for r in bad_flagged if r["missing_files_overlap_count"] > 0]

    # Which signals best predict failure? Check scope_assessment = "too_narrow"
    too_narrow_and_bad = [r for r in analysis_rows if r["scope_assessment"] == "too_narrow" and r["is_bad_plan"]]
    too_narrow_total = [r for r in analysis_rows if r["scope_assessment"] == "too_narrow"]

    summary_stats = {
        "total_tasks": len(analysis_rows),
        "bad_plans_count": len(bad_plans),
        "good_plans_count": len(good_plans),
        "bad_plan_threshold": BAD_PLAN_THRESHOLD,
        "critic_recall_on_bad_plans": critic_recall_on_bad,
        "critic_false_positive_rate": critic_fpr,
        "critic_precision_when_flagging": critic_precision,
        "bad_plans_flagged": [r["task_id"] for r in bad_flagged],
        "bad_plans_missed_by_critic": [r["task_id"] for r in bad_plans if not r["critic_flagged"]],
        "good_plans_wrongly_flagged": [r["task_id"] for r in good_flagged],
        "flagged_bad_with_overlap": len(flagged_bad_with_overlap),
        "too_narrow_is_bad_count": len(too_narrow_and_bad),
        "too_narrow_total_count": len(too_narrow_total),
        "too_narrow_precision": len(too_narrow_and_bad) / len(too_narrow_total) if too_narrow_total else 0.0,
    }

    return {
        "rows": analysis_rows,
        "summary": summary_stats,
    }


def print_summary_table(analysis: dict):
    """Print the formatted summary table."""
    rows = analysis["rows"]
    summary = analysis["summary"]

    print("\n" + "=" * 120)
    print("PLAN CRITIC EVALUATION SUMMARY")
    print("=" * 120)

    # Header
    header = (f"{'Task ID':<40} {'Recall':>7} {'Verdict':>8} {'Conf':>6} {'Scope':>14} "
              f"{'Suggested':>10} {'Miss Overlap':>14} {'Bad?':>5}")
    print(header)
    print("-" * 130)

    # Sort by plan recall ascending
    for r in sorted(rows, key=lambda x: x["plan_recall"]):
        overlap_str = (f"{r['missing_files_overlap_count']}/{r['actual_missed_count']}"
                       if r["actual_missed_count"] > 0 else "n/a")
        bad_str = "YES" if r["is_bad_plan"] else ""
        print(f"{r['task_id']:<40} {r['plan_recall']:>7.3f} {r['critic_verdict']:>8} "
              f"{r['critic_confidence']:>6.2f} {r['scope_assessment']:>14} "
              f"{r['suggested_missing_count']:>10} {overlap_str:>14} {bad_str:>5}")

    print("-" * 120)

    # Summary statistics
    print(f"\n{'CRITIC PERFORMANCE METRICS':}")
    print(f"  Bad plan threshold: recall < {summary['bad_plan_threshold']}")
    print(f"  Bad plans: {summary['bad_plans_count']}/{summary['total_tasks']}")
    print(f"  Good plans: {summary['good_plans_count']}/{summary['total_tasks']}")
    print()
    print(f"  Critic recall on bad plans: {summary['critic_recall_on_bad_plans']:.1%} "
          f"({len(summary['bad_plans_flagged'])}/{summary['bad_plans_count']})")
    print(f"    Caught: {summary['bad_plans_flagged']}")
    print(f"    Missed: {summary['bad_plans_missed_by_critic']}")
    print()
    print(f"  Critic false positive rate: {summary['critic_false_positive_rate']:.1%} "
          f"({len(summary['good_plans_wrongly_flagged'])}/{summary['good_plans_count']})")
    if summary['good_plans_wrongly_flagged']:
        print(f"    Wrongly flagged: {summary['good_plans_wrongly_flagged']}")
    print()
    print(f"  Critic precision (when flagging): {summary['critic_precision_when_flagging']:.1%}")
    print(f"  Flagged bad plans with missing-file overlap: "
          f"{summary['flagged_bad_with_overlap']}/{len(summary['bad_plans_flagged'])}")
    print()
    print(f"  'too_narrow' scope as predictor: precision = {summary['too_narrow_precision']:.1%} "
          f"({summary['too_narrow_is_bad_count']}/{summary['too_narrow_total_count']})")

    # Detailed concern analysis for the 4 worst failures
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS OF WORST FAILURES")
    print("=" * 120)
    worst_ids = ["django__django-10989", "sympy__sympy-13091", "sympy__sympy-16597", "pylint-dev__pylint-8898"]
    for r in rows:
        if r["task_id"] in worst_ids:
            print(f"\n  {r['task_id']} (recall={r['plan_recall']:.3f}):")
            print(f"    Critic verdict: {r['critic_verdict']} (confidence: {r['critic_confidence']:.2f})")
            print(f"    Scope assessment: {r['scope_assessment']}")
            print(f"    Key risk: {r['key_risk']}")
            print(f"    Concerns: {r['concerns']}")
            print(f"    Missing files suggested: {r['suggested_missing_count']}")
            print(f"    Overlap with actual missed: {r['missing_files_overlap_count']}/{r['actual_missed_count']}")
            if r['overlap_files']:
                print(f"    Overlapping files: {r['overlap_files']}")

    # Check success case
    print("\n" + "=" * 120)
    print("SUCCESS CASE CHECK")
    print("=" * 120)
    for r in rows:
        if r["task_id"] == "django__django-13841":
            print(f"\n  {r['task_id']} (recall={r['plan_recall']:.3f}):")
            print(f"    Critic verdict: {r['critic_verdict']} (confidence: {r['critic_confidence']:.2f})")
            print(f"    Scope assessment: {r['scope_assessment']}")
            rejected = r["critic_verdict"] == "reject"
            print(f"    Wrongly rejected? {'YES - FAILURE' if rejected else 'NO - PASS'}")

    # Signal analysis
    print("\n" + "=" * 120)
    print("SIGNAL ANALYSIS: WHICH CRITIC SIGNALS BEST PREDICT FAILURE?")
    print("=" * 120)

    # 1. Verdict
    for verdict in ("reject", "flag", "approve"):
        subset = [r for r in rows if r["critic_verdict"] == verdict]
        bad_in_subset = [r for r in subset if r["is_bad_plan"]]
        if subset:
            print(f"  Verdict '{verdict}': {len(subset)} tasks, {len(bad_in_subset)} actually bad "
                  f"({len(bad_in_subset)/len(subset):.0%} precision)")

    # 2. Scope assessment
    print()
    for scope in ("too_narrow", "appropriate", "too_broad"):
        subset = [r for r in rows if r["scope_assessment"] == scope]
        bad_in_subset = [r for r in subset if r["is_bad_plan"]]
        if subset:
            print(f"  Scope '{scope}': {len(subset)} tasks, {len(bad_in_subset)} actually bad "
                  f"({len(bad_in_subset)/len(subset):.0%} precision)")

    # 3. Confidence buckets
    print()
    for lo, hi, label in [(0.0, 0.5, "low (0-0.5)"), (0.5, 0.75, "med (0.5-0.75)"), (0.75, 1.01, "high (0.75-1.0)")]:
        subset = [r for r in rows if lo <= r["critic_confidence"] < hi]
        bad_in_subset = [r for r in subset if r["is_bad_plan"]]
        flagged_in_subset = [r for r in subset if r["critic_flagged"]]
        if subset:
            print(f"  Confidence {label}: {len(subset)} tasks, {len(flagged_in_subset)} flagged, "
                  f"{len(bad_in_subset)} actually bad")

    # 4. Suggested missing files count thresholds
    print()
    print("  Suggested missing files count as predictor:")
    for threshold in [3, 5, 6, 8, 10]:
        above = [r for r in rows if r["suggested_missing_count"] >= threshold]
        bad_above = [r for r in above if r["is_bad_plan"]]
        if above:
            prec = len(bad_above) / len(above)
            # Recall: of bad plans, how many have >= threshold suggestions?
            rec = len(bad_above) / len([r for r in rows if r["is_bad_plan"]]) if any(r["is_bad_plan"] for r in rows) else 0
            print(f"    >= {threshold:>2} suggestions: {len(above):>2} tasks, {len(bad_above)} bad "
                  f"(precision={prec:.0%}, recall={rec:.0%})")

    # 5. Composite score: confidence * suggested_count
    print()
    print("  Composite score (confidence * suggested_count) as predictor:")
    scored = [(r, r["critic_confidence"] * r["suggested_missing_count"]) for r in rows]
    scored.sort(key=lambda x: -x[1])
    print(f"    {'Task ID':<40} {'Score':>7} {'Recall':>7} {'Bad?':>5}")
    for r, score in scored:
        bad_str = "YES" if r["is_bad_plan"] else ""
        print(f"    {r['task_id']:<40} {score:>7.2f} {r['plan_recall']:>7.3f} {bad_str:>5}")

    # Best composite threshold
    print()
    for thresh in [4.0, 5.0, 6.0]:
        above = [(r, s) for r, s in scored if s >= thresh]
        bad_above = [r for r, s in above if r["is_bad_plan"]]
        bad_total = len([r for r in rows if r["is_bad_plan"]])
        if above:
            prec = len(bad_above) / len(above)
            rec = len(bad_above) / bad_total if bad_total else 0
            print(f"    Composite >= {thresh:.1f}: {len(above)} flagged, {len(bad_above)} bad "
                  f"(precision={prec:.0%}, recall={rec:.0%})")

    print()


def main():
    # Load data
    with open("data/tasks.json") as f:
        tasks = json.load(f)
    with open("data/plans.json") as f:
        plans = json.load(f)
    with open("data/metrics.json") as f:
        metrics = json.load(f)

    tasks_by_id = {t["task_id"]: t for t in tasks}
    plans_by_id = {p["task_id"]: p for p in plans}

    total = len(tasks)
    print(f"Loaded {total} tasks, {len(plans)} plans, {len(metrics)} metrics entries")

    # Load existing critic evaluations if available (to skip already-succeeded tasks)
    output_path = "data/critic_evaluations.json"
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_list = json.load(f)
        for entry in existing_list:
            if entry.get("evaluation") is not None and "error" not in entry:
                existing[entry["task_id"]] = entry
        print(f"Found {len(existing)} existing critic evaluations -- will skip those\n")
    else:
        print("No existing evaluations found -- generating all from scratch\n")

    # Run critic on each task
    critic_results = []
    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        # Skip if we already have a valid evaluation
        if task_id in existing:
            print(f"[{i}/{total}] Skipping {task_id} (already have evaluation)")
            critic_results.append(existing[task_id])
            continue

        plan_entry = plans_by_id.get(task_id)
        if plan_entry is None or plan_entry.get("plan") is None:
            print(f"[{i}/{total}] Skipping {task_id} (no plan available)")
            continue

        result = run_critic(task, plan_entry, i, total)
        critic_results.append(result)

        # 1 second delay between API calls
        if i < total:
            time.sleep(1)

    # Save critic evaluations
    with open(output_path, "w") as f:
        json.dump(critic_results, f, indent=2)
    print(f"\nCritic evaluations saved to {output_path}")

    # Compute analysis
    analysis = compute_analysis(critic_results, metrics, tasks)

    # Print summary
    print_summary_table(analysis)

    # Save analysis summary alongside evaluations
    summary_output = "data/critic_analysis_summary.json"
    with open(summary_output, "w") as f:
        json.dump(analysis["summary"], f, indent=2)
    print(f"Analysis summary saved to {summary_output}")


if __name__ == "__main__":
    main()
