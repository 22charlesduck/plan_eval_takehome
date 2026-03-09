#!/usr/bin/env python3
"""
Extended Plan Generation (100+ tasks)

Reads data/tasks_extended.json, writes data/plans_extended.json.
Processes tasks in batches with resume support and rate limiting.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
BATCH_SIZE = 10          # Save progress after every N tasks
RATE_LIMIT_DELAY = 0.5   # Seconds between API calls (configurable)
MAX_REPO_STRUCTURE_CHARS = 60000  # Truncation threshold for repo_structure_summary

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


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code fences if present."""
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()
    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = MAX_REPO_STRUCTURE_CHARS) -> str:
    """Truncate repo structure if it exceeds max_chars, keeping directory-level summary."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    truncated = repo_structure[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def generate_plan(task: dict, index: int, total: int, max_retries: int = 2) -> dict:
    """Generate a plan for a single task via the Anthropic API."""
    task_id = task["task_id"]
    print(f"  [{index}/{total}] Generating plan for {task_id}...")

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
                time.sleep(2 ** attempt)  # Exponential backoff

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            stop_reason = response.stop_reason

            if stop_reason == "max_tokens":
                print(f"           -> WARNING: response was truncated (max_tokens reached)")

            plan = parse_json_response(raw_text)

            # Basic validation
            for key in ("files_to_inspect", "files_to_modify", "implementation_steps"):
                if key not in plan:
                    raise ValueError(f"Plan missing required key: {key}")

            inspect_count = len(plan.get("files_to_inspect", []))
            modify_count = len(plan.get("files_to_modify", []))
            steps_count = len(plan.get("implementation_steps", []))
            print(f"           -> {inspect_count} inspect, {modify_count} modify, {steps_count} steps")

            return {
                "task_id": task_id,
                "plan": plan,
                "raw_response": raw_text,
            }

        except json.JSONDecodeError as e:
            last_error = e
            print(f"           -> JSON parse error on attempt {attempt + 1}: {e}")
            if raw_text:
                print(f"           -> Raw response preview: {raw_text[:200]}...")
        except anthropic.APIError as e:
            last_error = e
            print(f"           -> API error on attempt {attempt + 1}: {e}")
        except Exception as e:
            last_error = e
            print(f"           -> Unexpected error on attempt {attempt + 1}: {e}")

    # All retries exhausted
    return {
        "task_id": task_id,
        "plan": None,
        "raw_response": raw_text,
        "error": str(last_error),
    }


def load_existing_plans(output_path: str) -> dict:
    """Load existing plans from output file and index by task_id."""
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_list = json.load(f)
        for entry in existing_list:
            if entry.get("plan") is not None:
                existing[entry["task_id"]] = entry
    return existing


def save_plans(results: list, output_path: str):
    """Save plans list to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    input_path = "data/tasks_extended.json"
    output_path = "data/plans_extended.json"

    # Load tasks
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run curate_tasks.py for extended data first.")
        return

    with open(input_path) as f:
        tasks = json.load(f)

    total = len(tasks)
    print(f"Loaded {total} tasks from {input_path}")

    # Load existing plans for resume support
    existing = load_existing_plans(output_path)
    if existing:
        print(f"Found {len(existing)} existing successful plans -- will resume from where we left off\n")
    else:
        print("No existing plans found -- generating all from scratch\n")

    results = list(existing.values())  # Start with existing successful plans
    existing_ids = set(existing.keys())

    successes = len(existing)
    failures = 0
    skipped = len(existing)
    batch_count = 0

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        # Skip if we already have a valid plan
        if task_id in existing_ids:
            # Progress print every 10 tasks even for skips
            if i % 10 == 0:
                print(f"  Progress: {i}/{total} processed ({successes} success, {failures} fail, {skipped} cached)")
            continue

        result = generate_plan(task, i, total)
        results.append(result)

        if result.get("plan") is not None:
            successes += 1
        else:
            failures += 1

        batch_count += 1

        # Save progress after each batch
        if batch_count % BATCH_SIZE == 0:
            save_plans(results, output_path)
            print(f"\n  --- Batch checkpoint: saved {len(results)} plans to {output_path} ---")
            print(f"  --- Progress: {i}/{total} processed ({successes} success, {failures} fail) ---\n")

        # Progress print every 10 tasks
        if i % 10 == 0:
            print(f"  Progress: {i}/{total} processed ({successes} success, {failures} fail)")

        # Rate limiting between API calls
        if i < total and task_id not in existing_ids:
            time.sleep(RATE_LIMIT_DELAY)

    # Final save
    save_plans(results, output_path)

    print(f"\n{'='*60}")
    print(f"Done. {successes} succeeded ({skipped} from cache), {failures} failed.")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
