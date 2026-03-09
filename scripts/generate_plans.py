#!/usr/bin/env python3
"""
Workstream 2: Plan Generation

Calls the Anthropic API to generate a structured implementation plan for each
coding task in data/tasks.json. Saves results to data/plans.json.
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
    # Try to extract JSON from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()
    else:
        json_str = text.strip()

    return json.loads(json_str)


def truncate_repo_structure(repo_structure: str, max_chars: int = 80000) -> str:
    """Truncate repo structure if it's too large, keeping directory-level summary."""
    if len(repo_structure) <= max_chars:
        return repo_structure
    # Keep the first max_chars characters plus a note
    truncated = repo_structure[:max_chars]
    # Try to cut at a newline boundary
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    truncated += "\n\n... [repository structure truncated for length] ..."
    return truncated


def generate_plan(task: dict, index: int, total: int, max_retries: int = 2) -> dict:
    """Generate a plan for a single task via the Anthropic API."""
    task_id = task["task_id"]
    print(f"[{index}/{total}] Generating plan for {task_id}...")

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
                print(f"       -> Retry {attempt}/{max_retries}...")
                time.sleep(2)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            stop_reason = response.stop_reason

            if stop_reason == "max_tokens":
                print(f"       -> WARNING: response was truncated (max_tokens reached)")

            plan = parse_json_response(raw_text)

            # Basic validation
            for key in ("files_to_inspect", "files_to_modify", "implementation_steps"):
                if key not in plan:
                    raise ValueError(f"Plan missing required key: {key}")

            inspect_count = len(plan.get("files_to_inspect", []))
            modify_count = len(plan.get("files_to_modify", []))
            steps_count = len(plan.get("implementation_steps", []))
            print(f"       -> {inspect_count} files to inspect, {modify_count} files to modify, {steps_count} steps")

            return {
                "task_id": task_id,
                "plan": plan,
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

    # All retries exhausted
    raise last_error


def main():
    # Load tasks
    with open("data/tasks.json") as f:
        tasks = json.load(f)

    total = len(tasks)
    print(f"Loaded {total} tasks from data/tasks.json")

    # Load existing plans if available (to skip already-succeeded tasks)
    output_path = "data/plans.json"
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_list = json.load(f)
        for entry in existing_list:
            if entry.get("plan") is not None:
                existing[entry["task_id"]] = entry
        print(f"Found {len(existing)} existing successful plans — will skip those\n")
    else:
        print("No existing plans found — generating all from scratch\n")

    results = []
    successes = 0
    failures = 0
    skipped = 0

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        # Skip if we already have a valid plan
        if task_id in existing:
            print(f"[{i}/{total}] Skipping {task_id} (already have plan)")
            results.append(existing[task_id])
            successes += 1
            skipped += 1
            continue

        try:
            result = generate_plan(task, i, total)
            results.append(result)
            successes += 1
        except Exception as e:
            print(f"       -> FINAL ERROR for {task_id}: {e}")
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "plan": None,
                "raw_response": None,
                "error": str(e),
            })
            failures += 1

        # Be polite: sleep 1 second between API calls
        if i < total:
            time.sleep(1)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {successes} succeeded ({skipped} from cache), {failures} failed.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
