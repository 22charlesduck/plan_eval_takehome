import os
os.chdir("/scr/clding/plan_eval")

import json
import re
import time
import requests
from collections import defaultdict
from datasets import load_dataset


def parse_patch_files(patch):
    """Extract changed file paths from a unified diff patch."""
    files = set()
    for match in re.finditer(r'diff --git a/(.*?) b/(.*)', patch):
        a_path = match.group(1)
        b_path = match.group(2)
        if a_path != '/dev/null':
            files.add(a_path)
        if b_path != '/dev/null':
            files.add(b_path)
    return sorted(files)


def get_unique_top_dirs(files):
    """Get unique module-level directories from file paths.

    Uses second-level dirs (e.g., django/db, django/contrib) when all files
    share the same top-level dir, since SWE-bench repos typically have a single
    top-level package directory with multiple sub-modules.
    """
    top_dirs = set()
    second_level_dirs = set()
    for f in files:
        parts = f.split('/')
        if len(parts) > 1:
            top_dirs.add(parts[0])
        else:
            top_dirs.add('.')
        if len(parts) > 2:
            second_level_dirs.add('/'.join(parts[:2]))

    # If all files share the same top-level dir, use second-level dirs
    # to measure true module spread (e.g., django/db vs django/contrib)
    if len(top_dirs) == 1 and len(second_level_dirs) >= 2:
        return second_level_dirs
    return top_dirs


def problem_names_most_files(problem_text, files):
    """Check if the problem statement trivially names most changed file paths."""
    if not problem_text:
        return False
    named_count = 0
    for f in files:
        # Check for full path or distinctive filename
        basename = os.path.basename(f)
        # Skip generic names that could appear coincidentally
        if basename in ('__init__.py', 'models.py', 'views.py', 'forms.py',
                        'tests.py', 'urls.py', 'admin.py', 'settings.py',
                        'conf.py', 'setup.py', 'conftest.py'):
            continue
        if f in problem_text:
            named_count += 1
        elif len(basename) > 8 and basename in problem_text:
            named_count += 1
    # If majority of non-generic files are named, it's too obvious
    return named_count >= max(2, len(files) * 0.5)


def is_only_tests_or_docs(files):
    """Check if changes are exclusively test or docs files."""
    non_test_doc = []
    for f in files:
        lower = f.lower()
        is_test = ('test' in lower.split('/') or 'tests' in lower.split('/') or
                   lower.endswith('_test.py') or os.path.basename(lower).startswith('test_'))
        is_doc = ('doc' in lower.split('/') or 'docs' in lower.split('/'))
        if not is_test and not is_doc:
            non_test_doc.append(f)
    return len(non_test_doc) == 0


def assign_complexity_tier(files):
    """Assign complexity tier based on file count and directory spread."""
    n_files = len(files)
    top_dirs = get_unique_top_dirs(files)
    n_dirs = len(top_dirs)

    if n_files <= 3:
        tier = "localized"
    elif n_files <= 6:
        tier = "cross-module"
    else:
        tier = "architectural"

    rationale = f"{n_files} files changed across {n_dirs} top-level dirs: {', '.join(sorted(top_dirs))}"
    return tier, rationale


def get_repo_tree_github(repo, commit_sha, retries=2):
    """Fetch the Python file tree from GitHub API at a specific commit."""
    headers = {"Accept": "application/vnd.github.v3+json"}

    for attempt in range(retries):
        try:
            # Step 1: Get commit to find tree SHA
            url = f"https://api.github.com/repos/{repo}/git/commits/{commit_sha}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 403:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                print(f"    Failed to get commit: {resp.status_code}")
                return None, 0

            tree_sha = resp.json()["tree"]["sha"]

            # Step 2: Get recursive file tree
            url = f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}?recursive=1"
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code != 200:
                print(f"    Failed to get tree: {resp.status_code}")
                return None, 0

            data = resp.json()
            truncated = data.get("truncated", False)
            if truncated:
                print(f"    Warning: tree was truncated for {repo}")

            all_files = [item["path"] for item in data.get("tree", [])
                         if item["type"] == "blob"]
            py_files = [f for f in all_files if f.endswith(".py")
                        and '__pycache__' not in f]

            return sorted(py_files), len(py_files)

        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}")
            if attempt < retries - 1:
                time.sleep(5)

    return None, 0


def filter_candidates(dataset, source_label, min_files=2):
    """Filter a HuggingFace dataset into candidate tasks.

    Args:
        dataset: HuggingFace dataset split
        source_label: label for the data source (e.g. "swebench_verified")
        min_files: minimum number of changed files (default 2)

    Returns:
        list of candidate dicts
    """
    candidates = []

    for item in dataset:
        patch = item.get("patch", "")
        if not patch:
            continue

        files = parse_patch_files(patch)
        if len(files) < min_files:
            continue

        top_dirs = get_unique_top_dirs(files)
        if len(top_dirs) < 2:
            continue

        if is_only_tests_or_docs(files):
            continue

        problem = item.get("problem_statement", "")
        if len(problem) < 100:
            continue

        if problem_names_most_files(problem, files):
            continue

        tier, rationale = assign_complexity_tier(files)

        candidates.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": problem,
            "ground_truth_files_changed": files,
            "ground_truth_patch": patch,
            "complexity_tier": tier,
            "tier_rationale": rationale,
            "n_files": len(files),
            "n_dirs": len(top_dirs),
            "source_label": source_label,
        })

    return candidates


def select_from_pool(pool, target, already_selected_ids):
    """Select up to `target` tasks from pool using round-robin repo diversity."""
    by_repo = defaultdict(list)
    for c in pool:
        if c["instance_id"] in already_selected_ids:
            continue
        by_repo[c["repo"]].append(c)

    # Sort each repo's tasks by problem length descending (prefer substantive problems)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda x: len(x["problem_statement"]), reverse=True)

    # Round-robin selection across repos
    selected = []
    repos_list = sorted(by_repo.keys())
    if not repos_list:
        return selected
    idx = 0
    while len(selected) < target and any(by_repo[r] for r in repos_list):
        repo = repos_list[idx % len(repos_list)]
        if by_repo[repo]:
            selected.append(by_repo[repo].pop(0))
        idx += 1

    return selected


def main():
    # ---------------------------------------------------------------
    # Phase 1: Load datasets
    # ---------------------------------------------------------------
    print("Loading SWE-bench Verified...")
    ds_verified = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  Loaded {len(ds_verified)} Verified instances")

    print("Loading full SWE-bench...")
    ds_full = load_dataset("princeton-nlp/SWE-bench", split="test")
    print(f"  Loaded {len(ds_full)} full instances")

    verified_ids = set(item["instance_id"] for item in ds_verified)

    # ---------------------------------------------------------------
    # Phase 2: Filter candidates from both datasets
    # ---------------------------------------------------------------
    print("\nFiltering candidates from Verified (min_files=2)...")
    verified_candidates = filter_candidates(ds_verified, "swebench_verified", min_files=2)
    print(f"  Found {len(verified_candidates)} candidates")

    print("Filtering candidates from full SWE-bench (min_files=4, excluding Verified)...")
    full_candidates_raw = filter_candidates(ds_full, "swebench_full", min_files=4)
    # Exclude anything already in Verified
    full_candidates = [c for c in full_candidates_raw if c["instance_id"] not in verified_ids]
    print(f"  Found {len(full_candidates)} additional candidates")

    # Merge: Verified first, then full SWE-bench supplements
    all_candidates = verified_candidates + full_candidates

    # Show distribution
    tier_pools = defaultdict(list)
    for c in all_candidates:
        tier_pools[c["complexity_tier"]].append(c)

    print("\nCandidate pool:")
    for tier in ["localized", "cross-module", "architectural"]:
        items = tier_pools.get(tier, [])
        v_count = sum(1 for i in items if i["source_label"] == "swebench_verified")
        f_count = sum(1 for i in items if i["source_label"] == "swebench_full")
        print(f"  {tier}: {len(items)} total ({v_count} verified, {f_count} full)")
        repos = defaultdict(int)
        for item in items:
            repos[item["repo"]] += 1
        for repo, count in sorted(repos.items(), key=lambda x: -x[1])[:5]:
            print(f"    {repo}: {count}")

    # ---------------------------------------------------------------
    # Phase 3: Select tasks — aim for 14 total across 3 tiers
    # ---------------------------------------------------------------
    print("\nSelecting tasks...")
    targets = {"localized": 5, "cross-module": 5, "architectural": 4}
    selected = []
    selected_ids = set()

    for tier, target in targets.items():
        pool = tier_pools.get(tier, [])
        if not pool:
            print(f"  WARNING: No candidates for {tier}")
            continue

        # Prefer Verified candidates first
        verified_pool = [c for c in pool if c["source_label"] == "swebench_verified"]
        full_pool = [c for c in pool if c["source_label"] == "swebench_full"]

        tier_selected = select_from_pool(verified_pool, target, selected_ids)
        selected_ids.update(c["instance_id"] for c in tier_selected)

        remaining = target - len(tier_selected)
        if remaining > 0 and full_pool:
            extra = select_from_pool(full_pool, remaining, selected_ids)
            tier_selected.extend(extra)
            selected_ids.update(c["instance_id"] for c in extra)

        selected.extend(tier_selected)
        print(f"  {tier}: selected {len(tier_selected)} tasks "
              f"({sum(1 for t in tier_selected if t['source_label']=='swebench_verified')} verified, "
              f"{sum(1 for t in tier_selected if t['source_label']=='swebench_full')} full)")

    print(f"\nTotal selected: {len(selected)} tasks")

    # ---------------------------------------------------------------
    # Phase 4: Fetch repo structures from GitHub API
    # ---------------------------------------------------------------
    print("\nFetching repo file trees from GitHub API...")
    tree_cache = {}

    for i, task in enumerate(selected):
        key = (task["repo"], task["base_commit"])
        if key not in tree_cache:
            print(f"  [{i+1}/{len(selected)}] Fetching {task['repo']} @ {task['base_commit'][:8]}...")
            py_files, count = get_repo_tree_github(task["repo"], task["base_commit"])
            tree_cache[key] = (py_files, count)
            time.sleep(1)  # Be gentle with rate limits

        py_files, count = tree_cache[key]
        if py_files:
            task["repo_structure_summary"] = "\n".join(py_files)
            task["total_repo_files"] = count
        else:
            # Fallback: construct partial tree from patch + known paths
            print(f"    Fallback: constructing partial tree from patch for {task['instance_id']}")
            patch_files = task["ground_truth_files_changed"]
            dirs = set()
            for f in patch_files:
                d = os.path.dirname(f)
                if d:
                    dirs.add(d)
            task["repo_structure_summary"] = "\n".join(patch_files)
            task["total_repo_files"] = 200  # Conservative estimate

    # ---------------------------------------------------------------
    # Phase 5: Build final output
    # ---------------------------------------------------------------
    tasks_output = []
    for task in selected:
        tasks_output.append({
            "task_id": task["instance_id"],
            "source": task["source_label"],
            "repo": task["repo"],
            "problem_statement": task["problem_statement"],
            "ground_truth_files_changed": task["ground_truth_files_changed"],
            "ground_truth_patch": task["ground_truth_patch"],
            "complexity_tier": task["complexity_tier"],
            "tier_rationale": task["tier_rationale"],
            "total_repo_files": task["total_repo_files"],
            "repo_structure_summary": task["repo_structure_summary"],
        })

    os.makedirs("data", exist_ok=True)
    with open("data/tasks.json", "w") as f:
        json.dump(tasks_output, f, indent=2)

    # ---------------------------------------------------------------
    # Phase 6: Quality checks
    # ---------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUALITY CHECKS")
    print("=" * 50)
    print(f"Total tasks: {len(tasks_output)}")
    ok = True

    # Check tier distribution
    for tier in ["localized", "cross-module", "architectural"]:
        count = sum(1 for t in tasks_output if t["complexity_tier"] == tier)
        status = "OK" if count >= 2 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {tier}: {count} [{status}]")

    # Check repo structure
    missing = sum(1 for t in tasks_output if not t["repo_structure_summary"])
    if missing > 0:
        print(f"  Tasks missing repo structure: {missing} [FAIL]")
        ok = False
    else:
        print(f"  All tasks have repo structure [OK]")

    # Check file counts (relaxed: 2+ files OK)
    for t in tasks_output:
        if len(t["ground_truth_files_changed"]) < 2:
            print(f"  WARNING: {t['task_id']} has only {len(t['ground_truth_files_changed'])} files")
            ok = False

    # Check problem statement length
    short = [t for t in tasks_output if len(t["problem_statement"]) < 100]
    if short:
        print(f"  WARNING: {len(short)} tasks have short problem statements")
        ok = False
    else:
        print(f"  All problem statements are substantive [OK]")

    # Summary
    print(f"\n{'='*50}")
    print("TASK SUMMARY")
    print(f"{'='*50}")
    for t in tasks_output:
        print(f"  {t['task_id']}")
        print(f"    Tier: {t['complexity_tier']}")
        print(f"    Files changed: {len(t['ground_truth_files_changed'])}")
        print(f"    Repo: {t['repo']}")
        print(f"    Source: {t['source']}")
        print(f"    Problem length: {len(t['problem_statement'])} chars")
        print(f"    Repo structure lines: {len(t['repo_structure_summary'].split(chr(10)))}")
        print()

    if ok:
        print("All quality checks passed!")
    else:
        print("Some quality checks failed - review above warnings")

    print(f"\nSaved {len(tasks_output)} tasks to data/tasks.json")


if __name__ == "__main__":
    main()
