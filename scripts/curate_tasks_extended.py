"""
Extended task curation for plan evaluation.

Produces 100+ tasks from SWE-bench (Verified + Full) with relaxed filters
and richer complexity metadata. Saves to data/tasks_extended.json.

Relaxed filters vs pilot:
  - Minimum 2 files changed (was 3)
  - At least 2 directories/modules
  - Problem statement at least 50 chars (was 100)
  - Exclude purely test/doc changes
  - Do NOT filter by "problem names files"

Strategy for repo trees:
  - Fetch one tree per REPO (not per commit) to stay within unauthenticated
    GitHub API rate limits (60 req/hour). The file tree of large repos like
    Django/SymPy is stable enough across commits for planning context.
  - If GitHub API fails, build a comprehensive fallback tree from all patch
    paths across all tasks in that repo.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import math
import re
import sys
import time
import statistics
import requests
from collections import defaultdict
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    if len(top_dirs) == 1 and len(second_level_dirs) >= 2:
        return second_level_dirs
    return top_dirs


def get_all_dirs(files):
    """Get all unique parent directories from file paths."""
    dirs = set()
    for f in files:
        d = os.path.dirname(f)
        if d:
            dirs.add(d)
    return dirs


def get_top_modules(files):
    """Get distinct top-level packages/directories."""
    modules = set()
    for f in files:
        parts = f.split('/')
        if len(parts) > 1:
            modules.add(parts[0])
        else:
            modules.add('.')
    return modules


def max_dir_depth(files):
    """Get the deepest directory level of any changed file."""
    max_depth = 0
    for f in files:
        depth = f.count('/')
        if depth > max_depth:
            max_depth = depth
    return max_depth


def has_test_files(files):
    """Check if any test files are in the changed files."""
    for f in files:
        lower = f.lower()
        parts = lower.split('/')
        if ('test' in parts or 'tests' in parts or
                lower.endswith('_test.py') or
                os.path.basename(lower).startswith('test_')):
            return True
    return False


def has_init_files(files):
    """Check if any __init__.py files are in the changed files."""
    for f in files:
        if os.path.basename(f) == '__init__.py':
            return True
    return False


def is_only_tests_or_docs(files):
    """Check if changes are exclusively test or docs files."""
    for f in files:
        lower = f.lower()
        parts = lower.split('/')
        is_test = ('test' in parts or 'tests' in parts or
                   lower.endswith('_test.py') or
                   os.path.basename(lower).startswith('test_'))
        is_doc = ('doc' in parts or 'docs' in parts)
        if not is_test and not is_doc:
            return False
    return True


def assign_complexity_tier(n_files):
    """Assign complexity tier based on file count."""
    if n_files <= 3:
        return "localized"
    elif n_files <= 6:
        return "cross-module"
    else:
        return "architectural"


def compute_complexity_score(n_files, n_dirs):
    """Continuous complexity score = n_files * log(n_dirs + 1)."""
    return round(n_files * math.log(n_dirs + 1), 3)


# ---------------------------------------------------------------------------
# GitHub API: repo tree fetching (one tree per repo, not per commit)
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO_TREE_CACHE = {}  # repo -> (py_files_list, total_count)


def get_repo_tree_github(repo, commit_sha, retries=3):
    """Fetch the Python file tree from GitHub API at a specific commit.

    Results are cached by repo (not by commit) to minimize API calls.
    """
    if repo in REPO_TREE_CACHE:
        return REPO_TREE_CACHE[repo]

    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    for attempt in range(retries):
        try:
            # Step 1: Get commit to find tree SHA
            url = f"https://api.github.com/repos/{repo}/git/commits/{commit_sha}"
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code == 403:
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                reset_ts = resp.headers.get("X-RateLimit-Reset", "")
                if reset_ts:
                    wait = max(int(reset_ts) - int(time.time()), 10) + 5
                else:
                    wait = 65
                print(f"    Rate limited (remaining={remaining}), "
                      f"waiting {wait}s... (attempt {attempt+1}/{retries})",
                      flush=True)
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                print(f"    Commit not found for {repo} @ {commit_sha[:8]}, "
                      f"trying default branch...", flush=True)
                # Try the default branch instead
                url2 = f"https://api.github.com/repos/{repo}"
                resp2 = requests.get(url2, headers=headers, timeout=30)
                if resp2.status_code == 200:
                    default_branch = resp2.json().get("default_branch", "main")
                    url3 = f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}"
                    time.sleep(1)
                    resp3 = requests.get(url3, headers=headers, timeout=30)
                    if resp3.status_code == 200:
                        commit_sha = resp3.json()["object"]["sha"]
                        url = f"https://api.github.com/repos/{repo}/git/commits/{commit_sha}"
                        time.sleep(1)
                        resp = requests.get(url, headers=headers, timeout=30)
                        if resp.status_code != 200:
                            REPO_TREE_CACHE[repo] = (None, 0)
                            return None, 0
                    else:
                        REPO_TREE_CACHE[repo] = (None, 0)
                        return None, 0
                else:
                    REPO_TREE_CACHE[repo] = (None, 0)
                    return None, 0

            if resp.status_code != 200:
                print(f"    Failed to get commit: {resp.status_code}", flush=True)
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                REPO_TREE_CACHE[repo] = (None, 0)
                return None, 0

            tree_sha = resp.json()["tree"]["sha"]
            time.sleep(1)

            # Step 2: Get recursive file tree
            url = f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}?recursive=1"
            resp = requests.get(url, headers=headers, timeout=60)

            if resp.status_code == 403:
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                reset_ts = resp.headers.get("X-RateLimit-Reset", "")
                if reset_ts:
                    wait = max(int(reset_ts) - int(time.time()), 10) + 5
                else:
                    wait = 65
                print(f"    Rate limited on tree (remaining={remaining}), "
                      f"waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    Failed to get tree: {resp.status_code}", flush=True)
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                REPO_TREE_CACHE[repo] = (None, 0)
                return None, 0

            data = resp.json()
            truncated = data.get("truncated", False)
            if truncated:
                print(f"    Warning: tree was truncated for {repo}", flush=True)

            all_files = [item["path"] for item in data.get("tree", [])
                         if item["type"] == "blob"]
            py_files = [f for f in all_files if f.endswith(".py")
                        and '__pycache__' not in f]

            result = (sorted(py_files), len(py_files))
            REPO_TREE_CACHE[repo] = result
            time.sleep(1)
            return result

        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(5)

    REPO_TREE_CACHE[repo] = (None, 0)
    return None, 0


def build_fallback_tree_from_all_tasks(repo, tasks_for_repo):
    """Build a comprehensive fallback tree from ALL patch paths in a repo.

    Aggregates all files from all tasks' patches for the repo, plus infers
    __init__.py files for each directory.
    """
    all_paths = set()
    for task in tasks_for_repo:
        all_paths.update(task["ground_truth_files_changed"])

    # Add __init__.py for each directory
    dirs = set()
    for f in all_paths:
        d = os.path.dirname(f)
        while d:
            dirs.add(d)
            d = os.path.dirname(d)

    for d in dirs:
        all_paths.add(os.path.join(d, "__init__.py"))

    py_files = sorted(f for f in all_paths if f.endswith(".py"))
    # Estimate total repo files conservatively
    estimated_total = max(len(py_files) * 8, 200)
    return py_files, estimated_total


# ---------------------------------------------------------------------------
# Pilot tree extraction: reuse trees already fetched in pilot
# ---------------------------------------------------------------------------

def load_pilot_trees():
    """Extract repo trees from the pilot tasks.json."""
    trees = {}  # repo -> (py_files, count)
    if not os.path.exists("data/tasks.json"):
        return trees
    with open("data/tasks.json") as f:
        pilot_tasks = json.load(f)

    for task in pilot_tasks:
        repo = task["repo"]
        if repo not in trees:
            summary = task.get("repo_structure_summary", "")
            if summary:
                py_files = [line for line in summary.split("\n") if line.strip()]
                total = task.get("total_repo_files", len(py_files))
                trees[repo] = (py_files, total)

    return trees


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # -------------------------------------------------------------------
    # Phase 1: Load datasets and pilot tasks
    # -------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("EXTENDED TASK CURATION", flush=True)
    print("=" * 60, flush=True)

    # Load pilot tasks to exclude
    pilot_ids = set()
    if os.path.exists("data/tasks.json"):
        with open("data/tasks.json") as f:
            pilot_tasks = json.load(f)
        pilot_ids = {t["task_id"] for t in pilot_tasks}
        print(f"Loaded {len(pilot_ids)} pilot task IDs to exclude", flush=True)
    else:
        print("No pilot tasks.json found, nothing to exclude", flush=True)

    # Load pilot trees for reuse
    pilot_trees = load_pilot_trees()
    if pilot_trees:
        print(f"Loaded trees for {len(pilot_trees)} repos from pilot data: "
              f"{', '.join(sorted(pilot_trees.keys()))}", flush=True)
        for repo, (files, count) in pilot_trees.items():
            REPO_TREE_CACHE[repo] = (files, count)

    print("\nLoading SWE-bench datasets...", flush=True)
    ds_verified = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  Verified: {len(ds_verified)} instances", flush=True)

    ds_lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"  Lite: {len(ds_lite)} instances", flush=True)

    ds_full = load_dataset("princeton-nlp/SWE-bench", split="test")
    print(f"  Full: {len(ds_full)} instances", flush=True)

    verified_ids = set(item["instance_id"] for item in ds_verified)
    lite_ids = set(item["instance_id"] for item in ds_lite)

    # -------------------------------------------------------------------
    # Phase 2: Filter candidates from Full SWE-bench (superset)
    # -------------------------------------------------------------------
    print("\nFiltering candidates with relaxed criteria...", flush=True)
    candidates = []
    skipped_reasons = defaultdict(int)

    for item in ds_full:
        instance_id = item["instance_id"]

        # Skip pilot tasks
        if instance_id in pilot_ids:
            skipped_reasons["in_pilot"] += 1
            continue

        patch = item.get("patch", "")
        if not patch:
            skipped_reasons["no_patch"] += 1
            continue

        files = parse_patch_files(patch)
        if len(files) < 2:
            skipped_reasons["<2_files"] += 1
            continue

        top_dirs = get_unique_top_dirs(files)
        if len(top_dirs) < 2:
            skipped_reasons["<2_dirs"] += 1
            continue

        if is_only_tests_or_docs(files):
            skipped_reasons["only_test_doc"] += 1
            continue

        problem = item.get("problem_statement", "")
        if len(problem) < 50:
            skipped_reasons["short_problem"] += 1
            continue

        # Compute rich complexity metadata
        n_files = len(files)
        all_dirs = get_all_dirs(files)
        n_dirs = len(all_dirs)
        top_mods = get_top_modules(files)
        n_top_modules = len(top_mods)
        depth = max_dir_depth(files)
        has_tests = has_test_files(files)
        has_inits = has_init_files(files)
        tier = assign_complexity_tier(n_files)
        score = compute_complexity_score(n_files, n_dirs)

        # Determine source quality label
        if instance_id in verified_ids:
            source = "swebench_verified"
        elif instance_id in lite_ids:
            source = "swebench_lite"
        else:
            source = "swebench_full"

        candidates.append({
            "instance_id": instance_id,
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": problem,
            "ground_truth_files_changed": files,
            "ground_truth_patch": patch,
            "source": source,
            # Complexity metadata
            "n_files_changed": n_files,
            "n_dirs_changed": n_dirs,
            "n_top_modules": n_top_modules,
            "max_dir_depth": depth,
            "has_test_files": has_tests,
            "has_init_files": has_inits,
            "complexity_tier": tier,
            "complexity_score": score,
            "tier_rationale": (
                f"{n_files} files across {len(top_dirs)} module-level dirs "
                f"({n_dirs} unique dirs, depth {depth}): "
                f"{', '.join(sorted(top_dirs)[:5])}"
                f"{' ...' if len(top_dirs) > 5 else ''}"
            ),
        })

    print(f"\nTotal candidates: {len(candidates)}", flush=True)
    print("\nSkip reasons:", flush=True)
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}", flush=True)

    # -------------------------------------------------------------------
    # Phase 3: Select tasks with tier balancing and repo diversity
    # -------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("SELECTING TASKS", flush=True)
    print("=" * 60, flush=True)

    # Group by tier
    tier_pools = defaultdict(list)
    for c in candidates:
        tier_pools[c["complexity_tier"]].append(c)

    print("\nCandidate pool by tier:", flush=True)
    for tier in ["localized", "cross-module", "architectural"]:
        items = tier_pools.get(tier, [])
        v_count = sum(1 for i in items if i["source"] == "swebench_verified")
        l_count = sum(1 for i in items if i["source"] == "swebench_lite")
        f_count = sum(1 for i in items if i["source"] == "swebench_full")
        print(f"  {tier}: {len(items)} total "
              f"(verified={v_count}, lite={l_count}, full={f_count})", flush=True)

    # Take ALL candidates — we want 100+ and have ~250
    # Use round-robin repo diversity to interleave repos within each tier
    # Prioritize: Verified > Lite > Full

    selected = []
    selected_ids = set()

    for tier in ["localized", "cross-module", "architectural"]:
        pool = tier_pools.get(tier, [])
        if not pool:
            continue

        # Sort: verified first, then lite, then full;
        # within each source, prefer longer problem statements (more context)
        source_order = {"swebench_verified": 0, "swebench_lite": 1, "swebench_full": 2}
        pool.sort(key=lambda x: (source_order.get(x["source"], 9),
                                  -len(x["problem_statement"])))

        # Round-robin across repos for diversity
        by_repo = defaultdict(list)
        for c in pool:
            by_repo[c["repo"]].append(c)

        repos = sorted(by_repo.keys())
        tier_selected = []
        idx = 0
        # Keep going until all repos are drained
        while any(by_repo[r] for r in repos):
            repo = repos[idx % len(repos)]
            if by_repo[repo]:
                task = by_repo[repo].pop(0)
                if task["instance_id"] not in selected_ids:
                    tier_selected.append(task)
                    selected_ids.add(task["instance_id"])
            idx += 1

        selected.extend(tier_selected)
        print(f"  Selected {len(tier_selected)} {tier} tasks", flush=True)

    print(f"\nTotal selected: {len(selected)} tasks", flush=True)

    if len(selected) < 100:
        print(f"WARNING: Only {len(selected)} tasks, target was 100+", flush=True)

    # -------------------------------------------------------------------
    # Phase 4: Fetch repo structures
    # -------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("FETCHING REPO STRUCTURES", flush=True)
    print("=" * 60, flush=True)

    # Group tasks by repo
    tasks_by_repo = defaultdict(list)
    for task in selected:
        tasks_by_repo[task["repo"]].append(task)

    unique_repos = sorted(tasks_by_repo.keys())
    print(f"Need trees for {len(unique_repos)} unique repos", flush=True)

    # For each repo, pick a representative commit and fetch tree
    # (or reuse from cache/pilot)
    from_cache = 0
    from_api = 0
    from_fallback = 0

    for repo in unique_repos:
        tasks_for_repo = tasks_by_repo[repo]

        if repo in REPO_TREE_CACHE:
            py_files, count = REPO_TREE_CACHE[repo]
            if py_files is not None:
                print(f"  {repo}: using cached tree "
                      f"({count} .py files, {len(tasks_for_repo)} tasks)", flush=True)
                from_cache += 1
                for task in tasks_for_repo:
                    task["repo_structure_summary"] = "\n".join(py_files)
                    task["total_repo_files"] = count
                continue

        # Set to True to skip GitHub API and use fallback trees from patches
        USE_FALLBACK_ONLY = False
        if USE_FALLBACK_ONLY:
            py_files, count = None, 0
        else:
            representative_commit = tasks_for_repo[0]["base_commit"]
            print(f"  {repo}: fetching tree @ {representative_commit[:8]}... "
                  f"({len(tasks_for_repo)} tasks)", flush=True)
            py_files, count = get_repo_tree_github(repo, representative_commit)

        if py_files is not None:
            from_api += 1
            print(f"    Got {count} .py files from GitHub API", flush=True)
            for task in tasks_for_repo:
                task["repo_structure_summary"] = "\n".join(py_files)
                task["total_repo_files"] = count
        else:
            from_fallback += 1
            print(f"    GitHub API failed, building fallback tree...", flush=True)
            py_files, count = build_fallback_tree_from_all_tasks(repo, tasks_for_repo)
            REPO_TREE_CACHE[repo] = (py_files, count)
            print(f"    Fallback: {len(py_files)} .py files "
                  f"(estimated {count} total)", flush=True)
            for task in tasks_for_repo:
                task["repo_structure_summary"] = "\n".join(py_files)
                task["total_repo_files"] = count

    print(f"\nTree fetch summary: {from_cache} cached, {from_api} from API, "
          f"{from_fallback} fallback", flush=True)

    # -------------------------------------------------------------------
    # Phase 5: Build final output
    # -------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("BUILDING OUTPUT", flush=True)
    print("=" * 60, flush=True)

    tasks_output = []
    for task in selected:
        tasks_output.append({
            "task_id": task["instance_id"],
            "source": task["source"],
            "repo": task["repo"],
            "problem_statement": task["problem_statement"],
            "ground_truth_files_changed": task["ground_truth_files_changed"],
            "ground_truth_patch": task["ground_truth_patch"],
            # Complexity metadata (extended)
            "n_files_changed": task["n_files_changed"],
            "n_dirs_changed": task["n_dirs_changed"],
            "n_top_modules": task["n_top_modules"],
            "max_dir_depth": task["max_dir_depth"],
            "has_test_files": task["has_test_files"],
            "has_init_files": task["has_init_files"],
            "complexity_tier": task["complexity_tier"],
            "complexity_score": task["complexity_score"],
            "tier_rationale": task["tier_rationale"],
            # Repo structure
            "total_repo_files": task["total_repo_files"],
            "repo_structure_summary": task["repo_structure_summary"],
        })

    os.makedirs("data", exist_ok=True)
    output_path = "data/tasks_extended.json"
    with open(output_path, "w") as f:
        json.dump(tasks_output, f, indent=2)

    print(f"Saved {len(tasks_output)} tasks to {output_path}", flush=True)

    # -------------------------------------------------------------------
    # Phase 6: Statistics and quality checks
    # -------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("FINAL STATISTICS", flush=True)
    print("=" * 60, flush=True)

    # Total
    print(f"\nTotal tasks: {len(tasks_output)}", flush=True)

    # Tier distribution
    print("\nTier distribution:", flush=True)
    for tier in ["localized", "cross-module", "architectural"]:
        count = sum(1 for t in tasks_output if t["complexity_tier"] == tier)
        pct = 100 * count / len(tasks_output) if tasks_output else 0
        print(f"  {tier}: {count} ({pct:.1f}%)", flush=True)

    # Source distribution
    print("\nSource distribution:", flush=True)
    for source in ["swebench_verified", "swebench_lite", "swebench_full"]:
        count = sum(1 for t in tasks_output if t["source"] == source)
        pct = 100 * count / len(tasks_output) if tasks_output else 0
        print(f"  {source}: {count} ({pct:.1f}%)", flush=True)

    # Repo distribution
    print("\nRepo distribution:", flush=True)
    repo_counts = defaultdict(int)
    for t in tasks_output:
        repo_counts[t["repo"]] += 1
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(tasks_output) if tasks_output else 0
        print(f"  {repo}: {count} ({pct:.1f}%)", flush=True)

    # Complexity score distribution
    scores = [t["complexity_score"] for t in tasks_output]
    if scores:
        print("\nComplexity score distribution:", flush=True)
        print(f"  min:    {min(scores):.3f}", flush=True)
        print(f"  max:    {max(scores):.3f}", flush=True)
        print(f"  mean:   {statistics.mean(scores):.3f}", flush=True)
        print(f"  median: {statistics.median(scores):.3f}", flush=True)
        if len(scores) > 1:
            print(f"  stdev:  {statistics.stdev(scores):.3f}", flush=True)

    # Files changed distribution
    file_counts = [t["n_files_changed"] for t in tasks_output]
    if file_counts:
        print("\nFiles changed distribution:", flush=True)
        print(f"  min:    {min(file_counts)}", flush=True)
        print(f"  max:    {max(file_counts)}", flush=True)
        print(f"  mean:   {statistics.mean(file_counts):.1f}", flush=True)
        print(f"  median: {statistics.median(file_counts):.1f}", flush=True)

    # Quality checks
    print("\n" + "=" * 60, flush=True)
    print("QUALITY CHECKS", flush=True)
    print("=" * 60, flush=True)

    issues = []

    # Check minimum count
    if len(tasks_output) < 100:
        issues.append(f"Only {len(tasks_output)} tasks (target: 100+)")

    # Check tier balance (each tier should have some)
    for tier in ["localized", "cross-module", "architectural"]:
        count = sum(1 for t in tasks_output if t["complexity_tier"] == tier)
        if count < 5:
            issues.append(f"{tier} tier has only {count} tasks (want >= 5)")

    # Check no overlap with pilot
    overlap = [t for t in tasks_output if t["task_id"] in pilot_ids]
    if overlap:
        issues.append(f"{len(overlap)} tasks overlap with pilot: "
                       f"{[t['task_id'] for t in overlap]}")

    # Check all have repo structures
    missing_tree = sum(1 for t in tasks_output if not t.get("repo_structure_summary"))
    if missing_tree:
        issues.append(f"{missing_tree} tasks missing repo structure")

    if issues:
        print("ISSUES:", flush=True)
        for issue in issues:
            print(f"  - {issue}", flush=True)
    else:
        print("All quality checks passed!", flush=True)

    # File size info
    file_size = os.path.getsize(output_path)
    print(f"\nOutput file size: {file_size / (1024*1024):.1f} MB", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
