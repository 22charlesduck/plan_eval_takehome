"""
Workstream 3, Script 2: Baseline B — keyword/retrieval baseline from issue text.

Extracts identifier-like tokens from each task's problem_statement, matches them
against file paths in repo_structure_summary, and computes the same recall/precision/
search-reduction metrics.

Reads: data/tasks.json, data/metrics.json (from compute_metrics.py)
Writes: data/metrics.json (updated with baseline_b_* fields)
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re

# Common tokens to filter out — these would match too many files
STOP_TOKENS = {
    # Python keywords and builtins
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
    # Python-specific common words
    "python", "code", "method", "function", "attribute", "object",
    "string", "number", "print", "call", "args", "kwargs", "param",
    "default", "example", "issue", "problem", "patch", "change",
    "changes", "added", "removed", "fixed", "need", "needs", "want",
    "case", "result", "expected", "actual", "version", "current",
    "new", "old", "original", "update", "updated", "instead",
    # Super common path fragments
    "docs", "doc", "src", "lib", "bin", "conf", "locale", "templates",
    "static", "contrib", "compat", "internal", "extern", "third_party",
    "vendor",
}


def extract_tokens(text: str) -> set:
    """
    Extract identifier-like tokens from problem statement text.
    Looks for:
    - CamelCase words (e.g., HttpResponse, SameSite)
    - snake_case words (e.g., delete_cookie, same_site)
    - Dot-separated paths (e.g., django.http.response)
    - File-path-like strings (e.g., django/http/response.py)
    - Python identifiers longer than 3 chars
    """
    tokens = set()

    # 1. File-path-like strings: sequences with slashes and dots
    # e.g., "django/http/response.py" or "path/to/file"
    file_paths = re.findall(r'[\w]+(?:/[\w]+)+(?:\.[\w]+)?', text)
    for fp in file_paths:
        tokens.add(fp)
        # Also add individual path components
        for part in fp.split('/'):
            if len(part) > 3 and part.lower() not in STOP_TOKENS:
                tokens.add(part)

    # 2. Dot-separated identifiers: e.g., django.http.response
    dot_paths = re.findall(r'[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+', text)
    for dp in dot_paths:
        tokens.add(dp)
        # Convert dot path to slash path for matching
        slash_path = dp.replace('.', '/')
        tokens.add(slash_path)
        # Add individual components
        for part in dp.split('.'):
            if len(part) > 3 and part.lower() not in STOP_TOKENS:
                tokens.add(part)

    # 3. CamelCase words (2+ capitalized segments)
    camel_words = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', text)
    for cw in camel_words:
        if len(cw) > 3 and cw.lower() not in STOP_TOKENS:
            tokens.add(cw)

    # 4. snake_case words (underscore-separated)
    snake_words = re.findall(r'[a-z]+(?:_[a-z]+)+', text)
    for sw in snake_words:
        if len(sw) > 3 and sw.lower() not in STOP_TOKENS:
            tokens.add(sw)

    # 5. General Python identifiers (longer than 3 chars, not common English)
    identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
    for ident in identifiers:
        if len(ident) > 5 and ident.lower() not in STOP_TOKENS:
            # Only keep identifiers that look like code (contain underscore,
            # are CamelCase, or are ALL_CAPS)
            if ('_' in ident or
                    re.match(r'[A-Z][a-z]+[A-Z]', ident) or
                    ident.isupper()):
                tokens.add(ident)

    # Filter out stop tokens (case-insensitive)
    filtered = set()
    for t in tokens:
        if t.lower() not in STOP_TOKENS and len(t) > 2:
            filtered.add(t)

    return filtered


def match_token_to_files(token: str, file_paths: list) -> list:
    """
    Match a token against a list of file paths using substring matching.
    Returns list of matching file paths.
    """
    matches = []
    token_lower = token.lower()

    for fp in file_paths:
        fp_lower = fp.lower()
        # Direct substring match
        if token_lower in fp_lower:
            matches.append(fp)
        # Also try slash-converted version of dot-separated tokens
        elif '.' in token and '/' not in token:
            slash_token = token.replace('.', '/').lower()
            if slash_token in fp_lower:
                matches.append(fp)

    return matches


def normalize_path(p: str) -> str:
    """Normalize a file path."""
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


def match_files(baseline_files: set, gt_files: set):
    """Match baseline files against ground truth using normalized path matching."""
    matched_gt = set()
    matched_baseline = set()

    for gf in gt_files:
        for bf in baseline_files:
            if paths_match(gf, bf):
                matched_gt.add(gf)
                matched_baseline.add(bf)

    missed_gt = gt_files - matched_gt
    extra_baseline = baseline_files - matched_baseline

    return matched_gt, matched_baseline, missed_gt, extra_baseline


def main():
    with open("data/tasks.json") as f:
        tasks = json.load(f)

    # Load existing metrics (from compute_metrics.py)
    with open("data/metrics.json") as f:
        metrics = json.load(f)

    # Index metrics by task_id
    metrics_by_task = {m["task_id"]: m for m in metrics}

    for task in tasks:
        task_id = task["task_id"]
        gt_files = set(task["ground_truth_files_changed"])
        total_repo_files = task["total_repo_files"]
        problem_statement = task["problem_statement"]
        repo_structure = task["repo_structure_summary"]

        # Parse repo file paths
        repo_files = [line.strip() for line in repo_structure.strip().split('\n')
                      if line.strip()]

        # Step 1: Extract identifier-like tokens from problem statement
        tokens = extract_tokens(problem_statement)

        # Step 2: Match tokens against repo file paths
        baseline_files = set()
        token_match_counts = {}

        for token in tokens:
            matched = match_token_to_files(token, repo_files)
            token_match_counts[token] = len(matched)

            # Skip tokens that match too many files (likely too generic)
            if len(matched) > 30:
                continue

            baseline_files.update(matched)

        # Step 3: Compute metrics using same matching logic as plan
        matched_gt, matched_baseline, missed_gt, extra_baseline = match_files(
            baseline_files, gt_files
        )

        file_recall = len(matched_gt) / len(gt_files) if gt_files else 0.0
        file_precision = len(matched_baseline) / len(baseline_files) if baseline_files else 0.0
        search_reduction = len(baseline_files) / total_repo_files if total_repo_files > 0 else 0.0

        # Top tokens (sorted by match count, excluding overly broad ones)
        useful_tokens = sorted(
            [(t, c) for t, c in token_match_counts.items() if 0 < c <= 30],
            key=lambda x: x[1],
            reverse=True
        )

        # Update metrics entry
        if task_id in metrics_by_task:
            m = metrics_by_task[task_id]
            m["baseline_b_file_recall"] = round(file_recall, 4)
            m["baseline_b_file_precision"] = round(file_precision, 4)
            m["baseline_b_search_reduction"] = round(search_reduction, 6)
            m["baseline_b_files_matched"] = sorted(matched_gt)
            m["baseline_b_files_missed"] = sorted(missed_gt)
            m["baseline_b_files_extra"] = sorted(extra_baseline)
            m["baseline_b_keywords"] = [t for t, _ in useful_tokens[:20]]
            m["baseline_b_files_count"] = len(baseline_files)

        print(f"{task_id:45s}  "
              f"recall={file_recall:.2f}  precision={file_precision:.2f}  "
              f"reduction={search_reduction:.4f}  "
              f"matched={len(matched_gt)}/{len(gt_files)}  "
              f"baseline_size={len(baseline_files)}  "
              f"tokens={len(tokens)}")

    # Write updated metrics
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nUpdated data/metrics.json with baseline B fields.")

    # Summary stats
    avg_recall = sum(m.get("baseline_b_file_recall", 0) for m in metrics) / len(metrics)
    avg_precision = sum(m.get("baseline_b_file_precision", 0) for m in metrics) / len(metrics)
    print(f"\nBaseline B averages: recall={avg_recall:.3f}  precision={avg_precision:.3f}")


if __name__ == "__main__":
    main()
