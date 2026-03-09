"""
File-Level Difficulty Analysis

For each task, classify every missed file into categories:
- test_file: test files
- config_file: __init__.py, setup.py, conf.py, etc.
- shared_utility: files in utils/, common/, helpers/, core/ shared dirs
- direct_dependency: files in the same package as a correctly identified file
- indirect_dependency: files in a different package
- documentation: docs, README, CHANGELOG files

Aggregates across all tasks to show which file types plans miss most.
Works on both pilot (data/metrics.json) and extended (data/metrics_extended.json).
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import re
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def classify_file(filepath, matched_files):
    """Classify a file into a difficulty category."""
    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    parts = filepath.lower().split("/")

    # Test file
    if ("test" in parts or "tests" in parts or
            basename.startswith("test_") or basename.endswith("_test.py") or
            "conftest" in basename):
        return "test_file"

    # Config / init file
    if basename in ("__init__.py", "setup.py", "setup.cfg", "conf.py",
                     "conftest.py", "pyproject.toml", "MANIFEST.in"):
        return "config_init"

    # Documentation
    if "doc" in parts or "docs" in parts or basename.endswith(".md") or basename.endswith(".rst"):
        return "documentation"

    # Shared utility (hub files)
    utility_indicators = {"utils", "util", "helpers", "helper", "common",
                          "compat", "compat.py", "base", "mixins"}
    if any(ind in parts for ind in utility_indicators) or any(ind in basename for ind in ["utils.py", "helpers.py", "compat.py", "mixins.py"]):
        return "shared_utility"

    # Check if it's in the same package as any matched file
    if matched_files:
        matched_dirs = {os.path.dirname(f) for f in matched_files}
        if dirname in matched_dirs:
            return "same_package"
        # Check one level up
        parent = os.path.dirname(dirname)
        matched_parents = {os.path.dirname(d) for d in matched_dirs}
        if parent and parent in matched_parents:
            return "sibling_package"

    return "distant_module"


def analyze_metrics_file(metrics_path, tasks_path, label):
    """Analyze missed files from a metrics file."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(tasks_path) as f:
        tasks = json.load(f)

    task_map = {t["task_id"]: t for t in tasks}

    # Aggregate missed file classifications
    all_missed_categories = Counter()
    all_found_categories = Counter()
    tier_missed = defaultdict(Counter)
    tier_found = defaultdict(Counter)

    total_missed = 0
    total_found = 0

    for m in metrics:
        task_id = m["task_id"]
        tier = m["complexity_tier"]
        matched = set(m.get("plan_files_matched", []))
        missed = set(m.get("plan_files_missed", []))

        for f in missed:
            cat = classify_file(f, matched)
            all_missed_categories[cat] += 1
            tier_missed[tier][cat] += 1
            total_missed += 1

        for f in matched:
            cat = classify_file(f, matched)
            all_found_categories[cat] += 1
            tier_found[tier][cat] += 1
            total_found += 1

    return {
        "label": label,
        "total_missed": total_missed,
        "total_found": total_found,
        "missed_categories": dict(all_missed_categories),
        "found_categories": dict(all_found_categories),
        "tier_missed": {t: dict(c) for t, c in tier_missed.items()},
        "tier_found": {t: dict(c) for t, c in tier_found.items()},
    }


def compute_miss_rates(result):
    """Compute miss rate per category: missed / (missed + found)."""
    categories = set(result["missed_categories"].keys()) | set(result["found_categories"].keys())
    rates = {}
    for cat in categories:
        missed = result["missed_categories"].get(cat, 0)
        found = result["found_categories"].get(cat, 0)
        total = missed + found
        rates[cat] = {
            "missed": missed,
            "found": found,
            "total": total,
            "miss_rate": missed / total if total > 0 else 0,
        }
    return rates


def plot_miss_rates(rates, output_path, title="File Miss Rate by Category"):
    """Bar chart of miss rates by file category."""
    # Sort by miss rate descending
    sorted_cats = sorted(rates.items(), key=lambda x: -x[1]["miss_rate"])

    categories = [c for c, _ in sorted_cats if _["total"] >= 2]
    miss_rates = [rates[c]["miss_rate"] for c in categories]
    totals = [rates[c]["total"] for c in categories]

    # Nicer category names
    nice_names = {
        "test_file": "Test Files",
        "config_init": "Config/__init__.py",
        "documentation": "Documentation",
        "shared_utility": "Shared Utilities",
        "same_package": "Same Package",
        "sibling_package": "Sibling Package",
        "distant_module": "Distant Module",
    }
    labels = [nice_names.get(c, c) for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("YlOrRd", len(categories))

    bars = ax.barh(range(len(categories)), miss_rates, color=colors)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Miss Rate (fraction of files in category that plans miss)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)

    # Add count annotations
    for i, (rate, total) in enumerate(zip(miss_rates, totals)):
        ax.text(rate + 0.02, i, f"{rate:.0%} (n={total})", va="center", fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    os.makedirs("analysis", exist_ok=True)

    # Analyze pilot data
    if os.path.exists("data/metrics.json"):
        print("=" * 60)
        print("PILOT DATA (14 tasks)")
        print("=" * 60)
        result = analyze_metrics_file("data/metrics.json", "data/tasks.json", "pilot")
        rates = compute_miss_rates(result)

        print(f"\nTotal files: {result['total_missed'] + result['total_found']}")
        print(f"  Found: {result['total_found']}")
        print(f"  Missed: {result['total_missed']}")
        print(f"\nMiss rate by category:")
        for cat, info in sorted(rates.items(), key=lambda x: -x[1]["miss_rate"]):
            print(f"  {cat:20s}: {info['miss_rate']:.0%} "
                  f"({info['missed']}/{info['total']} missed)")

        plot_miss_rates(rates, "analysis/file_difficulty_pilot.png",
                       "Plan Miss Rate by File Category (14 Pilot Tasks)")

        # Save to JSON
        with open("data/file_difficulty_pilot.json", "w") as f:
            json.dump({"rates": {k: v for k, v in rates.items()},
                       "detail": result}, f, indent=2)
        print(f"\nSaved data/file_difficulty_pilot.json")

    # Analyze extended data
    if os.path.exists("data/metrics_extended.json"):
        print("\n" + "=" * 60)
        print("EXTENDED DATA")
        print("=" * 60)
        result = analyze_metrics_file("data/metrics_extended.json",
                                       "data/tasks_extended.json", "extended")
        rates = compute_miss_rates(result)

        print(f"\nTotal files: {result['total_missed'] + result['total_found']}")
        print(f"  Found: {result['total_found']}")
        print(f"  Missed: {result['total_missed']}")
        print(f"\nMiss rate by category:")
        for cat, info in sorted(rates.items(), key=lambda x: -x[1]["miss_rate"]):
            print(f"  {cat:20s}: {info['miss_rate']:.0%} "
                  f"({info['missed']}/{info['total']} missed)")

        plot_miss_rates(rates, "analysis/file_difficulty_extended.png",
                       "Plan Miss Rate by File Category (Extended Tasks)")

        with open("data/file_difficulty_extended.json", "w") as f:
            json.dump({"rates": {k: v for k, v in rates.items()},
                       "detail": result}, f, indent=2)
        print(f"\nSaved data/file_difficulty_extended.json")


if __name__ == "__main__":
    main()
