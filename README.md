# Planning Evaluation for Autonomous Coding Agents

A research evaluation measuring whether and when planning helps autonomous coding agents navigate codebases. Built as a take-home project for Cognition.

**Core question:** Does generating a structured plan (file list + implementation steps) before coding reduce navigation waste in large repositories?

**Short answer:** Yes, by +35 percentage points in file recall on average — but with strongly diminishing returns as task complexity grows, and with important nuances about how plans should be consumed. Read Plan_Eval.pdf for a full report on experiments and results.

---

## Repository Structure

```
plan_eval/
├── data/                          # All datasets and model outputs
│   ├── tasks.json                 # 14 pilot tasks (from SWE-bench)
│   ├── tasks_extended.json        # 236 extended tasks
│   ├── plans.json                 # Sonnet-generated plans (14 pilot)
│   ├── plans_extended.json        # Sonnet-generated plans (236)
│   ├── plans_haiku.json           # Haiku plans for model comparison
│   ├── plans_opus.json            # Opus plans for model comparison (8 tasks)
│   ├── plans_minimal.json         # Minimal prompt variant
│   ├── plans_dependency.json      # Dependency-tracing prompt variant
│   ├── metrics.json               # File recall/precision/baseline for pilot
│   ├── metrics_extended.json      # Same for 234 extended tasks
│   ├── plan_content_scores.json   # Opus-as-judge content quality scores
│   ├── plan_content_metrics.json  # Content quality summary
│   ├── critic_evaluations.json    # Plan critic scores
│   ├── judge_scores.json          # LLM-as-judge dimension scores
│   ├── refinement_results.json    # Critique-revise loop results (8 tasks)
│   ├── plan_vs_noplan_*.json      # A/B code gen comparison results
│   ├── soft_context_results.json  # Soft vs hard plan framing experiment
│   ├── code_gen_extended_results.json  # Code gen eval (50 tasks)
│   ├── opus_comparison_metrics.json    # 3-model comparison metrics
│   ├── file_difficulty_*.json     # Miss rate by file category
│   ├── synthetic_*.json           # Synthetic feature task data
│   └── swebench_reports/          # SWE-bench eval output (Docker required)
│
├── analysis/                      # All plots and qualitative analyses
│   ├── extended_crossover.png     # HEADLINE: plan recall by tier, n=234
│   ├── extended_regression.png    # Recall vs complexity regression (r=-0.44)
│   ├── file_difficulty_extended.png  # Miss rate by file category
│   ├── plan_content_vs_recall.png # Navigation vs reasoning scatter
│   ├── opus_comparison.png        # Haiku vs Sonnet vs Opus on 8 hard tasks
│   ├── model_comparison.png       # Haiku vs Sonnet across all tiers
│   ├── prompt_sensitivity.png     # 3 prompt variants comparison
│   ├── plan_refinement.png        # Critique-revise trajectory
│   ├── synthetic_surface_coverage.png  # Feature task coverage by category
│   ├── plan_vs_noplan_extended.png     # Code gen A/B, n=40
│   ├── soft_context_comparison.png    # Hard/soft/no plan code gen
│   ├── failure_analysis.md        # Qualitative failure analysis
│   ├── plan_content_analysis.md   # Content quality deep dive
│   └── [other supporting plots]
│
└── scripts/                       # All evaluation scripts (standalone, runnable)
    ├── curate_tasks.py            # Task selection from SWE-bench
    ├── curate_tasks_extended.py   # Extended task curation (236 tasks)
    ├── generate_plans.py          # Plan generation via Anthropic API
    ├── generate_plans_extended.py # Batch plan generation with resume
    ├── compute_metrics.py         # File recall/precision computation
    ├── compute_metrics_extended.py
    ├── run_baseline.py            # Keyword matching baseline
    ├── analyze.py                 # Pilot crossover analysis + plots
    ├── analyze_extended.py        # Extended analysis with stats + CI
    ├── file_difficulty_analysis.py # Miss rate by file category
    ├── plan_content_evaluation.py # Opus-as-judge content quality
    ├── plan_critic.py             # LLM plan critic
    ├── llm_judge.py               # Multi-dimension LLM judge
    ├── model_comparison.py        # Haiku vs Sonnet comparison
    ├── opus_comparison.py         # 3-model comparison
    ├── prompt_sensitivity.py      # Prompt variant A/B/C
    ├── plan_refinement.py         # Critique-revise loop
    ├── code_generation.py         # Code gen from plans (pilot)
    ├── code_generation_extended.py # Code gen at scale (50 tasks)
    ├── plan_vs_noplan.py          # Plan vs no-plan code gen (n=10)
    ├── plan_vs_noplan_extended.py # Plan vs no-plan code gen (n=40)
    ├── soft_context_codegen.py    # Hard/soft/no plan framing experiment
    ├── synthetic_feature_eval.py  # Synthetic feature task eval
    └── swebench_eval.py           # SWE-bench Docker test-suite eval (needs Docker)
```

---

## Experiments at a Glance

| # | Experiment | Tasks | Key Finding |
|---|-----------|-------|-------------|
| 1 | Core navigation eval | 234 | Plans +35pp recall vs keyword baseline (p<0.0001) |
| 2 | Prompt sensitivity | 14 | Dependency-tracing: +19pp on cross-module |
| 3 | Haiku vs Sonnet | 14 | Haiku matches Sonnet at 3-4x lower cost |
| 4 | Plan critic | 14 | Binary verdicts useless; composite score ≥6.0 works |
| 5 | File-level difficulty | 234 | Distant modules 100% miss; shared utils 61% miss |
| 6 | Plan refinement | 8 | First pass +9pp; second pass regresses |
| 7 | Code gen from plans | 50 | Plans moderate code gen quality (r=0.356) |
| 8 | Plan vs no-plan code gen | 40 | NoPlan wins 12/40; plans cause over-specification |
| 9 | Failure analysis | 14 | 4 failure modes: scope collapse, hub blindness, wrong layer, downstream blindness |
| 10 | Plan content quality | 14 | Navigation and reasoning are decoupled (content flat, recall drops 3x) |
| 11 | Synthetic feature tasks | 3 | 0.89 surface coverage (vs 0.53 file recall for bug fixes) |
| 12 | Plan vs no-plan (scale) | 40 | Finding confirmed: NoPlan wins 12/40, Plans 3/40 |
| 13 | Soft-context framing | 15 | Soft framing doesn't fix anchoring; NoPlan still wins |

---

## Setup

```bash
conda create -n planning_eval python=3.11 -y
conda activate planning_eval
pip install matplotlib pandas numpy seaborn datasets huggingface_hub anthropic python-dotenv swebench
```

API key goes in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

All scripts are standalone and run from the repo root:
```bash
conda activate planning_eval
python scripts/curate_tasks.py
python scripts/generate_plans.py
# etc.
```

The core pipeline (tasks → plans → metrics → analysis) runs end-to-end in ~15 minutes of API time for the 14-task pilot. The 236-task extended run takes ~25 minutes.

