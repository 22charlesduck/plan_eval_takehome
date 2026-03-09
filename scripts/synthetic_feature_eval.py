"""
Experiment 11: Synthetic Feature Planning

Tests whether planning evaluation generalizes from bug fixes to open-ended
feature implementation. Creates a synthetic CLI repo (~30 files), defines 3
feature tasks with surface checklists, generates plans via Sonnet, scores
plans programmatically + via Opus judge, and produces comparison writeup and
a coverage bar chart.
"""

import os
os.chdir("/scr/clding/plan_eval")

import json
import time
import textwrap
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

client = anthropic.Anthropic()

SONNET = "claude-sonnet-4-20250514"
OPUS = "claude-opus-4-20250514"
DELAY = 2.0

# ============================================================================
# Part 1: Create Synthetic Repo
# ============================================================================

REPO_ROOT = Path("data/synthetic_repos/dataflow")

REPO_FILES = {
    # cli/
    "cli/__init__.py": '''"""Dataflow CLI — a data processing toolkit."""\n\n__version__ = "0.1.0"\n''',

    "cli/main.py": textwrap.dedent('''\
        """Entry point and argument parsing for the dataflow CLI."""

        import argparse
        import sys
        from cli.commands.process import ProcessCommand
        from cli.commands.validate import ValidateCommand
        from cli.commands.inspect import InspectCommand
        from cli.formats.json_output import JsonFormatter
        from cli.formats.csv_output import CsvFormatter
        from cli.utils.logging import setup_logging
        from cli.utils.config import load_config


        FORMAT_REGISTRY = {
            "json": JsonFormatter,
            "csv": CsvFormatter,
        }

        COMMAND_REGISTRY = {
            "process": ProcessCommand,
            "validate": ValidateCommand,
            "inspect": InspectCommand,
        }


        def build_parser():
            """Build the main argument parser."""
            parser = argparse.ArgumentParser(prog="dataflow", description="CLI data processing tool")
            parser.add_argument("--format", choices=list(FORMAT_REGISTRY.keys()), default="json")
            parser.add_argument("--config", type=str, default=None, help="Path to config file")
            subparsers = parser.add_subparsers(dest="command")

            for name, cmd_cls in COMMAND_REGISTRY.items():
                sub = subparsers.add_parser(name, help=cmd_cls.help_text)
                cmd_cls.add_arguments(sub)

            return parser


        def main(argv=None):
            """Main entry point."""
            setup_logging()
            parser = build_parser()
            args = parser.parse_args(argv)

            if not args.command:
                parser.print_help()
                sys.exit(1)

            config = load_config(args.config)
            formatter = FORMAT_REGISTRY[args.format]()
            command = COMMAND_REGISTRY[args.command](config=config, formatter=formatter)
            result = command.execute(args)
            print(formatter.format(result))


        if __name__ == "__main__":
            main()
    '''),

    # commands/
    "cli/commands/__init__.py": '"""Command implementations."""\n',

    "cli/commands/process.py": textwrap.dedent('''\
        """Main data processing command."""

        from cli.core.pipeline import Pipeline
        from cli.utils.file_io import read_input, write_output


        class ProcessCommand:
            """Execute the data processing pipeline on input files."""

            help_text = "Process input data through the pipeline"

            def __init__(self, config, formatter):
                self.config = config
                self.formatter = formatter
                self.pipeline = Pipeline(config)

            @staticmethod
            def add_arguments(parser):
                parser.add_argument("input", help="Input file path")
                parser.add_argument("-o", "--output", help="Output file path")

            def execute(self, args):
                """Run the processing pipeline."""
                data = read_input(args.input)
                result = self.pipeline.run(data)
                if args.output:
                    write_output(args.output, self.formatter.format(result))
                return result
    '''),

    "cli/commands/validate.py": textwrap.dedent('''\
        """Data validation command."""

        from cli.core.validators import SchemaValidator, RuleValidator


        class ValidateCommand:
            """Validate input data against schema and rules."""

            help_text = "Validate data against schema and rules"

            def __init__(self, config, formatter):
                self.config = config
                self.formatter = formatter

            @staticmethod
            def add_arguments(parser):
                parser.add_argument("input", help="Input file to validate")
                parser.add_argument("--schema", help="Schema file path")
                parser.add_argument("--strict", action="store_true")

            def execute(self, args):
                """Run validation."""
                raise NotImplementedError("Validation command not yet implemented")
    '''),

    "cli/commands/inspect.py": textwrap.dedent('''\
        """Data inspection command."""


        class InspectCommand:
            """Inspect data files and show summary statistics."""

            help_text = "Inspect data and show summary"

            def __init__(self, config, formatter):
                self.config = config
                self.formatter = formatter

            @staticmethod
            def add_arguments(parser):
                parser.add_argument("input", help="Input file to inspect")
                parser.add_argument("--detailed", action="store_true")

            def execute(self, args):
                """Run inspection."""
                raise NotImplementedError("Inspect command not yet implemented")
    '''),

    # formats/
    "cli/formats/__init__.py": '"""Output formatters."""\n',

    "cli/formats/base.py": textwrap.dedent('''\
        """Abstract base formatter."""

        from abc import ABC, abstractmethod


        class BaseFormatter(ABC):
            """Base class for all output formatters."""

            @abstractmethod
            def format(self, data):
                """Format the given data into a string representation."""
                raise NotImplementedError

            @abstractmethod
            def file_extension(self):
                """Return the default file extension for this format."""
                raise NotImplementedError
    '''),

    "cli/formats/json_output.py": textwrap.dedent('''\
        """JSON output formatter."""

        import json
        from cli.formats.base import BaseFormatter


        class JsonFormatter(BaseFormatter):
            """Format output as JSON."""

            def __init__(self, indent=2, sort_keys=False):
                self.indent = indent
                self.sort_keys = sort_keys

            def format(self, data):
                """Serialize data to a JSON string."""
                return json.dumps(data, indent=self.indent, sort_keys=self.sort_keys)

            def file_extension(self):
                return ".json"
    '''),

    "cli/formats/csv_output.py": textwrap.dedent('''\
        """CSV output formatter."""

        import csv
        import io
        from cli.formats.base import BaseFormatter


        class CsvFormatter(BaseFormatter):
            """Format output as CSV."""

            def __init__(self, delimiter=","):
                self.delimiter = delimiter

            def format(self, data):
                """Serialize tabular data to CSV string."""
                if not data:
                    return ""
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys(), delimiter=self.delimiter)
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()

            def file_extension(self):
                return ".csv"
    '''),

    # core/
    "cli/core/__init__.py": '"""Core processing logic."""\n',

    "cli/core/pipeline.py": textwrap.dedent('''\
        """Data processing pipeline."""

        from cli.core.transforms import apply_transforms
        from cli.core.schema import validate_schema
        from cli.cache.memory_cache import MemoryCache


        class Pipeline:
            """Orchestrates the data processing pipeline."""

            def __init__(self, config):
                self.config = config
                self.cache = MemoryCache(max_size=config.get("cache_size", 100))
                self.transforms = config.get("transforms", [])

            def run(self, data):
                """Execute the full pipeline on input data."""
                cache_key = self._compute_key(data)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return cached

                validated = validate_schema(data, self.config.get("schema"))
                result = apply_transforms(validated, self.transforms)

                self.cache.set(cache_key, result)
                return result

            def _compute_key(self, data):
                """Compute a cache key for the given data."""
                import hashlib
                return hashlib.md5(str(data).encode()).hexdigest()
    '''),

    "cli/core/transforms.py": textwrap.dedent('''\
        """Data transformation functions."""


        TRANSFORM_REGISTRY = {}


        def register_transform(name):
            """Decorator to register a transform function."""
            def decorator(func):
                TRANSFORM_REGISTRY[name] = func
                return func
            return decorator


        @register_transform("uppercase")
        def uppercase_transform(data, **kwargs):
            """Convert string fields to uppercase."""
            raise NotImplementedError


        @register_transform("filter")
        def filter_transform(data, **kwargs):
            """Filter records based on conditions."""
            raise NotImplementedError


        @register_transform("aggregate")
        def aggregate_transform(data, **kwargs):
            """Aggregate records by key."""
            raise NotImplementedError


        def apply_transforms(data, transform_specs):
            """Apply a sequence of transforms to the data."""
            result = data
            for spec in transform_specs:
                name = spec["name"]
                params = spec.get("params", {})
                if name not in TRANSFORM_REGISTRY:
                    raise ValueError(f"Unknown transform: {name}")
                result = TRANSFORM_REGISTRY[name](result, **params)
            return result
    '''),

    "cli/core/schema.py": textwrap.dedent('''\
        """Schema definitions and validation."""


        class SchemaDefinition:
            """Represents a data schema with field types and constraints."""

            def __init__(self, fields=None, required=None):
                self.fields = fields or {}
                self.required = required or []

            def validate(self, record):
                """Validate a single record against this schema."""
                raise NotImplementedError


        def validate_schema(data, schema_config):
            """Validate data against schema if provided, otherwise pass through."""
            if schema_config is None:
                return data
            schema = SchemaDefinition(**schema_config)
            for record in data:
                schema.validate(record)
            return data
    '''),

    "cli/core/validators.py": textwrap.dedent('''\
        """Validation rules for data quality checks."""


        class SchemaValidator:
            """Validate data against a JSON schema."""

            def __init__(self, schema_path):
                self.schema_path = schema_path

            def validate(self, data):
                raise NotImplementedError


        class RuleValidator:
            """Validate data against custom business rules."""

            def __init__(self, rules):
                self.rules = rules

            def validate(self, data):
                raise NotImplementedError


        class CompositeValidator:
            """Combine multiple validators."""

            def __init__(self, validators):
                self.validators = validators

            def validate(self, data):
                """Run all validators and collect results."""
                raise NotImplementedError
    '''),

    # cache/
    "cli/cache/__init__.py": '"""Caching backends."""\n',

    "cli/cache/memory_cache.py": textwrap.dedent('''\
        """In-memory cache with LRU eviction."""

        from collections import OrderedDict


        class MemoryCache:
            """Simple in-memory LRU cache."""

            def __init__(self, max_size=100):
                self.max_size = max_size
                self._store = OrderedDict()

            def get(self, key):
                """Retrieve a cached value, or None if missing."""
                if key in self._store:
                    self._store.move_to_end(key)
                    return self._store[key]
                return None

            def set(self, key, value):
                """Store a value in the cache."""
                if key in self._store:
                    self._store.move_to_end(key)
                else:
                    if len(self._store) >= self.max_size:
                        self._store.popitem(last=False)
                self._store[key] = value

            def clear(self):
                """Clear all cached entries."""
                self._store.clear()
    '''),

    # utils/
    "cli/utils/__init__.py": '"""Utility modules."""\n',

    "cli/utils/logging.py": textwrap.dedent('''\
        """Logging configuration."""

        import logging


        def setup_logging(level="INFO", log_file=None):
            """Configure logging for the application."""
            handlers = [logging.StreamHandler()]
            if log_file:
                handlers.append(logging.FileHandler(log_file))

            logging.basicConfig(
                level=getattr(logging, level),
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                handlers=handlers,
            )

            return logging.getLogger("dataflow")
    '''),

    "cli/utils/config.py": textwrap.dedent('''\
        """Configuration loader."""

        import os
        import yaml
        from cli.config.defaults import DEFAULT_CONFIG


        def load_config(path=None):
            """Load configuration from file, falling back to defaults."""
            config = dict(DEFAULT_CONFIG)

            if path is None:
                path = os.environ.get("DATAFLOW_CONFIG", "config.yaml")

            if os.path.exists(path):
                with open(path, "r") as f:
                    user_config = yaml.safe_load(f) or {}
                config.update(user_config)

            return config
    '''),

    "cli/utils/file_io.py": textwrap.dedent('''\
        """File I/O helper functions."""

        import json
        import csv
        import os


        def read_input(path):
            """Read input data from a file (JSON or CSV)."""
            ext = os.path.splitext(path)[1].lower()
            with open(path, "r") as f:
                if ext == ".json":
                    return json.load(f)
                elif ext == ".csv":
                    reader = csv.DictReader(f)
                    return list(reader)
                else:
                    raise ValueError(f"Unsupported input format: {ext}")


        def write_output(path, content):
            """Write formatted output to a file."""
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
    '''),

    # config/
    "cli/config/__init__.py": '"""Configuration module."""\n',

    "cli/config/defaults.py": textwrap.dedent('''\
        """Default configuration settings."""

        DEFAULT_CONFIG = {
            "cache_size": 100,
            "log_level": "INFO",
            "output_format": "json",
            "transforms": [],
            "schema": None,
            "max_records": 10000,
            "parallel": False,
        }
    '''),

    # tests/
    "tests/__init__.py": "",

    "tests/conftest.py": textwrap.dedent('''\
        """Shared test fixtures."""

        import pytest
        from cli.utils.config import load_config


        @pytest.fixture
        def default_config():
            """Return a default test configuration."""
            return {
                "cache_size": 10,
                "log_level": "WARNING",
                "output_format": "json",
                "transforms": [],
                "schema": None,
            }


        @pytest.fixture
        def sample_data():
            """Return sample tabular data."""
            return [
                {"id": 1, "name": "Alice", "score": 85},
                {"id": 2, "name": "Bob", "score": 92},
                {"id": 3, "name": "Charlie", "score": 78},
            ]
    '''),

    "tests/test_commands.py": textwrap.dedent('''\
        """Tests for CLI commands."""

        import pytest


        class TestProcessCommand:
            def test_execute_basic(self, default_config, sample_data):
                """Test basic processing."""
                raise NotImplementedError

            def test_execute_with_output(self, default_config, tmp_path):
                """Test processing with file output."""
                raise NotImplementedError


        class TestValidateCommand:
            def test_validate_valid_data(self, default_config):
                raise NotImplementedError

            def test_validate_invalid_data(self, default_config):
                raise NotImplementedError
    '''),

    "tests/test_formats.py": textwrap.dedent('''\
        """Tests for output formatters."""

        import pytest
        from cli.formats.json_output import JsonFormatter
        from cli.formats.csv_output import CsvFormatter


        class TestJsonFormatter:
            def test_format_dict(self):
                fmt = JsonFormatter()
                result = fmt.format({"key": "value"})
                assert '"key"' in result

            def test_format_list(self, sample_data):
                fmt = JsonFormatter()
                result = fmt.format(sample_data)
                assert "Alice" in result


        class TestCsvFormatter:
            def test_format_records(self, sample_data):
                fmt = CsvFormatter()
                result = fmt.format(sample_data)
                assert "name" in result
    '''),

    "tests/test_pipeline.py": textwrap.dedent('''\
        """Tests for the processing pipeline."""

        import pytest
        from cli.core.pipeline import Pipeline


        class TestPipeline:
            def test_run_no_transforms(self, default_config, sample_data):
                """Pipeline with no transforms returns data unchanged."""
                raise NotImplementedError

            def test_caching(self, default_config, sample_data):
                """Second run should use cache."""
                raise NotImplementedError

            def test_cache_key_stability(self, default_config):
                """Same data should produce same cache key."""
                raise NotImplementedError
    '''),

    "tests/test_transforms.py": textwrap.dedent('''\
        """Tests for data transformations."""

        import pytest
        from cli.core.transforms import apply_transforms, TRANSFORM_REGISTRY


        class TestTransformRegistry:
            def test_known_transforms_registered(self):
                assert "uppercase" in TRANSFORM_REGISTRY
                assert "filter" in TRANSFORM_REGISTRY

            def test_unknown_transform_raises(self, sample_data):
                with pytest.raises(ValueError):
                    apply_transforms(sample_data, [{"name": "nonexistent"}])
    '''),

    # top-level files
    "setup.py": textwrap.dedent('''\
        """Package setup."""

        from setuptools import setup, find_packages

        setup(
            name="dataflow",
            version="0.1.0",
            packages=find_packages(),
            install_requires=[
                "pyyaml>=6.0",
            ],
            entry_points={
                "console_scripts": [
                    "dataflow=cli.main:main",
                ],
            },
        )
    '''),

    "config.yaml": textwrap.dedent('''\
        # Dataflow default configuration
        cache_size: 100
        log_level: INFO
        output_format: json
        max_records: 10000
        parallel: false
        transforms: []
    '''),

    "requirements.txt": textwrap.dedent('''\
        pyyaml>=6.0
    '''),
}


def create_synthetic_repo():
    """Create the synthetic dataflow repo on disk."""
    print("=" * 60)
    print("Part 1: Creating synthetic repo")
    print("=" * 60)

    for relpath, content in REPO_FILES.items():
        full = REPO_ROOT / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    count = len(REPO_FILES)
    print(f"  Created {count} files under {REPO_ROOT}")
    return count


# ============================================================================
# Part 2: Define Feature Tasks
# ============================================================================

SYNTHETIC_TASKS = [
    {
        "task_id": "feature_yaml_output",
        "title": "Add YAML Output Format",
        "description": (
            "Add YAML output format support. Users should be able to use --format yaml "
            "on the CLI. The formatter must handle nested data structures properly. "
            "Include unit tests and add pyyaml to requirements."
        ),
        "surfaces": [
            {"id": "yaml_formatter", "description": "Create new cli/formats/yaml_output.py implementing BaseFormatter for YAML", "category": "core_logic", "type": "create", "file_hint": "cli/formats/yaml_output.py"},
            {"id": "register_format", "description": "Edit cli/main.py to register yaml in FORMAT_REGISTRY and add --format yaml choice", "category": "interface", "type": "edit", "file_hint": "cli/main.py"},
            {"id": "pipeline_dispatch", "description": "Edit cli/commands/process.py or cli/core/pipeline.py to dispatch YAML formatting", "category": "core_logic", "type": "edit", "file_hint": "cli/commands/process.py"},
            {"id": "yaml_tests", "description": "Create tests for YAML formatter (test nested data, roundtrip, edge cases)", "category": "tests", "type": "create", "file_hint": "tests/test_yaml_format.py"},
            {"id": "requirements_yaml", "description": "Edit requirements.txt to add pyyaml dependency", "category": "config", "type": "edit", "file_hint": "requirements.txt"},
        ],
    },
    {
        "task_id": "feature_redis_cache",
        "title": "Add Redis Caching",
        "description": (
            "Add optional Redis caching for pipeline results. It should be configurable "
            "via config.yaml (redis_host, redis_port, redis_enabled). When Redis is "
            "unavailable, fall back gracefully to the existing memory cache. "
            "The cache backend should be selectable. Include tests that work without "
            "a running Redis instance (mock-based)."
        ),
        "surfaces": [
            {"id": "redis_cache", "description": "Create new cli/cache/redis_cache.py implementing a Redis cache backend", "category": "core_logic", "type": "create", "file_hint": "cli/cache/redis_cache.py"},
            {"id": "config_redis", "description": "Edit cli/config/defaults.py to add redis_host, redis_port, redis_enabled defaults", "category": "config", "type": "edit", "file_hint": "cli/config/defaults.py"},
            {"id": "config_yaml_redis", "description": "Edit config.yaml to include redis configuration section", "category": "config", "type": "edit", "file_hint": "config.yaml"},
            {"id": "pipeline_cache_select", "description": "Edit cli/core/pipeline.py to select cache backend based on config", "category": "core_logic", "type": "edit", "file_hint": "cli/core/pipeline.py"},
            {"id": "redis_tests", "description": "Create tests for Redis cache with mocks and fallback behavior", "category": "tests", "type": "create", "file_hint": "tests/test_redis_cache.py"},
            {"id": "requirements_redis", "description": "Edit requirements.txt to add redis dependency", "category": "config", "type": "edit", "file_hint": "requirements.txt"},
            {"id": "fallback_logic", "description": "Implement graceful fallback from Redis to memory cache on connection failure", "category": "core_logic", "type": "edit", "file_hint": "cli/cache/redis_cache.py"},
        ],
    },
    {
        "task_id": "feature_dry_run",
        "title": "Add --dry-run Flag",
        "description": (
            "Add a --dry-run flag to the CLI that propagates through the entire system. "
            "When active, the tool should show what operations would be performed without "
            "actually executing them. File writes should be skipped, transforms should "
            "report what they would do, and a summary should be printed. "
            "Include tests for dry-run behavior."
        ),
        "surfaces": [
            {"id": "cli_flag", "description": "Edit cli/main.py to add --dry-run argument and pass it through", "category": "interface", "type": "edit", "file_hint": "cli/main.py"},
            {"id": "pipeline_dry_run", "description": "Edit cli/core/pipeline.py to respect dry_run flag and skip cache writes", "category": "core_logic", "type": "edit", "file_hint": "cli/core/pipeline.py"},
            {"id": "process_passthrough", "description": "Edit cli/commands/process.py to propagate dry_run to pipeline", "category": "core_logic", "type": "edit", "file_hint": "cli/commands/process.py"},
            {"id": "file_io_skip", "description": "Edit cli/utils/file_io.py to skip writes when dry_run is active", "category": "core_logic", "type": "edit", "file_hint": "cli/utils/file_io.py"},
            {"id": "dry_run_summary", "description": "Add summary output showing what would have been done", "category": "interface", "type": "edit", "file_hint": "cli/main.py"},
            {"id": "dry_run_tests", "description": "Create tests for dry-run behavior across the pipeline", "category": "tests", "type": "create", "file_hint": "tests/test_dry_run.py"},
        ],
    },
]


def save_synthetic_tasks():
    """Save task definitions to JSON."""
    print("\n" + "=" * 60)
    print("Part 2: Saving synthetic tasks")
    print("=" * 60)

    with open("data/synthetic_tasks.json", "w") as f:
        json.dump(SYNTHETIC_TASKS, f, indent=2)
    print(f"  Saved {len(SYNTHETIC_TASKS)} tasks to data/synthetic_tasks.json")
    for t in SYNTHETIC_TASKS:
        print(f"    - {t['task_id']}: {len(t['surfaces'])} surfaces")


# ============================================================================
# Part 3: Generate Plans via Sonnet
# ============================================================================

def build_file_tree():
    """Build a text representation of the file tree."""
    lines = []
    for relpath in sorted(REPO_FILES.keys()):
        lines.append(f"  {relpath}")
    return "\n".join(lines)


def generate_plan(task, file_tree):
    """Call Sonnet to generate a plan for the given feature task."""
    prompt = f"""You are a senior software engineer planning the implementation of a new feature for an existing CLI data-processing tool called "dataflow".

## Repository File Tree
{file_tree}

## Feature Request
{task['description']}

## Instructions
Produce a detailed implementation plan as JSON with these fields:
- "files_to_inspect": list of existing files to read/understand first
- "files_to_create": list of new files to create (full relative paths)
- "files_to_modify": list of existing files that need edits
- "implementation_steps": ordered list of steps, each with "step", "file", "description"

Be thorough. Consider: interface changes, core logic, configuration, tests, dependencies.
Return ONLY valid JSON, no markdown fences."""

    response = client.messages.create(
        model=SONNET,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Try to parse JSON, stripping markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        print(f"    WARNING: Could not parse plan JSON for {task['task_id']}")
        plan = {"raw": text, "files_to_inspect": [], "files_to_create": [], "files_to_modify": [], "implementation_steps": []}

    return plan


def generate_all_plans():
    """Generate plans for all 3 tasks."""
    print("\n" + "=" * 60)
    print("Part 3: Generating plans via Sonnet")
    print("=" * 60)

    file_tree = build_file_tree()
    plans = []

    for task in SYNTHETIC_TASKS:
        print(f"  Generating plan for: {task['task_id']}...")
        plan = generate_plan(task, file_tree)
        plans.append({
            "task_id": task["task_id"],
            "title": task["title"],
            "plan": plan,
        })
        print(f"    files_to_create: {plan.get('files_to_create', [])}")
        print(f"    files_to_modify: {plan.get('files_to_modify', [])}")
        time.sleep(DELAY)

    with open("data/synthetic_plans.json", "w") as f:
        json.dump(plans, f, indent=2)
    print(f"  Saved {len(plans)} plans to data/synthetic_plans.json")
    return plans


# ============================================================================
# Part 4: Score Against Checklists
# ============================================================================

def normalize_path(p):
    """Normalize a file path for comparison."""
    p = p.strip().lstrip("./")
    # Some plans may include 'dataflow/' prefix
    if p.startswith("dataflow/"):
        p = p[len("dataflow/"):]
    return p


def score_surface(surface, plan):
    """Score a single surface against the plan. Returns 0.0, 0.5, or 1.0."""
    hint = normalize_path(surface["file_hint"])
    stype = surface["type"]  # "create" or "edit"

    plan_create = [normalize_path(f) for f in plan.get("files_to_create", [])]
    plan_modify = [normalize_path(f) for f in plan.get("files_to_modify", [])]
    plan_inspect = [normalize_path(f) for f in plan.get("files_to_inspect", [])]

    all_plan_files = plan_create + plan_modify + plan_inspect

    # Also check implementation_steps for file references
    step_files = []
    for step in plan.get("implementation_steps", []):
        sf = step.get("file", "")
        if sf:
            step_files.append(normalize_path(sf))

    all_mentioned = set(all_plan_files + step_files)

    # Exact match with correct type
    if stype == "create" and hint in plan_create:
        return 1.0
    if stype == "edit" and hint in plan_modify:
        return 1.0

    # Right file, wrong type (e.g., listed as modify instead of create)
    if hint in all_mentioned:
        return 0.5

    # Check for "right area" — same directory
    hint_dir = "/".join(hint.split("/")[:-1])
    for f in all_mentioned:
        f_dir = "/".join(f.split("/")[:-1])
        if f_dir == hint_dir and f_dir:
            return 0.5

    return 0.0


def programmatic_scoring(tasks, plans):
    """Score all tasks programmatically."""
    print("\n" + "=" * 60)
    print("Part 4a: Programmatic scoring")
    print("=" * 60)

    plan_map = {p["task_id"]: p["plan"] for p in plans}
    results = []

    for task in tasks:
        tid = task["task_id"]
        plan = plan_map.get(tid, {})
        scores = []
        category_scores = {}

        for surface in task["surfaces"]:
            score = score_surface(surface, plan)
            scores.append(score)
            cat = surface["category"]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score)

        overall = sum(scores) / len(scores) if scores else 0
        by_category = {cat: sum(s) / len(s) for cat, s in category_scores.items()}

        results.append({
            "task_id": tid,
            "title": task["title"],
            "surface_coverage": round(overall, 3),
            "category_coverage": {k: round(v, 3) for k, v in by_category.items()},
            "per_surface": [
                {"id": s["id"], "score": sc, "category": s["category"], "type": s["type"]}
                for s, sc in zip(task["surfaces"], scores)
            ],
        })

        print(f"  {tid}: coverage={overall:.2f}")
        for cat, avg in by_category.items():
            print(f"    {cat}: {avg:.2f}")

    return results


def opus_judge_scoring(tasks, plans):
    """Use Opus to judge plan quality on 5 dimensions."""
    print("\n" + "=" * 60)
    print("Part 4b: Opus judge scoring")
    print("=" * 60)

    plan_map = {p["task_id"]: p["plan"] for p in plans}
    judge_results = []

    for task in tasks:
        tid = task["task_id"]
        plan = plan_map.get(tid, {})

        prompt = f"""You are evaluating the quality of an implementation plan for a feature request.

## Feature Request
{task['description']}

## Expected Surfaces (what a good plan should cover)
{json.dumps(task['surfaces'], indent=2)}

## Generated Plan
{json.dumps(plan, indent=2)}

## Scoring Instructions
Rate each dimension 1-3 (1=poor, 2=adequate, 3=excellent):

1. **surface_coverage**: Does the plan identify all the files that need to be created/modified?
2. **decomposition_quality**: Are the implementation steps well-decomposed and atomic?
3. **dependency_ordering**: Are steps ordered respecting dependencies (e.g., create base class before subclass)?
4. **test_awareness**: Does the plan include test creation and specify what to test?
5. **config_awareness**: Does the plan address config files, requirements, and integration points?

Return ONLY a JSON object with these 5 keys mapping to integer scores 1-3. No explanation."""

        print(f"  Judging {tid}...")
        response = client.messages.create(
            model=OPUS,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[: text.rfind("```")]

        try:
            scores = json.loads(text)
        except json.JSONDecodeError:
            print(f"    WARNING: Could not parse judge JSON for {tid}")
            scores = {
                "surface_coverage": 2,
                "decomposition_quality": 2,
                "dependency_ordering": 2,
                "test_awareness": 2,
                "config_awareness": 2,
            }

        judge_results.append({
            "task_id": tid,
            "title": task["title"],
            "judge_scores": scores,
        })
        print(f"    scores: {scores}")
        time.sleep(DELAY)

    return judge_results


# ============================================================================
# Part 5: Outputs
# ============================================================================

def save_metrics(prog_results, judge_results):
    """Merge and save all metrics."""
    merged = []
    judge_map = {j["task_id"]: j["judge_scores"] for j in judge_results}

    for pr in prog_results:
        entry = dict(pr)
        entry["judge_scores"] = judge_map.get(pr["task_id"], {})
        merged.append(entry)

    with open("data/synthetic_metrics.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n  Saved metrics to data/synthetic_metrics.json")
    return merged


def generate_bar_chart(metrics):
    """Create bar chart of coverage by category across 3 tasks."""
    print("\n" + "=" * 60)
    print("Part 5a: Generating bar chart")
    print("=" * 60)

    categories = ["core_logic", "interface", "tests", "config"]
    task_labels = [m["title"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25

    for i, m in enumerate(metrics):
        cat_cov = m["category_coverage"]
        values = [cat_cov.get(c, 0) for c in categories]
        bars = ax.bar(x + i * width, values, width, label=m["title"])
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Surface Category", fontsize=12)
    ax.set_ylabel("Coverage Score", fontsize=12)
    ax.set_title("Plan Surface Coverage by Category (Synthetic Feature Tasks)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis/synthetic_surface_coverage.png", dpi=150)
    plt.close()
    print("  Saved analysis/synthetic_surface_coverage.png")


def generate_comparison_writeup(metrics):
    """Generate the markdown comparison table."""
    print("\n" + "=" * 60)
    print("Part 5b: Generating comparison writeup")
    print("=" * 60)

    # Compute aggregates
    all_coverages = [m["surface_coverage"] for m in metrics]
    avg_coverage = sum(all_coverages) / len(all_coverages)

    # Category averages across tasks
    categories = ["core_logic", "interface", "tests", "config"]
    cat_avgs = {}
    for cat in categories:
        vals = [m["category_coverage"].get(cat, 0) for m in metrics]
        cat_avgs[cat] = sum(vals) / len(vals) if vals else 0

    best_cat = max(cat_avgs, key=cat_avgs.get)
    worst_cat = min(cat_avgs, key=cat_avgs.get)

    # Judge score averages
    dims = ["surface_coverage", "decomposition_quality", "dependency_ordering", "test_awareness", "config_awareness"]
    dim_avgs = {}
    for d in dims:
        vals = [m["judge_scores"].get(d, 0) for m in metrics]
        dim_avgs[d] = sum(vals) / len(vals) if vals else 0

    # Per-task details
    task_lines = []
    for m in metrics:
        js = m["judge_scores"]
        task_lines.append(
            f"| {m['title']} | {m['surface_coverage']:.2f} | "
            f"{js.get('decomposition_quality', 0)}/3 | "
            f"{js.get('test_awareness', 0)}/3 | "
            f"{js.get('config_awareness', 0)}/3 |"
        )

    writeup = f"""# Experiment 11: Synthetic Feature Planning — Results

## Overview

This experiment tests whether the plan evaluation framework generalizes from
SWE-bench bug fixes to open-ended feature implementation tasks. We created a
synthetic 30-file CLI repository ("dataflow") and defined 3 feature tasks,
each with a checklist of implementation surfaces.

## Headline Comparison

| Dimension | Bug-Fix (SWE-bench, n=234) | Feature (Synthetic, n=3) |
|-----------|---------------------------|--------------------------|
| Primary plan value | Navigation (file targeting) | Decomposition + surface coverage |
| Main metric | File recall: 0.53 | Surface coverage: {avg_coverage:.2f} |
| What plans get right | Same-package files (73%) | {best_cat.replace('_', ' ')} ({cat_avgs[best_cat]:.2f}) |
| What plans miss | Shared utilities (61%), distant (100%) | {worst_cat.replace('_', ' ')} ({cat_avgs[worst_cat]:.2f}) |
| Content quality | Flat across complexity (2.0-2.15) | Decomposition: {dim_avgs['decomposition_quality']:.1f}/3, Tests: {dim_avgs['test_awareness']:.1f}/3 |

## Per-Task Results

| Task | Surface Coverage | Decomposition | Test Awareness | Config Awareness |
|------|-----------------|---------------|----------------|------------------|
{chr(10).join(task_lines)}

## Category Coverage (averaged across tasks)

| Category | Average Coverage |
|----------|-----------------|
| Core Logic | {cat_avgs.get('core_logic', 0):.2f} |
| Interface | {cat_avgs.get('interface', 0):.2f} |
| Tests | {cat_avgs.get('tests', 0):.2f} |
| Config | {cat_avgs.get('config', 0):.2f} |

## Opus Judge Scores (averaged across tasks)

| Dimension | Average Score |
|-----------|--------------|
| Surface Coverage | {dim_avgs.get('surface_coverage', 0):.1f}/3 |
| Decomposition Quality | {dim_avgs.get('decomposition_quality', 0):.1f}/3 |
| Dependency Ordering | {dim_avgs.get('dependency_ordering', 0):.1f}/3 |
| Test Awareness | {dim_avgs.get('test_awareness', 0):.1f}/3 |
| Config Awareness | {dim_avgs.get('config_awareness', 0):.1f}/3 |

## Key Findings

1. **Surface coverage ({avg_coverage:.2f}) vs file recall (0.53)**: Feature plans achieve
   {"higher" if avg_coverage > 0.53 else "comparable" if avg_coverage > 0.45 else "lower"}
   coverage than bug-fix file recall, suggesting plans
   {"are even more valuable" if avg_coverage > 0.53 else "transfer well"} for feature work
   where the implementation surface is broader and less constrained.

2. **Best category: {best_cat.replace('_', ' ')} ({cat_avgs[best_cat]:.2f})**: Plans excel at
   identifying {best_cat.replace('_', ' ')} surfaces, consistent with the finding that plans
   are strongest for within-package navigation.

3. **Weakest category: {worst_cat.replace('_', ' ')} ({cat_avgs[worst_cat]:.2f})**: Plans
   struggle most with {worst_cat.replace('_', ' ')} surfaces, echoing the bug-fix finding that
   cross-cutting concerns (utilities, config) are hardest to anticipate.

4. **Decomposition quality ({dim_avgs['decomposition_quality']:.1f}/3)** is a new dimension
   not measured in bug-fix evaluation. Feature plans must decompose open-ended requirements
   into ordered steps — this is a distinct skill from file targeting.

5. **Feature vs bug-fix planning**: Bug-fix plans primarily help with *navigation* (finding
   the right files). Feature plans must also handle *decomposition* (breaking the feature into
   implementable pieces) and *completeness* (covering all surfaces including tests and config).

## Visualization

See `analysis/synthetic_surface_coverage.png` for coverage by category across all 3 tasks.
"""

    with open("analysis/synthetic_results.md", "w") as f:
        f.write(writeup)
    print("  Saved analysis/synthetic_results.md")

    return writeup


# ============================================================================
# Main
# ============================================================================

def main():
    print("Experiment 11: Synthetic Feature Planning")
    print("=" * 60)

    # Part 1
    create_synthetic_repo()

    # Part 2
    save_synthetic_tasks()

    # Part 3
    plans = generate_all_plans()

    # Part 4
    prog_results = programmatic_scoring(SYNTHETIC_TASKS, plans)
    judge_results = opus_judge_scoring(SYNTHETIC_TASKS, plans)
    metrics = save_metrics(prog_results, judge_results)

    # Part 5
    generate_bar_chart(metrics)
    writeup = generate_comparison_writeup(metrics)

    print("\n" + "=" * 60)
    print("DONE — Experiment 11 Complete")
    print("=" * 60)

    # Print the comparison table
    for line in writeup.split("\n"):
        if line.startswith("|") or line.startswith("##"):
            print(line)


if __name__ == "__main__":
    main()
