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
