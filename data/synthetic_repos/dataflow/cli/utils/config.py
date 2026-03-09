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
