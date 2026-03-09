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
