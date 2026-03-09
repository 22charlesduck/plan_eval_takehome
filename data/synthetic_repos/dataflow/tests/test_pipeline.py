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
