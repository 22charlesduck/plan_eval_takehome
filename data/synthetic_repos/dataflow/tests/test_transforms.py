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
