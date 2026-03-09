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
