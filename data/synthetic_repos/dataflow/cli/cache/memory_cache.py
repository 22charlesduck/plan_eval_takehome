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
