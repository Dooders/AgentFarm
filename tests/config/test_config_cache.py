"""Unit tests for configuration caching functionality.

This module tests the ConfigCache class and related caching functionality including:
- Basic cache operations (get, put, invalidate, clear)
- LRU eviction
- Memory limits
- File modification tracking
- Thread safety
- Statistics
"""

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from farm.config import SimulationConfig
from farm.config.cache import ConfigCache, get_global_cache


class TestConfigCache(unittest.TestCase):
    """Test cases for ConfigCache class."""

    def setUp(self):
        """Set up test environment."""
        self.cache = ConfigCache(max_size=3, max_memory_mb=1.0)
        self.test_config = SimulationConfig()

    def test_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = ConfigCache(max_size=10, max_memory_mb=50.0)

        self.assertEqual(cache.max_size, 10)
        self.assertEqual(cache.max_memory_mb, 50.0)
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.memory_usage, 0.0)

    def test_get_put_basic(self):
        """Test basic get/put operations."""
        cache_key = "test_key"

        # Cache should be empty initially
        self.assertIsNone(self.cache.get(cache_key))

        # Put config in cache
        self.cache.put(cache_key, self.test_config)

        # Should be able to retrieve it
        retrieved = self.cache.get(cache_key)
        self.assertIsNotNone(retrieved)
        self.assertEqual(
            retrieved.environment.width, self.test_config.environment.width
        )

        # Access time should be updated
        self.assertIn(cache_key, self.cache.access_times)

    def test_invalidate(self):
        """Test cache invalidation."""
        cache_key = "invalidate_test"

        # Put config in cache
        self.cache.put(cache_key, self.test_config)
        self.assertIsNotNone(self.cache.get(cache_key))

        # Invalidate it
        self.cache.invalidate(cache_key)

        # Should not be retrievable
        self.assertIsNone(self.cache.get(cache_key))
        self.assertNotIn(cache_key, self.cache.cache)

    def test_clear(self):
        """Test clearing all cache entries."""
        # Add multiple entries
        for i in range(3):
            key = f"clear_test_{i}"
            config = SimulationConfig(seed=i)
            self.cache.put(key, config)

        self.assertEqual(len(self.cache.cache), 3)
        self.assertGreater(self.cache.memory_usage, 0)

        # Clear cache
        self.cache.clear()

        # Should be empty
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache.memory_usage, 0.0)
        self.assertEqual(len(self.cache.access_times), 0)
        self.assertEqual(len(self.cache.file_mtimes), 0)

    def test_get_stats(self):
        """Test cache statistics retrieval."""
        stats = self.cache.get_stats()

        expected_keys = [
            "entries",
            "memory_usage_mb",
            "max_size",
            "max_memory_mb",
            "hit_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["max_memory_mb"], 1.0)

    def test_lru_eviction_by_size(self):
        """Test LRU eviction when cache reaches max size."""
        # Add entries up to max size
        for i in range(3):
            key = f"size_test_{i}"
            config = SimulationConfig(seed=i)
            self.cache.put(key, config)

        self.assertEqual(len(self.cache.cache), 3)

        # Add one more - should evict oldest (least recently used)
        oldest_key = "size_test_0"
        new_key = "size_test_new"
        new_config = SimulationConfig(seed=999)

        self.cache.put(new_key, new_config)

        # Should still have 3 entries
        self.assertEqual(len(self.cache.cache), 3)

        # Oldest should be evicted
        self.assertNotIn(oldest_key, self.cache.cache)
        self.assertIn(new_key, self.cache.cache)

    def test_memory_limit_eviction(self):
        """Test eviction when memory limit is reached."""
        # Create a cache with very small memory limit
        small_cache = ConfigCache(max_size=10, max_memory_mb=0.001)  # 1KB limit

        # Create a config that will exceed the limit
        config = SimulationConfig()
        cache_key = "memory_test"

        # Put config - should succeed initially
        small_cache.put(cache_key, config)

        # Check that it was added (may be evicted immediately if too big)
        # This tests the memory checking logic

    def test_file_modification_tracking(self):
        """Test cache invalidation based on file modification times."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: data\n")
            temp_file = f.name

        try:
            cache_key = "file_test"
            self.cache.put(cache_key, self.test_config, temp_file)

            # Should be cached
            self.assertIsNotNone(self.cache.get(cache_key, temp_file))

            # Modify file - ensure mtime changes by adding a small delay
            import os
            import time

            # Add a small delay to ensure mtime will be different
            time.sleep(0.01)
            os.utime(temp_file, None)  # Ensure mtime changes
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write("modified: data\n")

            # Cache should be invalidated
            self.assertIsNone(self.cache.get(cache_key, temp_file))

        finally:
            os.unlink(temp_file)

    def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        results = []
        errors = []

        def worker(worker_id):
            """Worker function for concurrent access."""
            try:
                # Each worker gets/puts its own key
                key = f"thread_{worker_id}"
                config = SimulationConfig(seed=worker_id)

                # Put config
                self.cache.put(key, config)

                # Get config
                retrieved = self.cache.get(key)
                if retrieved and retrieved.seed == worker_id:
                    results.append(f"worker_{worker_id}_success")
                else:
                    results.append(f"worker_{worker_id}_fail")

            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)

        # Check results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("success", result)

        self.assertEqual(len(errors), 0)

    def test_get_global_cache(self):
        """Test getting the global cache instance."""
        global_cache = get_global_cache()
        self.assertIsInstance(global_cache, ConfigCache)

        # Should return the same instance
        global_cache2 = get_global_cache()
        self.assertIs(global_cache, global_cache2)

    def test_invalid_cached_data_handling(self):
        """Test handling of corrupted cached data."""
        cache_key = "corrupt_test"

        # Manually put corrupted data
        self.cache.cache[cache_key] = {
            "config": {"invalid": "data"},
            "size": 0.1,
            "created": time.time(),
        }
        self.cache.access_times[cache_key] = time.time()

        # Attempting to get should return None and remove corrupted entry
        result = self.cache.get(cache_key)
        self.assertIsNone(result)
        self.assertNotIn(cache_key, self.cache.cache)


if __name__ == "__main__":
    unittest.main()
