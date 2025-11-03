"""
Performance tests for Hydra config loading.

These tests measure:
- Config loading time
- Memory usage
- Override performance
"""

import os
import time
from typing import Dict, List

import pytest

from farm.config import load_config


class TestConfigLoadingPerformance:
    """Performance benchmarks for config loading."""

    def test_hydra_loading_time(self, benchmark):
        """Benchmark Hydra config loading time."""
        def load():
            return load_config(environment="development")
        
        config = benchmark(load)
        assert config is not None

    def test_loading_time_benchmark(self):
        """Benchmark config loading time."""
        # Warm up
        load_config(environment="development")
        
        # Measure loading time
        times = []
        for _ in range(10):
            start = time.time()
            load_config(environment="development")
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # Should be reasonably fast (< 1 second)
        assert avg_time < 1.0, f"Loading too slow: {avg_time:.4f}s"
        
        # Log results
        print(f"\nAverage loading time: {avg_time:.4f}s")

    def test_override_performance(self):
        """Test performance with overrides."""
        overrides = [
            "simulation_steps=200",
            "population.system_agents=50",
            "environment.width=200",
            "environment.height=200",
        ]
        
        times = []
        for _ in range(10):
            start = time.time()
            load_config(
                environment="development",
                use_hydra=True,
                overrides=overrides,
            )
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Override loading too slow: {avg_time:.4f}s"
        print(f"\nOverride avg: {avg_time:.4f}s")

    def test_multiple_environment_loading(self):
        """Test loading different environments."""
        environments = ["development", "production", "testing"]
        
        times = []
        for env in environments:
            start = time.time()
            load_config(environment=env)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Multi-env loading too slow: {avg_time:.4f}s"


class TestConfigMemoryUsage:
    """Test memory usage of config objects."""

    def test_config_memory_size(self):
        """Test that config objects are reasonably sized."""
        import sys
        
        config = load_config(environment="development")
        
        config_size = sys.getsizeof(config)
        
        # Config should be reasonably sized (< 10MB for the object itself)
        print(f"\nConfig size: {config_size} bytes")
        assert config_size < 10 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
