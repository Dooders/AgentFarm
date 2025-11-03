"""
Performance tests comparing Hydra vs legacy config loading.

These tests measure:
- Config loading time
- Memory usage
- Caching effectiveness
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
            return load_config(environment="development", use_hydra=True)
        
        config = benchmark(load)
        assert config is not None

    def test_legacy_loading_time(self, benchmark):
        """Benchmark legacy config loading time."""
        def load():
            return load_config(environment="development", use_hydra=False)
        
        config = benchmark(load)
        assert config is not None

    def test_comparative_loading_time(self):
        """Compare loading times between Hydra and legacy."""
        # Warm up
        load_config(environment="development", use_hydra=True)
        load_config(environment="development", use_hydra=False)
        
        # Measure Hydra
        hydra_times = []
        for _ in range(10):
            start = time.time()
            load_config(environment="development", use_hydra=True)
            hydra_times.append(time.time() - start)
        
        # Measure Legacy
        legacy_times = []
        for _ in range(10):
            start = time.time()
            load_config(environment="development", use_hydra=False)
            legacy_times.append(time.time() - start)
        
        avg_hydra = sum(hydra_times) / len(hydra_times)
        avg_legacy = sum(legacy_times) / len(legacy_times)
        
        # Both should be reasonably fast (< 1 second)
        assert avg_hydra < 1.0, f"Hydra loading too slow: {avg_hydra:.4f}s"
        assert avg_legacy < 1.0, f"Legacy loading too slow: {avg_legacy:.4f}s
        
        # Log results
        print(f"\nHydra avg: {avg_hydra:.4f}s")
        print(f"Legacy avg: {avg_legacy:.4f}s")
        print(f"Ratio: {avg_hydra/avg_legacy:.2f}x")

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
            load_config(environment=env, use_hydra=True)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Multi-env loading too slow: {avg_time:.4f}s"


class TestConfigMemoryUsage:
    """Test memory usage of config objects."""

    def test_config_memory_size(self):
        """Test that config objects are reasonably sized."""
        import sys
        
        hydra_config = load_config(environment="development", use_hydra=True)
        legacy_config = load_config(environment="development", use_hydra=False)
        
        hydra_size = sys.getsizeof(hydra_config)
        legacy_size = sys.getsizeof(legacy_config)
        
        # Both should be similar (same dataclass structure)
        # Hydra might be slightly larger due to OmegaConf conversion overhead
        print(f"\nHydra config size: {hydra_size} bytes")
        print(f"Legacy config size: {legacy_size} bytes")
        
        # Configs should be reasonably sized (< 10MB for the object itself)
        assert hydra_size < 10 * 1024 * 1024
        assert legacy_size < 10 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
