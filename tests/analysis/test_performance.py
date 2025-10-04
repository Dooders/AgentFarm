"""
Performance tests for analysis modules.

Benchmarks execution time, memory usage, and scalability.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple
import gc

from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.analysis.common.context import AnalysisContext
from farm.core.services import EnvConfigService


@pytest.fixture
def config_service():
    """Create configuration service for testing."""
    return EnvConfigService()


@pytest.fixture
def large_experiment_data(tmp_path):
    """Create large experiment dataset for performance testing."""
    exp_path = tmp_path / "large_experiment"
    exp_path.mkdir()

    # Create large simulation database
    db_path = exp_path / "simulation.db"
    db_path.write_bytes(b"large mock database content")

    # Generate large datasets
    n_steps = 1000  # Large dataset
    n_agents = 500

    # Population data
    pop_data = pd.DataFrame({
        'step': np.repeat(range(n_steps), n_agents),
        'agent_id': np.tile(range(n_agents), n_steps),
        'total_agents': np.random.randint(400, 600, n_steps * n_agents),
        'system_agents': np.random.randint(200, 300, n_steps * n_agents),
        'independent_agents': np.random.randint(150, 250, n_steps * n_agents),
        'control_agents': np.random.randint(50, 100, n_steps * n_agents),
        'avg_resources': np.random.uniform(40, 60, n_steps * n_agents),
        'births': np.random.randint(0, 10, n_steps * n_agents),
        'deaths': np.random.randint(0, 8, n_steps * n_agents),
    })

    # Resource data
    resource_data = pd.DataFrame({
        'step': range(n_steps),
        'total_resources': np.linspace(10000, 1000, n_steps) + np.random.normal(0, 500, n_steps),
        'consumed_resources': np.random.uniform(50, 150, n_steps),
        'resource_efficiency': np.random.uniform(0.7, 0.95, n_steps),
        'hotspot_count': np.random.randint(1, 10, n_steps),
    })

    # Action data
    action_types = ['move', 'gather', 'attack', 'reproduce', 'defend']
    action_data = pd.DataFrame({
        'step': range(n_steps),
        'action_type': np.random.choice(action_types, n_steps),
        'success_rate': np.random.uniform(0.6, 1.0, n_steps),
        'reward': np.random.normal(1.0, 0.5, n_steps),
        'frequency': np.random.randint(5, 50, n_steps),
    })

    # Agent data
    agent_data = pd.DataFrame({
        'agent_id': range(n_agents),
        'lifespan': np.random.normal(80, 20, n_agents),
        'total_actions': np.random.randint(100, 1000, n_agents),
        'success_rate': np.random.uniform(0.5, 0.95, n_agents),
        'avg_reward': np.random.normal(1.2, 0.3, n_agents),
        'agent_type': np.random.choice(['system', 'independent', 'control'], n_agents),
    })

    # Save to files
    data_dir = exp_path / "data"
    data_dir.mkdir()
    pop_data.to_csv(data_dir / "population.csv", index=False)
    resource_data.to_csv(data_dir / "resources.csv", index=False)
    action_data.to_csv(data_dir / "actions.csv", index=False)
    agent_data.to_csv(data_dir / "agents.csv", index=False)

    return exp_path


class PerformanceMetrics:
    """Helper class to track performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None

    def start_measurement(self):
        """Start performance measurement."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss

    def end_measurement(self):
        """End performance measurement."""
        self.end_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        self.end_memory = process.memory_info().rss
        # Note: peak memory tracking would require more complex monitoring

    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        return self.end_time - self.start_time

    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self.end_memory - self.start_memory

    def get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.get_memory_usage() / (1024 * 1024)


class TestAnalysisPerformance:
    """Performance tests for analysis modules."""

    def test_population_analysis_performance(self, config_service, large_experiment_data, tmp_path, benchmark):
        """Benchmark population analysis performance."""
        service = AnalysisService(config_service)

        def run_population_analysis():
            request = AnalysisRequest(
                module_name="population",
                experiment_path=large_experiment_data,
                output_path=tmp_path / "perf_pop",
                enable_caching=False
            )
            return service.run(request)

        # Use pytest-benchmark for proper benchmarking
        result = benchmark(run_population_analysis)
        assert result.success

        # Additional performance checks
        execution_time = benchmark.stats['mean']
        assert execution_time < 30.0  # Should complete in under 30 seconds

    def test_resources_analysis_performance(self, config_service, large_experiment_data, tmp_path, benchmark):
        """Benchmark resources analysis performance."""
        service = AnalysisService(config_service)

        def run_resources_analysis():
            request = AnalysisRequest(
                module_name="resources",
                experiment_path=large_experiment_data,
                output_path=tmp_path / "perf_res",
                enable_caching=False
            )
            return service.run(request)

        result = benchmark(run_resources_analysis)
        assert result.success

        execution_time = benchmark.stats['mean']
        assert execution_time < 25.0  # Should complete in under 25 seconds

    def test_actions_analysis_performance(self, config_service, large_experiment_data, tmp_path, benchmark):
        """Benchmark actions analysis performance."""
        service = AnalysisService(config_service)

        def run_actions_analysis():
            request = AnalysisRequest(
                module_name="actions",
                experiment_path=large_experiment_data,
                output_path=tmp_path / "perf_act",
                enable_caching=False
            )
            return service.run(request)

        result = benchmark(run_actions_analysis)
        assert result.success

        execution_time = benchmark.stats['mean']
        assert execution_time < 20.0  # Should complete in under 20 seconds

    def test_agents_analysis_performance(self, config_service, large_experiment_data, tmp_path, benchmark):
        """Benchmark agents analysis performance."""
        service = AnalysisService(config_service)

        def run_agents_analysis():
            request = AnalysisRequest(
                module_name="agents",
                experiment_path=large_experiment_data,
                output_path=tmp_path / "perf_agents",
                enable_caching=False
            )
            return service.run(request)

        result = benchmark(run_agents_analysis)
        assert result.success

        execution_time = benchmark.stats['mean']
        assert execution_time < 35.0  # Should complete in under 35 seconds (clustering can be expensive)

    def test_batch_analysis_performance(self, config_service, large_experiment_data, tmp_path, benchmark):
        """Benchmark batch analysis performance."""
        service = AnalysisService(config_service)

        def run_batch_analysis():
            requests = [
                AnalysisRequest(
                    module_name=module,
                    experiment_path=large_experiment_data,
                    output_path=tmp_path / f"batch_perf_{module}",
                    group="basic",
                    enable_caching=False
                )
                for module in ["population", "resources", "actions", "agents"]
            ]
            return service.run_batch(requests)

        results = benchmark(run_batch_analysis)
        assert len(results) == 4
        assert all(r.success for r in results)

        # Batch should be more efficient than running individually
        total_time = benchmark.stats['mean']
        # Allow reasonable time for batch processing
        assert total_time < 100.0

    def test_memory_usage_analysis(self, config_service, large_experiment_data, tmp_path):
        """Test memory usage during analysis."""
        service = AnalysisService(config_service)

        metrics = PerformanceMetrics()

        metrics.start_measurement()

        request = AnalysisRequest(
            module_name="population",
            experiment_path=large_experiment_data,
            output_path=tmp_path / "memory_test",
            enable_caching=False
        )

        result = service.run(request)

        metrics.end_measurement()

        assert result.success

        # Check memory usage is reasonable
        memory_mb = metrics.get_memory_usage_mb()
        assert memory_mb < 500  # Should use less than 500MB

        print(".2f")

    def test_caching_performance(self, config_service, large_experiment_data, tmp_path):
        """Test performance improvement with caching."""
        cache_dir = tmp_path / "cache"
        service = AnalysisService(config_service, cache_dir=cache_dir)

        request = AnalysisRequest(
            module_name="population",
            experiment_path=large_experiment_data,
            output_path=tmp_path / "cache_test",
            enable_caching=True
        )

        # First run
        metrics1 = PerformanceMetrics()
        metrics1.start_measurement()
        result1 = service.run(request)
        metrics1.end_measurement()

        assert result1.success
        assert not result1.cache_hit

        # Second run (cached)
        metrics2 = PerformanceMetrics()
        metrics2.start_measurement()
        result2 = service.run(request)
        metrics2.end_measurement()

        assert result2.success
        assert result2.cache_hit

        # Cached run should be significantly faster
        speedup = metrics1.get_execution_time() / metrics2.get_execution_time()
        assert speedup > 2.0  # At least 2x speedup

        print(".2f")

    def test_scalability_with_data_size(self, config_service, tmp_path):
        """Test how performance scales with data size."""
        service = AnalysisService(config_service)

        sizes = [100, 500, 1000]
        execution_times = []

        for size in sizes:
            # Create experiment data of different sizes
            exp_path = tmp_path / f"scale_test_{size}"
            exp_path.mkdir()

            # Create mock database
            db_path = exp_path / "simulation.db"
            db_path.write_bytes(b"mock database")

            # Create population data of specified size
            pop_data = pd.DataFrame({
                'step': range(size),
                'total_agents': [100 + i for i in range(size)],
                'system_agents': [50 + i//2 for i in range(size)],
            })

            data_dir = exp_path / "data"
            data_dir.mkdir()
            pop_data.to_csv(data_dir / "population.csv", index=False)

            # Time the analysis
            request = AnalysisRequest(
                module_name="population",
                experiment_path=exp_path,
                output_path=tmp_path / f"scale_output_{size}",
                enable_caching=False
            )

            start_time = time.perf_counter()
            result = service.run(request)
            end_time = time.perf_counter()

            assert result.success
            execution_times.append(end_time - start_time)

        # Check that performance scales reasonably (should be roughly linear or better)
        # For perfect linear scaling: time2/time1 ≈ size2/size1
        scaling_factor_1_to_2 = execution_times[1] / execution_times[0]
        expected_scaling_1_to_2 = sizes[1] / sizes[0]  # 500/100 = 5

        scaling_factor_2_to_3 = execution_times[2] / execution_times[1]
        expected_scaling_2_to_3 = sizes[2] / sizes[1]  # 1000/500 = 2

        # Allow some overhead but should be roughly proportional
        assert scaling_factor_1_to_2 < expected_scaling_1_to_2 * 2
        assert scaling_factor_2_to_3 < expected_scaling_2_to_3 * 2

    def test_concurrent_analysis_performance(self, config_service, large_experiment_data, tmp_path):
        """Test performance with concurrent analysis requests."""
        import concurrent.futures
        import threading

        service = AnalysisService(config_service)

        # Create multiple concurrent requests
        num_concurrent = 3
        requests = []
        for i in range(num_concurrent):
            request = AnalysisRequest(
                module_name="population",
                experiment_path=large_experiment_data,
                output_path=tmp_path / f"concurrent_{i}",
                enable_caching=False
            )
            requests.append(request)

        # Run concurrently
        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(service.run, req) for req in requests]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        end_time = time.perf_counter()

        assert len(results) == num_concurrent
        assert all(r.success for r in results)

        concurrent_time = end_time - start_time

        # Concurrent execution should not be dramatically slower than sequential
        # (though in practice it might be due to GIL and I/O bound nature)
        print(".2f")

    def test_analysis_with_different_output_formats(self, config_service, large_experiment_data, tmp_path):
        """Test performance with different output configurations."""
        service = AnalysisService(config_service)

        configs = [
            {"group": "basic", "name": "basic"},
            {"group": "all", "name": "full"},
            {"group": "plots", "name": "plots_only"},
        ]

        for config in configs:
            request = AnalysisRequest(
                module_name="population",
                experiment_path=large_experiment_data,
                output_path=tmp_path / f"format_{config['name']}",
                group=config['group'],
                enable_caching=False
            )

            start_time = time.perf_counter()
            result = service.run(request)
            end_time = time.perf_counter()

            assert result.success

            execution_time = end_time - start_time
            print(".3f")


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    def test_baseline_performance(self, config_service, large_experiment_data, tmp_path):
        """Establish baseline performance metrics."""
        service = AnalysisService(config_service)

        # Test with known dataset size
        request = AnalysisRequest(
            module_name="population",
            experiment_path=large_experiment_data,
            output_path=tmp_path / "baseline",
            enable_caching=False
        )

        start_time = time.perf_counter()
        result = service.run(request)
        end_time = time.perf_counter()

        assert result.success

        execution_time = end_time - start_time

        # Store baseline - in practice this would be compared against
        # previous runs to detect regressions
        baseline_time = execution_time

        # Assert that it's within reasonable bounds
        assert 1.0 < baseline_time < 60.0  # Between 1 second and 1 minute

        print(".3f")

    def test_memory_efficiency(self, config_service, large_experiment_data, tmp_path):
        """Test memory efficiency of analysis operations."""
        service = AnalysisService(config_service)

        # Monitor memory during analysis
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        request = AnalysisRequest(
            module_name="agents",  # Agents module can be memory intensive due to clustering
            experiment_path=large_experiment_data,
            output_path=tmp_path / "memory_efficiency",
            enable_caching=False
        )

        result = service.run(request)

        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory

        assert result.success

        # Memory usage should be reasonable for the data size
        memory_mb = memory_used / (1024 * 1024)
        assert memory_mb < 1000  # Less than 1GB for this test

        print(".2f")

    def test_io_performance(self, config_service, large_experiment_data, tmp_path):
        """Test I/O performance of analysis operations."""
        service = AnalysisService(config_service)

        # Test with file caching disabled to measure raw I/O
        request = AnalysisRequest(
            module_name="population",
            experiment_path=large_experiment_data,
            output_path=tmp_path / "io_test",
            enable_caching=False
        )

        start_time = time.perf_counter()
        result = service.run(request)
        end_time = time.perf_counter()

        assert result.success

        io_time = end_time - start_time

        # I/O bound operations should complete in reasonable time
        assert io_time < 45.0  # Under 45 seconds for I/O intensive work

        print(".3f")
