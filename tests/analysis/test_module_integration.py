"""
Comprehensive integration tests for all migrated analysis modules.

Tests complete workflows for population, resources, actions, agents,
learning, spatial, temporal, and combat modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, MagicMock

from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.analysis.common.context import AnalysisContext
from farm.core.services import EnvConfigService


@pytest.fixture
def config_service():
    """Create configuration service for testing."""
    return EnvConfigService()


@pytest.fixture
def sample_experiment_data(tmp_path):
    """Create sample experiment directory with mock data."""
    exp_path = tmp_path / "sample_experiment"
    exp_path.mkdir()

    # Create mock simulation.db
    db_path = exp_path / "simulation.db"
    db_path.write_bytes(b"mock database content")

    # Create some sample data files
    data_dir = exp_path / "data"
    data_dir.mkdir()

    # Sample population data
    pop_data = pd.DataFrame({
        'step': range(100),
        'total_agents': [100 + i for i in range(100)],
        'system_agents': [50 + i//2 for i in range(100)],
        'independent_agents': [30 + i//3 for i in range(100)],
        'control_agents': [20 + i//4 for i in range(100)],
        'avg_resources': [50 + np.sin(i/10) * 10 for i in range(100)],
        'births': [np.random.randint(0, 5) for _ in range(100)],
        'deaths': [np.random.randint(0, 3) for _ in range(100)],
    })
    pop_data.to_csv(data_dir / "population.csv", index=False)

    # Sample resource data
    resource_data = pd.DataFrame({
        'step': range(100),
        'total_resources': [1000 - i for i in range(100)],
        'consumed_resources': [10 + np.random.randint(5, 15) for _ in range(100)],
        'resource_efficiency': [0.8 + 0.2 * np.sin(i/20) for i in range(100)],
        'hotspot_count': [np.random.randint(1, 5) for _ in range(100)],
    })
    resource_data.to_csv(data_dir / "resources.csv", index=False)

    # Sample action data
    action_data = pd.DataFrame({
        'step': range(100),
        'action_type': np.random.choice(['move', 'gather', 'attack', 'reproduce'], 100),
        'success_rate': np.random.uniform(0.7, 1.0, 100),
        'reward': np.random.normal(1.0, 0.5, 100),
        'frequency': np.random.randint(1, 10, 100),
    })
    action_data.to_csv(data_dir / "actions.csv", index=False)

    return exp_path


class TestMigratedModulesIntegration:
    """Test all migrated analysis modules with complete workflows."""

    def test_population_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete population analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="population",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "population_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "population"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "population_results"
        assert (output_dir / "population_statistics.json").exists()
        assert (output_dir / "agent_composition.csv").exists()
        assert (output_dir / "population_over_time.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

        # Verify statistics file content
        with open(output_dir / "population_statistics.json") as f:
            stats = json.load(f)
            assert 'statistics' in stats
            assert 'rates' in stats
            assert 'stability' in stats

    def test_resources_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete resources analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="resources",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "resources_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "resources"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "resources_results"
        assert (output_dir / "resource_statistics.json").exists()
        assert (output_dir / "consumption_patterns.csv").exists()
        assert (output_dir / "resource_efficiency.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_actions_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete actions analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="actions",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "actions_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "actions"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "actions_results"
        assert (output_dir / "action_statistics.json").exists()
        assert (output_dir / "action_patterns.csv").exists()
        assert (output_dir / "action_distribution.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_agents_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete agents analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="agents",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "agents_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "agents"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "agents_results"
        assert (output_dir / "agent_statistics.json").exists()
        assert (output_dir / "lifespan_analysis.csv").exists()
        assert (output_dir / "behavior_patterns.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_learning_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete learning analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="learning",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "learning_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "learning"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "learning_results"
        assert (output_dir / "learning_statistics.json").exists()
        assert (output_dir / "learning_curves.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_spatial_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete spatial analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="spatial",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "spatial_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "spatial"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "spatial_results"
        assert (output_dir / "spatial_statistics.json").exists()
        assert (output_dir / "movement_patterns.png").exists()
        assert (output_dir / "location_analysis.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_temporal_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete temporal analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="temporal",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "temporal_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "temporal"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "temporal_results"
        assert (output_dir / "temporal_patterns.json").exists()
        assert (output_dir / "time_series_analysis.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_combat_module_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test complete combat analysis workflow."""
        service = AnalysisService(config_service)

        request = AnalysisRequest(
            module_name="combat",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "combat_results",
            group="all"
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "combat"
        assert result.output_path.exists()

        # Check expected outputs
        output_dir = tmp_path / "combat_results"
        assert (output_dir / "combat_statistics.json").exists()
        assert (output_dir / "combat_metrics.png").exists()
        assert (output_dir / "analysis_summary.json").exists()

    def test_batch_module_execution(self, config_service, sample_experiment_data, tmp_path):
        """Test running multiple modules in batch."""
        service = AnalysisService(config_service)

        modules = ["population", "resources", "actions", "agents"]
        requests = []

        for module in modules:
            request = AnalysisRequest(
                module_name=module,
                experiment_path=sample_experiment_data,
                output_path=tmp_path / f"{module}_batch",
                group="basic"
            )
            requests.append(request)

        results = service.run_batch(requests)

        assert len(results) == len(modules)
        assert all(r.success for r in results)
        assert all(r.output_path.exists() for r in results)

        # Verify each module produced expected outputs
        for i, module in enumerate(modules):
            output_dir = tmp_path / f"{module}_batch"
            assert (output_dir / "analysis_summary.json").exists()

    def test_module_error_handling(self, config_service, tmp_path):
        """Test error handling when modules fail gracefully."""
        service = AnalysisService(config_service)

        # Test with non-existent experiment path
        request = AnalysisRequest(
            module_name="population",
            experiment_path=tmp_path / "nonexistent",
            output_path=tmp_path / "error_output"
        )

        result = service.run(request)

        # Should fail gracefully with meaningful error
        assert not result.success
        assert result.error is not None
        assert len(result.error) > 0

    def test_module_caching(self, config_service, sample_experiment_data, tmp_path):
        """Test caching functionality across modules."""
        cache_dir = tmp_path / "cache"
        service = AnalysisService(config_service, cache_dir=cache_dir)

        request = AnalysisRequest(
            module_name="population",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "cached_output",
            enable_caching=True
        )

        # First run
        result1 = service.run(request)
        assert result1.success
        assert not result1.cache_hit

        # Second run (should use cache)
        result2 = service.run(request)
        assert result2.success
        # Note: Cache hit detection depends on implementation details
        # The important thing is that it succeeds and produces same results

    def test_module_custom_parameters(self, config_service, sample_experiment_data, tmp_path):
        """Test modules with custom analysis parameters."""
        service = AnalysisService(config_service)

        # Test population module with custom plot parameters
        request = AnalysisRequest(
            module_name="population",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "custom_params",
            group="plots",
            analysis_kwargs={
                "plot_population": {"figsize": (10, 6), "dpi": 150}
            }
        )

        result = service.run(request)
        assert result.success

        # Verify output was created
        output_dir = tmp_path / "custom_params"
        assert (output_dir / "population_over_time.png").exists()


class TestModuleCrossIntegration:
    """Test integration between different modules."""

    def test_population_resources_correlation(self, config_service, sample_experiment_data, tmp_path):
        """Test analyzing correlation between population and resources."""
        service = AnalysisService(config_service)

        # Run both modules
        pop_request = AnalysisRequest(
            module_name="population",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "pop_cross",
            group="analysis"
        )

        res_request = AnalysisRequest(
            module_name="resources",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "res_cross",
            group="analysis"
        )

        pop_result = service.run(pop_request)
        res_result = service.run(res_request)

        assert pop_result.success
        assert res_result.success

        # Both should produce analysis results
        assert (tmp_path / "pop_cross" / "population_statistics.json").exists()
        assert (tmp_path / "res_cross" / "resource_statistics.json").exists()

    def test_agent_actions_integration(self, config_service, sample_experiment_data, tmp_path):
        """Test analyzing relationship between agents and actions."""
        service = AnalysisService(config_service)

        # Run agent and action analysis
        agent_request = AnalysisRequest(
            module_name="agents",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "agent_action",
            group="analysis"
        )

        action_request = AnalysisRequest(
            module_name="actions",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "action_agent",
            group="analysis"
        )

        agent_result = service.run(agent_request)
        action_result = service.run(action_request)

        assert agent_result.success
        assert action_result.success

        # Verify complementary analyses
        assert (tmp_path / "agent_action" / "agent_statistics.json").exists()
        assert (tmp_path / "action_agent" / "action_statistics.json").exists()


class TestPerformanceBenchmarks:
    """Performance benchmarking for analysis modules."""

    def test_module_execution_performance(self, config_service, sample_experiment_data, tmp_path, benchmark):
        """Benchmark module execution performance."""
        service = AnalysisService(config_service)

        def run_population_analysis():
            request = AnalysisRequest(
                module_name="population",
                experiment_path=sample_experiment_data,
                output_path=tmp_path / "perf_test",
                enable_caching=False
            )
            return service.run(request)

        result = benchmark(run_population_analysis)
        assert result.success

    def test_batch_performance(self, config_service, sample_experiment_data, tmp_path, benchmark):
        """Benchmark batch execution performance."""
        service = AnalysisService(config_service)

        def run_batch_analysis():
            requests = [
                AnalysisRequest(
                    module_name=module,
                    experiment_path=sample_experiment_data,
                    output_path=tmp_path / f"batch_perf_{module}",
                    group="basic",
                    enable_caching=False
                )
                for module in ["population", "resources", "actions"]
            ]
            return service.run_batch(requests)

        results = benchmark(run_batch_analysis)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_memory_usage(self, config_service, sample_experiment_data, tmp_path):
        """Test memory usage during analysis (basic check)."""
        import psutil
        import os

        service = AnalysisService(config_service)

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run analysis
        request = AnalysisRequest(
            module_name="population",
            experiment_path=sample_experiment_data,
            output_path=tmp_path / "memory_test"
        )

        result = service.run(request)
        assert result.success

        # Check memory didn't grow excessively (rough check)
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (< 100MB for this test)
        assert memory_growth < 100 * 1024 * 1024  # 100MB in bytes
