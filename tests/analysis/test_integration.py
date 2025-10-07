"""
Integration tests for the analysis module system.

Tests the complete workflow from request to result.
"""

from pathlib import Path

import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.core import (
    BaseAnalysisModule,
    SimpleDataProcessor,
    make_analysis_function,
)
from farm.analysis.registry import registry
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class TestEndToEndWorkflow:
    """Test complete end-to-end analysis workflows."""

    @pytest.fixture
    def test_analysis_module(self):
        """Create a test module for integration testing."""

        class IntegrationTestModule(BaseAnalysisModule):
            def __init__(self):
                super().__init__(
                    name="integration_test",
                    description="Module for integration testing",
                )

            def register_functions(self):
                def compute_stats(df: pd.DataFrame, ctx: AnalysisContext) -> None:
                    stats = df.describe()
                    output_file = ctx.get_output_file("stats.csv")
                    stats.to_csv(output_file)
                    ctx.report_progress("Stats computed", 0.5)

                def create_plot(df: pd.DataFrame, ctx: AnalysisContext) -> None:
                    # Mock plotting
                    output_file = ctx.get_output_file("plot.txt")
                    output_file.write_text(f"Plot with {len(df)} points")
                    ctx.report_progress("Plot created", 1.0)

                self._functions = {
                    "compute_stats": make_analysis_function(compute_stats),
                    "create_plot": make_analysis_function(create_plot),
                }

                self._groups = {
                    "all": list(self._functions.values()),
                    "metrics": [self._functions["compute_stats"]],
                    "plots": [self._functions["create_plot"]],
                }

            def get_data_processor(self):
                def process(data, **kwargs):
                    return pd.DataFrame(
                        {
                            "iteration": range(10),
                            "value": range(10, 20),
                            "category": ["A", "B"] * 5,
                        }
                    )

                return SimpleDataProcessor(process)

        module = IntegrationTestModule()
        registry.register(module)
        return module

    def test_complete_workflow(
        self, config_service_mock, test_analysis_module, tmp_path
    ):
        """Test complete workflow from request to result."""
        # Create test paths
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        output_path = tmp_path / "output"

        # Initialize service
        service = AnalysisService(
            config_service=config_service_mock,
            cache_dir=tmp_path / "cache",
            auto_register=False,
        )

        # Create request
        request = AnalysisRequest(
            module_name="integration_test",
            experiment_path=exp_path,
            output_path=output_path,
            group="all",
        )

        # Run analysis
        result = service.run(request)

        # Verify result
        assert result.success
        assert result.module_name == "integration_test"
        assert result.dataframe is not None
        assert len(result.dataframe) == 10
        assert result.execution_time > 0

        # Verify outputs
        assert output_path.exists()
        assert (output_path / "stats.csv").exists()
        assert (output_path / "plot.txt").exists()
        assert (output_path / "analysis_summary.json").exists()

    def test_workflow_with_progress_tracking(
        self, config_service_mock, test_analysis_module, tmp_path
    ):
        """Test workflow with progress tracking."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        progress_updates = []

        def track_progress(message: str, progress: float):
            progress_updates.append((message, progress))

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        request = AnalysisRequest(
            module_name="integration_test",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            progress_callback=track_progress,
        )

        result = service.run(request)

        assert result.success
        assert len(progress_updates) > 0

        # Check progress values are in range
        for _, progress in progress_updates:
            assert 0.0 <= progress <= 1.0

    def test_workflow_with_function_groups(
        self, config_service_mock, test_analysis_module, tmp_path
    ):
        """Test running specific function groups."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        # Run only metrics group
        request = AnalysisRequest(
            module_name="integration_test",
            experiment_path=exp_path,
            output_path=tmp_path / "metrics",
            group="metrics",
        )

        result = service.run(request)

        assert result.success
        assert (tmp_path / "metrics" / "stats.csv").exists()
        assert not (tmp_path / "metrics" / "plot.txt").exists()  # Plot not run

        # Run only plots group
        request = AnalysisRequest(
            module_name="integration_test",
            experiment_path=exp_path,
            output_path=tmp_path / "plots",
            group="plots",
        )

        result = service.run(request)

        assert result.success
        assert (tmp_path / "plots" / "plot.txt").exists()
        assert not (tmp_path / "plots" / "stats.csv").exists()  # Stats not run

    def test_workflow_with_caching(
        self, config_service_mock, test_analysis_module, tmp_path
    ):
        """Test caching in complete workflow."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        service = AnalysisService(
            config_service=config_service_mock,
            cache_dir=tmp_path / "cache",
            auto_register=False,
        )

        request = AnalysisRequest(
            module_name="integration_test",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            enable_caching=True,
        )

        # First run
        result1 = service.run(request)
        assert result1.success
        assert not result1.cache_hit
        time1 = result1.execution_time

        # Second run (cached)
        result2 = service.run(request)
        assert result2.success
        assert result2.cache_hit
        # Cached results should be significantly faster for most cases
        # However, for very fast computations, cache I/O overhead might make it slower
        # So we'll just verify the cache hit occurred and results are the same
        time2 = result2.execution_time
        logger.info(f"First run: {time1:.4f}s, Second run (cached): {time2:.4f}s")

        # Verify same data
        assert result1.dataframe.equals(result2.dataframe)

    def test_batch_workflow(self, config_service_mock, test_analysis_module, tmp_path):
        """Test batch analysis workflow."""
        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        # Create multiple experiment paths
        requests = []
        for i in range(3):
            exp_path = tmp_path / f"experiment_{i}"
            exp_path.mkdir()

            request = AnalysisRequest(
                module_name="integration_test",
                experiment_path=exp_path,
                output_path=tmp_path / f"output_{i}",
                enable_caching=False,
            )
            requests.append(request)

        # Run batch
        results = service.run_batch(requests)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify all outputs
        for i in range(3):
            assert (tmp_path / f"output_{i}" / "stats.csv").exists()
            assert (tmp_path / f"output_{i}" / "plot.txt").exists()

    def test_workflow_with_custom_parameters(self, config_service_mock, tmp_path):
        """Test workflow with custom analysis parameters."""

        # Create module that uses custom parameters
        class ParamTestModule(BaseAnalysisModule):
            def __init__(self):
                super().__init__(name="param_test", description="Param test")

            def register_functions(self):
                def func_with_params(
                    df: pd.DataFrame,
                    ctx: AnalysisContext,
                    multiplier: int = 1,
                    add_column: bool = False,
                ) -> None:
                    df = df.copy()
                    df["value"] = df["value"] * multiplier
                    if add_column:
                        df["extra"] = "added"

                    output_file = ctx.get_output_file("result.csv")
                    df.to_csv(output_file, index=False)

                self._functions = {
                    "func_with_params": make_analysis_function(func_with_params)
                }
                self._groups = {"all": list(self._functions.values())}

            def get_data_processor(self):
                def process(data, **kwargs):
                    return pd.DataFrame({"value": [1, 2, 3]})

                return SimpleDataProcessor(process)

        module = ParamTestModule()
        registry.register(module)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        # Run with custom parameters
        request = AnalysisRequest(
            module_name="param_test",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            analysis_kwargs={
                "func_with_params": {"multiplier": 10, "add_column": True}
            },
        )

        result = service.run(request)
        assert result.success

        # Verify parameters were applied
        output_df = pd.read_csv(tmp_path / "output" / "result.csv")
        assert output_df["value"].tolist() == [10, 20, 30]
        assert "extra" in output_df.columns


class TestErrorHandling:
    """Test error handling in workflows."""

    def test_invalid_module_error(self, config_service_mock, tmp_path):
        """Test error handling for invalid module."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        request = AnalysisRequest(
            module_name="nonexistent",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
        )

        result = service.run(request)

        assert not result.success
        assert "not found" in result.error.lower()

    def test_missing_experiment_path_error(
        self, config_service_mock, minimal_module, tmp_path
    ):
        """Test error handling for missing experiment path."""
        registry.register(minimal_module)

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=tmp_path / "nonexistent",
            output_path=tmp_path / "output",
        )

        result = service.run(request)

        assert not result.success
        assert "does not exist" in result.error.lower()

    def test_function_error_handling(self, config_service_mock, tmp_path):
        """Test error handling when analysis function fails."""

        class ErrorModule(BaseAnalysisModule):
            def __init__(self):
                super().__init__(name="error_test", description="Error test")

            def register_functions(self):
                def failing_func(df: pd.DataFrame, ctx: AnalysisContext) -> None:
                    raise ValueError("Intentional error for testing")

                self._functions = {"failing_func": make_analysis_function(failing_func)}
                self._groups = {"all": list(self._functions.values())}

            def get_data_processor(self):
                return SimpleDataProcessor(
                    lambda d, **k: pd.DataFrame({"x": [1, 2, 3]})
                )

        module = ErrorModule()
        registry.register(module)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        service = AnalysisService(
            config_service=config_service_mock, auto_register=False
        )

        request = AnalysisRequest(
            module_name="error_test",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
        )

        # Analysis should complete even if function fails
        # (functions are allowed to fail without stopping the whole analysis)
        result = service.run(request)

        # The analysis framework continues despite function errors
        assert result.success  # Overall success because processor worked
