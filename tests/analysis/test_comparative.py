"""
Comprehensive tests for comparative analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.comparative import (
    comparative_module,
    compare_simulations,
    compare_experiments,
    compute_comparison_metrics,
    compute_parameter_differences,
    compute_performance_comparison,
    analyze_simulation_comparison,
    analyze_parameter_differences,
    analyze_performance_comparison,
    plot_comparison_metrics,
    plot_parameter_differences,
    plot_performance_comparison,
    plot_simulation_comparison,
    plot_comparative_analysis,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def sample_comparative_data():
    """Create sample comparative data."""
    experiments = ["exp_1", "exp_2", "exp_3"]
    data = []

    for exp in experiments:
        for i in range(10):
            data.append(
                {
                    "experiment": exp,
                    "simulation_id": exp,
                    "iteration": i,
                    "performance_metric": np.random.uniform(0, 1),
                    "efficiency_score": np.random.uniform(0, 1),
                    "stability_metric": np.random.uniform(0, 1),
                    "reward": np.random.uniform(0, 10),
                    "fitness": np.random.uniform(0, 1),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_parameter_data():
    """Create sample parameter data."""
    data = []
    for sim_id in ["sim_1", "sim_2", "sim_3"]:
        for param in ["learning_rate", "population_size", "mutation_rate"]:
            data.append(
                {
                    "simulation_id": sim_id,
                    "parameter_name": param,
                    "parameter_value": np.random.uniform(0, 1),
                }
            )
    return pd.DataFrame(data)


class TestComparativeComputations:
    """Test comparative statistical computations."""

    def test_compute_comparison_metrics(self, sample_comparative_data):
        """Test comparison metrics computation."""
        metrics = compute_comparison_metrics(sample_comparative_data)

        assert isinstance(metrics, dict)
        assert "performance_metric" in metrics
        assert "efficiency_score" in metrics
        assert "stability_metric" in metrics

        # Check statistics structure
        for metric_name, stats in metrics.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

    def test_compute_comparison_metrics_empty(self):
        """Test comparison metrics with empty DataFrame."""
        metrics = compute_comparison_metrics(pd.DataFrame())
        assert metrics == {}

    def test_compute_comparison_metrics_single_column(self):
        """Test comparison with single numeric column."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5],
            }
        )

        metrics = compute_comparison_metrics(df)

        assert "value" in metrics
        assert metrics["value"]["mean"] == 3.0

    def test_compute_parameter_differences(self, sample_parameter_data):
        """Test parameter differences computation."""
        differences = compute_parameter_differences(sample_parameter_data)

        assert isinstance(differences, dict)
        assert "learning_rate" in differences
        assert "population_size" in differences
        assert "mutation_rate" in differences

        # Check difference structure
        for param_name, diff_stats in differences.items():
            assert "values" in diff_stats
            assert "range" in diff_stats
            assert "mean" in diff_stats
            assert "std" in diff_stats

    def test_compute_parameter_differences_empty(self):
        """Test parameter differences with empty DataFrame."""
        differences = compute_parameter_differences(pd.DataFrame())
        assert differences == {}

    def test_compute_parameter_differences_no_simulation_id(self):
        """Test parameter differences without simulation_id column."""
        df = pd.DataFrame(
            {
                "parameter_name": ["param1", "param2"],
                "parameter_value": [1.0, 2.0],
            }
        )

        differences = compute_parameter_differences(df)
        assert differences == {}

    def test_compute_parameter_differences_single_value(self):
        """Test parameter differences with single value per parameter."""
        df = pd.DataFrame(
            {
                "simulation_id": ["sim_1"],
                "parameter_name": ["learning_rate"],
                "parameter_value": [0.01],
            }
        )

        differences = compute_parameter_differences(df)

        # Should not compute differences for single value
        assert differences == {}

    def test_compute_performance_comparison(self, sample_comparative_data):
        """Test performance comparison computation."""
        performance = compute_performance_comparison(sample_comparative_data)

        assert isinstance(performance, dict)
        assert "performance_metric" in performance
        assert "reward" in performance
        assert "fitness" in performance

        # Check statistics
        for metric_name, stats in performance.items():
            assert isinstance(stats, dict)

    def test_compute_performance_comparison_empty(self):
        """Test performance comparison with empty DataFrame."""
        performance = compute_performance_comparison(pd.DataFrame())
        assert performance == {}

    def test_compute_performance_comparison_no_performance_cols(self):
        """Test performance comparison without performance columns."""
        df = pd.DataFrame(
            {
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        )

        performance = compute_performance_comparison(df)

        # Should return empty dict if no performance columns
        assert performance == {}


class TestComparativeAnalysis:
    """Test comparative analysis functions."""

    def test_analyze_simulation_comparison(self, sample_comparative_data, tmp_path):
        """Test simulation comparison analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_simulation_comparison(sample_comparative_data, ctx)

        # Check output file
        output_file = tmp_path / "comparison_metrics.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert len(data) > 0

    def test_analyze_parameter_differences(self, sample_parameter_data, tmp_path):
        """Test parameter differences analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_parameter_differences(sample_parameter_data, ctx)

        # Check output file
        output_file = tmp_path / "parameter_differences.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_analyze_performance_comparison(self, sample_comparative_data, tmp_path):
        """Test performance comparison analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_performance_comparison(sample_comparative_data, ctx)

        # Check output file
        output_file = tmp_path / "performance_comparison.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_compare_experiments(self, sample_comparative_data):
        """Test experiment comparison."""
        result = compare_experiments(sample_comparative_data)

        assert isinstance(result, dict)
        assert "experiment_count" in result
        assert "performance_comparison" in result
        assert "statistical_tests" in result

        assert result["experiment_count"] == 3

    def test_compare_experiments_empty(self):
        """Test experiment comparison with empty DataFrame."""
        result = compare_experiments(pd.DataFrame())

        assert result["experiment_count"] == 0
        assert result["performance_comparison"] == {}
        assert result["statistical_tests"] == {}

    def test_compare_experiments_no_experiment_column(self):
        """Test experiment comparison without experiment column."""
        df = pd.DataFrame(
            {
                "performance_metric": [1, 2, 3],
                "reward": [10, 20, 30],
            }
        )

        result = compare_experiments(df)

        assert result["experiment_count"] == 0


class TestComparativeVisualization:
    """Test comparative visualization functions."""

    @patch("farm.analysis.comparative.plot.plt")
    @patch("farm.analysis.comparative.plot.sns")
    def test_plot_comparison_metrics(self, mock_sns, mock_plt, sample_comparative_data, tmp_path):
        """Test comparison metrics plotting."""
        # Mock subplots to return proper figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.boxplot = MagicMock()
        mock_ax.set_title = MagicMock()

        # Create a proper 2D array structure that supports tuple indexing
        class MockAxesArray:
            def __init__(self, mock_ax):
                self.mock_ax = mock_ax
                self._array = [[mock_ax for _ in range(2)] for _ in range(2)]

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return self._array[key[0]][key[1]]
                return self._array[key]

        mock_axes = MockAxesArray(mock_ax)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Mock seaborn to avoid compatibility issues
        mock_sns.boxplot.side_effect = AttributeError("Mocked seaborn error")

        ctx = AnalysisContext(output_path=tmp_path)

        plot_comparison_metrics(sample_comparative_data, ctx)

        # Check that plot was created
        assert mock_plt.savefig.called
        assert mock_plt.close.called

    @patch("farm.analysis.comparative.plot.plt")
    def test_plot_comparison_metrics_empty(self, mock_plt, tmp_path):
        """Test comparison metrics plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_comparison_metrics(pd.DataFrame(), ctx)

        # Should not create plot with no data
        assert not mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    @patch("farm.analysis.comparative.plot.sns")
    def test_plot_parameter_differences(self, mock_sns, mock_plt, sample_parameter_data, tmp_path):
        """Test parameter differences plotting."""
        # Mock subplots to return proper figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.bar = MagicMock()
        mock_ax.set_title = MagicMock()
        mock_axes = [mock_ax for _ in range(3)]  # 3 unique parameters
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Mock seaborn to avoid compatibility issues
        mock_sns.barplot.side_effect = AttributeError("Mocked seaborn error")

        ctx = AnalysisContext(output_path=tmp_path)

        plot_parameter_differences(sample_parameter_data, ctx)

        # Check that plot was created
        assert mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    def test_plot_parameter_differences_no_data(self, mock_plt, tmp_path):
        """Test parameter plotting without parameter data."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        plot_parameter_differences(df, ctx)

        # Should not create plot
        assert not mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    @patch("farm.analysis.comparative.plot.sns")
    def test_plot_performance_comparison(self, mock_sns, mock_plt, sample_comparative_data, tmp_path):
        """Test performance comparison plotting."""
        # Mock figure and subplot
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.boxplot = MagicMock()
        mock_ax.bar = MagicMock()
        mock_ax.set_xlabel = MagicMock()
        mock_ax.set_ylabel = MagicMock()
        mock_ax.set_title = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot.return_value = mock_ax

        # Mock seaborn to avoid compatibility issues
        mock_sns.boxplot.side_effect = AttributeError("Mocked seaborn error")

        ctx = AnalysisContext(output_path=tmp_path)

        plot_performance_comparison(sample_comparative_data, ctx)

        # Check that plot was created
        assert mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    def test_plot_performance_comparison_no_performance_cols(self, mock_plt, tmp_path):
        """Test performance plotting without performance columns."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        plot_performance_comparison(df, ctx)

        # Should not create plot
        assert not mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    @patch("farm.analysis.comparative.plot.sns")
    def test_plot_simulation_comparison(self, mock_sns, mock_plt, sample_comparative_data, tmp_path):
        """Test comprehensive simulation comparison plotting."""
        # Mock subplots and figure for both functions called
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.boxplot = MagicMock()
        mock_ax.bar = MagicMock()
        mock_ax.set_xlabel = MagicMock()
        mock_ax.set_ylabel = MagicMock()
        mock_ax.set_title = MagicMock()

        # Create a proper 2D array structure that supports tuple indexing
        class MockAxesArray:
            def __init__(self, mock_ax):
                self.mock_ax = mock_ax
                self._array = [[mock_ax for _ in range(2)] for _ in range(2)]

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return self._array[key[0]][key[1]]
                return self._array[key]

        mock_axes = MockAxesArray(mock_ax)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot.return_value = mock_ax

        # Mock seaborn to avoid compatibility issues
        mock_sns.boxplot.side_effect = AttributeError("Mocked seaborn error")

        ctx = AnalysisContext(output_path=tmp_path)

        plot_simulation_comparison(sample_comparative_data, ctx)

        # Should create at least one plot
        assert mock_plt.savefig.called

    @patch("farm.analysis.comparative.plot.plt")
    def test_plot_comparative_analysis(self, mock_plt, sample_comparative_data, tmp_path):
        """Test comparative analysis plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        result = plot_comparative_analysis(sample_comparative_data, ctx)

        assert result is None
        assert mock_plt.savefig.called


class TestSimulationComparison:
    """Test simulation comparison functionality."""

    def test_compare_simulations_missing_path(self):
        """Test compare_simulations with non-existent path."""
        with pytest.raises(ValueError, match="Search path does not exist"):
            compare_simulations("/nonexistent/path", "/output/path")

    def test_compare_simulations_insufficient_simulations(self, tmp_path):
        """Test compare_simulations with less than 2 simulations."""
        search_path = tmp_path / "simulations"
        search_path.mkdir()

        # Create only one simulation
        sim1 = search_path / "sim1"
        sim1.mkdir()
        (sim1 / "simulation.db").touch()

        output_path = tmp_path / "output"

        with pytest.raises(ValueError, match="Need at least 2 simulations"):
            compare_simulations(str(search_path), str(output_path))

    def test_compare_simulations_success(self, tmp_path):
        """Test successful simulation comparison."""
        search_path = tmp_path / "simulations"
        search_path.mkdir()

        # Create multiple simulations
        for i in range(3):
            sim_dir = search_path / f"sim{i}"
            sim_dir.mkdir()
            (sim_dir / "simulation.db").touch()

        output_path = tmp_path / "output"

        # Should not raise
        compare_simulations(str(search_path), str(output_path))

        # Check that output was created
        assert output_path.exists()
        result_file = output_path / "comparison_summary.txt"
        assert result_file.exists()


class TestComparativeModule:
    """Test comparative module integration."""

    def test_comparative_module_registration(self):
        """Test module registration."""
        assert comparative_module.name == "comparative"
        assert comparative_module.description == "Analysis for comparing multiple simulations and their differences"

    def test_comparative_module_function_names(self):
        """Test module function names."""
        functions = comparative_module.get_function_names()
        expected_functions = [
            "plot_comparative_analysis",
            "plot_performance_comparison",
            "plot_metric_correlations",
            "plot_experiment_differences",
            "plot_statistical_comparison",
            "compare_experiments",
            "analyze_experiment_variability",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_comparative_module_function_groups(self):
        """Test module function groups."""
        groups = comparative_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_comparative_module_data_processor(self):
        """Test module data processor."""
        processor = comparative_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = comparative_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = comparative_module.get_functions()
        assert len(functions) >= 7


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_metrics_with_nan_values(self):
        """Test comparison metrics with NaN values."""
        df = pd.DataFrame(
            {
                "metric1": [1.0, np.nan, 3.0, 4.0, 5.0],
                "metric2": [np.nan, 2.0, 3.0, np.nan, 5.0],
            }
        )

        metrics = compute_comparison_metrics(df)

        # Should handle NaN gracefully
        assert "metric1" in metrics
        assert "metric2" in metrics

    def test_compute_parameter_differences_extreme_values(self):
        """Test parameter differences with extreme values."""
        df = pd.DataFrame(
            {
                "simulation_id": ["sim_1", "sim_2", "sim_3"],
                "parameter_name": ["param1", "param1", "param1"],
                "parameter_value": [0.001, 1000.0, 0.5],
            }
        )

        differences = compute_parameter_differences(df)

        assert "param1" in differences
        assert differences["param1"]["range"] > 0

    def test_analyze_with_progress_callback(self, sample_comparative_data, tmp_path):
        """Test analysis with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)

        analyze_simulation_comparison(sample_comparative_data, ctx)

        # Should have called progress callback
        assert len(progress_calls) > 0
        assert any("complete" in msg.lower() for msg, _ in progress_calls)

    def test_plot_with_single_simulation(self, tmp_path):
        """Test plotting with single simulation data."""
        df = pd.DataFrame(
            {
                "simulation_id": ["sim_1"] * 10,
                "performance_metric": np.random.uniform(0, 1, 10),
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.comparative.plot.plt") as mock_plt, patch(
            "farm.analysis.comparative.plot.sns"
        ) as mock_sns:
            # Mock subplots to return proper figure and axes
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_ax.boxplot = MagicMock()
            mock_ax.set_title = MagicMock()

            # Create a proper 2D array structure that supports tuple indexing
            class MockAxesArray:
                def __init__(self, mock_ax):
                    self.mock_ax = mock_ax
                    self._array = [[mock_ax for _ in range(2)] for _ in range(2)]

                def __getitem__(self, key):
                    if isinstance(key, tuple):
                        return self._array[key[0]][key[1]]
                    return self._array[key]

            mock_axes = MockAxesArray(mock_ax)
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            # Mock seaborn to avoid compatibility issues
            mock_sns.boxplot.side_effect = AttributeError("Mocked seaborn error")

            # Should handle single simulation gracefully
            plot_comparison_metrics(df, ctx)

    def test_compare_experiments_with_missing_data(self):
        """Test experiment comparison with missing data points."""
        df = pd.DataFrame(
            {
                "experiment": ["A", "A", "B", "B", "C"],
                "performance_metric": [1.0, 2.0, np.nan, 3.0, 4.0],
            }
        )

        result = compare_experiments(df)

        # Should handle missing data
        assert result["experiment_count"] == 3

    def test_parameter_differences_identical_values(self):
        """Test parameter differences when all values are identical."""
        df = pd.DataFrame(
            {
                "simulation_id": ["sim_1", "sim_2", "sim_3"],
                "parameter_name": ["param1", "param1", "param1"],
                "parameter_value": [0.5, 0.5, 0.5],
            }
        )

        differences = compute_parameter_differences(df)

        assert "param1" in differences
        assert differences["param1"]["range"] == 0.0
        assert differences["param1"]["std"] == 0.0

    def test_compute_metrics_non_numeric_columns(self):
        """Test compute metrics ignores non-numeric columns."""
        df = pd.DataFrame(
            {
                "text_col": ["a", "b", "c"],
                "numeric_col": [1, 2, 3],
                "mixed_col": ["1", "2", 3],
            }
        )

        metrics = compute_comparison_metrics(df)

        # Should only include numeric columns
        assert "numeric_col" in metrics
        assert "text_col" not in metrics

    def test_plot_parameter_differences_single_parameter(self, tmp_path):
        """Test plotting with single parameter."""
        df = pd.DataFrame(
            {
                "simulation_id": ["sim_1", "sim_2"],
                "parameter_name": ["learning_rate", "learning_rate"],
                "parameter_value": [0.01, 0.02],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.comparative.plot.plt") as mock_plt, patch(
            "farm.analysis.comparative.plot.sns"
        ) as mock_sns:
            # Mock subplots to return proper figure and axes for single parameter
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_ax.bar = MagicMock()
            mock_ax.set_title = MagicMock()
            # For single parameter, subplots returns a single axis, not a list
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock seaborn to avoid compatibility issues
            mock_sns.barplot.side_effect = AttributeError("Mocked seaborn error")

            plot_parameter_differences(df, ctx)
