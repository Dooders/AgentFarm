"""
Comprehensive tests for learning analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.learning import (
    learning_module,
    compute_learning_statistics,
    compute_agent_learning_curves,
    compute_learning_efficiency_metrics,
    compute_module_performance_comparison,
    analyze_learning_performance,
    analyze_agent_learning_curves,
    analyze_module_performance,
    analyze_learning_progress,
    plot_learning_curves,
    plot_reward_distribution,
    plot_module_performance,
    plot_action_frequencies,
    plot_learning_efficiency,
    plot_reward_vs_step,
)


@pytest.fixture
def sample_learning_data():
    """Create sample learning data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "step": range(200),
            "reward": [i * 0.05 + np.random.randn() * 0.5 for i in range(200)],
            "agent_id": [f"agent_{i % 10}" for i in range(200)],
            "module_type": ["decision" if i % 2 == 0 else "memory" for i in range(200)],
            "action_taken": [i % 20 for i in range(200)],
            "action_taken_mapped": [i % 5 for i in range(200)],
            "action_count": [i % 10 + 1 for i in range(200)],
            "success_rate": [0.5 + (i % 50) * 0.01 for i in range(200)],
        }
    )


@pytest.fixture
def sample_agent_data():
    """Create sample agent-specific learning data."""
    return pd.DataFrame(
        {
            "step": list(range(50)) * 3,
            "agent_id": ["agent_1"] * 50 + ["agent_2"] * 50 + ["agent_3"] * 50,
            "reward": [i * 0.1 + (i % 5) for i in range(50)] * 3,
        }
    )


@pytest.fixture
def sample_module_data():
    """Create sample module-specific data."""
    return pd.DataFrame(
        {
            "step": range(100),
            "reward": [5 + i * 0.05 + np.random.randn() * 0.3 for i in range(100)],
            "module_type": ["decision"] * 50 + ["memory"] * 50,
            "agent_id": ["agent_1"] * 100,
        }
    )


class TestLearningComputations:
    """Test learning statistical computations."""

    def test_compute_learning_statistics(self, sample_learning_data):
        """Test learning statistics computation."""
        stats = compute_learning_statistics(sample_learning_data)

        assert isinstance(stats, dict)
        assert "total_experiences" in stats
        assert "reward" in stats
        assert "reward_trend" in stats
        assert "unique_agents" in stats
        assert "unique_actions" in stats
        assert "learning_efficiency" in stats
        assert "module_performance" in stats

        assert stats["total_experiences"] == 200
        assert stats["unique_agents"] == 10

    def test_compute_learning_statistics_empty(self):
        """Test learning statistics with empty DataFrame."""
        result = compute_learning_statistics(pd.DataFrame())

        assert result["total_experiences"] == 0
        assert result["avg_reward"] == 0.0

    def test_compute_learning_statistics_minimal(self):
        """Test learning statistics with minimal columns."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [i * 0.1 for i in range(20)],
                "agent_id": ["agent_1"] * 20,
            }
        )

        stats = compute_learning_statistics(df)

        assert stats["total_experiences"] == 20
        assert stats["unique_agents"] == 1

    def test_compute_learning_statistics_module_performance(self, sample_module_data):
        """Test module performance in statistics."""
        stats = compute_learning_statistics(sample_module_data)

        assert "module_performance" in stats
        assert "decision" in stats["module_performance"]
        assert "memory" in stats["module_performance"]

        decision_perf = stats["module_performance"]["decision"]
        assert "avg_reward" in decision_perf
        assert "total_experiences" in decision_perf

    def test_compute_agent_learning_curves(self, sample_agent_data):
        """Test agent learning curves computation."""
        curves = compute_agent_learning_curves(sample_agent_data)

        assert isinstance(curves, dict)
        assert "agent_1" in curves
        assert "agent_2" in curves
        assert "agent_3" in curves
        assert len(curves["agent_1"]) == 50

    def test_compute_agent_learning_curves_empty(self):
        """Test learning curves with empty DataFrame."""
        result = compute_agent_learning_curves(pd.DataFrame())

        assert result == {}

    def test_compute_agent_learning_curves_no_agent_id(self):
        """Test learning curves without agent_id column."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": range(20),
            }
        )

        result = compute_agent_learning_curves(df)

        assert result == {}

    def test_compute_agent_learning_curves_with_moving_average(self):
        """Test learning curves with pre-computed moving average."""
        df = pd.DataFrame(
            {
                "step": range(30),
                "agent_id": ["agent_1"] * 30,
                "reward": range(30),
                "reward_ma": [i * 0.9 for i in range(30)],  # Pre-computed MA
            }
        )

        curves = compute_agent_learning_curves(df)

        assert "agent_1" in curves
        assert len(curves["agent_1"]) == 30

    def test_compute_learning_efficiency_metrics(self, sample_learning_data):
        """Test learning efficiency metrics computation."""
        metrics = compute_learning_efficiency_metrics(sample_learning_data)

        assert isinstance(metrics, dict)
        assert "reward_efficiency" in metrics
        assert "action_diversity" in metrics
        assert "learning_stability" in metrics
        assert "convergence_rate" in metrics

        assert 0 <= metrics["reward_efficiency"] <= 1
        assert 0 <= metrics["learning_stability"] <= 1

    def test_compute_learning_efficiency_metrics_empty(self):
        """Test efficiency metrics with empty DataFrame."""
        result = compute_learning_efficiency_metrics(pd.DataFrame())

        assert result["reward_efficiency"] == 0.0
        assert result["convergence_rate"] == 0.0

    def test_compute_learning_efficiency_metrics_short_data(self):
        """Test efficiency metrics with short time series."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "reward": range(10),
                "action_taken_mapped": [i % 3 for i in range(10)],
            }
        )

        metrics = compute_learning_efficiency_metrics(df)

        # Short data should have zero convergence rate
        assert metrics["convergence_rate"] == 0.0

    def test_compute_module_performance_comparison(self, sample_module_data):
        """Test module performance comparison computation."""
        comparison = compute_module_performance_comparison(sample_module_data)

        assert isinstance(comparison, dict)
        assert "decision" in comparison
        assert "memory" in comparison

        decision_stats = comparison["decision"]
        assert "reward_stats" in decision_stats
        assert "experience_count" in decision_stats
        assert "trend" in decision_stats

    def test_compute_module_performance_comparison_empty(self):
        """Test module comparison with empty DataFrame."""
        result = compute_module_performance_comparison(pd.DataFrame())

        assert result == {}

    def test_compute_module_performance_comparison_no_module(self):
        """Test module comparison without module_type column."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": range(20),
            }
        )

        result = compute_module_performance_comparison(df)

        assert result == {}


class TestLearningAnalysis:
    """Test learning analysis functions."""

    def test_analyze_learning_performance(self, tmp_path, sample_learning_data):
        """Test learning performance analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_learning_performance(sample_learning_data, ctx)

        output_file = tmp_path / "learning_performance.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "statistics" in data
        assert "efficiency" in data

    def test_analyze_agent_learning_curves(self, tmp_path, sample_agent_data):
        """Test agent learning curves analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_agent_learning_curves(sample_agent_data, ctx)

        output_file = tmp_path / "agent_learning_curves.json"
        assert output_file.exists()

        with open(output_file) as f:
            curves = json.load(f)

        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_analyze_module_performance(self, tmp_path, sample_module_data):
        """Test module performance analysis."""
        ctx = AnalysisContext(output_path=tmp_path)

        analyze_module_performance(sample_module_data, ctx)

        output_file = tmp_path / "module_performance_comparison.json"
        assert output_file.exists()

    def test_analyze_learning_progress(self, tmp_path):
        """Test learning progress analysis."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.learning.analyze.process_learning_progress_data") as mock_process:
            mock_process.return_value = pd.DataFrame(
                {
                    "step": range(50),
                    "reward": [i * 0.1 for i in range(50)],
                    "action_count": [5 + i % 10 for i in range(50)],
                    "unique_actions": [3 + i % 5 for i in range(50)],
                }
            )

            analyze_learning_progress(str(exp_path), ctx)

            output_file = tmp_path / "learning_progress.json"
            assert output_file.exists()

    def test_analyze_learning_progress_no_data(self, tmp_path):
        """Test learning progress with no data."""
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.learning.analyze.process_learning_progress_data") as mock_process:
            mock_process.return_value = pd.DataFrame()

            analyze_learning_progress(str(exp_path), ctx)

            # Should handle gracefully


class TestLearningVisualization:
    """Test learning visualization functions."""

    def test_plot_learning_curves(self, tmp_path, sample_learning_data):
        """Test learning curves plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_learning_curves(sample_learning_data, ctx)

        plot_file = tmp_path / "learning_curves.png"
        assert plot_file.exists()

    def test_plot_learning_curves_no_agent(self, tmp_path):
        """Test learning curves without agent_id column."""
        df = pd.DataFrame(
            {
                "step": range(50),
                "reward": [i * 0.1 for i in range(50)],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_learning_curves(df, ctx)

        plot_file = tmp_path / "learning_curves.png"
        assert plot_file.exists()

    def test_plot_learning_curves_empty(self, tmp_path):
        """Test learning curves with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_learning_curves(pd.DataFrame(), ctx)

        # Should handle gracefully

    def test_plot_reward_distribution(self, tmp_path, sample_learning_data):
        """Test reward distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_reward_distribution(sample_learning_data, ctx)

        plot_file = tmp_path / "reward_distribution.png"
        assert plot_file.exists()

    def test_plot_reward_distribution_no_reward(self, tmp_path):
        """Test reward distribution without reward column."""
        df = pd.DataFrame({"step": range(10)})

        ctx = AnalysisContext(output_path=tmp_path)
        plot_reward_distribution(df, ctx)

        # Should handle gracefully

    def test_plot_module_performance(self, tmp_path, sample_module_data):
        """Test module performance plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_module_performance(sample_module_data, ctx)

        plot_file = tmp_path / "module_performance.png"
        assert plot_file.exists()

    def test_plot_module_performance_no_module(self, tmp_path):
        """Test module performance without module_type column."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": range(20),
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_module_performance(df, ctx)

        # Should handle gracefully

    def test_plot_action_frequencies(self, tmp_path, sample_learning_data):
        """Test action frequencies plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_action_frequencies(sample_learning_data, ctx)

        # Should create output

    def test_plot_learning_efficiency(self, tmp_path, sample_learning_data):
        """Test learning efficiency plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_learning_efficiency(sample_learning_data, ctx)

        # Should create output

    def test_plot_reward_vs_step(self, tmp_path, sample_learning_data):
        """Test reward vs step plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_reward_vs_step(sample_learning_data, ctx)

        # Should create output

    def test_plot_with_custom_options(self, tmp_path, sample_learning_data):
        """Test plotting with custom options."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_learning_curves(sample_learning_data, ctx, figsize=(10, 8), dpi=150)

        plot_file = tmp_path / "learning_curves.png"
        assert plot_file.exists()


class TestLearningModule:
    """Test learning module integration."""

    def test_learning_module_registration(self):
        """Test module registration."""
        assert learning_module.name == "learning"
        assert (
            learning_module.description
            == "Analysis of learning performance, agent learning curves, and module efficiency"
        )

    def test_learning_module_function_names(self):
        """Test module function names."""
        functions = learning_module.get_function_names()

        assert "analyze_performance" in functions
        assert "analyze_curves" in functions
        assert "analyze_modules" in functions
        assert "plot_curves" in functions
        assert "plot_distribution" in functions

    def test_learning_module_function_groups(self):
        """Test module function groups."""
        groups = learning_module.get_function_groups()

        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "performance" in groups
        assert "comparison" in groups
        assert "basic" in groups

    def test_learning_module_data_processor(self):
        """Test module data processor."""
        processor = learning_module.get_data_processor()
        assert processor is not None

    def test_learning_module_supports_database(self):
        """Test database support."""
        assert learning_module.supports_database() is True
        assert learning_module.get_db_filename() == "simulation.db"

    def test_module_validator(self):
        """Test module validator."""
        validator = learning_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = learning_module.get_functions()
        assert len(functions) >= 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_statistics_single_step(self):
        """Test statistics with single time step."""
        df = pd.DataFrame(
            {
                "step": [0],
                "reward": [1.0],
                "agent_id": ["agent_1"],
            }
        )

        stats = compute_learning_statistics(df)

        assert stats["total_experiences"] == 1

    def test_compute_statistics_with_nan(self):
        """Test statistics with NaN values."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [i if i % 3 != 0 else np.nan for i in range(20)],
                "agent_id": ["agent_1"] * 20,
            }
        )

        # Should handle NaN values
        stats = compute_learning_statistics(df)
        assert isinstance(stats, dict)

    def test_compute_efficiency_zero_reward_range(self):
        """Test efficiency with zero reward range."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [5.0] * 20,  # Constant reward
                "action_taken_mapped": [i % 3 for i in range(20)],
            }
        )

        metrics = compute_learning_efficiency_metrics(df)

        # Zero range should result in 0.5 efficiency
        assert metrics["reward_efficiency"] == 0.5

    def test_compute_efficiency_negative_rewards(self):
        """Test efficiency with negative rewards."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [-5 + i * 0.5 for i in range(20)],
                "action_taken_mapped": [i % 4 for i in range(20)],
            }
        )

        metrics = compute_learning_efficiency_metrics(df)

        assert isinstance(metrics, dict)
        assert "reward_efficiency" in metrics

    def test_compute_curves_single_agent_single_step(self):
        """Test learning curves with single step per agent."""
        df = pd.DataFrame(
            {
                "step": [0, 1],
                "agent_id": ["agent_1", "agent_2"],
                "reward": [1.0, 2.0],
            }
        )

        curves = compute_agent_learning_curves(df)

        # Single step agents should still have curves
        assert "agent_1" in curves
        assert "agent_2" in curves

    def test_analyze_performance_with_progress(self, tmp_path, sample_learning_data):
        """Test performance analysis with progress reporting."""
        ctx = AnalysisContext(output_path=tmp_path)

        progress_calls = []
        ctx.report_progress = lambda msg, prog: progress_calls.append((msg, prog))

        analyze_learning_performance(sample_learning_data, ctx)

        # Should have called progress
        assert len(progress_calls) > 0

    def test_plot_curves_single_point_per_agent(self, tmp_path):
        """Test plotting curves with single point per agent."""
        df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "agent_id": ["agent_1", "agent_2", "agent_3"],
                "reward": [1.0, 2.0, 3.0],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_learning_curves(df, ctx)

        # Should handle gracefully (agents with single point won't be plotted)

    def test_compute_module_performance_single_module(self):
        """Test module performance with single module type."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [i * 0.1 for i in range(20)],
                "module_type": ["decision"] * 20,
            }
        )

        comparison = compute_module_performance_comparison(df)

        assert "decision" in comparison
        assert len(comparison) == 1

    def test_compute_efficiency_high_variance(self):
        """Test efficiency with high reward variance."""
        df = pd.DataFrame(
            {
                "step": range(50),
                "reward": [i * 10 * (1 if i % 2 == 0 else -1) for i in range(50)],
                "action_taken_mapped": [i % 5 for i in range(50)],
            }
        )

        metrics = compute_learning_efficiency_metrics(df)

        # High variance should result in low stability
        assert metrics["learning_stability"] < 0.5

    def test_compute_efficiency_perfect_convergence(self):
        """Test efficiency with perfect convergence."""
        df = pd.DataFrame(
            {
                "step": range(100),
                "reward": [5.0] * 100,  # Perfect convergence - same reward throughout
                "action_taken_mapped": [i % 3 for i in range(100)],
            }
        )

        metrics = compute_learning_efficiency_metrics(df)

        # Perfect convergence should have high convergence rate
        assert metrics["convergence_rate"] > 0.5

    def test_compute_statistics_all_same_action(self):
        """Test statistics when all actions are the same."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "reward": [i * 0.1 for i in range(20)],
                "agent_id": ["agent_1"] * 20,
                "action_taken": [5] * 20,
                "action_taken_mapped": [2] * 20,
            }
        )

        stats = compute_learning_statistics(df)

        assert stats["unique_actions"] == 1
        assert stats["unique_actions_mapped"] == 1


class TestLearningHelperFunctions:
    """Test helper functions in learning module."""

    def test_process_learning_data(self, tmp_path):
        """Test processing learning data from experiment."""
        from farm.analysis.learning.data import process_learning_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        # Create mock database
        db_path = exp_path / "simulation.db"
        db_path.touch()

        with patch("farm.analysis.learning.data.SessionManager") as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []

            result = process_learning_data(exp_path)

            assert isinstance(result, pd.DataFrame)

    def test_process_learning_progress_data(self, tmp_path):
        """Test processing learning progress data."""
        from farm.analysis.learning.data import process_learning_progress_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.learning.data.SessionManager") as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []

            result = process_learning_progress_data(exp_path)

            assert isinstance(result, pd.DataFrame)

    def test_calculate_trend_increasing(self):
        """Test trend calculation with increasing rewards."""
        from farm.analysis.common.utils import calculate_trend

        rewards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        trend = calculate_trend(rewards)

        assert trend > 0

    def test_calculate_trend_decreasing(self):
        """Test trend calculation with decreasing rewards."""
        from farm.analysis.common.utils import calculate_trend

        rewards = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        trend = calculate_trend(rewards)

        assert trend < 0
