"""
Comprehensive tests for temporal analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.temporal import (
    temporal_module,
    compute_temporal_patterns,
    compute_temporal_statistics,
    compute_event_segmentation_metrics,
    compute_temporal_efficiency_metrics,
    analyze_temporal_patterns,
    analyze_event_segmentation,
    analyze_time_series_overview,
    analyze_temporal_efficiency,
    plot_temporal_patterns,
    plot_rolling_averages,
    plot_event_segmentation,
    plot_temporal_efficiency,
    plot_action_type_evolution,
    plot_reward_trends,
    process_temporal_data,
    process_time_series_data,
    process_event_segmentation_data,
    extract_temporal_patterns,
)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data."""
    return pd.DataFrame(
        {
            "time_period": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "action_type": ["move", "gather", "attack"] * 4,
            "action_count": [10, 8, 3, 12, 10, 5, 15, 12, 7, 18, 15, 9],
            "avg_reward": [1.0, 2.5, 0.8, 1.2, 2.8, 1.0, 1.5, 3.0, 1.2, 1.8, 3.2, 1.5],
        }
    )


@pytest.fixture
def sample_patterns_data():
    """Create sample temporal patterns data."""
    return pd.DataFrame(
        {
            "time_period": range(10),
            "action_type": ["move"] * 10,
            "action_count": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32],
            "avg_reward": [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2],
            "success_rate": [0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92],
            "rolling_avg_reward": [1.0, 1.1, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.8, 3.0],
            "rolling_action_count": [10, 11, 12, 14, 16, 18, 21, 24, 27, 30],
        }
    )


@pytest.fixture
def sample_segments_data():
    """Create sample segment data for event analysis."""
    pre_event = pd.DataFrame(
        {
            "agent_id": ["agent_1", "agent_2", "agent_3"] * 3,
            "action_type": ["move", "gather", "attack"] * 3,
            "reward": [1.0, 2.0, 0.5] * 3,
        }
    )

    post_event = pd.DataFrame(
        {
            "agent_id": ["agent_1", "agent_2", "agent_3"] * 2,
            "action_type": ["move", "gather", "attack"] * 2,
            "reward": [1.5, 2.5, 1.0] * 2,
        }
    )

    return {
        "pre_event": pre_event,
        "post_event": post_event,
    }


class TestTemporalComputations:
    """Test temporal statistical computations."""

    def test_compute_temporal_statistics(self, sample_time_series_data):
        """Test temporal statistics computation."""
        stats = compute_temporal_statistics(sample_time_series_data)

        assert "total_periods" in stats
        assert "total_actions" in stats
        assert "action_types" in stats
        assert "action_diversity" in stats
        assert "avg_reward_trend" in stats
        assert "action_patterns" in stats

        assert stats["total_periods"] == 4
        assert stats["action_diversity"] == 3
        assert "move" in stats["action_patterns"]

    def test_compute_temporal_statistics_empty(self):
        """Test temporal statistics with empty DataFrame."""
        stats = compute_temporal_statistics(pd.DataFrame())

        assert stats["total_periods"] == 0
        assert stats["total_actions"] == 0
        assert stats["avg_reward_trend"] == 0.0

    def test_compute_event_segmentation_metrics(self, sample_segments_data):
        """Test event segmentation metrics computation."""
        event_steps = [50]  # Event at step 50

        metrics = compute_event_segmentation_metrics(sample_segments_data, event_steps)

        assert "segment_metrics" in metrics
        assert "event_impacts" in metrics

        # Check segment metrics
        assert "pre_event" in metrics["segment_metrics"]
        assert "post_event" in metrics["segment_metrics"]

        pre_metrics = metrics["segment_metrics"]["pre_event"]
        assert "action_count" in pre_metrics
        assert "unique_actions" in pre_metrics
        assert "avg_reward" in pre_metrics

    def test_compute_event_segmentation_metrics_empty(self):
        """Test event segmentation with empty data."""
        metrics = compute_event_segmentation_metrics({}, [])
        assert metrics == {}

    def test_compute_event_segmentation_empty_segment(self):
        """Test event segmentation with some empty segments."""
        segments = {
            "segment1": pd.DataFrame(),
            "segment2": pd.DataFrame(
                {
                    "agent_id": ["agent_1"],
                    "action_type": ["move"],
                    "reward": [1.0],
                }
            ),
        }

        metrics = compute_event_segmentation_metrics(segments, [10])

        assert metrics["segment_metrics"]["segment1"]["action_count"] == 0
        assert metrics["segment_metrics"]["segment2"]["action_count"] == 1

    def test_compute_temporal_patterns(self, sample_patterns_data):
        """Test temporal patterns computation."""
        patterns = compute_temporal_patterns(sample_patterns_data)

        assert "move" in patterns
        assert "peak_period" in patterns["move"]
        assert "peak_count" in patterns["move"]
        assert "reward_trend" in patterns["move"]

    def test_compute_temporal_patterns_empty(self):
        """Test temporal patterns with empty DataFrame."""
        patterns = compute_temporal_patterns(pd.DataFrame())
        assert patterns == {}

    def test_compute_temporal_efficiency_metrics(self):
        """Test temporal efficiency metrics computation."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_count": [10, 12, 15, 13, 18, 20, 17, 22, 25, 23],
                "reward": [10.0, 14.4, 22.5, 16.9, 32.4, 40.0, 28.9, 48.4, 62.5, 52.9],
                "success_count": [7, 9, 11, 10, 13, 15, 12, 16, 18, 17],
            }
        )

        metrics = compute_temporal_efficiency_metrics(df)

        assert "avg_actions_per_step" in metrics
        assert "avg_reward_per_step" in metrics
        assert "avg_success_rate" in metrics
        assert "efficiency_trend" in metrics

        assert metrics["avg_actions_per_step"] > 0
        assert metrics["avg_reward_per_step"] > 0

    def test_compute_temporal_efficiency_metrics_empty(self):
        """Test efficiency metrics with empty DataFrame."""
        metrics = compute_temporal_efficiency_metrics(pd.DataFrame())

        assert metrics["avg_actions_per_step"] == 0.0
        assert metrics["avg_reward_per_step"] == 0.0


class TestTemporalAnalysis:
    """Test temporal analysis functions."""

    @patch("farm.analysis.temporal.analyze.extract_temporal_patterns")
    def test_analyze_temporal_patterns(self, mock_extract, tmp_path, sample_time_series_data):
        """Test temporal patterns analysis."""
        mock_extract.return_value = sample_time_series_data

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_temporal_patterns(str(tmp_path), ctx)

        output_file = tmp_path / "temporal_statistics.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "patterns" in data
        assert "efficiency" in data

    @patch("farm.analysis.temporal.analyze.process_event_segmentation_data")
    def test_analyze_event_segmentation(self, mock_process, tmp_path, sample_segments_data):
        """Test event segmentation analysis."""
        mock_process.return_value = sample_segments_data

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_event_segmentation(str(tmp_path), ctx, event_steps=[50])

        output_file = tmp_path / "event_segmentation.json"
        assert output_file.exists()

    @patch("farm.analysis.temporal.analyze.process_time_series_data")
    def test_analyze_time_series_overview(self, mock_process, tmp_path, sample_time_series_data):
        """Test time series overview analysis."""
        mock_process.return_value = sample_time_series_data

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_time_series_overview(str(tmp_path), ctx)

        # Should create some output
        assert len(list(tmp_path.glob("*.json"))) > 0

    @patch("farm.analysis.temporal.analyze.process_time_series_data")
    def test_analyze_temporal_efficiency(self, mock_process, tmp_path):
        """Test temporal efficiency analysis."""
        mock_data = pd.DataFrame(
            {
                "step": range(10),
                "action_count": [10] * 10,
                "reward": [10.0] * 10,
                "success_count": [7] * 10,
            }
        )
        mock_process.return_value = mock_data

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_temporal_efficiency(str(tmp_path), ctx)

        output_file = tmp_path / "temporal_efficiency.json"
        assert output_file.exists()


class TestTemporalVisualization:
    """Test temporal visualization functions."""

    def test_plot_temporal_patterns(self, sample_patterns_data, tmp_path):
        """Test temporal patterns plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_temporal_patterns(sample_patterns_data, ctx)

        plot_file = tmp_path / "temporal_patterns.png"
        assert plot_file.exists()

    def test_plot_temporal_patterns_empty(self, tmp_path):
        """Test temporal patterns plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_temporal_patterns(pd.DataFrame(), ctx)

        # Should not create file when no data
        plot_file = tmp_path / "temporal_patterns.png"
        assert not plot_file.exists()

    def test_plot_rolling_averages(self, sample_patterns_data, tmp_path):
        """Test rolling averages plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_rolling_averages(sample_patterns_data, ctx)

        plot_file = tmp_path / "rolling_averages.png"
        assert plot_file.exists()

    def test_plot_event_segmentation(self, sample_segments_data, tmp_path):
        """Test event segmentation plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        event_steps = [50]

        # Create the expected segmentation data structure
        segmentation_data = {
            "segment_metrics": {
                "pre_event": {
                    "action_count": 9,
                    "avg_reward": 1.17,
                    "unique_actions": 3,
                    "unique_agents": 3,
                },
                "post_event": {
                    "action_count": 6,
                    "avg_reward": 1.67,
                    "unique_actions": 3,
                    "unique_agents": 3,
                },
            }
        }

        plot_event_segmentation(segmentation_data, event_steps, ctx)

        plot_file = tmp_path / "event_segmentation.png"
        assert plot_file.exists()

    def test_plot_temporal_efficiency(self, tmp_path):
        """Test temporal efficiency plotting."""
        efficiency_data = {
            "avg_actions_per_step": 18.5,
            "avg_reward_per_step": 10.0,
            "avg_success_rate": 0.7,
            "efficiency_trend": 0.1,
        }

        ctx = AnalysisContext(output_path=tmp_path)
        plot_temporal_efficiency(efficiency_data, ctx)

        plot_file = tmp_path / "temporal_efficiency.png"
        assert plot_file.exists()

    def test_plot_action_type_evolution(self, sample_time_series_data, tmp_path):
        """Test action type evolution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_type_evolution(sample_time_series_data, ctx)

        plot_file = tmp_path / "action_evolution.png"
        assert plot_file.exists()

    def test_plot_reward_trends(self, sample_time_series_data, tmp_path):
        """Test reward trends plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_reward_trends(sample_time_series_data, ctx)

        plot_file = tmp_path / "reward_trends.png"
        assert plot_file.exists()


class TestTemporalModule:
    """Test temporal module integration."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert temporal_module.name == "temporal"
        assert len(temporal_module.get_function_names()) > 0
        assert "analyze_overview" in temporal_module.get_function_names()

    def test_module_groups(self):
        """Test module function groups."""
        groups = temporal_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = temporal_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert temporal_module.supports_database() is True
        assert temporal_module.get_db_filename() == "simulation.db"

    def test_module_validator(self):
        """Test module validator."""
        validator = temporal_module.get_validator()
        assert validator is not None

        # Test with valid data
        valid_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "action_count": [10, 15, 20],
                "avg_reward": [1.0, 1.5, 2.0],
            }
        )
        validator.validate(valid_df)  # Should not raise

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        expected_functions = [
            "analyze_overview",
            "analyze_patterns",
            "analyze_efficiency",
            "analyze_segmentation",
            "plot_patterns",
            "plot_rolling",
            "plot_efficiency",
        ]

        function_names = temporal_module.get_function_names()

        for func in expected_functions:
            assert func in function_names


class TestDataProcessing:
    """Test temporal data processing functions."""

    @patch("farm.analysis.temporal.data.SessionManager")
    @patch("farm.analysis.temporal.data.ActionRepository")
    def test_process_temporal_data(self, mock_repo_class, mock_session_class, tmp_path):
        """Test processing temporal data from database."""
        # Create mock database file
        db_path = tmp_path / "simulation.db"
        db_path.touch()

        # Mock session and repository
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        # Mock action data
        mock_action = MagicMock()
        mock_action.step_number = 0
        mock_action.action_type = "move"
        mock_action.reward = 1.0

        mock_repo.get_all_actions.return_value = [mock_action] * 10

        # Process data
        df = process_temporal_data(tmp_path, use_database=True)

        # Should return DataFrame
        assert isinstance(df, pd.DataFrame)

    def test_extract_temporal_patterns(self):
        """Test temporal pattern extraction."""
        df = pd.DataFrame(
            {
                "step": range(20),
                "agent_id": ["agent_1"] * 20,
                "action_type": ["move"] * 10 + ["gather"] * 10,
                "reward": list(range(10)) + list(range(10, 20)),
            }
        )

        patterns = extract_temporal_patterns(df, rolling_window_size=5)

        assert isinstance(patterns, pd.DataFrame)
        assert "time_period" in patterns.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_temporal_statistics_single_period(self):
        """Test temporal statistics with single time period."""
        df = pd.DataFrame(
            {
                "time_period": [0, 0, 0],
                "action_type": ["move", "gather", "attack"],
                "action_count": [10, 8, 3],
                "avg_reward": [1.0, 2.5, 0.8],
            }
        )

        stats = compute_temporal_statistics(df)

        assert stats["total_periods"] == 1
        assert stats["action_diversity"] == 3

    def test_compute_patterns_with_nan_values(self, sample_patterns_data):
        """Test pattern computation with NaN values."""
        df = sample_patterns_data.copy()
        df.loc[0, "action_count"] = np.nan
        df.loc[1, "avg_reward"] = np.nan

        # Should handle NaN gracefully
        patterns = compute_temporal_patterns(df)
        assert "move" in patterns

    def test_plot_with_progress_callback(self, sample_patterns_data, tmp_path):
        """Test plotting with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)
        plot_temporal_patterns(sample_patterns_data, ctx)

        plot_file = tmp_path / "temporal_patterns.png"
        assert plot_file.exists()

    def test_zero_action_counts(self):
        """Test handling of zero action counts."""
        df = pd.DataFrame(
            {
                "time_period": range(5),
                "action_type": ["move"] * 5,
                "action_count": [0, 0, 0, 0, 0],
                "avg_reward": [0.0] * 5,
            }
        )

        stats = compute_temporal_statistics(df)

        assert stats["total_actions"] == 0
        # Should not raise errors

    def test_compute_efficiency_all_failed_actions(self):
        """Test efficiency metrics when all actions fail."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_count": [10] * 10,
                "reward": [0.0] * 10,
                "success_count": [0] * 10,
            }
        )

        metrics = compute_temporal_efficiency_metrics(df)

        assert metrics["avg_reward_per_step"] == 0.0
        assert metrics["avg_success_rate"] == 0.0
