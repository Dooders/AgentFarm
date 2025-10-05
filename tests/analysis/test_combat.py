"""
Comprehensive tests for combat analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pandas as pd
import numpy as np
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.combat import (
    combat_module,
    compute_combat_statistics,
    compute_agent_combat_performance,
    compute_combat_efficiency_metrics,
    compute_combat_temporal_patterns,
    analyze_combat_overview,
    analyze_agent_combat_performance,
    analyze_combat_efficiency,
    analyze_combat_temporal_patterns,
    plot_combat_overview,
    plot_combat_success_rate,
    plot_agent_combat_performance,
    plot_combat_efficiency,
    plot_damage_distribution,
    plot_combat_temporal_patterns,
    process_combat_data,
    process_combat_metrics_data,
    process_agent_combat_stats,
)


@pytest.fixture
def sample_combat_data():
    """Create sample combat action data."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "agent_id": [
                "agent_1",
                "agent_2",
                "agent_1",
                "agent_2",
                "agent_3",
                "agent_1",
                "agent_2",
                "agent_3",
                "agent_1",
                "agent_3",
            ],
            "target_id": [
                "agent_4",
                "agent_5",
                "agent_4",
                "agent_5",
                "agent_6",
                "agent_5",
                "agent_6",
                "agent_4",
                "agent_6",
                "agent_5",
            ],
            "damage_dealt": [10, 15, 12, 0, 20, 18, 14, 16, 0, 22],
            "reward": [1.0, 1.5, 1.2, 0.0, 2.0, 1.8, 1.4, 1.6, 0.0, 2.2],
        }
    )


@pytest.fixture
def sample_metrics_data():
    """Create sample combat metrics data."""
    return pd.DataFrame(
        {
            "step": range(5),
            "combat_encounters": [2, 2, 2, 2, 2],
            "successful_attacks": [2, 1, 2, 2, 1],
        }
    )


@pytest.fixture
def sample_agent_combat_data():
    """Create sample agent combat statistics."""
    return pd.DataFrame(
        {
            "agent_id": ["agent_1", "agent_2", "agent_3"],
            "total_damage": [40, 29, 38],
            "total_attacks": [4, 3, 3],
            "success_rate": [0.75, 0.67, 1.0],
        }
    )


class TestCombatComputations:
    """Test combat statistical computations."""

    def test_compute_combat_statistics(self, sample_combat_data, sample_metrics_data):
        """Test combat statistics computation."""
        stats = compute_combat_statistics(sample_combat_data, sample_metrics_data)

        assert "combat_actions" in stats
        assert "combat_metrics" in stats
        assert "overall_combat" in stats

        # Check combat actions stats
        assert "total_attacks" in stats["combat_actions"]
        assert "successful_attacks" in stats["combat_actions"]
        assert "success_rate" in stats["combat_actions"]
        assert stats["combat_actions"]["total_attacks"] == 10

    def test_compute_combat_statistics_empty(self):
        """Test combat statistics with empty DataFrames."""
        stats = compute_combat_statistics(pd.DataFrame(), pd.DataFrame())
        assert stats == {}

    def test_compute_combat_statistics_only_actions(self, sample_combat_data):
        """Test combat statistics with only combat actions data."""
        stats = compute_combat_statistics(sample_combat_data, pd.DataFrame())

        assert "combat_actions" in stats
        assert "combat_metrics" not in stats
        assert "overall_combat" not in stats

    def test_compute_agent_combat_performance(self, sample_agent_combat_data):
        """Test agent combat performance computation."""
        performance = compute_agent_combat_performance(sample_agent_combat_data)

        assert "top_performers" in performance
        assert "rankings" in performance
        assert "performance_tiers" in performance

        # Check top performers
        assert "by_damage" in performance["top_performers"]
        assert "by_success_rate" in performance["top_performers"]
        assert "by_activity" in performance["top_performers"]

        # Check rankings
        assert len(performance["rankings"]) == 3
        for agent_id in ["agent_1", "agent_2", "agent_3"]:
            assert agent_id in performance["rankings"]
            assert "damage_rank" in performance["rankings"][agent_id]

    def test_compute_agent_combat_performance_empty(self):
        """Test agent combat performance with empty DataFrame."""
        performance = compute_agent_combat_performance(pd.DataFrame())
        assert performance == {}

    def test_compute_combat_efficiency_metrics(self, sample_combat_data):
        """Test combat efficiency metrics computation."""
        efficiency = compute_combat_efficiency_metrics(sample_combat_data)

        assert "overall_success_rate" in efficiency
        assert "damage_efficiency" in efficiency
        assert "combat_intensity" in efficiency
        assert "reward_efficiency" in efficiency

        # Check values are reasonable
        assert 0 <= efficiency["overall_success_rate"] <= 1
        assert efficiency["damage_efficiency"] >= 0
        assert efficiency["combat_intensity"] >= 0

    def test_compute_combat_efficiency_metrics_empty(self):
        """Test combat efficiency with empty DataFrame."""
        efficiency = compute_combat_efficiency_metrics(pd.DataFrame())

        assert efficiency["overall_success_rate"] == 0.0
        assert efficiency["damage_efficiency"] == 0.0
        assert efficiency["combat_intensity"] == 0.0

    def test_compute_combat_temporal_patterns(self, sample_combat_data, sample_metrics_data):
        """Test combat temporal patterns computation."""
        patterns = compute_combat_temporal_patterns(sample_combat_data, sample_metrics_data)

        assert "frequency_trend" in patterns
        assert "success_rate_trend" in patterns
        assert "damage_trend" in patterns
        assert "peak_combat" in patterns

        # Check peak combat info
        assert "step" in patterns["peak_combat"]
        assert "encounters" in patterns["peak_combat"]

    def test_compute_combat_temporal_patterns_empty(self):
        """Test combat temporal patterns with empty DataFrames."""
        patterns = compute_combat_temporal_patterns(pd.DataFrame(), pd.DataFrame())
        assert patterns == {}

    def test_compute_combat_temporal_patterns_only_metrics(self, sample_metrics_data):
        """Test temporal patterns with only metrics data."""
        patterns = compute_combat_temporal_patterns(pd.DataFrame(), sample_metrics_data)

        assert "frequency_trend" in patterns
        assert "peak_combat" in patterns
        assert "damage_trend" not in patterns


class TestCombatAnalysis:
    """Test combat analysis functions."""

    @patch("farm.analysis.combat.analyze.process_combat_data")
    @patch("farm.analysis.combat.analyze.process_combat_metrics_data")
    def test_analyze_combat_overview(self, mock_metrics, mock_combat, tmp_path):
        """Test combat overview analysis."""
        # Mock data processing
        mock_combat.return_value = pd.DataFrame(
            {
                "step": range(5),
                "agent_id": ["agent_1"] * 5,
                "target_id": ["agent_2"] * 5,
                "damage_dealt": [10, 12, 15, 0, 18],
                "reward": [1.0, 1.2, 1.5, 0.0, 1.8],
            }
        )
        mock_metrics.return_value = pd.DataFrame(
            {
                "step": range(5),
                "combat_encounters": [1] * 5,
                "successful_attacks": [1, 1, 1, 0, 1],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_combat_overview(str(tmp_path), ctx)

        output_file = tmp_path / "combat_overview.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "combat_actions" in data or "combat_metrics" in data

    @patch("farm.analysis.combat.analyze.process_agent_combat_stats")
    def test_analyze_agent_combat_performance(self, mock_stats, tmp_path):
        """Test agent combat performance analysis."""
        mock_stats.return_value = pd.DataFrame(
            {
                "agent_id": ["agent_1", "agent_2", "agent_3"],
                "total_damage": [100, 80, 120],
                "total_attacks": [10, 8, 12],
                "success_rate": [0.8, 0.75, 0.9],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_agent_combat_performance(str(tmp_path), ctx)

        output_file = tmp_path / "agent_combat_performance.json"
        assert output_file.exists()

    @patch("farm.analysis.combat.analyze.process_agent_combat_stats")
    def test_analyze_agent_combat_performance_empty(self, mock_stats, tmp_path):
        """Test agent combat performance with no data."""
        mock_stats.return_value = pd.DataFrame()

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_agent_combat_performance(str(tmp_path), ctx)

        # Should not create file when no data
        output_file = tmp_path / "agent_combat_performance.json"
        assert not output_file.exists()

    @patch("farm.analysis.combat.analyze.process_combat_data")
    def test_analyze_combat_efficiency(self, mock_combat, tmp_path):
        """Test combat efficiency analysis."""
        mock_combat.return_value = pd.DataFrame(
            {
                "step": range(10),
                "agent_id": ["agent_1"] * 10,
                "damage_dealt": [10] * 10,
                "reward": [1.0] * 10,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_combat_efficiency(str(tmp_path), ctx)

        output_file = tmp_path / "combat_efficiency.json"
        assert output_file.exists()

    @patch("farm.analysis.combat.analyze.process_combat_data")
    @patch("farm.analysis.combat.analyze.process_combat_metrics_data")
    def test_analyze_combat_temporal_patterns(self, mock_metrics, mock_combat, tmp_path):
        """Test combat temporal patterns analysis."""
        mock_combat.return_value = pd.DataFrame(
            {
                "step": range(10),
                "agent_id": ["agent_1"] * 10,
                "damage_dealt": list(range(10)),
            }
        )
        mock_metrics.return_value = pd.DataFrame(
            {
                "step": range(10),
                "combat_encounters": list(range(1, 11)),
                "successful_attacks": list(range(1, 11)),
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_combat_temporal_patterns(str(tmp_path), ctx)

        output_file = tmp_path / "combat_temporal_patterns.json"
        assert output_file.exists()


class TestCombatVisualization:
    """Test combat visualization functions."""

    def test_plot_combat_overview(self, sample_metrics_data, tmp_path):
        """Test combat overview plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_combat_overview(sample_metrics_data, ctx)

        plot_file = tmp_path / "combat_overview.png"
        assert plot_file.exists()

    def test_plot_combat_overview_empty(self, tmp_path):
        """Test combat overview plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_combat_overview(pd.DataFrame(), ctx)

        # Should not create file when no data
        plot_file = tmp_path / "combat_overview.png"
        assert not plot_file.exists()

    def test_plot_combat_success_rate(self, sample_metrics_data, tmp_path):
        """Test combat success rate plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_combat_success_rate(sample_metrics_data, ctx)

        plot_file = tmp_path / "combat_success_rate.png"
        assert plot_file.exists()

    def test_plot_agent_combat_performance(self, sample_agent_combat_data, tmp_path):
        """Test agent combat performance plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_agent_combat_performance(sample_agent_combat_data, ctx)

        plot_file = tmp_path / "agent_combat_performance.png"
        assert plot_file.exists()

    def test_plot_combat_efficiency(self, sample_combat_data, tmp_path):
        """Test combat efficiency plotting."""
        from farm.analysis.combat.compute import compute_combat_efficiency_metrics

        ctx = AnalysisContext(output_path=tmp_path)
        efficiency_data = compute_combat_efficiency_metrics(sample_combat_data)
        plot_combat_efficiency(efficiency_data, ctx)

        plot_file = tmp_path / "combat_efficiency.png"
        assert plot_file.exists()

    def test_plot_damage_distribution(self, sample_combat_data, tmp_path):
        """Test damage distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        plot_damage_distribution(sample_combat_data, ctx)

        plot_file = tmp_path / "damage_distribution.png"
        assert plot_file.exists()

    def test_plot_combat_temporal_patterns(self, sample_metrics_data, tmp_path):
        """Test combat temporal patterns plotting."""
        from farm.analysis.combat.compute import compute_combat_temporal_patterns

        ctx = AnalysisContext(output_path=tmp_path)
        patterns_data = compute_combat_temporal_patterns(pd.DataFrame(), sample_metrics_data)
        plot_combat_temporal_patterns(patterns_data, ctx)

        plot_file = tmp_path / "combat_temporal_patterns.png"
        assert plot_file.exists()


class TestCombatModule:
    """Test combat module integration."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert combat_module.name == "combat"
        assert len(combat_module.get_function_names()) > 0
        assert "analyze_overview" in combat_module.get_function_names()

    def test_module_groups(self):
        """Test module function groups."""
        groups = combat_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = combat_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert combat_module.supports_database() is True
        assert combat_module.get_db_filename() == "simulation.db"

    def test_module_validator(self):
        """Test module validator."""
        validator = combat_module.get_validator()
        assert validator is not None

        # Test with valid data
        valid_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "agent_id": ["agent_1", "agent_2", "agent_3"],
                "damage_dealt": [10, 15, 20],
            }
        )
        validator.validate(valid_df)  # Should not raise

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        expected_functions = [
            "analyze_overview",
            "analyze_agent_performance",
            "analyze_efficiency",
            "analyze_temporal",
            "plot_overview",
            "plot_success_rate",
            "plot_agent_performance",
            "plot_efficiency",
        ]

        function_names = combat_module.get_function_names()

        for func in expected_functions:
            assert func in function_names


class TestDataProcessing:
    """Test combat data processing functions."""

    @patch("farm.analysis.combat.data.SessionManager")
    @patch("farm.analysis.combat.data.ActionRepository")
    def test_process_combat_data_from_database(self, mock_repo_class, mock_session_class, tmp_path):
        """Test processing combat data from database."""
        # Create mock database file
        db_path = tmp_path / "simulation.db"
        db_path.touch()

        # Mock session and repository
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        # Mock combat action data
        mock_action = MagicMock()
        mock_action.agent_id = 1
        mock_action.action_type = "attack"
        mock_action.step_number = 0
        mock_action.action_target_id = 2
        mock_action.damage_dealt = 10
        mock_action.reward = 1.0

        mock_repo.get_actions_by_scope.return_value = [mock_action] * 5

        # Process data
        df = process_combat_data(tmp_path, use_database=True)

        # Should return DataFrame with combat data
        assert isinstance(df, pd.DataFrame)

    @patch("farm.analysis.combat.data.find_database_path")
    def test_process_combat_metrics_data(self, mock_find_db, tmp_path):
        """Test processing combat metrics data."""
        # Mock database path
        db_path = tmp_path / "simulation.db"
        db_path.touch()
        mock_find_db.return_value = db_path

        # Create mock metrics CSV
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        metrics_csv = data_dir / "step_metrics.csv"

        test_df = pd.DataFrame(
            {
                "step": range(5),
                "combat_encounters": [2, 3, 1, 4, 2],
                "successful_attacks": [1, 2, 1, 3, 1],
            }
        )
        test_df.to_csv(metrics_csv, index=False)

        # Process data
        df = process_combat_metrics_data(tmp_path, use_database=False)

        assert not df.empty
        assert "combat_encounters" in df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_combat_statistics_with_nan(self, sample_combat_data):
        """Test combat statistics with NaN values."""
        # Add some NaN values
        combat_df = sample_combat_data.copy()
        combat_df.loc[0, "damage_dealt"] = np.nan

        metrics_df = pd.DataFrame(
            {
                "step": range(5),
                "combat_encounters": [2, np.nan, 2, 2, 2],
                "successful_attacks": [2, 1, np.nan, 2, 1],
            }
        )

        # Should handle NaN gracefully
        stats = compute_combat_statistics(combat_df, metrics_df)
        assert "combat_actions" in stats

    def test_compute_agent_performance_single_agent(self):
        """Test agent performance with single agent."""
        agent_df = pd.DataFrame(
            {
                "agent_id": ["agent_1"],
                "total_damage": [100],
                "total_attacks": [10],
                "success_rate": [0.8],
            }
        )

        performance = compute_agent_combat_performance(agent_df)

        assert "top_performers" in performance
        assert len(performance["rankings"]) == 1

    def test_compute_efficiency_all_failed_attacks(self):
        """Test efficiency metrics when all attacks fail."""
        combat_df = pd.DataFrame(
            {
                "step": range(10),
                "agent_id": ["agent_1"] * 10,
                "damage_dealt": [0] * 10,
                "reward": [0.0] * 10,
            }
        )

        efficiency = compute_combat_efficiency_metrics(combat_df)

        assert efficiency["overall_success_rate"] == 0.0
        assert efficiency["damage_efficiency"] == 0.0

    def test_plot_with_progress_callback(self, sample_metrics_data, tmp_path):
        """Test plotting with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)
        plot_combat_overview(sample_metrics_data, ctx)

        plot_file = tmp_path / "combat_overview.png"
        assert plot_file.exists()

    def test_zero_combat_encounters(self):
        """Test handling of zero combat encounters."""
        metrics_df = pd.DataFrame(
            {
                "step": range(5),
                "combat_encounters": [0] * 5,
                "successful_attacks": [0] * 5,
            }
        )

        patterns = compute_combat_temporal_patterns(pd.DataFrame(), metrics_df)

        # Should still compute patterns without errors
        assert "peak_combat" in patterns
