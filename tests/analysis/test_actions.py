"""
Tests for actions analysis module.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.exceptions import DataValidationError
from farm.analysis.actions import (
    actions_module,
    analyze_action_patterns,
    analyze_action_sequences,
    analyze_success_rates,
    analyze_sequence_patterns,
    analyze_decision_patterns,
    analyze_reward_analysis,
    compute_action_sequences,
    compute_action_statistics,
    compute_success_rates,
    compute_sequence_patterns,
    compute_decision_patterns,
    compute_reward_metrics,
    plot_action_distribution,
    plot_action_sequences,
    plot_success_rates,
    plot_action_frequencies,
    plot_sequence_patterns,
    plot_decision_patterns,
    plot_reward_distributions,
)
from farm.analysis.actions.data import process_action_data
from farm.analysis.common.context import AnalysisContext


class TestActionComputations:
    """Test action statistical computations."""

    def test_compute_action_statistics(self):
        """Test action statistics computation."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": [
                    "move",
                    "gather",
                    "attack",
                    "move",
                    "gather",
                    "move",
                    "attack",
                    "gather",
                    "move",
                    "reproduce",
                ],
                "frequency": [15, 8, 3, 12, 10, 18, 5, 7, 14, 2],
                "success_rate": [0.8, 0.75, 0.67, 0.83, 0.8, 0.83, 0.8, 0.71, 0.79, 0.5],
                "avg_reward": [1.2, 2.5, 0.8, 1.1, 2.8, 1.3, 0.9, 2.2, 1.0, 3.0],
            }
        )

        stats = compute_action_statistics(df)

        assert "total_actions" in stats
        assert "action_types" in stats
        assert "most_common_action" in stats
        assert "most_common_frequency" in stats

        # The function returns statistics as nested dict, so we check the structure
        assert isinstance(stats["total_actions"], dict)
        assert isinstance(stats["action_types"], dict)
        assert len(stats["action_types"]) == 4  # move, gather, attack, reproduce
        assert stats["most_common_action"] == "move"

    def test_compute_success_rates(self):
        """Test success rate computation."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather", "attack", "reproduce"] * 5,
                "success_count": [12, 8, 3, 1, 15, 10, 5, 2, 18, 12, 7, 3, 20, 15, 8, 4, 22, 18, 10, 5],
                "total_attempts": [15, 10, 5, 2, 18, 12, 8, 3, 20, 15, 10, 4, 25, 18, 12, 6, 28, 20, 15, 8],
            }
        )

        success_rates = compute_success_rates(df)

        assert len(success_rates) == 4  # one per action type

        for action_type, rate in success_rates.items():
            assert 0 <= rate <= 1
            assert isinstance(rate, float)

        # Check that rates are reasonable aggregates
        assert 0.7 < success_rates["move"] < 0.9  # aggregate success rate for move
        assert 0.7 < success_rates["gather"] < 0.9  # aggregate success rate for gather

    def test_compute_action_sequences(self):
        """Test action sequence computation."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_sequence": [
                    ["move", "gather"],
                    ["gather", "attack"],
                    ["move", "gather", "attack"],
                    ["move"],
                    ["gather", "move"],
                    ["attack", "move"],
                    ["move", "gather"],
                    ["gather"],
                    ["move", "attack"],
                    ["gather", "move", "attack"],
                ],
                "sequence_length": [2, 2, 3, 1, 2, 2, 2, 1, 2, 3],
            }
        )

        sequences = compute_action_sequences(df)

        assert "common_sequences" in sequences
        assert "avg_sequence_length" in sequences
        assert "max_sequence_length" in sequences
        assert "transition_matrix" in sequences

        assert sequences["avg_sequence_length"] == 2.0  # average of lengths
        assert sequences["max_sequence_length"] == 3
        assert isinstance(sequences["transition_matrix"], dict)

    def test_compute_sequence_patterns(self):
        """Test action sequence pattern computation."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "seq_move_to_gather": [0.5, 0.6, 0.55, 0.7, 0.65, 0.6, 0.55, 0.7, 0.65, 0.6],
                "seq_gather_to_attack": [0.2, 0.3, 0.25, 0.35, 0.3, 0.25, 0.3, 0.35, 0.3, 0.25],
                "seq_attack_to_move": [0.8, 0.75, 0.8, 0.85, 0.8, 0.75, 0.8, 0.85, 0.8, 0.75],
            }
        )

        patterns = compute_sequence_patterns(df)

        assert "move->gather" in patterns
        assert "gather->attack" in patterns
        assert "attack->move" in patterns
        assert "most_common_sequence" in patterns

        # Check pattern structure
        for key in ["move->gather", "gather->attack", "attack->move"]:
            assert "avg_probability" in patterns[key]
            assert "max_probability" in patterns[key]
            assert "min_probability" in patterns[key]
            assert "probability_trend" in patterns[key]
            assert 0 <= patterns[key]["avg_probability"] <= 1

        # Most common sequence should be attack->move (highest avg probability)
        assert patterns["most_common_sequence"]["sequence"] == "attack->move"

    def test_compute_sequence_patterns_empty(self):
        """Test sequence pattern computation with no sequence columns."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        patterns = compute_sequence_patterns(df)

        assert patterns == {}

    def test_compute_decision_patterns(self):
        """Test decision pattern computation."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "action_type": ["move", "gather", "attack"] * 3,
                "success_rate": [0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.9, 0.8, 0.7],
            }
        )

        patterns = compute_decision_patterns(df)

        assert "avg_success_rate" in patterns
        assert "success_trend" in patterns
        assert "decision_consistency" in patterns
        assert "avg_action_diversity" in patterns
        assert "diversity_trend" in patterns

        # Check values are reasonable
        assert 0 <= patterns["avg_success_rate"] <= 1
        assert patterns["avg_action_diversity"] == 3.0  # 3 action types per step
        assert isinstance(patterns["decision_consistency"], float)

    def test_compute_decision_patterns_no_success_rate(self):
        """Test decision patterns without success_rate column."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move", "gather"] * 5,
                "frequency": [5] * 10,
            }
        )

        patterns = compute_decision_patterns(df)

        assert patterns == {}

    def test_compute_reward_metrics(self):
        """Test reward metrics computation."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 2, 2],
                "action_type": ["move", "gather", "move", "gather", "move", "gather"],
                "avg_reward": [1.0, 2.5, 1.2, 2.8, 1.1, 2.6],
                "reward_variance": [0.1, 0.2, 0.15, 0.25, 0.12, 0.22],
                "total_reward": [10.0, 25.0, 12.0, 28.0, 11.0, 26.0],
            }
        )

        metrics = compute_reward_metrics(df)

        assert "overall_avg_reward" in metrics
        assert "reward_trend" in metrics
        assert "reward_volatility" in metrics
        assert "avg_reward_variance" in metrics
        assert "total_rewards" in metrics
        assert "best_performing_action" in metrics
        assert "best_action_reward" in metrics

        # Check values are reasonable
        assert metrics["overall_avg_reward"] > 0
        assert isinstance(metrics["reward_volatility"], float)
        assert metrics["best_performing_action"] in ["move", "gather"]
        assert metrics["best_action_reward"] > 0

    def test_compute_reward_metrics_no_reward_column(self):
        """Test reward metrics without avg_reward column."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move", "gather"] * 5,
                "frequency": [5] * 10,
            }
        )

        metrics = compute_reward_metrics(df)

        assert metrics == {}

    def test_compute_action_statistics_empty_dataframe(self):
        """Test action statistics with empty DataFrame."""
        df = pd.DataFrame()

        stats = compute_action_statistics(df)

        assert stats == {}

    def test_compute_success_rates_zero_attempts(self):
        """Test success rates when total_attempts is zero."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather"],
                "success_count": [0, 10],
                "total_attempts": [0, 20],
            }
        )

        success_rates = compute_success_rates(df)

        assert success_rates["move"] == 0.0
        assert success_rates["gather"] == 0.5

    def test_compute_success_rates_missing_columns(self):
        """Test success rates with missing required columns."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather"],
                "frequency": [10, 20],
            }
        )

        success_rates = compute_success_rates(df)

        assert success_rates == {}

    def test_compute_action_sequences_no_sequence_column(self):
        """Test action sequences without sequence column."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        sequences = compute_action_sequences(df)

        assert sequences == {}


class TestActionAnalysis:
    """Test action analysis functions."""

    def test_analyze_action_patterns(self, tmp_path):
        """Test action pattern analysis."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": [
                    "move",
                    "gather",
                    "attack",
                    "move",
                    "gather",
                    "move",
                    "attack",
                    "gather",
                    "move",
                    "reproduce",
                ],
                "frequency": [15, 8, 3, 12, 10, 18, 5, 7, 14, 2],
                "success_rate": [0.8, 0.75, 0.67, 0.83, 0.8, 0.83, 0.8, 0.71, 0.79, 0.5],
                "avg_reward": [1.2, 2.5, 0.8, 1.1, 2.8, 1.3, 0.9, 2.2, 1.0, 3.0],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_action_patterns(df, ctx)

        stats_file = tmp_path / "action_statistics.json"
        assert stats_file.exists()

        with open(stats_file) as f:
            data = json.load(f)

        assert "total_actions" in data
        assert "action_types" in data

    def test_analyze_success_rates(self, tmp_path):
        """Test success rate analysis."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather", "attack", "reproduce"] * 3,
                "success_count": [12, 8, 3, 1, 15, 10, 5, 2, 18, 12, 7, 3],
                "total_attempts": [15, 10, 5, 2, 18, 12, 8, 3, 20, 15, 10, 4],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_success_rates(df, ctx)

        rates_file = tmp_path / "success_rates.csv"
        assert rates_file.exists()

        rates_df = pd.read_csv(rates_file)
        assert len(rates_df) == 4  # one per action type
        assert "action_type" in rates_df.columns
        assert "success_rate" in rates_df.columns

    def test_analyze_action_sequences(self, tmp_path):
        """Test action sequence analysis."""
        df = pd.DataFrame(
            {
                "step": range(5),
                "action_sequence": [
                    ["move", "gather"],
                    ["gather", "attack"],
                    ["move", "gather", "attack"],
                    ["move"],
                    ["gather", "move"],
                ],
                "sequence_length": [2, 2, 3, 1, 2],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_action_sequences(df, ctx)

        sequences_file = tmp_path / "action_sequences.json"
        assert sequences_file.exists()

        with open(sequences_file) as f:
            data = json.load(f)

        assert "common_sequences" in data
        assert "avg_sequence_length" in data

    def test_analyze_sequence_patterns(self, tmp_path):
        """Test action sequence pattern analysis."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "seq_move_to_gather": [0.5, 0.6, 0.55, 0.7, 0.65, 0.6, 0.55, 0.7, 0.65, 0.6],
                "seq_gather_to_attack": [0.2, 0.3, 0.25, 0.35, 0.3, 0.25, 0.3, 0.35, 0.3, 0.25],
                "seq_attack_to_move": [0.8, 0.75, 0.8, 0.85, 0.8, 0.75, 0.8, 0.85, 0.8, 0.75],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_sequence_patterns(df, ctx)

        patterns_file = tmp_path / "sequence_patterns.json"
        assert patterns_file.exists()

        with open(patterns_file) as f:
            data = json.load(f)

        assert "move->gather" in data
        assert "most_common_sequence" in data

    def test_analyze_decision_patterns(self, tmp_path):
        """Test decision pattern analysis."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "action_type": ["move", "gather", "attack"] * 3,
                "success_rate": [0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.9, 0.8, 0.7],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_decision_patterns(df, ctx)

        decisions_file = tmp_path / "decision_patterns.json"
        assert decisions_file.exists()

        with open(decisions_file) as f:
            data = json.load(f)

        assert "avg_success_rate" in data
        assert "decision_consistency" in data

    def test_analyze_reward_analysis(self, tmp_path):
        """Test reward analysis."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 2, 2],
                "action_type": ["move", "gather", "move", "gather", "move", "gather"],
                "avg_reward": [1.0, 2.5, 1.2, 2.8, 1.1, 2.6],
                "reward_variance": [0.1, 0.2, 0.15, 0.25, 0.12, 0.22],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_reward_analysis(df, ctx)

        rewards_file = tmp_path / "reward_analysis.json"
        assert rewards_file.exists()

        with open(rewards_file) as f:
            data = json.load(f)

        assert "overall_avg_reward" in data
        assert "best_performing_action" in data


class TestActionVisualization:
    """Test action visualization functions."""

    def test_plot_action_distribution(self, tmp_path):
        """Test action distribution plotting."""
        df = pd.DataFrame(
            {
                "action_type": ["move"] * 50 + ["gather"] * 30 + ["attack"] * 15 + ["reproduce"] * 5,
                "frequency": [1] * 100,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_distribution(df, ctx)

        plot_file = tmp_path / "action_distribution.png"
        assert plot_file.exists()

    def test_plot_success_rates(self, tmp_path):
        """Test success rates plotting."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather", "attack", "reproduce"],
                "success_rate": [0.8, 0.75, 0.6, 0.5],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_success_rates(df, ctx)

        plot_file = tmp_path / "success_rates.png"
        assert plot_file.exists()

    def test_plot_action_sequences(self, tmp_path):
        """Test action sequence plotting."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "sequence_length": [2, 2, 3, 1, 2, 2, 2, 1, 2, 3],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_sequences(df, ctx)

        plot_file = tmp_path / "action_sequences.png"
        assert plot_file.exists()

    def test_plot_action_frequencies(self, tmp_path):
        """Test action frequencies plotting."""
        df = pd.DataFrame(
            {
                "step": list(range(10)) * 3,
                "action_type": ["move"] * 10 + ["gather"] * 10 + ["attack"] * 10,
                "frequency": [15, 12, 18, 14, 16, 13, 17, 15, 19, 14] * 3,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_frequencies(df, ctx)

        plot_file = tmp_path / "action_distribution.png"
        assert plot_file.exists()

    def test_plot_action_frequencies_empty(self, tmp_path):
        """Test action frequencies plotting with empty DataFrame."""
        df = pd.DataFrame()

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error, just log a warning
        plot_action_frequencies(df, ctx)

    def test_plot_sequence_patterns(self, tmp_path):
        """Test sequence patterns plotting."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "seq_move_to_gather": [0.5, 0.6, 0.55, 0.7, 0.65, 0.6, 0.55, 0.7, 0.65, 0.6],
                "seq_gather_to_attack": [0.2, 0.3, 0.25, 0.35, 0.3, 0.25, 0.3, 0.35, 0.3, 0.25],
                "seq_attack_to_move": [0.8, 0.75, 0.8, 0.85, 0.8, 0.75, 0.8, 0.85, 0.8, 0.75],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_sequence_patterns(df, ctx)

        plot_file = tmp_path / "sequence_patterns.png"
        assert plot_file.exists()

    def test_plot_sequence_patterns_no_data(self, tmp_path):
        """Test sequence patterns plotting without sequence columns."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error, just log a warning
        plot_sequence_patterns(df, ctx)

    def test_plot_decision_patterns(self, tmp_path):
        """Test decision patterns plotting."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "action_type": ["move", "gather", "attack"] * 3,
                "success_rate": [0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.9, 0.8, 0.7],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_decision_patterns(df, ctx)

        plot_file = tmp_path / "decision_patterns.png"
        assert plot_file.exists()

    def test_plot_decision_patterns_no_success_rate(self, tmp_path):
        """Test decision patterns plotting without success_rate."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error, just log a warning
        plot_decision_patterns(df, ctx)

    def test_plot_reward_distributions(self, tmp_path):
        """Test reward distributions plotting."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 2, 2],
                "action_type": ["move", "gather", "move", "gather", "move", "gather"],
                "avg_reward": [1.0, 2.5, 1.2, 2.8, 1.1, 2.6],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_reward_distributions(df, ctx)

        plot_file = tmp_path / "reward_distributions.png"
        assert plot_file.exists()

    def test_plot_reward_distributions_no_reward(self, tmp_path):
        """Test reward distributions plotting without avg_reward."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error, just log a warning
        plot_reward_distributions(df, ctx)

    def test_plot_action_distribution_no_frequency(self, tmp_path):
        """Test action distribution plotting without frequency column."""
        df = pd.DataFrame(
            {
                "action_type": ["move"] * 50 + ["gather"] * 30 + ["attack"] * 15 + ["reproduce"] * 5,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_distribution(df, ctx)

        plot_file = tmp_path / "action_distribution.png"
        assert plot_file.exists()

    def test_plot_success_rates_no_data(self, tmp_path):
        """Test success rates plotting with missing columns."""
        df = pd.DataFrame(
            {
                "action_type": ["move", "gather", "attack"],
                "frequency": [10, 20, 5],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        plot_success_rates(df, ctx)

        # Should still create file, but may be empty or show warning
        plot_file = tmp_path / "success_rates.png"
        assert plot_file.exists()


class TestActionsModule:
    """Test actions module integration."""

    def test_actions_module_registration(self):
        """Test module is properly registered."""
        assert actions_module.name == "actions"
        assert len(actions_module.get_function_names()) > 0
        assert "analyze_patterns" in actions_module.get_function_names()
        assert "plot_frequencies" in actions_module.get_function_names()

    def test_actions_module_function_groups(self):
        """Test module function groups."""
        groups = actions_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "basic" in groups
        assert "sequences" in groups
        assert "performance" in groups

    def test_actions_module_data_processor(self):
        """Test module data processor."""
        processor = actions_module.get_data_processor()
        assert processor is not None

    def test_actions_module_validator(self):
        """Test module validator."""
        validator = actions_module.get_validator()
        assert validator is not None

        # Test valid data
        valid_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "action_type": ["move", "gather", "attack"],
                "frequency": [10, 20, 15],
            }
        )
        validator.validate(valid_df)  # Should not raise

    def test_actions_module_validator_missing_columns(self):
        """Test module validator with missing columns."""
        validator = actions_module.get_validator()

        # Missing required columns
        invalid_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "frequency": [10, 20, 15],
            }
        )

        with pytest.raises(DataValidationError, match="Missing required columns"):
            validator.validate(invalid_df)

    def test_actions_module_validator_empty_data(self):
        """Test module validator with empty data."""
        validator = actions_module.get_validator()

        empty_df = pd.DataFrame()

        with pytest.raises(DataValidationError):
            validator.validate(empty_df)

    def test_actions_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        expected_functions = [
            "analyze_patterns",
            "analyze_sequences",
            "analyze_decisions",
            "analyze_rewards",
            "plot_frequencies",
            "plot_sequences",
            "plot_decisions",
            "plot_rewards",
        ]

        function_names = actions_module.get_function_names()

        for func in expected_functions:
            assert func in function_names

    def test_actions_module_function_groups_contents(self):
        """Test that function groups contain the right functions."""
        functions = actions_module.get_functions()
        groups = actions_module.get_function_groups()

        # Test basic group
        basic_group = groups["basic"]
        assert len(basic_group) == 2
        assert any(f.name == "analyze_action_patterns" for f in basic_group)
        assert any(f.name == "plot_action_frequencies" for f in basic_group)

        # Test sequences group
        sequences_group = groups["sequences"]
        assert len(sequences_group) == 2
        assert any(f.name == "analyze_sequence_patterns" for f in sequences_group)
        assert any(f.name == "plot_sequence_patterns" for f in sequences_group)

        # Test performance group
        performance_group = groups["performance"]
        assert len(performance_group) == 3
        assert any(f.name == "analyze_decision_patterns" for f in performance_group)
        assert any(f.name == "analyze_reward_analysis" for f in performance_group)
        assert any(f.name == "plot_reward_distributions" for f in performance_group)


class TestDataProcessing:
    """Test data processing functions."""

    @patch("farm.analysis.actions.data.SessionManager")
    @patch("farm.analysis.actions.data.ActionRepository")
    def test_process_action_data_from_database(self, mock_repo_class, mock_session_class, tmp_path):
        """Test processing action data from database."""
        # Create mock database file
        db_path = tmp_path / "simulation.db"
        db_path.touch()

        # Mock session manager and repository
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        # Mock action data
        mock_action = MagicMock()
        mock_action.agent_id = 1
        mock_action.action_type = "move"
        mock_action.step_number = 0
        mock_action.action_target_id = None
        mock_action.resources_before = 100
        mock_action.resources_after = 95
        mock_action.reward = 1.0
        mock_action.details = {}

        mock_repo.get_actions_by_scope.return_value = [mock_action] * 10

        # Process data
        df = process_action_data(tmp_path, use_database=True)

        # Verify result
        assert not df.empty
        assert "step" in df.columns
        assert "action_type" in df.columns
        assert "frequency" in df.columns

    def test_process_action_data_from_csv(self, tmp_path):
        """Test processing action data from CSV file."""
        # Create mock data directory and CSV
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "actions.csv"
        test_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "action_type": ["move", "gather", "attack"],
                "frequency": [10, 20, 15],
            }
        )
        test_df.to_csv(csv_path, index=False)

        # Process data
        df = process_action_data(tmp_path, use_database=False)

        # Verify result
        assert not df.empty
        assert len(df) == 3
        assert "step" in df.columns
        assert "action_type" in df.columns
        assert "frequency" in df.columns

    def test_process_action_data_file_not_found(self, tmp_path):
        """Test processing action data when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            process_action_data(tmp_path, use_database=False)

    @patch("farm.analysis.actions.data.find_database_path")
    def test_process_action_data_database_fallback_to_csv(self, mock_find_db, tmp_path):
        """Test fallback to CSV when database fails."""
        # Mock database path to raise exception
        mock_find_db.side_effect = FileNotFoundError("Database not found")

        # Create CSV fallback
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "actions.csv"
        test_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "action_type": ["move", "gather", "attack"],
                "frequency": [10, 20, 15],
            }
        )
        test_df.to_csv(csv_path, index=False)

        # Process data (should fallback to CSV)
        df = process_action_data(tmp_path, use_database=True)

        # Verify result
        assert not df.empty
        assert len(df) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_action_statistics_with_nan_values(self):
        """Test action statistics with NaN values."""
        df = pd.DataFrame(
            {
                "step": range(5),
                "action_type": ["move", "gather", "attack", "move", "gather"],
                "frequency": [15, 8, np.nan, 12, 10],
                "success_rate": [0.8, np.nan, 0.67, 0.83, 0.8],
                "avg_reward": [1.2, 2.5, 0.8, np.nan, 2.8],
            }
        )

        stats = compute_action_statistics(df)

        # Should handle NaN values gracefully
        assert "total_actions" in stats
        assert "action_types" in stats
        assert "most_common_action" in stats

    def test_compute_sequence_patterns_with_missing_values(self):
        """Test sequence patterns with missing values."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "seq_move_to_gather": [0.5, np.nan, 0.55, 0.7, np.nan, 0.6, 0.55, 0.7, 0.65, 0.6],
                "seq_gather_to_attack": [np.nan] * 10,  # All NaN
            }
        )

        patterns = compute_sequence_patterns(df)

        # Should handle NaN values by dropping them
        assert "move->gather" in patterns
        # All NaN column should not be included
        assert "gather->attack" not in patterns or patterns["gather->attack"]["avg_probability"] != patterns["gather->attack"]["avg_probability"]  # NaN check

    def test_analyze_action_patterns_with_minimal_data(self, tmp_path):
        """Test action pattern analysis with minimal data."""
        df = pd.DataFrame(
            {
                "step": [0],
                "action_type": ["move"],
                "frequency": [1],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error
        analyze_action_patterns(df, ctx)

        stats_file = tmp_path / "action_statistics.json"
        assert stats_file.exists()

    def test_plot_with_single_data_point(self, tmp_path):
        """Test plotting with single data point."""
        df = pd.DataFrame(
            {
                "step": [0],
                "action_type": ["move"],
                "frequency": [10],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise an error
        plot_action_frequencies(df, ctx)

        plot_file = tmp_path / "action_distribution.png"
        assert plot_file.exists()

    def test_compute_reward_metrics_with_negative_rewards(self):
        """Test reward metrics with negative rewards."""
        df = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "action_type": ["move", "attack", "move", "attack"],
                "avg_reward": [1.0, -2.0, 1.2, -1.8],
            }
        )

        metrics = compute_reward_metrics(df)

        assert "overall_avg_reward" in metrics
        # Average should be less than 1.0 due to negative rewards
        assert metrics["overall_avg_reward"] < 1.0
        # Best action should still be determined
        assert "best_performing_action" in metrics

    def test_analyze_with_progress_callback(self, tmp_path):
        """Test analysis with progress callback."""
        df = pd.DataFrame(
            {
                "step": range(10),
                "action_type": ["move"] * 10,
                "frequency": [5] * 10,
            }
        )

        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)
        analyze_action_patterns(df, ctx)

        # Should have called progress callback
        assert len(progress_calls) > 0
        assert any("complete" in msg.lower() for msg, _ in progress_calls)
