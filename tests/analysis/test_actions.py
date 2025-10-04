"""
Tests for actions analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from farm.analysis.actions import (
    actions_module,
    compute_action_statistics,
    compute_success_rates,
    compute_action_sequences,
    analyze_action_patterns,
    analyze_success_rates,
    analyze_action_sequences,
    plot_action_distribution,
    plot_success_rates,
    plot_action_sequences,
)
from farm.analysis.common.context import AnalysisContext


class TestActionComputations:
    """Test action statistical computations."""

    def test_compute_action_statistics(self):
        """Test action statistics computation."""
        df = pd.DataFrame({
            'step': range(10),
            'action_type': ['move', 'gather', 'attack', 'move', 'gather', 'move', 'attack', 'gather', 'move', 'reproduce'],
            'frequency': [15, 8, 3, 12, 10, 18, 5, 7, 14, 2],
            'success_count': [12, 6, 2, 10, 8, 15, 4, 5, 11, 1],
            'reward': [1.2, 2.5, 0.8, 1.1, 2.8, 1.3, 0.9, 2.2, 1.0, 3.0],
        })

        stats = compute_action_statistics(df)

        assert 'action_counts' in stats
        assert 'total_actions' in stats
        assert 'unique_actions' in stats
        assert 'most_common' in stats

        assert stats['total_actions'] == 84  # sum of frequencies
        assert stats['unique_actions'] == 4   # move, gather, attack, reproduce
        assert stats['most_common'] == 'move'  # appears most

    def test_compute_success_rates(self):
        """Test success rate computation."""
        df = pd.DataFrame({
            'action_type': ['move', 'gather', 'attack', 'reproduce'] * 5,
            'success_count': [12, 8, 3, 1, 15, 10, 5, 2, 18, 12, 7, 3, 20, 15, 8, 4, 22, 18, 10, 5],
            'total_attempts': [15, 10, 5, 2, 18, 12, 8, 3, 20, 15, 10, 4, 25, 18, 12, 6, 28, 20, 15, 8],
        })

        success_rates = compute_success_rates(df)

        assert len(success_rates) == 4  # one per action type

        for action_type, rate in success_rates.items():
            assert 0 <= rate <= 1
            assert isinstance(rate, float)

        # Check specific calculations
        assert success_rates['move'] == 12/15  # first move entry
        assert success_rates['gather'] == 8/10  # first gather entry

    def test_compute_action_sequences(self):
        """Test action sequence computation."""
        df = pd.DataFrame({
            'step': range(10),
            'action_sequence': [
                ['move', 'gather'],
                ['gather', 'attack'],
                ['move', 'gather', 'attack'],
                ['move'],
                ['gather', 'move'],
                ['attack', 'move'],
                ['move', 'gather'],
                ['gather'],
                ['move', 'attack'],
                ['gather', 'move', 'attack']
            ],
            'sequence_length': [2, 2, 3, 1, 2, 2, 2, 1, 2, 3],
        })

        sequences = compute_action_sequences(df)

        assert 'common_sequences' in sequences
        assert 'avg_sequence_length' in sequences
        assert 'max_sequence_length' in sequences
        assert 'transition_matrix' in sequences

        assert sequences['avg_sequence_length'] == 2.0  # average of lengths
        assert sequences['max_sequence_length'] == 3
        assert isinstance(sequences['transition_matrix'], dict)


class TestActionAnalysis:
    """Test action analysis functions."""

    def test_analyze_action_patterns(self, tmp_path):
        """Test action pattern analysis."""
        df = pd.DataFrame({
            'step': range(10),
            'action_type': ['move', 'gather', 'attack', 'move', 'gather', 'move', 'attack', 'gather', 'move', 'reproduce'],
            'frequency': [15, 8, 3, 12, 10, 18, 5, 7, 14, 2],
            'success_count': [12, 6, 2, 10, 8, 15, 4, 5, 11, 1],
            'reward': [1.2, 2.5, 0.8, 1.1, 2.8, 1.3, 0.9, 2.2, 1.0, 3.0],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_action_patterns(df, ctx)

        stats_file = tmp_path / "action_patterns.json"
        assert stats_file.exists()

        with open(stats_file) as f:
            data = json.load(f)

        assert 'statistics' in data
        assert 'patterns' in data

    def test_analyze_success_rates(self, tmp_path):
        """Test success rate analysis."""
        df = pd.DataFrame({
            'action_type': ['move', 'gather', 'attack', 'reproduce'] * 3,
            'success_count': [12, 8, 3, 1, 15, 10, 5, 2, 18, 12, 7, 3],
            'total_attempts': [15, 10, 5, 2, 18, 12, 8, 3, 20, 15, 10, 4],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_success_rates(df, ctx)

        rates_file = tmp_path / "success_rates.csv"
        assert rates_file.exists()

        rates_df = pd.read_csv(rates_file)
        assert len(rates_df) == 4  # one per action type
        assert 'action_type' in rates_df.columns
        assert 'success_rate' in rates_df.columns

    def test_analyze_action_sequences(self, tmp_path):
        """Test action sequence analysis."""
        df = pd.DataFrame({
            'step': range(5),
            'action_sequence': [
                ['move', 'gather'],
                ['gather', 'attack'],
                ['move', 'gather', 'attack'],
                ['move'],
                ['gather', 'move']
            ],
            'sequence_length': [2, 2, 3, 1, 2],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_action_sequences(df, ctx)

        sequences_file = tmp_path / "action_sequences.json"
        assert sequences_file.exists()

        with open(sequences_file) as f:
            data = json.load(f)

        assert 'sequences' in data
        assert 'transitions' in data


class TestActionVisualization:
    """Test action visualization functions."""

    def test_plot_action_distribution(self, tmp_path):
        """Test action distribution plotting."""
        df = pd.DataFrame({
            'action_type': ['move'] * 50 + ['gather'] * 30 + ['attack'] * 15 + ['reproduce'] * 5,
            'frequency': [1] * 100,
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_distribution(df, ctx)

        plot_file = tmp_path / "action_distribution.png"
        assert plot_file.exists()

    def test_plot_success_rates(self, tmp_path):
        """Test success rates plotting."""
        df = pd.DataFrame({
            'action_type': ['move', 'gather', 'attack', 'reproduce'],
            'success_rate': [0.8, 0.75, 0.6, 0.5],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_success_rates(df, ctx)

        plot_file = tmp_path / "success_rates.png"
        assert plot_file.exists()

    def test_plot_action_sequences(self, tmp_path):
        """Test action sequence plotting."""
        df = pd.DataFrame({
            'step': range(10),
            'sequence_length': [2, 2, 3, 1, 2, 2, 2, 1, 2, 3],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_action_sequences(df, ctx)

        plot_file = tmp_path / "action_sequences.png"
        assert plot_file.exists()


class TestActionsModule:
    """Test actions module integration."""

    def test_actions_module_registration(self):
        """Test module is properly registered."""
        assert actions_module.name == "actions"
        assert len(actions_module.get_function_names()) > 0
        assert "analyze_patterns" in actions_module.get_function_names()
        assert "plot_distribution" in actions_module.get_function_names()

    def test_actions_module_function_groups(self):
        """Test module function groups."""
        groups = actions_module.get_group_names()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_actions_module_data_processor(self):
        """Test module data processor."""
        processor = actions_module.get_data_processor()
        assert processor is not None

        # Test with mock data
        mock_data = pd.DataFrame({
            'step': range(5),
            'action_type': ['move', 'gather', 'attack', 'move', 'gather'],
        })

        result = processor.process(mock_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
