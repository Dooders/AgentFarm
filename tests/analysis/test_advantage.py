"""
Tests for advantage analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.analysis.advantage import (
    advantage_module,
    plot_advantage_results,
    plot_advantage_correlation_matrix,
    plot_advantage_distribution,
    plot_advantage_timeline,
    analyze_advantage_patterns,
)
from farm.analysis.common.context import AnalysisContext


class TestAdvantageModule:
    """Test advantage module functionality."""

    def test_advantage_module_registration(self):
        """Test module registration."""
        assert advantage_module.name == "advantage"
        assert advantage_module.description == "Analysis of relative advantages between agent types and their impact on dominance patterns"
        assert not advantage_module._registered

    def test_advantage_module_function_names(self):
        """Test module function names."""
        functions = advantage_module.get_function_names()
        expected_functions = [
            "plot_advantage_distribution",
            "plot_advantage_timeline",
            "plot_advantage_correlations",
            "plot_advantage_evolution",
            "plot_advantage_comparison",
            "plot_advantage_optimization",
            "analyze_advantage_patterns",
            "analyze_advantage_evolution",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_advantage_module_function_groups(self):
        """Test module function groups."""
        groups = advantage_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_advantage_module_data_processor(self):
        """Test module data processor."""
        processor = advantage_module.get_data_processor()
        assert processor is not None


class TestAdvantageAnalysis:
    """Test advantage analysis functions."""

    @pytest.fixture
    def sample_advantage_data(self):
        """Create sample advantage data for testing."""
        return pd.DataFrame({
            'iteration': range(25),
            'agent_id': [f'agent_{i}' for i in range(25)],
            'advantage_score': np.random.uniform(0, 1, 25),
            'relative_advantage': np.random.uniform(-0.5, 0.5, 25),
            'evolutionary_fitness': np.random.uniform(0, 1, 25),
            'resource_advantage': np.random.uniform(0, 1, 25),
            'survival_advantage': np.random.uniform(0, 1, 25),
            'reproduction_advantage': np.random.uniform(0, 1, 25),
        })

    def test_analyze_advantage_patterns(self, sample_advantage_data):
        """Test advantage pattern analysis."""
        result = analyze_advantage_patterns(sample_advantage_data)

        assert isinstance(result, dict)
        assert 'avg_advantage_score' in result
        assert 'advantage_distribution' in result
        assert 'correlation_matrix' in result


class TestAdvantageVisualization:
    """Test advantage visualization functions."""

    @pytest.fixture
    def sample_advantage_data(self):
        """Create sample advantage data for testing."""
        return pd.DataFrame({
            'iteration': range(20),
            'agent_id': [f'agent_{i}' for i in range(20)],
            'advantage_score': np.random.uniform(0, 1, 20),
            'relative_advantage': np.random.uniform(-0.5, 0.5, 20),
            'evolutionary_fitness': np.random.uniform(0, 1, 20),
        })

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "advantage_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def analysis_context(self, temp_output_dir):
        """Create analysis context for testing."""
        return AnalysisContext(
            output_path=temp_output_dir,
            config={'test_mode': True},
            metadata={'test': 'advantage'}
        )

    @patch('farm.analysis.advantage.plot.plt')
    def test_plot_advantage_distribution(self, mock_plt, sample_advantage_data, analysis_context):
        """Test advantage distribution plotting."""
        result = plot_advantage_distribution(sample_advantage_data, analysis_context)
        assert result is None  # Plot functions return None
        mock_plt.savefig.assert_called_once()

    @patch('farm.analysis.advantage.plot.plt')
    def test_plot_advantage_timeline(self, mock_plt, sample_advantage_data, analysis_context):
        """Test advantage timeline plotting."""
        result = plot_advantage_timeline(sample_advantage_data, analysis_context)
        assert result is None
        mock_plt.savefig.assert_called_once()
