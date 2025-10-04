"""
Tests for genesis analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.analysis.genesis import (
    genesis_module,
    analyze_genesis_patterns,
    plot_genesis_patterns,
    plot_genesis_timeline,
    plot_initial_state_comparison,
    plot_critical_period_analysis,
    plot_genesis_analysis_results,
)
from farm.analysis.common.context import AnalysisContext


class TestGenesisModule:
    """Test genesis module functionality."""

    def test_genesis_module_registration(self):
        """Test module registration."""
        assert genesis_module.name == "genesis"
        assert genesis_module.description == "Analysis of initial conditions and their impact on dominance patterns and simulation outcomes"
        assert not genesis_module._registered

    def test_genesis_module_function_names(self):
        """Test module function names."""
        functions = genesis_module.get_function_names()
        expected_functions = [
            "analyze_factors",
            "analyze_across_simulations",
            "analyze_critical_period",
            "analyze_genesis_patterns",
            "plot_results",
            "plot_initial_comparison",
            "plot_critical_period",
            "plot_genesis_patterns",
            "plot_genesis_timeline",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_genesis_module_function_groups(self):
        """Test module function groups."""
        groups = genesis_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_genesis_module_data_processor(self):
        """Test module data processor."""
        processor = genesis_module.get_data_processor()
        assert processor is not None


class TestGenesisAnalysis:
    """Test genesis analysis functions."""

    @pytest.fixture
    def sample_genesis_data(self):
        """Create sample genesis data for testing."""
        return pd.DataFrame({
            'iteration': range(20),
            'agent_id': [f'agent_{i}' for i in range(20)],
            'genesis_time': np.random.uniform(0, 1000, 20),
            'parent_id': [f'parent_{i%5}' for i in range(20)],
            'success_rate': np.random.uniform(0, 1, 20),
            'efficiency_score': np.random.uniform(0, 1, 20),
            'resource_cost': np.random.uniform(10, 100, 20),
            'survival_time': np.random.uniform(100, 1000, 20),
        })

    def test_analyze_genesis_patterns(self, sample_genesis_data):
        """Test genesis pattern analysis."""
        result = analyze_genesis_patterns(sample_genesis_data)

        assert isinstance(result, dict)
        assert 'total_genesis_events' in result
        assert 'avg_success_rate' in result
        assert 'avg_efficiency' in result


class TestGenesisVisualization:
    """Test genesis visualization functions."""

    @pytest.fixture
    def sample_genesis_data(self):
        """Create sample genesis data for testing."""
        return pd.DataFrame({
            'iteration': range(15),
            'agent_id': [f'agent_{i}' for i in range(15)],
            'genesis_time': np.random.uniform(0, 800, 15),
            'success_rate': np.random.uniform(0, 1, 15),
            'efficiency_score': np.random.uniform(0, 1, 15),
            'resource_cost': np.random.uniform(10, 80, 15),
        })

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "genesis_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def analysis_context(self, temp_output_dir):
        """Create analysis context for testing."""
        return AnalysisContext(
            output_path=temp_output_dir,
            config={'test_mode': True},
            metadata={'test': 'genesis'}
        )

    @patch('farm.analysis.genesis.plot.plt')
    def test_plot_genesis_patterns(self, mock_plt, sample_genesis_data, analysis_context):
        """Test genesis patterns plotting."""
        result = plot_genesis_patterns(sample_genesis_data, analysis_context)
        assert result is None  # Plot functions return None
        mock_plt.savefig.assert_called_once()

    @patch('farm.analysis.genesis.plot.plt')
    def test_plot_genesis_timeline(self, mock_plt, sample_genesis_data, analysis_context):
        """Test genesis timeline plotting."""
        result = plot_genesis_timeline(sample_genesis_data, analysis_context)
        assert result is None
        mock_plt.savefig.assert_called_once()
