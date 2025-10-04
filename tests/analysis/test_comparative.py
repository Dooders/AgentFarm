"""
Tests for comparative analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.analysis.comparative import (
    comparative_module,
    plot_comparison_metrics,
    plot_performance_comparison,
    plot_simulation_comparison,
    compare_simulations,
    compare_experiments,
    plot_comparative_analysis,
)
from farm.analysis.common.context import AnalysisContext


class TestComparativeModule:
    """Test comparative module functionality."""

    def test_comparative_module_registration(self):
        """Test module registration."""
        assert comparative_module.name == "comparative"
        assert comparative_module.description == "Analysis for comparing multiple simulations and their differences"
        assert not comparative_module._registered

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


class TestComparativeAnalysis:
    """Test comparative analysis functions."""

    @pytest.fixture
    def sample_comparative_data(self):
        """Create sample comparative data for testing."""
        experiments = ['exp_1', 'exp_2', 'exp_3']
        data = []

        for exp in experiments:
            for i in range(10):
                data.append({
                    'experiment': exp,
                    'iteration': i,
                    'performance_metric': np.random.uniform(0, 1),
                    'efficiency_score': np.random.uniform(0, 1),
                    'stability_metric': np.random.uniform(0, 1),
                    'diversity_index': np.random.uniform(0, 1),
                })

        return pd.DataFrame(data)

    def test_compare_experiments(self, sample_comparative_data):
        """Test experiment comparison."""
        result = compare_experiments(sample_comparative_data)

        assert isinstance(result, dict)
        assert 'experiment_count' in result
        assert 'performance_comparison' in result
        assert 'statistical_tests' in result


class TestComparativeVisualization:
    """Test comparative visualization functions."""

    @pytest.fixture
    def sample_comparative_data(self):
        """Create sample comparative data for testing."""
        return pd.DataFrame({
            'experiment': ['A', 'A', 'B', 'B', 'C', 'C'] * 5,
            'iteration': list(range(10)) * 3,
            'performance_metric': np.random.uniform(0, 1, 30),
            'efficiency_score': np.random.uniform(0, 1, 30),
        })

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "comparative_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def analysis_context(self, temp_output_dir):
        """Create analysis context for testing."""
        return AnalysisContext(
            output_path=temp_output_dir,
            config={'test_mode': True},
            metadata={'test': 'comparative'}
        )

    @patch('farm.analysis.comparative.plot.plt')
    def test_plot_comparative_analysis(self, mock_plt, sample_comparative_data, analysis_context):
        """Test comparative analysis plotting."""
        result = plot_comparative_analysis(sample_comparative_data, analysis_context)
        assert result is None  # Plot functions return None
        mock_plt.savefig.assert_called_once()

    @patch('farm.analysis.comparative.plot.plt')
    def test_plot_performance_comparison(self, mock_plt, sample_comparative_data, analysis_context):
        """Test performance comparison plotting."""
        result = plot_performance_comparison(sample_comparative_data, analysis_context)
        assert result is None
        mock_plt.savefig.assert_called_once()
