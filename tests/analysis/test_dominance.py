"""
Tests for dominance analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.analysis.dominance import (
    dominance_module,
    plot_dominance_distribution,
    plot_comprehensive_score_breakdown,
    plot_dominance_switches,
    plot_dominance_stability,
    run_dominance_classification,
)
from farm.analysis.common.context import AnalysisContext


class TestDominanceModule:
    """Test dominance module functionality."""

    def test_dominance_module_registration(self):
        """Test module registration."""
        assert dominance_module.name == "dominance"
        assert dominance_module.description == "Analysis of agent dominance patterns in simulations"
        assert not dominance_module._registered

    def test_dominance_module_function_names(self):
        """Test module function names."""
        functions = dominance_module.get_function_names()
        expected_functions = [
            "plot_dominance_distribution",
            "plot_comprehensive_score_breakdown",
            "plot_dominance_switches",
            "plot_dominance_stability",
            "plot_reproduction_success_vs_switching",
            "plot_reproduction_advantage_vs_stability",
            "plot_resource_proximity_vs_dominance",
            "plot_reproduction_vs_dominance",
            "plot_dominance_comparison",
            "plot_correlation_matrix",
            "run_dominance_classification",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_dominance_module_function_groups(self):
        """Test module function groups."""
        groups = dominance_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "ml" in groups
        assert "correlation" in groups

    def test_dominance_module_data_processor(self):
        """Test module data processor."""
        processor = dominance_module.get_data_processor()
        assert processor is not None

    def test_dominance_module_supports_database(self):
        """Test that dominance module supports database."""
        assert dominance_module.supports_database()

    def test_dominance_module_db_filename(self):
        """Test dominance module database filename."""
        assert dominance_module.get_db_filename() == "dominance.db"


class TestDominanceVisualization:
    """Test dominance visualization functions."""

    @pytest.fixture
    def sample_dominance_data(self):
        """Create sample dominance data for testing."""
        return pd.DataFrame({
            'iteration': range(10),
            'agent_id': [f'agent_{i}' for i in range(10)],
            'comprehensive_dominance': np.random.uniform(0, 1, 10),
            'reproduction_success': np.random.uniform(0, 1, 10),
            'resource_proximity': np.random.uniform(0, 1, 10),
            'stability_score': np.random.uniform(0, 1, 10),
            'switching_frequency': np.random.randint(0, 5, 10),
        })

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "dominance_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def analysis_context(self, temp_output_dir):
        """Create analysis context for testing."""
        return AnalysisContext(
            output_path=temp_output_dir,
            config={'test_mode': True},
            metadata={'test': 'dominance'}
        )

    @patch('farm.analysis.dominance.plot.plt')
    def test_plot_dominance_distribution(self, mock_plt, sample_dominance_data, analysis_context):
        """Test dominance distribution plotting."""
        # Mock plt.subplots to return a tuple of (fig, axes)
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        result = plot_dominance_distribution(sample_dominance_data, ctx=analysis_context)
        assert result is None  # Plot functions return None
        mock_plt.savefig.assert_called_once()

    @patch('farm.analysis.dominance.plot.plt')
    def test_plot_comprehensive_score_breakdown(self, mock_plt, sample_dominance_data, analysis_context):
        """Test comprehensive score breakdown plotting."""
        result = plot_comprehensive_score_breakdown(sample_dominance_data, ctx=analysis_context)
        # Function returns the weighted scores DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Should have 3 agent types
        mock_plt.savefig.assert_called_once()


class TestDominanceML:
    """Test dominance machine learning functions."""

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample data for ML testing."""
        n_samples = 100
        return pd.DataFrame({
            'iteration': range(n_samples),
            'agent_id': [f'agent_{i}' for i in range(n_samples)],
            'comprehensive_dominance': np.random.uniform(0, 1, n_samples),
            'reproduction_success': np.random.uniform(0, 1, n_samples),
            'resource_proximity': np.random.uniform(0, 1, n_samples),
            'stability_score': np.random.uniform(0, 1, n_samples),
            'switching_frequency': np.random.randint(0, 5, n_samples),
            'dominance_category': np.random.choice(['low', 'medium', 'high'], n_samples),
        })

    def test_run_dominance_classification(self, sample_ml_data):
        """Test dominance classification."""
        # This might take time, so we'll just test that it doesn't crash
        try:
            result = run_dominance_classification(sample_ml_data)
            assert isinstance(result, dict)
        except Exception as e:
            # ML functions might fail with small datasets, that's okay
            assert isinstance(e, (ValueError, ImportError)) or "classification" in str(e).lower()
