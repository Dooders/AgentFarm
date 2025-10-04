"""
Tests for social behavior analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.analysis.social_behavior import (
    social_behavior_module,
    plot_social_network_overview,
    plot_cooperation_competition_balance,
    plot_resource_sharing_patterns,
)
from farm.analysis.common.context import AnalysisContext


class TestSocialBehaviorModule:
    """Test social behavior module functionality."""

    def test_social_behavior_module_registration(self):
        """Test module registration."""
        assert social_behavior_module.name == "social_behavior"
        assert social_behavior_module.description == "Analysis of social behaviors including cooperation, competition, networks, and group dynamics"
        assert not social_behavior_module._registered

    def test_social_behavior_module_function_names(self):
        """Test module function names."""
        functions = social_behavior_module.get_function_names()
        expected_functions = [
            "analyze_comprehensive",
            "plot_network_overview",
            "plot_cooperation_balance",
            "plot_sharing_patterns",
            "plot_clustering",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_social_behavior_module_function_groups(self):
        """Test module function groups."""
        groups = social_behavior_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_social_behavior_module_data_processor(self):
        """Test module data processor."""
        processor = social_behavior_module.get_data_processor()
        assert processor is not None


class TestSocialBehaviorAnalysis:
    """Test social behavior analysis functions."""

    @pytest.fixture
    def sample_social_data(self):
        """Create sample social behavior data for testing."""
        return pd.DataFrame({
            'iteration': range(30),
            'agent_id': [f'agent_{i}' for i in range(30)],
            'interaction_count': np.random.randint(0, 20, 30),
            'cooperation_score': np.random.uniform(0, 1, 30),
            'conflict_score': np.random.uniform(0, 1, 30),
            'social_network_density': np.random.uniform(0, 1, 30),
            'cluster_coefficient': np.random.uniform(0, 1, 30),
            'reciprocity_score': np.random.uniform(0, 1, 30),
        })

    def test_analyze_social_patterns(self, sample_social_data):
        """Test social pattern analysis."""
        # Note: analyze_social_behaviors requires a database session, so we'll skip this test
        # The function is designed to work with database data, not DataFrame inputs
        pytest.skip("analyze_social_behaviors requires database session, not DataFrame input")


class TestSocialBehaviorVisualization:
    """Test social behavior visualization functions."""

    @pytest.fixture
    def sample_social_data(self):
        """Create sample social behavior data for testing."""
        return pd.DataFrame({
            'iteration': range(20),
            'agent_id': [f'agent_{i}' for i in range(20)],
            'interaction_count': np.random.randint(0, 15, 20),
            'cooperation_score': np.random.uniform(0, 1, 20),
            'conflict_score': np.random.uniform(0, 1, 20),
        })

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "social_output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def analysis_context(self, temp_output_dir):
        """Create analysis context for testing."""
        return AnalysisContext(
            output_path=temp_output_dir,
            config={'test_mode': True},
            metadata={'test': 'social_behavior'}
        )

    @patch('farm.analysis.social_behavior.plot.plt')
    def test_plot_social_network_overview(self, mock_plt, sample_social_data, analysis_context):
        """Test social network overview plotting."""
        result = plot_social_network_overview(sample_social_data, analysis_context)
        assert result is None  # Plot functions return None
        mock_plt.savefig.assert_called_once()

    @patch('farm.analysis.social_behavior.plot.plt')
    def test_plot_cooperation_competition_balance(self, mock_plt, sample_social_data, analysis_context):
        """Test cooperation competition balance plotting."""
        result = plot_cooperation_competition_balance(sample_social_data, analysis_context)
        assert result is None
        mock_plt.savefig.assert_called_once()
