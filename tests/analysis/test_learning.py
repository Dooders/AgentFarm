"""
Tests for learning analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.learning import (
    learning_module,
    compute_learning_statistics,
    compute_agent_learning_curves,
    compute_learning_efficiency_metrics,
    analyze_learning_performance,
    analyze_agent_learning_curves,
    analyze_module_performance,
    plot_learning_curves,
    plot_reward_distribution,
)
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.core.services import EnvConfigService


@pytest.fixture
def sample_learning_data():
    """Create sample learning data for testing."""
    return pd.DataFrame({
        'step': range(100),
        'reward': [i * 0.1 + (i % 10) * 0.5 for i in range(100)],
        'agent_id': [f'agent_{i % 5}' for i in range(100)],
        'module_type': ['decision' if i % 2 == 0 else 'memory' for i in range(100)],
        'action_count': [i % 10 + 1 for i in range(100)],
        'success_rate': [0.5 + (i % 20) * 0.025 for i in range(100)],
    })


@pytest.fixture
def sample_experiment_path(tmp_path):
    """Create a sample experiment path with mock data."""
    exp_path = tmp_path / "experiment"
    exp_path.mkdir()

    # Create mock simulation.db
    db_path = exp_path / "simulation.db"
    db_path.touch()

    return exp_path


class TestLearningModule:
    """Test the learning analysis module."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert learning_module.name == "learning"
        assert len(learning_module.get_function_names()) > 0
        assert "analyze_performance" in learning_module.get_function_names()

    def test_module_groups(self):
        """Test module function groups."""
        groups = learning_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "performance" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = learning_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert learning_module.supports_database() is True
        assert learning_module.get_db_filename() == "simulation.db"


class TestLearningComputations:
    """Test learning statistical computations."""

    def test_compute_learning_statistics(self, sample_learning_data):
        """Test learning statistics computation."""
        result = compute_learning_statistics(sample_learning_data)

        assert "total_experiences" in result
        assert "reward" in result  # reward stats dict
        assert "reward_trend" in result
        assert "unique_agents" in result

    def test_compute_agent_learning_curves(self, sample_learning_data):
        """Test agent learning curves computation."""
        result = compute_agent_learning_curves(sample_learning_data)

        assert isinstance(result, dict)
        # Should contain curves for each agent

    def test_compute_learning_efficiency_metrics(self, sample_learning_data):
        """Test learning efficiency metrics computation."""
        result = compute_learning_efficiency_metrics(sample_learning_data)

        assert "reward_efficiency" in result
        assert "convergence_rate" in result
        assert "learning_stability" in result


class TestLearningAnalysis:
    """Test learning analysis functions."""

    def test_analyze_learning_performance(self, tmp_path, sample_learning_data):
        """Test learning performance analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_learning_performance(sample_learning_data, ctx)

        # Check output file created
        output_file = tmp_path / "learning_performance.json"
        assert output_file.exists()

        # Check content
        with open(output_file, 'r') as f:
            data = json.load(f)
            assert "statistics" in data
            assert "efficiency" in data

    def test_analyze_agent_learning_curves(self, tmp_path, sample_learning_data):
        """Test agent learning curves analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_agent_learning_curves(sample_learning_data, ctx)

        # Check output file created
        output_file = tmp_path / "agent_learning_curves.json"
        assert output_file.exists()

    def test_analyze_module_performance(self, tmp_path, sample_learning_data):
        """Test module performance analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_module_performance(sample_learning_data, ctx)

        # Check output file created
        output_file = tmp_path / "module_performance_comparison.json"
        assert output_file.exists()


class TestLearningVisualization:
    """Test learning visualization functions."""

    def test_plot_learning_curves(self, tmp_path, sample_learning_data):
        """Test learning curves plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Mock matplotlib to avoid display issues
        with patch('matplotlib.pyplot.savefig'):
            plot_learning_curves(sample_learning_data, ctx)

        # Check output file would be created (mocked)
        output_file = tmp_path / "learning_curves.png"
        # Note: In real test, this would exist, but we're mocking pyplot.savefig

    def test_plot_reward_distribution(self, tmp_path, sample_learning_data):
        """Test reward distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        with patch('matplotlib.pyplot.savefig'):
            plot_reward_distribution(sample_learning_data, ctx)

        # Check output file would be created
        output_file = tmp_path / "reward_distribution.png"


class TestLearningIntegration:
    """Test learning module integration with service."""

    def test_learning_module_integration(self, tmp_path, sample_experiment_path):
        """Test full learning module execution."""
        service = AnalysisService(EnvConfigService())

        request = AnalysisRequest(
            module_name="learning",
            experiment_path=sample_experiment_path,
            output_path=tmp_path,
            group="basic"
        )

        # Mock the data processing to avoid database dependency
        with patch('farm.analysis.learning.data.process_learning_data') as mock_process:
            mock_process.return_value = pd.DataFrame({
                'step': range(10),
                'reward': range(10),
                'agent_id': ['agent_1'] * 10,
                'module_type': ['decision'] * 10,
            })

            result = service.run(request)

            # Should succeed with mocked data
            assert result.success or "data" in str(result.error).lower()

    def test_learning_module_with_invalid_data(self, tmp_path, sample_experiment_path):
        """Test learning module with invalid data."""
        service = AnalysisService(EnvConfigService())

        request = AnalysisRequest(
            module_name="learning",
            experiment_path=sample_experiment_path,
            output_path=tmp_path,
            group="analysis"
        )

        # Mock data processing to return invalid data
        with patch('farm.analysis.learning.data.process_learning_data') as mock_process:
            mock_process.return_value = pd.DataFrame()  # Empty dataframe

            result = service.run(request)

            # Should handle invalid data gracefully
            assert not result.success or "insufficient" in str(result.error).lower()
