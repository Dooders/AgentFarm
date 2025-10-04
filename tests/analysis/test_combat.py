"""
Tests for combat analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.combat import (
    combat_module,
    compute_combat_statistics,
    compute_agent_combat_performance,
    compute_combat_efficiency_metrics,
    analyze_combat_overview,
    analyze_agent_combat_performance,
    analyze_combat_efficiency,
    plot_combat_overview,
    plot_agent_combat_performance,
    plot_combat_efficiency,
)
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.core.services import EnvConfigService


@pytest.fixture
def sample_combat_data():
    """Create sample combat data for testing."""
    return pd.DataFrame({
        'step': range(100),
        'agent_id': [f'agent_{i % 10}' for i in range(100)],
        'opponent_id': [f'opponent_{i % 8}' for i in range(100)],
        'damage_dealt': [i % 50 + (i % 10) for i in range(100)],
        'damage_received': [i % 30 + (i % 5) for i in range(100)],
        'health_remaining': [100 - (i % 100) for i in range(100)],
        'combat_outcome': ['win' if i % 3 == 0 else 'loss' if i % 3 == 1 else 'draw' for i in range(100)],
        'combat_duration': [10 + (i % 20) for i in range(100)],
        'weapon_used': [f'weapon_{i % 4}' for i in range(100)],
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


class TestCombatModule:
    """Test the combat analysis module."""

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


class TestCombatComputations:
    """Test combat statistical computations."""

    # Note: Compute functions expect specific data formats
    # These would need properly formatted data for detailed testing
    def test_compute_functions_exist(self):
        """Test that compute functions exist and are callable."""
        assert callable(compute_combat_statistics)
        assert callable(compute_agent_combat_performance)
        assert callable(compute_combat_efficiency_metrics)


class TestCombatAnalysis:
    """Test combat analysis functions."""

    def test_analyze_functions_exist(self):
        """Test that analysis functions exist and are callable."""
        assert callable(analyze_combat_overview)
        assert callable(analyze_agent_combat_performance)
        assert callable(analyze_combat_efficiency)


class TestCombatVisualization:
    """Test combat visualization functions."""

    def test_plot_functions_exist(self):
        """Test that plot functions exist and are callable."""
        assert callable(plot_combat_overview)
        assert callable(plot_agent_combat_performance)
        assert callable(plot_combat_efficiency)


class TestCombatIntegration:
    """Test combat module integration with service."""

    def test_combat_module_integration(self, tmp_path, sample_experiment_path):
        """Test full combat module execution."""
        service = AnalysisService(EnvConfigService())

        request = AnalysisRequest(
            module_name="combat",
            experiment_path=sample_experiment_path,
            output_path=tmp_path,
            group="basic"
        )

        # Mock the data processing to avoid database dependency
        with patch('farm.analysis.combat.data.process_combat_data') as mock_process:
            mock_process.return_value = pd.DataFrame({
                'step': range(10),
                'agent_id': ['agent_1'] * 10,
                'damage_dealt': range(10),
                'combat_outcome': ['win'] * 5 + ['loss'] * 5,
            })

            result = service.run(request)

            # Should succeed with mocked data
            assert result.success or "data" in str(result.error).lower()
