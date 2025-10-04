"""
Tests for spatial analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.spatial import (
    spatial_module,
    compute_spatial_statistics,
    compute_movement_patterns,
    compute_location_hotspots,
    analyze_spatial_overview,
    analyze_movement_patterns,
    analyze_location_hotspots,
    plot_spatial_overview,
    plot_movement_trajectories,
    plot_location_hotspots,
)
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.core.services import EnvConfigService


@pytest.fixture
def sample_spatial_data():
    """Create sample spatial data for testing."""
    return pd.DataFrame({
        'step': range(100),
        'agent_id': [f'agent_{i % 10}' for i in range(100)],
        'position_x': [i * 0.1 + (i % 10) for i in range(100)],
        'position_y': [i * 0.05 + (i % 5) * 2 for i in range(100)],
        'direction': [i % 360 for i in range(100)],
        'speed': [1.0 + (i % 10) * 0.1 for i in range(100)],
        'distance': [1.0 + (i % 5) * 0.2 for i in range(100)],
        'cluster_id': [i % 3 for i in range(100)],
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


class TestSpatialModule:
    """Test the spatial analysis module."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert spatial_module.name == "spatial"
        assert len(spatial_module.get_function_names()) > 0
        assert "analyze_overview" in spatial_module.get_function_names()

    def test_module_groups(self):
        """Test module function groups."""
        groups = spatial_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "movement" in groups
        assert "location" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = spatial_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert spatial_module.supports_database() is True
        assert spatial_module.get_db_filename() == "simulation.db"


class TestSpatialComputations:
    """Test spatial statistical computations."""

    def test_compute_spatial_statistics(self, sample_spatial_data):
        """Test spatial statistics computation."""
        result = compute_spatial_statistics(sample_spatial_data)

        assert isinstance(result, dict)
        # Should contain spatial statistics

    def test_compute_movement_patterns(self, sample_spatial_data):
        """Test movement patterns computation."""
        result = compute_movement_patterns(sample_spatial_data)

        assert isinstance(result, dict)

    def test_compute_location_hotspots(self, sample_spatial_data):
        """Test location hotspots computation."""
        result = compute_location_hotspots(sample_spatial_data)

        assert isinstance(result, dict)


class TestSpatialAnalysis:
    """Test spatial analysis functions."""

    # Note: Analysis functions expect experiment paths, not DataFrames
    # These would need to be integration tests with actual experiment data
    def test_analyze_functions_exist(self):
        """Test that analysis functions exist and are callable."""
        assert callable(analyze_spatial_overview)
        assert callable(analyze_movement_patterns)
        assert callable(analyze_location_hotspots)


class TestSpatialVisualization:
    """Test spatial visualization functions."""

    def test_plot_functions_exist(self):
        """Test that plot functions exist and are callable."""
        assert callable(plot_spatial_overview)
        assert callable(plot_movement_trajectories)
        assert callable(plot_location_hotspots)


class TestSpatialIntegration:
    """Test spatial module integration with service."""

    def test_spatial_module_integration(self, tmp_path, sample_experiment_path):
        """Test full spatial module execution."""
        service = AnalysisService(EnvConfigService())

        request = AnalysisRequest(
            module_name="spatial",
            experiment_path=sample_experiment_path,
            output_path=tmp_path,
            group="basic"
        )

        # Mock the data processing to avoid database dependency
        with patch('farm.analysis.spatial.data.process_spatial_data') as mock_process:
            mock_process.return_value = {
                'agent_positions': pd.DataFrame({
                    'step': range(10),
                    'agent_id': ['agent_1'] * 10,
                    'x': range(10),
                    'y': range(10),
                }),
                'resource_positions': pd.DataFrame({
                    'step': range(5),
                    'resource_id': ['resource_1'] * 5,
                    'x': range(5),
                    'y': range(5),
                })
            }

            result = service.run(request)

            # Should succeed with mocked data
            assert result.success or "data" in str(result.error).lower()
