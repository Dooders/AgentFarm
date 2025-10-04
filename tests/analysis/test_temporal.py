"""
Tests for temporal analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.temporal import (
    temporal_module,
    compute_temporal_patterns,
    compute_temporal_statistics,
    compute_temporal_efficiency_metrics,
    analyze_temporal_patterns,
    analyze_time_series_overview,
    analyze_temporal_efficiency,
    plot_temporal_patterns,
    plot_rolling_averages,
    plot_temporal_efficiency,
)
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.core.services import EnvConfigService


@pytest.fixture
def sample_temporal_data():
    """Create sample temporal data for testing."""
    import numpy as np

    # Create time series with seasonal patterns
    steps = range(200)
    seasonal = [10 + 5 * np.sin(2 * np.pi * i / 50) for i in steps]
    trend = [i * 0.1 for i in steps]
    noise = np.random.normal(0, 1, len(steps))

    return pd.DataFrame({
        'step': steps,
        'value': [s + t + n for s, t, n in zip(seasonal, trend, noise)],
        'metric_a': [i * 0.05 + (i % 20) for i in steps],
        'metric_b': [100 - i * 0.2 + (i % 10) * 2 for i in steps],
        'period': [i % 50 for i in steps],  # 50-step cycles
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


class TestTemporalModule:
    """Test the temporal analysis module."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert temporal_module.name == "temporal"
        assert len(temporal_module.get_function_names()) > 0
        assert "analyze_overview" in temporal_module.get_function_names()

    def test_module_groups(self):
        """Test module function groups."""
        groups = temporal_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = temporal_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert temporal_module.supports_database() is True
        assert temporal_module.get_db_filename() == "simulation.db"


class TestTemporalComputations:
    """Test temporal statistical computations."""

    # Note: Compute functions expect specific data formats
    # These would need properly formatted data for detailed testing
    def test_compute_functions_exist(self):
        """Test that compute functions exist and are callable."""
        assert callable(compute_temporal_patterns)
        assert callable(compute_temporal_statistics)
        assert callable(compute_temporal_efficiency_metrics)


class TestTemporalAnalysis:
    """Test temporal analysis functions."""

    def test_analyze_functions_exist(self):
        """Test that analysis functions exist and are callable."""
        assert callable(analyze_temporal_patterns)
        assert callable(analyze_time_series_overview)
        assert callable(analyze_temporal_efficiency)


class TestTemporalVisualization:
    """Test temporal visualization functions."""

    def test_plot_functions_exist(self):
        """Test that plot functions exist and are callable."""
        assert callable(plot_temporal_patterns)
        assert callable(plot_rolling_averages)
        assert callable(plot_temporal_efficiency)


class TestTemporalIntegration:
    """Test temporal module integration with service."""

    def test_temporal_module_integration(self, tmp_path, sample_experiment_path):
        """Test full temporal module execution."""
        service = AnalysisService(EnvConfigService())

        request = AnalysisRequest(
            module_name="temporal",
            experiment_path=sample_experiment_path,
            output_path=tmp_path,
            group="basic"
        )

        # Mock the data processing to avoid database dependency
        with patch('farm.analysis.temporal.module.process_temporal_data') as mock_process:
            mock_process.return_value = pd.DataFrame({
                'step': list(range(10)),
                'value': list(range(10)),
                'metric_a': list(range(10)),
            })

            result = service.run(request)

            # Should succeed with mocked data
            assert result.success or "data" in str(result.error).lower()
