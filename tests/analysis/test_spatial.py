"""
Comprehensive tests for spatial analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.spatial import (
    spatial_module,
    compute_spatial_statistics,
    compute_movement_patterns,
    compute_location_hotspots,
    compute_spatial_distribution_metrics,
    analyze_spatial_overview,
    analyze_movement_patterns,
    analyze_location_hotspots,
    analyze_spatial_distribution,
    plot_spatial_overview,
    plot_movement_trajectories,
    plot_location_hotspots,
    plot_spatial_density,
    plot_movement_directions,
    plot_clustering_analysis,
    calculate_euclidean_distance,
)


@pytest.fixture
def sample_agent_positions():
    """Create sample agent position data."""
    return pd.DataFrame(
        {
            "step": list(range(50)) * 2,
            "agent_id": ["agent_1"] * 50 + ["agent_2"] * 50,
            "position_x": np.random.uniform(0, 100, 100),
            "position_y": np.random.uniform(0, 100, 100),
        }
    )


@pytest.fixture
def sample_resource_positions():
    """Create sample resource position data."""
    return pd.DataFrame(
        {
            "step": list(range(20)) * 3,
            "resource_id": ["resource_1"] * 20 + ["resource_2"] * 20 + ["resource_3"] * 20,
            "position_x": np.random.uniform(0, 100, 60),
            "position_y": np.random.uniform(0, 100, 60),
        }
    )


@pytest.fixture
def sample_spatial_data(sample_agent_positions, sample_resource_positions):
    """Create sample spatial data dictionary."""
    return {
        "agent_positions": sample_agent_positions,
        "resource_positions": sample_resource_positions,
    }


@pytest.fixture
def sample_movement_data():
    """Create sample movement trajectory data."""
    return pd.DataFrame(
        {
            "step": range(100),
            "agent_id": [f"agent_{i % 10}" for i in range(100)],
            "position_x": [i * 0.1 + (i % 10) for i in range(100)],
            "position_y": [i * 0.05 + (i % 5) * 2 for i in range(100)],
            "direction": [i % 360 for i in range(100)],
            "speed": [1.0 + (i % 10) * 0.1 for i in range(100)],
            "distance": [1.0 + (i % 5) * 0.2 for i in range(100)],
        }
    )


@pytest.fixture
def sample_location_data():
    """Create sample location activity data."""
    return {
        "location_activity": pd.DataFrame(
            {
                "position_x": np.random.uniform(0, 100, 50),
                "position_y": np.random.uniform(0, 100, 50),
                "activity": np.random.randint(1, 100, 50),
            }
        )
    }


class TestSpatialComputations:
    """Test spatial statistical computations."""

    def test_compute_spatial_statistics(self, sample_spatial_data):
        """Test comprehensive spatial statistics computation."""
        result = compute_spatial_statistics(sample_spatial_data)

        assert isinstance(result, dict)
        assert "agent_spatial" in result
        assert "resource_spatial" in result
        assert "interaction_spatial" in result

    def test_compute_spatial_statistics_empty(self):
        """Test spatial statistics with empty data."""
        result = compute_spatial_statistics(
            {
                "agent_positions": pd.DataFrame(),
                "resource_positions": pd.DataFrame(),
            }
        )

        assert isinstance(result, dict)
        # Should not have spatial stats for empty data
        assert "agent_spatial" not in result

    def test_compute_spatial_statistics_agents_only(self, sample_agent_positions):
        """Test spatial statistics with only agent data."""
        result = compute_spatial_statistics(
            {
                "agent_positions": sample_agent_positions,
                "resource_positions": pd.DataFrame(),
            }
        )

        assert "agent_spatial" in result
        assert "resource_spatial" not in result

    def test_compute_movement_patterns(self, sample_movement_data):
        """Test movement pattern computation."""
        result = compute_movement_patterns(sample_movement_data)

        assert isinstance(result, dict)
        assert "total_movements" in result
        assert "avg_distance" in result
        assert "total_distance" in result
        assert "movement_frequency" in result
        assert result["total_movements"] == len(sample_movement_data)

    def test_compute_movement_patterns_empty(self):
        """Test movement patterns with empty DataFrame."""
        result = compute_movement_patterns(pd.DataFrame())

        assert isinstance(result, dict)
        assert result["total_movements"] == 0
        assert result["avg_distance"] == 0.0

    def test_compute_location_hotspots(self, sample_location_data):
        """Test location hotspots computation."""
        result = compute_location_hotspots(sample_location_data)

        assert isinstance(result, dict)
        assert "hotspots" in result
        assert "clusters" in result

    def test_compute_location_hotspots_empty(self):
        """Test location hotspots with empty data."""
        result = compute_location_hotspots({"location_activity": pd.DataFrame()})

        assert result == {"hotspots": [], "clusters": {}}

    def test_compute_spatial_distribution_metrics(self, sample_agent_positions):
        """Test spatial distribution metrics."""
        result = compute_spatial_distribution_metrics(sample_agent_positions)

        assert isinstance(result, dict)
        assert "centroid" in result
        assert "spread" in result
        assert len(result["centroid"]) == 2  # x, y coordinates

    def test_compute_spatial_distribution_metrics_empty(self):
        """Test distribution metrics with empty data."""
        result = compute_spatial_distribution_metrics(pd.DataFrame())

        assert result == {}

    def test_compute_spatial_distribution_metrics_insufficient_data(self):
        """Test distribution metrics with single point."""
        df = pd.DataFrame(
            {
                "position_x": [50.0],
                "position_y": [50.0],
            }
        )

        result = compute_spatial_distribution_metrics(df)

        # Should return empty dict for insufficient data
        assert result == {}

    def test_calculate_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        point1 = (0, 0)
        point2 = (3, 4)

        distance = calculate_euclidean_distance(point1, point2)

        assert distance == 5.0

    def test_calculate_euclidean_distance_same_point(self):
        """Test distance for same point."""
        point = (10, 20)

        distance = calculate_euclidean_distance(point, point)

        assert distance == 0.0


class TestSpatialAnalysis:
    """Test spatial analysis functions."""

    def test_analyze_spatial_overview(self, tmp_path, sample_spatial_data):
        """Test spatial overview analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_spatial_data") as mock_process:
            mock_process.return_value = sample_spatial_data

            analyze_spatial_overview(str(exp_path), ctx)

            # Should create output file
            output_file = tmp_path / "spatial_overview.json"
            assert output_file.exists()

    def test_analyze_movement_patterns(self, tmp_path, sample_movement_data):
        """Test movement patterns analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_movement_data") as mock_process:
            mock_process.return_value = sample_movement_data

            analyze_movement_patterns(str(exp_path), ctx)

            # Should create output file
            output_file = tmp_path / "movement_patterns.json"
            assert output_file.exists()

    def test_analyze_movement_patterns_no_data(self, tmp_path):
        """Test movement patterns with no data."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_movement_data") as mock_process:
            mock_process.return_value = pd.DataFrame()

            # Should handle gracefully
            analyze_movement_patterns(str(exp_path), ctx)

    def test_analyze_location_hotspots(self, tmp_path, sample_location_data):
        """Test location hotspots analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_location_analysis_data") as mock_process:
            mock_process.return_value = sample_location_data

            analyze_location_hotspots(str(exp_path), ctx)

            # Should create output file
            output_file = tmp_path / "location_hotspots.json"
            assert output_file.exists()

    def test_analyze_spatial_distribution(self, tmp_path, sample_spatial_data):
        """Test spatial distribution analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_spatial_data") as mock_process:
            mock_process.return_value = sample_spatial_data

            analyze_spatial_distribution(str(exp_path), ctx)

            # Should create output file
            output_file = tmp_path / "spatial_distribution.json"
            assert output_file.exists()

    def test_analyze_with_agent_ids_filter(self, tmp_path, sample_movement_data):
        """Test movement analysis with agent ID filter."""
        ctx = AnalysisContext(output_path=tmp_path)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.analyze.process_movement_data") as mock_process:
            mock_process.return_value = sample_movement_data

            # Pass agent_ids filter
            analyze_movement_patterns(str(exp_path), ctx, agent_ids=["agent_1", "agent_2"])

            # Should call with agent_ids parameter
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert "agent_ids" in call_kwargs


class TestSpatialVisualization:
    """Test spatial visualization functions."""

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_spatial_overview(self, mock_plt, sample_spatial_data, tmp_path):
        """Test spatial overview plotting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ctx = AnalysisContext(output_path=tmp_path)

        plot_spatial_overview(sample_spatial_data, ctx)

        assert mock_plt.subplots.called
        assert mock_plt.close.called

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_spatial_overview_empty(self, mock_plt, tmp_path):
        """Test spatial overview with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        empty_data = {
            "agent_positions": pd.DataFrame(),
            "resource_positions": pd.DataFrame(),
        }

        plot_spatial_overview(empty_data, ctx)

        # Should handle empty data gracefully
        # May not create plot

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_movement_trajectories(self, mock_plt, sample_movement_data, tmp_path):
        """Test movement trajectories plotting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Mock the colormap to return a proper array
        mock_plt.cm.viridis.return_value = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

        ctx = AnalysisContext(output_path=tmp_path)

        plot_movement_trajectories(sample_movement_data, ctx)

        assert mock_plt.subplots.called
        assert mock_plt.close.called

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_movement_trajectories_empty(self, mock_plt, tmp_path):
        """Test trajectories plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_movement_trajectories(pd.DataFrame(), ctx)

        # Should handle empty data gracefully

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_location_hotspots(self, mock_plt, tmp_path):
        """Test location hotspots plotting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ctx = AnalysisContext(output_path=tmp_path)

        hotspots_data = {
            "hotspots": [
                {"position_x": 10, "position_y": 20, "activity": 50},
                {"position_x": 30, "position_y": 40, "activity": 75},
            ]
        }

        plot_location_hotspots(hotspots_data, ctx)

        assert mock_plt.subplots.called
        assert mock_plt.close.called

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_location_hotspots_empty(self, mock_plt, tmp_path):
        """Test hotspots plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_location_hotspots({"hotspots": []}, ctx)

        # Should handle empty data gracefully

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_spatial_density(self, mock_plt, tmp_path):
        """Test spatial density plotting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ctx = AnalysisContext(output_path=tmp_path)

        density_data = {
            "density_map": np.random.random((10, 10)),
            "x_edges": np.linspace(0, 100, 11),
            "y_edges": np.linspace(0, 100, 11),
        }

        plot_spatial_density(density_data, ctx)

        assert mock_plt.subplots.called
        assert mock_plt.close.called

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_movement_directions(self, mock_plt, sample_movement_data, tmp_path):
        """Test movement directions plotting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ctx = AnalysisContext(output_path=tmp_path)

        # Add direction column if not present
        if "direction" not in sample_movement_data.columns:
            sample_movement_data["direction"] = np.random.uniform(0, 360, len(sample_movement_data))

        plot_movement_directions(sample_movement_data, ctx)

        assert mock_plt.subplots.called
        assert mock_plt.close.called

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_clustering_analysis(self, mock_plt, sample_agent_positions, tmp_path):
        """Test clustering analysis plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        clustering_data = {
            "positions": sample_agent_positions,
            "cluster_labels": np.random.randint(0, 3, len(sample_agent_positions)),
        }

        plot_clustering_analysis(clustering_data, ctx)

        # Should attempt to create plot

    @patch("farm.analysis.spatial.plot.plt")
    def test_plot_with_custom_dpi(self, mock_plt, sample_spatial_data, tmp_path):
        """Test plotting with custom DPI setting."""
        # Mock the subplots return value
        mock_fig = mock_plt.subplots.return_value[0]
        mock_ax = mock_plt.subplots.return_value[1]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        ctx = AnalysisContext(output_path=tmp_path)

        plot_spatial_overview(sample_spatial_data, ctx, dpi=150)

        assert mock_plt.subplots.called
        assert mock_plt.close.called


class TestSpatialModule:
    """Test spatial module integration."""

    def test_module_registration(self):
        """Test module is properly registered."""
        assert spatial_module.name == "spatial"
        assert (
            spatial_module.description
            == "Analysis of spatial patterns, movement trajectories, location effects, and clustering"
        )

    def test_module_function_names(self):
        """Test module function names."""
        functions = spatial_module.get_function_names()

        assert "analyze_overview" in functions
        assert "analyze_movement" in functions
        assert "analyze_hotspots" in functions
        assert "plot_overview" in functions
        assert "plot_trajectories" in functions

    def test_module_groups(self):
        """Test module function groups."""
        groups = spatial_module.get_function_groups()

        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "movement" in groups
        assert "location" in groups
        assert "basic" in groups

    def test_data_processor(self):
        """Test data processor creation."""
        processor = spatial_module.get_data_processor()
        assert processor is not None

    def test_supports_database(self):
        """Test database support."""
        assert spatial_module.supports_database() is True
        assert spatial_module.get_db_filename() == "simulation.db"

    def test_module_validator(self):
        """Test module validator."""
        validator = spatial_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = spatial_module.get_functions()
        assert len(functions) >= 10


class TestSpatialDataProcessing:
    """Test spatial data processing functions."""

    def test_process_spatial_data(self, tmp_path):
        """Test processing spatial data from experiment."""
        from farm.analysis.spatial.data import process_spatial_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        # Create mock database
        db_path = exp_path / "simulation.db"
        db_path.touch()

        with patch("farm.analysis.spatial.data.SessionManager") as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session

            # Mock query results
            mock_session.query.return_value.all.return_value = []

            result = process_spatial_data(exp_path)

            assert isinstance(result, dict)

    def test_process_movement_data(self, tmp_path):
        """Test processing movement data."""
        from farm.analysis.spatial.data import process_movement_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.data.SessionManager") as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []

            result = process_movement_data(exp_path)

            assert isinstance(result, pd.DataFrame)

    def test_process_location_analysis_data(self, tmp_path):
        """Test processing location analysis data."""
        from farm.analysis.spatial.data import process_location_analysis_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.spatial.data.SessionManager") as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []

            result = process_location_analysis_data(exp_path)

            assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_statistics_missing_columns(self):
        """Test spatial statistics with missing position columns."""
        data = {
            "agent_positions": pd.DataFrame({"agent_id": [1, 2, 3]}),
            "resource_positions": pd.DataFrame(),
        }

        result = compute_spatial_statistics(data)

        # Should handle missing columns gracefully
        assert isinstance(result, dict)

    def test_compute_movement_patterns_missing_distance(self):
        """Test movement patterns without distance column."""
        df = pd.DataFrame(
            {
                "agent_id": ["agent_1"] * 10,
                "position_x": range(10),
                "position_y": range(10),
            }
        )

        # Should handle missing distance column
        try:
            result = compute_movement_patterns(df)
            # If it doesn't error, should return dict
            assert isinstance(result, dict)
        except KeyError:
            # Expected if distance is required
            pass

    def test_compute_distribution_with_nan_values(self):
        """Test distribution metrics with NaN values."""
        df = pd.DataFrame(
            {
                "position_x": [1.0, 2.0, np.nan, 4.0, 5.0],
                "position_y": [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )

        result = compute_spatial_distribution_metrics(df)

        # Should handle NaN values
        assert isinstance(result, dict)

    def test_plot_trajectories_single_agent(self, tmp_path):
        """Test plotting trajectories for single agent."""
        ctx = AnalysisContext(output_path=tmp_path)

        df = pd.DataFrame(
            {
                "agent_id": ["agent_1"] * 20,
                "position_x": range(20),
                "position_y": range(20),
            }
        )

        with patch("farm.analysis.spatial.plot.plt") as mock_plt:
            # Mock the subplots return value
            mock_fig = mock_plt.subplots.return_value[0]
            mock_ax = mock_plt.subplots.return_value[1]
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the colormap to return a proper array
            mock_plt.cm.viridis.return_value = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

            plot_movement_trajectories(df, ctx)

    def test_plot_trajectories_many_agents(self, tmp_path):
        """Test plotting trajectories for many agents (>10)."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Create data for 20 agents
        df = pd.DataFrame(
            {
                "agent_id": [f"agent_{i}" for i in range(20) for _ in range(10)],
                "position_x": np.random.uniform(0, 100, 200),
                "position_y": np.random.uniform(0, 100, 200),
            }
        )

        with patch("farm.analysis.spatial.plot.plt") as mock_plt:
            # Mock the subplots return value
            mock_fig = mock_plt.subplots.return_value[0]
            mock_ax = mock_plt.subplots.return_value[1]
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the colormap to return a proper array
            mock_plt.cm.viridis.return_value = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

            plot_movement_trajectories(df, ctx)

    def test_hotspots_zero_activity(self):
        """Test hotspots with zero activity."""
        data = {
            "location_activity": pd.DataFrame(
                {
                    "position_x": [10, 20, 30],
                    "position_y": [10, 20, 30],
                    "activity": [0, 0, 0],
                }
            )
        }

        result = compute_location_hotspots(data)

        assert isinstance(result, dict)

    def test_spatial_statistics_extreme_coordinates(self):
        """Test spatial statistics with extreme coordinate values."""
        df = pd.DataFrame(
            {
                "position_x": [0, 1000000],
                "position_y": [0, 1000000],
            }
        )

        result = compute_spatial_distribution_metrics(df)

        # Should handle extreme values
        assert isinstance(result, dict)

    def test_movement_patterns_zero_distance(self):
        """Test movement patterns with zero movement."""
        df = pd.DataFrame(
            {
                "agent_id": ["agent_1"] * 10,
                "distance": [0.0] * 10,
            }
        )

        result = compute_movement_patterns(df)

        assert result["total_distance"] == 0.0
        assert result["avg_distance"] == 0.0

    def test_analyze_with_invalid_path(self, tmp_path):
        """Test analysis with invalid experiment path."""
        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.spatial.analyze.process_spatial_data") as mock_process:
            mock_process.side_effect = FileNotFoundError("Path not found")

            # Should raise error
            with pytest.raises(FileNotFoundError):
                analyze_spatial_overview("/invalid/path", ctx)

    def test_validator_with_invalid_data(self):
        """Test spatial data validator with invalid data."""
        from farm.analysis.spatial.module import SpatialDataValidator
        from farm.analysis.validation import DataValidationError

        validator = SpatialDataValidator(min_rows=1)

        # Test with non-dict
        with pytest.raises(DataValidationError):
            validator.validate("not a dict")

        # Test with missing keys
        with pytest.raises(DataValidationError):
            validator.validate({"agent_positions": pd.DataFrame()})

        # Test with non-DataFrame values
        with pytest.raises(DataValidationError):
            validator.validate({"agent_positions": "not a dataframe", "resource_positions": pd.DataFrame()})

    def test_validator_with_insufficient_rows(self):
        """Test validator with insufficient rows."""
        from farm.analysis.spatial.module import SpatialDataValidator
        from farm.analysis.validation import DataValidationError

        validator = SpatialDataValidator(min_rows=10)

        data = {
            "agent_positions": pd.DataFrame({"x": [1, 2, 3]}),  # Only 3 rows
            "resource_positions": pd.DataFrame({"x": [1, 2, 3]}),
        }

        with pytest.raises(DataValidationError):
            validator.validate(data)


class TestSpatialHelperFunctions:
    """Test helper functions in spatial module."""

    def test_analyze_movement_trajectories(self, sample_movement_data):
        """Test movement trajectory analysis."""
        from farm.analysis.spatial.movement import analyze_movement_trajectories

        result = analyze_movement_trajectories(sample_movement_data)

        assert isinstance(result, dict)

    def test_analyze_location_effects(self, sample_location_data):
        """Test location effects analysis."""
        from farm.analysis.spatial.location import analyze_location_effects

        result = analyze_location_effects(sample_location_data)

        assert isinstance(result, dict)

    def test_analyze_clustering_patterns(self, sample_agent_positions):
        """Test clustering patterns analysis."""
        from farm.analysis.spatial.location import analyze_clustering_patterns

        result = analyze_clustering_patterns(sample_agent_positions)

        assert isinstance(result, dict)

    def test_analyze_resource_location_patterns(self, sample_resource_positions):
        """Test resource location patterns analysis."""
        from farm.analysis.spatial.location import analyze_resource_location_patterns

        result = analyze_resource_location_patterns(sample_resource_positions)

        assert isinstance(result, dict)

    def test_analyze_movement_patterns_detailed(self, sample_movement_data):
        """Test detailed movement patterns analysis."""
        from farm.analysis.spatial.movement import analyze_movement_patterns_detailed

        result = analyze_movement_patterns_detailed(sample_movement_data)

        assert isinstance(result, dict)
