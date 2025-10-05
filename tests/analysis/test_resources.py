"""
Comprehensive tests for resources analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.resources import (
    resources_module,
    compute_resource_statistics,
    compute_consumption_patterns,
    compute_efficiency_metrics,
    compute_resource_efficiency,
    compute_resource_hotspots,
    analyze_resource_patterns,
    analyze_consumption,
    analyze_resource_efficiency,
    analyze_hotspots,
    plot_resource_distribution,
    plot_consumption_over_time,
    plot_efficiency_metrics,
    plot_resource_hotspots,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def sample_resource_data():
    """Create sample resource data."""
    return pd.DataFrame({
        'step': range(100),
        'total_resources': [1000 - i * 5 for i in range(100)],  # Depleting
        'consumed_resources': [20 + i * 0.2 for i in range(100)],  # Increasing consumption
        'available_resources': [800 - i * 4 for i in range(100)],
        'resource_efficiency': [0.8 + (i % 20) * 0.01 for i in range(100)],
        'utilization_rate': [0.75 + (i % 15) * 0.01 for i in range(100)],
        'distribution_efficiency': [0.7 + (i % 25) * 0.01 for i in range(100)],
        'consumption_efficiency': [0.85 + (i % 10) * 0.005 for i in range(100)],
    })


@pytest.fixture
def sample_consumption_data():
    """Create sample consumption data."""
    return pd.DataFrame({
        'step': range(50),
        'consumed_resources': [10 + i * 0.5 for i in range(50)],
        'consumption_rate': [0.01 + i * 0.0002 for i in range(50)],
        'total_resources': [1000 - i * 10 for i in range(50)],
    })


@pytest.fixture
def sample_efficiency_data():
    """Create sample efficiency data."""
    return pd.DataFrame({
        'step': range(50),
        'resource_efficiency': [0.8 + (i % 10) * 0.02 for i in range(50)],
        'utilization_rate': [0.75 + (i % 15) * 0.01 for i in range(50)],
        'distribution_efficiency': [0.7 + (i % 20) * 0.01 for i in range(50)],
        'consumption_efficiency': [0.85 + (i % 12) * 0.01 for i in range(50)],
        'efficiency_gain': [(i % 10 - 5) * 0.01 for i in range(50)],
    })


class TestResourceComputations:
    """Test resource statistical computations."""

    def test_compute_resource_statistics(self, sample_resource_data):
        """Test resource statistics computation."""
        stats = compute_resource_statistics(sample_resource_data)
        
        assert isinstance(stats, dict)
        assert 'total' in stats
        assert 'consumption' in stats
        assert 'efficiency' in stats
        assert 'depletion_rate' in stats
        assert 'peak_step' in stats
        assert 'peak_value' in stats
        assert 'final_value' in stats
        
        # Verify statistics structure
        assert 'mean' in stats['total']
        assert 'std' in stats['total']
        assert stats['consumption']['total'] > 0

    def test_compute_resource_statistics_minimal(self):
        """Test resource statistics with minimal data."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000] * 10,
        })
        
        stats = compute_resource_statistics(df)
        
        assert 'total' in stats
        assert stats['peak_value'] == 1000

    def test_compute_resource_statistics_with_entropy(self):
        """Test resource statistics with entropy data."""
        df = pd.DataFrame({
            'step': range(20),
            'total_resources': [1000 - i * 10 for i in range(20)],
            'distribution_entropy': [0.5 + i * 0.01 for i in range(20)],
        })
        
        stats = compute_resource_statistics(df)
        
        assert 'entropy' in stats
        assert 'avg_distribution_uniformity' in stats

    def test_compute_consumption_patterns(self, sample_consumption_data):
        """Test consumption pattern computation."""
        patterns = compute_consumption_patterns(sample_consumption_data)
        
        assert isinstance(patterns, dict)
        assert 'trend' in patterns
        assert 'volatility' in patterns
        assert 'peak_consumption' in patterns
        assert 'increasing_periods' in patterns
        
        assert patterns['peak_consumption'] > 0
        assert patterns['volatility'] >= 0

    def test_compute_consumption_patterns_empty(self):
        """Test consumption patterns with missing column."""
        df = pd.DataFrame({'step': range(10)})
        
        result = compute_consumption_patterns(df)
        
        assert result == {}

    def test_compute_consumption_patterns_constant(self):
        """Test consumption patterns with constant consumption."""
        df = pd.DataFrame({
            'consumed_resources': [20] * 50,
        })
        
        patterns = compute_consumption_patterns(df)
        
        assert abs(patterns['trend']) < 1e-10  # Should be approximately 0.0
        assert patterns['peak_consumption'] == 20

    def test_compute_efficiency_metrics(self, sample_efficiency_data):
        """Test efficiency metrics computation."""
        metrics = compute_efficiency_metrics(sample_efficiency_data)
        
        assert isinstance(metrics, dict)
        assert 'mean_efficiency' in metrics
        assert 'efficiency_trend' in metrics
        assert 'efficiency_volatility' in metrics
        assert 'peak_efficiency' in metrics
        
        assert 0 <= metrics['mean_efficiency'] <= 1
        assert metrics['peak_efficiency'] <= 1

    def test_compute_efficiency_metrics_missing_column(self):
        """Test efficiency metrics without efficiency column."""
        df = pd.DataFrame({'step': range(10)})
        
        result = compute_efficiency_metrics(df)
        
        assert result == {}

    def test_compute_resource_efficiency(self, sample_efficiency_data):
        """Test resource efficiency computation."""
        efficiency = compute_resource_efficiency(sample_efficiency_data)
        
        assert isinstance(efficiency, dict)
        assert 'avg_utilization_rate' in efficiency
        assert 'avg_distribution_efficiency' in efficiency
        assert 'avg_consumption_efficiency' in efficiency
        assert 'overall_efficiency_score' in efficiency

    def test_compute_resource_efficiency_partial_data(self):
        """Test resource efficiency with partial data."""
        df = pd.DataFrame({
            'step': range(20),
            'utilization_rate': [0.8] * 20,
        })
        
        efficiency = compute_resource_efficiency(df)
        
        assert 'avg_utilization_rate' in efficiency
        assert abs(efficiency['avg_utilization_rate'] - 0.8) < 1e-10

    def test_compute_resource_efficiency_empty(self):
        """Test resource efficiency with no efficiency columns."""
        df = pd.DataFrame({'step': range(10)})
        
        result = compute_resource_efficiency(df)
        
        assert isinstance(result, dict)

    def test_compute_resource_hotspots(self, sample_resource_data):
        """Test resource hotspot computation."""
        hotspots = compute_resource_hotspots(sample_resource_data)
        
        assert isinstance(hotspots, dict)
        assert 'max_concentration' in hotspots
        assert 'avg_concentration' in hotspots
        assert 'concentration_ratio' in hotspots
        assert 'hotspot_intensity' in hotspots

    def test_compute_resource_hotspots_empty(self):
        """Test hotspots with missing data."""
        df = pd.DataFrame({'step': range(10)})
        
        result = compute_resource_hotspots(df)
        
        assert result == {}

    def test_compute_resource_hotspots_uniform(self):
        """Test hotspots with uniform distribution."""
        df = pd.DataFrame({
            'total_resources': [100] * 20,
        })
        
        hotspots = compute_resource_hotspots(df)
        
        # Uniform distribution should have ratio of 1.0
        assert abs(hotspots['concentration_ratio'] - 1.0) < 1e-8
        assert abs(hotspots['hotspot_intensity']) < 1e-8


class TestResourceAnalysis:
    """Test resource analysis functions."""

    def test_analyze_resource_patterns(self, tmp_path, sample_resource_data):
        """Test resource pattern analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_resource_patterns(sample_resource_data, ctx)
        
        output_file = tmp_path / "resource_patterns.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'statistics' in data
        assert 'patterns' in data
        assert 'efficiency' in data
        assert 'hotspots' in data

    def test_analyze_consumption(self, tmp_path, sample_consumption_data):
        """Test consumption analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_consumption(sample_consumption_data, ctx)
        
        output_file = tmp_path / "consumption_patterns.csv"
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) == len(sample_consumption_data)

    def test_analyze_resource_efficiency(self, tmp_path, sample_efficiency_data):
        """Test resource efficiency analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_resource_efficiency(sample_efficiency_data, ctx)
        
        output_file = tmp_path / "efficiency_analysis.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'metrics' in data
        assert 'improvement_rate' in data

    def test_analyze_hotspots(self, tmp_path, sample_resource_data):
        """Test hotspots analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_hotspots(sample_resource_data, ctx)
        
        output_file = tmp_path / "hotspot_analysis.json"
        assert output_file.exists()

    def test_analyze_efficiency_without_gain_column(self, tmp_path):
        """Test efficiency analysis without efficiency_gain column."""
        df = pd.DataFrame({
            'step': range(10),
            'utilization_rate': [0.8] * 10,
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_resource_efficiency(df, ctx)
        
        output_file = tmp_path / "efficiency_analysis.json"
        assert output_file.exists()


class TestResourceVisualization:
    """Test resource visualization functions."""

    def test_plot_resource_distribution(self, tmp_path, sample_resource_data):
        """Test resource distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_resource_distribution(sample_resource_data, ctx)
        
        plot_file = tmp_path / "resource_distribution.png"
        assert plot_file.exists()

    def test_plot_resource_distribution_with_average(self, tmp_path):
        """Test resource distribution with average per cell."""
        df = pd.DataFrame({
            'step': range(20),
            'total_resources': [1000 - i * 10 for i in range(20)],
            'average_per_cell': [50 - i * 0.5 for i in range(20)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_resource_distribution(df, ctx)
        
        plot_file = tmp_path / "resource_distribution.png"
        assert plot_file.exists()

    def test_plot_consumption_over_time(self, tmp_path, sample_consumption_data):
        """Test consumption over time plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_consumption_over_time(sample_consumption_data, ctx)
        
        plot_file = tmp_path / "consumption_over_time.png"
        assert plot_file.exists()

    def test_plot_consumption_different_columns(self, tmp_path):
        """Test consumption plotting with different column names."""
        # Test with avg_consumption_rate
        df1 = pd.DataFrame({
            'step': range(20),
            'avg_consumption_rate': [0.01 + i * 0.001 for i in range(20)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_consumption_over_time(df1, ctx)
        
        plot_file = tmp_path / "consumption_over_time.png"
        assert plot_file.exists()

    def test_plot_consumption_no_data(self, tmp_path):
        """Test consumption plotting with no consumption data."""
        df = pd.DataFrame({'step': range(10)})
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_consumption_over_time(df, ctx)
        
        # Should handle gracefully

    def test_plot_efficiency_metrics(self, tmp_path, sample_efficiency_data):
        """Test efficiency metrics plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_efficiency_metrics(sample_efficiency_data, ctx)
        
        plot_file = tmp_path / "efficiency_metrics.png"
        assert plot_file.exists()

    def test_plot_efficiency_metrics_partial(self, tmp_path):
        """Test efficiency plotting with partial data."""
        df = pd.DataFrame({
            'step': range(20),
            'resource_efficiency': [0.8 + i * 0.01 for i in range(20)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_efficiency_metrics(df, ctx)
        
        plot_file = tmp_path / "efficiency_metrics.png"
        assert plot_file.exists()

    def test_plot_efficiency_metrics_no_data(self, tmp_path):
        """Test efficiency plotting with no efficiency data."""
        df = pd.DataFrame({'step': range(10)})
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_efficiency_metrics(df, ctx)
        
        # Should handle gracefully

    def test_plot_resource_hotspots(self, tmp_path, sample_resource_data):
        """Test resource hotspots plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_resource_hotspots(sample_resource_data, ctx)
        
        plot_file = tmp_path / "resource_hotspots.png"
        assert plot_file.exists()

    def test_plot_hotspots_short_data(self, tmp_path):
        """Test hotspots plotting with short time series."""
        df = pd.DataFrame({
            'step': range(5),
            'total_resources': [1000, 980, 960, 940, 920],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_resource_hotspots(df, ctx)
        
        plot_file = tmp_path / "resource_hotspots.png"
        assert plot_file.exists()

    def test_plot_with_custom_options(self, tmp_path, sample_resource_data):
        """Test plotting with custom options."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_resource_distribution(sample_resource_data, ctx, figsize=(10, 8), dpi=150)
        
        plot_file = tmp_path / "resource_distribution.png"
        assert plot_file.exists()


class TestResourcesModule:
    """Test resources module integration."""

    def test_resources_module_registration(self):
        """Test module registration."""
        assert resources_module.name == "resources"
        assert resources_module.description == "Analysis of resource distribution, consumption, efficiency, and hotspot patterns"

    def test_resources_module_function_names(self):
        """Test module function names."""
        functions = resources_module.get_function_names()
        
        assert "analyze_patterns" in functions
        assert "plot_resource" in functions
        assert "analyze_consumption" in functions
        assert "plot_consumption" in functions
        assert "analyze_efficiency" in functions
        assert "plot_efficiency" in functions

    def test_resources_module_function_groups(self):
        """Test module function groups."""
        groups = resources_module.get_function_groups()
        
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "basic" in groups
        assert "efficiency" in groups

    def test_resources_module_data_processor(self):
        """Test module data processor."""
        processor = resources_module.get_data_processor()
        assert processor is not None
        assert hasattr(processor, 'process')

    def test_module_validator(self):
        """Test module validator."""
        validator = resources_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = resources_module.get_functions()
        assert len(functions) >= 8

    def test_module_group_names(self):
        """Test getting group names."""
        groups = resources_module.get_group_names()
        assert isinstance(groups, list)
        assert len(groups) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_statistics_single_step(self):
        """Test statistics with single time step."""
        df = pd.DataFrame({
            'step': [0],
            'total_resources': [1000],
        })
        
        stats = compute_resource_statistics(df)
        
        assert stats['total']['mean'] == 1000
        assert stats['peak_value'] == 1000

    def test_compute_statistics_with_nan(self):
        """Test statistics with NaN values."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000, np.nan, 950, 920, np.nan, 830, 780, np.nan, 650, 580],
            'consumed_resources': [20, 30, np.nan, 40, 50, np.nan, 60, 70, 70, np.nan],
        })
        
        # Should handle NaN values
        stats = compute_resource_statistics(df)
        assert isinstance(stats, dict)

    def test_compute_consumption_zero_mean(self):
        """Test consumption patterns with zero mean."""
        df = pd.DataFrame({
            'consumed_resources': [0] * 10,
        })
        
        patterns = compute_consumption_patterns(df)
        
        # Should handle zero mean gracefully (division by zero protection)
        assert 'volatility' in patterns

    def test_compute_efficiency_constant_values(self):
        """Test efficiency metrics with constant efficiency."""
        df = pd.DataFrame({
            'resource_efficiency': [0.8] * 20,
        })
        
        metrics = compute_efficiency_metrics(df)
        
        assert abs(metrics['mean_efficiency'] - 0.8) < 1e-10
        assert abs(metrics['efficiency_trend']) < 1e-10

    def test_compute_efficiency_decreasing_trend(self):
        """Test efficiency metrics with decreasing trend."""
        df = pd.DataFrame({
            'resource_efficiency': [0.9 - i * 0.01 for i in range(20)],
        })
        
        metrics = compute_efficiency_metrics(df)
        
        assert metrics['efficiency_trend'] < 0

    def test_analyze_patterns_minimal_data(self, tmp_path):
        """Test pattern analysis with minimal data."""
        df = pd.DataFrame({
            'step': [0, 1],
            'total_resources': [1000, 990],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_resource_patterns(df, ctx)
        
        output_file = tmp_path / "resource_patterns.json"
        assert output_file.exists()

    def test_plot_resource_negative_values(self, tmp_path):
        """Test plotting with negative resource values."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000 - i * 150 for i in range(10)],  # Goes negative
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_resource_distribution(df, ctx)
        
        # Should handle negative values
        plot_file = tmp_path / "resource_distribution.png"
        assert plot_file.exists()

    def test_plot_consumption_all_zeros(self, tmp_path):
        """Test consumption plotting with all zeros."""
        df = pd.DataFrame({
            'step': range(20),
            'consumed_resources': [0] * 20,
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_consumption_over_time(df, ctx)

    def test_compute_hotspots_high_concentration(self):
        """Test hotspots with very high concentration."""
        df = pd.DataFrame({
            'total_resources': [10, 10, 10, 1000, 10],
        })
        
        hotspots = compute_resource_hotspots(df)
        
        # Should detect high concentration
        assert hotspots['concentration_ratio'] > 1.0
        assert hotspots['hotspot_intensity'] > 0.0

    def test_resource_stability_zero_resources(self):
        """Test resource stability with zero resources."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [0] * 10,
        })
        
        stats = compute_resource_statistics(df)
        
        # Should handle zero resources
        assert 'resource_stability' in stats

    def test_analyze_with_progress_callback(self, tmp_path, sample_resource_data):
        """Test analysis with progress reporting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        progress_calls = []
        ctx.report_progress = lambda msg, prog: progress_calls.append((msg, prog))
        
        analyze_resource_patterns(sample_resource_data, ctx)
        
        # Should have called progress
        assert len(progress_calls) > 0

    def test_efficiency_with_regeneration(self):
        """Test efficiency computation with regeneration rate."""
        df = pd.DataFrame({
            'step': range(20),
            'utilization_rate': [0.8] * 20,
            'regeneration_rate': [0.1 + i * 0.01 for i in range(20)],
        })
        
        efficiency = compute_resource_efficiency(df)
        
        assert 'avg_regeneration_rate' in efficiency


class TestResourceHelperFunctions:
    """Test helper functions in resources module."""

    def test_process_resource_data(self, tmp_path):
        """Test processing resource data from experiment."""
        from farm.analysis.resources.data import process_resource_data
        
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        
        # Create mock database
        db_path = exp_path / "simulation.db"
        db_path.touch()
        
        # Create data directory with a CSV file as fallback
        data_dir = exp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "resources.csv"
        csv_path.write_text("step,total_resources\n0,1000\n1,950\n")
        
        with patch('farm.database.session_manager.SessionManager') as mock_sm, \
             patch('farm.database.repositories.resource_repository.ResourceRepository') as mock_repo:
            
            # Mock the repository methods to return empty data
            mock_repo_instance = MagicMock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.resource_distribution.return_value = []
            mock_repo_instance.consumption_patterns.return_value = MagicMock(
                total_consumed=0, avg_consumption_rate=0.0, 
                peak_consumption=0, consumption_variance=0.0
            )
            mock_repo_instance.efficiency_metrics.return_value = MagicMock(
                utilization_rate=0.0, distribution_efficiency=0.0,
                consumption_efficiency=0.0, regeneration_rate=0.0
            )
            
            result = process_resource_data(exp_path)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2  # Should load from CSV fallback

    def test_calculate_trend_helper(self):
        """Test trend calculation helper."""
        from farm.analysis.common.utils import calculate_trend
        
        increasing = np.array([10, 20, 30, 40, 50])
        trend = calculate_trend(increasing)
        
        assert trend > 0

    def test_calculate_statistics_helper(self):
        """Test statistics calculation helper."""
        from farm.analysis.common.utils import calculate_statistics
        
        data = np.array([10, 20, 30, 40, 50])
        stats = calculate_statistics(data)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert stats['mean'] == 30.0
