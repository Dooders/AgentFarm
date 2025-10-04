"""
Tests for resources analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from farm.analysis.resources import (
    resources_module,
    compute_resource_statistics,
    compute_consumption_patterns,
    compute_efficiency_metrics,
    analyze_resource_patterns,
    analyze_consumption,
    analyze_efficiency,
    plot_resource_over_time,
    plot_consumption_patterns,
    plot_efficiency_trends,
)
from farm.analysis.common.context import AnalysisContext


class TestResourceComputations:
    """Test resource statistical computations."""

    def test_compute_resource_statistics(self):
        """Test resource statistics computation."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000, 980, 950, 920, 880, 830, 780, 720, 650, 580],
            'consumed_resources': [20, 30, 30, 40, 50, 50, 60, 70, 70, 70],
            'available_resources': [800, 750, 700, 650, 600, 550, 500, 450, 400, 350],
            'resource_efficiency': [0.8, 0.82, 0.79, 0.85, 0.83, 0.81, 0.87, 0.86, 0.84, 0.82],
        })

        stats = compute_resource_statistics(df)

        assert 'total' in stats
        assert 'consumption' in stats
        assert 'efficiency' in stats
        assert 'depletion_rate' in stats

        # Check basic statistics
        assert stats['total']['min'] == 580
        assert stats['total']['max'] == 1000
        assert 'mean' in stats['total']
        assert 'std' in stats['total']

        # Check consumption stats
        assert stats['consumption']['total'] == 490  # sum of consumed
        assert stats['consumption']['mean'] == 49.0   # average consumed per step

    def test_compute_consumption_patterns(self):
        """Test consumption pattern computation."""
        df = pd.DataFrame({
            'step': range(20),
            'consumed_resources': [10, 12, 15, 8, 20, 18, 25, 22, 30, 28,
                                 35, 32, 40, 38, 45, 42, 50, 48, 55, 52],
            'total_resources': range(1000, 980, -1),  # decreasing
        })

        patterns = compute_consumption_patterns(df)

        assert 'trend' in patterns
        assert 'volatility' in patterns
        assert 'peak_consumption' in patterns
        assert 'increasing_periods' in patterns

        assert patterns['peak_consumption'] == 55
        assert patterns['trend'] > 0  # Should be increasing
        assert patterns['volatility'] >= 0

    def test_compute_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        df = pd.DataFrame({
            'step': range(10),
            'resource_efficiency': [0.8, 0.82, 0.79, 0.85, 0.83, 0.81, 0.87, 0.86, 0.84, 0.82],
            'consumed_resources': [20, 22, 18, 25, 23, 21, 28, 26, 24, 22],
            'total_resources': [1000, 980, 960, 940, 920, 900, 880, 860, 840, 820],
        })

        efficiency = compute_efficiency_metrics(df)

        assert 'mean_efficiency' in efficiency
        assert 'efficiency_trend' in efficiency
        assert 'efficiency_volatility' in efficiency
        assert 'peak_efficiency' in efficiency

        assert 0 <= efficiency['mean_efficiency'] <= 1
        assert efficiency['peak_efficiency'] <= 1
        assert efficiency['efficiency_volatility'] >= 0


class TestResourceAnalysis:
    """Test resource analysis functions."""

    def test_analyze_resource_patterns(self, tmp_path):
        """Test resource pattern analysis."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000, 980, 950, 920, 880, 830, 780, 720, 650, 580],
            'consumed_resources': [20, 30, 30, 40, 50, 50, 60, 70, 70, 70],
            'resource_efficiency': [0.8, 0.82, 0.79, 0.85, 0.83, 0.81, 0.87, 0.86, 0.84, 0.82],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_resource_patterns(df, ctx)

        stats_file = tmp_path / "resource_patterns.json"
        assert stats_file.exists()

        with open(stats_file) as f:
            data = json.load(f)

        assert 'statistics' in data
        assert 'patterns' in data
        assert 'efficiency' in data

    def test_analyze_consumption(self, tmp_path):
        """Test consumption analysis."""
        df = pd.DataFrame({
            'step': range(15),
            'consumed_resources': [10, 12, 15, 8, 20, 18, 25, 22, 30, 28, 35, 32, 40, 38, 45],
            'consumption_rate': [0.01, 0.012, 0.015, 0.008, 0.02, 0.018, 0.025, 0.022, 0.03, 0.028,
                               0.035, 0.032, 0.04, 0.038, 0.045],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_consumption(df, ctx)

        patterns_file = tmp_path / "consumption_patterns.csv"
        assert patterns_file.exists()

        patterns_df = pd.read_csv(patterns_file)
        assert len(patterns_df) == 15
        assert 'consumed_resources' in patterns_df.columns

    def test_analyze_efficiency(self, tmp_path):
        """Test efficiency analysis."""
        df = pd.DataFrame({
            'step': range(10),
            'resource_efficiency': [0.8, 0.82, 0.79, 0.85, 0.83, 0.81, 0.87, 0.86, 0.84, 0.82],
            'efficiency_gain': [0.02, -0.03, 0.06, -0.02, -0.02, 0.06, -0.01, -0.02, -0.02],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_efficiency(df, ctx)

        efficiency_file = tmp_path / "efficiency_analysis.json"
        assert efficiency_file.exists()

        with open(efficiency_file) as f:
            data = json.load(f)

        assert 'metrics' in data
        assert 'improvement_rate' in data


class TestResourceVisualization:
    """Test resource visualization functions."""

    def test_plot_resource_over_time(self, tmp_path):
        """Test resource over time plotting."""
        df = pd.DataFrame({
            'step': range(10),
            'total_resources': [1000, 980, 950, 920, 880, 830, 780, 720, 650, 580],
            'consumed_resources': [20, 30, 30, 40, 50, 50, 60, 70, 70, 70],
            'available_resources': [800, 750, 700, 650, 600, 550, 500, 450, 400, 350],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_resource_over_time(df, ctx)

        plot_file = tmp_path / "resource_over_time.png"
        assert plot_file.exists()

    def test_plot_consumption_patterns(self, tmp_path):
        """Test consumption patterns plotting."""
        df = pd.DataFrame({
            'step': range(15),
            'consumed_resources': [10, 12, 15, 8, 20, 18, 25, 22, 30, 28, 35, 32, 40, 38, 45],
            'consumption_rate': [0.01, 0.012, 0.015, 0.008, 0.02, 0.018, 0.025, 0.022, 0.03, 0.028,
                               0.035, 0.032, 0.04, 0.038, 0.045],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_consumption_patterns(df, ctx)

        plot_file = tmp_path / "consumption_patterns.png"
        assert plot_file.exists()

    def test_plot_efficiency_trends(self, tmp_path):
        """Test efficiency trends plotting."""
        df = pd.DataFrame({
            'step': range(10),
            'resource_efficiency': [0.8, 0.82, 0.79, 0.85, 0.83, 0.81, 0.87, 0.86, 0.84, 0.82],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_efficiency_trends(df, ctx)

        plot_file = tmp_path / "efficiency_trends.png"
        assert plot_file.exists()


class TestResourcesModule:
    """Test resources module integration."""

    def test_resources_module_registration(self):
        """Test module is properly registered."""
        assert resources_module.name == "resources"
        assert len(resources_module.get_function_names()) > 0
        assert "analyze_patterns" in resources_module.get_function_names()
        assert "plot_resource" in resources_module.get_function_names()

    def test_resources_module_function_groups(self):
        """Test module function groups."""
        groups = resources_module.get_group_names()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_resources_module_data_processor(self):
        """Test module data processor."""
        processor = resources_module.get_data_processor()
        assert processor is not None

        # Test with mock data
        mock_data = pd.DataFrame({
            'step': range(5),
            'total_resources': range(1000, 995, -1),
        })

        result = processor.process(mock_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
