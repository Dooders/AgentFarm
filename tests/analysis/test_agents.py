"""
Tests for agents analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from farm.analysis.agents import (
    agents_module,
    compute_agent_statistics,
    compute_lifespan_metrics,
    compute_behavior_clusters,
    analyze_agent_patterns,
    analyze_lifespan,
    analyze_behavior,
    plot_agent_statistics,
    plot_lifespan_distribution,
    plot_behavior_clusters,
)
from farm.analysis.common.context import AnalysisContext


class TestAgentComputations:
    """Test agent statistical computations."""

    def test_compute_agent_statistics(self):
        """Test agent statistics computation."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, 75, 100, 25, 80, 60, 90, 40, 120, 30],
            'total_actions': [150, 200, 300, 80, 250, 180, 280, 120, 400, 90],
            'success_rate': [0.8, 0.85, 0.9, 0.7, 0.88, 0.82, 0.92, 0.75, 0.95, 0.72],
            'avg_reward': [1.2, 1.5, 1.8, 0.9, 1.6, 1.3, 1.9, 1.0, 2.0, 0.95],
            'agent_type': ['system', 'independent', 'system', 'control', 'independent',
                         'system', 'independent', 'control', 'system', 'control'],
        })

        stats = compute_agent_statistics(df)

        assert 'total_agents' in stats
        assert 'avg_lifespan' in stats
        assert 'avg_success_rate' in stats
        assert 'by_type' in stats

        assert stats['total_agents'] == 10
        assert abs(stats['avg_lifespan'] - 67.0) < 0.1  # mean of lifespans
        assert abs(stats['avg_success_rate'] - 0.829) < 0.01  # mean success rate

        # Check type breakdown
        assert 'system' in stats['by_type']
        assert 'independent' in stats['by_type']
        assert 'control' in stats['by_type']

    def test_compute_lifespan_metrics(self):
        """Test lifespan metrics computation."""
        df = pd.DataFrame({
            'agent_id': range(20),
            'lifespan': np.random.normal(75, 15, 20),  # Normal distribution
            'agent_type': np.random.choice(['system', 'independent', 'control'], 20),
        })

        metrics = compute_lifespan_metrics(df)

        assert 'mean_lifespan' in metrics
        assert 'median_lifespan' in metrics
        assert 'lifespan_std' in metrics
        assert 'max_lifespan' in metrics
        assert 'min_lifespan' in metrics
        assert 'survival_curve' in metrics

        assert metrics['mean_lifespan'] > 0
        assert metrics['median_lifespan'] > 0
        assert metrics['lifespan_std'] >= 0
        assert len(metrics['survival_curve']) > 0

    def test_compute_behavior_clusters(self):
        """Test behavior clustering computation."""
        # Create agents with different behavior patterns
        np.random.seed(42)
        df = pd.DataFrame({
            'agent_id': range(30),
            'action_frequency': np.random.normal(10, 2, 30),
            'success_rate': np.random.normal(0.8, 0.1, 30),
            'exploration_rate': np.random.normal(0.3, 0.1, 30),
            'avg_reward': np.random.normal(1.5, 0.3, 30),
        })

        clusters = compute_behavior_clusters(df, n_clusters=3)

        assert 'cluster_labels' in clusters
        assert 'cluster_centers' in clusters
        assert 'cluster_sizes' in clusters
        assert 'silhouette_score' in clusters

        assert len(clusters['cluster_labels']) == 30
        assert len(clusters['cluster_centers']) == 3
        assert len(clusters['cluster_sizes']) == 3
        assert sum(clusters['cluster_sizes']) == 30


class TestAgentAnalysis:
    """Test agent analysis functions."""

    def test_analyze_agent_patterns(self, tmp_path):
        """Test agent pattern analysis."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, 75, 100, 25, 80, 60, 90, 40, 120, 30],
            'total_actions': [150, 200, 300, 80, 250, 180, 280, 120, 400, 90],
            'success_rate': [0.8, 0.85, 0.9, 0.7, 0.88, 0.82, 0.92, 0.75, 0.95, 0.72],
            'avg_reward': [1.2, 1.5, 1.8, 0.9, 1.6, 1.3, 1.9, 1.0, 2.0, 0.95],
            'agent_type': ['system', 'independent', 'system', 'control', 'independent',
                         'system', 'independent', 'control', 'system', 'control'],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_agent_patterns(df, ctx)

        stats_file = tmp_path / "agent_patterns.json"
        assert stats_file.exists()

        with open(stats_file) as f:
            data = json.load(f)

        assert 'statistics' in data
        assert 'type_breakdown' in data

    def test_analyze_lifespan(self, tmp_path):
        """Test lifespan analysis."""
        df = pd.DataFrame({
            'agent_id': range(15),
            'lifespan': np.random.normal(75, 15, 15),
            'agent_type': np.random.choice(['system', 'independent', 'control'], 15),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_lifespan(df, ctx)

        lifespan_file = tmp_path / "lifespan_analysis.csv"
        assert lifespan_file.exists()

        lifespan_df = pd.read_csv(lifespan_file)
        assert len(lifespan_df) == 15
        assert 'lifespan' in lifespan_df.columns
        assert 'agent_type' in lifespan_df.columns

    def test_analyze_behavior(self, tmp_path):
        """Test behavior analysis."""
        np.random.seed(42)
        df = pd.DataFrame({
            'agent_id': range(20),
            'action_frequency': np.random.normal(10, 2, 20),
            'success_rate': np.random.normal(0.8, 0.1, 20),
            'exploration_rate': np.random.normal(0.3, 0.1, 20),
            'avg_reward': np.random.normal(1.5, 0.3, 20),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_behavior(df, ctx)

        behavior_file = tmp_path / "behavior_analysis.json"
        assert behavior_file.exists()

        with open(behavior_file) as f:
            data = json.load(f)

        assert 'clusters' in data
        assert 'behavior_patterns' in data


class TestAgentVisualization:
    """Test agent visualization functions."""

    def test_plot_agent_statistics(self, tmp_path):
        """Test agent statistics plotting."""
        df = pd.DataFrame({
            'agent_type': ['system'] * 5 + ['independent'] * 4 + ['control'] * 3,
            'lifespan': [80, 85, 90, 75, 82, 70, 72, 68, 65, 60, 58, 62],
            'success_rate': [0.9, 0.88, 0.92, 0.85, 0.87, 0.8, 0.82, 0.78, 0.75, 0.7, 0.72, 0.68],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_agent_statistics(df, ctx)

        plot_file = tmp_path / "agent_statistics.png"
        assert plot_file.exists()

    def test_plot_lifespan_distribution(self, tmp_path):
        """Test lifespan distribution plotting."""
        df = pd.DataFrame({
            'agent_id': range(20),
            'lifespan': np.random.normal(75, 15, 20),
            'agent_type': np.random.choice(['system', 'independent', 'control'], 20),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_lifespan_distribution(df, ctx)

        plot_file = tmp_path / "lifespan_distribution.png"
        assert plot_file.exists()

    def test_plot_behavior_clusters(self, tmp_path):
        """Test behavior cluster plotting."""
        np.random.seed(42)
        df = pd.DataFrame({
            'agent_id': range(25),
            'action_frequency': np.random.normal(10, 2, 25),
            'success_rate': np.random.normal(0.8, 0.1, 25),
            'exploration_rate': np.random.normal(0.3, 0.1, 25),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_behavior_clusters(df, ctx)

        plot_file = tmp_path / "behavior_clusters.png"
        assert plot_file.exists()


class TestAgentsModule:
    """Test agents module integration."""

    def test_agents_module_registration(self):
        """Test module is properly registered."""
        assert agents_module.name == "agents"
        assert len(agents_module.get_function_names()) > 0
        assert "analyze_patterns" in agents_module.get_function_names()
        assert "plot_statistics" in agents_module.get_function_names()

    def test_agents_module_function_groups(self):
        """Test module function groups."""
        groups = agents_module.get_group_names()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_agents_module_data_processor(self):
        """Test module data processor."""
        processor = agents_module.get_data_processor()
        assert processor is not None

        # Test with mock data
        mock_data = pd.DataFrame({
            'agent_id': range(5),
            'lifespan': [50, 60, 70, 80, 90],
        })

        result = processor.process(mock_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
