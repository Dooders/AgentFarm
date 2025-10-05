"""
Comprehensive tests for agents analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.agents import (
    agents_module,
    compute_lifespan_statistics,
    compute_behavior_patterns,
    compute_performance_metrics,
    compute_learning_curves,
    analyze_lifespan_patterns,
    analyze_behavior_clustering,
    analyze_performance_analysis,
    analyze_learning_curves,
    analyze_agent_lifespans,
    plot_lifespan_distributions,
    plot_behavior_clusters,
    plot_performance_metrics,
    plot_learning_curves,
    plot_lifespan_histogram,
    cluster_agent_behaviors,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def sample_agent_data():
    """Create sample agent data."""
    np.random.seed(42)
    return pd.DataFrame({
        'agent_id': [f'agent_{i}' for i in range(50)],
        'lifespan': np.random.normal(75, 15, 50),
        'death_time': [np.random.choice([None, i * 10]) for i in range(50)],
        'birth_time': [i * 2 for i in range(50)],
        'total_actions': np.random.randint(50, 500, 50),
        'successful_actions': np.random.randint(30, 400, 50),
        'total_rewards': np.random.uniform(100, 1000, 50),
        'agent_type': np.random.choice(['system', 'independent', 'control'], 50),
        'exploration_rate': np.random.uniform(0.1, 0.5, 50),
        'action_frequency': np.random.uniform(5, 20, 50),
    })


@pytest.fixture
def sample_behavior_data():
    """Create sample behavior data."""
    return pd.DataFrame({
        'agent_id': range(30),
        'action_frequency': np.random.normal(10, 2, 30),
        'success_rate': np.random.normal(0.8, 0.1, 30),
        'exploration_rate': np.random.normal(0.3, 0.1, 30),
        'avg_reward': np.random.normal(1.5, 0.3, 30),
    })


@pytest.fixture
def sample_learning_data():
    """Create sample learning data."""
    return pd.DataFrame({
        'agent_id': ['agent_1'] * 100,
        'step': range(100),
        'reward': np.cumsum(np.random.randn(100)) + 50,
        'success_rate': np.linspace(0.5, 0.9, 100),
    })


class TestAgentComputations:
    """Test agent statistical computations."""

    def test_compute_lifespan_statistics(self, sample_agent_data):
        """Test lifespan statistics computation."""
        stats = compute_lifespan_statistics(sample_agent_data)
        
        assert isinstance(stats, dict)
        assert 'lifespan' in stats
        assert 'total_agents' in stats
        assert 'survival_rate' in stats
        assert 'mortality_rate' in stats
        assert 'agent_type_distribution' in stats
        assert 'lifespan_by_type' in stats
        
        assert stats['total_agents'] == 50
        assert 'mean' in stats['lifespan']

    def test_compute_lifespan_statistics_empty(self):
        """Test lifespan statistics with empty DataFrame."""
        result = compute_lifespan_statistics(pd.DataFrame())
        
        assert result == {}

    def test_compute_lifespan_statistics_no_lifespan(self):
        """Test lifespan statistics without lifespan column."""
        df = pd.DataFrame({'agent_id': [1, 2, 3]})
        result = compute_lifespan_statistics(df)
        
        assert result == {}

    def test_compute_lifespan_by_type(self, sample_agent_data):
        """Test lifespan statistics by agent type."""
        stats = compute_lifespan_statistics(sample_agent_data)
        
        assert 'lifespan_by_type' in stats
        for agent_type in ['system', 'independent', 'control']:
            if agent_type in stats['lifespan_by_type']:
                type_stats = stats['lifespan_by_type'][agent_type]
                assert 'mean' in type_stats
                assert 'std' in type_stats

    def test_compute_behavior_patterns(self, sample_agent_data):
        """Test behavior pattern computation."""
        patterns = compute_behavior_patterns(sample_agent_data)
        
        assert isinstance(patterns, dict)
        assert 'mean_lifespan' in patterns
        assert 'median_lifespan' in patterns
        assert 'lifespan_std' in patterns
        assert 'max_lifespan' in patterns
        assert 'min_lifespan' in patterns
        assert 'survival_curve' in patterns

    def test_compute_behavior_patterns_empty(self):
        """Test behavior patterns with empty DataFrame."""
        result = compute_behavior_patterns(pd.DataFrame())
        
        assert result == {}

    def test_compute_behavior_patterns_with_rewards(self):
        """Test behavior patterns with reward data."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': np.random.uniform(50, 100, 10),
            'avg_reward': np.random.uniform(0.5, 2.0, 10),
        })
        
        patterns = compute_behavior_patterns(df)
        
        assert 'reward_distribution' in patterns

    def test_compute_performance_metrics(self, sample_behavior_data):
        """Test performance metrics computation."""
        metrics = compute_performance_metrics(sample_behavior_data)
        
        assert isinstance(metrics, dict)
        assert 'cluster_labels' in metrics
        assert 'cluster_centers' in metrics
        assert 'cluster_sizes' in metrics
        assert 'silhouette_score' in metrics
        
        if metrics['cluster_labels']:
            assert len(metrics['cluster_labels']) == 30
            assert len(metrics['cluster_centers']) == 3
            assert sum(metrics['cluster_sizes']) == 30

    def test_compute_performance_metrics_empty(self):
        """Test performance metrics with empty DataFrame."""
        result = compute_performance_metrics(pd.DataFrame())
        
        assert result == {}

    def test_compute_performance_metrics_insufficient_features(self):
        """Test performance metrics with insufficient features."""
        df = pd.DataFrame({'agent_id': range(10)})
        
        result = compute_performance_metrics(df)
        
        # Should return empty or minimal metrics
        assert isinstance(result, dict)

    def test_compute_performance_metrics_composite_score(self):
        """Test performance metrics with composite scoring."""
        df = pd.DataFrame({
            'agent_id': range(20),
            'successful_actions': np.random.randint(10, 100, 20),
            'total_actions': np.random.randint(50, 200, 20),
            'total_rewards': np.random.uniform(50, 500, 20),
            'lifespan': np.random.uniform(40, 120, 20),
        })
        
        metrics = compute_performance_metrics(df)
        
        if 'performance_score' in metrics:
            assert 'mean' in metrics['performance_score']
            assert 'top_performers' in metrics

    def test_compute_learning_curves(self, sample_learning_data):
        """Test learning curve computation."""
        curves = compute_learning_curves(sample_learning_data)
        
        assert isinstance(curves, dict)

    def test_survival_rate_calculation(self):
        """Test survival and mortality rate calculation."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50] * 10,
            'death_time': [None] * 5 + [50] * 5,  # 5 alive, 5 dead
        })
        
        stats = compute_lifespan_statistics(df)
        
        assert stats['survival_rate'] == 0.5
        assert stats['mortality_rate'] == 0.5


class TestAgentAnalysis:
    """Test agent analysis functions."""

    def test_analyze_lifespan_patterns(self, tmp_path, sample_agent_data):
        """Test lifespan pattern analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_lifespan_patterns(sample_agent_data, ctx)
        
        output_file = tmp_path / "lifespan_statistics.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'lifespan' in data
        assert 'total_agents' in data

    def test_analyze_behavior_clustering(self, tmp_path, sample_agent_data):
        """Test behavior clustering analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_behavior_clustering(sample_agent_data, ctx)
        
        output_file = tmp_path / "behavior_patterns.json"
        assert output_file.exists()

    def test_analyze_performance_analysis(self, tmp_path, sample_agent_data):
        """Test performance analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_performance_analysis(sample_agent_data, ctx)
        
        output_file = tmp_path / "performance_analysis.json"
        assert output_file.exists()

    def test_analyze_learning_curves(self, tmp_path, sample_learning_data):
        """Test learning curves analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_learning_curves(sample_learning_data, ctx)
        
        output_file = tmp_path / "learning_curves.json"
        assert output_file.exists()

    def test_analyze_agent_lifespans(self, tmp_path, sample_agent_data):
        """Test detailed agent lifespans analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_agent_lifespans(sample_agent_data, ctx)
        
        output_file = tmp_path / "detailed_lifespan_stats.json"
        assert output_file.exists()

    def test_cluster_agent_behaviors(self, tmp_path, sample_behavior_data):
        """Test agent behavior clustering."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        cluster_agent_behaviors(sample_behavior_data, ctx)
        
        # Should create output file
        output_files = list(tmp_path.glob("*.json"))
        assert len(output_files) > 0


class TestAgentVisualization:
    """Test agent visualization functions."""

    def test_plot_lifespan_distributions(self, tmp_path, sample_agent_data):
        """Test lifespan distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_lifespan_distributions(sample_agent_data, ctx)
        
        plot_file = tmp_path / "lifespan_distributions.png"
        assert plot_file.exists()

    def test_plot_lifespan_distributions_no_type(self, tmp_path):
        """Test lifespan plotting without agent type."""
        df = pd.DataFrame({
            'agent_id': range(20),
            'lifespan': np.random.uniform(50, 100, 20),
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_lifespan_distributions(df, ctx)
        
        plot_file = tmp_path / "lifespan_distributions.png"
        assert plot_file.exists()

    def test_plot_lifespan_distributions_empty(self, tmp_path):
        """Test lifespan plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_lifespan_distributions(pd.DataFrame(), ctx)
        
        # Should handle gracefully, may not create file

    def test_plot_behavior_clusters(self, tmp_path):
        """Test behavior cluster plotting."""
        df = pd.DataFrame({
            'agent_id': range(25),
            'successful_actions': np.random.randint(10, 100, 25),
            'total_actions': np.random.randint(50, 200, 25),
            'total_rewards': np.random.uniform(50, 500, 25),
            'lifespan': np.random.uniform(40, 120, 25),
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_behavior_clusters(df, ctx)
        
        plot_file = tmp_path / "behavior_clusters.png"
        assert plot_file.exists()

    def test_plot_behavior_clusters_empty(self, tmp_path):
        """Test behavior cluster plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_behavior_clusters(pd.DataFrame(), ctx)
        
        # Should handle gracefully

    def test_plot_performance_metrics(self, tmp_path, sample_agent_data):
        """Test performance metrics plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_performance_metrics(sample_agent_data, ctx)
        
        plot_file = tmp_path / "performance_metrics.png"
        assert plot_file.exists()

    def test_plot_learning_curves(self, tmp_path, sample_learning_data):
        """Test learning curves plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_learning_curves(sample_learning_data, ctx)
        
        # Should create output
        plot_files = list(tmp_path.glob("*.png"))
        assert len(plot_files) > 0

    def test_plot_lifespan_histogram(self, tmp_path, sample_agent_data):
        """Test lifespan histogram plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_lifespan_histogram(sample_agent_data, ctx)
        
        # Should create output
        plot_files = list(tmp_path.glob("*.png"))
        assert len(plot_files) > 0

    def test_plot_with_custom_options(self, tmp_path, sample_agent_data):
        """Test plotting with custom options."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_lifespan_distributions(sample_agent_data, ctx, figsize=(10, 8), dpi=150)
        
        plot_file = tmp_path / "lifespan_distributions.png"
        assert plot_file.exists()


class TestAgentsModule:
    """Test agents module integration."""

    def test_agents_module_registration(self):
        """Test module registration."""
        assert agents_module.name == "agents"
        assert agents_module.description == "Analysis of individual agent behavior, lifespan, performance, and learning patterns"

    def test_agents_module_function_names(self):
        """Test module function names."""
        functions = agents_module.get_function_names()
        
        assert "analyze_behaviors" in functions
        assert "plot_behaviors" in functions
        assert "analyze_lifespans" in functions
        assert "plot_lifespans" in functions
        assert "analyze_performance" in functions

    def test_agents_module_function_groups(self):
        """Test module function groups."""
        groups = agents_module.get_function_groups()
        
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "lifespan" in groups
        assert "behavior" in groups
        assert "basic" in groups

    def test_agents_module_data_processor(self):
        """Test module data processor."""
        processor = agents_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = agents_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = agents_module.get_functions()
        assert len(functions) >= 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_lifespan_with_nan_values(self):
        """Test lifespan computation with NaN values."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, np.nan, 75, np.nan, 100, 60, np.nan, 80, 90, 70],
            'death_time': [50, None, 75, None, 100, 60, None, 80, 90, 70],
            'agent_type': ['system'] * 10,
        })
        
        stats = compute_lifespan_statistics(df)
        
        # Should handle NaN values gracefully
        assert 'lifespan' in stats

    def test_compute_behavior_with_negative_values(self):
        """Test behavior patterns with negative values."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, -10, 75, 80, 100, 60, 55, 80, 90, 70],  # One negative
        })
        
        patterns = compute_behavior_patterns(df)
        
        # Should handle negative values
        assert isinstance(patterns, dict)

    def test_compute_performance_single_agent(self):
        """Test performance metrics with single agent."""
        df = pd.DataFrame({
            'agent_id': ['agent_1'],
            'action_frequency': [10.0],
            'success_rate': [0.8],
            'exploration_rate': [0.3],
            'avg_reward': [1.5],
        })
        
        metrics = compute_performance_metrics(df)
        
        # Should handle single agent (can't cluster)
        assert isinstance(metrics, dict)

    def test_compute_performance_two_agents(self):
        """Test performance metrics with two agents."""
        df = pd.DataFrame({
            'agent_id': ['agent_1', 'agent_2'],
            'action_frequency': [10.0, 12.0],
            'success_rate': [0.8, 0.85],
            'exploration_rate': [0.3, 0.35],
            'avg_reward': [1.5, 1.6],
        })
        
        metrics = compute_performance_metrics(df)
        
        # Should handle two agents (insufficient for 3 clusters)
        assert isinstance(metrics, dict)

    def test_analyze_with_missing_columns(self, tmp_path):
        """Test analysis with missing required columns."""
        df = pd.DataFrame({'agent_id': range(10)})
        ctx = AnalysisContext(output_path=tmp_path)
        
        # Should handle missing columns gracefully
        analyze_lifespan_patterns(df, ctx)
        
        # May create empty or minimal output

    def test_plot_with_zero_lifespan(self, tmp_path):
        """Test plotting with zero lifespan agents."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [0, 0, 50, 60, 0, 70, 80, 0, 90, 100],
            'agent_type': ['system'] * 10,
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_lifespan_distributions(df, ctx)

    def test_compute_lifespan_all_dead(self):
        """Test lifespan statistics with all agents dead."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, 75, 100, 25, 80, 60, 90, 40, 120, 30],
            'death_time': [50, 75, 100, 25, 80, 60, 90, 40, 120, 30],
        })
        
        stats = compute_lifespan_statistics(df)
        
        assert stats['survival_rate'] == 0.0
        assert stats['mortality_rate'] == 1.0

    def test_compute_lifespan_all_alive(self):
        """Test lifespan statistics with all agents alive."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [50, 75, 100, 25, 80, 60, 90, 40, 120, 30],
            'death_time': [None] * 10,
        })
        
        stats = compute_lifespan_statistics(df)
        
        assert stats['survival_rate'] == 1.0
        assert stats['mortality_rate'] == 0.0

    def test_behavior_patterns_extreme_values(self):
        """Test behavior patterns with extreme values."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'lifespan': [1, 2, 1000, 5, 10, 2000, 15, 20, 3000, 25],
        })
        
        patterns = compute_behavior_patterns(df)
        
        # Should handle extreme values
        assert 'mean_lifespan' in patterns
        assert patterns['mean_lifespan'] > 0

    def test_performance_metrics_with_zero_actions(self):
        """Test performance metrics with zero actions."""
        df = pd.DataFrame({
            'agent_id': range(10),
            'successful_actions': [0] * 10,
            'total_actions': [0] * 10,
            'total_rewards': [0] * 10,
            'lifespan': [50] * 10,
        })
        
        metrics = compute_performance_metrics(df)
        
        # Should handle zero actions
        assert isinstance(metrics, dict)

    def test_survival_curve_single_agent(self):
        """Test survival curve with single agent."""
        df = pd.DataFrame({
            'agent_id': ['agent_1'],
            'lifespan': [100.0],
        })
        
        patterns = compute_behavior_patterns(df)
        
        assert 'survival_curve' in patterns
        assert len(patterns['survival_curve']) == 1

    def test_analyze_with_progress_callback(self, tmp_path, sample_agent_data):
        """Test analysis with progress reporting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        # Mock progress callback
        progress_calls = []
        ctx.report_progress = lambda msg, prog: progress_calls.append((msg, prog))
        
        analyze_lifespan_patterns(sample_agent_data, ctx)
        
        # Should have called progress
        assert len(progress_calls) > 0


class TestAgentHelperFunctions:
    """Test helper functions in agents module."""

    def test_process_agent_data(self, tmp_path):
        """Test processing agent data from experiment."""
        from farm.analysis.agents.data import process_agent_data
        
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        
        # Create mock database
        db_path = exp_path / "simulation.db"
        db_path.touch()
        
        with patch('farm.analysis.agents.data.SessionManager') as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.all.return_value = []
            
            result = process_agent_data(exp_path)
            
            assert isinstance(result, pd.DataFrame)

    def test_compute_statistics_helper(self):
        """Test statistics calculation helper."""
        from farm.analysis.common.utils import calculate_statistics
        
        data = np.array([10, 20, 30, 40, 50])
        stats = calculate_statistics(data)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert stats['mean'] == 30.0

    def test_agent_type_distribution(self, sample_agent_data):
        """Test agent type distribution calculation."""
        stats = compute_lifespan_statistics(sample_agent_data)
        
        assert 'agent_type_distribution' in stats
        total = sum(stats['agent_type_distribution'].values())
        assert total == len(sample_agent_data)
