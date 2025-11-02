"""
Comprehensive tests for genesis analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.genesis import (
    genesis_module,
    compute_initial_state_metrics,
    compute_genesis_impact_scores,
    compute_critical_period_metrics,
    analyze_genesis_factors,
    analyze_genesis_across_simulations,
    analyze_critical_period,
    analyze_genesis_patterns,
    plot_genesis_analysis_results,
    plot_initial_state_comparison,
    plot_critical_period_analysis,
    plot_genesis_patterns,
    plot_genesis_timeline,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def sample_genesis_data():
    """Create sample genesis data."""
    return pd.DataFrame(
        {
            "iteration": range(20),
            "agent_id": [f"agent_{i}" for i in range(20)],
            "genesis_time": np.random.uniform(0, 1000, 20),
            "parent_id": [f"parent_{i % 5}" for i in range(20)],
            "success_rate": np.random.uniform(0, 1, 20),
            "efficiency_score": np.random.uniform(0, 1, 20),
            "resource_cost": np.random.uniform(10, 100, 20),
            "survival_time": np.random.uniform(100, 1000, 20),
        }
    )


@pytest.fixture
def mock_agents():
    """Create mock agent data."""
    agents = []
    for i in range(10):
        agent = {
            "agent_id": i,
            "agent_type": "SystemAgent" if i < 5 else "IndependentAgent",
            "position_x": i * 10.0,
            "position_y": i * 10.0,
            "initial_resources": 100.0,
            "starting_health": 100.0,
            "starvation_counter": 0,
            "action_weights": {"move": 0.3, "gather": 0.5, "attack": 0.2},
        }
        agents.append(agent)
    return agents


@pytest.fixture
def mock_resources():
    """Create mock resource data."""
    resources = []
    for i in range(5):
        resource = {
            "amount": 50.0,
            "position_x": i * 20.0,
            "position_y": i * 20.0,
        }
        resources.append(resource)
    return resources


class TestGenesisComputations:
    """Test genesis statistical computations."""

    @patch("farm.analysis.genesis.compute.compute_agent_resource_proximity")
    @patch("farm.analysis.genesis.compute.compute_agent_agent_proximity")
    @patch("farm.analysis.genesis.compute.compute_agent_starting_attributes")
    @patch("farm.analysis.genesis.compute.compute_initial_relative_advantages")
    def test_compute_initial_state_metrics(
        self,
        mock_advantages,
        mock_attributes,
        mock_agent_proximity,
        mock_resource_proximity,
        mock_session,
        mock_agents,
        mock_resources,
    ):
        """Test initial state metrics computation."""
        # Mock query results
        mock_session.query.return_value.filter.return_value.all.return_value = [
            (
                a["agent_id"],
                a["agent_type"],
                a["position_x"],
                a["position_y"],
                a["initial_resources"],
                a["starting_health"],
                a["starvation_counter"],
                a["action_weights"],
            )
            for a in mock_agents
        ]

        # Mock resource query
        resource_query = MagicMock()
        resource_query.filter.return_value.all.return_value = [
            (r["amount"], r["position_x"], r["position_y"]) for r in mock_resources
        ]
        mock_session.query.side_effect = [
            mock_session.query.return_value,  # agents query
            resource_query,  # resources query
        ]

        # Mock helper functions
        mock_resource_proximity.return_value = {
            "agent_resource_proximity": {},
            "agent_type_resource_proximity": {},
        }
        mock_agent_proximity.return_value = {
            "agent_agent_proximity": {},
            "agent_type_proximity": {},
        }
        mock_attributes.return_value = {"agent_starting_attributes": {}}
        mock_advantages.return_value = {"initial_relative_advantages": {}}

        metrics = compute_initial_state_metrics(mock_session)

        assert "initial_agent_count" in metrics
        assert "initial_resource_count" in metrics
        assert "agent_type_distribution" in metrics
        assert metrics["initial_agent_count"] == 10
        assert metrics["initial_resource_count"] == 5

    def test_compute_initial_state_metrics_empty(self, mock_session):
        """Test initial state metrics with no agents."""
        # Mock empty queries
        mock_session.query.return_value.filter.return_value.all.return_value = []

        metrics = compute_initial_state_metrics(mock_session)

        assert metrics["initial_agent_count"] == 0
        assert metrics["initial_resource_count"] == 0

    @patch("farm.analysis.genesis.compute.compute_initial_state_metrics")
    @patch("farm.analysis.genesis.compute.compute_simulation_outcomes")
    def test_compute_genesis_impact_scores(self, mock_outcomes, mock_initial, mock_session):
        """Test genesis impact scores computation."""
        mock_initial.return_value = {
            "initial_agent_count": 10,
            "initial_resource_count": 5,
            "avg_resource_amount": 50.0,
        }
        mock_outcomes.return_value = {
            "final_total_agents": 15,
            "survival_rate": 0.8,
        }

        scores = compute_genesis_impact_scores(mock_session)

        assert "outcome_specific_impacts" in scores
        assert "overall_impact_scores" in scores

    @patch("farm.analysis.genesis.compute.compute_initial_state_metrics")
    def test_compute_critical_period_metrics(self, mock_initial, mock_session):
        """Test critical period metrics computation."""
        # Mock initial metrics
        mock_initial.return_value = {
            "agent_type_distribution": {
                "SystemAgent": 5,
                "IndependentAgent": 5,
            }
        }

        # Mock simulation steps
        mock_steps = []
        for i in range(10):
            step = MagicMock()
            step.step_number = i
            step.as_dict.return_value = {
                "system_agents": 5,
                "independent_agents": 5,
                "control_agents": 0,
                "total_agents": 10,
            }
            mock_steps.append(step)

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_steps
        mock_session.query.return_value.filter.return_value.all.return_value = []

        metrics = compute_critical_period_metrics(mock_session, critical_period_end=10)

        assert "survival_rate" in metrics
        assert "reproduction_rate" in metrics
        assert "resource_efficiency" in metrics

    def test_compute_critical_period_metrics_no_steps(self, mock_session):
        """Test critical period with no steps."""
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        metrics = compute_critical_period_metrics(mock_session)

        assert "error" in metrics


class TestGenesisAnalysis:
    """Test genesis analysis functions."""

    def test_analyze_genesis_patterns(self, sample_genesis_data):
        """Test genesis pattern analysis."""
        result = analyze_genesis_patterns(sample_genesis_data)

        assert isinstance(result, dict)
        assert "total_genesis_events" in result
        assert "avg_success_rate" in result
        assert "avg_efficiency" in result
        assert result["total_genesis_events"] == 20

    def test_analyze_genesis_patterns_empty(self):
        """Test genesis patterns with empty DataFrame."""
        result = analyze_genesis_patterns(pd.DataFrame())

        assert result["total_genesis_events"] == 0

    @patch("farm.analysis.genesis.analyze.compute_initial_state_metrics")
    def test_analyze_genesis_factors(self, mock_compute, tmp_path):
        """Test genesis factors analysis."""
        mock_session = MagicMock()
        mock_compute.return_value = {
            "initial_agent_count": 10,
            "agent_type_distribution": {"SystemAgent": 5, "IndependentAgent": 5},
        }

        # Test the function directly with session parameter
        result = analyze_genesis_factors(mock_session)

        # Check that the function returns expected structure
        assert isinstance(result, dict)
        assert "initial_metrics" in result

    @patch("farm.analysis.genesis.analyze.glob.glob")
    def test_analyze_genesis_across_simulations(self, mock_glob, tmp_path):
        """Test genesis analysis across simulations."""
        # Mock multiple simulation paths
        sim_paths = [str(tmp_path / f"iteration_{i}") for i in range(3)]
        mock_glob.return_value = sim_paths

        for path in sim_paths:
            Path(path).mkdir(parents=True, exist_ok=True)
            # Create a mock database file
            (Path(path) / "simulation.db").touch()

        # Test the function directly with experiment path
        result = analyze_genesis_across_simulations(str(tmp_path))

        # Check that the function returns expected structure
        assert isinstance(result, dict)
        assert "simulations" in result

    @patch("farm.analysis.genesis.analyze.compute_critical_period_metrics")
    def test_analyze_critical_period(self, mock_compute, tmp_path):
        """Test critical period analysis."""
        mock_session = MagicMock()
        mock_compute.return_value = {
            "survival_rate": 0.8,
            "reproduction_rate": 0.3,
            "resource_efficiency": 0.7,
        }

        # Test the function directly with session parameter
        result = analyze_critical_period(mock_session)

        # Check that the function returns expected structure
        assert isinstance(result, dict)
        assert "metrics" in result


class TestGenesisVisualization:
    """Test genesis visualization functions."""

    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_genesis_patterns(self, mock_plt, sample_genesis_data, tmp_path):
        """Test genesis patterns plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_genesis_patterns(sample_genesis_data, ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_genesis_patterns_empty(self, mock_plt, tmp_path):
        """Test genesis patterns plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_genesis_patterns(pd.DataFrame(), ctx)

        # Should handle gracefully
        assert result is None

    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_genesis_timeline(self, mock_plt, sample_genesis_data, tmp_path):
        """Test genesis timeline plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_genesis_timeline(sample_genesis_data, ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.genesis.plot.sns")
    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_initial_state_comparison(self, mock_plt, mock_sns, tmp_path):
        """Test initial state comparison plotting."""
        # Create proper simulation data structure
        simulations = [
            {
                "iteration": "0",
                "results": {
                    "initial_metrics": {
                        "agent_starting_attributes": {
                            "SystemAgent": {"count": 5, "avg_initial_resources": 100.0},
                            "IndependentAgent": {"count": 5, "avg_initial_resources": 80.0},
                        }
                    }
                },
            }
        ]

        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_initial_state_comparison(simulations, ctx=ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.genesis.plot.sns")
    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_critical_period_analysis(self, mock_plt, mock_sns, tmp_path):
        """Test critical period analysis plotting."""
        # Create proper simulation data structure
        simulations = [
            {
                "iteration": "0",
                "results": {
                    "critical_period": {
                        "survival_rate": 0.8,
                        "reproduction_rate": 0.3,
                        "resource_efficiency": 0.7,
                        "SystemAgent_growth_rate": 0.2,
                        "IndependentAgent_growth_rate": 0.1,
                        "early_deaths": {"SystemAgent": 1, "IndependentAgent": 2},
                    }
                },
            }
        ]

        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_critical_period_analysis(simulations, ctx=ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.genesis.plot.sns")
    @patch("farm.analysis.genesis.plot.plt")
    def test_plot_genesis_analysis_results(self, mock_plt, mock_sns, tmp_path):
        """Test genesis analysis results plotting."""
        results = {
            "simulations": [
                {
                    "iteration": "0",
                    "results": {
                        "initial_metrics": {
                            "agent_starting_attributes": {
                                "SystemAgent": {"count": 5, "avg_initial_resources": 100.0},
                                "IndependentAgent": {"count": 5, "avg_initial_resources": 80.0},
                            }
                        },
                        "critical_period": {
                            "survival_rate": 0.8,
                            "reproduction_rate": 0.3,
                            "resource_efficiency": 0.7,
                        },
                    },
                }
            ]
        }

        ctx = AnalysisContext(output_path=tmp_path)
        # Fix parameter order: results, output_path, ctx
        result = plot_genesis_analysis_results(results, str(tmp_path), ctx=ctx)

        assert result is None


class TestGenesisModule:
    """Test genesis module integration."""

    def test_genesis_module_registration(self):
        """Test module registration."""
        assert genesis_module.name == "genesis"
        assert (
            genesis_module.description
            == "Analysis of initial conditions and their impact on dominance patterns and simulation outcomes"
        )

    def test_genesis_module_function_names(self):
        """Test module function names."""
        functions = genesis_module.get_function_names()
        expected_functions = [
            "analyze_factors",
            "analyze_across_simulations",
            "analyze_critical_period",
            "analyze_genesis_patterns",
            "plot_results",
            "plot_initial_comparison",
            "plot_critical_period",
            "plot_genesis_patterns",
            "plot_genesis_timeline",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_genesis_module_function_groups(self):
        """Test module function groups."""
        groups = genesis_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_genesis_module_data_processor(self):
        """Test module data processor."""
        processor = genesis_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = genesis_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = genesis_module.get_functions()
        assert len(functions) >= 9


class TestHelperFunctions:
    """Test helper computation functions."""

    def test_compute_agent_resource_proximity(self, mock_agents, mock_resources):
        """Test agent-resource proximity computation."""
        from farm.analysis.genesis.compute import compute_agent_resource_proximity

        metrics = compute_agent_resource_proximity(mock_agents, mock_resources)

        assert "agent_resource_proximity" in metrics
        assert "agent_type_resource_proximity" in metrics

    def test_compute_agent_resource_proximity_empty(self):
        """Test proximity with no agents or resources."""
        from farm.analysis.genesis.compute import compute_agent_resource_proximity

        metrics = compute_agent_resource_proximity([], [])

        assert metrics["agent_resource_proximity"] == {}

    def test_compute_agent_agent_proximity(self, mock_agents):
        """Test agent-agent proximity computation."""
        from farm.analysis.genesis.compute import compute_agent_agent_proximity

        metrics = compute_agent_agent_proximity(mock_agents)

        assert "agent_agent_proximity" in metrics
        assert "agent_type_proximity" in metrics

    def test_compute_agent_agent_proximity_single_agent(self):
        """Test proximity with single agent."""
        from farm.analysis.genesis.compute import compute_agent_agent_proximity

        single_agent = [
            {
                "agent_id": 1,
                "agent_type": "SystemAgent",
                "position_x": 10.0,
                "position_y": 10.0,
            }
        ]

        metrics = compute_agent_agent_proximity(single_agent)

        assert metrics["agent_agent_proximity"] == {}

    def test_compute_agent_starting_attributes(self, mock_agents):
        """Test agent starting attributes computation."""
        from farm.analysis.genesis.compute import compute_agent_starting_attributes

        metrics = compute_agent_starting_attributes(mock_agents)

        assert "agent_starting_attributes" in metrics
        assert "SystemAgent" in metrics["agent_starting_attributes"]
        assert "IndependentAgent" in metrics["agent_starting_attributes"]

    def test_compute_initial_relative_advantages(self, mock_agents, mock_resources):
        """Test initial relative advantages computation."""
        from farm.analysis.genesis.compute import compute_initial_relative_advantages

        metrics = compute_initial_relative_advantages(mock_agents, mock_resources)

        assert "initial_relative_advantages" in metrics
        advantages = metrics["initial_relative_advantages"]
        assert "resource_proximity_advantage" in advantages
        assert "attribute_advantage" in advantages

    def test_extract_features_from_metrics(self):
        """Test feature extraction from metrics."""
        from farm.analysis.genesis.compute import extract_features_from_metrics

        metrics = {
            "initial_agent_count": 10,
            "nested": {
                "value1": 5.0,
                "value2": 10.0,
            },
            "string_value": "test",
        }

        features = extract_features_from_metrics(metrics)

        assert "initial_agent_count" in features
        assert "nested_value1" in features
        assert "nested_value2" in features
        assert "string_value" not in features  # Strings should be excluded


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_analyze_genesis_patterns_with_nan(self):
        """Test genesis patterns with NaN values."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "agent_id": [f"agent_{i}" for i in range(10)],
                "genesis_time": [np.nan] * 5 + list(range(5)),
                "success_rate": [np.nan, 0.5] * 5,
                "efficiency_score": [0.7] * 10,
            }
        )

        result = analyze_genesis_patterns(df)

        # Should handle NaN gracefully
        assert "total_genesis_events" in result
        assert result["total_genesis_events"] == 10

    def test_compute_metrics_with_progress_callback(self, mock_session, tmp_path):
        """Test computation with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)

        # Should work with progress callback
        assert ctx.progress_callback is not None

    def test_plot_with_single_agent_type(self, tmp_path):
        """Test plotting with single agent type."""
        df = pd.DataFrame(
            {
                "agent_id": [f"agent_{i}" for i in range(5)],
                "genesis_time": range(5),
                "success_rate": [0.8] * 5,
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.genesis.plot.plt"):
            result = plot_genesis_patterns(df, ctx)

        # Should handle single type gracefully
        assert result is None

    @patch("farm.analysis.genesis.compute.logger")
    def test_compute_with_database_error(self, mock_logger, mock_session):
        """Test computation handles database errors."""
        mock_session.query.side_effect = Exception("Database error")

        # Should log error but not crash
        with pytest.raises(Exception):
            compute_initial_state_metrics(mock_session)

        # Should have attempted the operation
        assert mock_session.query.called

    def test_zero_agents_at_critical_period(self, mock_session):
        """Test critical period with zero agents."""
        from farm.analysis.genesis.compute import compute_critical_period_metrics

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        metrics = compute_critical_period_metrics(mock_session)

        # Should return error
        assert "error" in metrics
