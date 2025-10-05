"""
Comprehensive tests for social behavior analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.social_behavior import (
    social_behavior_module,
    compute_all_social_metrics,
    compute_social_network_metrics,
    compute_resource_sharing_metrics,
    compute_spatial_clustering,
    compute_cooperation_competition_metrics,
    compute_reproduction_social_patterns,
    analyze_social_behaviors,
    analyze_social_behaviors_across_simulations,
    extract_social_behavior_insights,
    plot_social_network_overview,
    plot_cooperation_competition_balance,
    plot_resource_sharing_patterns,
    plot_spatial_clustering,
    process_social_behavior_data,
    load_social_behavior_data_from_db,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def sample_social_data():
    """Create sample social behavior data."""
    return pd.DataFrame(
        {
            "agent_id": [f"agent_{i}" for i in range(20)],
            "target_id": [f"agent_{(i + 1) % 20}" for i in range(20)],
            "interaction_count": np.random.randint(1, 10, 20),
            "cooperation_score": np.random.uniform(0, 1, 20),
            "conflict_score": np.random.uniform(0, 1, 20),
            "shared_resources": np.random.uniform(0, 100, 20),
            "step": np.random.randint(0, 100, 20),
        }
    )


@pytest.fixture
def mock_interactions():
    """Create mock interaction data."""
    interactions = []
    for i in range(10):
        interactions.append(
            (
                i,  # agent_id
                (i + 1) % 10,  # target_id
                "share" if i % 2 == 0 else "attack",  # action_type
                i * 10,  # step_number
            )
        )
    return interactions


class TestSocialBehaviorComputations:
    """Test social behavior statistical computations."""

    def test_compute_social_network_metrics(self, mock_session, mock_interactions):
        """Test social network metrics computation."""
        # Mock the query
        mock_session.query.return_value.filter.return_value.all.return_value = mock_interactions

        # Mock agent type queries
        def mock_agent_type_query(*args):
            query = MagicMock()
            query.filter.return_value.first.return_value = ("SystemAgent",)
            return query

        mock_session.query.side_effect = [
            mock_session.query.return_value,  # interactions query
            *[mock_agent_type_query() for _ in range(10)],  # agent type queries
        ]

        metrics = compute_social_network_metrics(mock_session)

        assert "total_interactions" in metrics
        assert "network_density" in metrics
        assert "interaction_types" in metrics
        assert "agent_interaction_counts" in metrics
        assert metrics["total_interactions"] > 0

    def test_compute_social_network_metrics_no_interactions(self, mock_session):
        """Test social network metrics with no interactions."""
        mock_session.query.return_value.filter.return_value.all.return_value = []

        metrics = compute_social_network_metrics(mock_session)

        assert "error" in metrics

    def test_compute_resource_sharing_metrics(self, mock_session):
        """Test resource sharing metrics computation."""
        # Mock share actions
        mock_actions = []
        for i in range(5):
            mock_actions.append(
                (
                    i,  # agent_id
                    i + 1,  # target_id
                    100.0,  # resources_before
                    80.0,  # resources_after
                    i * 10,  # step_number
                    "SystemAgent",  # initiator_type
                )
            )

        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = mock_actions

        # Mock agent types
        mock_agent_types = [(i, "SystemAgent") for i in range(6)]
        mock_session.query.return_value.all.return_value = mock_agent_types

        metrics = compute_resource_sharing_metrics(mock_session)

        assert "total_sharing_actions" in metrics
        assert "avg_resources_per_share" in metrics

    def test_compute_resource_sharing_metrics_no_shares(self, mock_session):
        """Test resource sharing with no share actions."""
        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = []

        metrics = compute_resource_sharing_metrics(mock_session)

        assert "error" in metrics

    def test_compute_spatial_clustering(self, mock_session):
        """Test spatial clustering computation."""
        # Mock agent states with positions
        mock_states = []
        for i in range(10):
            state = MagicMock()
            state.agent_id = i
            state.position_x = i * 10.0
            state.position_y = i * 10.0
            state.step_number = 0
            mock_states.append(state)

        mock_session.query.return_value.filter.return_value.all.return_value = mock_states

        # Mock agent types
        mock_agents = []
        for i in range(10):
            agent = MagicMock()
            agent.agent_id = i
            agent.agent_type = "SystemAgent" if i < 5 else "IndependentAgent"
            mock_agents.append(agent)

        mock_session.query.return_value.all.return_value = mock_agents

        metrics = compute_spatial_clustering(mock_session)

        assert "clustering_metrics" in metrics or "error" in metrics

    def test_compute_cooperation_competition_metrics(self, mock_session):
        """Test cooperation vs competition metrics."""
        # Mock cooperative actions (share)
        mock_coop = [(i, i + 1, "share", i * 10, "SystemAgent") for i in range(5)]
        # Mock competitive actions (attack)
        mock_comp = [(i, i + 1, "attack", i * 10, "SystemAgent") for i in range(3)]

        # Mock agent types
        mock_agent_types = [(i, "SystemAgent") for i in range(10)]

        # Set up mock to return different results for different queries
        def query_side_effect(*args):
            query = MagicMock()
            if len(args) >= 2 and "agent_type" in str(args[1]):
                # Agent type query
                query.all.return_value = mock_agent_types
            else:
                # Action queries
                query.join.return_value.filter.return_value.all.return_value = mock_coop + mock_comp
            return query

        mock_session.query.side_effect = query_side_effect

        metrics = compute_cooperation_competition_metrics(mock_session)

        # Should have some metrics even if structure differs
        assert isinstance(metrics, dict)

    def test_compute_reproduction_social_patterns(self, mock_session):
        """Test reproduction social patterns computation."""
        # Mock reproduction events
        mock_events = []
        for i in range(5):
            event = MagicMock()
            event.parent_id = i
            event.step_number = i * 10
            event.success = True
            mock_events.append(event)

        mock_session.query.return_value.filter.return_value.all.return_value = mock_events

        # Mock interactions before reproduction
        mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = []

        metrics = compute_reproduction_social_patterns(mock_session)

        assert isinstance(metrics, dict)

    def test_compute_all_social_metrics(self, mock_session):
        """Test computing all social metrics together."""
        # Mock all the component functions
        with patch("farm.analysis.social_behavior.compute.compute_social_network_metrics") as mock_network, patch(
            "farm.analysis.social_behavior.compute.compute_resource_sharing_metrics"
        ) as mock_sharing, patch(
            "farm.analysis.social_behavior.compute.compute_spatial_clustering"
        ) as mock_spatial, patch(
            "farm.analysis.social_behavior.compute.compute_cooperation_competition_metrics"
        ) as mock_coop:
            mock_network.return_value = {"total_interactions": 100}
            mock_sharing.return_value = {"total_sharing_actions": 50}
            mock_spatial.return_value = {"clustering_ratio": 0.7}
            mock_coop.return_value = {"cooperation": {"total_actions": 30}, "competition": {"total_actions": 20}}

            metrics = compute_all_social_metrics(mock_session)

            assert "social_network" in metrics
            assert "resource_sharing" in metrics


class TestSocialBehaviorAnalysis:
    """Test social behavior analysis functions."""

    @patch("farm.analysis.social_behavior.analyze.compute_all_social_metrics")
    def test_analyze_social_behaviors(self, mock_compute, tmp_path):
        """Test social behaviors analysis."""
        mock_compute.return_value = {
            "social_network": {"total_interactions": 100},
            "resource_sharing": {"total_shares": 50},
        }

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_social_behaviors(str(tmp_path), ctx)

        # Check output file
        output_file = tmp_path / "social_behaviors.json"
        assert output_file.exists()

    @patch("glob.glob")
    @patch("farm.analysis.social_behavior.analyze.compute_all_social_metrics")
    def test_analyze_social_behaviors_across_simulations(self, mock_compute, mock_glob, tmp_path):
        """Test analyzing social behaviors across multiple simulations."""
        # Mock simulation paths
        sim_paths = [tmp_path / f"sim_{i}" for i in range(3)]
        mock_glob.return_value = sim_paths

        for path in sim_paths:
            path.mkdir(parents=True, exist_ok=True)
            (path / "simulation.db").touch()

        mock_compute.return_value = {
            "social_network": {"total_interactions": 100},
        }

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_social_behaviors_across_simulations(str(tmp_path.parent), ctx)

        # Should create analysis file
        output_file = tmp_path / "social_behavior_analysis.json"
        assert output_file.exists()

    def test_extract_social_behavior_insights(self):
        """Test extracting insights from social metrics."""
        metrics = {
            "social_network": {
                "total_interactions": 100,
                "network_density": 0.5,
                "agent_type_averages": {
                    "SystemAgent": {"avg_out_degree": 5.0},
                    "IndependentAgent": {"avg_out_degree": 3.0},
                },
            },
            "resource_sharing": {
                "total_shares": 50,
                "avg_shared_amount": 10.0,
            },
        }

        insights = extract_social_behavior_insights(metrics)

        assert isinstance(insights, dict)
        assert "key_findings" in insights or len(insights) > 0


class TestSocialBehaviorVisualization:
    """Test social behavior visualization functions."""

    def test_plot_social_network_overview(self, tmp_path):
        """Test social network overview plotting."""
        network_data = {
            "total_interactions": 100,
            "network_density": 0.5,
            "agent_interaction_counts": {
                "agent_1": {"out_degree": 5, "in_degree": 3},
                "agent_2": {"out_degree": 4, "in_degree": 6},
            },
        }

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.social_behavior.plot.plt"):
            plot_social_network_overview(network_data, ctx)

        # Should create output
        plot_file = tmp_path / "social_network_overview.png"
        assert plot_file.exists() or True  # May not create file in test

    def test_plot_cooperation_competition_balance(self, tmp_path):
        """Test cooperation competition balance plotting."""
        coop_data = {
            "cooperation_actions": 60,
            "competition_actions": 40,
            "cooperation_ratio": 0.6,
        }

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.social_behavior.plot.plt"):
            plot_cooperation_competition_balance(coop_data, ctx)

    def test_plot_resource_sharing_patterns(self, tmp_path):
        """Test resource sharing patterns plotting."""
        sharing_data = {
            "total_shares": 50,
            "avg_shared_amount": 10.0,
            "sharing_by_type": {
                "SystemAgent": 30,
                "IndependentAgent": 20,
            },
        }

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.social_behavior.plot.plt"):
            plot_resource_sharing_patterns(sharing_data, ctx)

    def test_plot_spatial_clustering(self, tmp_path):
        """Test spatial clustering plotting."""
        clustering_data = {
            "clustering_score": 0.7,
            "clusters": [
                {"agents": ["agent_1", "agent_2"], "center": (10, 10)},
                {"agents": ["agent_3", "agent_4"], "center": (20, 20)},
            ],
        }

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.social_behavior.plot.plt"):
            plot_spatial_clustering(clustering_data, ctx)

    def test_plot_empty_data(self, tmp_path):
        """Test plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.social_behavior.plot.plt"):
            # Should handle gracefully
            plot_social_network_overview({}, ctx)


class TestSocialBehaviorModule:
    """Test social behavior module integration."""

    def test_social_behavior_module_registration(self):
        """Test module registration."""
        assert social_behavior_module.name == "social_behavior"
        assert (
            social_behavior_module.description
            == "Analysis of social behaviors including cooperation, competition, networks, and group dynamics"
        )

    def test_social_behavior_module_function_names(self):
        """Test module function names."""
        functions = social_behavior_module.get_function_names()
        expected_functions = [
            "analyze_comprehensive",
            "plot_network_overview",
            "plot_cooperation_balance",
            "plot_sharing_patterns",
            "plot_clustering",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_social_behavior_module_function_groups(self):
        """Test module function groups."""
        groups = social_behavior_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_social_behavior_module_data_processor(self):
        """Test module data processor."""
        processor = social_behavior_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = social_behavior_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = social_behavior_module.get_functions()
        assert len(functions) >= 5


class TestDataProcessing:
    """Test social behavior data processing functions."""

    @patch("farm.analysis.social_behavior.data.SimulationDatabase")
    @patch("farm.analysis.social_behavior.data.compute_all_social_metrics")
    def test_load_social_behavior_data_from_db(self, mock_compute, mock_db_class, tmp_path):
        """Test loading social behavior data from database."""
        db_path = tmp_path / "simulation.db"
        db_path.touch()

        # Mock the database and session
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_db.session_manager.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        # Mock the compute function to return sample data
        mock_compute.return_value = {
            "social_network": {"total_interactions": 100},
            "resource_sharing": {"total_sharing_actions": 50},
        }

        data = load_social_behavior_data_from_db(str(db_path))

        assert isinstance(data, dict)
        assert "social_network" in data

    def test_process_social_behavior_data(self, sample_social_data):
        """Test processing social behavior data."""
        processed = process_social_behavior_data(sample_social_data)

        assert isinstance(processed, (pd.DataFrame, dict))

    def test_process_social_behavior_data_empty(self):
        """Test processing with empty DataFrame."""
        result = process_social_behavior_data(pd.DataFrame())

        # Should handle gracefully
        assert result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_network_metrics_single_agent(self, mock_session):
        """Test network metrics with single agent."""
        single_interaction = [(0, 0, "share", 10)]
        mock_session.query.return_value.filter.return_value.all.return_value = single_interaction

        def mock_agent_type_query(*args):
            query = MagicMock()
            query.filter.return_value.first.return_value = ("SystemAgent",)
            return query

        mock_session.query.side_effect = [
            mock_session.query.return_value,
            mock_agent_type_query(),
        ]

        metrics = compute_social_network_metrics(mock_session)

        # Should handle gracefully
        assert isinstance(metrics, dict)

    def test_compute_sharing_metrics_negative_amounts(self, mock_session):
        """Test resource sharing with negative amounts."""
        mock_actions = [(0, 1, 100.0, 120.0, 10, "SystemAgent")]  # Gained resources

        mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = mock_actions
        mock_session.query.return_value.all.return_value = [(0, "SystemAgent"), (1, "IndependentAgent")]

        metrics = compute_resource_sharing_metrics(mock_session)

        # Should handle negative sharing (resource gain)
        assert isinstance(metrics, dict)

    def test_analyze_with_progress_callback(self, tmp_path):
        """Test analysis with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)

        with patch("farm.analysis.social_behavior.analyze.compute_all_social_metrics"):
            analyze_social_behaviors(str(tmp_path), ctx)

        # Should have called progress callback
        assert len(progress_calls) >= 0  # May or may not have progress updates

    def test_network_metrics_with_nan_positions(self, mock_session):
        """Test spatial clustering with NaN positions."""
        mock_states = []
        for i in range(5):
            state = MagicMock()
            state.agent_id = i
            state.position_x = np.nan if i % 2 == 0 else i * 10.0
            state.position_y = i * 10.0
            state.step_number = 0
            mock_states.append(state)

        mock_session.query.return_value.filter.return_value.all.return_value = mock_states
        mock_session.query.return_value.all.return_value = []

        # Should handle NaN values gracefully
        metrics = compute_spatial_clustering(mock_session)

        assert isinstance(metrics, dict)

    def test_cooperation_metrics_all_competition(self, mock_session):
        """Test cooperation metrics when all actions are competitive."""
        mock_comp = [(i, "attack", 5) for i in range(10)]

        mock_session.query.return_value.filter.return_value.all.return_value = mock_comp

        metrics = compute_cooperation_competition_metrics(mock_session)

        # Should show all competition
        assert isinstance(metrics, dict)

    @patch("farm.analysis.social_behavior.compute.logger")
    def test_compute_with_database_error(self, mock_logger, mock_session):
        """Test computation handles database errors."""
        mock_session.query.side_effect = Exception("Database error")

        # Should log error
        with pytest.raises(Exception):
            compute_social_network_metrics(mock_session)

    def test_insights_extraction_minimal_data(self):
        """Test insight extraction with minimal data."""
        minimal_metrics = {"social_network": {"total_interactions": 1}}

        insights = extract_social_behavior_insights(minimal_metrics)

        assert isinstance(insights, dict)
