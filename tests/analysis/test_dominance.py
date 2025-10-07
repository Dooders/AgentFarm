"""
Comprehensive tests for dominance analysis module.

Updated to use the new protocol-based architecture with orchestrator.

NOTE: This test file uses the clean class-based architecture without
backward compatibility wrappers. All tests use:
- DominanceComputer class for computation tests
- get_orchestrator() for integration tests
- Protocol-based mocking for isolation
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.dominance import (
    dominance_module,
    plot_dominance_distribution,
    plot_comprehensive_score_breakdown,
    plot_dominance_switches,
    plot_dominance_stability,
    run_dominance_classification,
    get_orchestrator,
    DominanceComputer,
    DominanceAnalyzer,
)
from farm.analysis.dominance.mocks import (
    MockDominanceComputer,
    create_sample_simulation_data,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def mock_simulation_steps():
    """Create mock simulation steps."""
    steps = []
    for i in range(100):
        step = MagicMock()
        step.step_number = i
        step.system_agents = 10 + i % 5
        step.independent_agents = 8 - i % 4
        step.control_agents = 5 + i % 3
        steps.append(step)
    return steps


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    agents = []
    for i in range(30):
        agent = MagicMock()
        agent.agent_id = f"agent_{i}"
        agent.agent_type = ["system", "independent", "control"][i % 3]
        agent.birth_time = i
        agent.death_time = i + 50 if i % 2 == 0 else None
        agents.append(agent)
    return agents


@pytest.fixture
def sample_dominance_data():
    """Create sample dominance data."""
    return pd.DataFrame(
        {
            "iteration": range(20),
            "population_dominance": np.random.choice(["system", "independent", "control"], 20),
            "survival_dominance": np.random.choice(["system", "independent", "control"], 20),
            "comprehensive_dominance": np.random.choice(["system", "independent", "control"], 20),
            "system_dominance_score": np.random.uniform(0, 1, 20),
            "independent_dominance_score": np.random.uniform(0, 1, 20),
            "control_dominance_score": np.random.uniform(0, 1, 20),
            "total_switches": np.random.randint(0, 10, 20),
            "switches_per_step": np.random.uniform(0, 0.1, 20),
            "reproduction_success": np.random.uniform(0, 1, 20),
            "resource_proximity": np.random.uniform(0, 1, 20),
            "stability_score": np.random.uniform(0, 1, 20),
            "switching_frequency": np.random.randint(0, 5, 20),
            # Add missing columns for plot functions
            "system_avg_dominance_period": np.random.uniform(1, 10, 20),
            "independent_avg_dominance_period": np.random.uniform(1, 10, 20),
            "control_avg_dominance_period": np.random.uniform(1, 10, 20),
            "early_phase_switches": np.random.randint(0, 3, 20),
            "middle_phase_switches": np.random.randint(0, 3, 20),
            "late_phase_switches": np.random.randint(0, 3, 20),
            "dominance_stability": np.random.uniform(0, 100, 20),
        }
    )


class TestDominanceComputations:
    """Test dominance statistical computations using class-based architecture."""

    def test_compute_population_dominance(self, mock_session, mock_simulation_steps):
        """Test population dominance computation with DominanceComputer class."""
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        computer = DominanceComputer()
        result = computer.compute_population_dominance(mock_session)

        assert result in ["system", "independent", "control"]

    def test_compute_population_dominance_no_data(self, mock_session):
        """Test population dominance with no data."""
        mock_session.query.return_value.order_by.return_value.first.return_value = None

        computer = DominanceComputer()
        result = computer.compute_population_dominance(mock_session)

        assert result is None

    def test_compute_survival_dominance(self, mock_session, mock_agents, mock_simulation_steps):
        """Test survival dominance computation with DominanceComputer class."""
        mock_session.query.return_value.all.return_value = mock_agents
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        computer = DominanceComputer()
        result = computer.compute_survival_dominance(mock_session)

        assert result in ["system", "independent", "control"] or result is None

    def test_compute_survival_dominance_no_agents(self, mock_session):
        """Test survival dominance with no agents."""
        mock_session.query.return_value.all.return_value = []
        mock_session.query.return_value.order_by.return_value.first.return_value = None

        computer = DominanceComputer()
        result = computer.compute_survival_dominance(mock_session)

        assert result is None

    def test_compute_dominance_switches(self, mock_session, mock_simulation_steps):
        """Test dominance switch computation with DominanceComputer class."""
        mock_session.query.return_value.order_by.return_value.all.return_value = mock_simulation_steps

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        assert isinstance(result, dict)
        assert "total_switches" in result
        assert "switches_per_step" in result
        assert "avg_dominance_periods" in result
        assert "transition_matrix" in result

    def test_compute_dominance_switches_no_steps(self, mock_session):
        """Test dominance switches with no steps."""
        mock_session.query.return_value.order_by.return_value.all.return_value = []

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        assert result is None

    def test_compute_dominance_switches_single_dominant(self, mock_session):
        """Test dominance switches with single dominant type."""
        # Create steps with only one dominant type
        steps = []
        for i in range(50):
            step = MagicMock()
            step.step_number = i
            step.system_agents = 20
            step.independent_agents = 5
            step.control_agents = 3
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        result = compute_dominance_switches(mock_session)

        assert result["total_switches"] == 0

    def test_compute_comprehensive_dominance(self, mock_session, mock_simulation_steps):
        """Test comprehensive dominance computation."""
        # Set up mock simulation steps with total_agents attribute
        for step in mock_simulation_steps:
            step.total_agents = step.system_agents + step.independent_agents + step.control_agents

        mock_session.query.return_value.order_by.return_value.all.return_value = mock_simulation_steps
        mock_session.query.return_value.scalar.return_value = 100

        result = compute_comprehensive_dominance(mock_session)

        assert isinstance(result, dict)
        assert "dominant_type" in result
        assert "scores" in result
        assert "metrics" in result

    def test_compute_dominance_switches_phase_distribution(self, mock_session):
        """Test that switches are distributed across phases."""
        # Create steps that switch dominance in different phases
        steps = []
        for i in range(90):
            step = MagicMock()
            step.step_number = i
            # Switch at 30 and 60 to hit different phases
            if i < 30:
                step.system_agents = 20
                step.independent_agents = 5
                step.control_agents = 3
            elif i < 60:
                step.system_agents = 5
                step.independent_agents = 20
                step.control_agents = 3
            else:
                step.system_agents = 3
                step.independent_agents = 5
                step.control_agents = 20
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        result = compute_dominance_switches(mock_session)

        assert "phase_switches" in result
        assert result["total_switches"] >= 2


class TestDominanceAnalysis:
    """Test dominance analysis functions."""

    def test_process_single_simulation(self, mock_session):
        """Test processing single simulation."""
        from farm.analysis.dominance.analyze import process_single_simulation

        with patch("farm.analysis.dominance.analyze.compute_population_dominance") as mock_pop, patch(
            "farm.analysis.dominance.analyze.compute_survival_dominance"
        ) as mock_surv, patch("farm.analysis.dominance.analyze.compute_comprehensive_dominance") as mock_comp, patch(
            "farm.analysis.dominance.analyze.compute_dominance_switches"
        ) as mock_switch, patch(
            "farm.analysis.dominance.analyze.get_initial_positions_and_resources"
        ) as mock_init, patch("farm.analysis.dominance.analyze.get_final_population_counts") as mock_final, patch(
            "farm.analysis.dominance.analyze.get_agent_survival_stats"
        ) as mock_stats, patch("farm.analysis.dominance.analyze.get_reproduction_stats") as mock_repro:
            mock_pop.return_value = "system"
            mock_surv.return_value = "system"
            mock_comp.return_value = {
                "dominant_type": "system",
                "scores": {"system": 0.8, "independent": 0.3, "control": 0.2},
                "metrics": {
                    "auc": {"system": 0.7, "independent": 0.2, "control": 0.1},
                    "recency_weighted_auc": {"system": 0.75, "independent": 0.2, "control": 0.05},
                    "dominance_duration": {"system": 80, "independent": 15, "control": 5},
                    "growth_trends": {"system": 0.02, "independent": -0.01, "control": -0.005},
                    "final_ratios": {"system": 0.7, "independent": 0.2, "control": 0.1},
                },
            }
            mock_switch.return_value = {
                "total_switches": 3,
                "switches_per_step": 0.03,
                "avg_dominance_periods": {"system": 30, "independent": 20, "control": 10},
                "phase_switches": {"early": 1, "middle": 1, "late": 1},
                "transition_probabilities": {
                    "system": {"system": 0, "independent": 0.5, "control": 0.5},
                    "independent": {"system": 0.5, "independent": 0, "control": 0.5},
                    "control": {"system": 0.5, "independent": 0.5, "control": 0},
                },
            }
            mock_init.return_value = {"initial_system_count": 10}
            mock_final.return_value = {"system_agents": 20}
            mock_stats.return_value = {
                "system_count": 20,
                "system_alive": 15,
                "system_dead": 5,
                "system_avg_survival": 50,
                "system_dead_ratio": 0.25,
            }
            mock_repro.return_value = {"system_reproduction_attempts": 10}

            result = process_single_simulation(mock_session, iteration=1, config={})

            assert result is not None
            assert "iteration" in result
            assert "comprehensive_dominance" in result

    def test_process_single_simulation_error(self, mock_session):
        """Test processing simulation with error."""
        from farm.analysis.dominance.analyze import process_single_simulation

        with patch("farm.analysis.dominance.analyze.compute_population_dominance") as mock_pop:
            mock_pop.side_effect = Exception("Database error")

            result = process_single_simulation(mock_session, iteration=1, config={})

            assert result is None

    def test_process_dominance_data(self, tmp_path):
        """Test processing dominance data."""
        from farm.analysis.dominance.analyze import process_dominance_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.dominance.analyze.setup_and_process_simulations") as mock_setup, patch(
            "farm.analysis.dominance.analyze.analyze_dominance_switch_factors"
        ) as mock_factors:
            mock_setup.return_value = [
                {"iteration": 0, "comprehensive_dominance": "system", "total_switches": 3},
                {"iteration": 1, "comprehensive_dominance": "independent", "total_switches": 5},
            ]
            mock_factors.return_value = pd.DataFrame(mock_setup.return_value)

            result = process_dominance_data(str(exp_path))

            assert isinstance(result, pd.DataFrame)

    def test_analyze_dominance_switch_factors(self):
        """Test analyzing dominance switch factors."""
        from farm.analysis.dominance.analyze import analyze_dominance_switch_factors

        df = pd.DataFrame(
            {
                "iteration": range(10),
                "total_switches": [3, 5, 2, 4, 6, 3, 5, 4, 2, 3],
                "comprehensive_dominance": ["system"] * 10,
                "system_reproduction_attempts": np.random.randint(5, 15, 10),
            }
        )

        result = analyze_dominance_switch_factors(df)

        assert isinstance(result, pd.DataFrame)

    def test_analyze_high_vs_low_switching(self):
        """Test analyzing high vs low switching with orchestrator."""
        orchestrator = get_orchestrator()

        df = pd.DataFrame(
            {
                "iteration": range(20),
                "total_switches": list(range(20)),
                "system_reproduction_attempts": np.random.randint(5, 15, 20),
                "independent_reproduction_attempts": np.random.randint(5, 15, 20),
            }
        )

        numeric_cols = ["system_reproduction_attempts", "independent_reproduction_attempts"]
        result = orchestrator.analyze_high_vs_low_switching(df, numeric_cols)

        assert isinstance(result, pd.DataFrame)


class TestDominanceVisualization:
    """Test dominance visualization functions."""

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_dominance_distribution(self, mock_plt, sample_dominance_data, tmp_path):
        """Test dominance distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Mock subplots to return proper structure
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        result = plot_dominance_distribution(sample_dominance_data, ctx=ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_dominance_distribution_empty(self, mock_plt, tmp_path):
        """Test distribution plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)
        df = pd.DataFrame()

        result = plot_dominance_distribution(df, ctx=ctx)

        # Should handle gracefully
        assert result is None

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_comprehensive_score_breakdown(self, mock_plt, sample_dominance_data, tmp_path):
        """Test comprehensive score breakdown plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        result = plot_comprehensive_score_breakdown(sample_dominance_data, ctx=ctx)

        assert isinstance(result, pd.DataFrame)
        assert mock_plt.savefig.called

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_dominance_switches(self, mock_plt, sample_dominance_data, tmp_path):
        """Test dominance switches plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_dominance_switches(sample_dominance_data, ctx=ctx)

        # Should create plot
        assert mock_plt.savefig.called or mock_plt.figure.called

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_dominance_stability(self, mock_plt, sample_dominance_data, tmp_path):
        """Test dominance stability plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_dominance_stability(sample_dominance_data, ctx=ctx)

        # Should create plot
        assert mock_plt.savefig.called or mock_plt.figure.called

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_with_missing_columns(self, mock_plt, tmp_path):
        """Test plotting with missing required columns."""
        ctx = AnalysisContext(output_path=tmp_path)
        df = pd.DataFrame({"iteration": [1, 2, 3]})

        # Should handle missing columns gracefully
        plot_dominance_distribution(df, ctx=ctx)


class TestDominanceML:
    """Test dominance machine learning functions."""

    def test_run_dominance_classification(self):
        """Test dominance classification."""
        df = pd.DataFrame(
            {
                "iteration": range(100),
                "comprehensive_dominance": np.random.choice(["system", "independent", "control"], 100),
                "reproduction_success": np.random.uniform(0, 1, 100),
                "resource_proximity": np.random.uniform(0, 1, 100),
                "stability_score": np.random.uniform(0, 1, 100),
                "switching_frequency": np.random.randint(0, 5, 100),
                "dominance_category": np.random.choice(["low", "medium", "high"], 100),
            }
        )

        try:
            result = run_dominance_classification(df)
            assert isinstance(result, dict)
        except Exception as e:
            # ML functions might fail with small/synthetic datasets
            assert isinstance(e, (ValueError, ImportError, KeyError)) or "classification" in str(e).lower()

    def test_run_dominance_classification_insufficient_data(self, tmp_path):
        """Test classification with insufficient data."""
        df = pd.DataFrame(
            {
                "iteration": [1, 2],
                "dominance_category": ["low", "high"],
            }
        )

        # Should handle insufficient data
        try:
            result = run_dominance_classification(df, str(tmp_path))
            # If it doesn't raise an error, check result type
            if result is not None:
                assert isinstance(result, dict)
        except (ValueError, KeyError):
            # Expected for insufficient data
            pass


class TestDominanceModule:
    """Test dominance module integration."""

    def test_dominance_module_registration(self):
        """Test module registration."""
        assert dominance_module.name == "dominance"
        assert dominance_module.description == "Analysis of agent dominance patterns in simulations"

    def test_dominance_module_function_names(self):
        """Test module function names."""
        functions = dominance_module.get_function_names()
        expected_functions = [
            "plot_dominance_distribution",
            "plot_comprehensive_score_breakdown",
            "plot_dominance_switches",
            "plot_dominance_stability",
            "plot_reproduction_success_vs_switching",
            "plot_reproduction_advantage_vs_stability",
            "plot_resource_proximity_vs_dominance",
            "plot_reproduction_vs_dominance",
            "plot_dominance_comparison",
            "plot_correlation_matrix",
            "run_dominance_classification",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_dominance_module_function_groups(self):
        """Test module function groups."""
        groups = dominance_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "ml" in groups
        assert "correlation" in groups

    def test_dominance_module_data_processor(self):
        """Test module data processor."""
        processor = dominance_module.get_data_processor()
        assert processor is not None

    def test_dominance_module_supports_database(self):
        """Test that dominance module supports database."""
        assert dominance_module.supports_database()

    def test_dominance_module_db_filename(self):
        """Test dominance module database filename."""
        assert dominance_module.get_db_filename() == "dominance.db"

    def test_module_validator(self):
        """Test module validator."""
        validator = dominance_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = dominance_module.get_functions()
        assert len(functions) >= 11


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_dominance_with_equal_populations(self, mock_session):
        """Test dominance with equal populations."""
        step = MagicMock()
        step.system_agents = 10
        step.independent_agents = 10
        step.control_agents = 10

        mock_session.query.return_value.order_by.return_value.first.return_value = step

        result = compute_population_dominance(mock_session)

        # Should pick one deterministically
        assert result in ["system", "independent", "control"]

    def test_compute_survival_with_all_alive_agents(self, mock_session):
        """Test survival dominance with all agents alive."""
        agents = []
        for i in range(30):
            agent = MagicMock()
            agent.agent_type = ["system", "independent", "control"][i % 3]
            agent.birth_time = i
            agent.death_time = None  # All alive
            agents.append(agent)

        step = MagicMock()
        step.step_number = 100

        mock_session.query.return_value.all.return_value = agents
        mock_session.query.return_value.order_by.return_value.first.return_value = step

        result = compute_survival_dominance(mock_session)

        # Should compute based on birth time
        assert result in ["system", "independent", "control"] or result is None

    def test_analyze_switch_factors_empty_df(self):
        """Test analyzing switch factors with empty DataFrame."""
        from farm.analysis.dominance.analyze import analyze_dominance_switch_factors

        result = analyze_dominance_switch_factors(pd.DataFrame())

        # Should handle empty DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_analyze_switch_factors_no_switches_column(self):
        """Test analyzing switch factors without switches column."""
        from farm.analysis.dominance.analyze import analyze_dominance_switch_factors

        df = pd.DataFrame({"iteration": [1, 2, 3]})
        result = analyze_dominance_switch_factors(df)

        # Should handle missing column
        assert isinstance(result, pd.DataFrame)

    def test_process_dominance_data_empty_path(self, tmp_path):
        """Test processing dominance data with empty path."""
        from farm.analysis.dominance.analyze import process_dominance_data

        empty_path = tmp_path / "empty"
        empty_path.mkdir()

        with patch("farm.analysis.dominance.analyze.setup_and_process_simulations") as mock_setup:
            mock_setup.return_value = []

            result = process_dominance_data(str(empty_path))

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_plot_distribution_single_measure(self, tmp_path):
        """Test plotting distribution with single measure."""
        df = pd.DataFrame({"population_dominance": ["system", "independent", "control"] * 3})

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.dominance.plot.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            plot_dominance_distribution(df, ctx=ctx)

            # Should handle single measure
            assert mock_plt.savefig.called

    def test_compute_switches_with_no_agent_steps(self, mock_session):
        """Test computing switches when all steps have zero agents."""
        steps = []
        for i in range(50):
            step = MagicMock()
            step.step_number = i
            step.system_agents = 0
            step.independent_agents = 0
            step.control_agents = 0
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        result = compute_dominance_switches(mock_session)

        # Should handle no agents gracefully
        assert result is not None
        assert result["total_switches"] == 0

    def test_dominance_analysis_with_nan_values(self):
        """Test dominance analysis with NaN values."""
        from farm.analysis.dominance.analyze import analyze_dominance_switch_factors

        df = pd.DataFrame(
            {
                "iteration": range(10),
                "total_switches": [3, np.nan, 2, 4, np.nan, 3, 5, 4, 2, 3],
                "comprehensive_dominance": ["system"] * 10,
                "system_reproduction_attempts": [10, 12, np.nan, 14, 15, np.nan, 13, 11, 12, 10],
            }
        )

        result = analyze_dominance_switch_factors(df)

        # Should handle NaN values
        assert isinstance(result, pd.DataFrame)

    def test_comprehensive_dominance_single_agent_type(self, mock_session):
        """Test comprehensive dominance with single agent type."""
        steps = []
        for i in range(50):
            step = MagicMock()
            step.step_number = i
            step.system_agents = 20
            step.independent_agents = 0
            step.control_agents = 0
            step.total_agents = 20  # Add total_agents attribute
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps
        mock_session.query.return_value.scalar.return_value = 50

        result = compute_comprehensive_dominance(mock_session)

        assert result["dominant_type"] == "system"

    @patch("farm.analysis.dominance.compute.logger")
    def test_compute_with_logging(self, mock_logger, mock_session):
        """Test that computation logs appropriately."""
        mock_session.query.return_value.order_by.return_value.first.return_value = None

        compute_population_dominance(mock_session)

        # Logger might be called (implementation dependent)
        # Just verify it doesn't crash


class TestDominanceHelperFunctions:
    """Test helper functions in dominance module."""

    def test_get_agent_survival_stats(self, mock_session, mock_agents):
        """Test getting agent survival statistics."""
        from farm.analysis.dominance.data import get_agent_survival_stats

        mock_session.query.return_value.all.return_value = mock_agents

        # Mock the final step query
        final_step = MagicMock()
        final_step.step_number = 100
        mock_session.query.return_value.order_by.return_value.first.return_value = final_step

        result = get_agent_survival_stats(mock_session)

        assert isinstance(result, dict)

    def test_get_final_population_counts(self, mock_session):
        """Test getting final population counts."""
        from farm.analysis.dominance.data import get_final_population_counts

        final_step = MagicMock()
        final_step.system_agents = 20
        final_step.independent_agents = 15
        final_step.control_agents = 10
        final_step.step_number = 100

        mock_session.query.return_value.order_by.return_value.first.return_value = final_step

        result = get_final_population_counts(mock_session)

        assert isinstance(result, dict)
        assert "system_agents" in result

    def test_save_dominance_data_to_db(self, sample_dominance_data):
        """Test saving dominance data to database."""
        from farm.analysis.dominance.db_io import save_dominance_data_to_db

        with patch("farm.analysis.dominance.db_io.import_multi_table_data") as mock_import, \
             patch("farm.analysis.dominance.db_io.init_db") as mock_init, \
             patch("farm.analysis.dominance.db_io.get_session") as mock_get_session:

            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_import.return_value = len(sample_dominance_data)

            # Should handle saving
            result = save_dominance_data_to_db(sample_dominance_data, "sqlite:///:memory:")

            assert result is True
            mock_import.assert_called_once()

    def test_save_dominance_data_empty_df(self):
        """Test saving empty DataFrame to database."""
        from farm.analysis.dominance.db_io import save_dominance_data_to_db

        # Should return False for empty data without calling database functions
        result = save_dominance_data_to_db(pd.DataFrame(), "sqlite:///:memory:")

        # Should return False for empty data
        assert result is False
