"""
Comprehensive tests for dominance analysis module.

Updated to use the new protocol-based architecture with orchestrator.

NOTE: This test file uses the clean class-based architecture without
backward compatibility wrappers. All tests use:
- DominanceComputer class for computation tests
- get_orchestrator() for integration tests
- Protocol-based mocking for isolation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.dominance import (
    DominanceComputer,
    dominance_module,
    get_orchestrator,
    load_data_from_db,
    plot_comprehensive_score_breakdown,
    plot_dominance_distribution,
    plot_dominance_stability,
    plot_dominance_switches,
    plot_feature_importance,
    run_dominance_classification,
)
from farm.analysis.dominance.ml import prepare_features_for_classification, train_classifier
from farm.analysis.dominance.validation import validate_sim_data


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
        # Use new JSON structure for agent type counts
        step.agent_type_counts = {
            "system": 10 + i % 5,
            "independent": 8 - i % 4,
            "control": 5 + i % 3,
        }
        # Keep old attributes for backwards compatibility in tests that still use them
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
            step.agent_type_counts = {"system": 20, "independent": 5, "control": 3}
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        assert result["total_switches"] == 0

    def test_compute_comprehensive_dominance(self, mock_session, mock_simulation_steps):
        """Test comprehensive dominance computation."""
        # Set up mock simulation steps with total_agents attribute
        for step in mock_simulation_steps:
            step.total_agents = step.system_agents + step.independent_agents + step.control_agents

        mock_session.query.return_value.order_by.return_value.all.return_value = mock_simulation_steps
        mock_session.query.return_value.scalar.return_value = 100

        computer = DominanceComputer()
        result = computer.compute_comprehensive_dominance(mock_session)

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
                step.agent_type_counts = {"system": 20, "independent": 5, "control": 3}
            elif i < 60:
                step.system_agents = 5
                step.independent_agents = 20
                step.control_agents = 3
                step.agent_type_counts = {"system": 5, "independent": 20, "control": 3}
            else:
                step.system_agents = 3
                step.independent_agents = 5
                step.control_agents = 20
                step.agent_type_counts = {"system": 3, "independent": 5, "control": 20}
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        assert "phase_switches" in result
        assert result["total_switches"] >= 2


class TestDominanceAnalysis:
    """Test dominance analysis functions."""

    def test_process_single_simulation(self, mock_session):
        """Test processing single simulation."""
        from farm.analysis.dominance.analyze import process_single_simulation

        with patch("farm.analysis.dominance.orchestrator.create_dominance_orchestrator") as mock_create_orch:
            mock_orchestrator = MagicMock()
            mock_orchestrator.compute_population_dominance.return_value = "system"
            mock_orchestrator.compute_survival_dominance.return_value = "system"
            mock_orchestrator.compute_comprehensive_dominance.return_value = {
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
            mock_orchestrator.compute_dominance_switches.return_value = {
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
            mock_orchestrator.get_initial_positions_and_resources.return_value = {"initial_system_count": 10}
            mock_orchestrator.get_final_population_counts.return_value = {"system_agents": 20}
            mock_orchestrator.get_agent_survival_stats.return_value = {
                "system_count": 20,
                "system_alive": 15,
                "system_dead": 5,
                "system_avg_survival": 50,
                "system_dead_ratio": 0.25,
            }
            mock_orchestrator.get_reproduction_stats.return_value = {"system_reproduction_attempts": 10}
            mock_create_orch.return_value = mock_orchestrator

            result = process_single_simulation(mock_session, iteration=1, config={})

            assert result is not None
            assert "iteration" in result
            assert "comprehensive_dominance" in result

    def test_process_single_simulation_error(self, mock_session):
        """Test processing simulation with error."""
        from farm.analysis.dominance.analyze import process_single_simulation

        with patch("farm.analysis.dominance.orchestrator.create_dominance_orchestrator") as mock_create_orch:
            mock_orchestrator = MagicMock()
            mock_orchestrator.compute_population_dominance.side_effect = Exception("Database error")
            mock_create_orch.return_value = mock_orchestrator

            result = process_single_simulation(mock_session, iteration=1, config={})

            assert result is None

    def test_process_dominance_data(self, tmp_path):
        """Test processing dominance data."""
        from farm.analysis.dominance.analyze import process_dominance_data

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("scripts.analysis_config.setup_and_process_simulations") as mock_setup, patch(
            "farm.analysis.dominance.orchestrator.create_dominance_orchestrator"
        ) as mock_create_orch:
            mock_setup.return_value = [
                {"iteration": 0, "comprehensive_dominance": "system", "total_switches": 3},
                {"iteration": 1, "comprehensive_dominance": "independent", "total_switches": 5},
            ]
            mock_orchestrator = MagicMock()
            mock_orchestrator.analyze_dataframe_comprehensively.return_value = pd.DataFrame(mock_setup.return_value)
            mock_create_orch.return_value = mock_orchestrator

            result = process_dominance_data(str(exp_path))

            assert isinstance(result, pd.DataFrame)

    def test_analyze_dominance_switch_factors(self):
        """Test analyzing dominance switch factors."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "total_switches": [3, 5, 2, 4, 6, 3, 5, 4, 2, 3],
                "comprehensive_dominance": ["system"] * 10,
                "system_reproduction_attempts": np.random.randint(5, 15, 10),
            }
        )

        computer = DominanceComputer()
        result = computer.compute_dominance_switch_factors(df)

        assert isinstance(result, dict)

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

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_feature_importance(self, mock_plt, tmp_path):
        """Test feature importance plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Sample feature importance data
        feat_imp = [
            ("feature1", 0.8),
            ("feature2", 0.7),
            ("feature3", 0.6),
            ("feature4", 0.5),
            ("feature5", 0.4),
            ("feature6", 0.3),
            ("feature7", 0.2),
            ("feature8", 0.1),
        ]

        plot_feature_importance(feat_imp=feat_imp, label_name="comprehensive_dominance", ctx=ctx)

        # Should create plot and save
        assert mock_plt.savefig.called
        assert mock_plt.figure.called

    @patch("farm.analysis.dominance.plot.plt")
    def test_plot_feature_importance_missing_parameters(self, mock_plt, tmp_path):
        """Test feature importance plotting with missing required parameters."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Test missing feat_imp
        with pytest.raises(ValueError, match="feat_imp parameter is required"):
            plot_feature_importance(label_name="test", ctx=ctx)

        # Test missing label_name
        with pytest.raises(ValueError, match="label_name parameter is required"):
            plot_feature_importance(feat_imp=[("f", 0.5)], ctx=ctx)

        # Test missing output_path
        with pytest.raises(ValueError, match="output_path parameter is required"):
            plot_feature_importance(feat_imp=[("f", 0.5)], label_name="test")


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

    @patch("farm.analysis.dominance.ml.logger")
    def test_train_classifier(self, mock_logger):
        """Test training a classifier."""
        # Create sample data
        X = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, 100),
                "feature2": np.random.uniform(0, 1, 100),
                "categorical": np.random.choice(["A", "B", "C"], 100),
            }
        )
        y = pd.Series(np.random.choice(["system", "independent", "control"], 100))

        clf, feat_imp = train_classifier(X, y, "test_dominance")

        # Should return classifier and feature importances
        assert clf is not None
        assert isinstance(feat_imp, list)
        assert len(feat_imp) > 0
        assert all(isinstance(feat, tuple) and len(feat) == 2 for feat in feat_imp)

        # Should have logged information
        assert mock_logger.info.called

    @patch("farm.analysis.dominance.ml.logger")
    def test_prepare_features_for_classification(self, mock_logger):
        """Test preparing features for classification."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "population_dominance": ["system"] * 10,
                "survival_dominance": ["independent"] * 10,
                "comprehensive_dominance": ["system"] * 10,
                "feature1": np.random.uniform(0, 1, 10),
                "feature2": np.random.uniform(0, 1, 10),
                "categorical": ["A", "B"] * 5,
            }
        )

        X, feature_cols, exclude_cols = prepare_features_for_classification(df)

        # Should return feature matrix and column lists
        assert isinstance(X, pd.DataFrame)
        assert isinstance(feature_cols, list)
        assert isinstance(exclude_cols, list)

        # Should exclude dominance columns
        assert "iteration" in exclude_cols
        assert "population_dominance" in exclude_cols
        assert "survival_dominance" in exclude_cols
        assert "comprehensive_dominance" in exclude_cols

        # Should include feature columns
        assert "feature1" in feature_cols
        assert "feature2" in feature_cols
        assert "categorical" in feature_cols

        # X should not contain excluded columns
        assert "iteration" not in X.columns
        assert "population_dominance" not in X.columns

    @patch("farm.analysis.dominance.ml.logger")
    def test_prepare_features_with_duplicates(self, mock_logger):
        """Test preparing features when DataFrame has duplicate columns."""
        # Create DataFrame with duplicate columns by using MultiIndex or direct assignment
        data = {
            "iteration": range(10),
            "feature1": np.random.uniform(0, 1, 10),
            "feature2": np.random.uniform(0, 1, 10),
        }
        df = pd.DataFrame(data)
        # Manually create duplicate column names
        df.columns = ["iteration", "feature1", "feature1"]  # Force duplicate

        X, feature_cols, exclude_cols = prepare_features_for_classification(df)

        # Should handle duplicates and log warning
        assert isinstance(X, pd.DataFrame)
        assert mock_logger.warning.called

    @patch("farm.analysis.dominance.ml.logger")
    def test_prepare_features_with_missing_values(self, mock_logger):
        """Test preparing features with missing values."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "population_dominance": ["system"] * 10,
                "numeric_feature": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "categorical_feature": ["A", "B", None, "A", "B", "A", "B", "A", "B", "A"],
            }
        )

        X, feature_cols, exclude_cols = prepare_features_for_classification(df)

        # Should fill missing values
        assert isinstance(X, pd.DataFrame)
        assert not X["numeric_feature"].isna().any()
        assert not X["categorical_feature"].isna().any()

    def test_create_dominance_metrics(self):
        """Test creating dominance metrics object from DataFrame row."""
        from farm.analysis.dominance.db_io import _create_dominance_metrics

        # Create test row data
        row = {
            "population_dominance": "system",
            "survival_dominance": "independent",
            "comprehensive_dominance": "system",
            "system_dominance_score": 0.8,
            "independent_dominance_score": 0.6,
            "control_dominance_score": 0.4,
            "system_auc": 0.85,
            "independent_auc": 0.65,
            "control_auc": 0.45,
            "system_recency_weighted_auc": 0.82,
            "independent_recency_weighted_auc": 0.62,
            "control_recency_weighted_auc": 0.42,
            "system_dominance_duration": 50,
            "independent_dominance_duration": 30,
            "control_dominance_duration": 20,
            "system_growth_trend": 0.1,
            "independent_growth_trend": 0.05,
            "control_growth_trend": -0.02,
            "system_final_ratio": 0.6,
            "independent_final_ratio": 0.3,
            "control_final_ratio": 0.1,
        }

        result = _create_dominance_metrics(row, sim_id=1)

        # Should create DominanceMetrics object
        assert result is not None
        assert result.simulation_id == 1
        assert result.population_dominance == "system"
        assert result.comprehensive_dominance == "system"
        assert result.system_dominance_score == 0.8

    def test_create_agent_population(self):
        """Test creating agent population object from DataFrame row."""
        from farm.analysis.dominance.db_io import _create_agent_population

        # Create test row data
        row = {
            "system_agents": 20,
            "independent_agents": 15,
            "control_agents": 10,
            "total_agents": 45,
            "final_step": 100,
            "system_count": 25,
            "system_alive": 20,
            "system_dead": 5,
            "system_avg_survival": 85.5,
            "system_dead_ratio": 0.2,
            "independent_count": 18,
            "independent_alive": 15,
            "independent_dead": 3,
            "independent_avg_survival": 78.2,
            "independent_dead_ratio": 0.167,
            "control_count": 12,
            "control_alive": 10,
            "control_dead": 2,
            "control_avg_survival": 72.1,
            "control_dead_ratio": 0.167,
        }

        result = _create_agent_population(row, sim_id=1)

        # Should create AgentPopulation object
        assert result is not None
        assert result.simulation_id == 1
        assert result.system_agents == 20
        assert result.total_agents == 45
        assert result.system_avg_survival == 85.5

    def test_create_reproduction_stats(self):
        """Test creating reproduction stats object from DataFrame row."""
        from farm.analysis.dominance.db_io import _create_reproduction_stats

        # Create test row data
        row = {
            "system_reproduction_attempts": 25,
            "system_reproduction_successes": 20,
            "system_reproduction_failures": 5,
            "system_reproduction_success_rate": 0.8,
            "system_first_reproduction_time": 10,
            "system_reproduction_efficiency": 0.75,
            "system_avg_resources_per_reproduction": 50.5,
            "system_avg_offspring_resources": 25.2,
            "independent_reproduction_attempts": 22,
            "independent_reproduction_successes": 18,
            "independent_reproduction_failures": 4,
            "independent_reproduction_success_rate": 0.818,
            "independent_first_reproduction_time": 12,
            "independent_reproduction_efficiency": 0.72,
            "independent_avg_resources_per_reproduction": 48.3,
            "independent_avg_offspring_resources": 24.1,
            "control_reproduction_attempts": 18,
            "control_reproduction_successes": 14,
            "control_reproduction_failures": 4,
            "control_reproduction_success_rate": 0.778,
            "control_first_reproduction_time": 15,
            "control_reproduction_efficiency": 0.68,
            "control_avg_resources_per_reproduction": 45.7,
            "control_avg_offspring_resources": 22.8,
        }

        result = _create_reproduction_stats(row, sim_id=1)

        # Should create ReproductionStats object
        assert result is not None
        assert result.simulation_id == 1
        assert result.system_reproduction_attempts == 25
        assert result.system_reproduction_success_rate == 0.8
        assert result.independent_reproduction_successes == 18


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
        step.agent_type_counts = {"system": 10, "independent": 10, "control": 10}

        mock_session.query.return_value.order_by.return_value.first.return_value = step

        computer = DominanceComputer()
        result = computer.compute_population_dominance(mock_session)

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

        computer = DominanceComputer()
        result = computer.compute_survival_dominance(mock_session)

        # Should compute based on birth time
        assert result in ["system", "independent", "control"] or result is None

    def test_analyze_switch_factors_empty_df(self):
        """Test analyzing switch factors with empty DataFrame."""
        computer = DominanceComputer()

        result = computer.compute_dominance_switch_factors(pd.DataFrame())

        # Should handle empty DataFrame
        assert result is None

    def test_analyze_switch_factors_no_switches_column(self):
        """Test analyzing switch factors without switches column."""
        computer = DominanceComputer()

        df = pd.DataFrame({"iteration": [1, 2, 3]})
        result = computer.compute_dominance_switch_factors(df)

        # Should handle missing column
        assert result is None

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
            step.agent_type_counts = {"system": 0, "independent": 0, "control": 0}
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        # Should handle no agents gracefully
        assert result is not None
        assert result["total_switches"] == 0

    def test_dominance_analysis_with_nan_values(self):
        """Test dominance analysis with NaN values."""
        computer = DominanceComputer()

        df = pd.DataFrame(
            {
                "iteration": range(10),
                "total_switches": [3, np.nan, 2, 4, np.nan, 3, 5, 4, 2, 3],
                "comprehensive_dominance": ["system"] * 10,
                "system_reproduction_attempts": [10, 12, np.nan, 14, 15, np.nan, 13, 11, 12, 10],
            }
        )

        result = computer.compute_dominance_switch_factors(df)

        # Should handle NaN values
        assert isinstance(result, dict)

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
            step.agent_type_counts = {"system": 20, "independent": 0, "control": 0}
            steps.append(step)

        mock_session.query.return_value.order_by.return_value.all.return_value = steps
        mock_session.query.return_value.scalar.return_value = 50

        computer = DominanceComputer()
        result = computer.compute_comprehensive_dominance(mock_session)

        assert result["dominant_type"] == "system"

    @patch("farm.analysis.dominance.compute.logger")
    def test_compute_with_logging(self, mock_logger, mock_session):
        """Test that computation logs appropriately."""
        mock_session.query.return_value.order_by.return_value.first.return_value = None

        computer = DominanceComputer()
        computer.compute_population_dominance(mock_session)

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
        final_step.agent_type_counts = {"system": 20, "independent": 15, "control": 10}

        mock_session.query.return_value.order_by.return_value.first.return_value = final_step

        result = get_final_population_counts(mock_session)

        assert isinstance(result, dict)
        assert "system_agents" in result

    def test_save_dominance_data_to_db(self, sample_dominance_data):
        """Test saving dominance data to database."""
        from farm.analysis.dominance.db_io import save_dominance_data_to_db

        with patch("farm.analysis.dominance.db_io.import_multi_table_data") as mock_import, patch(
            "farm.analysis.dominance.db_io.init_db"
        ) as mock_init, patch("farm.analysis.dominance.db_io.get_session") as mock_get_session:
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

    @patch("farm.analysis.dominance.query_dominance_db.create_engine")
    @patch("farm.analysis.dominance.query_dominance_db.sessionmaker")
    def test_load_data_from_db(self, mock_sessionmaker, mock_create_engine):
        """Test loading data from database."""
        # Mock database components
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_session_class = MagicMock()
        mock_sessionmaker.return_value = mock_session_class

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock DataFrame from SQL query
        mock_df = pd.DataFrame(
            {"iteration": [0, 1], "population_dominance": ["system", "independent"], "system_agents": [20, 15]}
        )

        with patch("farm.analysis.dominance.query_dominance_db.pd.read_sql", return_value=mock_df):
            result = load_data_from_db("sqlite:///test.db")

            # Should return DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "iteration" in result.columns
            assert "population_dominance" in result.columns

            # Should have called database functions
            mock_create_engine.assert_called_once_with("sqlite:///test.db")
            mock_sessionmaker.assert_called_once()

    @patch("farm.analysis.dominance.query_dominance_db.create_engine")
    @patch("farm.analysis.dominance.query_dominance_db.sessionmaker")
    def test_load_data_from_db_error(self, mock_sessionmaker, mock_create_engine):
        """Test loading data from database with error."""
        # Mock database components to raise exception
        mock_create_engine.side_effect = Exception("Database connection error")

        result = load_data_from_db("sqlite:///test.db")

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("farm.analysis.dominance.validation.DominanceDataModel")
    def test_validate_sim_data_with_model(self, mock_model):
        """Test validating simulation data with DominanceDataModel available."""
        # Mock model to return validated data
        mock_instance = MagicMock()
        mock_instance.dict.return_value = {"validated": True, "data": "test"}
        mock_model.return_value = mock_instance

        test_data = {"raw": "data", "value": 42}

        result = validate_sim_data(test_data)

        # Should use model validation
        assert result == {"validated": True, "data": "test"}
        mock_model.assert_called_once_with(**test_data)

    @patch("farm.analysis.dominance.validation.DominanceDataModel", None)
    def test_validate_sim_data_without_model(self):
        """Test validating simulation data when DominanceDataModel is not available."""
        test_data = {"raw": "data", "value": 42}

        result = validate_sim_data(test_data)

        # Should return original data unchanged
        assert result is test_data

    @patch("farm.analysis.dominance.validation.DominanceDataModel")
    def test_validate_sim_data_model_error(self, mock_model):
        """Test validating simulation data when model validation fails."""
        # Mock model to raise exception
        mock_model.side_effect = Exception("Validation error")

        test_data = {"raw": "data", "value": 42}

        result = validate_sim_data(test_data)

        # Should return original data on validation error
        assert result is test_data

    @patch("farm.analysis.dominance.query_dominance_db.print")
    def test_query_dominance_metrics(self, mock_print, mock_session):
        """Test querying dominance metrics."""
        from farm.analysis.dominance.query_dominance_db import query_dominance_metrics

        # Mock query results
        dominance_result = [("system", 10), ("independent", 5)]
        avg_result = MagicMock()
        avg_result.avg_system = 0.8
        avg_result.avg_independent = 0.6
        avg_result.avg_control = 0.4
        top_result = [(0, 0.9, 0.5, 0.3), (1, 0.85, 0.6, 0.4)]

        mock_session.query.return_value.group_by.return_value.all.return_value = dominance_result
        mock_session.query.return_value.one.return_value = avg_result
        mock_session.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = (
            top_result
        )

        # Should execute without error
        query_dominance_metrics(mock_session)

        # Should have made expected queries
        assert mock_session.query.call_count >= 3

    @patch("farm.analysis.dominance.query_dominance_db.print")
    def test_query_agent_populations(self, mock_print, mock_session):
        """Test querying agent populations."""
        from farm.analysis.dominance.query_dominance_db import query_agent_populations

        # Mock query results
        avg_counts = MagicMock()
        avg_counts.avg_system = 20.5
        avg_counts.avg_independent = 15.3
        avg_counts.avg_control = 10.2
        avg_counts.avg_total = 46.0

        avg_survival = MagicMock()
        avg_survival.avg_system_survival = 85.2
        avg_survival.avg_independent_survival = 78.5
        avg_survival.avg_control_survival = 72.1

        mock_session.query.return_value.one.side_effect = [avg_counts, avg_survival]

        # Should execute without error
        query_agent_populations(mock_session)

        # Should have made expected queries
        assert mock_session.query.call_count == 2

    @patch("farm.analysis.dominance.query_dominance_db.print")
    def test_query_reproduction_stats(self, mock_print, mock_session):
        """Test querying reproduction stats."""
        from farm.analysis.dominance.query_dominance_db import query_reproduction_stats

        # Mock query results - need to handle multiple .one() calls (3 total)
        success_rates = MagicMock()
        success_rates.avg_system = 0.75
        success_rates.avg_independent = 0.68
        success_rates.avg_control = 0.62

        first_repro_times = MagicMock()
        first_repro_times.avg_system = 15.2
        first_repro_times.avg_independent = 18.5
        first_repro_times.avg_control = 22.1

        efficiency = MagicMock()
        efficiency.avg_system = 0.82
        efficiency.avg_independent = 0.75
        efficiency.avg_control = 0.68

        # Set up the side_effect to return different results for different calls
        mock_session.query.return_value.one.side_effect = [success_rates, first_repro_times, efficiency]

        # Should execute without error
        query_reproduction_stats(mock_session)

        # Should have made expected queries
        assert mock_session.query.call_count == 3
