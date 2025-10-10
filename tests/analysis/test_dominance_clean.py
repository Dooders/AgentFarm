"""
Updated dominance analysis tests for clean class-based architecture.

This file contains modernized tests that use the orchestrator pattern
and protocol-based design without backward compatibility wrappers.
"""

from unittest.mock import Mock, MagicMock
import numpy as np
import pandas as pd
import pytest

from farm.analysis.dominance import (
    get_orchestrator,
    create_dominance_orchestrator,
    DominanceComputer,
    DominanceAnalyzer,
    DominanceDataProvider,
)
from farm.analysis.dominance.mocks import (
    MockDominanceComputer,
    MockDominanceAnalyzer,
    MockDominanceDataProvider,
    create_sample_simulation_data,
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock()


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
        step.total_agents = step.system_agents + step.independent_agents + step.control_agents
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
        agent.state_manager.birth_time = i
        agent.death_time = i + 50 if i % 2 == 0 else None
        agents.append(agent)
    return agents


class TestDominanceComputerClass:
    """Test DominanceComputer class directly."""

    def test_computer_instantiation(self):
        """Test creating DominanceComputer."""
        computer = DominanceComputer()
        assert computer is not None
        assert computer.analyzer is None

    def test_computer_with_analyzer_dependency(self):
        """Test creating DominanceComputer with analyzer."""
        analyzer = DominanceAnalyzer()
        computer = DominanceComputer(analyzer=analyzer)
        assert computer.analyzer is analyzer

    def test_compute_population_dominance(self, mock_session, mock_simulation_steps):
        """Test population dominance computation."""
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        computer = DominanceComputer()
        result = computer.compute_population_dominance(mock_session)

        assert result in ["system", "independent", "control"]

    def test_compute_survival_dominance(self, mock_session, mock_agents, mock_simulation_steps):
        """Test survival dominance computation."""
        mock_session.query.return_value.all.return_value = mock_agents
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        computer = DominanceComputer()
        result = computer.compute_survival_dominance(mock_session)

        assert result in ["system", "independent", "control"] or result is None

    def test_compute_comprehensive_dominance(self, mock_session, mock_simulation_steps):
        """Test comprehensive dominance computation."""
        mock_session.query.return_value.order_by.return_value.all.return_value = mock_simulation_steps

        computer = DominanceComputer()
        result = computer.compute_comprehensive_dominance(mock_session)

        assert result is not None
        assert "dominant_type" in result
        assert "scores" in result
        assert "metrics" in result

    def test_compute_dominance_switches(self, mock_session, mock_simulation_steps):
        """Test dominance switches computation."""
        mock_session.query.return_value.order_by.return_value.all.return_value = mock_simulation_steps

        computer = DominanceComputer()
        result = computer.compute_dominance_switches(mock_session)

        assert isinstance(result, dict)
        assert "total_switches" in result
        assert "switches_per_step" in result

    def test_compute_dominance_switch_factors(self):
        """Test computing switch factors."""
        df = create_sample_simulation_data()

        computer = DominanceComputer()
        result = computer.compute_dominance_switch_factors(df)

        assert result is not None
        assert "reproduction_correlations" in result
        assert "switches_by_dominant_type" in result


class TestDominanceAnalyzerClass:
    """Test DominanceAnalyzer class directly."""

    def test_analyzer_instantiation(self):
        """Test creating DominanceAnalyzer."""
        analyzer = DominanceAnalyzer()
        assert analyzer is not None
        assert analyzer.computer is None

    def test_analyzer_with_computer_dependency(self):
        """Test creating DominanceAnalyzer with computer."""
        computer = DominanceComputer()
        analyzer = DominanceAnalyzer(computer=computer)
        assert analyzer.computer is computer

    def test_analyze_by_agent_type(self):
        """Test analyze by agent type."""
        df = create_sample_simulation_data()
        numeric_cols = ["system_reproduction_success_rate"]

        analyzer = DominanceAnalyzer()
        result = analyzer.analyze_by_agent_type(df, numeric_cols)

        assert isinstance(result, pd.DataFrame)

    def test_analyze_high_vs_low_switching(self):
        """Test high vs low switching analysis."""
        df = create_sample_simulation_data()
        numeric_cols = ["system_reproduction_success_rate"]

        analyzer = DominanceAnalyzer()
        result = analyzer.analyze_high_vs_low_switching(df, numeric_cols)

        assert isinstance(result, pd.DataFrame)
        # Check that columns were added
        assert any("high_switching_mean" in col for col in result.columns)


class TestDominanceDataProviderClass:
    """Test DominanceDataProvider class directly."""

    def test_provider_instantiation(self):
        """Test creating DominanceDataProvider."""
        provider = DominanceDataProvider()
        assert provider is not None

    def test_get_final_population_counts(self, mock_session, mock_simulation_steps):
        """Test getting final population counts."""
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        provider = DominanceDataProvider()
        result = provider.get_final_population_counts(mock_session)

        assert result is not None
        assert "system_agents" in result


class TestOrchestratorIntegration:
    """Test orchestrator integration scenarios."""

    def test_get_orchestrator_singleton(self):
        """Test that get_orchestrator returns same instance."""
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        assert orch1 is orch2

    def test_orchestrator_has_all_components(self):
        """Test orchestrator has all components wired."""
        orchestrator = get_orchestrator()

        assert orchestrator.computer is not None
        assert orchestrator.analyzer is not None
        assert orchestrator.data_provider is not None
        assert orchestrator.computer.analyzer is orchestrator.analyzer
        assert orchestrator.analyzer.computer is orchestrator.computer

    def test_orchestrator_compute_methods(self, mock_session, mock_simulation_steps):
        """Test orchestrator computation methods."""
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_simulation_steps[-1]

        orchestrator = get_orchestrator()
        result = orchestrator.compute_population_dominance(mock_session)

        assert result in ["system", "independent", "control"]

    def test_orchestrator_analyze_dataframe(self):
        """Test orchestrator DataFrame analysis."""
        df = create_sample_simulation_data()

        orchestrator = get_orchestrator()
        result = orchestrator.analyze_dataframe_comprehensively(df)

        assert isinstance(result, pd.DataFrame)
        # Should have more columns after analysis
        assert len(result.columns) >= len(df.columns)


class TestMockImplementations:
    """Test mock implementations work correctly."""

    def test_mock_computer_returns_predictable_values(self):
        """Test MockDominanceComputer returns expected values."""
        mock = MockDominanceComputer()
        mock_session = Mock()

        result = mock.compute_population_dominance(mock_session)
        assert result == "system"  # Predictable mock value

    def test_mock_computer_tracks_calls(self):
        """Test MockDominanceComputer tracks calls."""
        mock = MockDominanceComputer()
        mock_session = Mock()

        mock.compute_population_dominance(mock_session)
        mock.compute_population_dominance(mock_session)

        assert mock.call_count["compute_population_dominance"] == 2

    def test_mock_analyzer_works(self):
        """Test MockDominanceAnalyzer works."""
        mock = MockDominanceAnalyzer()
        df = create_sample_simulation_data()

        result = mock.analyze_by_agent_type(df, ["system_reproduction_success_rate"])

        assert isinstance(result, pd.DataFrame)
        assert "agent_type_analysis_done" in result.columns


class TestProtocolCompliance:
    """Test that classes comply with their protocols."""

    def test_computer_implements_all_methods(self):
        """Test DominanceComputer has all protocol methods."""
        computer = DominanceComputer()

        required_methods = [
            "compute_population_dominance",
            "compute_survival_dominance",
            "compute_comprehensive_dominance",
            "compute_dominance_switches",
            "compute_dominance_switch_factors",
            "aggregate_reproduction_analysis_results",
        ]

        for method in required_methods:
            assert hasattr(computer, method)
            assert callable(getattr(computer, method))

    def test_analyzer_implements_all_methods(self):
        """Test DominanceAnalyzer has all protocol methods."""
        analyzer = DominanceAnalyzer()

        required_methods = [
            "analyze_by_agent_type",
            "analyze_high_vs_low_switching",
            "analyze_reproduction_advantage",
            "analyze_reproduction_efficiency",
            "analyze_reproduction_timing",
            "analyze_dominance_switch_factors",
            "analyze_reproduction_dominance_switching",
        ]

        for method in required_methods:
            assert hasattr(analyzer, method)
            assert callable(getattr(analyzer, method))


class TestDependencyInjection:
    """Test dependency injection patterns work correctly."""

    def test_computer_accepts_analyzer(self):
        """Test computer accepts analyzer via DI."""
        analyzer = DominanceAnalyzer()
        computer = DominanceComputer(analyzer=analyzer)

        assert computer.analyzer is analyzer

    def test_analyzer_accepts_computer(self):
        """Test analyzer accepts computer via DI."""
        computer = DominanceComputer()
        analyzer = DominanceAnalyzer(computer=computer)

        assert analyzer.computer is computer

    def test_bidirectional_injection(self):
        """Test bidirectional dependency injection."""
        computer = DominanceComputer()
        analyzer = DominanceAnalyzer(computer=computer)
        computer.analyzer = analyzer

        assert computer.analyzer is analyzer
        assert analyzer.computer is computer


class TestOrchestratorCreation:
    """Test orchestrator creation patterns."""

    def test_create_orchestrator_factory(self):
        """Test creating orchestrator via factory."""
        orchestrator = create_dominance_orchestrator()

        assert orchestrator is not None
        assert orchestrator.computer is not None
        assert orchestrator.analyzer is not None

    def test_create_orchestrator_with_custom_computer(self):
        """Test creating orchestrator with custom computer."""
        custom_computer = MockDominanceComputer()
        orchestrator = create_dominance_orchestrator(custom_computer=custom_computer)

        assert orchestrator.computer is custom_computer

    def test_orchestrator_wires_dependencies(self):
        """Test orchestrator wires dependencies correctly."""
        orchestrator = create_dominance_orchestrator()

        # Verify bidirectional wiring
        assert orchestrator.computer.analyzer is orchestrator.analyzer
        assert orchestrator.analyzer.computer is orchestrator.computer


# Run pytest if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
