"""
Tests for the dominance analysis orchestrator and protocol-based architecture.

This test file demonstrates the benefits of the protocol-based design:
1. Easy mocking with protocol implementations
2. Independent testing of components
3. Dependency injection for better testability
"""

from unittest.mock import Mock, MagicMock
import pandas as pd
import pytest

from farm.analysis.dominance import (
    DominanceAnalysisOrchestrator,
    create_dominance_orchestrator,
    get_orchestrator,
    DominanceComputer,
)
from farm.analysis.dominance.implementations import (
    DominanceAnalyzer,
    DominanceDataProvider,
)
from farm.analysis.dominance.mocks import (
    MockDominanceComputer,
    MockDominanceAnalyzer,
    MockDominanceDataProvider,
    create_mock_computer,
    create_sample_simulation_data,
    create_minimal_simulation_data,
)


class TestOrchestratorCreation:
    """Test orchestrator creation and dependency wiring."""

    def test_create_orchestrator_with_defaults(self):
        """Test creating orchestrator with default implementations."""
        orchestrator = create_dominance_orchestrator()

        assert orchestrator is not None
        assert isinstance(orchestrator, DominanceAnalysisOrchestrator)
        assert orchestrator.computer is not None
        assert orchestrator.analyzer is not None
        assert orchestrator.data_provider is not None

    def test_orchestrator_dependency_wiring(self):
        """Test that orchestrator wires dependencies correctly."""
        orchestrator = create_dominance_orchestrator()

        # Verify bidirectional wiring
        assert orchestrator.computer.analyzer is orchestrator.analyzer
        assert orchestrator.analyzer.computer is orchestrator.computer

    def test_create_orchestrator_with_custom_computer(self):
        """Test creating orchestrator with custom computer."""
        custom_computer = MockDominanceComputer()
        orchestrator = create_dominance_orchestrator(custom_computer=custom_computer)

        assert orchestrator.computer is custom_computer
        assert orchestrator.computer.analyzer is orchestrator.analyzer

    def test_get_orchestrator_returns_singleton(self):
        """Test that get_orchestrator returns the default instance."""
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()

        # Should return the same instance
        assert orch1 is orch2


class TestOrchestratorComputationMethods:
    """Test orchestrator computation method delegation."""

    def test_compute_population_dominance(self):
        """Test compute_population_dominance delegates to computer."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        mock_session = Mock()
        result = orchestrator.compute_population_dominance(mock_session)

        assert result == "system"  # Default mock return value
        assert mock_computer.call_count["compute_population_dominance"] == 1

    def test_compute_survival_dominance(self):
        """Test compute_survival_dominance delegates to computer."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        mock_session = Mock()
        result = orchestrator.compute_survival_dominance(mock_session)

        assert result == "independent"  # Default mock return value
        assert mock_computer.call_count["compute_survival_dominance"] == 1

    def test_compute_comprehensive_dominance(self):
        """Test compute_comprehensive_dominance delegates to computer."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        mock_session = Mock()
        result = orchestrator.compute_comprehensive_dominance(mock_session)

        assert result is not None
        assert result["dominant_type"] == "system"
        assert "scores" in result
        assert mock_computer.call_count["compute_comprehensive_dominance"] == 1

    def test_compute_dominance_switches(self):
        """Test compute_dominance_switches delegates to computer."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        mock_session = Mock()
        result = orchestrator.compute_dominance_switches(mock_session)

        assert result is not None
        assert result["total_switches"] == 5
        assert "switches_per_step" in result
        assert mock_computer.call_count["compute_dominance_switches"] == 1

    def test_compute_dominance_switch_factors(self):
        """Test compute_dominance_switch_factors delegates to computer."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        df = create_sample_simulation_data()
        result = orchestrator.compute_dominance_switch_factors(df)

        assert result is not None
        assert "top_positive_correlations" in result
        assert mock_computer.call_count["compute_dominance_switch_factors"] == 1


class TestOrchestratorAnalysisMethods:
    """Test orchestrator analysis method delegation."""

    def test_analyze_by_agent_type(self):
        """Test analyze_by_agent_type delegates to analyzer."""
        mock_analyzer = MockDominanceAnalyzer()
        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer)

        df = create_sample_simulation_data()
        numeric_cols = ["system_reproduction_success_rate"]
        result_df = orchestrator.analyze_by_agent_type(df, numeric_cols)

        assert "agent_type_analysis_done" in result_df.columns
        assert mock_analyzer.call_count["analyze_by_agent_type"] == 1

    def test_analyze_high_vs_low_switching(self):
        """Test analyze_high_vs_low_switching delegates to analyzer."""
        mock_analyzer = MockDominanceAnalyzer()
        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer)

        df = create_sample_simulation_data()
        numeric_cols = ["system_reproduction_success_rate"]
        result_df = orchestrator.analyze_high_vs_low_switching(df, numeric_cols)

        assert f"{numeric_cols[0]}_high_switching_mean" in result_df.columns
        assert mock_analyzer.call_count["analyze_high_vs_low_switching"] == 1

    def test_analyze_dominance_switch_factors(self):
        """Test analyze_dominance_switch_factors delegates to analyzer."""
        mock_analyzer = MockDominanceAnalyzer()
        mock_computer = MockDominanceComputer()
        mock_analyzer.computer = mock_computer
        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer, computer=mock_computer)

        df = create_sample_simulation_data()
        result_df = orchestrator.analyze_dominance_switch_factors(df)

        assert "positive_corr_resource_proximity" in result_df.columns
        assert mock_analyzer.call_count["analyze_dominance_switch_factors"] == 1


class TestOrchestratorDataProviderMethods:
    """Test orchestrator data provider method delegation."""

    def test_get_final_population_counts(self):
        """Test get_final_population_counts delegates to data provider."""
        mock_provider = MockDominanceDataProvider()
        orchestrator = DominanceAnalysisOrchestrator(data_provider=mock_provider)

        mock_session = Mock()
        result = orchestrator.get_final_population_counts(mock_session)

        assert result is not None
        assert result["system_agents"] == 15
        assert mock_provider.call_count["get_final_population_counts"] == 1

    def test_get_agent_survival_stats(self):
        """Test get_agent_survival_stats delegates to data provider."""
        mock_provider = MockDominanceDataProvider()
        orchestrator = DominanceAnalysisOrchestrator(data_provider=mock_provider)

        mock_session = Mock()
        result = orchestrator.get_agent_survival_stats(mock_session)

        assert result is not None
        assert result["system_count"] == 20
        assert mock_provider.call_count["get_agent_survival_stats"] == 1

    def test_get_reproduction_stats(self):
        """Test get_reproduction_stats delegates to data provider."""
        mock_provider = MockDominanceDataProvider()
        orchestrator = DominanceAnalysisOrchestrator(data_provider=mock_provider)

        mock_session = Mock()
        result = orchestrator.get_reproduction_stats(mock_session)

        assert result is not None
        assert result["system_reproduction_attempts"] == 25
        assert mock_provider.call_count["get_reproduction_stats"] == 1


class TestOrchestratorHighLevelMethods:
    """Test orchestrator high-level orchestration methods."""

    def test_run_full_analysis(self):
        """Test run_full_analysis orchestrates all components."""
        mock_computer = MockDominanceComputer()
        mock_provider = MockDominanceDataProvider()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer, data_provider=mock_provider)

        mock_session = Mock()
        config = {"gathering_range": 30}

        result = orchestrator.run_full_analysis(mock_session, config)

        # Verify all expected keys present
        assert "population_dominance" in result
        assert "survival_dominance" in result
        assert "comprehensive_dominance" in result
        assert "dominance_switches" in result
        assert "initial_data" in result
        assert "final_counts" in result
        assert "survival_stats" in result
        assert "reproduction_stats" in result

        # Verify all methods were called
        assert mock_computer.call_count["compute_population_dominance"] == 1
        assert mock_computer.call_count["compute_survival_dominance"] == 1
        assert mock_computer.call_count["compute_comprehensive_dominance"] == 1
        assert mock_computer.call_count["compute_dominance_switches"] == 1
        assert mock_provider.call_count["get_final_population_counts"] == 1

    def test_analyze_dataframe_comprehensively(self):
        """Test analyze_dataframe_comprehensively with auto-detection."""
        mock_analyzer = MockDominanceAnalyzer()
        mock_computer = MockDominanceComputer()
        mock_analyzer.computer = mock_computer
        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer, computer=mock_computer)

        df = create_sample_simulation_data()
        result_df = orchestrator.analyze_dataframe_comprehensively(df)

        # Verify DataFrame was modified
        assert len(result_df.columns) > len(df.columns)

        # Verify methods were called
        assert mock_analyzer.call_count["analyze_dominance_switch_factors"] == 1
        assert mock_analyzer.call_count["analyze_reproduction_dominance_switching"] == 1


class TestProtocolBasedTesting:
    """Demonstrate benefits of protocol-based testing."""

    def test_isolated_computer_testing(self):
        """Test computer in isolation without real analyzer."""
        # Create computer without analyzer dependency
        computer = DominanceComputer()

        # Mock session
        mock_session = MagicMock()
        mock_step = MagicMock()
        mock_step.system_agents = 10
        mock_step.independent_agents = 5
        mock_step.control_agents = 3
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_step

        # Test computer method
        result = computer.compute_population_dominance(mock_session)

        assert result == "system"

    def test_isolated_analyzer_testing(self):
        """Test analyzer in isolation with mock computer."""
        mock_computer = MockDominanceComputer()
        analyzer = DominanceAnalyzer(computer=mock_computer)

        df = create_sample_simulation_data()
        original_columns = len(df.columns)
        result_df = analyzer.analyze_dominance_switch_factors(df)

        # Verify analyzer called computer
        assert mock_computer.call_count["compute_dominance_switch_factors"] == 1

        # Verify DataFrame modified
        assert len(result_df.columns) > original_columns

    def test_mock_entire_orchestrator(self):
        """Test using completely mocked orchestrator."""
        mock_computer = MockDominanceComputer()
        mock_analyzer = MockDominanceAnalyzer()
        mock_provider = MockDominanceDataProvider()

        orchestrator = DominanceAnalysisOrchestrator(
            computer=mock_computer, analyzer=mock_analyzer, data_provider=mock_provider
        )

        # Test all components work together
        mock_session = Mock()
        config = {}

        result = orchestrator.run_full_analysis(mock_session, config)

        # Verify all components were used
        assert mock_computer.call_count["compute_population_dominance"] == 1
        assert mock_provider.call_count["get_final_population_counts"] == 1


class TestDependencyInjection:
    """Test dependency injection patterns."""

    def test_computer_accepts_analyzer_dependency(self):
        """Test that computer can accept analyzer via DI."""
        analyzer = DominanceAnalyzer()
        computer = DominanceComputer(analyzer=analyzer)

        assert computer.analyzer is analyzer

    def test_analyzer_accepts_computer_dependency(self):
        """Test that analyzer can accept computer via DI."""
        computer = DominanceComputer()
        analyzer = DominanceAnalyzer(computer=computer)

        assert analyzer.computer is computer

    def test_bidirectional_dependency_injection(self):
        """Test bidirectional dependency injection."""
        computer = DominanceComputer()
        analyzer = DominanceAnalyzer(computer=computer)
        computer.analyzer = analyzer

        # Verify bidirectional link
        assert computer.analyzer is analyzer
        assert analyzer.computer is computer


class TestOrchestratorIntegration:
    """Integration tests using mock implementations."""

    def test_full_analysis_with_mocks(self):
        """Test full analysis workflow with all mocks."""
        orchestrator = DominanceAnalysisOrchestrator(
            computer=MockDominanceComputer(),
            analyzer=MockDominanceAnalyzer(),
            data_provider=MockDominanceDataProvider(),
        )

        mock_session = Mock()
        config = {"gathering_range": 30}

        result = orchestrator.run_full_analysis(mock_session, config)

        # Verify comprehensive results
        assert result["population_dominance"] == "system"
        assert result["survival_dominance"] == "independent"
        assert result["comprehensive_dominance"]["dominant_type"] == "system"
        assert result["dominance_switches"]["total_switches"] == 5
        assert result["final_counts"]["system_agents"] == 15
        assert result["survival_stats"]["system_count"] == 20
        assert result["reproduction_stats"]["system_reproduction_attempts"] == 25

    def test_dataframe_analysis_with_mocks(self):
        """Test DataFrame analysis with mock components."""
        mock_analyzer = MockDominanceAnalyzer()
        mock_computer = MockDominanceComputer()
        mock_analyzer.computer = mock_computer

        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer, computer=mock_computer)

        df = create_sample_simulation_data()
        original_cols = len(df.columns)

        result_df = orchestrator.analyze_dataframe_comprehensively(df)

        # Verify DataFrame was enhanced
        assert len(result_df.columns) > original_cols

        # Verify methods were called
        assert mock_analyzer.call_count["analyze_dominance_switch_factors"] == 1
        assert mock_analyzer.call_count["analyze_reproduction_dominance_switching"] == 1


class TestRealImplementations:
    """Test with real implementations (no mocks)."""

    def test_real_computer_instantiation(self):
        """Test that real DominanceComputer can be created."""
        computer = DominanceComputer()
        assert computer is not None
        assert computer.analyzer is None  # No dependency injected

    def test_real_analyzer_instantiation(self):
        """Test that real DominanceAnalyzer can be created."""
        analyzer = DominanceAnalyzer()
        assert analyzer is not None
        assert analyzer.computer is None  # No dependency injected

    def test_real_data_provider_instantiation(self):
        """Test that real DominanceDataProvider can be created."""
        provider = DominanceDataProvider()
        assert provider is not None

    def test_real_orchestrator_creation(self):
        """Test creating orchestrator with real implementations."""
        orchestrator = create_dominance_orchestrator()

        # Verify all components are real implementations
        assert isinstance(orchestrator.computer, DominanceComputer)
        assert isinstance(orchestrator.analyzer, DominanceAnalyzer)
        assert isinstance(orchestrator.data_provider, DominanceDataProvider)

        # Verify wiring
        assert orchestrator.computer.analyzer is orchestrator.analyzer
        assert orchestrator.analyzer.computer is orchestrator.computer


class TestMockCallTracking:
    """Test that mock implementations track calls correctly."""

    def test_mock_computer_tracks_calls(self):
        """Test that mock computer tracks method calls."""
        mock = MockDominanceComputer()
        mock_session = Mock()

        # Make multiple calls
        mock.compute_population_dominance(mock_session)
        mock.compute_population_dominance(mock_session)
        mock.compute_survival_dominance(mock_session)

        # Verify call tracking
        assert mock.call_count["compute_population_dominance"] == 2
        assert mock.call_count["compute_survival_dominance"] == 1

    def test_mock_analyzer_tracks_calls(self):
        """Test that mock analyzer tracks method calls."""
        mock = MockDominanceAnalyzer()
        df = create_minimal_simulation_data()

        # Make multiple calls
        mock.analyze_by_agent_type(df, [])
        mock.analyze_high_vs_low_switching(df, [])
        mock.analyze_by_agent_type(df, [])

        # Verify call tracking
        assert mock.call_count["analyze_by_agent_type"] == 2
        assert mock.call_count["analyze_high_vs_low_switching"] == 1

    def test_mock_data_provider_tracks_calls(self):
        """Test that mock data provider tracks method calls."""
        mock = MockDominanceDataProvider()
        mock_session = Mock()

        # Make multiple calls
        mock.get_final_population_counts(mock_session)
        mock.get_agent_survival_stats(mock_session)
        mock.get_final_population_counts(mock_session)

        # Verify call tracking
        assert mock.call_count["get_final_population_counts"] == 2
        assert mock.call_count["get_agent_survival_stats"] == 1


class TestProtocolCompliance:
    """Test that implementations comply with protocols."""

    def test_computer_has_all_protocol_methods(self):
        """Test that DominanceComputer has all required methods."""
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
            assert hasattr(computer, method), f"Missing method: {method}"
            assert callable(getattr(computer, method))

    def test_analyzer_has_all_protocol_methods(self):
        """Test that DominanceAnalyzer has all required methods."""
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
            assert hasattr(analyzer, method), f"Missing method: {method}"
            assert callable(getattr(analyzer, method))

    def test_data_provider_has_all_protocol_methods(self):
        """Test that DominanceDataProvider has all required methods."""
        provider = DominanceDataProvider()

        required_methods = [
            "get_final_population_counts",
            "get_agent_survival_stats",
            "get_reproduction_stats",
            "get_initial_positions_and_resources",
        ]

        for method in required_methods:
            assert hasattr(provider, method), f"Missing method: {method}"
            assert callable(getattr(provider, method))


class TestOrchestratorEdgeCases:
    """Test edge cases and error handling."""

    def test_orchestrator_with_none_dependencies(self):
        """Test orchestrator handles None dependencies gracefully."""
        orchestrator = DominanceAnalysisOrchestrator(computer=None, analyzer=None, data_provider=None)

        # Should create defaults
        assert orchestrator.computer is not None
        assert orchestrator.analyzer is not None
        assert orchestrator.data_provider is not None

    def test_compute_with_empty_dataframe(self):
        """Test compute_dominance_switch_factors with empty DataFrame."""
        mock_computer = MockDominanceComputer()
        orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

        empty_df = pd.DataFrame()
        result = orchestrator.compute_dominance_switch_factors(empty_df)

        # Should return None for empty DataFrame
        assert result is None

    def test_analyze_with_missing_columns(self):
        """Test analysis with missing required columns."""
        mock_analyzer = MockDominanceAnalyzer()
        orchestrator = DominanceAnalysisOrchestrator(analyzer=mock_analyzer)

        # DataFrame missing 'total_switches' column
        df = pd.DataFrame({"iteration": [1, 2, 3]})

        # Should handle gracefully
        result_df = orchestrator.analyze_dominance_switch_factors(df)
        assert isinstance(result_df, pd.DataFrame)


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
