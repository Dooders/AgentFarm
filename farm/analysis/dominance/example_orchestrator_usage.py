"""
Example usage of the DominanceAnalysisOrchestrator.

This file demonstrates various ways to use the orchestrator for dominance analysis,
including simple usage, advanced patterns, and testing approaches.
"""

from typing import Any, Dict, Optional
from unittest.mock import Mock

import pandas as pd


# ============================================================================
# Example 1: Simple Usage (Recommended)
# ============================================================================

def example_1_simple_usage():
    """
    Simplest way to use the orchestrator.
    
    This is the recommended approach for most use cases.
    """
    from farm.analysis.dominance import get_orchestrator
    
    # Get pre-configured orchestrator
    orchestrator = get_orchestrator()
    
    # Use it for computation (assuming you have a session)
    # session = get_database_session()  # Your session
    
    # Compute dominance metrics
    # pop_dom = orchestrator.compute_population_dominance(session)
    # surv_dom = orchestrator.compute_survival_dominance(session)
    # comp_dom = orchestrator.compute_comprehensive_dominance(session)
    
    print("✅ Example 1: Simple usage - Use get_orchestrator()")


# ============================================================================
# Example 2: High-Level Workflow
# ============================================================================

def example_2_full_analysis():
    """
    Run complete analysis workflow using orchestrator.
    
    This shows how to use the high-level orchestration methods.
    """
    from farm.analysis.dominance import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    # Run complete analysis (assuming you have session and config)
    # session = get_database_session()
    # config = get_simulation_config()
    
    # Single method for complete analysis
    # results = orchestrator.run_full_analysis(session, config)
    
    # Access comprehensive results
    # print(f"Population dominance: {results['population_dominance']}")
    # print(f"Survival dominance: {results['survival_dominance']}")
    # print(f"Comprehensive: {results['comprehensive_dominance']['dominant_type']}")
    # print(f"Switches: {results['dominance_switches']['total_switches']}")
    # print(f"Initial data: {results['initial_data']}")
    # print(f"Survival stats: {results['survival_stats']}")
    
    print("✅ Example 2: Full analysis - Use run_full_analysis()")


# ============================================================================
# Example 3: DataFrame Analysis
# ============================================================================

def example_3_dataframe_analysis():
    """
    Analyze a DataFrame of simulation results.
    
    This shows auto-detection of reproduction columns.
    """
    from farm.analysis.dominance import get_orchestrator
    
    # Sample data
    df = pd.DataFrame({
        'iteration': range(10),
        'total_switches': [2, 3, 1, 4, 2, 3, 5, 2, 1, 3],
        'system_reproduction_success_rate': [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.5, 0.9, 0.8, 0.7],
        'independent_reproduction_success_rate': [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.9, 0.5, 0.6, 0.7],
        'comprehensive_dominance': ['system', 'system', 'independent', 'system', 'system',
                                   'independent', 'control', 'system', 'system', 'independent'],
        'switches_per_step': [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.05, 0.02, 0.01, 0.03],
    })
    
    orchestrator = get_orchestrator()
    
    # Comprehensive analysis (auto-detects reproduction columns)
    df_analyzed = orchestrator.analyze_dataframe_comprehensively(df)
    
    print(f"✅ Example 3: DataFrame analysis")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   After analysis: {len(df_analyzed.columns)}")
    print(f"   Columns added: {len(df_analyzed.columns) - len(df.columns)}")


# ============================================================================
# Example 4: Custom Implementation
# ============================================================================

def example_4_custom_implementation():
    """
    Create orchestrator with custom implementations.
    
    This shows how to provide custom analysis logic.
    """
    from farm.analysis.dominance import create_dominance_orchestrator
    from farm.analysis.dominance.interfaces import DominanceComputerProtocol
    
    class CustomComputer:
        """Custom computer with different weighting."""
        
        def __init__(self, analyzer=None):
            self.analyzer = analyzer
        
        def compute_comprehensive_dominance(self, sim_session) -> Optional[Dict[str, Any]]:
            """Custom implementation with different metric weights."""
            # Your custom logic here
            return {
                'dominant_type': 'custom_result',
                'scores': {'system': 0.5, 'independent': 0.3, 'control': 0.2},
                'metrics': {},
                'normalized_metrics': {},
            }
        
        # Implement other required protocol methods...
        def compute_population_dominance(self, sim_session):
            return "system"
        
        def compute_survival_dominance(self, sim_session):
            return "system"
        
        def compute_dominance_switches(self, sim_session):
            return None
        
        def compute_dominance_switch_factors(self, df):
            return None
        
        def aggregate_reproduction_analysis_results(self, df, numeric_repro_cols):
            return {}
    
    # Create orchestrator with custom computer
    orchestrator = create_dominance_orchestrator(
        custom_computer=CustomComputer()
    )
    
    # Use custom implementation
    # result = orchestrator.compute_comprehensive_dominance(session)
    
    print("✅ Example 4: Custom implementation - Use create_dominance_orchestrator()")


# ============================================================================
# Example 5: Testing with Mocks
# ============================================================================

def example_5_testing_with_mocks():
    """
    Write testable code using the orchestrator.
    
    This shows the testing benefits of the orchestrator pattern.
    """
    from farm.analysis.dominance import DominanceAnalysisOrchestrator
    
    # Create mock components
    mock_computer = Mock()
    mock_computer.compute_population_dominance.return_value = "system"
    mock_computer.compute_survival_dominance.return_value = "independent"
    mock_computer.compute_comprehensive_dominance.return_value = {
        'dominant_type': 'system',
        'scores': {'system': 0.6, 'independent': 0.3, 'control': 0.1}
    }
    
    mock_analyzer = Mock()
    mock_data_provider = Mock()
    mock_data_provider.get_final_population_counts.return_value = {
        'system_agents': 10,
        'independent_agents': 5,
        'control_agents': 3
    }
    
    # Create orchestrator with mocks
    orchestrator = DominanceAnalysisOrchestrator(
        computer=mock_computer,
        analyzer=mock_analyzer,
        data_provider=mock_data_provider
    )
    
    # Test orchestrator methods
    mock_session = Mock()
    result = orchestrator.compute_population_dominance(mock_session)
    
    # Verify
    assert result == "system"
    mock_computer.compute_population_dominance.assert_called_once_with(mock_session)
    
    print("✅ Example 5: Testing with mocks - Easy unit testing")


# ============================================================================
# Example 6: Multi-Simulation Analysis
# ============================================================================

def example_6_multi_simulation_analysis():
    """
    Analyze multiple simulations using the orchestrator.
    
    This shows how to efficiently process multiple simulations.
    """
    from farm.analysis.dominance import get_orchestrator
    
    # Create orchestrator once, reuse for all simulations
    orchestrator = get_orchestrator()
    
    # Collect results from multiple simulations
    all_results = []
    
    # for session in simulation_sessions:  # Your list of sessions
    #     results = orchestrator.run_full_analysis(session, config)
    #     all_results.append(results)
    
    # Convert to DataFrame
    # df = pd.DataFrame(all_results)
    
    # Comprehensive DataFrame analysis
    # df = orchestrator.analyze_dataframe_comprehensively(df)
    
    print("✅ Example 6: Multi-simulation - Reuse orchestrator instance")


# ============================================================================
# Example 7: Backward Compatibility
# ============================================================================

def example_7_backward_compatibility():
    """
    Show that legacy API still works.
    
    This demonstrates that existing code doesn't need to change.
    """
    # Legacy imports still work
    from farm.analysis.dominance.compute import compute_population_dominance
    from farm.analysis.dominance.analyze import analyze_dominance_switch_factors
    
    # Legacy function calls still work
    # result = compute_population_dominance(session)
    # df = analyze_dominance_switch_factors(df)
    
    print("✅ Example 7: Backward compatibility - Legacy API still works")


# ============================================================================
# Example 8: Component Access
# ============================================================================

def example_8_component_access():
    """
    Access individual components through the orchestrator.
    
    This shows how to work with individual components.
    """
    from farm.analysis.dominance import get_orchestrator
    
    orchestrator = get_orchestrator()
    
    # Access individual components
    computer = orchestrator.computer
    analyzer = orchestrator.analyzer
    data_provider = orchestrator.data_provider
    
    # Use components directly if needed
    # result = computer.compute_population_dominance(session)
    # df = analyzer.analyze_by_agent_type(df, numeric_cols)
    # data = data_provider.get_reproduction_stats(session)
    
    # Verify bidirectional wiring
    assert computer.analyzer is analyzer
    assert analyzer.computer is computer
    
    print("✅ Example 8: Component access - Direct component usage")
    print(f"   Computer: {type(computer).__name__}")
    print(f"   Analyzer: {type(analyzer).__name__}")
    print(f"   Data Provider: {type(data_provider).__name__}")


# ============================================================================
# Example 9: Real-World Usage Pattern
# ============================================================================

def example_9_real_world_pattern():
    """
    Realistic usage pattern for analysis pipeline.
    
    This demonstrates a complete analysis pipeline.
    """
    from farm.analysis.dominance import get_orchestrator
    
    def analyze_experiment(experiment_path: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze all simulations in an experiment.
        
        Parameters
        ----------
        experiment_path : str
            Path to experiment directory
        config : dict
            Experiment configuration
            
        Returns
        -------
        pd.DataFrame
            Comprehensive analysis results
        """
        orchestrator = get_orchestrator()
        results = []
        
        # Process each simulation
        # for sim_db in get_simulation_databases(experiment_path):
        #     session = connect_to_database(sim_db)
        #     
        #     # Run full analysis
        #     sim_results = orchestrator.run_full_analysis(session, config)
        #     results.append(sim_results)
        #     
        #     session.close()
        
        # Convert to DataFrame
        # df = pd.DataFrame(results)
        
        # Comprehensive DataFrame analysis
        # df = orchestrator.analyze_dataframe_comprehensively(df)
        
        # return df
        pass
    
    print("✅ Example 9: Real-world pattern - Complete analysis pipeline")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DOMINANCE ANALYSIS ORCHESTRATOR - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    example_1_simple_usage()
    example_2_full_analysis()
    example_3_dataframe_analysis()
    example_4_custom_implementation()
    example_5_testing_with_mocks()
    example_6_multi_simulation_analysis()
    example_7_backward_compatibility()
    example_8_component_access()
    example_9_real_world_pattern()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nFor more details, see:")
    print("  - ORCHESTRATOR_GUIDE.md - Complete API reference")
    print("  - MIGRATION_GUIDE.md - Migration from legacy API")
    print("  - orchestrator.py - Source code")
