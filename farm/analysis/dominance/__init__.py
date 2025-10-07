"""
Dominance Analysis Module

This module provides comprehensive dominance analysis using a protocol-based
architecture with dependency injection. The circular dependency between
analyze.py and compute.py has been eliminated.

Usage:
    from farm.analysis.dominance import get_orchestrator
    
    orchestrator = get_orchestrator()
    result = orchestrator.compute_population_dominance(session)
    df = orchestrator.analyze_dataframe_comprehensively(df)
"""

# Core orchestration - recommended for all new code
from farm.analysis.dominance.orchestrator import (
    DominanceAnalysisOrchestrator,
    create_dominance_orchestrator,
)

# Protocol interfaces
from farm.analysis.dominance.interfaces import (
    DominanceAnalyzerProtocol,
    DominanceComputerProtocol,
    DominanceDataProviderProtocol,
)

# Concrete implementations
from farm.analysis.dominance.compute import DominanceComputer
from farm.analysis.dominance.implementations import (
    DominanceAnalyzer,
    DominanceDataProvider,
)

# Legacy functions (retained for specific use cases)
from farm.analysis.dominance.analyze import (
    process_dominance_data,
    save_dominance_data_to_db,
)

# ML and plotting functions
from farm.analysis.dominance.ml import run_dominance_classification
from farm.analysis.dominance.plot import (
    plot_comprehensive_score_breakdown,
    plot_correlation_matrix,
    plot_dominance_comparison,
    plot_dominance_distribution,
    plot_dominance_stability,
    plot_dominance_switches,
    plot_reproduction_advantage_vs_stability,
    plot_reproduction_success_vs_switching,
    plot_reproduction_vs_dominance,
    plot_resource_proximity_vs_dominance,
)
from farm.analysis.dominance.query_dominance_db import load_data_from_db

# Convenience: Create a default orchestrator instance for easy access
_default_orchestrator = create_dominance_orchestrator()


def get_orchestrator() -> DominanceAnalysisOrchestrator:
    """
    Get the default dominance analysis orchestrator instance.
    
    This is the recommended way to access dominance analysis functionality.
    
    Returns
    -------
    DominanceAnalysisOrchestrator
        The default orchestrator with all dependencies wired
        
    Example
    -------
    >>> from farm.analysis.dominance import get_orchestrator
    >>> orchestrator = get_orchestrator()
    >>> result = orchestrator.compute_population_dominance(session)
    >>> df = orchestrator.analyze_dataframe_comprehensively(df)
    """
    return _default_orchestrator

# Dictionary to store all analysis functions
_ANALYSIS_FUNCTIONS = {}
_ANALYSIS_GROUPS = {}


def register_analysis():
    """
    Register all analysis functions for the dominance module.
    This makes it easy to get all functions or specific groups of functions.
    """
    global _ANALYSIS_FUNCTIONS, _ANALYSIS_GROUPS

    # Register all plot functions
    _ANALYSIS_FUNCTIONS.update(
        {
            "plot_dominance_distribution": plot_dominance_distribution,
            "plot_comprehensive_score_breakdown": plot_comprehensive_score_breakdown,
            "plot_dominance_switches": plot_dominance_switches,
            "plot_dominance_stability": plot_dominance_stability,
            "plot_reproduction_success_vs_switching": plot_reproduction_success_vs_switching,
            "plot_reproduction_advantage_vs_stability": plot_reproduction_advantage_vs_stability,
            "plot_resource_proximity_vs_dominance": plot_resource_proximity_vs_dominance,
            "plot_reproduction_vs_dominance": plot_reproduction_vs_dominance,
            "plot_dominance_comparison": plot_dominance_comparison,
            "plot_correlation_matrix": lambda df, output_path: plot_correlation_matrix(
                df, label_name="comprehensive_dominance", output_path=output_path
            ),
        }
    )

    # Register ML functions
    _ANALYSIS_FUNCTIONS.update(
        {
            "run_dominance_classification": run_dominance_classification,
        }
    )

    # Define function groups for easier access
    _ANALYSIS_GROUPS = {
        "all": list(_ANALYSIS_FUNCTIONS.values()),
        "plots": [
            plot_dominance_distribution,
            plot_comprehensive_score_breakdown,
            plot_dominance_switches,
            plot_dominance_stability,
            plot_reproduction_success_vs_switching,
            plot_reproduction_advantage_vs_stability,
            plot_resource_proximity_vs_dominance,
            plot_reproduction_vs_dominance,
            plot_dominance_comparison,
        ],
        "ml": [run_dominance_classification],
        "correlation": [_ANALYSIS_FUNCTIONS["plot_correlation_matrix"]],
        "basic": [
            plot_dominance_distribution,
            plot_comprehensive_score_breakdown,
            plot_dominance_comparison,
        ],
        "reproduction": [
            plot_reproduction_success_vs_switching,
            plot_reproduction_advantage_vs_stability,
            plot_reproduction_vs_dominance,
        ],
        "switching": [
            plot_dominance_switches,
            plot_dominance_stability,
        ],
    }


def get_analysis_function(name):
    """
    Get a specific analysis function by name.

    Parameters
    ----------
    name : str
        Name of the analysis function

    Returns
    -------
    callable
        The requested analysis function
    """
    if not _ANALYSIS_FUNCTIONS:
        register_analysis()
    return _ANALYSIS_FUNCTIONS.get(name)


def get_analysis_functions(group="all"):
    """
    Get a list of analysis functions by group.

    Parameters
    ----------
    group : str
        Name of the function group (all, plots, ml, correlation, basic, reproduction, switching)

    Returns
    -------
    list
        List of analysis functions in the requested group
    """
    if not _ANALYSIS_GROUPS:
        register_analysis()
    return _ANALYSIS_GROUPS.get(group, [])


def get_module_info():
    """
    Get information about the dominance analysis module.

    Returns
    -------
    dict
        Dictionary containing module information
    """
    if not _ANALYSIS_GROUPS:
        register_analysis()

    return {
        "name": "dominance",
        "description": "Analysis of agent dominance patterns in simulations",
        "data_processor": process_dominance_data,
        "db_loader": load_data_from_db,
        "db_filename": "dominance.db",
        "function_groups": list(_ANALYSIS_GROUPS.keys()),
        "functions": list(_ANALYSIS_FUNCTIONS.keys()),
    }


# Register analysis functions when module is imported
register_analysis()

# Import the module instance
from farm.analysis.dominance.module import dominance_module

# Define __all__ for explicit exports
__all__ = [
    # Orchestrator (recommended entry point)
    "DominanceAnalysisOrchestrator",
    "create_dominance_orchestrator",
    "get_orchestrator",
    # Protocols (interfaces)
    "DominanceAnalyzerProtocol",
    "DominanceComputerProtocol",
    "DominanceDataProviderProtocol",
    # Concrete implementations
    "DominanceComputer",
    "DominanceAnalyzer",
    "DominanceDataProvider",
    # Legacy functions (specific use cases)
    "process_dominance_data",
    "save_dominance_data_to_db",
    # ML and plotting
    "run_dominance_classification",
    "load_data_from_db",
    "plot_comprehensive_score_breakdown",
    "plot_correlation_matrix",
    "plot_dominance_comparison",
    "plot_dominance_distribution",
    "plot_dominance_stability",
    "plot_dominance_switches",
    "plot_reproduction_advantage_vs_stability",
    "plot_reproduction_success_vs_switching",
    "plot_reproduction_vs_dominance",
    "plot_resource_proximity_vs_dominance",
    # Module utilities
    "get_analysis_function",
    "get_analysis_functions",
    "get_module_info",
    "dominance_module",
]
