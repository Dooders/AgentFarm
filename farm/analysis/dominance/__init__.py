from farm.analysis.dominance.analyze import process_dominance_data
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
