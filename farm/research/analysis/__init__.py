"""
Analysis package for research simulations.

This package provides tools for analyzing simulation results, including:
- Database interaction functions
- Data processing and statistical calculations
- Plotting functions
- Orchestration and execution logic
"""

from farm.research.analysis.analysis import (
    analyze_final_agent_counts,
    detect_early_terminations,
    find_experiments,
    process_action_distributions,
    process_experiment,
    process_experiment_by_agent_type,
    process_experiment_resource_consumption,
    process_experiment_resource_levels,
    process_experiment_rewards_by_generation,
    validate_population_data,
    validate_resource_level_data,
)
from farm.research.analysis.database import (
    find_simulation_databases,
    get_action_distribution_data,
    get_columns_data,
    get_columns_data_by_agent_type,
    get_data,
    get_resource_consumption_data,
    get_resource_level_data,
    get_rewards_by_generation,
)
from farm.research.analysis.dataframes import create_population_df
from farm.research.analysis.main import main
from farm.research.analysis.plotting import (
    plot_action_distributions,
    plot_early_termination_analysis,
    plot_final_agent_counts,
    plot_marker_point,
    plot_mean_and_ci,
    plot_median_line,
    plot_population_trends_across_simulations,
    plot_population_trends_by_agent_type,
    plot_reference_line,
    plot_resource_consumption_trends,
    plot_resource_level_trends,
    plot_rewards_by_generation,
    setup_plot_aesthetics,
)
from farm.research.analysis.util import calculate_statistics

__all__ = [
    # Database functions
    "find_simulation_databases",
    "get_columns_data",
    "get_data",
    "get_columns_data_by_agent_type",
    "get_resource_consumption_data",
    "get_action_distribution_data",
    "get_resource_level_data",
    "get_rewards_by_generation",
    # Analysis functions
    "validate_population_data",
    "validate_resource_level_data",
    "detect_early_terminations",
    "analyze_final_agent_counts",
    "process_experiment",
    "find_experiments",
    "process_experiment_by_agent_type",
    "process_experiment_resource_consumption",
    "process_action_distributions",
    "process_experiment_resource_levels",
    "process_experiment_rewards_by_generation",
    # Dataframe functions
    "create_population_df",
    # Utility functions
    "calculate_statistics",
    # Plotting functions
    "plot_mean_and_ci",
    "plot_median_line",
    "plot_reference_line",
    "plot_marker_point",
    "setup_plot_aesthetics",
    "plot_population_trends_across_simulations",
    "plot_population_trends_by_agent_type",
    "plot_resource_consumption_trends",
    "plot_early_termination_analysis",
    "plot_final_agent_counts",
    "plot_rewards_by_generation",
    "plot_action_distributions",
    "plot_resource_level_trends",
    # Main functions
    "main",
]
