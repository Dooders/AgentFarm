"""
Dominance Analysis Functions and Utilities

This file contains legacy dominance analysis functions. For new code, use:
 - farm.analysis.dominance.module - Modern module implementation
 - farm.analysis.dominance.pipeline - Orchestration
 - farm.analysis.dominance.features - Feature engineering
 - farm.analysis.dominance.db_io - Persistence

The DominanceAnalysis class at the bottom uses the old BaseAnalysisModule
and is kept for backwards compatibility. New code should use the module system.
"""

import traceback

import pandas as pd

from farm.analysis.core import BaseAnalysisModule
from farm.analysis.common.metrics import (
    analyze_correlations,
    get_valid_numeric_columns,
    group_and_analyze,
    split_and_compare_groups,
)
from farm.analysis.database_utils import import_multi_table_data
from farm.analysis.dominance.compute import (
    aggregate_reproduction_analysis_results,
    compute_comprehensive_dominance,
    compute_dominance_switch_factors,
    compute_dominance_switches,
    compute_population_dominance,
    compute_survival_dominance,
)
from farm.analysis.dominance.data import (
    get_agent_survival_stats,
    get_final_population_counts,
    get_initial_positions_and_resources,
    get_reproduction_stats,
)
from farm.analysis.dominance.models import DominanceDataModel
from farm.analysis.dominance.sqlalchemy_models import (
    AgentPopulation,
    CorrelationAnalysis,
    DominanceMetrics,
    DominanceSwitching,
    HighLowSwitchingComparison,
    ReproductionStats,
    ResourceDistribution,
    Simulation,
    get_session,
    init_db,
)
from farm.utils.logging_config import get_logger
from scripts.analysis_config import setup_and_process_simulations

logger = get_logger(__name__)


def process_single_simulation(session, iteration, config, **kwargs):
    """
    Process a single simulation database for dominance analysis.

    Parameters
    ----------
    session : SQLAlchemy session
        Session connected to the simulation database
    iteration : int
        Iteration number of the simulation
    config : dict
        Configuration dictionary for the simulation
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    dict or None
        Dictionary containing processed data for this simulation,
        or None if processing failed
    """
    try:
        logger.info("processing_iteration", iteration=iteration)

        # Compute dominance metrics
        population_dominance = compute_population_dominance(session)
        survival_dominance = compute_survival_dominance(session)
        comprehensive_dominance = compute_comprehensive_dominance(session)

        # Compute dominance switching metrics
        dominance_switches = compute_dominance_switches(session)

        # Get initial positions and resource data
        initial_data = get_initial_positions_and_resources(session, config)

        # Get final population counts
        final_counts = get_final_population_counts(session)

        # Get agent survival statistics
        survival_stats = get_agent_survival_stats(session)
        logger.info(
            "survival_stats_for_iteration",
            iteration=iteration,
            survival_stats=survival_stats,
        )

        # Get reproduction statistics
        reproduction_stats = get_reproduction_stats(session)

        # Combine all data
        sim_data = {
            "iteration": iteration,
            "population_dominance": population_dominance,
            "survival_dominance": survival_dominance,
            "comprehensive_dominance": (comprehensive_dominance["dominant_type"] if comprehensive_dominance else None),
        }

        # Add dominance scores
        for agent_type in ["system", "independent", "control"]:
            if comprehensive_dominance:
                sim_data[f"{agent_type}_dominance_score"] = comprehensive_dominance["scores"][agent_type]
                sim_data[f"{agent_type}_auc"] = comprehensive_dominance["metrics"]["auc"][agent_type]
                sim_data[f"{agent_type}_recency_weighted_auc"] = comprehensive_dominance["metrics"][
                    "recency_weighted_auc"
                ][agent_type]
                sim_data[f"{agent_type}_dominance_duration"] = comprehensive_dominance["metrics"]["dominance_duration"][
                    agent_type
                ]
                sim_data[f"{agent_type}_growth_trend"] = comprehensive_dominance["metrics"]["growth_trends"][agent_type]
                sim_data[f"{agent_type}_final_ratio"] = comprehensive_dominance["metrics"]["final_ratios"][agent_type]
            else:
                # Set default values when comprehensive_dominance is None
                sim_data[f"{agent_type}_dominance_score"] = None
                sim_data[f"{agent_type}_auc"] = None
                sim_data[f"{agent_type}_recency_weighted_auc"] = None
                sim_data[f"{agent_type}_dominance_duration"] = None
                sim_data[f"{agent_type}_growth_trend"] = None
                sim_data[f"{agent_type}_final_ratio"] = None

        # Add dominance switching data
        if dominance_switches:
            sim_data["total_switches"] = dominance_switches["total_switches"]
            sim_data["switches_per_step"] = dominance_switches["switches_per_step"]

            # Add average dominance periods
            for agent_type in ["system", "independent", "control"]:
                sim_data[f"{agent_type}_avg_dominance_period"] = dominance_switches["avg_dominance_periods"][agent_type]

            # Add phase-specific switch counts
            for phase in ["early", "middle", "late"]:
                sim_data[f"{phase}_phase_switches"] = dominance_switches["phase_switches"][phase]

            # Add transition matrix data
            for from_type in ["system", "independent", "control"]:
                for to_type in ["system", "independent", "control"]:
                    sim_data[f"{from_type}_to_{to_type}"] = dominance_switches["transition_probabilities"][from_type][
                        to_type
                    ]

        # Add all other data
        if initial_data:
            sim_data.update(initial_data)
        if final_counts:
            sim_data.update(final_counts)
        if survival_stats:
            sim_data.update(survival_stats)
        if reproduction_stats:
            sim_data.update(reproduction_stats)

        # Check if survival stats were added
        survival_keys = [
            "system_count",
            "system_alive",
            "system_dead",
            "system_avg_survival",
            "system_dead_ratio",
        ]
        missing_keys = [key for key in survival_keys if key not in sim_data]
        if missing_keys:
            logger.warning(
                "missing_survival_stats_keys",
                iteration=iteration,
                missing_keys=missing_keys,
            )

        # Validate data with Pydantic model
        try:
            validated_data = DominanceDataModel(**sim_data).dict()
            logger.debug("successfully_validated_data", iteration=iteration)
            return validated_data
        except Exception as e:
            logger.warning("data_validation_failed", iteration=iteration, error=str(e))
            # Still return the data even if validation fails
            return sim_data

    except Exception as e:
        logger.error(
            "error_processing_iteration",
            iteration=iteration,
            error=str(e),
            exc_info=True,
        )
        logger.error("traceback_details", traceback=traceback.format_exc())
        return None


def process_dominance_data(experiment_path, save_to_db=False, db_path="sqlite:///dominance.db"):
    """
    Analyze all simulation databases in the experiment folder.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases
    save_to_db : bool, optional
        If True, save the data directly to the database instead of returning a DataFrame
    db_path : str, optional
        Path to the database to save the data to, defaults to 'sqlite:///dominance.db'

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with analysis results for each simulation if save_to_db is False,
        otherwise None
    """
    # Use the helper function to process all simulations
    data = setup_and_process_simulations(
        experiment_path=experiment_path,
        process_simulation_func=process_single_simulation,
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Compute dominance switch factors and add to the DataFrame
    df = analyze_dominance_switch_factors(df)

    # Analyze reproduction and dominance switching
    df = analyze_reproduction_dominance_switching(df)

    # Get valid numeric reproduction columns
    numeric_repro_cols = get_valid_numeric_columns(df, [col for col in df.columns if "reproduction" in col])

    # Analyze high vs low switching simulations
    df = analyze_high_vs_low_switching(df, numeric_repro_cols)

    # Analyze reproduction timing
    df = analyze_reproduction_timing(df, numeric_repro_cols)

    # Analyze reproduction efficiency
    df = analyze_reproduction_efficiency(df, numeric_repro_cols)

    # Analyze reproduction advantage
    df = analyze_reproduction_advantage(df, numeric_repro_cols)

    # Analyze by agent type
    df = analyze_by_agent_type(df, numeric_repro_cols)

    if save_to_db:
        save_dominance_data_to_db(df, db_path)
        return None
    else:
        return df


def _create_dominance_metrics(row, sim_id):
    """Create a DominanceMetrics object from a DataFrame row."""
    return DominanceMetrics(
        simulation_id=sim_id,
        population_dominance=row["population_dominance"],
        survival_dominance=row["survival_dominance"],
        comprehensive_dominance=row["comprehensive_dominance"],
        system_dominance_score=row["system_dominance_score"],
        independent_dominance_score=row["independent_dominance_score"],
        control_dominance_score=row["control_dominance_score"],
        system_auc=row["system_auc"],
        independent_auc=row["independent_auc"],
        control_auc=row["control_auc"],
        system_recency_weighted_auc=row["system_recency_weighted_auc"],
        independent_recency_weighted_auc=row["independent_recency_weighted_auc"],
        control_recency_weighted_auc=row["control_recency_weighted_auc"],
        system_dominance_duration=row["system_dominance_duration"],
        independent_dominance_duration=row["independent_dominance_duration"],
        control_dominance_duration=row["control_dominance_duration"],
        system_growth_trend=row["system_growth_trend"],
        independent_growth_trend=row["independent_growth_trend"],
        control_growth_trend=row["control_growth_trend"],
        system_final_ratio=row["system_final_ratio"],
        independent_final_ratio=row["independent_final_ratio"],
        control_final_ratio=row["control_final_ratio"],
    )


def _create_agent_population(row, sim_id):
    """Create an AgentPopulation object from a DataFrame row."""
    return AgentPopulation(
        simulation_id=sim_id,
        system_agents=row.get("system_agents"),
        independent_agents=row.get("independent_agents"),
        control_agents=row.get("control_agents"),
        total_agents=row.get("total_agents"),
        final_step=row.get("final_step"),
        system_count=row.get("system_count"),
        system_alive=row.get("system_alive"),
        system_dead=row.get("system_dead"),
        system_avg_survival=row.get("system_avg_survival"),
        system_dead_ratio=row.get("system_dead_ratio"),
        independent_count=row.get("independent_count"),
        independent_alive=row.get("independent_alive"),
        independent_dead=row.get("independent_dead"),
        independent_avg_survival=row.get("independent_avg_survival"),
        independent_dead_ratio=row.get("independent_dead_ratio"),
        control_count=row.get("control_count"),
        control_alive=row.get("control_alive"),
        control_dead=row.get("control_dead"),
        control_avg_survival=row.get("control_avg_survival"),
        control_dead_ratio=row.get("control_dead_ratio"),
        initial_system_count=row.get("initial_system_count"),
        initial_independent_count=row.get("initial_independent_count"),
        initial_control_count=row.get("initial_control_count"),
        initial_resource_count=row.get("initial_resource_count"),
        initial_resource_amount=row.get("initial_resource_amount"),
    )


def _create_reproduction_stats(row, sim_id):
    """Create a ReproductionStats object from a DataFrame row."""
    return ReproductionStats(
        simulation_id=sim_id,
        system_reproduction_attempts=row.get("system_reproduction_attempts"),
        system_reproduction_successes=row.get("system_reproduction_successes"),
        system_reproduction_failures=row.get("system_reproduction_failures"),
        system_reproduction_success_rate=row.get("system_reproduction_success_rate"),
        system_first_reproduction_time=row.get("system_first_reproduction_time"),
        system_reproduction_efficiency=row.get("system_reproduction_efficiency"),
        system_avg_resources_per_reproduction=row.get("system_avg_resources_per_reproduction"),
        system_avg_offspring_resources=row.get("system_avg_offspring_resources"),
        independent_reproduction_attempts=row.get("independent_reproduction_attempts"),
        independent_reproduction_successes=row.get("independent_reproduction_successes"),
        independent_reproduction_failures=row.get("independent_reproduction_failures"),
        independent_reproduction_success_rate=row.get("independent_reproduction_success_rate"),
        independent_first_reproduction_time=row.get("independent_first_reproduction_time"),
        independent_reproduction_efficiency=row.get("independent_reproduction_efficiency"),
        independent_avg_resources_per_reproduction=row.get("independent_avg_resources_per_reproduction"),
        independent_avg_offspring_resources=row.get("independent_avg_offspring_resources"),
        control_reproduction_attempts=row.get("control_reproduction_attempts"),
        control_reproduction_successes=row.get("control_reproduction_successes"),
        control_reproduction_failures=row.get("control_reproduction_failures"),
        control_reproduction_success_rate=row.get("control_reproduction_success_rate"),
        control_first_reproduction_time=row.get("control_first_reproduction_time"),
        control_reproduction_efficiency=row.get("control_reproduction_efficiency"),
        control_avg_resources_per_reproduction=row.get("control_avg_resources_per_reproduction"),
        control_avg_offspring_resources=row.get("control_avg_offspring_resources"),
        # Reproduction advantage metrics
        independent_vs_control_first_reproduction_advantage=row.get(
            "independent_vs_control_first_reproduction_advantage"
        ),
        independent_vs_control_reproduction_efficiency_advantage=row.get(
            "independent_vs_control_reproduction_efficiency_advantage"
        ),
        independent_vs_control_reproduction_rate_advantage=row.get(
            "independent_vs_control_reproduction_rate_advantage"
        ),
        system_vs_independent_reproduction_rate_advantage=row.get("system_vs_independent_reproduction_rate_advantage"),
        system_vs_control_reproduction_rate_advantage=row.get("system_vs_control_reproduction_rate_advantage"),
        system_vs_independent_reproduction_efficiency_advantage=row.get(
            "system_vs_independent_reproduction_efficiency_advantage"
        ),
        system_vs_control_first_reproduction_advantage=row.get("system_vs_control_first_reproduction_advantage"),
        system_vs_independent_first_reproduction_advantage=row.get(
            "system_vs_independent_first_reproduction_advantage"
        ),
        system_vs_control_reproduction_efficiency_advantage=row.get(
            "system_vs_control_reproduction_efficiency_advantage"
        ),
    )


def _create_dominance_switching(row, sim_id):
    """Create a DominanceSwitching object from a DataFrame row."""
    return DominanceSwitching(
        simulation_id=sim_id,
        system_to_independent_switches=row.get("system_to_independent_switches"),
        independent_to_system_switches=row.get("independent_to_system_switches"),
        system_to_control_switches=row.get("system_to_control_switches"),
        control_to_system_switches=row.get("control_to_system_switches"),
        independent_to_control_switches=row.get("independent_to_control_switches"),
        control_to_independent_switches=row.get("control_to_independent_switches"),
        total_switches=row.get("total_switches"),
        switch_rate=row.get("switch_rate"),
        avg_switch_duration=row.get("avg_switch_duration"),
        max_switch_duration=row.get("max_switch_duration"),
        min_switch_duration=row.get("min_switch_duration"),
        system_dominance_periods=row.get("system_dominance_periods"),
        independent_dominance_periods=row.get("independent_dominance_periods"),
        control_dominance_periods=row.get("control_dominance_periods"),
        longest_system_period=row.get("longest_system_period"),
        longest_independent_period=row.get("longest_independent_period"),
        longest_control_period=row.get("longest_control_period"),
        system_period_avg_length=row.get("system_period_avg_length"),
        independent_period_avg_length=row.get("independent_period_avg_length"),
        control_period_avg_length=row.get("control_period_avg_length"),
        dominance_stability_score=row.get("dominance_stability_score"),
    )


def _create_resource_distribution(row, sim_id):
    """Create a ResourceDistribution object from a DataFrame row."""
    return ResourceDistribution(
        simulation_id=sim_id,
        total_resources=row.get("total_resources"),
        system_resources=row.get("system_resources"),
        independent_resources=row.get("independent_resources"),
        control_resources=row.get("control_resources"),
        resource_efficiency=row.get("resource_efficiency"),
        resource_distribution_entropy=row.get("resource_distribution_entropy"),
        resource_sharing_events=row.get("resource_sharing_events"),
        resource_sharing_efficiency=row.get("resource_sharing_efficiency"),
        resource_competition_events=row.get("resource_competition_events"),
        resource_competition_intensity=row.get("resource_competition_intensity"),
        system_resource_advantage=row.get("system_resource_advantage"),
        independent_resource_advantage=row.get("independent_resource_advantage"),
        control_resource_advantage=row.get("control_resource_advantage"),
        resource_efficiency_trend=row.get("resource_efficiency_trend"),
    )


def _create_high_low_switching(row, sim_id):
    """Create a HighLowSwitchingComparison object from a DataFrame row."""
    return HighLowSwitchingComparison(
        simulation_id=sim_id,
        high_dominance_switches=row.get("high_dominance_switches"),
        low_dominance_switches=row.get("low_dominance_switches"),
        high_dominance_switch_rate=row.get("high_dominance_switch_rate"),
        low_dominance_switch_rate=row.get("low_dominance_switch_rate"),
        high_dominance_avg_duration=row.get("high_dominance_avg_duration"),
        low_dominance_avg_duration=row.get("low_dominance_avg_duration"),
        high_dominance_stability=row.get("high_dominance_stability"),
        low_dominance_stability=row.get("low_dominance_stability"),
        dominance_threshold_effect=row.get("dominance_threshold_effect"),
        switching_pattern_difference=row.get("switching_pattern_difference"),
    )


def _create_correlation_analysis(row, sim_id):
    """Create a CorrelationAnalysis object from a DataFrame row."""
    return CorrelationAnalysis(
        simulation_id=sim_id,
        dominance_population_correlation=row.get("dominance_population_correlation"),
        dominance_survival_correlation=row.get("dominance_survival_correlation"),
        dominance_reproduction_correlation=row.get("dominance_reproduction_correlation"),
        dominance_resource_correlation=row.get("dominance_resource_correlation"),
        dominance_combat_correlation=row.get("dominance_combat_correlation"),
        dominance_efficiency_correlation=row.get("dominance_efficiency_correlation"),
        dominance_stability_correlation=row.get("dominance_stability_correlation"),
        population_reproduction_correlation=row.get("population_reproduction_correlation"),
        population_resource_correlation=row.get("population_resource_correlation"),
        reproduction_resource_correlation=row.get("reproduction_resource_correlation"),
        reproduction_efficiency_correlation=row.get("reproduction_efficiency_correlation"),
        resource_efficiency_correlation=row.get("resource_efficiency_correlation"),
        dominant_type_correlation=row.get("dominant_type_correlation"),
        correlation_strength=row.get("correlation_strength"),
        correlation_consistency=row.get("correlation_consistency"),
    )


def save_dominance_data_to_db(df, db_path="sqlite:///dominance.db"):
    """
    Save the dominance analysis data to a SQLite database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with dominance analysis results
    db_path : str, optional
        Path to the database to save the data to, defaults to 'sqlite:///dominance.db'

    Returns
    -------
    bool
        True if the data was successfully saved, False otherwise
    """
    if df.empty:
        logger.warning("no_data_to_save_to_database")
        return False

    try:
        # Initialize database
        engine = init_db(db_path)
        session = get_session(engine)

        logger.info("dataframe_columns", columns=df.columns.tolist())
        logger.info("sample_data", sample=df.iloc[0].to_dict())
        print(df.head())

        # Define the data model configurations
        data_model_configs = [
            {"model_class": DominanceMetrics, "create_func": _create_dominance_metrics, "name": "dominance metrics"},
            {"model_class": AgentPopulation, "create_func": _create_agent_population, "name": "agent population"},
            {"model_class": ReproductionStats, "create_func": _create_reproduction_stats, "name": "reproduction stats"},
            {
                "model_class": DominanceSwitching,
                "create_func": _create_dominance_switching,
                "name": "dominance switching",
            },
            {
                "model_class": ResourceDistribution,
                "create_func": _create_resource_distribution,
                "name": "resource distribution",
            },
            {
                "model_class": HighLowSwitchingComparison,
                "create_func": _create_high_low_switching,
                "name": "high-low switching",
            },
            {
                "model_class": CorrelationAnalysis,
                "create_func": _create_correlation_analysis,
                "name": "correlation analysis",
            },
        ]

        # Import data using the shared utility
        count = import_multi_table_data(
            df=df,
            session=session,
            simulation_model_class=Simulation,
            data_model_configs=data_model_configs,
            log_prefix="dominance analysis",
        )
        logger.info("simulations_imported_successfully")
        return True

    except Exception as e:
        if "session" in locals():
            session.rollback()
        logger.error("error_importing_data_to_database", error=str(e), exc_info=True)
        return False
    finally:
        if "session" in locals():
            session.close()


def analyze_dominance_switch_factors(df):
    """
    Analyze what factors correlate with dominance switching patterns.

    This function calculates various factors that correlate with dominance switching patterns,
    including:
    1. Top positive correlations between factors and total switches
    2. Top negative correlations between factors and total switches
    3. Average number of switches per dominant type
    4. Correlations between reproduction metrics and total switches

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added dominance switch factor columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_dominance_switch_factors")
        return df

    if df.empty or "total_switches" not in df.columns:
        logger.warning("no_dominance_switch_data_available")
        return df

    # Calculate dominance switch factors
    results = compute_dominance_switch_factors(df)

    if results is None:
        return df

    # Add dominance switch factors to the DataFrame
    if results:
        # Add top positive correlations
        if "top_positive_correlations" in results:
            for factor, corr in results["top_positive_correlations"].items():
                df[f"positive_corr_{factor}"] = corr

        # Add top negative correlations
        if "top_negative_correlations" in results:
            for factor, corr in results["top_negative_correlations"].items():
                df[f"negative_corr_{factor}"] = corr

        # Add switches by dominant type
        if "switches_by_dominant_type" in results:
            for agent_type, avg_switches in results["switches_by_dominant_type"].items():
                df[f"{agent_type}_avg_switches"] = avg_switches

        # Add reproduction correlations
        if "reproduction_correlations" in results:
            for factor, corr in results["reproduction_correlations"].items():
                df[f"repro_corr_{factor}"] = corr

        logger.info("added_dominance_switch_factor_analysis")

    return df


def analyze_reproduction_dominance_switching(df):
    """
    Analyze the relationship between reproduction strategies and dominance switching patterns
    and add the results to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added reproduction analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_reproduction_dominance_switching")
        return df

    if df.empty or "total_switches" not in df.columns:
        logger.warning("no_dominance_switch_data_for_reproduction_switching")
        return df

    reproduction_cols = [col for col in df.columns if "reproduction" in col]

    # Filter to only include numeric reproduction columns
    numeric_repro_cols = get_valid_numeric_columns(df, reproduction_cols)

    if not numeric_repro_cols:
        logger.warning("no_numeric_reproduction_data_columns")
        return df

    # Use the aggregation function from compute.py to collect all results
    results = aggregate_reproduction_analysis_results(df, numeric_repro_cols)

    if not results:
        return df

    # Add results to the DataFrame
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            for key, value in category_results.items():
                if isinstance(value, dict):
                    # For nested dictionaries (like high vs low switching comparison)
                    for subkey, subvalue in value.items():
                        col_name = f"{category}_{key}_{subkey}"
                        df[col_name] = subvalue
                else:
                    # For simple key-value pairs
                    col_name = f"{category}_{key}"
                    df[col_name] = value

    logger.info("added_reproduction_analysis_categories", count=len(results))

    return df


def analyze_high_vs_low_switching(df, numeric_repro_cols):
    """
    Compare reproduction metrics between high and low switching groups.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added high vs low switching analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_high_vs_low_switching")
        return df

    if df.empty or "total_switches" not in df.columns:
        logger.warning("no_dominance_switch_data_for_high_vs_low_switching")
        return df

    # Use the utility function to split and compare groups
    comparison_results = split_and_compare_groups(
        df,
        split_column="total_switches",
        metrics=numeric_repro_cols,
        split_method="median",
    )

    # Extract comparison results
    if "comparison_results" in comparison_results:
        repro_comparison = comparison_results["comparison_results"]

        # Add these values to the DataFrame with the specific naming convention used in this module
        for col, stats in repro_comparison.items():
            df[f"{col}_high_switching_mean"] = stats["high_group_mean"]
            df[f"{col}_low_switching_mean"] = stats["low_group_mean"]
            df[f"{col}_difference"] = stats["difference"]
            df[f"{col}_percent_difference"] = stats["percent_difference"]

        # Log the most significant differences
        logger.info("reproduction_differences_high_vs_low_switching")
        sorted_diffs = sorted(
            repro_comparison.items(),
            key=lambda x: abs(x[1]["percent_difference"]),
            reverse=True,
        )

        for col, stats in sorted_diffs[:5]:  # Top 5 differences
            if abs(stats["percent_difference"]) > 10:  # Only report meaningful differences
                direction = "higher" if stats["difference"] > 0 else "lower"
                logger.info(
                    "reproduction_difference_detail",
                    column=col,
                    high_mean=stats["high_group_mean"],
                    low_mean=stats["low_group_mean"],
                    percent_difference=abs(stats["percent_difference"]),
                    direction=direction,
                )

    return df


def analyze_reproduction_timing(df, numeric_repro_cols):
    """
    Analyze how first reproduction timing relates to dominance switching.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added reproduction timing analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_reproduction_timing")
        return df

    # Filter to get only first reproduction columns
    first_repro_cols = [col for col in numeric_repro_cols if "first_reproduction_time" in col]

    if first_repro_cols:
        # Define a filter condition to exclude rows where first reproduction is -1 (no reproduction)
        def filter_valid_first_repro(data_df):
            filtered_dfs = {}
            for col in first_repro_cols:
                if col in data_df.columns:
                    filtered_dfs[col] = data_df[data_df[col] > 0]

            # If no valid columns, return original data
            if not filtered_dfs:
                return data_df

            # Return the filtered data for the first column (they should be similar)
            return next(iter(filtered_dfs.values()))

        # Use the utility function to analyze correlations with filtering
        first_repro_corr = {}
        for col in first_repro_cols:
            # Create a single-column filter for this specific column
            # Use function to capture col by value, not by reference
            def col_filter(df, col=col):
                return df[df[col] > 0]

            correlations = analyze_correlations(
                df,
                target_column="total_switches",
                metric_columns=[col],
                min_data_points=5,
                filter_condition=col_filter,
            )

            # Add the correlation if found
            if correlations and col in correlations:
                first_repro_corr[col] = correlations[col]

        # Log the results
        logger.info("correlation_reproduction_timing_dominance_switches")
        for col, corr in first_repro_corr.items():
            agent_type = col.split("_first_reproduction")[0]
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logger.info(
                    "reproduction_timing_correlation",
                    agent_type=agent_type,
                    direction=direction,
                    correlation=corr,
                )
    else:
        logger.info("no_first_reproduction_timing_data_available")

    return df


def analyze_reproduction_efficiency(df, numeric_repro_cols):
    """
    Analyze if reproduction efficiency correlates with dominance stability.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added reproduction efficiency analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_reproduction_efficiency")
        return df

    efficiency_cols = [col for col in numeric_repro_cols if "reproduction_efficiency" in col]

    if efficiency_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        # Use the utility function to analyze correlations
        def filter_valid(data_df, efficiency_cols=efficiency_cols):
            return data_df[(data_df[efficiency_cols].notna()).all(axis=1) & (data_df[efficiency_cols] != 0).all(axis=1)]

        efficiency_stability_corr = analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=efficiency_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )

        # Log the results
        logger.info("correlation_reproduction_efficiency_dominance_stability")
        for col, corr in efficiency_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                agent_type = col.split("_reproduction")[0]
                direction = "more" if corr > 0 else "less"
                logger.info(
                    "reproduction_efficiency_correlation",
                    agent_type=agent_type,
                    direction=direction,
                    correlation=corr,
                )

    return df


def analyze_reproduction_advantage(df, numeric_repro_cols):
    """
    Analyze reproduction advantage and dominance switching.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added reproduction advantage analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_reproduction_advantage")
        return df

    advantage_cols = [
        col
        for col in numeric_repro_cols
        if "reproduction_rate_advantage" in col or "reproduction_efficiency_advantage" in col
    ]

    if advantage_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        if "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        # Use the utility function to analyze correlations
        def filter_valid(data_df, advantage_cols=advantage_cols):
            return data_df[data_df[advantage_cols].notna().all(axis=1)]

        advantage_stability_corr = analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=advantage_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )

        # Log the results
        logger.info("correlation_reproduction_advantage_dominance_stability")
        for col, corr in advantage_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                if "_vs_" in col:
                    types = (
                        col.split("_vs_")[0],
                        col.split("_vs_")[1].split("_reproduction")[0],
                    )
                    direction = "more" if corr > 0 else "less"
                    logger.info(
                        "reproduction_advantage_correlation",
                        advantage_type=types[0],
                        over_type=types[1],
                        direction=direction,
                        correlation=corr,
                    )

    return df


def analyze_by_agent_type(df, numeric_repro_cols):
    """
    Analyze relationship between reproduction metrics and dominance switching by agent type.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added agent type analysis columns
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("input_not_dataframe", function="analyze_by_agent_type")
        return df

    results = {}

    if "comprehensive_dominance" in df.columns:
        # Define the analysis function to apply to each group
        def analyze_group_correlations(group_df):
            group_results = {}

            # Calculate correlations between reproduction metrics and switching
            group_correlations = analyze_correlations(
                group_df,
                target_column="total_switches",
                metric_columns=numeric_repro_cols,
                min_data_points=5,
            )

            # Add to results
            return group_correlations

        # Use the utility function to group and analyze
        agent_types = ["system", "independent", "control"]
        type_results = group_and_analyze(
            df,
            group_column="comprehensive_dominance",
            group_values=agent_types,
            analysis_func=analyze_group_correlations,
            min_group_size=5,
        )

        # Log top correlations for each agent type
        for agent_type, type_correlations in type_results.items():
            if type_correlations:
                logger.info(
                    "top_reproduction_factors_affecting_switching",
                    agent_type=agent_type,
                )
                sorted_corrs = sorted(type_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                for col, corr in sorted_corrs[:3]:  # Top 3
                    if abs(corr) > 0.2:  # Only report stronger correlations
                        direction = "more" if corr > 0 else "fewer"
                        logger.info(
                            "reproduction_factor_correlation",
                            column=col,
                            correlation=corr,
                            direction=direction,
                        )

    return df


class DominanceAnalysis(BaseAnalysisModule):
    """
    Module to analyze dominance patterns across simulations.

    This class extends BaseAnalysisModule to provide specific dominance analysis
    functionality.
    """

    def __init__(self, df=None):
        """Initialize the dominance analysis module."""
        super().__init__(df)

    def analyze_high_vs_low_switching(self, numeric_repro_cols=None):
        """
        Analyze high vs low switching simulations.

        Parameters
        ----------
        numeric_repro_cols : list, optional
            List of numeric reproduction columns to analyze

        Returns
        -------
        dict
            Dictionary with analysis results
        """
        if self.df is None:
            return {}

        # Get numeric reproduction columns if not provided
        if numeric_repro_cols is None:
            numeric_repro_cols = self.get_valid_columns("reproduction")

        # Use split_and_compare from parent class
        results = self.split_and_compare(split_column="total_switches", metrics=numeric_repro_cols)

        return results

    def analyze_dominance_factors(self):
        """
        Analyze factors correlating with dominance.

        Returns
        -------
        dict
            Dictionary with dominance factor analysis results
        """
        if self.df is None or "total_switches" not in self.df.columns:
            return {}

        # Find top correlations with switching
        switch_correlations = self.find_top_correlations(target_column="total_switches", top_n=10)

        # Find top correlations with dominance stability
        if "dominance_stability" not in self.df.columns and "switches_per_step" in self.df.columns:
            self.df["dominance_stability"] = 1 / (self.df["switches_per_step"] + 0.01)

        if "dominance_stability" in self.df.columns:
            stability_correlations = self.find_top_correlations(target_column="dominance_stability", top_n=10)
        else:
            stability_correlations = {}

        return {
            "switch_correlations": switch_correlations,
            "stability_correlations": stability_correlations,
        }

    def analyze_by_agent_type(self, metric_columns=None):
        """
        Analyze metrics by dominant agent type.

        Parameters
        ----------
        metric_columns : list, optional
            List of metric columns to analyze

        Returns
        -------
        dict
            Dictionary with analysis results by agent type
        """
        if self.df is None or "comprehensive_dominance" not in self.df.columns:
            return {}

        # Get metric columns if not provided
        if metric_columns is None:
            metric_columns = self.get_valid_columns()

        # Define analysis function
        def analyze_agent_type(group_df):
            return {
                "means": group_df[metric_columns].mean().to_dict(),
                "correlations": analyze_correlations(group_df, "total_switches", metric_columns),
            }

        # Group and analyze by agent type
        agent_types = ["system", "independent", "control"]
        return self.group_and_analyze(
            group_column="comprehensive_dominance",
            group_values=agent_types,
            analysis_func=analyze_agent_type,
        )
