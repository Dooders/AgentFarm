import glob
import json
import logging
import os
import traceback

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from farm.analysis.base_module import (
    BaseAnalysisModule,
    analyze_correlations,
    get_valid_numeric_columns,
    group_and_analyze,
    split_and_compare_groups,
)
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
from scripts.analysis_config import (
    get_valid_numeric_columns,
    setup_and_process_simulations,
)


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
        logging.info(f"Processing iteration {iteration}")

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
        logging.info(f"Survival stats for iteration {iteration}: {survival_stats}")

        # Get reproduction statistics
        reproduction_stats = get_reproduction_stats(session)

        # Combine all data
        sim_data = {
            "iteration": iteration,
            "population_dominance": population_dominance,
            "survival_dominance": survival_dominance,
            "comprehensive_dominance": comprehensive_dominance["dominant_type"],
        }

        # Add dominance scores
        for agent_type in ["system", "independent", "control"]:
            sim_data[f"{agent_type}_dominance_score"] = comprehensive_dominance[
                "scores"
            ][agent_type]
            sim_data[f"{agent_type}_auc"] = comprehensive_dominance["metrics"]["auc"][
                agent_type
            ]
            sim_data[f"{agent_type}_recency_weighted_auc"] = comprehensive_dominance[
                "metrics"
            ]["recency_weighted_auc"][agent_type]
            sim_data[f"{agent_type}_dominance_duration"] = comprehensive_dominance[
                "metrics"
            ]["dominance_duration"][agent_type]
            sim_data[f"{agent_type}_growth_trend"] = comprehensive_dominance["metrics"][
                "growth_trends"
            ][agent_type]
            sim_data[f"{agent_type}_final_ratio"] = comprehensive_dominance["metrics"][
                "final_ratios"
            ][agent_type]

        # Add dominance switching data
        if dominance_switches:
            sim_data["total_switches"] = dominance_switches["total_switches"]
            sim_data["switches_per_step"] = dominance_switches["switches_per_step"]

            # Add average dominance periods
            for agent_type in ["system", "independent", "control"]:
                sim_data[f"{agent_type}_avg_dominance_period"] = dominance_switches[
                    "avg_dominance_periods"
                ][agent_type]

            # Add phase-specific switch counts
            for phase in ["early", "middle", "late"]:
                sim_data[f"{phase}_phase_switches"] = dominance_switches[
                    "phase_switches"
                ][phase]

            # Add transition matrix data
            for from_type in ["system", "independent", "control"]:
                for to_type in ["system", "independent", "control"]:
                    sim_data[f"{from_type}_to_{to_type}"] = dominance_switches[
                        "transition_probabilities"
                    ][from_type][to_type]

        # Add all other data
        sim_data.update(initial_data)
        sim_data.update(final_counts)
        sim_data.update(survival_stats)
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
            logging.warning(
                f"Missing survival stats keys for iteration {iteration}: {missing_keys}"
            )

        # Validate data with Pydantic model
        try:
            validated_data = DominanceDataModel(**sim_data).dict()
            logging.debug(f"Successfully validated data for iteration {iteration}")
            return validated_data
        except Exception as e:
            logging.warning(f"Data validation failed for iteration {iteration}: {e}")
            # Still return the data even if validation fails
            return sim_data

    except Exception as e:
        logging.error(f"Error processing iteration {iteration}: {e}")
        logging.error(traceback.format_exc())
        return None


def process_dominance_data(
    experiment_path, save_to_db=False, db_path="sqlite:///dominance.db"
):
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
    numeric_repro_cols = get_valid_numeric_columns(
        df, [col for col in df.columns if "reproduction" in col]
    )

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
        logging.warning("No data to save to database")
        return False

    try:
        # Initialize database
        engine = init_db(db_path)
        session = get_session(engine)

        logging.info(f"DataFrame columns: {df.columns.tolist()}")
        logging.info(f"Sample data: {df.iloc[0].to_dict()}")
        print(df.head())

        # Import data row by row
        logging.info(f"Importing {len(df)} simulations into database...")
        for idx, row in df.iterrows():
            # Create Simulation record
            sim = Simulation(iteration=row["iteration"])
            session.add(sim)
            session.flush()  # Flush to get the ID

            # Log survival stats for this row
            survival_keys = [
                "system_count",
                "system_alive",
                "system_dead",
                "system_avg_survival",
                "system_dead_ratio",
            ]
            survival_values = {key: row.get(key) for key in survival_keys}
            logging.info(
                f"Survival stats for iteration {row['iteration']}: {survival_values}"
            )

            # Create DominanceMetrics record
            dominance_metrics = DominanceMetrics(
                simulation_id=sim.id,
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
                independent_recency_weighted_auc=row[
                    "independent_recency_weighted_auc"
                ],
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
            session.add(dominance_metrics)

            # Create AgentPopulation record
            agent_population = AgentPopulation(
                simulation_id=sim.id,
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
            session.add(agent_population)

            # Create ReproductionStats record
            reproduction_stats = ReproductionStats(
                simulation_id=sim.id,
                system_reproduction_attempts=row.get("system_reproduction_attempts"),
                system_reproduction_successes=row.get("system_reproduction_successes"),
                system_reproduction_failures=row.get("system_reproduction_failures"),
                system_reproduction_success_rate=row.get(
                    "system_reproduction_success_rate"
                ),
                system_first_reproduction_time=row.get(
                    "system_first_reproduction_time"
                ),
                system_reproduction_efficiency=row.get(
                    "system_reproduction_efficiency"
                ),
                system_avg_resources_per_reproduction=row.get(
                    "system_avg_resources_per_reproduction"
                ),
                system_avg_offspring_resources=row.get(
                    "system_avg_offspring_resources"
                ),
                independent_reproduction_attempts=row.get(
                    "independent_reproduction_attempts"
                ),
                independent_reproduction_successes=row.get(
                    "independent_reproduction_successes"
                ),
                independent_reproduction_failures=row.get(
                    "independent_reproduction_failures"
                ),
                independent_reproduction_success_rate=row.get(
                    "independent_reproduction_success_rate"
                ),
                independent_first_reproduction_time=row.get(
                    "independent_first_reproduction_time"
                ),
                independent_reproduction_efficiency=row.get(
                    "independent_reproduction_efficiency"
                ),
                independent_avg_resources_per_reproduction=row.get(
                    "independent_avg_resources_per_reproduction"
                ),
                independent_avg_offspring_resources=row.get(
                    "independent_avg_offspring_resources"
                ),
                control_reproduction_attempts=row.get("control_reproduction_attempts"),
                control_reproduction_successes=row.get(
                    "control_reproduction_successes"
                ),
                control_reproduction_failures=row.get("control_reproduction_failures"),
                control_reproduction_success_rate=row.get(
                    "control_reproduction_success_rate"
                ),
                control_first_reproduction_time=row.get(
                    "control_first_reproduction_time"
                ),
                control_reproduction_efficiency=row.get(
                    "control_reproduction_efficiency"
                ),
                control_avg_resources_per_reproduction=row.get(
                    "control_avg_resources_per_reproduction"
                ),
                control_avg_offspring_resources=row.get(
                    "control_avg_offspring_resources"
                ),
                independent_vs_control_first_reproduction_advantage=row.get(
                    "independent_vs_control_first_reproduction_advantage"
                ),
                independent_vs_control_reproduction_efficiency_advantage=row.get(
                    "independent_vs_control_reproduction_efficiency_advantage"
                ),
                independent_vs_control_reproduction_rate_advantage=row.get(
                    "independent_vs_control_reproduction_rate_advantage"
                ),
                system_vs_independent_reproduction_rate_advantage=row.get(
                    "system_vs_independent_reproduction_rate_advantage"
                ),
                system_vs_control_reproduction_rate_advantage=row.get(
                    "system_vs_control_reproduction_rate_advantage"
                ),
                system_vs_independent_reproduction_efficiency_advantage=row.get(
                    "system_vs_independent_reproduction_efficiency_advantage"
                ),
                system_vs_control_first_reproduction_advantage=row.get(
                    "system_vs_control_first_reproduction_advantage"
                ),
                system_vs_independent_first_reproduction_advantage=row.get(
                    "system_vs_independent_first_reproduction_advantage"
                ),
                system_vs_control_reproduction_efficiency_advantage=row.get(
                    "system_vs_control_reproduction_efficiency_advantage"
                ),
            )
            session.add(reproduction_stats)

            # Create DominanceSwitching record
            dominance_switching = DominanceSwitching(
                simulation_id=sim.id,
                total_switches=row.get("total_switches"),
                switches_per_step=row.get("switches_per_step"),
                dominance_stability=row.get("dominance_stability"),
                system_avg_dominance_period=row.get("system_avg_dominance_period"),
                independent_avg_dominance_period=row.get(
                    "independent_avg_dominance_period"
                ),
                control_avg_dominance_period=row.get("control_avg_dominance_period"),
                early_phase_switches=row.get("early_phase_switches"),
                middle_phase_switches=row.get("middle_phase_switches"),
                late_phase_switches=row.get("late_phase_switches"),
                control_avg_switches=row.get("control_avg_switches"),
                independent_avg_switches=row.get("independent_avg_switches"),
                system_avg_switches=row.get("system_avg_switches"),
                system_to_system=row.get("system_to_system"),
                system_to_independent=row.get("system_to_independent"),
                system_to_control=row.get("system_to_control"),
                independent_to_system=row.get("independent_to_system"),
                independent_to_independent=row.get("independent_to_independent"),
                independent_to_control=row.get("independent_to_control"),
                control_to_system=row.get("control_to_system"),
                control_to_independent=row.get("control_to_independent"),
                control_to_control=row.get("control_to_control"),
            )
            session.add(dominance_switching)

            # Create ResourceDistribution record
            resource_distribution = ResourceDistribution(
                simulation_id=sim.id,
                systemagent_avg_resource_dist=row.get("systemagent_avg_resource_dist"),
                systemagent_weighted_resource_dist=row.get(
                    "systemagent_weighted_resource_dist"
                ),
                systemagent_nearest_resource_dist=row.get(
                    "systemagent_nearest_resource_dist"
                ),
                systemagent_resources_in_range=row.get(
                    "systemagent_resources_in_range"
                ),
                systemagent_resource_amount_in_range=row.get(
                    "systemagent_resource_amount_in_range"
                ),
                independentagent_avg_resource_dist=row.get(
                    "independentagent_avg_resource_dist"
                ),
                independentagent_weighted_resource_dist=row.get(
                    "independentagent_weighted_resource_dist"
                ),
                independentagent_nearest_resource_dist=row.get(
                    "independentagent_nearest_resource_dist"
                ),
                independentagent_resources_in_range=row.get(
                    "independentagent_resources_in_range"
                ),
                independentagent_resource_amount_in_range=row.get(
                    "independentagent_resource_amount_in_range"
                ),
                controlagent_avg_resource_dist=row.get(
                    "controlagent_avg_resource_dist"
                ),
                controlagent_weighted_resource_dist=row.get(
                    "controlagent_weighted_resource_dist"
                ),
                controlagent_nearest_resource_dist=row.get(
                    "controlagent_nearest_resource_dist"
                ),
                controlagent_resources_in_range=row.get(
                    "controlagent_resources_in_range"
                ),
                controlagent_resource_amount_in_range=row.get(
                    "controlagent_resource_amount_in_range"
                ),
                positive_corr_controlagent_resource_amount_in_range=row.get(
                    "positive_corr_controlagent_resource_amount_in_range"
                ),
                positive_corr_systemagent_avg_resource_dist=row.get(
                    "positive_corr_systemagent_avg_resource_dist"
                ),
                positive_corr_systemagent_weighted_resource_dist=row.get(
                    "positive_corr_systemagent_weighted_resource_dist"
                ),
                positive_corr_independentagent_avg_resource_dist=row.get(
                    "positive_corr_independentagent_avg_resource_dist"
                ),
                positive_corr_independentagent_weighted_resource_dist=row.get(
                    "positive_corr_independentagent_weighted_resource_dist"
                ),
                negative_corr_systemagent_resource_amount_in_range=row.get(
                    "negative_corr_systemagent_resource_amount_in_range"
                ),
                negative_corr_systemagent_nearest_resource_dist=row.get(
                    "negative_corr_systemagent_nearest_resource_dist"
                ),
                negative_corr_independentagent_resource_amount_in_range=row.get(
                    "negative_corr_independentagent_resource_amount_in_range"
                ),
                negative_corr_controlagent_avg_resource_dist=row.get(
                    "negative_corr_controlagent_avg_resource_dist"
                ),
                negative_corr_controlagent_nearest_resource_dist=row.get(
                    "negative_corr_controlagent_nearest_resource_dist"
                ),
            )
            session.add(resource_distribution)

            # Create HighLowSwitchingComparison record
            # Include all fields from this table, only a few shown for brevity
            # (the original import_csv_to_db.py has the same pattern)
            high_low_switching = HighLowSwitchingComparison(
                simulation_id=sim.id,
                system_reproduction_attempts_high_switching_mean=row.get(
                    "system_reproduction_attempts_high_switching_mean"
                ),
                system_reproduction_attempts_low_switching_mean=row.get(
                    "system_reproduction_attempts_low_switching_mean"
                ),
                system_reproduction_attempts_difference=row.get(
                    "system_reproduction_attempts_difference"
                ),
                system_reproduction_attempts_percent_difference=row.get(
                    "system_reproduction_attempts_percent_difference"
                ),
                system_reproduction_successes_high_switching_mean=row.get(
                    "system_reproduction_successes_high_switching_mean"
                ),
                system_reproduction_successes_low_switching_mean=row.get(
                    "system_reproduction_successes_low_switching_mean"
                ),
                system_reproduction_successes_difference=row.get(
                    "system_reproduction_successes_difference"
                ),
                system_reproduction_successes_percent_difference=row.get(
                    "system_reproduction_successes_percent_difference"
                ),
                # Continue with all fields from the HighLowSwitchingComparison model
                # These should match the fields in the sqlalchemy_models.py file
                system_reproduction_failures_high_switching_mean=row.get(
                    "system_reproduction_failures_high_switching_mean"
                ),
                system_reproduction_failures_low_switching_mean=row.get(
                    "system_reproduction_failures_low_switching_mean"
                ),
                system_reproduction_failures_difference=row.get(
                    "system_reproduction_failures_difference"
                ),
                system_reproduction_failures_percent_difference=row.get(
                    "system_reproduction_failures_percent_difference"
                ),
                # Add all remaining fields...
                independent_reproduction_attempts_high_switching_mean=row.get(
                    "independent_reproduction_attempts_high_switching_mean"
                ),
                independent_reproduction_attempts_low_switching_mean=row.get(
                    "independent_reproduction_attempts_low_switching_mean"
                ),
                independent_reproduction_attempts_difference=row.get(
                    "independent_reproduction_attempts_difference"
                ),
                independent_reproduction_attempts_percent_difference=row.get(
                    "independent_reproduction_attempts_percent_difference"
                ),
                # Add remaining fields from the model...
            )
            session.add(high_low_switching)

            # Create CorrelationAnalysis record
            correlation_analysis = CorrelationAnalysis(
                simulation_id=sim.id,
                # Add all correlation analysis fields from the model
                repro_corr_system_reproduction_success_rate=row.get(
                    "repro_corr_system_reproduction_success_rate"
                ),
                repro_corr_independent_avg_resources_per_reproduction=row.get(
                    "repro_corr_independent_avg_resources_per_reproduction"
                ),
                repro_corr_independent_reproduction_success_rate=row.get(
                    "repro_corr_independent_reproduction_success_rate"
                ),
                repro_corr_independent_reproduction_failures=row.get(
                    "repro_corr_independent_reproduction_failures"
                ),
                repro_corr_independent_reproduction_attempts=row.get(
                    "repro_corr_independent_reproduction_attempts"
                ),
                # Add remaining correlation fields...
            )
            session.add(correlation_analysis)

        # Commit all changes
        session.commit()
        logging.info(f"Successfully imported {len(df)} simulations into the database")
        return True

    except Exception as e:
        if "session" in locals():
            session.rollback()
        logging.error(f"Error importing data to database: {e}")
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
        logging.warning("Input to analyze_dominance_switch_factors is not a DataFrame")
        return df

    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for analysis")
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
            for agent_type, avg_switches in results[
                "switches_by_dominant_type"
            ].items():
                df[f"{agent_type}_avg_switches"] = avg_switches

        # Add reproduction correlations
        if "reproduction_correlations" in results:
            for factor, corr in results["reproduction_correlations"].items():
                df[f"repro_corr_{factor}"] = corr

        logging.info("Added dominance switch factor analysis to the DataFrame")

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
        logging.warning(
            "Input to analyze_reproduction_dominance_switching is not a DataFrame"
        )
        return df

    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for reproduction-switching analysis"
        )
        return df

    reproduction_cols = [col for col in df.columns if "reproduction" in col]

    # Filter to only include numeric reproduction columns
    numeric_repro_cols = get_valid_numeric_columns(df, reproduction_cols)

    if not numeric_repro_cols:
        logging.warning("No numeric reproduction data columns found")
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

    logging.info(
        f"Added {len(results)} reproduction analysis categories to the DataFrame"
    )

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
        logging.warning("Input to analyze_high_vs_low_switching is not a DataFrame")
        return df

    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for high vs low switching analysis"
        )
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
        logging.info(
            "\nReproduction differences between high and low switching simulations:"
        )
        sorted_diffs = sorted(
            repro_comparison.items(),
            key=lambda x: abs(x[1]["percent_difference"]),
            reverse=True,
        )

        for col, stats in sorted_diffs[:5]:  # Top 5 differences
            if (
                abs(stats["percent_difference"]) > 10
            ):  # Only report meaningful differences
                direction = "higher" if stats["difference"] > 0 else "lower"
                logging.info(
                    f"  {col}: {stats['high_group_mean']:.3f} vs {stats['low_group_mean']:.3f} "
                    f"({abs(stats['percent_difference']):.1f}% {direction} in high-switching simulations)"
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
        logging.warning("Input to analyze_reproduction_timing is not a DataFrame")
        return df

    # Filter to get only first reproduction columns
    first_repro_cols = [
        col for col in numeric_repro_cols if "first_reproduction_time" in col
    ]

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
            col_filter = lambda df: df[df[col] > 0]

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
        logging.info(
            "\nCorrelation between first reproduction timing and dominance switches:"
        )
        for col, corr in first_repro_corr.items():
            agent_type = col.split("_first_reproduction")[0]
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(
                    f"  Earlier {agent_type} reproduction → {direction} dominance switches (r={corr:.3f})"
                )
    else:
        logging.info("No first reproduction timing data available for analysis")

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
        logging.warning("Input to analyze_reproduction_efficiency is not a DataFrame")
        return df

    efficiency_cols = [
        col for col in numeric_repro_cols if "reproduction_efficiency" in col
    ]

    if efficiency_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        # Use the utility function to analyze correlations
        filter_valid = lambda data_df: data_df[
            (data_df[efficiency_cols].notna()).all(axis=1)
            & (data_df[efficiency_cols] != 0).all(axis=1)
        ]

        efficiency_stability_corr = analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=efficiency_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )

        # Log the results
        logging.info(
            "\nCorrelation between reproduction efficiency and dominance stability:"
        )
        for col, corr in efficiency_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                agent_type = col.split("_reproduction")[0]
                direction = "more" if corr > 0 else "less"
                logging.info(
                    f"  Higher {agent_type} reproduction efficiency → {direction} stable dominance (r={corr:.3f})"
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
        logging.warning("Input to analyze_reproduction_advantage is not a DataFrame")
        return df

    advantage_cols = [
        col
        for col in numeric_repro_cols
        if "reproduction_rate_advantage" in col
        or "reproduction_efficiency_advantage" in col
    ]

    if advantage_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        if "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        # Use the utility function to analyze correlations
        filter_valid = lambda data_df: data_df[
            data_df[advantage_cols].notna().all(axis=1)
        ]

        advantage_stability_corr = analyze_correlations(
            df,
            target_column="dominance_stability",
            metric_columns=advantage_cols,
            min_data_points=5,
            filter_condition=filter_valid,
        )

        # Log the results
        logging.info(
            "\nCorrelation between reproduction advantage and dominance stability:"
        )
        for col, corr in advantage_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                if "_vs_" in col:
                    types = (
                        col.split("_vs_")[0],
                        col.split("_vs_")[1].split("_reproduction")[0],
                    )
                    direction = "more" if corr > 0 else "less"
                    logging.info(
                        f"  {types[0]} advantage over {types[1]} → {direction} stable dominance (r={corr:.3f})"
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
        logging.warning("Input to analyze_by_agent_type is not a DataFrame")
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
                logging.info(
                    f"\nTop reproduction factors affecting switching in {agent_type}-dominant simulations:"
                )
                sorted_corrs = sorted(
                    type_correlations.items(), key=lambda x: abs(x[1]), reverse=True
                )
                for col, corr in sorted_corrs[:3]:  # Top 3
                    if abs(corr) > 0.2:  # Only report stronger correlations
                        direction = "more" if corr > 0 else "fewer"
                        logging.info(f"  {col}: {corr:.3f} ({direction} switches)")

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
        results = self.split_and_compare(
            split_column="total_switches", metrics=numeric_repro_cols
        )

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
        switch_correlations = self.find_top_correlations(
            target_column="total_switches", top_n=10
        )

        # Find top correlations with dominance stability
        if (
            "dominance_stability" not in self.df.columns
            and "switches_per_step" in self.df.columns
        ):
            self.df["dominance_stability"] = 1 / (self.df["switches_per_step"] + 0.01)

        if "dominance_stability" in self.df.columns:
            stability_correlations = self.find_top_correlations(
                target_column="dominance_stability", top_n=10
            )
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
                "correlations": analyze_correlations(
                    group_df, "total_switches", metric_columns
                ),
            }

        # Group and analyze by agent type
        agent_types = ["system", "independent", "control"]
        return self.group_and_analyze(
            group_column="comprehensive_dominance",
            group_values=agent_types,
            analysis_func=analyze_agent_type,
        )
