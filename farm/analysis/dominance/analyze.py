"""
Dominance Analysis Functions and Utilities

This file contains legacy dominance analysis functions. For new code, use:
 - farm.analysis.dominance.module - Modern module implementation
 - farm.analysis.dominance.pipeline - Orchestration
 - farm.analysis.dominance.features - Feature engineering
 - farm.analysis.dominance.db_io - Persistence

The DominanceAnalysis class at the bottom uses the old BaseAnalysisModule
and is kept for backwards compatibility. New code should use the module system.

Phase 2 Refactoring:
 - Classes now implement protocol interfaces with dependency injection
 - Backward compatibility maintained through module-level functions
 - Import from implementations.py for concrete classes
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
from farm.analysis.dominance.compute import DominanceComputer
from farm.analysis.dominance.data import (
    get_agent_survival_stats,
    get_final_population_counts,
    get_initial_positions_and_resources,
    get_reproduction_stats,
)
from farm.analysis.dominance.implementations import (
    DominanceAnalyzer,
    DominanceDataProvider,
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

        # Use orchestrator for analysis
        from farm.analysis.dominance.orchestrator import create_dominance_orchestrator
        orchestrator = create_dominance_orchestrator()

        # Compute dominance metrics using orchestrator
        population_dominance = orchestrator.compute_population_dominance(session)
        survival_dominance = orchestrator.compute_survival_dominance(session)
        comprehensive_dominance = orchestrator.compute_comprehensive_dominance(session)

        # Compute dominance switching metrics
        dominance_switches = orchestrator.compute_dominance_switches(session)

        # Get data using orchestrator
        initial_data = orchestrator.get_initial_positions_and_resources(session, config)
        final_counts = orchestrator.get_final_population_counts(session)
        survival_stats = orchestrator.get_agent_survival_stats(session)
        logger.info(
            "survival_stats_for_iteration",
            iteration=iteration,
            survival_stats=survival_stats,
        )

        # Get reproduction statistics
        reproduction_stats = orchestrator.get_reproduction_stats(session)

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

    # Use orchestrator for comprehensive DataFrame analysis
    from farm.analysis.dominance.orchestrator import create_dominance_orchestrator
    orchestrator = create_dominance_orchestrator()
    
    # Run comprehensive analysis (auto-detects reproduction columns)
    df = orchestrator.analyze_dataframe_comprehensively(df)

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
