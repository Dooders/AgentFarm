import pandas as pd

from farm.analysis.common.metrics import get_valid_numeric_columns
from farm.analysis.dominance.compute import (
    compute_comprehensive_dominance,
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
from farm.analysis.dominance.features import (
    analyze_by_agent_type,
    analyze_dominance_switch_factors,
    analyze_high_vs_low_switching,
    analyze_reproduction_advantage,
    analyze_reproduction_dominance_switching,
    analyze_reproduction_efficiency,
    analyze_reproduction_timing,
)
from farm.utils.logging_config import get_logger
from scripts.analysis_config import setup_and_process_simulations

logger = get_logger(__name__)


def process_single_simulation(session, iteration, config, **_):
    try:
        logger.info("processing_iteration", iteration=iteration)

        population_dominance = compute_population_dominance(session)
        survival_dominance = compute_survival_dominance(session)
        comprehensive_dominance = compute_comprehensive_dominance(session)
        dominance_switches = compute_dominance_switches(session)

        initial_data = get_initial_positions_and_resources(session, config)
        final_counts = get_final_population_counts(session)
        survival_stats = get_agent_survival_stats(session)
        reproduction_stats = get_reproduction_stats(session)

        sim_data = {
            "iteration": iteration,
            "population_dominance": population_dominance,
            "survival_dominance": survival_dominance,
            "comprehensive_dominance": (
                comprehensive_dominance["dominant_type"]
                if comprehensive_dominance
                else None
            ),
        }

        for agent_type in ["system", "independent", "control"]:
            if comprehensive_dominance:
                sim_data[f"{agent_type}_dominance_score"] = comprehensive_dominance[
                    "scores"
                ][agent_type]
                sim_data[f"{agent_type}_auc"] = comprehensive_dominance["metrics"][
                    "auc"
                ][agent_type]
                sim_data[f"{agent_type}_recency_weighted_auc"] = (
                    comprehensive_dominance["metrics"]["recency_weighted_auc"][
                        agent_type
                    ]
                )
                sim_data[f"{agent_type}_dominance_duration"] = comprehensive_dominance[
                    "metrics"
                ]["dominance_duration"][agent_type]
                sim_data[f"{agent_type}_growth_trend"] = comprehensive_dominance[
                    "metrics"
                ]["growth_trends"][agent_type]
                sim_data[f"{agent_type}_final_ratio"] = comprehensive_dominance[
                    "metrics"
                ]["final_ratios"][agent_type]
            else:
                sim_data[f"{agent_type}_dominance_score"] = None
                sim_data[f"{agent_type}_auc"] = None
                sim_data[f"{agent_type}_recency_weighted_auc"] = None
                sim_data[f"{agent_type}_dominance_duration"] = None
                sim_data[f"{agent_type}_growth_trend"] = None
                sim_data[f"{agent_type}_final_ratio"] = None

        if dominance_switches:
            sim_data["total_switches"] = dominance_switches["total_switches"]
            sim_data["switches_per_step"] = dominance_switches["switches_per_step"]
            for agent_type in ["system", "independent", "control"]:
                sim_data[f"{agent_type}_avg_dominance_period"] = dominance_switches[
                    "avg_dominance_periods"
                ][agent_type]
            for phase in ["early", "middle", "late"]:
                sim_data[f"{phase}_phase_switches"] = dominance_switches[
                    "phase_switches"
                ][phase]
            for from_type in ["system", "independent", "control"]:
                for to_type in ["system", "independent", "control"]:
                    sim_data[f"{from_type}_to_{to_type}"] = dominance_switches[
                        "transition_probabilities"
                    ][from_type][to_type]

        if initial_data:
            sim_data.update(initial_data)
        if final_counts:
            sim_data.update(final_counts)
        if survival_stats:
            sim_data.update(survival_stats)
        if reproduction_stats:
            sim_data.update(reproduction_stats)

        return sim_data
    except Exception as exc:
        logger.error("iteration_processing_failed", iteration=iteration, error=str(exc))
        return None


def process_dominance_data(
    experiment_path, save_to_db=False, db_path="sqlite:///dominance.db"
):
    data = setup_and_process_simulations(
        experiment_path=experiment_path,
        process_simulation_func=process_single_simulation,
    )

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = analyze_dominance_switch_factors(df)
    df = analyze_reproduction_dominance_switching(df)

    numeric_repro_cols = get_valid_numeric_columns(
        df, [col for col in df.columns if "reproduction" in col]
    )

    df = analyze_high_vs_low_switching(df, numeric_repro_cols)
    df = analyze_reproduction_timing(df, numeric_repro_cols)
    df = analyze_reproduction_efficiency(df, numeric_repro_cols)
    df = analyze_reproduction_advantage(df, numeric_repro_cols)
    df = analyze_by_agent_type(df, numeric_repro_cols)

    if save_to_db:
        from farm.analysis.dominance.db_io import save_dominance_data_to_db

        save_dominance_data_to_db(df, db_path)
        return None
    return df
