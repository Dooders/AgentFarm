import pandas as pd

from farm.analysis.common.metrics import get_valid_numeric_columns
from farm.analysis.dominance.constants import DOMINANCE_AGENT_TYPES
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
from farm.analysis.dominance.orchestrator import create_dominance_orchestrator
from farm.utils.logging import get_logger
from scripts.analysis_config import setup_and_process_simulations

logger = get_logger(__name__)


def process_single_simulation(session, iteration, config, **_):
    try:
        logger.info("processing_iteration", iteration=iteration)

        orchestrator = create_dominance_orchestrator()

        population_dominance = orchestrator.compute_population_dominance(session)
        survival_dominance = orchestrator.compute_survival_dominance(session)
        comprehensive_dominance = orchestrator.compute_comprehensive_dominance(session)
        dominance_switches = orchestrator.compute_dominance_switches(session)

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

        for agent_type in DOMINANCE_AGENT_TYPES:
            if comprehensive_dominance:
                scores = comprehensive_dominance.get("scores") or {}
                metrics = comprehensive_dominance.get("metrics") or {}
                auc = metrics.get("auc") or {}
                rw_auc = metrics.get("recency_weighted_auc") or {}
                dom_dur = metrics.get("dominance_duration") or {}
                growth = metrics.get("growth_trends") or {}
                final_r = metrics.get("final_ratios") or {}
                sim_data[f"{agent_type}_dominance_score"] = scores.get(agent_type)
                sim_data[f"{agent_type}_auc"] = auc.get(agent_type)
                sim_data[f"{agent_type}_recency_weighted_auc"] = rw_auc.get(agent_type)
                sim_data[f"{agent_type}_dominance_duration"] = dom_dur.get(agent_type)
                sim_data[f"{agent_type}_growth_trend"] = growth.get(agent_type)
                sim_data[f"{agent_type}_final_ratio"] = final_r.get(agent_type)
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
            avg_periods = dominance_switches.get("avg_dominance_periods") or {}
            for agent_type in DOMINANCE_AGENT_TYPES:
                sim_data[f"{agent_type}_avg_dominance_period"] = avg_periods.get(agent_type)
            phase_sw = dominance_switches.get("phase_switches") or {}
            for phase in ["early", "middle", "late"]:
                sim_data[f"{phase}_phase_switches"] = phase_sw.get(phase)
            trans_prob = dominance_switches.get("transition_probabilities") or {}
            for from_type in DOMINANCE_AGENT_TYPES:
                row = trans_prob.get(from_type) or {}
                for to_type in DOMINANCE_AGENT_TYPES:
                    sim_data[f"{from_type}_to_{to_type}"] = row.get(to_type)

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
