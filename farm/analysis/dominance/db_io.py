from farm.analysis.database_utils import import_multi_table_data
from farm.analysis.dominance.sqlalchemy_models import (
    AgentPopulation,
    DominanceMetrics,
    ReproductionStats,
    Simulation,
    get_session,
    init_db,
)
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


def _create_dominance_metrics(row, sim_id):
    """Create a DominanceMetrics object from a DataFrame row."""
    return DominanceMetrics(
        simulation_id=sim_id,
        population_dominance=row.get("population_dominance"),
        survival_dominance=row.get("survival_dominance"),
        comprehensive_dominance=row.get("comprehensive_dominance"),
        system_dominance_score=row.get("system_dominance_score"),
        independent_dominance_score=row.get("independent_dominance_score"),
        control_dominance_score=row.get("control_dominance_score"),
        system_auc=row.get("system_auc"),
        independent_auc=row.get("independent_auc"),
        control_auc=row.get("control_auc"),
        system_recency_weighted_auc=row.get("system_recency_weighted_auc"),
        independent_recency_weighted_auc=row.get("independent_recency_weighted_auc"),
        control_recency_weighted_auc=row.get("control_recency_weighted_auc"),
        system_dominance_duration=row.get("system_dominance_duration"),
        independent_dominance_duration=row.get("independent_dominance_duration"),
        control_dominance_duration=row.get("control_dominance_duration"),
        system_growth_trend=row.get("system_growth_trend"),
        independent_growth_trend=row.get("independent_growth_trend"),
        control_growth_trend=row.get("control_growth_trend"),
        system_final_ratio=row.get("system_final_ratio"),
        independent_final_ratio=row.get("independent_final_ratio"),
        control_final_ratio=row.get("control_final_ratio"),
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
    )


def save_dominance_data_to_db(df, db_path="sqlite:///dominance.db"):
    if df.empty:
        logger.warning("no_data_to_save")
        return False
    try:
        engine = init_db(db_path)
        session = get_session(engine)
        # Define the data model configurations
        data_model_configs = [
            {"model_class": DominanceMetrics, "create_func": _create_dominance_metrics, "name": "dominance metrics"},
            {"model_class": AgentPopulation, "create_func": _create_agent_population, "name": "agent population"},
            {"model_class": ReproductionStats, "create_func": _create_reproduction_stats, "name": "reproduction stats"},
        ]

        # Import data using the shared utility
        count = import_multi_table_data(
            df=df,
            session=session,
            simulation_model_class=Simulation,
            data_model_configs=data_model_configs,
            log_prefix="dominance",
        )

        logger.info("simulations_imported_successfully")
        return True
    except Exception as exc:
        if "session" in locals():
            session.rollback()
        logger.error("database_import_failed", error=str(exc))
        return False
    finally:
        if "session" in locals():
            session.close()
