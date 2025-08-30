import logging

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


def save_dominance_data_to_db(df, db_path="sqlite:///dominance.db"):
    if df.empty:
        logging.warning("No data to save to database")
        return False
    try:
        engine = init_db(db_path)
        session = get_session(engine)
        logging.info(f"Importing {len(df)} simulations into database...")
        for _, row in df.iterrows():
            sim = Simulation(iteration=row["iteration"])
            session.add(sim)
            session.flush()

            dominance_metrics = DominanceMetrics(
                simulation_id=sim.id,
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
            session.add(dominance_metrics)

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

            reproduction_stats = ReproductionStats(
                simulation_id=sim.id,
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
            session.add(reproduction_stats)

            # Note: Additional tables like DominanceSwitching, ResourceDistribution, HighLowSwitchingComparison
            # can be populated here similarly if needed.

        session.commit()
        logging.info("Successfully imported simulations into the database")
        return True
    except Exception as exc:
        if "session" in locals():
            session.rollback()
        logging.error(f"Error importing data to database: {exc}")
        return False
    finally:
        if "session" in locals():
            session.close()

