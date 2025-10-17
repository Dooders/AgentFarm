import argparse
import logging
from farm.utils.logging import get_logger

logger = get_logger(__name__)

import pandas as pd
from sqlalchemy import create_engine, desc, func, text
from sqlalchemy.orm import sessionmaker

from farm.analysis.dominance.sqlalchemy_models import (
    AgentPopulation,
    CorrelationAnalysis,
    DominanceMetrics,
    DominanceSwitching,
    HighLowSwitchingComparison,
    ReproductionStats,
    ResourceDistribution,
    Simulation,
    init_db,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_session(db_path="sqlite:///dominance.db"):
    """Create a session for database operations"""
    engine = create_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def query_dominance_metrics(session):
    """Query and display basic dominance metrics"""
    logger.info("Querying basic dominance metrics...")

    # Get counts of each dominance type
    dominance_counts = (
        session.query(
            DominanceMetrics.comprehensive_dominance,
            func.count(DominanceMetrics.comprehensive_dominance).label("count"),
        )
        .group_by(DominanceMetrics.comprehensive_dominance)
        .all()
    )

    print("\nComprehensive Dominance Counts:")
    for dominance_type, count in dominance_counts:
        print(f"{dominance_type}: {count}")

    # Get average dominance scores
    avg_scores = session.query(
        func.avg(DominanceMetrics.system_dominance_score).label("avg_system"),
        func.avg(DominanceMetrics.independent_dominance_score).label("avg_independent"),
        func.avg(DominanceMetrics.control_dominance_score).label("avg_control"),
    ).one()

    print("\nAverage Dominance Scores:")
    print(f"System: {avg_scores.avg_system:.4f}")
    print(f"Independent: {avg_scores.avg_independent:.4f}")
    print(f"Control: {avg_scores.avg_control:.4f}")

    # Get top 5 simulations with highest system dominance
    top_system = (
        session.query(
            Simulation.iteration,
            DominanceMetrics.system_dominance_score,
            DominanceMetrics.independent_dominance_score,
            DominanceMetrics.control_dominance_score,
        )
        .join(DominanceMetrics)
        .order_by(desc(DominanceMetrics.system_dominance_score))
        .limit(5)
        .all()
    )

    print("\nTop 5 Simulations with Highest System Dominance:")
    for iteration, sys_score, ind_score, ctrl_score in top_system:
        print(
            f"Iteration {iteration}: System={sys_score:.4f}, Independent={ind_score:.4f}, Control={ctrl_score:.4f}"
        )


def query_agent_populations(session):
    """Query and display agent population statistics"""
    logger.info("Querying agent population statistics...")

    # Get average population counts
    avg_counts = session.query(
        func.avg(AgentPopulation.system_count).label("avg_system"),
        func.avg(AgentPopulation.independent_count).label("avg_independent"),
        func.avg(AgentPopulation.control_count).label("avg_control"),
        func.avg(AgentPopulation.total_agents).label("avg_total"),
    ).one()

    print("\nAverage Agent Counts:")
    print(f"System: {avg_counts.avg_system:.2f}")
    print(f"Independent: {avg_counts.avg_independent:.2f}")
    print(f"Control: {avg_counts.avg_control:.2f}")
    print(f"Total: {avg_counts.avg_total:.2f}")

    # Get average survival statistics
    avg_survival = session.query(
        func.avg(AgentPopulation.system_avg_survival).label("avg_system_survival"),
        func.avg(AgentPopulation.independent_avg_survival).label(
            "avg_independent_survival"
        ),
        func.avg(AgentPopulation.control_avg_survival).label("avg_control_survival"),
    ).one()

    print("\nAverage Survival Times:")
    print(f"System: {avg_survival.avg_system_survival:.2f}")
    print(f"Independent: {avg_survival.avg_independent_survival:.2f}")
    print(f"Control: {avg_survival.avg_control_survival:.2f}")


def query_reproduction_stats(session):
    """Query and display reproduction statistics"""
    logger.info("Querying reproduction statistics...")

    # Get average reproduction success rates
    avg_success_rates = session.query(
        func.avg(ReproductionStats.system_reproduction_success_rate).label(
            "avg_system"
        ),
        func.avg(ReproductionStats.independent_reproduction_success_rate).label(
            "avg_independent"
        ),
        func.avg(ReproductionStats.control_reproduction_success_rate).label(
            "avg_control"
        ),
    ).one()

    print("\nAverage Reproduction Success Rates:")
    print(f"System: {avg_success_rates.avg_system:.4f}")
    print(f"Independent: {avg_success_rates.avg_independent:.4f}")
    print(f"Control: {avg_success_rates.avg_control:.4f}")

    # Get average first reproduction times
    avg_first_repro = session.query(
        func.avg(ReproductionStats.system_first_reproduction_time).label("avg_system"),
        func.avg(ReproductionStats.independent_first_reproduction_time).label(
            "avg_independent"
        ),
        func.avg(ReproductionStats.control_first_reproduction_time).label(
            "avg_control"
        ),
    ).one()

    print("\nAverage First Reproduction Times:")
    print(f"System: {avg_first_repro.avg_system:.2f}")
    print(f"Independent: {avg_first_repro.avg_independent:.2f}")
    print(f"Control: {avg_first_repro.avg_control:.2f}")

    # Get average reproduction efficiency
    avg_efficiency = session.query(
        func.avg(ReproductionStats.system_reproduction_efficiency).label("avg_system"),
        func.avg(ReproductionStats.independent_reproduction_efficiency).label(
            "avg_independent"
        ),
        func.avg(ReproductionStats.control_reproduction_efficiency).label(
            "avg_control"
        ),
    ).one()

    print("\nAverage Reproduction Efficiency:")
    print(f"System: {avg_efficiency.avg_system:.4f}")
    print(f"Independent: {avg_efficiency.avg_independent:.4f}")
    print(f"Control: {avg_efficiency.avg_control:.4f}")


def query_dominance_switching(session):
    """Query and display dominance switching statistics"""
    logger.info("Querying dominance switching statistics...")

    # Get average switching metrics
    avg_switching = session.query(
        func.avg(DominanceSwitching.total_switches).label("avg_switches"),
        func.avg(DominanceSwitching.switches_per_step).label("avg_switches_per_step"),
        func.avg(DominanceSwitching.early_phase_switches).label("avg_early"),
        func.avg(DominanceSwitching.middle_phase_switches).label("avg_middle"),
        func.avg(DominanceSwitching.late_phase_switches).label("avg_late"),
    ).one()

    print("\nAverage Switching Metrics:")
    print(f"Total Switches: {avg_switching.avg_switches:.2f}")
    print(f"Switches per Step: {avg_switching.avg_switches_per_step:.4f}")
    print(f"Early Phase Switches: {avg_switching.avg_early:.2f}")
    print(f"Middle Phase Switches: {avg_switching.avg_middle:.2f}")
    print(f"Late Phase Switches: {avg_switching.avg_late:.2f}")

    # Get average transition probabilities
    avg_transitions = session.query(
        func.avg(DominanceSwitching.system_to_system).label("sys_to_sys"),
        func.avg(DominanceSwitching.system_to_independent).label("sys_to_ind"),
        func.avg(DominanceSwitching.system_to_control).label("sys_to_ctrl"),
        func.avg(DominanceSwitching.independent_to_system).label("ind_to_sys"),
        func.avg(DominanceSwitching.independent_to_independent).label("ind_to_ind"),
        func.avg(DominanceSwitching.independent_to_control).label("ind_to_ctrl"),
        func.avg(DominanceSwitching.control_to_system).label("ctrl_to_sys"),
        func.avg(DominanceSwitching.control_to_independent).label("ctrl_to_ind"),
        func.avg(DominanceSwitching.control_to_control).label("ctrl_to_ctrl"),
    ).one()

    print("\nAverage Transition Probabilities:")
    print("From System:")
    print(f"  To System: {avg_transitions.sys_to_sys:.4f}")
    print(f"  To Independent: {avg_transitions.sys_to_ind:.4f}")
    print(f"  To Control: {avg_transitions.sys_to_ctrl:.4f}")
    print("From Independent:")
    print(f"  To System: {avg_transitions.ind_to_sys:.4f}")
    print(f"  To Independent: {avg_transitions.ind_to_ind:.4f}")
    print(f"  To Control: {avg_transitions.ind_to_ctrl:.4f}")
    print("From Control:")
    print(f"  To System: {avg_transitions.ctrl_to_sys:.4f}")
    print(f"  To Independent: {avg_transitions.ctrl_to_ind:.4f}")
    print(f"  To Control: {avg_transitions.ctrl_to_ctrl:.4f}")


def query_resource_distribution(session):
    """Query and display resource distribution statistics"""
    logger.info("Querying resource distribution statistics...")

    # Get average resource distances
    avg_distances = session.query(
        func.avg(ResourceDistribution.systemagent_avg_resource_dist).label(
            "sys_avg_dist"
        ),
        func.avg(ResourceDistribution.independentagent_avg_resource_dist).label(
            "ind_avg_dist"
        ),
        func.avg(ResourceDistribution.controlagent_avg_resource_dist).label(
            "ctrl_avg_dist"
        ),
    ).one()

    print("\nAverage Resource Distances:")
    print(f"System Agents: {avg_distances.sys_avg_dist:.4f}")
    print(f"Independent Agents: {avg_distances.ind_avg_dist:.4f}")
    print(f"Control Agents: {avg_distances.ctrl_avg_dist:.4f}")

    # Get average resources in range
    avg_resources = session.query(
        func.avg(ResourceDistribution.systemagent_resources_in_range).label(
            "sys_resources"
        ),
        func.avg(ResourceDistribution.independentagent_resources_in_range).label(
            "ind_resources"
        ),
        func.avg(ResourceDistribution.controlagent_resources_in_range).label(
            "ctrl_resources"
        ),
    ).one()

    print("\nAverage Resources in Range:")
    print(f"System Agents: {avg_resources.sys_resources:.4f}")
    print(f"Independent Agents: {avg_resources.ind_resources:.4f}")
    print(f"Control Agents: {avg_resources.ctrl_resources:.4f}")


def query_high_low_switching(session):
    """Query and display high vs low switching comparison statistics"""
    logger.info("Querying high vs low switching comparison statistics...")

    # Get average reproduction success rates for high vs low switching
    avg_success_rates = session.query(
        func.avg(
            HighLowSwitchingComparison.system_reproduction_success_rate_high_switching_mean
        ).label("sys_high"),
        func.avg(
            HighLowSwitchingComparison.system_reproduction_success_rate_low_switching_mean
        ).label("sys_low"),
        func.avg(
            HighLowSwitchingComparison.independent_reproduction_success_rate_high_switching_mean
        ).label("ind_high"),
        func.avg(
            HighLowSwitchingComparison.independent_reproduction_success_rate_low_switching_mean
        ).label("ind_low"),
        func.avg(
            HighLowSwitchingComparison.control_reproduction_success_rate_high_switching_mean
        ).label("ctrl_high"),
        func.avg(
            HighLowSwitchingComparison.control_reproduction_success_rate_low_switching_mean
        ).label("ctrl_low"),
    ).one()

    print("\nAverage Reproduction Success Rates (High vs Low Switching):")
    print(f"System - High Switching: {avg_success_rates.sys_high:.4f}")
    print(f"System - Low Switching: {avg_success_rates.sys_low:.4f}")
    print(f"Independent - High Switching: {avg_success_rates.ind_high:.4f}")
    print(f"Independent - Low Switching: {avg_success_rates.ind_low:.4f}")
    print(f"Control - High Switching: {avg_success_rates.ctrl_high:.4f}")
    print(f"Control - Low Switching: {avg_success_rates.ctrl_low:.4f}")


def query_correlation_analysis(session):
    """Query and display correlation analysis statistics"""
    logger.info("Querying correlation analysis statistics...")

    # Get average reproduction correlations
    avg_repro_corr = session.query(
        func.avg(CorrelationAnalysis.repro_corr_system_reproduction_success_rate).label(
            "sys_success_rate"
        ),
        func.avg(
            CorrelationAnalysis.repro_corr_independent_reproduction_success_rate
        ).label("ind_success_rate"),
    ).one()

    print("\nAverage Reproduction Correlations:")
    print(f"System Success Rate: {avg_repro_corr.sys_success_rate:.4f}")
    print(f"Independent Success Rate: {avg_repro_corr.ind_success_rate:.4f}")

    # Get average timing correlations
    avg_timing_corr = session.query(
        func.avg(
            CorrelationAnalysis.first_reproduction_timing_correlation_system_first_reproduction_time
        ).label("sys_time"),
        func.avg(
            CorrelationAnalysis.first_reproduction_timing_correlation_independent_first_reproduction_time
        ).label("ind_time"),
        func.avg(
            CorrelationAnalysis.first_reproduction_timing_correlation_control_first_reproduction_time
        ).label("ctrl_time"),
    ).one()

    print("\nAverage First Reproduction Timing Correlations:")
    print(f"System: {avg_timing_corr.sys_time:.4f}")
    print(f"Independent: {avg_timing_corr.ind_time:.4f}")
    print(f"Control: {avg_timing_corr.ctrl_time:.4f}")


def export_query_to_csv(session, query, output_file):
    """Execute a custom SQL query and export results to CSV"""
    logger.info(f"Executing custom query and exporting to {output_file}...")

    try:
        # Execute the query
        result = session.execute(text(query))

        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall())
        if not df.empty:
            df.columns = result.keys()

        # Export to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully exported query results to {output_file}")
        print(f"\nExported query results to {output_file}")
        print(f"Result shape: {df.shape}")

        return True
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return False


def load_data_from_db(db_path="sqlite:///dominance.db"):
    """
    Load data from the database into a pandas DataFrame for analysis and visualization.

    Parameters
    ----------
    db_path : str
        Path to the SQLAlchemy database

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with all data from the database
    """
    logger.info(f"Loading data from database: {db_path}")

    try:
        # Initialize database connection
        engine = create_engine(db_path)
        Session = sessionmaker(bind=engine)
        session = Session()

        # SQL query to join all tables with explicit column selection to avoid duplicates
        query = """
        SELECT
            s.iteration,
            dm.population_dominance,
            dm.survival_dominance,
            dm.comprehensive_dominance,
            dm.system_dominance_score,
            dm.independent_dominance_score,
            dm.control_dominance_score,
            dm.system_auc,
            dm.independent_auc,
            dm.control_auc,
            dm.system_recency_weighted_auc,
            dm.independent_recency_weighted_auc,
            dm.control_recency_weighted_auc,
            dm.system_dominance_duration,
            dm.independent_dominance_duration,
            dm.control_dominance_duration,
            dm.system_growth_trend,
            dm.independent_growth_trend,
            dm.control_growth_trend,
            dm.system_final_ratio,
            dm.independent_final_ratio,
            dm.control_final_ratio,

            ap.system_agents,
            ap.independent_agents,
            ap.control_agents,
            ap.total_agents,
            ap.final_step,
            ap.system_count,
            ap.system_alive,
            ap.system_dead,
            ap.system_avg_survival,
            ap.system_dead_ratio,
            ap.independent_count,
            ap.independent_alive,
            ap.independent_dead,
            ap.independent_avg_survival,
            ap.independent_dead_ratio,
            ap.control_count,
            ap.control_alive,
            ap.control_dead,
            ap.control_avg_survival,
            ap.control_dead_ratio,
            ap.initial_system_count,
            ap.initial_independent_count,
            ap.initial_control_count,
            ap.initial_resource_count,
            ap.initial_resource_amount,

            rs.system_reproduction_attempts,
            rs.system_reproduction_successes,
            rs.system_reproduction_failures,
            rs.system_reproduction_success_rate,
            rs.system_first_reproduction_time,
            rs.system_reproduction_efficiency,
            rs.system_avg_resources_per_reproduction,
            rs.system_avg_offspring_resources,
            rs.independent_reproduction_attempts,
            rs.independent_reproduction_successes,
            rs.independent_reproduction_failures,
            rs.independent_reproduction_success_rate,
            rs.independent_first_reproduction_time,
            rs.independent_reproduction_efficiency,
            rs.independent_avg_resources_per_reproduction,
            rs.independent_avg_offspring_resources,
            rs.control_reproduction_attempts,
            rs.control_reproduction_successes,
            rs.control_reproduction_failures,
            rs.control_reproduction_success_rate,
            rs.control_first_reproduction_time,
            rs.control_reproduction_efficiency,
            rs.control_avg_resources_per_reproduction,
            rs.control_avg_offspring_resources,
            rs.independent_vs_control_first_reproduction_advantage,
            rs.independent_vs_control_reproduction_efficiency_advantage,
            rs.independent_vs_control_reproduction_rate_advantage,
            rs.system_vs_independent_reproduction_rate_advantage,
            rs.system_vs_control_reproduction_rate_advantage,
            rs.system_vs_independent_reproduction_efficiency_advantage,
            rs.system_vs_control_first_reproduction_advantage,
            rs.system_vs_independent_first_reproduction_advantage,
            rs.system_vs_control_reproduction_efficiency_advantage,

            ds.total_switches,
            ds.switches_per_step,
            ds.dominance_stability,
            ds.system_avg_dominance_period,
            ds.independent_avg_dominance_period,
            ds.control_avg_dominance_period,
            ds.early_phase_switches,
            ds.middle_phase_switches,
            ds.late_phase_switches,
            ds.control_avg_switches,
            ds.independent_avg_switches,
            ds.system_avg_switches,
            ds.system_to_system,
            ds.system_to_independent,
            ds.system_to_control,
            ds.independent_to_system,
            ds.independent_to_independent,
            ds.independent_to_control,
            ds.control_to_system,
            ds.control_to_independent,
            ds.control_to_control,

            rd.systemagent_avg_resource_dist,
            rd.systemagent_weighted_resource_dist,
            rd.systemagent_nearest_resource_dist,
            rd.systemagent_resources_in_range,
            rd.systemagent_resource_amount_in_range,
            rd.independentagent_avg_resource_dist,
            rd.independentagent_weighted_resource_dist,
            rd.independentagent_nearest_resource_dist,
            rd.independentagent_resources_in_range,
            rd.independentagent_resource_amount_in_range,
            rd.controlagent_avg_resource_dist,
            rd.controlagent_weighted_resource_dist,
            rd.controlagent_nearest_resource_dist,
            rd.controlagent_resources_in_range,
            rd.controlagent_resource_amount_in_range,
            rd.positive_corr_controlagent_resource_amount_in_range,
            rd.positive_corr_systemagent_avg_resource_dist,
            rd.positive_corr_systemagent_weighted_resource_dist,
            rd.positive_corr_independentagent_avg_resource_dist,
            rd.positive_corr_independentagent_weighted_resource_dist,
            rd.negative_corr_systemagent_resource_amount_in_range,
            rd.negative_corr_systemagent_nearest_resource_dist,
            rd.negative_corr_independentagent_resource_amount_in_range,
            rd.negative_corr_controlagent_avg_resource_dist,
            rd.negative_corr_controlagent_nearest_resource_dist
        FROM
            simulations s
        LEFT JOIN
            dominance_metrics dm ON s.id = dm.simulation_id
        LEFT JOIN
            agent_populations ap ON s.id = ap.simulation_id
        LEFT JOIN
            reproduction_stats rs ON s.id = rs.simulation_id
        LEFT JOIN
            dominance_switching ds ON s.id = ds.simulation_id
        LEFT JOIN
            resource_distribution rd ON s.id = rd.simulation_id
        LEFT JOIN
            high_low_switching_comparison hl ON s.id = hl.simulation_id
        LEFT JOIN
            correlation_analysis ca ON s.id = ca.simulation_id
        """

        # Execute query and load into DataFrame
        df = pd.read_sql(text(query), engine)

        logger.info(f"Loaded {len(df)} rows from database")
        return df

    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        if "session" in locals():
            session.close()


def main():
    parser = argparse.ArgumentParser(description="Query the dominance database")
    parser.add_argument(
        "--db-path", default="sqlite:///dominance.db", help="Path to the database"
    )
    parser.add_argument(
        "--query-type",
        choices=[
            "all",
            "dominance",
            "population",
            "reproduction",
            "switching",
            "resources",
            "high-low",
            "correlation",
        ],
        default="all",
        help="Type of query to run",
    )
    parser.add_argument("--custom-query", help="Custom SQL query to execute")
    parser.add_argument("--output-file", help="Output file for custom query results")

    args = parser.parse_args()

    # Create session
    session, engine = get_session(args.db_path)

    try:
        if args.custom_query and args.output_file:
            export_query_to_csv(session, args.custom_query, args.output_file)
        else:
            if args.query_type in ["all", "dominance"]:
                query_dominance_metrics(session)

            if args.query_type in ["all", "population"]:
                query_agent_populations(session)

            if args.query_type in ["all", "reproduction"]:
                query_reproduction_stats(session)

            if args.query_type in ["all", "switching"]:
                query_dominance_switching(session)

            if args.query_type in ["all", "resources"]:
                query_resource_distribution(session)

            if args.query_type in ["all", "high-low"]:
                query_high_low_switching(session)

            if args.query_type in ["all", "correlation"]:
                query_correlation_analysis(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
