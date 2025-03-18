"""Query script for the experiment database.

This script shows how to query the experiment database to get statistics
about multiple simulations.
"""

import logging
import os
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def query_database(db_path: str) -> None:
    """Query the experiment database and print statistics.

    Parameters
    ----------
    db_path : str
        Path to the database file
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Query experiment details
        cursor.execute("SELECT * FROM experiments")
        experiment = cursor.fetchone()
        if experiment:
            (
                experiment_id,
                name,
                description,
                hypothesis,
                creation_date,
                last_updated,
                status,
                tags,
                variables,
                results_summary,
                notes,
            ) = experiment
            logger.info(f"Experiment: {name} (ID: {experiment_id})")
            logger.info(f"Status: {status}")
            logger.info(f"Created: {creation_date}")
            logger.info(f"Last updated: {last_updated}")

        # Query simulations
        cursor.execute(
            "SELECT simulation_id, status, start_time, end_time FROM simulations"
        )
        simulations = cursor.fetchall()
        logger.info(f"Number of simulations: {len(simulations)}")
        for sim in simulations:
            sim_id, status, start_time, end_time = sim
            logger.info(f"  Simulation {sim_id}: {status}, {start_time} -> {end_time}")

            # Query step data for this simulation
            cursor.execute(
                f"SELECT COUNT(*) FROM simulation_steps WHERE simulation_id = ?",
                (sim_id,),
            )
            step_count = cursor.fetchone()[0]
            logger.info(f"    Steps: {step_count}")

            # Query agent states
            cursor.execute(
                f"SELECT COUNT(DISTINCT agent_id) FROM agent_states WHERE simulation_id = ?",
                (sim_id,),
            )
            agent_count = cursor.fetchone()[0]
            logger.info(f"    Agents: {agent_count}")

            # Query actions
            cursor.execute(
                f"SELECT COUNT(*) FROM agent_actions WHERE simulation_id = ?", (sim_id,)
            )
            action_count = cursor.fetchone()[0]
            logger.info(f"    Actions: {action_count}")

            # Query actions by type
            cursor.execute(
                f"""
                SELECT action_type, COUNT(*) 
                FROM agent_actions 
                WHERE simulation_id = ?
                GROUP BY action_type
            """,
                (sim_id,),
            )
            action_types = cursor.fetchall()
            if action_types:
                logger.info(f"    Action types:")
                for action_type, count in action_types:
                    logger.info(f"      {action_type}: {count}")

            # Query reproduction events
            cursor.execute(
                f"SELECT COUNT(*) FROM reproduction_events WHERE simulation_id = ?",
                (sim_id,),
            )
            reproduction_count = cursor.fetchone()[0]
            logger.info(f"    Reproduction events: {reproduction_count}")

            # Query health incidents
            cursor.execute(
                f"SELECT COUNT(*) FROM health_incidents WHERE simulation_id = ?",
                (sim_id,),
            )
            health_count = cursor.fetchone()[0]
            logger.info(f"    Health incidents: {health_count}")

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")

    finally:
        conn.close()


def main():
    """Run the query script."""
    db_path = "test_experiment.db"
    logger.info(f"Querying database: {db_path}")
    query_database(db_path)
    logger.info("Query completed")


if __name__ == "__main__":
    main()
