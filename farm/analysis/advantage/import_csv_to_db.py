#!/usr/bin/env python3
"""
Import CSV data into the Relative Advantage database.

This script reads CSV files containing relative advantage analysis results
and imports them into a structured SQLite database using SQLAlchemy models.
"""

import argparse
import logging
import sys
from typing import Dict

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from farm.analysis.advantage.sqlalchemy_models import (
    AdvantageDominanceCorrelation,
    CompositeAdvantage,
    ReproductionAdvantage,
    ResourceAcquisition,
    Simulation,
    get_session,
    init_db,
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def read_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Read CSV data into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the CSV data
    """
    logging.info(f"Reading CSV data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully read {len(df)} rows from CSV")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)


def import_simulation_data(df: pd.DataFrame, session) -> Dict[int, int]:
    """
    Import simulation data into the database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing simulation data
    session : SQLAlchemy session
        Database session

    Returns
    -------
    dict
        Mapping of iteration numbers to simulation IDs
    """
    logging.info("Importing simulation data")
    iteration_to_id = {}

    for _, row in df.iterrows():
        iteration = int(row["iteration"])

        # Check if simulation already exists
        existing = session.query(Simulation).filter_by(iteration=iteration).first()
        if existing:
            logging.debug(f"Simulation {iteration} already exists, skipping")
            iteration_to_id[iteration] = existing.id
            continue

        # Create new simulation
        sim = Simulation(iteration=iteration)
        session.add(sim)

        try:
            session.flush()
            iteration_to_id[iteration] = sim.id
            logging.debug(f"Added simulation {iteration} with ID {sim.id}")
        except SQLAlchemyError as e:
            logging.error(f"Error adding simulation {iteration}: {e}")
            session.rollback()

    session.commit()
    logging.info(f"Imported {len(iteration_to_id)} simulations")
    return iteration_to_id


def import_resource_acquisition_data(
    df: pd.DataFrame, iteration_to_id: Dict[int, int], session
) -> None:
    """
    Import resource acquisition data into the database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing resource acquisition data
    iteration_to_id : dict
        Mapping of iteration numbers to simulation IDs
    session : SQLAlchemy session
        Database session
    """
    logging.info("Importing resource acquisition data")
    count = 0

    for _, row in df.iterrows():
        iteration = int(row["iteration"])
        sim_id = iteration_to_id.get(iteration)
        if not sim_id:
            logging.warning(
                f"No simulation ID found for iteration {iteration}, skipping"
            )
            continue

        # Check if data already exists
        existing = (
            session.query(ResourceAcquisition).filter_by(simulation_id=sim_id).first()
        )
        if existing:
            logging.debug(
                f"Resource acquisition data for simulation {iteration} already exists, skipping"
            )
            continue

        # Create new resource acquisition data
        resource_data = ResourceAcquisition(
            simulation_id=sim_id,
            # Raw resource metrics for each agent type
            system_early_phase=float(row.get("system_early_phase_resources", 0)),
            system_mid_phase=float(row.get("system_mid_phase_resources", 0)),
            system_late_phase=float(row.get("system_late_phase_resources", 0)),
            independent_early_phase=float(
                row.get("independent_early_phase_resources", 0)
            ),
            independent_mid_phase=float(row.get("independent_mid_phase_resources", 0)),
            independent_late_phase=float(
                row.get("independent_late_phase_resources", 0)
            ),
            control_early_phase=float(row.get("control_early_phase_resources", 0)),
            control_mid_phase=float(row.get("control_mid_phase_resources", 0)),
            control_late_phase=float(row.get("control_late_phase_resources", 0)),
            # System vs Independent advantages
            system_vs_independent_early_phase_advantage=float(
                row.get("system_vs_independent_early_phase_resource_advantage", 0)
            ),
            system_vs_independent_mid_phase_advantage=float(
                row.get("system_vs_independent_mid_phase_resource_advantage", 0)
            ),
            system_vs_independent_late_phase_advantage=float(
                row.get("system_vs_independent_late_phase_resource_advantage", 0)
            ),
            system_vs_independent_advantage_trajectory=float(
                row.get("system_vs_independent_resource_advantage_trajectory", 0)
            ),
            # System vs Control advantages
            system_vs_control_early_phase_advantage=float(
                row.get("system_vs_control_early_phase_resource_advantage", 0)
            ),
            system_vs_control_mid_phase_advantage=float(
                row.get("system_vs_control_mid_phase_resource_advantage", 0)
            ),
            system_vs_control_late_phase_advantage=float(
                row.get("system_vs_control_late_phase_resource_advantage", 0)
            ),
            system_vs_control_advantage_trajectory=float(
                row.get("system_vs_control_resource_advantage_trajectory", 0)
            ),
            # Independent vs Control advantages
            independent_vs_control_early_phase_advantage=float(
                row.get("independent_vs_control_early_phase_resource_advantage", 0)
            ),
            independent_vs_control_mid_phase_advantage=float(
                row.get("independent_vs_control_mid_phase_resource_advantage", 0)
            ),
            independent_vs_control_late_phase_advantage=float(
                row.get("independent_vs_control_late_phase_resource_advantage", 0)
            ),
            independent_vs_control_advantage_trajectory=float(
                row.get("independent_vs_control_resource_advantage_trajectory", 0)
            ),
        )

        session.add(resource_data)
        count += 1

        if count % 100 == 0:
            session.commit()
            logging.debug(f"Committed {count} resource acquisition records")

    session.commit()
    logging.info(f"Imported {count} resource acquisition records")


def import_reproduction_advantage_data(
    df: pd.DataFrame, iteration_to_id: Dict[int, int], session
) -> None:
    """
    Import reproduction advantage data into the database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing reproduction advantage data
    iteration_to_id : dict
        Mapping of iteration numbers to simulation IDs
    session : SQLAlchemy session
        Database session
    """
    logging.info("Importing reproduction advantage data")
    count = 0

    for _, row in df.iterrows():
        iteration = int(row["iteration"])
        sim_id = iteration_to_id.get(iteration)
        if not sim_id:
            logging.warning(
                f"No simulation ID found for iteration {iteration}, skipping"
            )
            continue

        # Check if data already exists
        existing = (
            session.query(ReproductionAdvantage).filter_by(simulation_id=sim_id).first()
        )
        if existing:
            logging.debug(
                f"Reproduction advantage data for simulation {iteration} already exists, skipping"
            )
            continue

        # Create new reproduction advantage data
        reproduction_data = ReproductionAdvantage(
            simulation_id=sim_id,
            # Raw reproduction metrics for each agent type
            system_success_rate=float(row.get("system_reproduction_success_rate", 0)),
            system_total_offspring=int(row.get("system_total_offspring", 0)),
            system_reproduction_efficiency=float(
                row.get("system_reproduction_efficiency", 0)
            ),
            system_first_reproduction_time=float(
                row.get("system_first_reproduction_time", 0)
            ),
            independent_success_rate=float(
                row.get("independent_reproduction_success_rate", 0)
            ),
            independent_total_offspring=int(row.get("independent_total_offspring", 0)),
            independent_reproduction_efficiency=float(
                row.get("independent_reproduction_efficiency", 0)
            ),
            independent_first_reproduction_time=float(
                row.get("independent_first_reproduction_time", 0)
            ),
            control_success_rate=float(row.get("control_reproduction_success_rate", 0)),
            control_total_offspring=int(row.get("control_total_offspring", 0)),
            control_reproduction_efficiency=float(
                row.get("control_reproduction_efficiency", 0)
            ),
            control_first_reproduction_time=float(
                row.get("control_first_reproduction_time", 0)
            ),
            # System vs Independent advantages
            system_vs_independent_success_rate_advantage=float(
                row.get("system_vs_independent_reproduction_success_rate_advantage", 0)
            ),
            system_vs_independent_offspring_advantage=float(
                row.get("system_vs_independent_offspring_advantage", 0)
            ),
            system_vs_independent_efficiency_advantage=float(
                row.get("system_vs_independent_reproduction_efficiency_advantage", 0)
            ),
            system_vs_independent_timing_advantage=float(
                row.get("system_vs_independent_first_reproduction_advantage", 0)
            ),
            # System vs Control advantages
            system_vs_control_success_rate_advantage=float(
                row.get("system_vs_control_reproduction_success_rate_advantage", 0)
            ),
            system_vs_control_offspring_advantage=float(
                row.get("system_vs_control_offspring_advantage", 0)
            ),
            system_vs_control_efficiency_advantage=float(
                row.get("system_vs_control_reproduction_efficiency_advantage", 0)
            ),
            system_vs_control_timing_advantage=float(
                row.get("system_vs_control_first_reproduction_advantage", 0)
            ),
            # Independent vs Control advantages
            independent_vs_control_success_rate_advantage=float(
                row.get("independent_vs_control_reproduction_success_rate_advantage", 0)
            ),
            independent_vs_control_offspring_advantage=float(
                row.get("independent_vs_control_offspring_advantage", 0)
            ),
            independent_vs_control_efficiency_advantage=float(
                row.get("independent_vs_control_reproduction_efficiency_advantage", 0)
            ),
            independent_vs_control_timing_advantage=float(
                row.get("independent_vs_control_first_reproduction_advantage", 0)
            ),
        )

        session.add(reproduction_data)
        count += 1

        if count % 100 == 0:
            session.commit()
            logging.debug(f"Committed {count} reproduction advantage records")

    session.commit()
    logging.info(f"Imported {count} reproduction advantage records")


def import_composite_advantage_data(
    df: pd.DataFrame, iteration_to_id: Dict[int, int], session
) -> None:
    """
    Import composite advantage data into the database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing composite advantage data
    iteration_to_id : dict
        Mapping of iteration numbers to simulation IDs
    session : SQLAlchemy session
        Database session
    """
    logging.info("Importing composite advantage data")
    count = 0

    for _, row in df.iterrows():
        iteration = int(row["iteration"])
        sim_id = iteration_to_id.get(iteration)
        if not sim_id:
            logging.warning(
                f"No simulation ID found for iteration {iteration}, skipping"
            )
            continue

        # Check if data already exists
        existing = (
            session.query(CompositeAdvantage).filter_by(simulation_id=sim_id).first()
        )
        if existing:
            logging.debug(
                f"Composite advantage data for simulation {iteration} already exists, skipping"
            )
            continue

        # Create new composite advantage data
        composite_data = CompositeAdvantage(
            simulation_id=sim_id,
            # System vs Independent composite advantage
            system_vs_independent_score=float(
                row.get("system_vs_independent_composite_advantage", 0)
            ),
            system_vs_independent_resource_component=float(
                row.get("system_vs_independent_resource_component", 0)
            ),
            system_vs_independent_reproduction_component=float(
                row.get("system_vs_independent_reproduction_component", 0)
            ),
            system_vs_independent_survival_component=float(
                row.get("system_vs_independent_survival_component", 0)
            ),
            system_vs_independent_population_component=float(
                row.get("system_vs_independent_population_component", 0)
            ),
            system_vs_independent_combat_component=float(
                row.get("system_vs_independent_combat_component", 0)
            ),
            system_vs_independent_positioning_component=float(
                row.get("system_vs_independent_positioning_component", 0)
            ),
            # System vs Control composite advantage
            system_vs_control_score=float(
                row.get("system_vs_control_composite_advantage", 0)
            ),
            system_vs_control_resource_component=float(
                row.get("system_vs_control_resource_component", 0)
            ),
            system_vs_control_reproduction_component=float(
                row.get("system_vs_control_reproduction_component", 0)
            ),
            system_vs_control_survival_component=float(
                row.get("system_vs_control_survival_component", 0)
            ),
            system_vs_control_population_component=float(
                row.get("system_vs_control_population_component", 0)
            ),
            system_vs_control_combat_component=float(
                row.get("system_vs_control_combat_component", 0)
            ),
            system_vs_control_positioning_component=float(
                row.get("system_vs_control_positioning_component", 0)
            ),
            # Independent vs Control composite advantage
            independent_vs_control_score=float(
                row.get("independent_vs_control_composite_advantage", 0)
            ),
            independent_vs_control_resource_component=float(
                row.get("independent_vs_control_resource_component", 0)
            ),
            independent_vs_control_reproduction_component=float(
                row.get("independent_vs_control_reproduction_component", 0)
            ),
            independent_vs_control_survival_component=float(
                row.get("independent_vs_control_survival_component", 0)
            ),
            independent_vs_control_population_component=float(
                row.get("independent_vs_control_population_component", 0)
            ),
            independent_vs_control_combat_component=float(
                row.get("independent_vs_control_combat_component", 0)
            ),
            independent_vs_control_positioning_component=float(
                row.get("independent_vs_control_positioning_component", 0)
            ),
        )

        session.add(composite_data)
        count += 1

        if count % 100 == 0:
            session.commit()
            logging.debug(f"Committed {count} composite advantage records")

    session.commit()
    logging.info(f"Imported {count} composite advantage records")


def import_advantage_dominance_correlation_data(
    df: pd.DataFrame, iteration_to_id: Dict[int, int], session
) -> None:
    """
    Import advantage-dominance correlation data into the database.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage-dominance correlation data
    iteration_to_id : dict
        Mapping of iteration numbers to simulation IDs
    session : SQLAlchemy session
        Database session
    """
    logging.info("Importing advantage-dominance correlation data")
    count = 0

    for _, row in df.iterrows():
        iteration = int(row["iteration"])
        sim_id = iteration_to_id.get(iteration)
        if not sim_id:
            logging.warning(
                f"No simulation ID found for iteration {iteration}, skipping"
            )
            continue

        # Check if data already exists
        existing = (
            session.query(AdvantageDominanceCorrelation)
            .filter_by(simulation_id=sim_id)
            .first()
        )
        if existing:
            logging.debug(
                f"Advantage-dominance correlation data for simulation {iteration} already exists, skipping"
            )
            continue

        # Create new advantage-dominance correlation data
        correlation_data = AdvantageDominanceCorrelation(
            simulation_id=sim_id,
            # Dominant type in the simulation
            dominant_type=row.get("dominant_type", ""),
            # Resource acquisition correlations
            resource_early_phase_correlation=float(
                row.get("resource_early_phase_correlation", 0)
            ),
            resource_mid_phase_correlation=float(
                row.get("resource_mid_phase_correlation", 0)
            ),
            resource_late_phase_correlation=float(
                row.get("resource_late_phase_correlation", 0)
            ),
            resource_trajectory_correlation=float(
                row.get("resource_trajectory_correlation", 0)
            ),
            # Reproduction correlations
            reproduction_success_rate_correlation=float(
                row.get("reproduction_success_rate_correlation", 0)
            ),
            reproduction_offspring_correlation=float(
                row.get("reproduction_offspring_correlation", 0)
            ),
            reproduction_efficiency_correlation=float(
                row.get("reproduction_efficiency_correlation", 0)
            ),
            reproduction_timing_correlation=float(
                row.get("reproduction_timing_correlation", 0)
            ),
            # Survival correlations
            survival_rate_correlation=float(row.get("survival_rate_correlation", 0)),
            lifespan_correlation=float(row.get("lifespan_correlation", 0)),
            death_rate_correlation=float(row.get("death_rate_correlation", 0)),
            # Population growth correlations
            early_growth_correlation=float(row.get("early_growth_correlation", 0)),
            mid_growth_correlation=float(row.get("mid_growth_correlation", 0)),
            late_growth_correlation=float(row.get("late_growth_correlation", 0)),
            final_population_correlation=float(
                row.get("final_population_correlation", 0)
            ),
            # Combat correlations
            win_rate_correlation=float(row.get("win_rate_correlation", 0)),
            damage_correlation=float(row.get("damage_correlation", 0)),
            # Initial positioning correlations
            avg_distance_correlation=float(row.get("avg_distance_correlation", 0)),
            nearest_distance_correlation=float(
                row.get("nearest_distance_correlation", 0)
            ),
            resources_in_range_correlation=float(
                row.get("resources_in_range_correlation", 0)
            ),
            # Composite advantage correlation
            composite_advantage_correlation=float(
                row.get("composite_advantage_correlation", 0)
            ),
        )

        session.add(correlation_data)
        count += 1

        if count % 100 == 0:
            session.commit()
            logging.debug(f"Committed {count} advantage-dominance correlation records")

    session.commit()
    logging.info(f"Imported {count} advantage-dominance correlation records")


def main():
    """Main function to import CSV data into the database."""
    parser = argparse.ArgumentParser(
        description="Import CSV data into the Relative Advantage database"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/relative_advantage_results.csv",
        help="Path to the CSV file containing relative advantage data",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="sqlite:///relative_advantage.db",
        help="Path to the SQLite database",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Read CSV data
    df = read_csv_data(args.csv_path)

    # Initialize database
    engine = init_db(args.db_path)
    session = get_session(engine)

    try:
        # Import data
        iteration_to_id = import_simulation_data(df, session)
        import_resource_acquisition_data(df, iteration_to_id, session)
        import_reproduction_advantage_data(df, iteration_to_id, session)
        import_composite_advantage_data(df, iteration_to_id, session)
        import_advantage_dominance_correlation_data(df, iteration_to_id, session)

        # Add other import functions as needed

        logging.info("Data import completed successfully")
    except Exception as e:
        logging.error(f"Error importing data: {e}")
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
