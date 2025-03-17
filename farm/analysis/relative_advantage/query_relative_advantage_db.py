#!/usr/bin/env python3
"""
Query utility for the Relative Advantage database.

This script provides functions to query the Relative Advantage database
and export results to CSV files.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from farm.analysis.relative_advantage.sqlalchemy_models import (
    AdvantageDominanceCorrelation,
    CompositeAdvantage,
    PopulationGrowth,
    ReproductionAdvantage,
    ResourceAcquisition,
    Simulation,
    SurvivalAdvantage,
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


def query_resource_advantages(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query resource acquisition advantages.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying resource acquisition advantages")

    query = (
        session.query(
            Simulation.iteration,
            ResourceAcquisition.system_late_phase,
            ResourceAcquisition.independent_late_phase,
            ResourceAcquisition.control_late_phase,
            ResourceAcquisition.system_vs_independent_late_phase_advantage,
            ResourceAcquisition.system_vs_control_late_phase_advantage,
            ResourceAcquisition.independent_vs_control_late_phase_advantage,
            ResourceAcquisition.system_vs_independent_advantage_trajectory,
            ResourceAcquisition.system_vs_control_advantage_trajectory,
            ResourceAcquisition.independent_vs_control_advantage_trajectory,
        )
        .join(ResourceAcquisition, Simulation.id == ResourceAcquisition.simulation_id)
        .order_by(ResourceAcquisition.system_vs_independent_late_phase_advantage.desc())
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved resource acquisition query results to {output_file}")

    return df


def query_reproduction_advantages(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query reproduction advantages.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying reproduction advantages")

    query = (
        session.query(
            Simulation.iteration,
            ReproductionAdvantage.system_success_rate,
            ReproductionAdvantage.independent_success_rate,
            ReproductionAdvantage.control_success_rate,
            ReproductionAdvantage.system_vs_independent_success_rate_advantage,
            ReproductionAdvantage.system_vs_control_success_rate_advantage,
            ReproductionAdvantage.independent_vs_control_success_rate_advantage,
            ReproductionAdvantage.system_first_reproduction_time,
            ReproductionAdvantage.independent_first_reproduction_time,
            ReproductionAdvantage.control_first_reproduction_time,
            ReproductionAdvantage.system_vs_independent_timing_advantage,
            ReproductionAdvantage.system_vs_control_timing_advantage,
            ReproductionAdvantage.independent_vs_control_timing_advantage,
        )
        .join(
            ReproductionAdvantage, Simulation.id == ReproductionAdvantage.simulation_id
        )
        .order_by(
            ReproductionAdvantage.system_vs_independent_success_rate_advantage.desc()
        )
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved reproduction advantages query results to {output_file}")

    return df


def query_survival_advantages(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query survival advantages.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying survival advantages")

    query = (
        session.query(
            Simulation.iteration,
            SurvivalAdvantage.system_survival_rate,
            SurvivalAdvantage.independent_survival_rate,
            SurvivalAdvantage.control_survival_rate,
            SurvivalAdvantage.system_vs_independent_survival_rate_advantage,
            SurvivalAdvantage.system_vs_control_survival_rate_advantage,
            SurvivalAdvantage.independent_vs_control_survival_rate_advantage,
            SurvivalAdvantage.system_avg_lifespan,
            SurvivalAdvantage.independent_avg_lifespan,
            SurvivalAdvantage.control_avg_lifespan,
        )
        .join(SurvivalAdvantage, Simulation.id == SurvivalAdvantage.simulation_id)
        .order_by(
            SurvivalAdvantage.system_vs_independent_survival_rate_advantage.desc()
        )
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved survival advantages query results to {output_file}")

    return df


def query_population_growth(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query population growth advantages.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying population growth advantages")

    query = (
        session.query(
            Simulation.iteration,
            PopulationGrowth.system_final_population,
            PopulationGrowth.independent_final_population,
            PopulationGrowth.control_final_population,
            PopulationGrowth.system_vs_independent_final_population_advantage,
            PopulationGrowth.system_vs_control_final_population_advantage,
            PopulationGrowth.independent_vs_control_final_population_advantage,
            PopulationGrowth.system_late_growth_rate,
            PopulationGrowth.independent_late_growth_rate,
            PopulationGrowth.control_late_growth_rate,
        )
        .join(PopulationGrowth, Simulation.id == PopulationGrowth.simulation_id)
        .order_by(
            PopulationGrowth.system_vs_independent_final_population_advantage.desc()
        )
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved population growth query results to {output_file}")

    return df


def query_composite_advantages(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query composite advantages.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying composite advantages")

    query = (
        session.query(
            Simulation.iteration,
            CompositeAdvantage.system_vs_independent_score,
            CompositeAdvantage.system_vs_control_score,
            CompositeAdvantage.independent_vs_control_score,
            CompositeAdvantage.system_vs_independent_resource_component,
            CompositeAdvantage.system_vs_independent_reproduction_component,
            CompositeAdvantage.system_vs_independent_survival_component,
            CompositeAdvantage.system_vs_independent_population_component,
            CompositeAdvantage.system_vs_control_resource_component,
            CompositeAdvantage.system_vs_control_reproduction_component,
            CompositeAdvantage.system_vs_control_survival_component,
            CompositeAdvantage.system_vs_control_population_component,
        )
        .join(CompositeAdvantage, Simulation.id == CompositeAdvantage.simulation_id)
        .order_by(CompositeAdvantage.system_vs_independent_score.desc())
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved composite advantages query results to {output_file}")

    return df


def query_advantage_dominance_correlations(
    session: Session, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Query advantage-dominance correlations.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Querying advantage-dominance correlations")

    query = (
        session.query(
            Simulation.iteration,
            AdvantageDominanceCorrelation.dominant_type,
            AdvantageDominanceCorrelation.resource_late_phase_correlation,
            AdvantageDominanceCorrelation.reproduction_success_rate_correlation,
            AdvantageDominanceCorrelation.reproduction_efficiency_correlation,
            AdvantageDominanceCorrelation.survival_rate_correlation,
            AdvantageDominanceCorrelation.final_population_correlation,
            AdvantageDominanceCorrelation.composite_advantage_correlation,
        )
        .join(
            AdvantageDominanceCorrelation,
            Simulation.id == AdvantageDominanceCorrelation.simulation_id,
        )
        .order_by(AdvantageDominanceCorrelation.composite_advantage_correlation.desc())
    )

    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)

    # Save to CSV if output file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(
            f"Saved advantage-dominance correlations query results to {output_file}"
        )

    return df


def run_custom_query(
    session: Session, query_str: str, output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Run a custom SQL query.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session
    query_str : str
        SQL query string
    output_file : str, optional
        Path to save the query results as CSV

    Returns
    -------
    pandas.DataFrame
        DataFrame containing query results
    """
    logging.info("Running custom query")

    try:
        # Execute the query
        result = session.execute(text(query_str))

        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall())
        if not df.empty:
            df.columns = result.keys()

        # Save to CSV if output file is provided
        if output_file and not df.empty:
            df.to_csv(output_file, index=False)
            logging.info(f"Saved custom query results to {output_file}")

        return df
    except Exception as e:
        logging.error(f"Error executing custom query: {e}")
        return pd.DataFrame()


def main():
    """Main function to query the database."""
    parser = argparse.ArgumentParser(
        description="Query the Relative Advantage database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="sqlite:///relative_advantage.db",
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        choices=[
            "resource",
            "reproduction",
            "survival",
            "population",
            "composite",
            "correlation",
            "all",
        ],
        default="all",
        help="Type of query to run",
    )
    parser.add_argument("--custom-query", type=str, help="Custom SQL query to run")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="query_results",
        help="Directory to save query results",
    )
    parser.add_argument(
        "--output-file", type=str, help="File to save custom query results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize database connection
    engine = init_db(args.db_path)
    session = get_session(engine)

    try:
        # Run queries based on query type
        if args.custom_query:
            output_file = args.output_file or os.path.join(
                args.output_dir, "custom_query_results.csv"
            )
            df = run_custom_query(session, args.custom_query, output_file)
            if df.empty:
                logging.warning("Custom query returned no results")
            else:
                logging.info(f"Custom query returned {len(df)} rows")
                print(df.head(10))
        else:
            if args.query_type in ["resource", "all"]:
                output_file = os.path.join(args.output_dir, "resource_advantages.csv")
                df = query_resource_advantages(session, output_file)
                print("\nResource Acquisition Advantages:")
                print(df.head(10))

            if args.query_type in ["reproduction", "all"]:
                output_file = os.path.join(
                    args.output_dir, "reproduction_advantages.csv"
                )
                df = query_reproduction_advantages(session, output_file)
                print("\nReproduction Advantages:")
                print(df.head(10))

            if args.query_type in ["survival", "all"]:
                output_file = os.path.join(args.output_dir, "survival_advantages.csv")
                df = query_survival_advantages(session, output_file)
                print("\nSurvival Advantages:")
                print(df.head(10))

            if args.query_type in ["population", "all"]:
                output_file = os.path.join(args.output_dir, "population_growth.csv")
                df = query_population_growth(session, output_file)
                print("\nPopulation Growth Advantages:")
                print(df.head(10))

            if args.query_type in ["composite", "all"]:
                output_file = os.path.join(args.output_dir, "composite_advantages.csv")
                df = query_composite_advantages(session, output_file)
                print("\nComposite Advantages:")
                print(df.head(10))

            if args.query_type in ["correlation", "all"]:
                output_file = os.path.join(
                    args.output_dir, "advantage_dominance_correlations.csv"
                )
                df = query_advantage_dominance_correlations(session, output_file)
                print("\nAdvantage-Dominance Correlations:")
                print(df.head(10))

    except Exception as e:
        logging.error(f"Error querying database: {e}")
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
