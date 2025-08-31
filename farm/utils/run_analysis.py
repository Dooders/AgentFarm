import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from farm.charts.chart_analyzer import ChartAnalyzer
from farm.database.database import SimulationDatabase
from farm.core.services import IConfigService, EnvConfigService


def setup_environment(config_service: IConfigService):
    """Load environment variables from .env file and check required variables."""
    # Load environment variables from .env file
    load_dotenv()

    if not config_service.get_openai_api_key():
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set in .env file. "
            "Please check your .env file contains the API key."
        )


def load_data(database_path: str) -> pd.DataFrame:
    """Load data from the SQLite database."""
    try:
        engine = create_engine(f"sqlite:///{database_path}")
        df = pd.read_sql("SELECT * FROM agent_actions", engine)
        print(f"Successfully loaded {len(df)} records from database.")
        return df
    except Exception as e:
        raise Exception(f"Error loading data from database: {str(e)}")


def run_analysis(database_path: str, output_dir: str, save_charts: bool = True):
    """
    Run the complete chart analysis pipeline.

    Args:
        database_path: Path to the SQLite database
        output_dir: Directory to save output files
        save_charts: Whether to save charts to files or keep in memory
    """
    # Setup environment
    setup_environment()

    # Create output directory if saving charts
    if save_charts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data from database...")
    df = load_data(database_path)

    # Initialize analyzer and run analysis
    print("Initializing chart analyzer...")
    db = SimulationDatabase(database_path)
    analyzer = ChartAnalyzer(
        db, output_path if save_charts else None, save_charts=save_charts
    )

    print("Running chart analysis...")
    analyses = analyzer.analyze_all_charts(output_path if save_charts else None)

    # Print summary
    print("\nAnalysis Complete!")
    if save_charts:
        print(f"Charts and analyses have been saved to: {output_dir}")
    print("\nAnalysis Summary:")
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} ===")
        # Print first 200 characters of each analysis as preview
        preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
        print(preview)


def main():
    print("Loading data from database...")
    connection_string = "sqlite:///simulations/simulation.db"
    engine = create_engine(connection_string)

    # Load both actions and agents data
    actions_df = pd.read_sql("SELECT * FROM agent_actions", engine)
    agents_df = pd.read_sql(
        "SELECT * FROM agents", engine
    )  # Note: table name might be case-sensitive

    print(
        f"Successfully loaded {len(actions_df)} action records and {len(agents_df)} agent records from database."
    )

    print("Initializing chart analyzer...")
    db = SimulationDatabase("sqlite:///simulations/simulation.db")
    analyzer = ChartAnalyzer(db)

    print("Running chart analysis...")
    analyses = analyzer.analyze_all_charts()

    print("\nAnalysis Complete!")
    print("Charts and analyses have been saved to: chart_analysis")

    print("\nAnalysis Summary:\n")
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} Analysis ===")
        print(analysis)


if __name__ == "__main__":
    main()
