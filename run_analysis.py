import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from farm.charts.chart_analyzer import ChartAnalyzer


def setup_environment():
    """Load environment variables from .env file and check required variables."""
    # Load environment variables from .env file
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
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


def run_analysis(database_path: str, output_dir: str):
    """Run the complete chart analysis pipeline."""
    # Setup environment
    setup_environment()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data from database...")
    df = load_data(database_path)

    # Initialize analyzer and run analysis
    print("Initializing chart analyzer...")
    analyzer = ChartAnalyzer(str(output_dir))

    print("Running chart analysis...")
    analyses = analyzer.analyze_all_charts(df)

    # Print summary
    print("\nAnalysis Complete!")
    print(f"Charts and analyses have been saved to: {output_dir}")
    print("\nAnalysis Summary:")
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} ===")
        # Print first 200 characters of each analysis as preview
        preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
        print(preview)


def main():
    print("Loading data from database...")
    connection_string = "sqlite:///simulations/simulation_results.db"
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
    analyzer = ChartAnalyzer()

    print("Running chart analysis...")
    analyses = analyzer.analyze_all_charts(actions_df, agents_df)

    print("\nAnalysis Complete!")
    print("Charts and analyses have been saved to: chart_analysis")

    print("\nAnalysis Summary:\n")
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} Analysis ===")
        print(analysis)


if __name__ == "__main__":
    main()
