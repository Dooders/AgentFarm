import logging
import os

# Import analysis configuration
from analysis_config import (
    find_latest_experiment_path,
    save_analysis_data,
    setup_analysis_directory,
)

from farm.analysis.dominance.analyze import process_dominance_data
from farm.analysis.dominance.ml import run_dominance_classification
from farm.analysis.dominance.plot import (
    plot_comprehensive_score_breakdown,
    plot_correlation_matrix,
    plot_dominance_comparison,
    plot_dominance_distribution,
    plot_dominance_stability,
    plot_dominance_switches,
    plot_reproduction_advantage_vs_stability,
    plot_reproduction_success_vs_switching,
    plot_reproduction_vs_dominance,
    plot_resource_proximity_vs_dominance,
)


def main():
    # Set up the dominance analysis directory
    dominance_output_path, log_file = setup_analysis_directory("dominance")

    logging.info(f"Saving results to {dominance_output_path}")

    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return

    logging.info(f"Analyzing simulations in {experiment_path}...")

    # Process data and directly insert it into the database in the dominance output path
    db_filename = "dominance.db"
    db_path = os.path.join(dominance_output_path, db_filename)
    db_uri = f"sqlite:///{db_path}"
    logging.info(f"Processing data and inserting directly into {db_uri}")
    df = process_dominance_data(experiment_path, save_to_db=True, db_path=db_uri)

    # For visualization and plotting, we still need the DataFrame
    # We'll retrieve it if it's not returned (save_to_db=True returns None)
    if df is None:
        from farm.analysis.dominance.query_dominance_db import load_data_from_db

        logging.info("Loading data from database for visualization...")
        df = load_data_from_db(db_uri)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Continue with visualization but skip saving the raw data
    # since it's already in the database
    logging.info(f"Analyzed {len(df)} simulations.")
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

    # Calculate and display dominance distributions as percentages
    logging.info("\nDominance distribution (percentages):")

    #! Need a better way to execute analysis types, cleaner and easier to execute
    # Plot dominance distribution
    plot_dominance_distribution(df, dominance_output_path)

    # Generate comprehensive score breakdown chart
    plot_comprehensive_score_breakdown(df, dominance_output_path)

    # Plot dominance switching patterns
    plot_dominance_switches(df, dominance_output_path)

    # Plot dominance stability
    plot_dominance_stability(df, dominance_output_path)

    # Plot reproduction success vs switching
    plot_reproduction_success_vs_switching(df, dominance_output_path)

    # Plot reproduction advantage vs stability
    plot_reproduction_advantage_vs_stability(df, dominance_output_path)

    # Plot resource proximity vs dominance
    plot_resource_proximity_vs_dominance(df, dominance_output_path)

    # Plot reproduction metrics vs dominance
    plot_reproduction_vs_dominance(df, dominance_output_path)

    # Plot correlation matrices
    for label in ["population_dominance", "survival_dominance"]:
        if label in df.columns and df[label].nunique() > 1:
            plot_correlation_matrix(df, label, dominance_output_path)

    # Plot comparison of different dominance measures
    plot_dominance_comparison(df, dominance_output_path)

    # Run ML classification analysis
    run_dominance_classification(df, dominance_output_path)

    logging.info("\nAnalysis complete. Results saved to database and PNG files.")
    logging.info(f"Database saved to: {db_path}")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {dominance_output_path}")


if __name__ == "__main__":
    main()
