from analysis_config import run_analysis

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
    # Import here to avoid circular imports
    from farm.analysis.dominance.query_dominance_db import load_data_from_db

    # Define all analysis functions to run
    analysis_functions = [
        plot_dominance_distribution,
        plot_comprehensive_score_breakdown,
        plot_dominance_switches,
        plot_dominance_stability,
        plot_reproduction_success_vs_switching,
        plot_reproduction_advantage_vs_stability,
        plot_resource_proximity_vs_dominance,
        plot_reproduction_vs_dominance,
        plot_dominance_comparison,
        run_dominance_classification,
    ]

    # Define any special keyword arguments for specific functions
    analysis_kwargs = {}

    # Run the analysis using the generic function
    output_path, df = run_analysis(
        analysis_type="dominance",
        data_processor=process_dominance_data,
        analysis_functions=analysis_functions,
        db_filename="dominance.db",
        load_data_function=load_data_from_db,
        processor_kwargs={},
        analysis_kwargs=analysis_kwargs,
    )

    # Add any additional post-processing specific to dominance analysis
    if df is not None and not df.empty:
        # Plot correlation matrices for specific columns
        for label in ["population_dominance", "survival_dominance"]:
            if label in df.columns and df[label].nunique() > 1:
                plot_correlation_matrix(df, output_path, label=label)


if __name__ == "__main__":
    main()
