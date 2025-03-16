import logging

# Import analysis configuration
from analysis_config import run_analysis


def main():
    """
    Run dominance analysis using the module system.
    """
    # Run the analysis using the generic function with the module system
    output_path, df = run_analysis(
        analysis_type="dominance",
        function_group="all",  # Use all analysis functions
        # Alternatively, you can specify a specific group:
        # function_group="basic",  # Only basic analysis
        # function_group="reproduction",  # Only reproduction analysis
        # function_group="switching",  # Only switching analysis
        # function_group="ml",  # Only machine learning analysis
    )

    # The module system handles all the details:
    # - Setting up the output directory
    # - Finding the experiment path
    # - Processing the data
    # - Running the analysis functions
    # - Saving the results

    if df is not None and not df.empty:
        logging.info(
            f"Analysis complete. Processed {len(df)} simulations. Output saved to {output_path}"
        )


if __name__ == "__main__":
    main()
