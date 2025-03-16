import logging
import os

# Import analysis configuration
from analysis_config import run_analysis

"""
Template for creating a new analysis module.
Replace 'template' with your analysis type (e.g., 'reproduction', 'resource', etc.)
"""

# Import your analysis module's functions
# from farm.analysis.template.analyze import process_template_data
# from farm.analysis.template.query_template_db import load_data_from_db
# from farm.analysis.template.plot import (
#     plot_template_distribution,
#     plot_template_over_time,
#     # Add other plotting functions here
# )


def main():
    """
    Run template analysis using the module system.
    Replace 'template' with your analysis type (e.g., 'reproduction', 'resource', etc.)
    """
    # Run the analysis using the generic function with the module system
    output_path, df = run_analysis(
        analysis_type="template",  # Replace with your module name
        function_group="all",  # Use all analysis functions
        # Alternatively, you can specify a specific group:
        # function_group="basic",  # Only basic analysis
        # function_group="advanced",  # Only advanced analysis
    )

    # The module system handles all the details:
    # - Setting up the output directory
    # - Finding the experiment path
    # - Processing the data
    # - Running the analysis functions
    # - Saving the results

    if df is not None and not df.empty:
        logging.info(f"Analysis complete. Processed {len(df)} simulations.")

        # Add any additional post-processing specific to your analysis
        # For example:
        # if "important_column" in df.columns:
        #     special_analysis_function(df, output_path)


if __name__ == "__main__":
    main()
