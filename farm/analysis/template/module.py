"""
Template analysis module implementation.

This is a template for creating new analysis modules using the modern
protocol-based architecture.

Usage:
    1. Copy this file to your new module directory
    2. Replace 'template' with your analysis type
    3. Implement the required methods
    4. Register your module via environment variable or programmatically
"""

from typing import Callable, Optional
import pandas as pd

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
from farm.analysis.common.context import AnalysisContext

# Import your analysis module's functions
# from farm.analysis.template.analyze import process_template_data
# from farm.analysis.template.plot import (
#     plot_template_distribution,
#     plot_template_over_time,
# )


class TemplateModule(BaseAnalysisModule):
    """
    Template module for analyzing [your analysis type] in simulations.
    
    Replace this description with details about what your module analyzes.
    
    Example:
        Module for analyzing resource efficiency patterns across simulations.
    """

    def __init__(self):
        """Initialize the template analysis module."""
        super().__init__(
            name="template",  # Replace with your module name
            description="Template analysis module - replace this description"
        )
        
        # Optional: Set up validation
        # validator = CompositeValidator([
        #     ColumnValidator(
        #         required_columns=['iteration', 'agent_type'],
        #         column_types={'iteration': int}
        #     ),
        #     DataQualityValidator(min_rows=1)
        # ])
        # self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all analysis functions for this module.
        
        Note: Use make_analysis_function() to wrap functions to ensure
        they match the standard (df, ctx, **kwargs) signature.
        """
        # Example function registration:
        # self._functions = {
        #     "plot_distribution": make_analysis_function(plot_template_distribution),
        #     "plot_over_time": make_analysis_function(plot_template_over_time),
        # }
        
        self._functions = {}
        
        # Define function groups for easier access
        self._groups = {
            "all": list(self._functions.values()),
            # Example groups:
            # "plots": [
            #     self._functions["plot_distribution"],
            #     self._functions["plot_over_time"],
            # ],
            # "metrics": [
            #     self._functions["compute_stats"],
            # ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get the data processor for this module.
        
        The processor transforms raw experiment data into a DataFrame
        ready for analysis.
        
        Returns:
            SimpleDataProcessor wrapping your processing function
        """
        def process_template_data(experiment_path, **kwargs) -> pd.DataFrame:
            """Process raw data into analysis-ready format.
            
            Args:
                experiment_path: Path to experiment directory
                **kwargs: Additional processing parameters
                
            Returns:
                Processed DataFrame
            """
            # Implement your data processing logic here
            # Example:
            # data = []
            # for sim_dir in experiment_path.glob("iteration_*"):
            #     sim_data = load_simulation_data(sim_dir)
            #     data.append(sim_data)
            # return pd.DataFrame(data)
            
            raise NotImplementedError("Implement your data processor")
        
        return SimpleDataProcessor(process_template_data)
    
    # Optional: Database support
    # def supports_database(self) -> bool:
    #     return True
    # 
    # def get_db_filename(self) -> str:
    #     return "template.db"
    # 
    # def get_db_loader(self) -> Optional[Callable]:
    #     from farm.analysis.template.query_db import load_data_from_db
    #     return load_data_from_db


# Example analysis function
def example_analysis_function(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Example analysis function using the standard signature.
    
    Args:
        df: Input DataFrame with processed data
        ctx: Analysis context with output_path, logger, config, etc.
        **kwargs: Additional function-specific parameters
    """
    ctx.logger.info("Running example analysis")
    
    # Your analysis logic here
    results = df.describe()
    
    # Save outputs using context
    output_file = ctx.get_output_file("results.csv")
    results.to_csv(output_file)
    
    # Report progress
    ctx.report_progress("Analysis complete", 1.0)


# Create singleton instance (uncomment when ready to use)
# template_module = TemplateModule()

# Register via environment variable:
# export FARM_ANALYSIS_MODULES="farm.analysis.template.module.template_module"

# Or register programmatically:
# from farm.analysis.registry import registry
# registry.register(template_module)
