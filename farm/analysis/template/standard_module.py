"""
Template for creating new analysis modules.

Copy this template to create a new analysis module.
Follow the existing pattern from dominance, genesis, etc.
"""

from typing import Optional, Callable
from pathlib import Path
import pandas as pd

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
from farm.analysis.common.context import AnalysisContext


# ============================================================================
# Data Processing
# ============================================================================

def process_MODULE_data(experiment_path: Path, **kwargs) -> pd.DataFrame:
    """Process raw experiment data for MODULE analysis.

    Args:
        experiment_path: Path to experiment directory
        **kwargs: Additional processing options

    Returns:
        Processed DataFrame ready for analysis
    """
    # TODO: Implement data processing
    # 1. Load data from experiment_path
    # 2. Transform and clean data
    # 3. Compute derived metrics
    # 4. Return DataFrame

    raise NotImplementedError("Implement data processing")


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_MODULE_metrics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze MODULE metrics and save results.

    Args:
        df: Processed data
        ctx: Analysis context with output_path, logger, etc.
        **kwargs: Additional analysis options
    """
    ctx.logger.info("Analyzing MODULE metrics...")

    # TODO: Implement analysis
    # 1. Calculate metrics
    # 2. Save results to ctx.get_output_file()
    # 3. Report progress via ctx.report_progress()

    raise NotImplementedError("Implement analysis")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_MODULE_overview(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create overview visualization of MODULE data.

    Args:
        df: Processed data
        ctx: Analysis context
        **kwargs: Plot options
    """
    import matplotlib.pyplot as plt

    ctx.logger.info("Creating MODULE overview plot...")

    # TODO: Implement visualization
    # 1. Create figure
    # 2. Plot data
    # 3. Save to ctx.get_output_file()

    raise NotImplementedError("Implement visualization")


# ============================================================================
# Module Definition
# ============================================================================

class MODULEModule(BaseAnalysisModule):
    """Module for analyzing MODULE in simulations."""

    def __init__(self):
        super().__init__(
            name="MODULE",
            description="Analysis of MODULE patterns in simulations"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['iteration'],  # TODO: Adjust columns
                column_types={'iteration': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all analysis functions."""
        self._functions = {
            "analyze_metrics": make_analysis_function(analyze_MODULE_metrics),
            "plot_overview": make_analysis_function(plot_MODULE_overview),
            # TODO: Add more functions
        }

        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [self._functions["analyze_metrics"]],
            "plots": [self._functions["plot_overview"]],
            # TODO: Add more groups
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for MODULE."""
        return SimpleDataProcessor(process_MODULE_data)

    # Optional: Database support
    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return False  # TODO: Set to True if using database

    def get_db_filename(self) -> Optional[str]:
        """Get database filename if using database."""
        return None  # TODO: Return filename if supports_database=True

    def get_db_loader(self) -> Optional[Callable]:
        """Get database loader if using database."""
        return None  # TODO: Return loader function if supports_database=True


# Create singleton instance
MODULE_module = MODULEModule()

