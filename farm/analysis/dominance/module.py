"""
Dominance analysis module implementation.
"""

from typing import Callable, Dict, List, Optional

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
from farm.analysis.dominance.pipeline import process_dominance_data
from farm.analysis.dominance.ml import run_dominance_classification
from farm.analysis.dominance.plot import (
    plot_comprehensive_score_breakdown,
    plot_correlation_matrix,
    plot_dominance_comparison,
    plot_dominance_distribution,
    plot_dominance_stability,
    plot_dominance_switches,
    plot_feature_importance,
    plot_reproduction_advantage_vs_stability,
    plot_reproduction_success_vs_switching,
    plot_reproduction_vs_dominance,
    plot_resource_proximity_vs_dominance,
)
from farm.analysis.dominance.query_dominance_db import load_data_from_db


class DominanceModule(BaseAnalysisModule):
    """
    Module for analyzing agent dominance patterns in simulations.
    """

    def __init__(self):
        """Initialize the dominance analysis module."""
        super().__init__(
            name="dominance",
            description="Analysis of agent dominance patterns in simulations",
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['iteration'],
                column_types={'iteration': int}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all analysis functions for the dominance module."""
        # Wrap plot functions to ensure they match protocol
        wrapped_correlation_matrix = make_analysis_function(
            lambda df, ctx, **kwargs: plot_correlation_matrix(
                df, label_name="comprehensive_dominance", output_path=str(ctx.output_path)
            ),
            name="plot_correlation_matrix"
        )

        # Register all functions with wrapping
        self._functions = {
            "plot_dominance_distribution": make_analysis_function(plot_dominance_distribution),
            "plot_comprehensive_score_breakdown": make_analysis_function(plot_comprehensive_score_breakdown),
            "plot_dominance_switches": make_analysis_function(plot_dominance_switches),
            "plot_dominance_stability": make_analysis_function(plot_dominance_stability),
            "plot_reproduction_success_vs_switching": make_analysis_function(plot_reproduction_success_vs_switching),
            "plot_reproduction_advantage_vs_stability": make_analysis_function(plot_reproduction_advantage_vs_stability),
            "plot_resource_proximity_vs_dominance": make_analysis_function(plot_resource_proximity_vs_dominance),
            "plot_reproduction_vs_dominance": make_analysis_function(plot_reproduction_vs_dominance),
            "plot_dominance_comparison": make_analysis_function(plot_dominance_comparison),
            "plot_correlation_matrix": wrapped_correlation_matrix,
            "run_dominance_classification": make_analysis_function(run_dominance_classification),
        }

        # Define function groups for easier access
        self._groups = {
            "all": list(self._functions.values()),
            "plots": [
                self._functions["plot_dominance_distribution"],
                self._functions["plot_comprehensive_score_breakdown"],
                self._functions["plot_dominance_switches"],
                self._functions["plot_dominance_stability"],
                self._functions["plot_reproduction_success_vs_switching"],
                self._functions["plot_reproduction_advantage_vs_stability"],
                self._functions["plot_resource_proximity_vs_dominance"],
                self._functions["plot_reproduction_vs_dominance"],
                self._functions["plot_dominance_comparison"],
            ],
            "ml": [self._functions["run_dominance_classification"]],
            "correlation": [self._functions["plot_correlation_matrix"]],
            "basic": [
                self._functions["plot_dominance_distribution"],
                self._functions["plot_comprehensive_score_breakdown"],
                self._functions["plot_dominance_comparison"],
            ],
            "reproduction": [
                self._functions["plot_reproduction_success_vs_switching"],
                self._functions["plot_reproduction_advantage_vs_stability"],
                self._functions["plot_reproduction_vs_dominance"],
            ],
            "switching": [
                self._functions["plot_dominance_switches"],
                self._functions["plot_dominance_stability"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get the data processor function for the dominance module."""
        return SimpleDataProcessor(process_dominance_data)

    def supports_database(self) -> bool:
        """This module uses a database for intermediate storage."""
        return True

    def get_db_loader(self) -> Optional[Callable]:
        """Get the database loader function for the dominance module."""
        return load_data_from_db

    def get_db_filename(self) -> str:
        """Get the database filename for the dominance module."""
        return "dominance.db"


# Create a singleton instance
dominance_module = DominanceModule()
