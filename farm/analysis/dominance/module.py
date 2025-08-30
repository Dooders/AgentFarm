"""
Dominance analysis module implementation.
"""

from typing import Callable, Dict, List, Optional

from farm.analysis.base_module import AnalysisModule
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


class DominanceModule(AnalysisModule):
    """
    Module for analyzing agent dominance patterns in simulations.
    """

    def __init__(self):
        """Initialize the dominance analysis module."""
        super().__init__(
            name="dominance",
            description="Analysis of agent dominance patterns in simulations",
        )

    def register_analysis(self) -> None:
        """Register all analysis functions for the dominance module."""
        # Register all plot functions
        self._analysis_functions.update(
            {
                "plot_dominance_distribution": plot_dominance_distribution,
                "plot_comprehensive_score_breakdown": plot_comprehensive_score_breakdown,
                "plot_dominance_switches": plot_dominance_switches,
                "plot_dominance_stability": plot_dominance_stability,
                "plot_reproduction_success_vs_switching": plot_reproduction_success_vs_switching,
                "plot_reproduction_advantage_vs_stability": plot_reproduction_advantage_vs_stability,
                "plot_resource_proximity_vs_dominance": plot_resource_proximity_vs_dominance,
                "plot_reproduction_vs_dominance": plot_reproduction_vs_dominance,
                "plot_dominance_comparison": plot_dominance_comparison,
                "plot_correlation_matrix": lambda df, output_path: plot_correlation_matrix(
                    df, label_name="comprehensive_dominance", output_path=output_path
                ),
            }
        )

        # Register ML functions
        self._analysis_functions.update(
            {
                "run_dominance_classification": run_dominance_classification,
            }
        )

        # Define function groups for easier access
        self._analysis_groups = {
            "all": list(self._analysis_functions.values()),
            "plots": [
                plot_dominance_distribution,
                plot_comprehensive_score_breakdown,
                plot_dominance_switches,
                plot_dominance_stability,
                plot_reproduction_success_vs_switching,
                plot_reproduction_advantage_vs_stability,
                plot_resource_proximity_vs_dominance,
                plot_reproduction_vs_dominance,
                plot_dominance_comparison,
            ],
            "ml": [run_dominance_classification],
            "correlation": [plot_correlation_matrix],
            "basic": [
                plot_dominance_distribution,
                plot_comprehensive_score_breakdown,
                plot_dominance_comparison,
            ],
            "reproduction": [
                plot_reproduction_success_vs_switching,
                plot_reproduction_advantage_vs_stability,
                plot_reproduction_vs_dominance,
            ],
            "switching": [
                plot_dominance_switches,
                plot_dominance_stability,
            ],
        }

    def get_data_processor(self) -> Callable:
        """Get the data processor function for the dominance module."""
        return process_dominance_data

    def get_db_loader(self) -> Optional[Callable]:
        """Get the database loader function for the dominance module."""
        return load_data_from_db

    def get_db_filename(self) -> str:
        """Get the database filename for the dominance module."""
        return "dominance.db"


# Create a singleton instance
dominance_module = DominanceModule()
