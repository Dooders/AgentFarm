"""
Social Behavior Analysis Module

This module provides comprehensive analysis of social behaviors in agent simulations,
including cooperation, competition, social networks, and group dynamics.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.social_behavior.data import process_social_behavior_data
from farm.analysis.social_behavior.analyze import (
    analyze_social_behaviors,
)
from farm.analysis.social_behavior.plot import (
    plot_social_network_overview,
    plot_cooperation_competition_balance,
    plot_resource_sharing_patterns,
    plot_spatial_clustering,
)


class SocialBehaviorModule(BaseAnalysisModule):
    """Module for analyzing social behavior patterns in simulations."""

    def __init__(self):
        super().__init__(
            name="social_behavior",
            description="Analysis of social behaviors including cooperation, competition, networks, and group dynamics"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=[],  # Social behavior analysis works with raw database data
                column_types={}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all social behavior analysis functions."""

        # Analysis functions - simplified to use the main analysis function
        def analyze_comprehensive(df, ctx):
            """Run comprehensive social behavior analysis."""
            # Since social behavior analysis works with database directly,
            # we'll use the existing analyze_social_behaviors function
            # The data processor should handle getting the database session
            ctx.logger.info("Running comprehensive social behavior analysis...")
            # This would need to be adapted to work with the framework
            # For now, just create a placeholder
            pass

        # Analysis functions
        self._functions = {
            "analyze_comprehensive": make_analysis_function(analyze_comprehensive),
            "plot_network_overview": make_analysis_function(plot_social_network_overview),
            "plot_cooperation_balance": make_analysis_function(plot_cooperation_competition_balance),
            "plot_sharing_patterns": make_analysis_function(plot_resource_sharing_patterns),
            "plot_clustering": make_analysis_function(plot_spatial_clustering),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [self._functions["analyze_comprehensive"]],
            "plots": [
                self._functions["plot_network_overview"],
                self._functions["plot_cooperation_balance"],
                self._functions["plot_sharing_patterns"],
                self._functions["plot_clustering"],
            ],
            "basic": [
                self._functions["analyze_comprehensive"],
                self._functions["plot_cooperation_balance"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for social behavior analysis."""
        return SimpleDataProcessor(process_social_behavior_data)

    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return True  # Social behavior analysis requires direct database access

    def get_db_filename(self) -> str:
        """Get database filename if using database."""
        return "simulation.db"

    def get_db_loader(self):
        """Get database loader if using database."""
        return None  # Uses direct database access in data processor


# Create singleton instance
social_behavior_module = SocialBehaviorModule()
