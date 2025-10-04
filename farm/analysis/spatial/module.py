"""
Spatial analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.spatial.data import process_spatial_data
from farm.analysis.spatial.analyze import (
    analyze_spatial_overview,
    analyze_movement_patterns,
    analyze_location_hotspots,
    analyze_spatial_distribution,
)
from farm.analysis.spatial.plot import (
    plot_spatial_overview,
    plot_movement_trajectories,
    plot_location_hotspots,
    plot_spatial_density,
    plot_movement_directions,
    plot_clustering_analysis,
)


class SpatialModule(BaseAnalysisModule):
    """Module for analyzing spatial patterns in simulations."""

    def __init__(self):
        super().__init__(
            name="spatial",
            description="Analysis of spatial patterns, movement trajectories, location effects, and clustering"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=[],  # Flexible since we handle multiple data types
                column_types={}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all spatial analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_overview": make_analysis_function(analyze_spatial_overview),
            "analyze_movement": make_analysis_function(analyze_movement_patterns),
            "analyze_hotspots": make_analysis_function(analyze_location_hotspots),
            "analyze_distribution": make_analysis_function(analyze_spatial_distribution),
            "plot_overview": make_analysis_function(plot_spatial_overview),
            "plot_trajectories": make_analysis_function(plot_movement_trajectories),
            "plot_hotspots": make_analysis_function(plot_location_hotspots),
            "plot_density": make_analysis_function(plot_spatial_density),
            "plot_directions": make_analysis_function(plot_movement_directions),
            "plot_clustering": make_analysis_function(plot_clustering_analysis),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_overview"],
                self._functions["analyze_movement"],
                self._functions["analyze_hotspots"],
                self._functions["analyze_distribution"],
            ],
            "plots": [
                self._functions["plot_overview"],
                self._functions["plot_trajectories"],
                self._functions["plot_hotspots"],
                self._functions["plot_density"],
                self._functions["plot_directions"],
                self._functions["plot_clustering"],
            ],
            "movement": [
                self._functions["analyze_movement"],
                self._functions["plot_trajectories"],
                self._functions["plot_directions"],
            ],
            "location": [
                self._functions["analyze_hotspots"],
                self._functions["analyze_distribution"],
                self._functions["plot_hotspots"],
                self._functions["plot_density"],
            ],
            "basic": [
                self._functions["analyze_overview"],
                self._functions["plot_overview"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for spatial analysis."""
        return SimpleDataProcessor(process_spatial_data)

    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename if using database."""
        return "simulation.db"


# Create singleton instance
spatial_module = SpatialModule()
