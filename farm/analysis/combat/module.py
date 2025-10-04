"""
Combat analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.combat.data import process_combat_data
from farm.analysis.combat.analyze import (
    analyze_combat_overview,
    analyze_agent_combat_performance,
    analyze_combat_efficiency,
    analyze_combat_temporal_patterns,
)
from farm.analysis.combat.plot import (
    plot_combat_overview,
    plot_combat_success_rate,
    plot_agent_combat_performance,
    plot_combat_efficiency,
    plot_damage_distribution,
    plot_combat_temporal_patterns,
)


class CombatModule(BaseAnalysisModule):
    """Module for analyzing combat patterns in simulations."""

    def __init__(self):
        super().__init__(
            name="combat",
            description="Analysis of combat encounters, attack success rates, damage patterns, and agent combat performance"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=[],  # Flexible since we handle multiple data types
                column_types={}
            ),
            DataQualityValidator(min_rows=0)  # Allow empty data (no combat)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all combat analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_overview": make_analysis_function(analyze_combat_overview),
            "analyze_performance": make_analysis_function(analyze_agent_combat_performance),
            "analyze_efficiency": make_analysis_function(analyze_combat_efficiency),
            "analyze_temporal": make_analysis_function(analyze_combat_temporal_patterns),
            "plot_overview": make_analysis_function(plot_combat_overview),
            "plot_success_rate": make_analysis_function(plot_combat_success_rate),
            "plot_performance": make_analysis_function(plot_agent_combat_performance),
            "plot_efficiency": make_analysis_function(plot_combat_efficiency),
            "plot_damage": make_analysis_function(plot_damage_distribution),
            "plot_temporal": make_analysis_function(plot_combat_temporal_patterns),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_overview"],
                self._functions["analyze_performance"],
                self._functions["analyze_efficiency"],
                self._functions["analyze_temporal"],
            ],
            "plots": [
                self._functions["plot_overview"],
                self._functions["plot_success_rate"],
                self._functions["plot_performance"],
                self._functions["plot_efficiency"],
                self._functions["plot_damage"],
                self._functions["plot_temporal"],
            ],
            "performance": [
                self._functions["analyze_performance"],
                self._functions["plot_performance"],
                self._functions["plot_efficiency"],
            ],
            "temporal": [
                self._functions["analyze_temporal"],
                self._functions["plot_temporal"],
            ],
            "basic": [
                self._functions["analyze_overview"],
                self._functions["plot_overview"],
                self._functions["plot_success_rate"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for combat analysis."""
        return SimpleDataProcessor(process_combat_data)

    def supports_database(self) -> bool:
        """Whether this module uses database storage."""
        return True

    def get_db_filename(self) -> str:
        """Get database filename if using database."""
        return "simulation.db"


# Create singleton instance
combat_module = CombatModule()
