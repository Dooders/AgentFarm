"""
Agents analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.agents.data import process_agent_data
from farm.analysis.agents.analyze import (
    analyze_lifespan_patterns,
    analyze_behavior_clustering,
    analyze_performance_analysis,
    analyze_learning_curves,
    analyze_agent_statistics,
)
from farm.analysis.agents.plot import (
    plot_lifespan_distributions,
    plot_behavior_clusters,
    plot_performance_metrics,
    plot_learning_curves,
)
from farm.analysis.agents.lifespan import (
    analyze_agent_lifespans,
    plot_lifespan_histogram,
)
from farm.analysis.agents.behavior import (
    cluster_agent_behaviors,
    plot_behavior_clusters as plot_behavior_clusters_clustering,
)


class AgentsModule(BaseAnalysisModule):
    """Module for analyzing individual agents in simulations."""

    def __init__(self):
        super().__init__(
            name="agents",
            description="Analysis of individual agent behavior, lifespan, performance, and learning patterns"
        )

        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['agent_id'],
                column_types={'agent_id': str}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)

    def register_functions(self) -> None:
        """Register all agent analysis functions."""

        # Analysis functions
        self._functions = {
            "analyze_statistics": make_analysis_function(analyze_agent_statistics),
            "analyze_lifespans": make_analysis_function(analyze_lifespan_patterns),
            "analyze_behaviors": make_analysis_function(analyze_behavior_clustering),
            "analyze_performance": make_analysis_function(analyze_performance_analysis),
            "analyze_learning": make_analysis_function(analyze_learning_curves),
            "analyze_detailed_lifespans": make_analysis_function(analyze_agent_lifespans),
            "cluster_behaviors": make_analysis_function(cluster_agent_behaviors),
            "plot_lifespans": make_analysis_function(plot_lifespan_distributions),
            "plot_behaviors": make_analysis_function(plot_behavior_clusters),
            "plot_performance": make_analysis_function(plot_performance_metrics),
            "plot_learning": make_analysis_function(plot_learning_curves),
            "plot_lifespan_histogram": make_analysis_function(plot_lifespan_histogram),
            "plot_behavior_clustering": make_analysis_function(plot_behavior_clusters_clustering),
        }

        # Function groups
        self._groups = {
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_statistics"],
                self._functions["analyze_lifespans"],
                self._functions["analyze_behaviors"],
                self._functions["analyze_performance"],
                self._functions["analyze_learning"],
            ],
            "plots": [
                self._functions["plot_lifespans"],
                self._functions["plot_behaviors"],
                self._functions["plot_performance"],
                self._functions["plot_learning"],
            ],
            "lifespan": [
                self._functions["analyze_lifespans"],
                self._functions["analyze_detailed_lifespans"],
                self._functions["plot_lifespans"],
                self._functions["plot_lifespan_histogram"],
            ],
            "behavior": [
                self._functions["analyze_behaviors"],
                self._functions["cluster_behaviors"],
                self._functions["plot_behaviors"],
                self._functions["plot_behavior_clustering"],
            ],
            "basic": [
                self._functions["analyze_lifespans"],
                self._functions["plot_lifespans"],
            ],
        }

    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for agent analysis."""
        return SimpleDataProcessor(process_agent_data)


# Create singleton instance
agents_module = AgentsModule()
