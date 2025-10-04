from typing import Any, Dict, List, Tuple

from farm.database.analyzers.location_analysis import LocationAnalyzer
from farm.database.analyzers.movement_analysis import MovementAnalyzer


import warnings
class SpatialAnalyzer:
    """Analyzes spatial patterns and behaviors of agents.

    This class provides static methods for analyzing various aspects of agent spatial behavior,
    integrating location-specific analysis and movement patterns into a comprehensive
    spatial behavior analysis.
    """

    @staticmethod
    def analyze_spatial_patterns(results: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Analyze spatial patterns in agent behavior.

        Performs comprehensive spatial analysis by examining position effects,
        clustering patterns, and movement behaviors. This method aggregates results
        from multiple specialized analyses into a single cohesive view of spatial behavior.

        Args:
            results: List of action-state pairs to analyze, where each pair contains:
                - action: Agent action data with reward information
                - state: Environment state data with position information

        Returns:
            Dict[str, Any]: Spatial analysis results containing:
                - position_effects: Dict mapping positions to correlation coefficients
                  between resource levels and rewards at each location
                - clustering: Dict containing clustering metrics (see LocationAnalyzer)
                - movement_patterns: Dict containing movement analysis (see MovementAnalyzer)

        Example:
            >>> analyzer = SpatialAnalyzer()
            >>> results = [(action1, state1), (action2, state2)]
            >>> patterns = analyzer.analyze_spatial_patterns(results)
            >>> print(patterns['clustering']['average_density'])
            >>> print(patterns['movement_patterns']['average_distance'])
        """
        # Group actions by location
        location_actions = {}
        for action, state in results:
            if hasattr(state, "position"):
                pos = state.position
                if pos not in location_actions:
                    location_actions[pos] = []
                location_actions[pos].append((action, state))

        # Use specialized analyzers for different aspects
        position_effects = LocationAnalyzer.analyze_position_effects(location_actions)
        clustering = LocationAnalyzer.analyze_clustering(location_actions)
        movement_patterns = MovementAnalyzer.analyze_movement_patterns(results)

        return {
            "position_effects": position_effects,
            "clustering": clustering,
            "movement_patterns": movement_patterns,
        }
