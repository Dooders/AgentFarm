from typing import Any, Dict, List, Tuple

from farm.database.analyzers.analysis_utils import _calculate_correlation


class LocationAnalyzer:
    """Analyzes location-specific patterns and behaviors of agents.

    This class provides static methods for analyzing how agents interact with and
    are influenced by specific locations in their environment, including clustering
    patterns and position-based performance analysis.
    """

    @staticmethod
    def analyze_position_effects(
        location_actions: Dict[Any, List[Tuple[Any, Any]]]
    ) -> Dict[Any, float]:
        """Analyze how different positions affect agent performance.

        Examines the correlation between resource levels and rewards at each location
        to understand how position influences agent success.

        Args:
            location_actions: Dict mapping locations to lists of action-state pairs,
                where each location is a position in the environment and the list
                contains all actions taken at that position.

        Returns:
            Dict[Any, float]: Position effects mapping each position to its correlation
                coefficient between resource levels and rewards. Higher values indicate
                stronger positive correlation between resources and rewards at that location.

        Note:
            - Correlation is calculated between resource levels and action rewards
            - Only actions with valid rewards and states with resource levels are considered
            - Positions with insufficient data return a correlation of 0.0
        """
        return {
            pos: _calculate_correlation(
                [
                    state.resource_level
                    for _, state in actions
                    if hasattr(state, "resource_level")
                ],
                [action.reward for action, _ in actions if action.reward is not None],
            )
            for pos, actions in location_actions.items()
        }

    @staticmethod
    def analyze_clustering(
        location_actions: Dict[Any, List[Tuple[Any, Any]]]
    ) -> Dict[str, Any]:
        """Analyze clustering of agents based on location.

        Examines how agents cluster in different locations by analyzing the density
        and frequency of actions at each position. This helps identify popular areas,
        potential bottlenecks, or underutilized spaces.

        Args:
            location_actions: Dict mapping locations to lists of action-state pairs,
                where each location is a position in the environment and the list
                contains all actions taken at that position.

        Returns:
            Dict[str, Any]: Clustering metrics including:
                - average_density: Mean number of actions per location (float)
                - location_frequencies: Dict mapping locations to their normalized
                  visit frequencies (0.0 to 1.0)
                - density_map: Dict mapping locations to their raw action counts
                - hotspots: List of (location, frequency) tuples for locations with
                  above-average density, sorted by frequency

        Note:
            - Density is calculated as the raw count of actions at each location
            - Frequencies are normalized to sum to 1.0 across all locations
            - Hotspots are locations with density above the mean
        """
        # Calculate raw density for each location
        density_map = {
            location: len(actions) for location, actions in location_actions.items()
        }

        total_actions = sum(density_map.values())
        avg_density = (
            sum(density_map.values()) / len(density_map) if density_map else 0.0
        )

        # Calculate normalized frequencies
        frequencies = {
            location: count / total_actions if total_actions > 0 else 0.0
            for location, count in density_map.items()
        }

        # Identify hotspots (locations with above-average density)
        hotspots = [
            (loc, freq)
            for loc, freq in frequencies.items()
            if density_map[loc] > avg_density
        ]
        hotspots.sort(key=lambda x: x[1], reverse=True)

        return {
            "average_density": avg_density,
            "location_frequencies": frequencies,
            "density_map": density_map,
            "hotspots": hotspots,
        }
