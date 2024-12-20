from typing import Any, Dict, List, Tuple

from database.analyzers.analysis_utils import _calculate_correlation


class SpatialAnalyzer:
    """Analyzes spatial patterns and behaviors of agents.

    This class provides static methods for analyzing various aspects of agent spatial behavior,
    including movement patterns, clustering, and position-based performance analysis.

    The analyzer works with action-state pairs that contain position information and can
    process both individual and group spatial behaviors.
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
                - clustering: Dict containing clustering metrics:
                    - average_density: Mean number of actions per location
                    - location_frequencies: Normalized visit counts per location
                - movement_patterns: Dict containing movement analysis:
                    - common_paths: List of (path, frequency) tuples
                    - average_distance: Mean distance traveled per movement

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

        # Analyze position effects
        position_effects = {
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

        # Analyze clustering patterns
        clustering = SpatialAnalyzer._analyze_clustering(location_actions)

        # Analyze movement patterns
        movement_patterns = SpatialAnalyzer._analyze_movement_patterns(results)

        return {
            "position_effects": position_effects,
            "clustering": clustering,
            "movement_patterns": movement_patterns,
        }

    @staticmethod
    def _analyze_clustering(
        location_actions: Dict[Any, List[Tuple[Any, Any]]]
    ) -> Dict[str, float]:
        """Analyze clustering of agents based on location.

        Examines how agents cluster in different locations by analyzing the density
        and frequency of actions at each position. This helps identify popular areas,
        potential bottlenecks, or underutilized spaces.

        Args:
            location_actions: Dict mapping locations to lists of action-state pairs,
                where each location is a position in the environment and the list
                contains all actions taken at that position.

        Returns:
            Dict[str, float]: Clustering metrics including:
                - average_density: Mean number of actions per location (float)
                - location_frequencies: Dict mapping locations to their normalized
                  visit frequencies (0.0 to 1.0)

        Note:
            - Density is calculated as the raw count of actions at each location
            - Frequencies are normalized to sum to 1.0 across all locations
        """
        cluster_density = {}
        for location, actions in location_actions.items():
            cluster_density[location] = len(actions)

        total_actions = sum(cluster_density.values())
        return {
            "average_density": (
                sum(cluster_density.values()) / len(cluster_density)
                if cluster_density
                else 0.0
            ),
            "location_frequencies": {
                location: count / total_actions if total_actions > 0 else 0.0
                for location, count in cluster_density.items()
            },
        }

    @staticmethod
    def _analyze_movement_patterns(results: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Analyze movement patterns of agents.

        Examines how agents move through the environment by tracking paths taken
        and calculating movement statistics. This analysis helps identify common
        routes, movement efficiency, and spatial preferences.

        Args:
            results: List of action-state pairs in chronological order, where each pair
                contains:
                - action: Agent action data
                - state: Environment state with position information

        Returns:
            Dict[str, Any]: Movement patterns including:
                - common_paths: List of ((start_pos, end_pos), frequency) tuples,
                  sorted by frequency in descending order
                - average_distance: Mean Euclidean distance traveled per movement

        Note:
            - Paths are represented as tuples of (start_position, end_position)
            - Distance is calculated using Euclidean distance between positions
            - Only consecutive positions with valid position data are analyzed
        """
        movements = {}
        distances = []

        for i in range(1, len(results)):
            prev_action, prev_state = results[i - 1]
            curr_action, curr_state = results[i]

            if hasattr(prev_state, "position") and hasattr(curr_state, "position"):
                prev_pos = prev_state.position
                curr_pos = curr_state.position

                path = (prev_pos, curr_pos)
                if path not in movements:
                    movements[path] = 0
                movements[path] += 1

                # Calculate Euclidean distance
                distance = sum((a - b) ** 2 for a, b in zip(prev_pos, curr_pos)) ** 0.5
                distances.append(distance)

        common_paths = sorted(movements.items(), key=lambda x: x[1], reverse=True)
        average_distance = sum(distances) / len(distances) if distances else 0.0

        return {
            "common_paths": common_paths,
            "average_distance": average_distance,
        }
