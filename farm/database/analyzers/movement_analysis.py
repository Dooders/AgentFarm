from math import sqrt
from typing import Any, Dict, List, Tuple


class MovementAnalyzer:
    """Analyzes movement patterns and trajectories of agents.

    This class provides static methods for analyzing how agents move through their
    environment, including path analysis, distance calculations, and movement patterns.
    """

    @staticmethod
    def calculate_euclidean_distance(
        pos1: Tuple[float, ...], pos2: Tuple[float, ...]
    ) -> float:
        """Calculate the Euclidean distance between two positions.

        Args:
            pos1: First position coordinates as a tuple of floats
            pos2: Second position coordinates as a tuple of floats

        Returns:
            float: Euclidean distance between the positions

        Raises:
            ValueError: If positions have different dimensions
        """
        if len(pos1) != len(pos2):
            raise ValueError("Positions must have the same number of dimensions")

        return sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    @staticmethod
    def analyze_movement_patterns(results: List[Tuple[Any, Any]]) -> Dict[str, Any]:
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
                - total_distance: Sum of all distances traveled
                - path_lengths: Dict mapping paths to their distances
                - movement_directions: Distribution of movement directions
                - stationary_periods: List of periods with no movement

        Note:
            - Paths are represented as tuples of (start_position, end_position)
            - Only consecutive positions with valid position data are analyzed
            - Movement directions are binned into cardinal/intercardinal directions
        """
        movements = {}
        distances = []
        path_lengths = {}
        stationary_periods = []
        current_stationary = 0

        for i in range(1, len(results)):
            prev_action, prev_state = results[i - 1]
            curr_action, curr_state = results[i]

            if hasattr(prev_state, "position") and hasattr(curr_state, "position"):
                prev_pos = prev_state.position
                curr_pos = curr_state.position

                # Record path and frequency
                path = (prev_pos, curr_pos)
                if path not in movements:
                    movements[path] = 0
                    path_lengths[path] = MovementAnalyzer.calculate_euclidean_distance(
                        prev_pos, curr_pos
                    )
                movements[path] += 1

                # Calculate and record distance
                distance = path_lengths[path]
                distances.append(distance)

                # Track stationary periods
                if distance < 1e-6:  # Consider very small movements as stationary
                    current_stationary += 1
                elif current_stationary > 0:
                    stationary_periods.append(current_stationary)
                    current_stationary = 0

        # Add final stationary period if exists
        if current_stationary > 0:
            stationary_periods.append(current_stationary)

        # Calculate movement directions
        movement_directions = MovementAnalyzer._analyze_movement_directions(movements)

        common_paths = sorted(movements.items(), key=lambda x: x[1], reverse=True)
        total_distance = sum(distances)
        average_distance = total_distance / len(distances) if distances else 0.0

        return {
            "common_paths": common_paths,
            "average_distance": average_distance,
            "total_distance": total_distance,
            "path_lengths": path_lengths,
            "movement_directions": movement_directions,
            "stationary_periods": stationary_periods,
        }

    @staticmethod
    def _analyze_movement_directions(
        movements: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], int]
    ) -> Dict[str, float]:
        """Analyze the distribution of movement directions.

        Args:
            movements: Dict mapping paths to their frequencies

        Returns:
            Dict[str, float]: Distribution of movement directions as proportions
        """
        directions = {
            "north": 0,
            "northeast": 0,
            "east": 0,
            "southeast": 0,
            "south": 0,
            "southwest": 0,
            "west": 0,
            "northwest": 0,
        }
        total = 0

        for (start, end), freq in movements.items():
            if len(start) >= 2 and len(end) >= 2:  # Only analyze 2D+ movements
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                # Skip if no movement
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue

                # Determine direction based on angle
                angle = MovementAnalyzer._calculate_angle(dx, dy)
                direction = MovementAnalyzer._angle_to_direction(angle)
                directions[direction] += freq
                total += freq

        # Normalize to proportions
        return {
            direction: count / total if total > 0 else 0.0
            for direction, count in directions.items()
        }

    @staticmethod
    def _calculate_angle(dx: float, dy: float) -> float:
        """Calculate angle in degrees from delta x and y.

        Args:
            dx: Change in x coordinate
            dy: Change in y coordinate

        Returns:
            float: Angle in degrees (0-360)
        """
        from math import atan2, degrees

        angle = degrees(atan2(dy, dx))
        return (angle + 360) % 360

    @staticmethod
    def _angle_to_direction(angle: float) -> str:
        """Convert angle to cardinal/intercardinal direction.

        Args:
            angle: Angle in degrees (0-360)

        Returns:
            str: Direction name (e.g., "north", "northeast", etc.)
        """
        directions = [
            "east",
            "northeast",
            "north",
            "northwest",
            "west",
            "southwest",
            "south",
            "southeast",
        ]
        index = round(angle / 45) % 8
        return directions[index]
