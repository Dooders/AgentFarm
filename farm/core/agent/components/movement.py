"""
Movement component for agent locomotion.

Handles agent movement in the environment, including position validation,
distance constraints, and spatial service integration.
"""

import math
from typing import TYPE_CHECKING, Optional, Tuple
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import MovementConfig

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class MovementComponent(IAgentComponent):
    """
    Component handling agent movement in 2D/3D space.

    Responsibilities:
    - Move agent to target positions
    - Validate movement distances
    - Apply discretization rules
    - Integrate with spatial indexing

    Single Responsibility: Only movement logic.
    Open-Closed: Extend for new movement types without modification.
    """

    def __init__(self, config: MovementConfig):
        """
        Initialize movement component.

        Args:
            config: Movement configuration
        """
        self._config = config
        self._agent: Optional["AgentCore"] = None

    @property
    def name(self) -> str:
        """Component identifier."""
        return "movement"

    @property
    def max_movement(self) -> float:
        """Maximum distance agent can move per turn."""
        return self._config.max_movement

    def move_to(self, target_position: Tuple[float, float]) -> bool:
        """
        Move agent towards target position.

        Movement is constrained by max_movement distance. If target is
        farther than max_movement, agent moves as far as possible toward it.

        Args:
            target_position: Desired (x, y) position

        Returns:
            bool: True if move was successful, False otherwise

        Example:
            >>> movement.move_to((100, 100))  # Move toward (100, 100)
            True
        """
        if self._agent is None:
            return False

        current_pos = self._agent.state_manager.position

        # Calculate direction and distance to target
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return True  # Already at target

        # Limit movement to max_movement
        if distance > self._config.max_movement:
            scale = self._config.max_movement / distance
            dx *= scale
            dy *= scale

        # Calculate new position
        new_position = (
            current_pos[0] + dx,
            current_pos[1] + dy,
        )

        # Update position through state manager
        self._agent.state_manager.set_position(new_position)
        return True

    def move_by(self, delta_x: float, delta_y: float) -> bool:
        """
        Move agent by relative offset.

        Movement is constrained by max_movement distance. If delta is
        larger than max_movement, it will be scaled down.

        Args:
            delta_x: X offset to move
            delta_y: Y offset to move

        Returns:
            bool: True if move was successful, False otherwise

        Example:
            >>> movement.move_by(5, 10)  # Move 5 right, 10 up
            True
        """
        if self._agent is None:
            return False

        # Check if delta exceeds max movement
        distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if distance > self._config.max_movement:
            scale = self._config.max_movement / distance
            delta_x *= scale
            delta_y *= scale

        current_pos = self._agent.state_manager.position
        new_position = (
            current_pos[0] + delta_x,
            current_pos[1] + delta_y,
        )

        self._agent.state_manager.set_position(new_position)
        return True

    def random_move(self, distance: Optional[float] = None) -> bool:
        """
        Move in a random direction.

        Args:
            distance: Distance to move (uses max_movement if None)

        Returns:
            bool: True if move was successful, False otherwise

        Example:
            >>> movement.random_move()  # Random move up to max_movement
            True
            >>> movement.random_move(5.0)  # Random move exactly 5 units
            True
        """
        if self._agent is None:
            return False

        import random

        # Use provided distance or random up to max
        if distance is None:
            distance = random.uniform(0, self._config.max_movement)
        else:
            distance = min(distance, self._config.max_movement)

        # Random angle
        angle = random.uniform(0, 2 * math.pi)

        # Calculate offset
        delta_x = distance * math.cos(angle)
        delta_y = distance * math.sin(angle)

        return self.move_by(delta_x, delta_y)

    def move_toward_entity(self, target_position: Tuple[float, float], stop_distance: float = 0.0) -> bool:
        """
        Move toward an entity, stopping at specified distance.

        Useful for approaching resources, other agents, etc. while maintaining
        a minimum distance.

        Args:
            target_position: Position to move toward
            stop_distance: Minimum distance to maintain (default: 0)

        Returns:
            bool: True if move was successful, False otherwise

        Example:
            >>> # Move toward resource but stop 1 unit away
            >>> movement.move_toward_entity(resource.position, stop_distance=1.0)
            True
        """
        if self._agent is None:
            return False

        current_pos = self._agent.state_manager.position

        # Calculate distance to target
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Already within stop distance
        if distance <= stop_distance:
            return True

        # Calculate how far to move (distance to target minus stop distance)
        move_distance = distance - stop_distance
        move_distance = min(move_distance, self._config.max_movement)

        # Calculate movement vector
        if distance > 0:
            scale = move_distance / distance
            delta_x = dx * scale
            delta_y = dy * scale
        else:
            delta_x, delta_y = 0.0, 0.0

        return self.move_by(delta_x, delta_y)

    def can_reach(self, target_position: Tuple[float, float]) -> bool:
        """
        Check if agent can reach target in one move.

        Args:
            target_position: Position to check

        Returns:
            bool: True if target is within max_movement distance

        Example:
            >>> if movement.can_reach(resource.position):
            ...     movement.move_to(resource.position)
        """
        if self._agent is None:
            return False

        current_pos = self._agent.state_manager.position
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= self._config.max_movement

    def distance_to(self, target_position: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance to target position.

        Args:
            target_position: Position to measure distance to

        Returns:
            float: Distance to target

        Example:
            >>> dist = movement.distance_to(resource.position)
            >>> print(f"Resource is {dist} units away")
        """
        if self._agent is None:
            return float('inf')

        current_pos = self._agent.state_manager.position
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_state(self) -> dict:
        """
        Get serializable state.

        Returns:
            dict: Component state (movement has no state to persist)
        """
        return {
            "max_movement": self._config.max_movement,
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary

        Note:
            Movement component is mostly stateless, configuration is
            immutable, so this is primarily for consistency.
        """
        # Movement component has no mutable state to load
        pass