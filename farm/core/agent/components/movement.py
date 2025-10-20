"""
Movement component for agent spatial positioning and movement.

This module provides the MovementComponent class that manages agent position,
validates movement operations, handles spatial queries, and integrates with
validation and spatial services. It enforces movement constraints and maintains
position consistency across the simulation.
"""

import math
from typing import Tuple

from farm.core.agent.config.component_configs import MovementConfig
from farm.core.agent.services import AgentServices
from farm.utils.logging import get_logger

from .base import AgentComponent

logger = get_logger(__name__)


class MovementComponent(AgentComponent):
    """
    Manages agent position and movement within the simulation.

    This component handles all aspects of agent spatial positioning including:
    - Tracking and updating agent position coordinates
    - Validating movement operations against distance constraints
    - Integrating with validation services for position validation
    - Updating spatial indices when position changes
    - Providing spatial query capabilities for nearby positions
    - Enforcing movement limits defined in configuration

    The component maintains position consistency and handles error recovery
    when external services (validation, spatial) encounter issues.
    """

    def __init__(self, services: AgentServices, config: MovementConfig):
        """
        Initialize movement component with services and configuration.

        Args:
            services: AgentServices container providing access to validation,
                spatial, logging, and other required services
            config: MovementConfig containing max_movement distance limit and
                perception_radius for spatial queries
        """
        super().__init__(services, "MovementComponent")
        self.config = config
        self.position = (0.0, 0.0)

    def attach(self, core) -> None:
        """Attach component to agent core.

        Args:
            core: AgentCore instance to attach to
        """
        super().attach(core)

    def on_step_start(self) -> None:
        """Called at the start of each simulation step.

        Currently no-op, but available for future step initialization logic.
        """
        pass

    def on_step_end(self) -> None:
        """Called at the end of each simulation step.

        Currently no-op, but available for future step cleanup logic.
        """
        pass

    def on_terminate(self) -> None:
        """Called when the agent terminates or is removed from simulation.

        Currently no-op, but available for cleanup when agent lifecycle ends.
        """
        pass

    def set_position(self, position: Tuple[float, float]) -> bool:
        """
        Set agent position with comprehensive validation.

        Validates position format, coordinates, and integrates with validation
        service if available. Updates spatial indices when position changes.

        Args:
            position: New (x, y) position as tuple of two numbers

        Returns:
            bool: True if position was valid and updated, False if validation
                service rejected the position

        Raises:
            ValueError: If position format is invalid (wrong type, length, or
                contains NaN/infinity values)
        """
        # Input validation
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            logger.debug(f"[{self.name}] Invalid position format: {position}")
            raise ValueError(f"Position must be a tuple/list of 2 numbers, got: {type(position)}")

        x, y = position
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            logger.debug(f"[{self.name}] Position coordinates must be numbers: {position}")
            raise ValueError(f"Position coordinates must be numbers, got: ({type(x)}, {type(y)})")

        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            logger.debug(f"[{self.name}] Position contains invalid values (NaN/inf): {position}")
            raise ValueError(f"Position coordinates cannot be NaN or infinity: {position}")

        # Validate position if validation service available
        if self.validation_service:
            try:
                if not self.validation_service.is_valid_position(position):
                    logger.debug(f"[{self.name}] Position validation failed: {position}")
                    return False
            except Exception as e:
                logger.warning(f"[{self.name}] Position validation service error: {e}", exc_info=True)
                # Continue without validation if service fails
                pass

        # Update position if changed
        if self.position != position:
            old_position = self.position
            self.position = position
            logger.debug(f"[{self.name}] Position updated from {old_position} to {position}")

            # Mark spatial structures as dirty
            if self.spatial_service:
                try:
                    self.spatial_service.mark_positions_dirty()
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to mark spatial positions dirty: {e}", exc_info=True)
                    # Don't fail the position update if spatial service fails
                    pass

        return True

    def move_to(self, position: Tuple[float, float]) -> bool:
        """
        Move agent to target position with distance constraint validation.

        Calculates Euclidean distance from current position to target and
        validates against max_movement limit. If valid, calls set_position
        to update position with full validation.

        Args:
            position: Target (x, y) position as tuple of two numbers

        Returns:
            bool: True if move was within distance limit and position was
                successfully updated, False if distance exceeded or position
                validation failed

        Raises:
            ValueError: If position format is invalid (propagated from
                set_position validation)
        """
        try:
            # Check distance
            dx = position[0] - self.position[0]
            dy = position[1] - self.position[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > self.config.max_movement:
                logger.debug(f"[{self.name}] Movement distance {distance:.2f} exceeds max {self.config.max_movement}")
                return False

            return self.set_position(position)

        except (TypeError, IndexError) as e:
            logger.debug(f"[{self.name}] Invalid position format for movement: {position}, error: {e}")
            raise ValueError(f"Invalid position format: {position}") from e
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error during movement: {e}", exc_info=True)
            raise

    def get_nearby_positions(self, radius: int) -> list[Tuple[float, float]]:
        """
        Get positions of entities within specified radius of agent.

        Queries spatial service for nearby entities and extracts their positions.
        Handles both dictionary and list result formats from spatial service.
        Returns empty list if spatial service unavailable or encounters errors.

        Args:
            radius: Search radius in simulation units (must be non-negative)

        Returns:
            List of (x, y) position tuples for entities within radius.
            Only includes entities with valid position attributes.

        Raises:
            ValueError: If radius is negative or not a number
        """
        if not isinstance(radius, (int, float)):
            logger.debug(f"[{self.name}] Invalid radius type: {type(radius)}")
            raise ValueError(f"Radius must be a number, got: {type(radius)}")

        if radius < 0:
            logger.debug(f"[{self.name}] Negative radius not allowed: {radius}")
            raise ValueError(f"Radius must be non-negative, got: {radius}")

        if not self.spatial_service:
            logger.debug(f"[{self.name}] No spatial service available for nearby position query")
            return []

        try:
            nearby = self.spatial_service.get_nearby(self.position, radius, [])
            if isinstance(nearby, dict):
                positions = []
                for items in nearby.values():
                    if isinstance(items, list):
                        positions.extend(
                            [item.position for item in items if hasattr(item, "position") and item.position is not None]
                        )
                return positions
            elif isinstance(nearby, list):
                return [item.position for item in nearby if hasattr(item, "position") and item.position is not None]
            else:
                logger.warning(f"[{self.name}] Unexpected nearby result type: {type(nearby)}")
                return []

        except Exception as e:
            logger.warning(f"[{self.name}] Error querying nearby positions: {e}", exc_info=True)
            return []

    @property
    def x(self) -> float:
        """Get X coordinate of current position."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Get Y coordinate of current position."""
        return self.position[1]

    @property
    def perception_radius(self) -> int:
        """Get perception radius from configuration for spatial queries."""
        return self.config.perception_radius
