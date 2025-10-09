"""
Perception component for agent environment observation.

Handles gathering information about nearby entities and environment state.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Fallback numpy-like interface for basic testing
    class np:
        int8 = int

        @staticmethod
        def zeros(shape, dtype=None):
            """Simple fallback zeros implementation."""
            if isinstance(shape, tuple) and len(shape) == 2:
                return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
            return []

        @staticmethod
        def all(arr):
            """Simple fallback all implementation."""
            if isinstance(arr, list):
                return all(all(row) if isinstance(row, list) else row for row in arr)
            return bool(arr)


from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import PerceptionConfig
from farm.utils.logging import get_logger

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore
    from farm.core.services.interfaces import ISpatialQueryService

logger = get_logger(__name__)


class PerceptionComponent(IAgentComponent):
    """
    Component handling agent perception and observation.

    Responsibilities:
    - Query spatial service for nearby entities
    - Create perception grids
    - Track visible agents and resources
    - Provide structured observation data

    Single Responsibility: Only perception logic.
    """

    def __init__(self, spatial_service: "ISpatialQueryService", config: PerceptionConfig):
        """
        Initialize perception component.

        Args:
            spatial_service: Service for querying nearby entities
            config: Perception configuration
        """
        self._spatial_service = spatial_service
        self._config = config
        self._agent: Optional["AgentCore"] = None

    @property
    def name(self) -> str:
        """Component identifier."""
        return "perception"

    @property
    def radius(self) -> int:
        """Perception radius in world units."""
        return self._config.perception_radius

    @property
    def grid_size(self) -> int:
        """Size of perception grid (2 * radius + 1)."""
        return self._config.perception_grid_size

    def get_nearby_entities(
        self, entity_types: Optional[List[str]] = None, radius: Optional[float] = None
    ) -> Dict[str, List[Any]]:
        """
        Get nearby entities from spatial service.

        Args:
            entity_types: Types to query (e.g., ["agents", "resources"])
                         If None, queries all types
            radius: Search radius (uses config radius if None)

        Returns:
            dict: Map of entity type to list of entities

        Example:
            >>> nearby = perception.get_nearby_entities(["resources"])
            >>> resources = nearby["resources"]
            >>> print(f"Found {len(resources)} nearby resources")
        """
        if self._agent is None:
            return {}

        query_radius = radius if radius is not None else self._config.perception_radius
        position = self._agent.state_manager.position

        try:
            return self._spatial_service.get_nearby(position=position, radius=query_radius, index_names=entity_types)
        except Exception:
            return {}

    def get_nearest_entity(self, entity_types: Optional[List[str]] = None) -> Dict[str, Optional[Any]]:
        """
        Get nearest entity of each type.

        Args:
            entity_types: Types to query (queries all if None)

        Returns:
            dict: Map of entity type to nearest entity (or None)

        Example:
            >>> nearest = perception.get_nearest_entity(["resources"])
            >>> if nearest["resources"]:
            ...     print(f"Nearest resource at {nearest['resources'].position}")
        """
        if self._agent is None:
            return {}

        position = self._agent.state_manager.position

        try:
            return self._spatial_service.get_nearest(position=position, index_names=entity_types)
        except Exception:
            return {}

    def create_perception_grid(self):
        """
        Create a grid representation of nearby environment.

        Grid encoding:
        - 0: Empty space
        - 1: Resource
        - 2: Other agent
        - 3: Boundary/obstacle

        Returns:
            np.ndarray or list: Perception grid of shape (grid_size, grid_size)

        Example:
            >>> grid = perception.create_perception_grid()
            >>> print(f"Grid shape: {grid.shape}")
            >>> print(f"Resources visible: {(grid == 1).sum()}")
        """
        if self._agent is None:
            size = self._config.perception_grid_size
            if HAS_NUMPY:
                return np.zeros((size, size), dtype=np.int8)
            else:
                return [[0 for _ in range(size)] for _ in range(size)]

        size = self._config.perception_grid_size
        radius = self._config.perception_radius

        if HAS_NUMPY:
            perception = np.zeros((size, size), dtype=np.int8)
        else:
            perception = [[0 for _ in range(size)] for _ in range(size)]

        # Get nearby entities
        nearby = self.get_nearby_entities(["resources", "agents"])
        nearby_resources = nearby.get("resources", [])
        nearby_agents = nearby.get("agents", [])

        # Get agent position
        agent_pos = self._agent.state_manager.position

        # Helper to convert world to grid coordinates
        def world_to_grid(wx: float, wy: float) -> tuple:
            gx = int(wx - agent_pos[0] + radius)
            gy = int(wy - agent_pos[1] + radius)
            return gx, gy

        # Add resources to grid
        for resource in nearby_resources:
            # Validate position attribute exists and is properly structured
            if not hasattr(resource, "position") or not resource.position:
                logger.warning(f"Resource {getattr(resource, 'id', 'unknown')} missing position attribute")
                continue
            try:
                pos = resource.position
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    logger.warning(f"Resource {getattr(resource, 'id', 'unknown')} has invalid position format: {pos}")
                    continue
                gx, gy = world_to_grid(float(pos[0]), float(pos[1]))
                if 0 <= gx < size and 0 <= gy < size:
                    if HAS_NUMPY:
                        perception[gy, gx] = 1
                    else:
                        perception[gy][gx] = 1
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Resource {getattr(resource, 'id', 'unknown')} position conversion failed: {e}, position: {getattr(resource, 'position', 'None')}")
                continue

        # Add other agents to grid
        for agent in nearby_agents:
            if agent.agent_id != self._agent.agent_id:
                # Validate position attribute exists and is properly structured
                if not hasattr(agent, "position") or not agent.position:
                    logger.warning(f"Agent {getattr(agent, 'agent_id', 'unknown')} missing position attribute")
                    continue
                try:
                    pos = agent.position
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        logger.warning(f"Agent {getattr(agent, 'agent_id', 'unknown')} has invalid position format: {pos}")
                        continue
                    gx, gy = world_to_grid(float(pos[0]), float(pos[1]))
                    if 0 <= gx < size and 0 <= gy < size:
                        if HAS_NUMPY:
                            perception[gy, gx] = 2
                        else:
                            perception[gy][gx] = 2
                except (TypeError, ValueError, IndexError) as e:
                    logger.warning(f"Agent {getattr(agent, 'agent_id', 'unknown')} position conversion failed: {e}, position: {getattr(agent, 'position', 'None')}")
                    continue

        # Mark boundaries (if we have validation service)
        # This would require access to environment bounds
        # For now, leave boundaries unmarked (could be added later)

        return perception

    def can_see(self, target_position: tuple) -> bool:
        """
        Check if a position is within perception radius.

        Args:
            target_position: Position to check visibility of

        Returns:
            bool: True if position is within perception radius

        Example:
            >>> if perception.can_see(resource.position):
            ...     print("I can see that resource!")
        """
        if self._agent is None:
            return False

        agent_pos = self._agent.state_manager.position
        dx = target_position[0] - agent_pos[0]
        dy = target_position[1] - agent_pos[1]
        distance = (dx * dx + dy * dy) ** 0.5

        return distance <= self._config.perception_radius

    def count_nearby(self, entity_type: str, radius: Optional[float] = None) -> int:
        """
        Count entities of a specific type nearby.

        Args:
            entity_type: Type to count (e.g., "resources", "agents")
            radius: Search radius (uses config radius if None)

        Returns:
            int: Number of entities found

        Example:
            >>> resource_count = perception.count_nearby("resources")
            >>> print(f"Found {resource_count} nearby resources")
        """
        nearby = self.get_nearby_entities([entity_type], radius)
        return len(nearby.get(entity_type, []))

    def get_state(self) -> dict:
        """
        Get serializable state.

        Returns:
            dict: Component state (perception has no state to persist)
        """
        return {
            "perception_radius": self._config.perception_radius,
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary

        Note:
            Perception component is stateless, configuration is immutable.
        """
        # Perception component has no mutable state to load
        pass
