"""
Centralized state management for agents.

The StateManager handles all agent state in one place, following the
Single Responsibility Principle and making state changes traceable.
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


from farm.utils.logging import get_logger

logger = get_logger(__name__)


class StateManager:
    """
    Manages agent state including position, orientation, and metadata.

    The StateManager follows SRP by handling only state tracking and updates.
    It provides a clean interface for reading and modifying agent state while
    ensuring consistency and proper event notification.

    Benefits:
    - Centralized state: All state changes go through one place
    - Traceable: Easy to log or debug state changes
    - Consistent: State updates trigger proper side effects
    - Testable: Can test state logic in isolation

    State managed:
    - Position (x, y, z)
    - Orientation (rotation in degrees)
    - Birth time
    - Death time (if applicable)
    - Generation
    - Genome ID
    """

    def __init__(self, agent: "AgentCore"):
        """
        Initialize state manager for an agent.

        Args:
            agent: The agent whose state this manager tracks
        """
        self._agent = agent

        # Position state
        self._position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._orientation: float = 0.0  # Degrees, 0 = north/up

        # Lifecycle state
        self._birth_time: int = 0
        self._death_time: Optional[int] = None

        # Genealogy state
        self._generation: int = 0
        self._genome_id: str = ""
        self._parent_ids: list[str] = []

    # Position management

    @property
    def position(self) -> Tuple[float, float]:
        """
        Get current 2D position.

        Returns:
            tuple: (x, y) position
        """
        return (self._position[0], self._position[1])

    @property
    def position_3d(self) -> Tuple[float, float, float]:
        """
        Get current 3D position.

        Returns:
            tuple: (x, y, z) position
        """
        return self._position

    def set_position(
        self, position: Optional[Tuple[float, float]] | Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Update agent position and notify spatial service.

        Args:
            position: New (x, y) or (x, y, z) position

        Note:
            This automatically marks spatial structures as dirty for rebuilding.
        """
        if position is None:
            return

        old_position = self._position

        # Handle 2D or 3D position
        if len(position) == 2:
            self._position = (position[0], position[1], 0.0)
        else:
            # For 3D position, ensure Z coordinate is not None
            z_coord = position[2] if position[2] is not None else 0.0
            self._position = (position[0], position[1], z_coord)

        # Notify spatial service if position changed
        if old_position != self._position:
            try:
                # Access spatial service through agent
                if hasattr(self._agent, "_spatial_service"):
                    self._agent._spatial_service.mark_positions_dirty()
            except Exception as e:
                logger.warning(
                    f"Failed to mark spatial positions dirty: {e}",
                    agent_id=self._agent.agent_id,
                )

    # Orientation management

    @property
    def orientation(self) -> float:
        """
        Get current orientation in degrees.

        Returns:
            float: Orientation (0 = north, 90 = east, 180 = south, 270 = west)
        """
        return self._orientation

    def set_orientation(self, degrees: float) -> None:
        """
        Set agent orientation.

        Args:
            degrees: New orientation in degrees (0-360, or any value which
                    will be normalized to 0-360)
        """
        # Normalize to 0-360 range
        self._orientation = degrees % 360.0

    def rotate(self, delta_degrees: float) -> None:
        """
        Rotate agent by delta degrees.

        Args:
            delta_degrees: Rotation amount (positive = clockwise)
        """
        self._orientation = (self._orientation + delta_degrees) % 360.0

    # Lifecycle management

    @property
    def birth_time(self) -> int:
        """Get the simulation step when this agent was born."""
        return self._birth_time

    def set_birth_time(self, time: int) -> None:
        """
        Set birth time.

        Args:
            time: Simulation step of birth
        """
        self._birth_time = time

    @property
    def death_time(self) -> Optional[int]:
        """Get the simulation step when this agent died (None if alive)."""
        return self._death_time

    def set_death_time(self, time: int) -> None:
        """
        Set death time.

        Args:
            time: Simulation step of death
        """
        self._death_time = time

    @property
    def age(self) -> int:
        """
        Calculate agent's current age in simulation steps.

        Returns:
            int: Steps since birth (or until death if dead)
        """
        if self._death_time is not None:
            return self._death_time - self._birth_time

        # Get current time from agent's time service if available
        current_time = self._birth_time
        try:
            if hasattr(self._agent, "_time_service") and self._agent._time_service:
                current_time = self._agent._time_service.current_time()
        except Exception:
            pass

        return current_time - self._birth_time

    # Genealogy management

    @property
    def generation(self) -> int:
        """Get agent's generation number."""
        return self._generation

    def set_generation(self, generation: int) -> None:
        """
        Set generation number.

        Args:
            generation: Generation number (0 for initial population)
        """
        self._generation = generation

    @property
    def genome_id(self) -> str:
        """Get agent's genome identifier."""
        return self._genome_id

    def set_genome_id(self, genome_id: str) -> None:
        """
        Set genome identifier.

        Args:
            genome_id: Unique genome identifier string
        """
        self._genome_id = genome_id

    @property
    def parent_ids(self) -> list[str]:
        """Get list of parent agent IDs."""
        return self._parent_ids.copy()  # Return copy to prevent external modification

    def set_parent_ids(self, parent_ids: list[str]) -> None:
        """
        Set parent agent IDs.

        Args:
            parent_ids: List of parent agent IDs
        """
        self._parent_ids = parent_ids.copy()  # Store copy to prevent external modification

    # State serialization

    def get_state_dict(self) -> dict:
        """
        Get complete state as dictionary for serialization.

        Returns:
            dict: All state values

        Note:
            This is used for genome serialization and state persistence.
        """
        return {
            "position": self._position,
            "orientation": self._orientation,
            "birth_time": self._birth_time,
            "death_time": self._death_time,
            "generation": self._generation,
            "genome_id": self._genome_id,
            "parent_ids": self._parent_ids,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary (from get_state_dict())
        """
        self._position = tuple(state.get("position", (0.0, 0.0, 0.0)))
        self._orientation = state.get("orientation", 0.0)
        self._birth_time = state.get("birth_time", 0)
        self._death_time = state.get("death_time", None)
        self._generation = state.get("generation", 0)
        self._genome_id = state.get("genome_id", "")
        self._parent_ids = state.get("parent_ids", [])
