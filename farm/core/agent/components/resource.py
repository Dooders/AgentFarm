"""
Resource component for agent resource management.

Handles resource tracking, consumption, and starvation mechanics.
"""

from typing import TYPE_CHECKING, Optional
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import ResourceConfig

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class ResourceComponent(IAgentComponent):
    """
    Component handling agent resource management.

    Responsibilities:
    - Track current resource level
    - Add/consume resources
    - Monitor starvation
    - Trigger death when starved

    Single Responsibility: Only resource tracking and starvation.
    """

    def __init__(self, initial_resources: int, config: ResourceConfig):
        """
        Initialize resource component.

        Args:
            initial_resources: Starting resource level
            config: Resource configuration
        """
        self._resources = initial_resources
        self._config = config
        self._agent: Optional["AgentCore"] = None
        self._starvation_counter = 0

    @property
    def name(self) -> str:
        """Component identifier."""
        return "resource"

    @property
    def level(self) -> int:
        """Current resource level."""
        return self._resources

    @property
    def is_starving(self) -> bool:
        """Whether agent is currently starving (resources <= 0)."""
        return self._resources <= 0

    @property
    def starvation_steps(self) -> int:
        """Number of consecutive steps with zero resources."""
        return self._starvation_counter

    def add(self, amount: int) -> None:
        """
        Add resources.

        Args:
            amount: Amount to add (can be negative to consume)

        Example:
            >>> resource.add(50)  # Gather 50 resources
            >>> resource.add(-10)  # Consume 10 resources
        """
        self._resources += amount

        # Reset starvation counter if we have resources
        if self._resources > 0:
            self._starvation_counter = 0

    def consume(self, amount: int) -> bool:
        """
        Consume resources if available.

        Args:
            amount: Amount to consume

        Returns:
            bool: True if consumption was successful, False if insufficient

        Example:
            >>> if resource.consume(20):
            ...     print("Successfully consumed 20 resources")
            ... else:
            ...     print("Insufficient resources")
        """
        if self._resources >= amount:
            self._resources -= amount
            # Reset starvation counter if we still have resources
            if self._resources > 0:
                self._starvation_counter = 0
            return True
        return False

    def set_level(self, amount: int) -> None:
        """
        Set resource level directly.

        Args:
            amount: New resource level

        Note:
            Use sparingly - prefer add() or consume() for tracking changes.
        """
        self._resources = amount
        if self._resources > 0:
            self._starvation_counter = 0

    def has_resources(self, amount: int) -> bool:
        """
        Check if agent has at least the specified amount.

        Args:
            amount: Minimum resources required

        Returns:
            bool: True if agent has enough resources

        Example:
            >>> if resource.has_resources(50):
            ...     # Agent can afford something costing 50
        """
        return self._resources >= amount

    def on_step_end(self) -> None:
        """
        Called at end of each step.

        Applies base resource consumption and checks for starvation death.
        """
        # Base consumption per turn
        self._resources -= self._config.base_consumption_rate

        # Check starvation
        if self._resources <= 0:
            self._starvation_counter += 1

            # Trigger death if starved too long
            if self._starvation_counter >= self._config.starvation_threshold:
                if self._agent:
                    self._agent.terminate()
        else:
            self._starvation_counter = 0

    def get_state(self) -> dict:
        """
        Get serializable state.

        Returns:
            dict: Component state including resource level and starvation counter
        """
        return {
            "resources": self._resources,
            "starvation_counter": self._starvation_counter,
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self._resources = state.get("resources", 0)
        self._starvation_counter = state.get("starvation_counter", 0)