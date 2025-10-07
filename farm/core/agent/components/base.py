"""
Base interface for agent components.

Components are pluggable units of behavior that can be attached to agents.
Each component has a single, well-defined responsibility.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class IAgentComponent(ABC):
    """
    Interface for agent components.

    Components are composable units of agent behavior following the
    Single Responsibility Principle. Each component manages one specific
    aspect of an agent's capabilities (movement, resources, combat, etc.).

    This interface follows:
    - ISP (Interface Segregation Principle): Small, focused interface
    - SRP (Single Responsibility Principle): Each component has one purpose
    - OCP (Open-Closed Principle): Extend via new components, not modification

    Lifecycle:
    1. Component is created with configuration
    2. Component is attached to an agent via attach()
    3. Component receives lifecycle events (on_step_start, on_step_end, on_terminate)
    4. Component can be queried for its state and capabilities
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name identifier for this component type.

        Returns:
            str: Component type name (e.g., "movement", "resource", "combat")
        """
        pass

    def attach(self, agent: "AgentCore") -> None:
        """
        Attach this component to an agent.

        Called when the component is added to an agent. This gives the component
        a reference to its parent agent for accessing other components or services.

        Args:
            agent: The AgentCore instance this component belongs to
        """
        self._agent = agent

    def on_step_start(self) -> None:
        """
        Called at the start of each simulation step.

        Use this for:
        - Resetting per-turn state
        - Preparing for the upcoming turn
        - Checking preconditions

        This is called BEFORE the agent's behavior executes.
        """
        pass

    def on_step_end(self) -> None:
        """
        Called at the end of each simulation step.

        Use this for:
        - Applying per-turn effects (resource consumption, timer decrements)
        - Updating state based on turn results
        - Checking postconditions

        This is called AFTER the agent's behavior executes.
        """
        pass

    def on_terminate(self) -> None:
        """
        Called when the agent is terminated (dies).

        Use this for:
        - Cleanup operations
        - Final state recording
        - Resource release

        After this is called, the component should assume the agent is no longer active.
        """
        pass

    def get_state(self) -> dict:
        """
        Get the current state of this component for serialization.

        Returns:
            dict: Serializable state dictionary

        Note:
            Override this if your component has state that needs to be saved/loaded.
        """
        return {}

    def load_state(self, state: dict) -> None:
        """
        Load state from a serialized dictionary.

        Args:
            state: State dictionary (from get_state())

        Note:
            Override this if your component has state that needs to be saved/loaded.
        """
        pass