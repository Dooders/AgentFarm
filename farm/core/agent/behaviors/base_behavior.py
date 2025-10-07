"""
Base interface for agent behaviors.

Behaviors implement the Strategy pattern, allowing different decision-making
algorithms to be plugged into agents without modifying the agent core.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class IAgentBehavior(ABC):
    """
    Interface for agent behavior strategies.

    Behaviors define how an agent makes decisions and acts each simulation turn.
    Different behaviors can implement different strategies (random, learning,
    scripted, etc.) while maintaining a consistent interface.

    This interface follows:
    - Strategy Pattern: Encapsulates algorithm, makes it interchangeable
    - OCP (Open-Closed Principle): New behaviors extend, don't modify existing
    - SRP (Single Responsibility Principle): Only handles decision-making logic
    - DIP (Dependency Inversion Principle): Depends on AgentCore abstraction

    Examples:
        - RandomBehavior: Random action selection
        - LearningBehavior: Reinforcement learning based decisions
        - ScriptedBehavior: Predefined action sequences
        - HumanBehavior: Decisions from external input
    """

    @abstractmethod
    def execute_turn(self, agent: "AgentCore") -> None:
        """
        Execute one simulation turn for the agent.

        This method is called once per simulation step and is responsible for:
        1. Observing the current state
        2. Deciding what action to take
        3. Executing the chosen action
        4. Updating any learning/memory systems

        Args:
            agent: The AgentCore instance to control

        Note:
            The behavior can access agent components via agent.get_component(name).
            The behavior should respect component availability and handle missing
            components gracefully.
        """
        pass

    def reset(self) -> None:
        """
        Reset the behavior state.

        Called when starting a new episode or simulation run. Use this to:
        - Clear episode-specific state
        - Reset exploration parameters
        - Reinitialize temporary buffers

        Default implementation does nothing (stateless behavior).
        """
        pass

    def get_state(self) -> dict:
        """
        Get the current state of this behavior for serialization.

        Returns:
            dict: Serializable state dictionary

        Note:
            Override this if your behavior has state that needs to be saved/loaded
            (e.g., learned policies, experience buffers).
        """
        return {}

    def load_state(self, state: dict) -> None:
        """
        Load state from a serialized dictionary.

        Args:
            state: State dictionary (from get_state())

        Note:
            Override this if your behavior has state that needs to be saved/loaded.
        """
        pass