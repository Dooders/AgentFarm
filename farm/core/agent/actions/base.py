"""
Base interface for agent actions.

Actions follow the Command pattern, encapsulating requests as objects.
This allows for validation, cost estimation, and execution as separate concerns.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class IAction(ABC):
    """
    Interface for agent actions.

    Actions are executable commands that agents can perform. Each action
    encapsulates its own validation logic, cost estimation, and execution.

    This interface follows:
    - Command Pattern: Encapsulates action as object
    - SRP (Single Responsibility Principle): Each action class does one thing
    - OCP (Open-Closed Principle): New actions extend, don't modify existing
    - ISP (Interface Segregation Principle): Small, focused interface

    Actions are stateless and can be shared across agents. Any agent-specific
    state is passed as parameters to the methods.

    Lifecycle:
    1. Action is instantiated once and registered
    2. can_execute() checks if action is valid for an agent
    3. estimate_cost() provides cost information for planning
    4. execute() performs the actual action
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this action type.

        Returns:
            str: Action name (e.g., "move", "gather", "attack")

        Note:
            This should be a constant string that uniquely identifies
            the action type. Used for action selection and logging.
        """
        pass

    @property
    def description(self) -> str:
        """
        Human-readable description of what this action does.

        Returns:
            str: Action description

        Note:
            Override this to provide helpful documentation.
        """
        return f"Action: {self.name}"

    @abstractmethod
    def can_execute(self, agent: "AgentCore") -> bool:
        """
        Check if this action can be executed by the agent.

        This method validates preconditions without modifying state:
        - Does the agent have required components?
        - Does the agent have sufficient resources?
        - Is the action valid in the current context?

        Args:
            agent: The agent attempting to execute this action

        Returns:
            bool: True if the action can be executed, False otherwise

        Note:
            This should be a pure function with no side effects.
            It's called before execute() to validate the action.
        """
        pass

    @abstractmethod
    def execute(self, agent: "AgentCore") -> Dict[str, Any]:
        """
        Execute the action for the given agent.

        This method performs the actual action and may modify agent state,
        environment state, or interact with other agents.

        Args:
            agent: The agent executing this action

        Returns:
            dict: Result dictionary with structure:
                {
                    "success": bool,        # Whether action succeeded
                    "error": str,           # Error message if failed (optional)
                    "details": dict,        # Action-specific details (optional)
                    "changes": dict,        # State changes made (optional)
                }

        Note:
            This method should be idempotent when it fails - if the action
            cannot be completed, it should leave the agent in a consistent state.
        """
        pass

    def estimate_cost(self, agent: "AgentCore") -> float:
        """
        Estimate the resource cost of executing this action.

        Used by planning algorithms to estimate action costs before execution.

        Args:
            agent: The agent that would execute this action

        Returns:
            float: Estimated resource cost (0 if no cost or cost unknown)

        Note:
            This is an estimate and may not match the actual cost.
            Override this if your action has a predictable resource cost.
        """
        return 0.0

    def estimate_reward(self, agent: "AgentCore") -> float:
        """
        Estimate the expected reward from executing this action.

        Used by planning algorithms to evaluate action value before execution.

        Args:
            agent: The agent that would execute this action

        Returns:
            float: Estimated reward (0 if unknown)

        Note:
            This is an estimate for planning purposes. The actual reward
            is calculated after execution based on state changes.
            Override this if your action has a predictable reward.
        """
        return 0.0

    def get_requirements(self) -> Dict[str, Any]:
        """
        Get the requirements for executing this action.

        Returns:
            dict: Requirements dictionary:
                {
                    "components": List[str],  # Required component names
                    "resources": float,       # Minimum resources needed
                    "other": dict,           # Other requirements
                }

        Note:
            This is used for validation and planning. Override to specify
            what the agent needs to execute this action.
        """
        return {
            "components": [],
            "resources": 0.0,
            "other": {},
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}')"