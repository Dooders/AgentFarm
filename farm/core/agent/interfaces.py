"""
Agent component interfaces for the new component-based architecture.

Defines the core interfaces that all components must implement to integrate
with AgentCore and participate in the agent lifecycle.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class IAgentComponent(ABC):
    """
    Interface for pluggable agent components.
    
    Each component implements a specific agent capability (movement, combat, resources, etc.)
    and participates in the agent's lifecycle through well-defined hooks.
    
    Components are loosely coupled and communicate through the agent core or
    well-defined interfaces rather than direct component-to-component dependencies.
    """
    
    @abstractmethod
    def attach(self, core: "AgentCore") -> None:
        """
        Called when component is attached to an agent core.
        
        Use this to store a reference to the core and perform initialization
        that requires knowing about the agent's other properties.
        
        Args:
            core: The AgentCore instance this component is being attached to
        """
        pass
    
    @abstractmethod
    def on_step_start(self) -> None:
        """
        Called at the start of each simulation step, before actions are decided.
        
        Use this for pre-turn logic like:
        - Countdown timers
        - Resource regeneration
        - Status updates
        """
        pass
    
    @abstractmethod
    def on_step_end(self) -> None:
        """
        Called at the end of each simulation step, after action execution.
        
        Use this for post-turn logic like:
        - Resource consumption
        - Damage application
        - State cleanup
        """
        pass
    
    @abstractmethod
    def on_terminate(self) -> None:
        """
        Called when the agent is being terminated/killed.
        
        Use this for cleanup, final logging, or state persistence.
        """
        pass
