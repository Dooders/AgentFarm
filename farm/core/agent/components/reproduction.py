"""
Reproduction component.

Handles agent reproduction and offspring creation mechanics.
"""

from typing import Optional

from farm.core.agent.config.component_configs import ReproductionConfig
from farm.core.agent.services import AgentServices

from .base import AgentComponent


class ReproductionComponent(AgentComponent):
    """
    Manages agent reproduction.
    
    Responsibilities:
    - Track offspring creation
    - Manage reproduction cost
    - Handle parent-offspring relationships
    """
    
    def __init__(self, services: AgentServices, config: ReproductionConfig):
        """
        Initialize reproduction component.
        
        Args:
            services: Agent services
            config: Reproduction configuration
        """
        super().__init__(services, "ReproductionComponent")
        self.config = config
        self.offspring_created = 0
    
    def attach(self, core) -> None:
        """Attach to core."""
        super().attach(core)
    
    def on_step_start(self) -> None:
        """Called at start of step."""
        pass
    
    def on_step_end(self) -> None:
        """Called at end of step."""
        pass
    
    def on_terminate(self) -> None:
        """Called when agent dies."""
        pass
    
    def can_reproduce(self) -> bool:
        """
        Check if agent has sufficient resources to reproduce.
        
        Returns:
            bool: True if agent can afford offspring cost
        """
        if not self.core:
            return False
        
        resource_component = self.core.get_component("resource")
        if not resource_component:
            return False
        
        return resource_component.level >= self.config.offspring_cost
    
    def reproduce(self) -> Optional["AgentCore"]:  # noqa: F821
        """
        Create offspring agent.
        
        This is a template method - actual offspring creation should be
        implemented by the factory or lifecycle service.
        
        Returns:
            New agent instance or None if reproduction failed
        """
        if not self.can_reproduce():
            return None
        
        if not self.core:
            return None
        
        # Deduct cost from parent
        resource_component = self.core.get_component("resource")
        if resource_component:
            resource_component.remove(self.config.offspring_cost)
        
        self.offspring_created += 1
        
        return None
    
    @property
    def total_offspring(self) -> int:
        """Get total number of offspring created."""
        return self.offspring_created
