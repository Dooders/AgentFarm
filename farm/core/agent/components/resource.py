"""
Resource management component.

Handles resource levels, consumption, starvation mechanics, and reproduction cost tracking.
"""

from farm.core.agent.config.component_configs import ResourceConfig
from farm.core.agent.services import AgentServices

from .base import AgentComponent


class ResourceComponent(AgentComponent):
    """
    Manages agent resource levels and starvation.
    
    Responsibilities:
    - Track resource level
    - Apply consumption each step
    - Handle starvation when resources depleted
    - Track when agent will starve
    """
    
    def __init__(self, services: AgentServices, config: ResourceConfig):
        """
        Initialize resource component.
        
        Args:
            services: Agent services
            config: Resource configuration
        """
        super().__init__(services, "ResourceComponent")
        self.config = config
        self.level = 0.0
        self.starvation_counter = 0
    
    def attach(self, core) -> None:
        """Attach to core and initialize resource level from core."""
        super().attach(core)
        # Will be set by caller after attachment
    
    def on_step_start(self) -> None:
        """Apply resource consumption at start of step."""
        self.level -= self.config.base_consumption_rate
        self._check_starvation()
    
    def on_step_end(self) -> None:
        """Called at end of step for any post-action resource updates."""
        pass
    
    def on_terminate(self) -> None:
        """Called when agent dies."""
        pass
    
    def _check_starvation(self) -> bool:
        """
        Check if agent should die from starvation.
        
        Returns:
            True if agent died from starvation, False otherwise
        """
        if self.level <= 0:
            self.starvation_counter += 1
            if self.starvation_counter >= self.config.starvation_threshold:
                if self.core:
                    self.core.terminate()
                return True
        else:
            self.starvation_counter = 0
        return False
    
    def add(self, amount: float) -> None:
        """Add resources to the agent."""
        self.level += amount
    
    def remove(self, amount: float) -> bool:
        """
        Remove resources from the agent.
        
        Args:
            amount: Amount to remove
            
        Returns:
            True if successful, False if insufficient resources
        """
        if self.level >= amount:
            self.level -= amount
            return True
        return False
    
    @property
    def has_resources(self) -> bool:
        """Check if agent has any resources."""
        return self.level > 0
    
    @property
    def is_starving(self) -> bool:
        """Check if agent is currently starving (counter > 0)."""
        return self.starvation_counter > 0
    
    @property
    def turns_until_starvation(self) -> int:
        """Get number of turns until starvation if resources remain depleted."""
        if self.level > 0:
            return self.config.starvation_threshold
        return max(0, self.config.starvation_threshold - self.starvation_counter)
