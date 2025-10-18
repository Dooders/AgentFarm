"""
Base component class with common functionality.

All agent components inherit from this base to implement the IAgentComponent
interface and access common utilities.
"""

from typing import TYPE_CHECKING, Optional

from farm.core.agent.interfaces import IAgentComponent
from farm.core.agent.services import AgentServices

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore


class AgentComponent(IAgentComponent):
    """
    Base class for all agent components.
    
    Provides common functionality for components including:
    - Lifecycle hook implementation
    - Services access
    - Agent core reference
    - Common logging/validation helpers
    """
    
    def __init__(self, services: AgentServices, name: str = ""):
        """
        Initialize component.
        
        Args:
            services: AgentServices container with all required services
            name: Component name for logging and debugging
        """
        self.services = services
        self.name = name or self.__class__.__name__
        self.core: Optional["AgentCore"] = None
    
    def attach(self, core: "AgentCore") -> None:
        """Store reference to agent core."""
        self.core = core
    
    def on_step_start(self) -> None:
        """Called at start of simulation step - implement in subclass if needed."""
        pass
    
    def on_step_end(self) -> None:
        """Called at end of simulation step - implement in subclass if needed."""
        pass
    
    def on_terminate(self) -> None:
        """Called when agent terminates - implement in subclass if needed."""
        pass
    
    def _log_debug(self, message: str) -> None:
        """Log debug message if logging service available."""
        if self.logging_service:
            try:
                self.logging_service.log_debug(f"[{self.name}] {message}")
            except Exception:
                pass
    
    @property
    def logging_service(self):
        """Access logging service."""
        return self.services.logging_service
    
    @property
    def metrics_service(self):
        """Access metrics service."""
        return self.services.metrics_service
    
    @property
    def validation_service(self):
        """Access validation service."""
        return self.services.validation_service
    
    @property
    def time_service(self):
        """Access time service."""
        return self.services.time_service
    
    @property
    def spatial_service(self):
        """Access spatial service."""
        return self.services.spatial_service
    
    @property
    def lifecycle_service(self):
        """Access lifecycle service."""
        return self.services.lifecycle_service
    
    @property
    def current_time(self) -> int:
        """Get current simulation time."""
        return self.services.get_current_time()
