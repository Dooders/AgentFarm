"""
Movement component.

Handles agent position, movement validation, and updates to spatial indices.
"""

from farm.core.agent.config.component_configs import MovementConfig
from farm.core.agent.services import AgentServices

from .base import AgentComponent


class MovementComponent(AgentComponent):
    """
    Manages agent position and movement.
    
    Responsibilities:
    - Track agent position
    - Validate movement to new positions
    - Update spatial indices when position changes
    - Handle position boundary checking
    """
    
    def __init__(self, services: AgentServices, config: MovementConfig):
        """
        Initialize movement component.
        
        Args:
            services: Agent services
            config: Movement configuration
        """
        super().__init__(services, "MovementComponent")
        self.config = config
        self.position = (0.0, 0.0)
    
    def attach(self, core) -> None:
        """Attach to core and initialize position from core."""
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
    
    def set_position(self, position: tuple[float, float]) -> bool:
        """
        Set agent position after validation.
        
        Args:
            position: New (x, y) position
            
        Returns:
            bool: True if position was valid and updated, False otherwise
        """
        # Validate position if validation service available
        if self.validation_service:
            if not self.validation_service.is_valid_position(position):
                return False
        
        # Update position if changed
        if self.position != position:
            self.position = position
            # Mark spatial structures as dirty
            if self.spatial_service:
                try:
                    self.spatial_service.mark_positions_dirty()
                except Exception:
                    pass
        
        return True
    
    def move_to(self, position: tuple[float, float]) -> bool:
        """
        Move agent to position with distance validation.
        
        Args:
            position: Target (x, y) position
            
        Returns:
            bool: True if move was valid and executed
        """
        # Check distance
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        
        if distance > self.config.max_movement:
            return False
        
        return self.set_position(position)
    
    def get_nearby_positions(self, radius: int) -> list[tuple[float, float]]:
        """
        Get positions within a radius of agent.
        
        Args:
            radius: Search radius
            
        Returns:
            List of positions within radius
        """
        if not self.spatial_service:
            return []
        
        try:
            nearby = self.spatial_service.get_nearby(self.position, radius, [])
            return [item.position for item in nearby if hasattr(item, 'position')]
        except Exception:
            return []
    
    @property
    def x(self) -> float:
        """Get X coordinate."""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Get Y coordinate."""
        return self.position[1]
    
    @property
    def perception_radius(self) -> int:
        """Get perception radius for queries."""
        return self.config.perception_radius
