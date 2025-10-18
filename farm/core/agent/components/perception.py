"""
Perception component.

Handles agent perception, spatial awareness, and observation generation for decision-making.
"""

import math

import numpy as np
import torch

from farm.core.agent.config.component_configs import PerceptionConfig
from farm.core.agent.services import AgentServices
from farm.core.perception import PerceptionData

from .base import AgentComponent


class PerceptionComponent(AgentComponent):
    """
    Manages agent perception and observation.
    
    Responsibilities:
    - Query spatial service for nearby entities
    - Generate perception grids
    - Build observation tensors for decision-making
    - Handle egocentric perception with agent orientation
    """
    
    def __init__(self, services: AgentServices, config: PerceptionConfig):
        """
        Initialize perception component.
        
        Args:
            services: Agent services
            config: Perception configuration
        """
        super().__init__(services, "PerceptionComponent")
        self.config = config
        self.last_perception = None
    
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
    
    def get_perception(self) -> PerceptionData:
        """
        Get agent's current perception of surrounding environment.
        
        Creates a grid representation of nearby entities:
        - 0: Empty space
        - 1: Resource
        - 2: Other agent
        - 3: Boundary/obstacle
        
        Returns:
            PerceptionData: Structured perception grid
        """
        if not self.core:
            return PerceptionData(np.zeros((2 * self.config.perception_radius + 1, 2 * self.config.perception_radius + 1), dtype=np.int8))
        
        radius = self.config.perception_radius
        size = 2 * radius + 1
        perception = np.zeros((size, size), dtype=np.int8)
        
        # Get nearby entities
        nearby_resources = []
        nearby_agents = []
        
        if self.spatial_service:
            try:
                nearby = self.spatial_service.get_nearby(self.core.position, radius, ["resources"])
                nearby_resources = nearby.get("resources", [])
            except Exception:
                pass
            
            try:
                nearby = self.spatial_service.get_nearby(self.core.position, radius, ["agents"])
                nearby_agents = nearby.get("agents", [])
            except Exception:
                pass
        
        # Helper to convert world coords to grid
        def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
            if self.config.position_discretization_method == "round":
                gx = int(round(wx - self.core.position[0] + radius))
                gy = int(round(wy - self.core.position[1] + radius))
            elif self.config.position_discretization_method == "ceil":
                gx = int(math.ceil(wx - self.core.position[0] + radius))
                gy = int(math.ceil(wy - self.core.position[1] + radius))
            else:  # "floor" (default)
                gx = int(math.floor(wx - self.core.position[0] + radius))
                gy = int(math.floor(wy - self.core.position[1] + radius))
            return gx, gy
        
        # Add resources
        for resource in nearby_resources:
            try:
                gx, gy = world_to_grid(resource.position[0], resource.position[1])
                if 0 <= gx < size and 0 <= gy < size:
                    perception[gy, gx] = 1
            except Exception:
                pass
        
        # Add other agents
        for agent in nearby_agents:
            try:
                if agent.agent_id != self.core.agent_id:
                    gx, gy = world_to_grid(agent.position[0], agent.position[1])
                    if 0 <= gx < size and 0 <= gy < size:
                        perception[gy, gx] = 2
            except Exception:
                pass
        
        # Mark boundaries
        x_min = self.core.position[0] - radius
        y_min = self.core.position[1] - radius
        
        for i in range(size):
            for j in range(size):
                world_x = x_min + j
                world_y = y_min + i
                try:
                    if self.validation_service and not self.validation_service.is_valid_position((world_x, world_y)):
                        perception[i, j] = 3
                except Exception:
                    perception[i, j] = 3
        
        self.last_perception = PerceptionData(perception)
        return self.last_perception
    
    def get_observation_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Get perception as a torch tensor for decision-making.
        
        Args:
            device: Torch device to place tensor on
            
        Returns:
            Observation tensor from environment if available, else perception grid
        """
        if not self.core:
            return torch.zeros((1, 11, 11), dtype=torch.float32)
        
        # Prefer environment observation if available
        if hasattr(self.core, 'environment') and self.core.environment:
            try:
                observation_np = self.core.environment.observe(self.core.agent_id)
                if device is None:
                    device = getattr(self.core, 'device', torch.device('cpu'))
                return torch.from_numpy(observation_np).to(device=device, dtype=torch.float32)
            except Exception:
                pass
        
        # Fallback to perception grid
        perception = self.get_perception()
        if device is None:
            device = getattr(self.core, 'device', torch.device('cpu'))
        
        # Convert to multi-channel tensor
        grid = perception.grid.astype(np.float32)
        tensor = torch.from_numpy(grid).unsqueeze(0).to(device=device)
        return tensor
