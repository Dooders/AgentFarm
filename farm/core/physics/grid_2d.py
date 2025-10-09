"""Grid2D physics engine implementation for AgentFarm environments.

This module implements the IPhysicsEngine protocol for 2D grid-based environments,
wrapping the existing spatial indexing and observation systems from the Environment class.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

from farm.core.channels import NUM_CHANNELS
from farm.core.observations import ObservationConfig
from farm.core.physics.interface import IPhysicsEngine
from farm.core.spatial import SpatialIndex
from farm.utils.config_utils import resolve_spatial_index_config
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class Grid2DPhysics(IPhysicsEngine):
    """2D grid-based physics engine for AgentFarm environments.
    
    This implementation wraps the existing spatial indexing system and provides
    a clean interface for 2D grid-based environments. It handles position validation,
    spatial queries, distance calculations, and observation space definition.
    
    Attributes:
        width (int): Environment width in grid units
        height (int): Environment height in grid units
        spatial_index (SpatialIndex): Spatial indexing system for efficient queries
        observation_config (ObservationConfig): Configuration for observation spaces
        seed (Optional[int]): Random seed for deterministic behavior
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        spatial_config: Optional[Any] = None,
        observation_config: Optional[ObservationConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Grid2D physics engine.
        
        Args:
            width: Environment width in grid units
            height: Environment height in grid units
            spatial_config: Configuration for spatial indexing (optional)
            observation_config: Configuration for observation spaces (optional)
            seed: Random seed for deterministic behavior (optional)
        """
        self.width = width
        self.height = height
        self.seed = seed
        
        # Set up random number generator
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize spatial index
        self._setup_spatial_index(spatial_config)
        
        # Initialize observation configuration
        self.observation_config = observation_config or ObservationConfig()
        self._setup_observation_space()
        
        # Entity storage for spatial queries
        self._entities: Dict[str, List[Any]] = {
            "agents": [],
            "resources": [],
            "objects": []
        }
    
    def _setup_spatial_index(self, spatial_config: Optional[Any]) -> None:
        """Initialize spatial indexing system."""
        if spatial_config:
            resolved_config = resolve_spatial_index_config(spatial_config)
        else:
            resolved_config = None
        
        if resolved_config:
            self.spatial_index = SpatialIndex(
                self.width,
                self.height,
                enable_batch_updates=resolved_config.enable_batch_updates,
                region_size=resolved_config.region_size,
                max_batch_size=resolved_config.max_batch_size,
                dirty_region_batch_size=getattr(
                    resolved_config, "dirty_region_batch_size", 10
                ),
            )
            
            # Enable additional index types if configured
            if resolved_config.enable_quadtree_indices:
                self._enable_quadtree_indices()
            if resolved_config.enable_spatial_hash_indices:
                self._enable_spatial_hash_indices(resolved_config.spatial_hash_cell_size)
        else:
            # Default configuration with batch updates enabled
            self.spatial_index = SpatialIndex(
                self.width,
                self.height,
                enable_batch_updates=True,
                region_size=50.0,
                max_batch_size=100,
                dirty_region_batch_size=10,
            )
    
    def _setup_observation_space(self) -> None:
        """Setup observation space based on configuration."""
        S = 2 * self.observation_config.R + 1
        
        # Robust numpy dtype mapping from torch dtype or string
        torch_dtype = (
            getattr(torch, self.observation_config.dtype)
            if isinstance(self.observation_config.dtype, str)
            else self.observation_config.dtype
        )
        
        if torch_dtype in (torch.float32, torch.float):
            np_dtype = np.float32
        elif torch_dtype in (torch.float64, torch.double):
            np_dtype = np.float64
        elif torch_dtype in (torch.float16, torch.half):
            np_dtype = np.float16
        elif torch_dtype == torch.bfloat16:
            # numpy has no bfloat16; use float32 for the observation space dtype
            np_dtype = np.float32
        else:
            np_dtype = np.float32
            
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_CHANNELS, S, S), dtype=np_dtype
        )
    
    def validate_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid within the environment bounds.
        
        Args:
            position: (x, y) coordinates to validate
            
        Returns:
            True if position is within bounds, False otherwise
        """
        if not isinstance(position, tuple) or len(position) != 2:
            return False
        
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_nearby_entities(
        self, 
        position: Tuple[float, float], 
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Find entities near a position using spatial indexing.
        
        Args:
            position: Center position to search from
            radius: Search radius
            entity_type: Type of entities to search for ("agents", "resources", "objects")
            
        Returns:
            List of nearby entities
        """
        if not self.validate_position(position):
            return []
        
        # Use spatial index for efficient queries
        if entity_type == "agents":
            return self.spatial_index.get_nearby_range(position, radius, "agents")
        elif entity_type == "resources":
            return self.spatial_index.get_nearby_range(position, radius, "resources")
        else:
            # For other entity types, fall back to brute force
            nearby = []
            for entity in self._entities.get(entity_type, []):
                if hasattr(entity, 'position'):
                    if self.compute_distance(position, entity.position) <= radius:
                        nearby.append(entity)
            return nearby
    
    def compute_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two positions.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Euclidean distance
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get shape of state representation.
        
        Returns:
            Tuple describing state dimensions (width, height)
        """
        return (self.width, self.height)
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Get observation space for an agent.
        
        Args:
            agent_id: Agent identifier (not used for Grid2D, all agents have same space)
            
        Returns:
            Gymnasium Box space for observations
        """
        return self._observation_space
    
    def sample_position(self) -> Tuple[float, float]:
        """Sample a random valid position.
        
        Returns:
            Random position within environment bounds
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return (x, y)
    
    def update(self, dt: float = 1.0) -> None:
        """Update physics simulation.
        
        Args:
            dt: Time step (not used for static grid physics)
        """
        # Update spatial index if needed
        if hasattr(self.spatial_index, 'update'):
            self.spatial_index.update()
    
    def reset(self) -> None:
        """Reset physics state."""
        # Clear entity lists
        for entity_type in self._entities:
            self._entities[entity_type].clear()
        
        # Reset spatial index
        if hasattr(self.spatial_index, 'reset'):
            self.spatial_index.reset()
    
    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get environment bounds.
        
        Returns:
            Tuple of (min_bounds, max_bounds) where min_bounds=(0, 0) and 
            max_bounds=(width, height)
        """
        return ((0.0, 0.0), (float(self.width), float(self.height)))
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for this physics engine.
        
        Returns:
            Dictionary describing physics configuration
        """
        return {
            "type": "grid_2d",
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "observation_config": {
                "R": self.observation_config.R,
                "dtype": str(self.observation_config.dtype),
            },
            "spatial_index": {
                "enable_batch_updates": getattr(self.spatial_index, 'enable_batch_updates', False),
                "region_size": getattr(self.spatial_index, 'region_size', 50.0),
                "max_batch_size": getattr(self.spatial_index, 'max_batch_size', 100),
            }
        }
    
    # Additional methods for spatial index management
    
    def set_entity_references(self, agents: List[Any], resources: List[Any]) -> None:
        """Set entity references for spatial indexing.
        
        Args:
            agents: List of agent objects
            resources: List of resource objects
        """
        self._entities["agents"] = agents
        self._entities["resources"] = resources
        
        # Update spatial index references
        self.spatial_index.set_references(agents, resources)
        self.spatial_index.update()
    
    def mark_positions_dirty(self) -> None:
        """Mark positions as dirty for batch updates."""
        self.spatial_index.mark_positions_dirty()
    
    def process_batch_spatial_updates(self, force: bool = False) -> None:
        """Process any pending batch spatial updates."""
        if hasattr(self.spatial_index, 'process_batch_updates'):
            self.spatial_index.process_batch_updates(force=force)
    
    def get_spatial_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for spatial indexing."""
        stats = {}
        
        # Get basic spatial index stats
        if hasattr(self.spatial_index, "get_stats"):
            stats.update(self.spatial_index.get_stats())
        
        # Add physics-specific stats
        stats.update({
            "width": self.width,
            "height": self.height,
            "total_entities": sum(len(entities) for entities in self._entities.values()),
            "agents_count": len(self._entities["agents"]),
            "resources_count": len(self._entities["resources"]),
        })
        
        return stats
    
    def enable_batch_spatial_updates(
        self, region_size: float = 50.0, max_batch_size: int = 100
    ) -> None:
        """Enable batch spatial updates with the specified configuration."""
        if hasattr(self.spatial_index, "enable_batch_updates"):
            self.spatial_index.enable_batch_updates(region_size, max_batch_size)
    
    def disable_batch_spatial_updates(self) -> None:
        """Disable batch spatial updates and process any pending updates."""
        if hasattr(self.spatial_index, "disable_batch_updates"):
            self.spatial_index.disable_batch_updates()
    
    def _enable_quadtree_indices(self) -> None:
        """Enable Quadtree indices alongside existing KD-tree indices."""
        if hasattr(self.spatial_index, "enable_quadtree_indices"):
            self.spatial_index.enable_quadtree_indices()
    
    def _enable_spatial_hash_indices(self, cell_size: Optional[float] = None) -> None:
        """Enable Spatial Hash Grid indices alongside existing KD-tree indices."""
        if hasattr(self.spatial_index, "enable_spatial_hash_indices"):
            self.spatial_index.enable_spatial_hash_indices(cell_size)
    
    # Convenience methods for backward compatibility
    
    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[Any]:
        """Find all agents within radius of position."""
        return self.get_nearby_entities(position, radius, "agents")
    
    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List[Any]:
        """Find all resources within radius of position."""
        return self.get_nearby_entities(position, radius, "resources")
    
    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        """Find nearest resource to position."""
        if not self.validate_position(position):
            return None
        
        # Use spatial index if available
        if hasattr(self.spatial_index, 'get_nearest_resource'):
            return self.spatial_index.get_nearest_resource(position)
        
        # Fall back to brute force
        nearest = None
        min_distance = float('inf')
        
        for resource in self._entities["resources"]:
            if hasattr(resource, 'position'):
                distance = self.compute_distance(position, resource.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest = resource
        
        return nearest
