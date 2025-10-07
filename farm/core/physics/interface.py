"""Physics engine interface for AgentFarm environments.

This module defines the protocol that all physics engines must implement.
Physics engines handle spatial validation, entity queries, distance calculations,
and state representation for different environment types.

The abstraction allows swapping between:
- Grid-based 2D environments (current default)
- Static/fixed position environments (e.g., catapult aiming)
- Continuous physics environments (e.g., robotics, navigation)
- Custom domain-specific physics

Example:
    >>> from farm.core.physics import Grid2DPhysics, StaticPhysics
    >>> 
    >>> # Use grid physics (default)
    >>> physics = Grid2DPhysics(width=100, height=100)
    >>> env = Environment(physics_engine=physics, config=config)
    >>> 
    >>> # Or use static physics
    >>> physics = StaticPhysics(valid_positions=[(0,0), (1,1)])
    >>> env = Environment(physics_engine=physics, config=config)
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from gymnasium import spaces


@runtime_checkable
class IPhysicsEngine(Protocol):
    """Protocol defining physics and spatial operations for environments.
    
    This abstraction allows different environment types (2D grid, continuous,
    static, etc.) to be swapped transparently without changing the core
    Environment or agent logic.
    
    All physics engines must implement these methods to be compatible with
    the AgentFarm Environment class.
    """
    
    def validate_position(self, position: Any) -> bool:
        """Check if a position is valid in this environment.
        
        The position format depends on the physics engine implementation:
        - Grid2D: (x: float, y: float) tuple
        - Static: Any hashable position identifier
        - Continuous: numpy array
        
        Args:
            position: Position representation (format depends on implementation)
            
        Returns:
            True if position is valid, False otherwise
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> physics.validate_position((50, 50))  # True
            >>> physics.validate_position((-1, 50))  # False
        """
        ...
    
    def get_nearby_entities(
        self, 
        position: Any, 
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Find entities near a position.
        
        What "nearby" means depends on the physics engine:
        - Grid2D: Euclidean distance
        - Static: Same state cluster / discrete hops
        - Continuous: Configurable distance metric
        
        Args:
            position: Center position to search from
            radius: Search radius (interpretation depends on implementation)
            entity_type: Type of entities to search for ("agents", "resources", "objects")
            
        Returns:
            List of nearby entities
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> nearby_agents = physics.get_nearby_entities((50, 50), radius=10.0)
        """
        ...
    
    def compute_distance(self, pos1: Any, pos2: Any) -> float:
        """Compute distance between two positions.
        
        The distance metric depends on the physics engine:
        - Grid2D: Euclidean distance sqrt((x1-x2)^2 + (y1-y2)^2)
        - Static: Discrete hops or state difference
        - Continuous: Configurable (Euclidean, Manhattan, etc.)
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Distance value (metric depends on implementation)
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> dist = physics.compute_distance((0, 0), (3, 4))
            >>> print(dist)  # 5.0 (Euclidean)
        """
        ...
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get shape of state representation.
        
        Returns:
            Tuple describing state dimensions
            - Grid2D: (width, height)
            - Static: (state_dim,)
            - Continuous: (n_dimensions,)
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> physics.get_state_shape()  # (100, 100)
        """
        ...
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Get observation space for an agent.
        
        Returns a Gymnasium space describing the format of observations
        that agents will receive in this environment.
        
        Args:
            agent_id: Identifier of agent (may affect observation space)
            
        Returns:
            Gymnasium space describing observations
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> obs_space = physics.get_observation_space("agent_1")
            >>> print(obs_space)  # Box(low=0, high=1, shape=(channels, h, w))
        """
        ...
    
    def sample_position(self) -> Any:
        """Sample a random valid position.
        
        Useful for initializing agents and resources at random locations.
        
        Returns:
            Random valid position in this environment
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> pos = physics.sample_position()
            >>> print(pos)  # (45.3, 67.8)
        """
        ...
    
    def update(self, dt: float = 1.0) -> None:
        """Update physics simulation.
        
        Called once per simulation step. Can be used for:
        - Updating spatial indices
        - Physics integration
        - State transitions
        
        Args:
            dt: Time step (default 1.0)
            
        Example:
            >>> physics.update(dt=0.1)  # Update with smaller time step
        """
        ...
    
    def reset(self) -> None:
        """Reset physics state.
        
        Called when environment resets. Should clear all dynamic state
        and return to initial configuration.
        
        Example:
            >>> physics.reset()
        """
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for this physics engine.
        
        Returns:
            Dictionary describing physics configuration
            
        Example:
            >>> physics = Grid2DPhysics(width=100, height=100)
            >>> config = physics.get_config()
            >>> print(config)
            {'type': 'grid_2d', 'width': 100, 'height': 100}
        """
        ...


@runtime_checkable
class IObservationBuilder(Protocol):
    """Protocol for building observations from environment state.
    
    Different physics engines may require different observation formats:
    - Grid2D: Multi-channel 2D grids
    - Static: State vectors
    - Continuous: Feature vectors or sensor readings
    
    This is an optional extension point for custom observation building.
    By default, physics engines can define observation space directly.
    """
    
    def build_observation(
        self, 
        agent_id: str,
        physics_state: Any,
        entities: Dict[str, List[Any]]
    ) -> np.ndarray:
        """Build observation for an agent.
        
        Args:
            agent_id: Agent to build observation for
            physics_state: Current physics state
            entities: Dict of entity lists by type ("agents", "resources", etc.)
            
        Returns:
            Observation array
            
        Example:
            >>> obs = builder.build_observation(
            ...     agent_id="agent_1",
            ...     physics_state=state,
            ...     entities={"agents": [...], "resources": [...]}
            ... )
        """
        ...
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space definition.
        
        Returns:
            Gymnasium space for observations
            
        Example:
            >>> obs_space = builder.get_observation_space()
            >>> print(obs_space)  # Box(low=0, high=1, shape=(n,))
        """
        ...


# Type alias for convenience
PhysicsEngine = IPhysicsEngine
ObservationBuilder = IObservationBuilder
