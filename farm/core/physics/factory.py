"""Physics engine factory for creating physics engines from configuration.

This module provides a factory function to create physics engines based on
configuration, making it easy to swap between different physics implementations.
"""

from typing import Any, Optional

from farm.config import SimulationConfig
from farm.core.physics.interface import IPhysicsEngine
from farm.core.physics.grid_2d import Grid2DPhysics
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def create_physics_engine(
    config: SimulationConfig,
    seed: Optional[int] = None
) -> IPhysicsEngine:
    """Create a physics engine from configuration.
    
    Args:
        config: Simulation configuration containing physics settings
        seed: Random seed for deterministic behavior (optional)
        
    Returns:
        Physics engine implementing IPhysicsEngine protocol
        
    Raises:
        ValueError: If physics type is not supported
        AttributeError: If required configuration is missing
        
    Example:
        >>> from farm.config import SimulationConfig
        >>> from farm.core.physics import create_physics_engine
        >>> 
        >>> config = SimulationConfig(
        ...     environment=EnvironmentConfig(width=100, height=100)
        ... )
        >>> physics = create_physics_engine(config, seed=42)
        >>> print(physics.get_config())
        {'type': 'grid_2d', 'width': 100, 'height': 100, ...}
    """
    # Use seed from config if not provided
    if seed is None:
        seed = getattr(config, 'seed', None)
    
    # Get physics configuration from environment config
    if hasattr(config, 'environment') and config.environment:
        env_config = config.environment
        
        # Check if physics config is specified
        if hasattr(env_config, 'physics') and env_config.physics:
            physics_type = getattr(env_config.physics, 'type', 'grid_2d')
        else:
            # Default to grid_2d and extract dimensions from environment config
            physics_type = 'grid_2d'
    else:
        # Fallback to grid_2d with default dimensions
        physics_type = 'grid_2d'
    
    # Create physics engine based on type
    if physics_type == 'grid_2d':
        return _create_grid_2d_physics(config, seed)
    else:
        raise ValueError(f"Unsupported physics type: {physics_type}")


def _create_grid_2d_physics(config: SimulationConfig, seed: Optional[int]) -> Grid2DPhysics:
    """Create a Grid2D physics engine from configuration.
    
    Args:
        config: Simulation configuration
        seed: Random seed
        
    Returns:
        Grid2D physics engine
        
    Raises:
        AttributeError: If required configuration is missing
    """
    # Extract dimensions from environment config
    if hasattr(config, 'environment') and config.environment:
        env_config = config.environment
        width = getattr(env_config, 'width', 100)
        height = getattr(env_config, 'height', 100)
        
        # Get physics-specific config if available
        if hasattr(env_config, 'physics') and env_config.physics:
            physics_config = env_config.physics
            # Override dimensions if specified in physics config
            width = getattr(physics_config, 'width', width)
            height = getattr(physics_config, 'height', height)
            spatial_config = getattr(physics_config, 'spatial_config', None)
            observation_config = getattr(physics_config, 'observation_config', None)
        else:
            spatial_config = None
            observation_config = None
    else:
        # Default dimensions
        width = 100
        height = 100
        spatial_config = None
        observation_config = None
    
    logger.info(
        "Creating Grid2D physics engine",
        width=width,
        height=height,
        seed=seed,
        has_spatial_config=spatial_config is not None,
        has_observation_config=observation_config is not None
    )
    
    return Grid2DPhysics(
        width=width,
        height=height,
        spatial_config=spatial_config,
        observation_config=observation_config,
        seed=seed
    )


def get_available_physics_types() -> list[str]:
    """Get list of available physics engine types.
    
    Returns:
        List of supported physics engine type names
    """
    return ["grid_2d"]


def validate_physics_config(config: Any) -> bool:
    """Validate physics configuration.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check if config has required attributes
        if not hasattr(config, 'environment'):
            return False
        
        env_config = config.environment
        if not hasattr(env_config, 'width') or not hasattr(env_config, 'height'):
            return False
        
        # Validate dimensions
        width = env_config.width
        height = env_config.height
        if not isinstance(width, int) or not isinstance(height, int):
            return False
        if width <= 0 or height <= 0:
            return False
        
        # Check physics-specific config if present
        if hasattr(env_config, 'physics') and env_config.physics:
            physics_config = env_config.physics
            physics_type = getattr(physics_config, 'type', 'grid_2d')
            
            if physics_type not in get_available_physics_types():
                return False
        
        return True
        
    except Exception as e:
        logger.warning("Physics config validation failed", error=str(e))
        return False
