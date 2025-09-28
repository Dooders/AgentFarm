"""
Simplified bridge between Hydra configuration system and existing SimulationConfig.

This module provides a bridge that allows the existing SimulationConfig class
to work with the new Hydra-based configuration system, without requiring
complex dependencies like torch.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .config_hydra_simple import SimpleHydraConfigManager

logger = logging.getLogger(__name__)


@dataclass
class SimpleVisualizationConfig:
    """Simplified visualization configuration."""
    canvas_size: tuple = (400, 400)
    padding: int = 20
    background_color: str = "black"
    max_animation_frames: int = 5
    animation_min_delay: int = 50
    max_resource_amount: int = 30
    resource_size: int = 2
    agent_radius_scale: int = 2
    birth_radius_scale: int = 4
    death_mark_scale: float = 1.5
    min_font_size: int = 10
    font_scale_factor: int = 40
    font_family: str = "arial"


@dataclass
class SimpleRedisMemoryConfig:
    """Simplified Redis memory configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    environment: str = "development"


@dataclass
class SimpleSimulationConfig:
    """
    Simplified SimulationConfig that works with Hydra configuration.
    
    This provides the same interface as the original SimulationConfig
    but without complex dependencies.
    """
    # Environment settings
    width: int = 100
    height: int = 100
    
    # Position discretization settings
    position_discretization_method: str = "floor"
    use_bilinear_interpolation: bool = True
    
    # Agent settings
    system_agents: int = 10
    independent_agents: int = 10
    control_agents: int = 10
    
    # Simulation settings
    max_steps: int = 1000
    max_population: int = 100
    simulation_id: str = "simulation"
    
    # Database settings
    use_in_memory_db: bool = False
    in_memory_db_memory_limit_mb: int = 1000
    persist_db_on_completion: bool = True
    db_pragma_profile: str = "balanced"
    db_cache_size_mb: int = 100
    
    # Learning parameters
    learning_rate: float = 0.01
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    
    # Debug settings
    debug: bool = False
    verbose_logging: bool = False
    
    # Nested configurations
    visualization: SimpleVisualizationConfig = field(default_factory=SimpleVisualizationConfig)
    redis: SimpleRedisMemoryConfig = field(default_factory=SimpleRedisMemoryConfig)
    agent_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result


class HydraSimulationConfig(SimpleSimulationConfig):
    """
    Bridge class that converts Hydra configuration to SimulationConfig format.
    
    This allows existing simulation code to work with the new Hydra configuration
    system without requiring changes to the simulation logic.
    """
    
    def __init__(self, hydra_config_manager: SimpleHydraConfigManager):
        """
        Initialize SimulationConfig from Hydra configuration manager.
        
        Args:
            hydra_config_manager: Hydra configuration manager instance
        """
        # Get configuration dictionary from Hydra
        config_dict = hydra_config_manager.to_dict()
        
        # Convert Hydra config to SimulationConfig format
        simulation_config_dict = self._convert_hydra_to_simulation_config(config_dict)
        
        # Initialize SimulationConfig with converted data
        super().__init__(**simulation_config_dict)
        
        # Store reference to Hydra manager for future use
        self._hydra_manager = hydra_config_manager
        
        logger.info("Created HydraSimulationConfig from Hydra configuration")
    
    def _convert_hydra_to_simulation_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Hydra configuration dictionary to SimulationConfig format.
        
        Args:
            config_dict: Hydra configuration dictionary
            
        Returns:
            Dictionary in SimulationConfig format
        """
        # Create base configuration dictionary
        simulation_config = {}
        
        # Map Hydra configuration keys to SimulationConfig attributes
        key_mapping = {
            # Environment settings
            'width': 'width',
            'height': 'height',
            'position_discretization_method': 'position_discretization_method',
            'use_bilinear_interpolation': 'use_bilinear_interpolation',
            
            # Agent settings
            'system_agents': 'system_agents',
            'independent_agents': 'independent_agents',
            'control_agents': 'control_agents',
            
            # Simulation settings
            'max_steps': 'max_steps',
            'max_population': 'max_population',
            'simulation_id': 'simulation_id',
            
            # Database settings
            'use_in_memory_db': 'use_in_memory_db',
            'in_memory_db_memory_limit_mb': 'in_memory_db_memory_limit_mb',
            'persist_db_on_completion': 'persist_db_on_completion',
            'db_pragma_profile': 'db_pragma_profile',
            'db_cache_size_mb': 'db_cache_size_mb',
            
            # Learning parameters
            'learning_rate': 'learning_rate',
            'epsilon_start': 'epsilon_start',
            'epsilon_min': 'epsilon_min',
            'epsilon_decay': 'epsilon_decay',
            
            # Debug settings
            'debug': 'debug',
            'verbose_logging': 'verbose_logging',
        }
        
        # Copy mapped values
        for hydra_key, sim_key in key_mapping.items():
            if hydra_key in config_dict:
                simulation_config[sim_key] = config_dict[hydra_key]
        
        # Handle nested configurations
        if 'visualization' in config_dict:
            # Only use compatible fields
            viz_config = config_dict['visualization']
            compatible_viz = {k: v for k, v in viz_config.items() 
                            if k in SimpleVisualizationConfig.__annotations__}
            simulation_config['visualization'] = SimpleVisualizationConfig(**compatible_viz)
        
        if 'redis' in config_dict:
            # Only use compatible fields
            redis_config = config_dict['redis']
            compatible_redis = {k: v for k, v in redis_config.items() 
                              if k in SimpleRedisMemoryConfig.__annotations__}
            simulation_config['redis'] = SimpleRedisMemoryConfig(**compatible_redis)
        
        # Handle agent parameters
        if 'agent_parameters' in config_dict:
            simulation_config['agent_parameters'] = config_dict['agent_parameters']
        
        # Set default values for any missing required fields
        self._set_defaults(simulation_config)
        
        return simulation_config
    
    def _set_defaults(self, config_dict: Dict[str, Any]) -> None:
        """
        Set default values for any missing required fields.
        
        Args:
            config_dict: Configuration dictionary to update
        """
        defaults = {
            'width': 100,
            'height': 100,
            'position_discretization_method': 'floor',
            'use_bilinear_interpolation': True,
            'system_agents': 10,
            'independent_agents': 10,
            'control_agents': 10,
            'max_steps': 1000,
            'max_population': 100,
            'simulation_id': 'hydra-simulation',
            'use_in_memory_db': False,
            'in_memory_db_memory_limit_mb': 1000,
            'persist_db_on_completion': True,
            'db_pragma_profile': 'balanced',
            'db_cache_size_mb': 100,
            'learning_rate': 0.01,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'debug': False,
            'verbose_logging': False,
        }
        
        for key, default_value in defaults.items():
            if key not in config_dict:
                config_dict[key] = default_value
    
    def get_hydra_manager(self) -> SimpleHydraConfigManager:
        """
        Get the underlying Hydra configuration manager.
        
        Returns:
            Hydra configuration manager instance
        """
        return self._hydra_manager
    
    def update_from_hydra(self) -> None:
        """
        Update SimulationConfig from current Hydra configuration.
        
        This is useful when the Hydra configuration has been updated
        (e.g., through hot-reloading) and you want to sync the SimulationConfig.
        """
        config_dict = self._hydra_manager.to_dict()
        simulation_config_dict = self._convert_hydra_to_simulation_config(config_dict)
        
        # Update attributes
        for key, value in simulation_config_dict.items():
            setattr(self, key, value)
        
        logger.info("Updated SimulationConfig from Hydra configuration")
    
    @classmethod
    def from_hydra_manager(cls, hydra_config_manager: SimpleHydraConfigManager) -> 'HydraSimulationConfig':
        """
        Create HydraSimulationConfig from Hydra configuration manager.
        
        Args:
            hydra_config_manager: Hydra configuration manager instance
            
        Returns:
            HydraSimulationConfig instance
        """
        return cls(hydra_config_manager)
    
    @classmethod
    def from_hydra_config(
        cls,
        config_dir: str = "/workspace/config_hydra/conf",
        environment: str = "development",
        agent: str = "system_agent",
        overrides: Optional[list] = None
    ) -> 'HydraSimulationConfig':
        """
        Create HydraSimulationConfig directly from Hydra configuration.
        
        Args:
            config_dir: Hydra configuration directory
            environment: Environment to use
            agent: Agent type to use
            overrides: Configuration overrides
            
        Returns:
            HydraSimulationConfig instance
        """
        from .config_hydra_simple import create_simple_hydra_config_manager
        
        hydra_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment=environment,
            agent=agent,
            overrides=overrides or []
        )
        
        return cls(hydra_manager)


def create_simulation_config_from_hydra(
    config_dir: str = "/workspace/config_hydra/conf",
    environment: str = "development",
    agent: str = "system_agent",
    overrides: Optional[list] = None
) -> HydraSimulationConfig:
    """
    Convenience function to create SimulationConfig from Hydra configuration.
    
    Args:
        config_dir: Hydra configuration directory
        environment: Environment to use
        agent: Agent type to use
        overrides: Configuration overrides
        
    Returns:
        HydraSimulationConfig instance
    """
    return HydraSimulationConfig.from_hydra_config(
        config_dir=config_dir,
        environment=environment,
        agent=agent,
        overrides=overrides
    )


# Backward compatibility alias
SimulationConfigFromHydra = HydraSimulationConfig