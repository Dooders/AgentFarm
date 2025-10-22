"""
Configuration integration for analysis modules.

This module provides backward compatibility for analysis modules by importing
configurations from the main config system and providing legacy access patterns.
"""

from typing import Optional
from farm.config.config import (
    SpatialAnalysisConfig,
    GenesisAnalysisConfig, 
    AgentAnalysisConfig,
    PopulationAnalysisConfig,
    LearningAnalysisConfig,
    AnalysisGlobalConfig,
    SimulationConfig
)

# Global configuration instances for backward compatibility
# These will be populated from the main SimulationConfig when available
spatial_config: Optional[SpatialAnalysisConfig] = None
genesis_config: Optional[GenesisAnalysisConfig] = None
agent_config: Optional[AgentAnalysisConfig] = None
population_config: Optional[PopulationAnalysisConfig] = None
learning_config: Optional[LearningAnalysisConfig] = None
global_config: Optional[AnalysisGlobalConfig] = None


def initialize_from_simulation_config(sim_config: SimulationConfig):
    """Initialize analysis configs from main simulation configuration.
    
    Args:
        sim_config: Main simulation configuration
    """
    global spatial_config, genesis_config, agent_config, population_config, learning_config, global_config
    
    spatial_config = sim_config.spatial_analysis
    genesis_config = sim_config.genesis_analysis
    agent_config = sim_config.agent_analysis
    population_config = sim_config.population_analysis
    learning_config = sim_config.learning_analysis
    global_config = sim_config.analysis_global


def reset_to_defaults():
    """Reset all configurations to default values."""
    global spatial_config, genesis_config, agent_config, population_config, learning_config, global_config

    spatial_config = SpatialAnalysisConfig()
    genesis_config = GenesisAnalysisConfig()
    agent_config = AgentAnalysisConfig()
    population_config = PopulationAnalysisConfig()
    learning_config = LearningAnalysisConfig()
    global_config = AnalysisGlobalConfig()


def get_config(module_name: str):
    """Get configuration for a specific module.

    Args:
        module_name: Name of the analysis module

    Returns:
        Configuration instance for the module

    Raises:
        ValueError: If module name is not recognized
    """
    # Initialize with defaults if not already set
    if spatial_config is None:
        reset_to_defaults()
    
    config_map = {
        'spatial': spatial_config,
        'genesis': genesis_config,
        'agents': agent_config,
        'population': population_config,
        'learning': learning_config,
        'global': global_config,
    }

    if module_name not in config_map:
        raise ValueError(
            f"Unknown module '{module_name}'. "
            f"Available modules: {', '.join(config_map.keys())}"
        )

    return config_map[module_name]
