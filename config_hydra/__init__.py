"""
Hydra Configuration System

A modern configuration management system using Hydra with Pydantic validation.
"""

from .config_hydra import HydraConfigManager, create_hydra_config_manager
from .config_hydra_simple import SimpleHydraConfigManager, create_simple_hydra_config_manager
from .config_hydra_models import (
    HydraSimulationConfig,
    HydraEnvironmentConfig,
    HydraAgentConfig,
    VisualizationConfig,
    RedisMemoryConfig,
    AgentParameters,
    AgentTypeRatios,
    validate_config_dict,
    validate_environment_config,
    validate_agent_config
)
from .config_schema import generate_combined_config_schema

__all__ = [
    "HydraConfigManager",
    "create_hydra_config_manager",
    "SimpleHydraConfigManager",
    "create_simple_hydra_config_manager",
    "HydraSimulationConfig",
    "HydraEnvironmentConfig",
    "HydraAgentConfig",
    "VisualizationConfig",
    "RedisMemoryConfig",
    "AgentParameters",
    "AgentTypeRatios",
    "validate_config_dict",
    "validate_environment_config",
    "validate_agent_config",
    "generate_combined_config_schema",
]
