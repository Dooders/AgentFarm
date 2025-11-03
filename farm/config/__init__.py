"""
Configuration system for Agent Farm.

This package provides comprehensive configuration management including:
- Centralized configuration loading with orchestrator pattern
- Hydra-based configuration management (Phase 2+)
- Validation and error handling with automatic repair
- Caching and performance optimization
- Templating for parameter sweeps
- File watching and reloading
- Monitoring and observability
- Component isolation with broken circular dependencies
"""

import os
from typing import Optional

from .cache import ConfigCache, LazyConfigLoader, OptimizedConfigLoader
from .config import (
    AgentBehaviorConfig,
    EnvironmentConfig,
    PopulationConfig,
    RedisMemoryConfig,
    ResourceConfig,
    SimulationConfig,
    VisualizationConfig,
)
from .monitor import (
    ConfigMetrics,
    ConfigMonitor,
    get_config_system_health,
    get_global_monitor,
    log_config_system_status,
)
from .template import ConfigTemplate, ConfigTemplateManager
from .validation import (
    ConfigurationError,
    ConfigurationValidator,
    SafeConfigLoader,
    ValidationError,
)
from .orchestrator import ConfigurationOrchestrator, get_global_orchestrator
from .watcher import ConfigWatcher, ReloadableConfig, create_reloadable_config

# Import Hydra loader (required)
from .hydra_loader import (
    HydraConfigLoader,
    get_global_hydra_loader,
    load_hydra_config,
)

def load_config(
    environment: str = "development",
    profile: Optional[str] = None,
    **kwargs
) -> SimulationConfig:
    """
    Load configuration using Hydra.
    
    This function loads configurations using the Hydra configuration system.
    Hydra provides advanced features like command-line overrides, config composition,
    and multi-run support.
    
    Args:
        environment: Environment name (development, production, testing)
        profile: Optional profile name (benchmark, simulation, research)
        **kwargs: Additional arguments:
            - overrides: List of override strings (e.g., ["simulation_steps=200"])
    
    Returns:
        SimulationConfig: Loaded configuration
    
    Raises:
        ImportError: If Hydra is not installed
        ValueError: If Hydra config cannot be loaded
    
    Example:
        ```python
        # Basic usage
        config = load_config(environment="production", profile="benchmark")
        
        # With overrides
        config = load_config(
            environment="production",
            overrides=["simulation_steps=2000", "population.system_agents=50"]
        )
        ```
    """
    loader = get_global_hydra_loader()
    overrides = kwargs.get("overrides")
    return loader.load_config(
        environment=environment,
        profile=profile,
        overrides=overrides,
    )


__all__ = [
    # Core configuration classes
    "SimulationConfig",
    "VisualizationConfig",
    "RedisMemoryConfig",
    # Sub-configuration classes
    "AgentBehaviorConfig",
    "EnvironmentConfig",
    "PopulationConfig",
    "ResourceConfig",
    # Caching
    "ConfigCache",
    "LazyConfigLoader",
    "OptimizedConfigLoader",
    # Orchestrator (main entry point)
    "ConfigurationOrchestrator",
    "get_global_orchestrator",
    "load_config",  # Unified loader with Hydra support
    # Hydra support
    "HydraConfigLoader",
    "get_global_hydra_loader",
    "load_hydra_config",
    # Monitoring
    "ConfigMetrics",
    "ConfigMonitor",
    "get_config_system_health",
    "get_global_monitor",
    "log_config_system_status",
    # Templating
    "ConfigTemplate",
    "ConfigTemplateManager",
    # Validation
    "ConfigurationError",
    "ConfigurationValidator",
    "SafeConfigLoader",
    "ValidationError",
    # Watching
    "ConfigWatcher",
    "ReloadableConfig",
    "create_reloadable_config",
]
