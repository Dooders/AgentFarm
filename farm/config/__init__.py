"""
Configuration system for Agent Farm.

This package provides comprehensive configuration management including:
- Centralized configuration loading with orchestrator pattern
- Validation and error handling with automatic repair
- Caching and performance optimization
- Templating for parameter sweeps
- File watching and reloading
- Monitoring and observability
- Component isolation with broken circular dependencies
"""

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
from farm.core.agent.config import AgentComponentConfig
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
from .orchestrator import ConfigurationOrchestrator, get_global_orchestrator, load_config
from .watcher import ConfigWatcher, ReloadableConfig, create_reloadable_config

__all__ = [
    # Core configuration classes
    "SimulationConfig",
    "VisualizationConfig",
    "RedisMemoryConfig",
    # Sub-configuration classes
    "AgentBehaviorConfig",
    "AgentComponentConfig",
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
    "load_config",
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
