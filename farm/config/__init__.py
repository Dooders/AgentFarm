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
from .orchestrator import ConfigurationOrchestrator, get_global_orchestrator, load_config as _load_config_legacy
from .watcher import ConfigWatcher, ReloadableConfig, create_reloadable_config

# Try to import Hydra loader (may not be available if Hydra not installed)
try:
    from .hydra_loader import (
        HydraConfigLoader,
        get_global_hydra_loader,
        load_hydra_config,
    )
    _HYDRA_AVAILABLE = True
except ImportError:
    _HYDRA_AVAILABLE = False
    HydraConfigLoader = None
    get_global_hydra_loader = None
    load_hydra_config = None

def load_config(
    environment: str = "development",
    profile: Optional[str] = None,
    use_hydra: Optional[bool] = None,
    **kwargs
) -> SimulationConfig:
    """
    Load configuration with backward compatibility.
    
    This function provides a unified interface for loading configurations,
    supporting both the legacy config system and Hydra. By default, it uses
    the legacy system unless USE_HYDRA_CONFIG environment variable is set
    or use_hydra=True is explicitly passed.
    
    Args:
        environment: Environment name (development, production, testing)
        profile: Optional profile name (benchmark, simulation, research)
        use_hydra: Whether to use Hydra (None = auto-detect from env var)
        **kwargs: Additional arguments passed to the underlying loader
    
    Returns:
        SimulationConfig: Loaded configuration
    
    Raises:
        ImportError: If use_hydra=True but Hydra is not installed
        ValueError: If Hydra config cannot be loaded
    
    Example:
        ```python
        # Use legacy system (default)
        config = load_config(environment="production", profile="benchmark")
        
        # Force Hydra usage
        config = load_config(environment="production", use_hydra=True)
        
        # Use environment variable
        # export USE_HYDRA_CONFIG=true
        config = load_config(environment="production")
        ```
    """
    # Determine which loader to use
    if use_hydra is None:
        # Auto-detect from environment variable
        use_hydra = os.getenv("USE_HYDRA_CONFIG", "false").lower() == "true"
    
    if use_hydra:
        if not _HYDRA_AVAILABLE:
            raise ImportError(
                "Hydra is not available. Install it with: pip install hydra-core>=1.3.0"
            )
        
        # Use Hydra loader
        loader = get_global_hydra_loader()
        overrides = kwargs.get("overrides")
        return loader.load_config(
            environment=environment,
            profile=profile,
            overrides=overrides,
        )
    else:
        # Use legacy loader
        return _load_config_legacy(
            environment=environment,
            profile=profile,
            **kwargs
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
    # Hydra support (if available)
    *(["HydraConfigLoader", "get_global_hydra_loader", "load_hydra_config"] if _HYDRA_AVAILABLE else []),
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
