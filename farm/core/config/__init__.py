"""
Hierarchical Configuration Management System

This module provides a comprehensive configuration management system with:
- Hierarchical configuration with inheritance
- Environment-specific overrides
- Runtime configuration validation
- Configuration migration system
- Hot-reloading capabilities
"""

from .hierarchical import HierarchicalConfig
from .validation import ConfigurationValidator, ValidationResult, DEFAULT_SIMULATION_SCHEMA
from .environment import EnvironmentConfigManager
from .exceptions import (
    ConfigurationError,
    ValidationException,
    ConfigurationMigrationError,
    ConfigurationLoadError,
    ConfigurationSaveError,
)

__all__ = [
    "HierarchicalConfig",
    "ConfigurationValidator", 
    "ValidationResult",
    "DEFAULT_SIMULATION_SCHEMA",
    "EnvironmentConfigManager",
    "ConfigurationError",
    "ValidationException",
    "ConfigurationMigrationError",
    "ConfigurationLoadError",
    "ConfigurationSaveError",
]