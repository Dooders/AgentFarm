"""
Hierarchical configuration management system.

This module provides the core HierarchicalConfig class that implements
configuration inheritance with support for global, environment-specific,
and agent-specific configuration layers.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .exceptions import ValidationException

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Hierarchical configuration with inheritance support.
    
    This class implements a three-tier configuration system:
    1. Global configuration (base defaults)
    2. Environment-specific configuration (environment overrides)
    3. Agent-specific configuration (agent-specific overrides)
    
    Configuration values are resolved using a hierarchical lookup where
    more specific configurations override more general ones.
    """
    
    global_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with hierarchical lookup.
        
        The lookup order is:
        1. Agent-specific config (highest priority)
        2. Environment-specific config (medium priority)
        3. Global config (lowest priority)
        4. Default value (if key not found in any layer)
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            Configuration value from the highest priority layer where the key exists,
            or the default value if the key is not found in any layer.
            
        Examples:
            >>> config = HierarchicalConfig(
            ...     global_config={'debug': False, 'timeout': 30},
            ...     environment_config={'debug': True},
            ...     agent_config={'timeout': 60}
            ... )
            >>> config.get('debug')  # Returns True (from environment_config)
            >>> config.get('timeout')  # Returns 60 (from agent_config)
            >>> config.get('missing_key', 'default')  # Returns 'default'
        """
        # Check agent-specific config first (highest priority)
        if key in self.agent_config:
            logger.debug(f"Found '{key}' in agent config: {self.agent_config[key]}")
            return self.agent_config[key]
        
        # Check environment-specific config (medium priority)
        if key in self.environment_config:
            logger.debug(f"Found '{key}' in environment config: {self.environment_config[key]}")
            return self.environment_config[key]
        
        # Fall back to global config (lowest priority)
        if key in self.global_config:
            logger.debug(f"Found '{key}' in global config: {self.global_config[key]}")
            return self.global_config[key]
        
        # Return default if key not found in any layer
        logger.debug(f"Key '{key}' not found in any config layer, using default: {default}")
        return default
    
    def get_nested(self, key_path: str, default: Any = None, separator: str = '.') -> Any:
        """Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the nested key (e.g., 'database.host')
            default: Default value if key path is not found
            separator: Separator character for key path (default: '.')
            
        Returns:
            Configuration value at the specified path, or default if not found.
            
        Examples:
            >>> config = HierarchicalConfig(
            ...     global_config={'database': {'host': 'localhost', 'port': 5432}},
            ...     environment_config={'database': {'host': 'prod-db'}}
            ... )
            >>> config.get_nested('database.host')  # Returns 'prod-db'
            >>> config.get_nested('database.port')  # Returns 5432
            >>> config.get_nested('database.ssl', False)  # Returns False
        """
        keys = key_path.split(separator)
        current_value = self.get(keys[0])
        
        if current_value is None:
            return default
        
        # Navigate through nested dictionaries
        for key in keys[1:]:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                return default
        
        return current_value
    
    def set(self, key: str, value: Any, layer: str = 'agent') -> None:
        """Set configuration value in specified layer.
        
        Args:
            key: Configuration key to set
            value: Value to set
            layer: Configuration layer ('global', 'environment', or 'agent')
            
        Raises:
            ValueError: If layer is not one of the valid options
            
        Examples:
            >>> config = HierarchicalConfig()
            >>> config.set('debug', True, 'environment')
            >>> config.get('debug')  # Returns True
        """
        if layer not in ['global', 'environment', 'agent']:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: global, environment, agent")
        
        if layer == 'global':
            self.global_config[key] = value
        elif layer == 'environment':
            self.environment_config[key] = value
        elif layer == 'agent':
            self.agent_config[key] = value
        
        logger.debug(f"Set '{key}' = {value} in {layer} config")
    
    def set_nested(self, key_path: str, value: Any, layer: str = 'agent', separator: str = '.') -> None:
        """Set nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the nested key
            value: Value to set
            layer: Configuration layer to set the value in
            separator: Separator character for key path
            
        Examples:
            >>> config = HierarchicalConfig()
            >>> config.set_nested('database.host', 'localhost', 'environment')
            >>> config.get_nested('database.host')  # Returns 'localhost'
        """
        keys = key_path.split(separator)
        
        # Get the target dictionary
        if layer == 'global':
            target_dict = self.global_config
        elif layer == 'environment':
            target_dict = self.environment_config
        elif layer == 'agent':
            target_dict = self.agent_config
        else:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: global, environment, agent")
        
        # Navigate to the parent dictionary
        current_dict = target_dict
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        
        # Set the final value
        current_dict[keys[-1]] = value
        logger.debug(f"Set nested '{key_path}' = {value} in {layer} config")
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists in any layer.
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if the key exists in any configuration layer, False otherwise.
        """
        return (key in self.agent_config or 
                key in self.environment_config or 
                key in self.global_config)
    
    def has_nested(self, key_path: str, separator: str = '.') -> bool:
        """Check if nested configuration key exists in any layer.
        
        Args:
            key_path: Dot-separated path to the nested key
            separator: Separator character for key path
            
        Returns:
            True if the key path exists in any configuration layer, False otherwise.
        """
        return self.get_nested(key_path, separator=separator) is not None
    
    def get_all_keys(self) -> List[str]:
        """Get all configuration keys from all layers.
        
        Returns:
            List of all unique configuration keys across all layers.
        """
        all_keys = set()
        all_keys.update(self.global_config.keys())
        all_keys.update(self.environment_config.keys())
        all_keys.update(self.agent_config.keys())
        return sorted(list(all_keys))
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration after applying all overrides.
        
        This method returns a flat dictionary with all configuration values
        resolved according to the hierarchical precedence rules.
        
        Returns:
            Dictionary containing the effective configuration values.
        """
        effective_config = {}
        
        # Start with global config (lowest priority)
        effective_config.update(self.global_config)
        
        # Apply environment overrides (medium priority)
        effective_config.update(self.environment_config)
        
        # Apply agent-specific overrides (highest priority)
        effective_config.update(self.agent_config)
        
        return effective_config
    
    def validate(self, required_keys: Optional[List[str]] = None) -> None:
        """Validate configuration consistency.
        
        Args:
            required_keys: List of keys that must be present in the configuration
            
        Raises:
            ValidationException: If required keys are missing or validation fails
            
        Examples:
            >>> config = HierarchicalConfig()
            >>> config.validate(['simulation_id', 'max_steps', 'environment'])
            ValidationException: Required configuration key 'simulation_id' is missing
        """
        if required_keys is None:
            required_keys = ['simulation_id', 'max_steps', 'environment']
        
        for key in required_keys:
            if not self.has(key):
                raise ValidationException(
                    field=key,
                    value=None,
                    message=f"Required configuration key '{key}' is missing"
                )
        
        logger.debug(f"Configuration validation passed for {len(required_keys)} required keys")
    
    def merge(self, other: 'HierarchicalConfig') -> 'HierarchicalConfig':
        """Merge another HierarchicalConfig into this one.
        
        The merge operation applies the other configuration's layers to this
        configuration, with the other configuration taking precedence.
        
        Args:
            other: Another HierarchicalConfig to merge
            
        Returns:
            New HierarchicalConfig with merged values.
        """
        merged_config = HierarchicalConfig(
            global_config=copy.deepcopy(self.global_config),
            environment_config=copy.deepcopy(self.environment_config),
            agent_config=copy.deepcopy(self.agent_config)
        )
        
        # Merge global config
        merged_config.global_config.update(other.global_config)
        
        # Merge environment config
        merged_config.environment_config.update(other.environment_config)
        
        # Merge agent config
        merged_config.agent_config.update(other.agent_config)
        
        logger.debug("Merged HierarchicalConfig instances")
        return merged_config
    
    def copy(self) -> 'HierarchicalConfig':
        """Create a deep copy of this configuration.
        
        Returns:
            Deep copy of this HierarchicalConfig instance.
        """
        return HierarchicalConfig(
            global_config=copy.deepcopy(self.global_config),
            environment_config=copy.deepcopy(self.environment_config),
            agent_config=copy.deepcopy(self.agent_config)
        )
    
    def clear_layer(self, layer: str) -> None:
        """Clear all configuration values from a specific layer.
        
        Args:
            layer: Configuration layer to clear ('global', 'environment', or 'agent')
            
        Raises:
            ValueError: If layer is not one of the valid options
        """
        if layer not in ['global', 'environment', 'agent']:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: global, environment, agent")
        
        if layer == 'global':
            self.global_config.clear()
        elif layer == 'environment':
            self.environment_config.clear()
        elif layer == 'agent':
            self.agent_config.clear()
        
        logger.debug(f"Cleared {layer} configuration layer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary representation.
        
        Returns:
            Dictionary containing all configuration layers.
        """
        return {
            'global': self.global_config,
            'environment': self.environment_config,
            'agent': self.agent_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchicalConfig':
        """Create HierarchicalConfig from dictionary representation.
        
        Args:
            data: Dictionary containing configuration layers
            
        Returns:
            New HierarchicalConfig instance.
        """
        return cls(
            global_config=data.get('global', {}),
            environment_config=data.get('environment', {}),
            agent_config=data.get('agent', {})
        )
    
    def __repr__(self) -> str:
        """Return string representation of the configuration."""
        return (f"HierarchicalConfig("
                f"global_keys={len(self.global_config)}, "
                f"environment_keys={len(self.environment_config)}, "
                f"agent_keys={len(self.agent_config)})")
    
    def __len__(self) -> int:
        """Return total number of configuration keys across all layers."""
        return len(self.get_all_keys())
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists in any layer."""
        return self.has(key)