"""
Environment-specific configuration management system.

This module provides the EnvironmentConfigManager class that handles
loading and managing environment-specific configuration overrides,
including automatic environment detection and configuration file loading.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .hierarchical import HierarchicalConfig
from .exceptions import ConfigurationLoadError, ConfigurationError

logger = logging.getLogger(__name__)


class EnvironmentConfigManager:
    """Manages environment-specific configuration overrides.
    
    This class handles loading configuration files with environment-specific
    overrides, automatic environment detection, and configuration inheritance.
    """
    
    def __init__(
        self, 
        base_config_path: str, 
        environment: Optional[str] = None,
        config_dir: Optional[str] = None
    ):
        """Initialize environment configuration manager.
        
        Args:
            base_config_path: Path to the base configuration file
            environment: Environment name (e.g., 'development', 'staging', 'production')
                        If None, will be auto-detected from environment variables
            config_dir: Directory containing configuration files
                       If None, will use the directory of base_config_path
        """
        self.base_config_path = Path(base_config_path)
        self.environment = environment or self._detect_environment()
        self.config_dir = Path(config_dir) if config_dir else self.base_config_path.parent
        
        # Configuration file paths
        self.environments_dir = self.config_dir / "environments"
        self.agents_dir = self.config_dir / "agents"
        
        # Loaded configuration
        self._config_hierarchy: Optional[HierarchicalConfig] = None
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"EnvironmentConfigManager initialized for environment: {self.environment}")
    
    def _detect_environment(self) -> str:
        """Detect the current environment from environment variables.
        
        Returns:
            Environment name (default: 'development')
        """
        # Check multiple environment variables in order of preference
        env_vars = [
            'FARM_ENVIRONMENT',
            'ENVIRONMENT', 
            'ENV',
            'NODE_ENV',
            'PYTHON_ENV'
        ]
        
        for env_var in env_vars:
            env_value = os.getenv(env_var)
            if env_value:
                logger.debug(f"Detected environment '{env_value}' from {env_var}")
                return env_value.lower()
        
        # Default environment
        default_env = 'development'
        logger.debug(f"No environment variable found, using default: {default_env}")
        return default_env
    
    def _load_yaml_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration data
            
        Raises:
            ConfigurationLoadError: If the file cannot be loaded or parsed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationLoadError(
                source="file",
                message=f"Configuration file not found: {file_path}",
                file_path=str(file_path)
            )
        
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded configuration from {file_path}")
            return config_data
            
        except yaml.YAMLError as e:
            raise ConfigurationLoadError(
                source="file",
                message=f"Invalid YAML syntax in {file_path}",
                file_path=str(file_path),
                original_error=e
            )
        except Exception as e:
            raise ConfigurationLoadError(
                source="file",
                message=f"Failed to load configuration from {file_path}",
                file_path=str(file_path),
                original_error=e
            )
    
    def _load_json_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            Dictionary containing the configuration data
            
        Raises:
            ConfigurationLoadError: If the file cannot be loaded or parsed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationLoadError(
                source="file",
                message=f"Configuration file not found: {file_path}",
                file_path=str(file_path)
            )
        
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f) or {}
            
            logger.debug(f"Loaded configuration from {file_path}")
            return config_data
            
        except json.JSONDecodeError as e:
            raise ConfigurationLoadError(
                source="file",
                message=f"Invalid JSON syntax in {file_path}",
                file_path=str(file_path),
                original_error=e
            )
        except Exception as e:
            raise ConfigurationLoadError(
                source="file",
                message=f"Failed to load configuration from {file_path}",
                file_path=str(file_path),
                original_error=e
            )
    
    def _load_config_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a file (YAML or JSON).
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary containing the configuration data
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return self._load_yaml_config(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._load_json_config(file_path)
        else:
            # Try YAML first, then JSON
            try:
                return self._load_yaml_config(file_path)
            except ConfigurationLoadError:
                return self._load_json_config(file_path)
    
    def _get_environment_config_path(self) -> Path:
        """Get the path to the environment-specific configuration file.
        
        Returns:
            Path to the environment configuration file
        """
        return self.environments_dir / f"{self.environment}.yaml"
    
    def _get_agent_specific_config_path(self, agent_type: str) -> Path:
        """Get the path to an agent-specific configuration file.
        
        Args:
            agent_type: Type of agent (e.g., 'system_agent', 'independent_agent')
            
        Returns:
            Path to the agent-specific configuration file
        """
        return self.agents_dir / f"{agent_type}.yaml"
    
    def _load_agent_specific_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration if available.
        
        Returns:
            Dictionary containing agent-specific configuration
        """
        agent_config = {}
        
        # Try to load agent-specific configs
        if self.agents_dir.exists():
            for agent_file in self.agents_dir.glob("*.yaml"):
                try:
                    agent_type = agent_file.stem
                    agent_data = self._load_config_file(agent_file)
                    agent_config.update(agent_data)
                    logger.debug(f"Loaded agent-specific config for {agent_type}")
                except ConfigurationLoadError as e:
                    logger.warning(f"Failed to load agent config {agent_file}: {e}")
        
        return agent_config
    
    def _load_config_hierarchy(self) -> HierarchicalConfig:
        """Load configuration with environment-specific overrides.
        
        Returns:
            HierarchicalConfig with all layers loaded
        """
        # Load base configuration
        try:
            base_config = self._load_config_file(self.base_config_path)
        except ConfigurationLoadError as e:
            raise ConfigurationError(f"Failed to load base configuration: {e}")
        
        # Load environment-specific overrides
        env_config = {}
        env_config_path = self._get_environment_config_path()
        if env_config_path.exists():
            try:
                env_config = self._load_config_file(env_config_path)
                logger.info(f"Loaded environment-specific config from {env_config_path}")
            except ConfigurationLoadError as e:
                logger.warning(f"Failed to load environment config {env_config_path}: {e}")
        else:
            logger.debug(f"No environment-specific config found at {env_config_path}")
        
        # Load agent-specific overrides
        agent_config = self._load_agent_specific_config()
        
        return HierarchicalConfig(
            global_config=base_config,
            environment_config=env_config,
            agent_config=agent_config
        )
    
    def get_config_hierarchy(self, force_reload: bool = False) -> HierarchicalConfig:
        """Get the configuration hierarchy.
        
        Args:
            force_reload: If True, reload configuration from files
            
        Returns:
            HierarchicalConfig with all layers loaded
        """
        if self._config_hierarchy is None or force_reload:
            self._config_hierarchy = self._load_config_hierarchy()
        
        return self._config_hierarchy
    
    def get_effective_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Get the effective configuration after applying all overrides.
        
        Args:
            force_reload: If True, reload configuration from files
            
        Returns:
            Dictionary containing the effective configuration values
        """
        config_hierarchy = self.get_config_hierarchy(force_reload)
        return config_hierarchy.get_effective_config()
    
    def get(self, key: str, default: Any = None, force_reload: bool = False) -> Any:
        """Get a configuration value with hierarchical lookup.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            force_reload: If True, reload configuration from files
            
        Returns:
            Configuration value from the highest priority layer where the key exists
        """
        config_hierarchy = self.get_config_hierarchy(force_reload)
        return config_hierarchy.get(key, default)
    
    def get_nested(self, key_path: str, default: Any = None, force_reload: bool = False) -> Any:
        """Get a nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the nested key
            default: Default value if key path is not found
            force_reload: If True, reload configuration from files
            
        Returns:
            Configuration value at the specified path
        """
        config_hierarchy = self.get_config_hierarchy(force_reload)
        return config_hierarchy.get_nested(key_path, default)
    
    def set_environment(self, environment: str) -> None:
        """Change the current environment and reload configuration.
        
        Args:
            environment: New environment name
        """
        if environment != self.environment:
            self.environment = environment
            self._config_hierarchy = None  # Force reload
            logger.info(f"Environment changed to: {environment}")
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments based on configuration files.
        
        Returns:
            List of environment names
        """
        environments = []
        
        if self.environments_dir.exists():
            for env_file in self.environments_dir.glob("*.yaml"):
                environments.append(env_file.stem)
        
        return sorted(environments)
    
    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types based on configuration files.
        
        Returns:
            List of agent type names
        """
        agent_types = []
        
        if self.agents_dir.exists():
            for agent_file in self.agents_dir.glob("*.yaml"):
                agent_types.append(agent_file.stem)
        
        return sorted(agent_types)
    
    def validate_configuration_files(self) -> Dict[str, List[str]]:
        """Validate all configuration files for syntax errors.
        
        Returns:
            Dictionary mapping file paths to lists of validation errors
        """
        validation_results = {}
        
        # Validate base config
        try:
            self._load_config_file(self.base_config_path)
            validation_results[str(self.base_config_path)] = []
        except ConfigurationLoadError as e:
            validation_results[str(self.base_config_path)] = [str(e)]
        
        # Validate environment configs
        if self.environments_dir.exists():
            for env_file in self.environments_dir.glob("*.yaml"):
                try:
                    self._load_config_file(env_file)
                    validation_results[str(env_file)] = []
                except ConfigurationLoadError as e:
                    validation_results[str(env_file)] = [str(e)]
        
        # Validate agent configs
        if self.agents_dir.exists():
            for agent_file in self.agents_dir.glob("*.yaml"):
                try:
                    self._load_config_file(agent_file)
                    validation_results[str(agent_file)] = []
                except ConfigurationLoadError as e:
                    validation_results[str(agent_file)] = [str(e)]
        
        return validation_results
    
    def create_environment_config(self, environment: str, config_data: Dict[str, Any]) -> None:
        """Create a new environment-specific configuration file.
        
        Args:
            environment: Environment name
            config_data: Configuration data to write
        """
        # Ensure environments directory exists
        self.environments_dir.mkdir(parents=True, exist_ok=True)
        
        env_config_path = self.environments_dir / f"{environment}.yaml"
        
        try:
            import yaml
            with open(env_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created environment configuration: {env_config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment config {env_config_path}: {e}")
    
    def create_agent_config(self, agent_type: str, config_data: Dict[str, Any]) -> None:
        """Create a new agent-specific configuration file.
        
        Args:
            agent_type: Agent type name
            config_data: Configuration data to write
        """
        # Ensure agents directory exists
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        agent_config_path = self.agents_dir / f"{agent_type}.yaml"
        
        try:
            import yaml
            with open(agent_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created agent configuration: {agent_config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create agent config {agent_config_path}: {e}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration setup.
        
        Returns:
            Dictionary containing configuration summary information
        """
        config_hierarchy = self.get_config_hierarchy()
        
        return {
            'environment': self.environment,
            'base_config_path': str(self.base_config_path),
            'config_dir': str(self.config_dir),
            'available_environments': self.get_available_environments(),
            'available_agent_types': self.get_available_agent_types(),
            'config_layers': {
                'global_keys': len(config_hierarchy.global_config),
                'environment_keys': len(config_hierarchy.environment_config),
                'agent_keys': len(config_hierarchy.agent_config),
                'total_unique_keys': len(config_hierarchy.get_all_keys())
            },
            'effective_config_keys': len(config_hierarchy.get_effective_config())
        }
    
    def __repr__(self) -> str:
        """Return string representation of the configuration manager."""
        return (f"EnvironmentConfigManager("
                f"environment='{self.environment}', "
                f"base_config='{self.base_config_path.name}', "
                f"config_dir='{self.config_dir.name}')")