"""
Simplified Hydra-based configuration management system.

This module provides a simplified Hydra-based replacement for the custom hierarchical
configuration system, without complex dependencies.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from .config_hydra_models import (
    validate_config_dict,
    validate_environment_config,
    validate_agent_config,
    HydraSimulationConfig
)

logger = logging.getLogger(__name__)


class SimpleHydraConfigManager:
    """Simplified Hydra-based configuration manager.
    
    This class provides a Hydra-based replacement for the custom hierarchical
    configuration system, without complex dependencies.
    """
    
    def __init__(
        self,
        config_dir: str,
        config_name: str = "config",
        environment: Optional[str] = None,
        agent: Optional[str] = None,
        overrides: Optional[List[str]] = None
    ):
        """Initialize the Hydra configuration manager.
        
        Args:
            config_dir: Directory containing Hydra configuration files
            config_name: Name of the main configuration file (without .yaml)
            environment: Environment name (e.g., 'development', 'staging', 'production')
            agent: Agent type (e.g., 'system_agent', 'independent_agent', 'control_agent')
            overrides: List of configuration overrides
        """
        self.config_dir = Path(config_dir)
        self.config_name = config_name
        self.environment = environment or self._detect_environment()
        self.agent = agent or "system_agent"
        self.overrides = overrides or []
        
        # Current configuration
        self._config: Optional[DictConfig] = None
        
        # Initialize Hydra
        self._initialize_hydra()
        
        logger.info(f"SimpleHydraConfigManager initialized for environment: {self.environment}, agent: {self.agent}")
    
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
    
    def _initialize_hydra(self) -> None:
        """Initialize Hydra with the configuration directory."""
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize Hydra
            with initialize_config_dir(
                config_dir=str(self.config_dir),
                version_base=None
            ):
                # Build overrides list
                overrides = []
                
                # Add environment override
                if self.environment:
                    overrides.append(f"environments={self.environment}")
                
                # Always add agent override
                if self.agent:
                    overrides.append(f"agents={self.agent}")
                
                # Add custom overrides
                overrides.extend(self.overrides)
                
                # Compose configuration
                self._config = compose(
                    config_name=self.config_name,
                    overrides=overrides
                )
                
                logger.debug(f"Hydra initialized with overrides: {overrides}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Hydra: {e}")
            raise RuntimeError(f"Failed to initialize Hydra: {e}")
    
    def get_config(self) -> DictConfig:
        """Get the current Hydra configuration.
        
        Returns:
            DictConfig containing the current configuration
        """
        if self._config is None:
            self._initialize_hydra()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config()
        try:
            return OmegaConf.select(config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        config = self.get_config()
        OmegaConf.set(config, key, value)
    
    def update_environment(self, environment: str) -> None:
        """Update the environment and reload configuration.
        
        Args:
            environment: New environment name
        """
        if environment != self.environment:
            self.environment = environment
            self._config = None
            self._initialize_hydra()
            logger.info(f"Environment updated to: {environment}")
    
    def update_agent(self, agent: str) -> None:
        """Update the agent type and reload configuration.
        
        Args:
            agent: New agent type
        """
        if agent != self.agent:
            self.agent = agent
            self._config = None
            self._initialize_hydra()
            logger.info(f"Agent updated to: {agent}")
    
    def reload(self) -> None:
        """Reload the configuration from files.
        
        This method can be called to refresh the configuration
        when files have been modified externally.
        """
        self._config = None
        self._initialize_hydra()
        logger.info("Configuration reloaded")
    
    def add_override(self, override: str) -> None:
        """Add a configuration override and reload.
        
        Args:
            override: Configuration override (e.g., 'max_steps=500')
        """
        if override not in self.overrides:
            self.overrides.append(override)
            self._config = None
            self._initialize_hydra()
            logger.info(f"Added override: {override}")
    
    def remove_override(self, override: str) -> None:
        """Remove a configuration override and reload.
        
        Args:
            override: Configuration override to remove
        """
        if override in self.overrides:
            self.overrides.remove(override)
            self._config = None
            self._initialize_hydra()
            logger.info(f"Removed override: {override}")
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments.
        
        Returns:
            List of environment names
        """
        env_dir = self.config_dir / "environments"
        if not env_dir.exists():
            return []
        
        environments = []
        for env_file in env_dir.glob("*.yaml"):
            environments.append(env_file.stem)
        
        return sorted(environments)
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent types.
        
        Returns:
            List of agent type names
        """
        agent_dir = self.config_dir / "agents"
        if not agent_dir.exists():
            return []
        
        agents = []
        for agent_file in agent_dir.glob("*.yaml"):
            agents.append(agent_file.stem)
        
        return sorted(agents)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate the current configuration using Pydantic models.
        
        Returns:
            Dictionary of validation errors by category
        """
        errors = {}
        
        try:
            config_dict = self.to_dict()
            
            # Use Pydantic validation
            validated_config = validate_config_dict(config_dict)
            logger.debug("Configuration validation passed")
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            errors['pydantic'] = []
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                errors['pydantic'].append(f'{field}: {message}')
        
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            errors['general'] = [f'Validation error: {str(e)}']
        
        return errors
    
    def validate_environment_config(self) -> Dict[str, List[str]]:
        """
        Validate environment-specific configuration.
        
        Returns:
            Dictionary of validation errors by category
        """
        errors = {}
        
        try:
            config_dict = self.to_dict()
            
            # Extract environment-specific fields
            env_fields = [
                'debug', 'verbose_logging', 'max_steps', 'max_population',
                'use_in_memory_db', 'persist_db_on_completion', 'learning_rate',
                'epsilon_start', 'epsilon_min', 'db_pragma_profile', 'db_cache_size_mb'
            ]
            env_config = {k: v for k, v in config_dict.items() if k in env_fields}
            
            # Use Pydantic validation
            validated_config = validate_environment_config(env_config)
            logger.debug("Environment configuration validation passed")
            
        except ValidationError as e:
            logger.error(f"Environment configuration validation failed: {e}")
            errors['environment'] = []
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                errors['environment'].append(f'{field}: {message}')
        
        except Exception as e:
            logger.error(f"Unexpected environment validation error: {e}")
            errors['general'] = [f'Environment validation error: {str(e)}']
        
        return errors
    
    def validate_agent_config(self) -> Dict[str, List[str]]:
        """
        Validate agent-specific configuration.
        
        Returns:
            Dictionary of validation errors by category
        """
        errors = {}
        
        try:
            config_dict = self.to_dict()
            
            # Extract agent-specific fields
            agent_config = {
                'agent_parameters': config_dict.get('agent_parameters', {})
            }
            
            # Use Pydantic validation
            validated_config = validate_agent_config(agent_config)
            logger.debug("Agent configuration validation passed")
            
        except ValidationError as e:
            logger.error(f"Agent configuration validation failed: {e}")
            errors['agent'] = []
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                errors['agent'].append(f'{field}: {message}')
        
        except Exception as e:
            logger.error(f"Unexpected agent validation error: {e}")
            errors['general'] = [f'Agent validation error: {str(e)}']
        
        return errors
    
    def get_validated_config(self) -> HydraSimulationConfig:
        """
        Get the current configuration as a validated Pydantic model.
        
        Returns:
            Validated HydraSimulationConfig instance
            
        Raises:
            ValidationError: If configuration validation fails
        """
        config_dict = self.to_dict()
        return validate_config_dict(config_dict)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        config = self.get_config()
        
        return {
            'environment': self.environment,
            'agent': self.agent,
            'config_dir': str(self.config_dir),
            'available_environments': self.get_available_environments(),
            'available_agents': self.get_available_agents(),
            'overrides': self.overrides.copy(),
            'config_keys': list(OmegaConf.to_container(config, resolve=True).keys()),
            'validation_errors': self.validate_configuration()
        }
    
    def save_config(self, file_path: str) -> None:
        """Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration
        """
        config = self.get_config()
        OmegaConf.save(config, file_path)
        logger.info(f"Configuration saved to: {file_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dictionary containing the configuration
        """
        config = self.get_config()
        return OmegaConf.to_container(config, resolve=True)
    
    def __repr__(self) -> str:
        """Return string representation of the configuration manager."""
        return (f"SimpleHydraConfigManager("
                f"environment='{self.environment}', "
                f"agent='{self.agent}', "
                f"config_dir='{self.config_dir.name}')")


# Convenience function for backward compatibility
def create_simple_hydra_config_manager(
    config_dir: str = "config_hydra/conf",
    environment: Optional[str] = None,
    agent: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> SimpleHydraConfigManager:
    """Create a simple Hydra configuration manager with default settings.
    
    Args:
        config_dir: Directory containing Hydra configuration files
        environment: Environment name
        agent: Agent type
        overrides: List of configuration overrides
        
    Returns:
        SimpleHydraConfigManager instance
    """
    return SimpleHydraConfigManager(
        config_dir=config_dir,
        environment=environment,
        agent=agent,
        overrides=overrides
    )