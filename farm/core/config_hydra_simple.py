"""
Simple Hydra configuration manager.

This module provides a simplified interface to the Hydra configuration system,
making it easier to use for basic configuration needs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

from .config_hydra import HydraConfigManager, create_hydra_config_manager
from .config_hydra_models import HydraSimulationConfig

logger = logging.getLogger(__name__)


class SimpleHydraConfigManager:
    """Simple wrapper around HydraConfigManager for easier usage."""

    def __init__(
        self,
        config_dir: str = "config_hydra/conf",
        environment: str = "development",
        agent: str = "system_agent",
        overrides: Optional[List[str]] = None
    ):
        """Initialize the simple Hydra configuration manager.

        Args:
            config_dir: Directory containing Hydra configuration files
            environment: Environment name (e.g., 'development', 'staging', 'production')
            agent: Agent type (e.g., 'system_agent', 'independent_agent', 'control_agent')
            overrides: List of configuration overrides
        """
        self.config_manager = create_hydra_config_manager(
            config_dir=config_dir,
            environment=environment,
            agent=agent,
            overrides=overrides
        )

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config_manager.get_config()

    def to_dict(self) -> Dict[str, Any]:
        """Get the current configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        config = self.config_manager.get_config()
        return config_manager.OmegaConf.to_container(config, resolve=True)

    def get_simulation_config(self) -> HydraSimulationConfig:
        """Get the configuration as a HydraSimulationConfig object.

        Returns:
            HydraSimulationConfig object
        """
        return self.config_manager.get_simulation_config()


def create_simple_hydra_config_manager(
    config_dir: str = "config_hydra/conf",
    environment: str = "development",
    agent: str = "system_agent",
    overrides: Optional[List[str]] = None
) -> SimpleHydraConfigManager:
    """Create a simple Hydra configuration manager.

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
