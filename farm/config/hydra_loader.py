"""
Hydra-based configuration loader for Agent Farm.

This module provides configuration loading using Hydra framework, enabling
command-line overrides, config composition, and multi-run capabilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config import SimulationConfig


class HydraConfigLoader:
    """
    Load configurations using Hydra framework.
    
    This loader provides Hydra-based configuration management with support for:
    - Config groups (environments, profiles)
    - Command-line overrides
    - Automatic config composition
    - Multi-run and sweep support (via Hydra decorators)
    
    Example:
        ```python
        loader = HydraConfigLoader()
        config = loader.load_config(
            environment="production",
            profile="benchmark",
            overrides=["simulation_steps=2000"]
        )
        ```
    """

    def __init__(
        self,
        config_path: str = "conf",
        config_name: str = "config",
        version_base: Optional[str] = None,
    ):
        """
        Initialize the Hydra config loader.
        
        Args:
            config_path: Path to the Hydra config directory (relative to workspace root)
            config_name: Name of the main config file (without .yaml extension)
            version_base: Hydra version base (None for latest, or specific version)
        """
        self.config_path = Path(config_path)
        self.config_name = config_name
        self.version_base = version_base
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure Hydra is initialized with the config directory."""
        if not self._initialized:
            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()
            
            # Resolve config path relative to workspace root
            workspace_root = Path(__file__).parent.parent.parent
            config_dir = workspace_root / self.config_path
            
            if not config_dir.exists():
                raise FileNotFoundError(
                    f"Config directory not found: {config_dir}. "
                    f"Make sure Phase 1 of Hydra migration is complete."
                )
            
            # Initialize Hydra with the config directory
            initialize_config_dir(
                config_dir=str(config_dir.absolute()),
                config_name=self.config_name,
                version_base=self.version_base,
            )
            self._initialized = True

    def load_config(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> SimulationConfig:
        """
        Load configuration using Hydra.
        
        Args:
            environment: Environment name (development, production, testing)
            profile: Optional profile name (benchmark, simulation, research)
            overrides: Optional list of Hydra override strings (e.g., ["simulation_steps=200"])
        
        Returns:
            SimulationConfig: Loaded and composed configuration
        
        Raises:
            FileNotFoundError: If config directory or files are missing
            ValueError: If environment or profile is invalid
        """
        self._ensure_initialized()
        
        # Build override list
        override_list = [f"environment={environment}"]
        
        if profile:
            override_list.append(f"profile={profile}")
        else:
            # Explicitly set to null if not provided
            override_list.append("profile=null")
        
        # Add any additional overrides
        if overrides:
            override_list.extend(overrides)
        
        try:
            # Compose config with Hydra
            cfg = compose(
                config_name=self.config_name,
                overrides=override_list,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load Hydra config with environment={environment}, "
                f"profile={profile}: {e}"
            ) from e
        
        # Convert OmegaConf DictConfig to SimulationConfig
        return self._omega_to_simulation_config(cfg)

    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> SimulationConfig:
        """
        Load configuration from a dictionary (bypassing Hydra).
        
        This is useful for testing or when you already have a config dict.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            SimulationConfig: Loaded configuration
        """
        return SimulationConfig.from_dict(config_dict)

    def _omega_to_simulation_config(self, cfg: DictConfig) -> SimulationConfig:
        """
        Convert OmegaConf DictConfig to SimulationConfig.
        
        Args:
            cfg: OmegaConf DictConfig from Hydra
        
        Returns:
            SimulationConfig: Converted configuration object
        """
        # Convert OmegaConf to a regular Python dict
        # resolve=True resolves variable interpolations
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Ensure it's a dict (not a list or other type)
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Expected dict from OmegaConf, got {type(config_dict)}"
            )
        
        # Use existing from_dict method to create SimulationConfig
        return SimulationConfig.from_dict(config_dict)

    def load_config_with_overrides(
        self,
        *override_strings: str,
        environment: str = "development",
        profile: Optional[str] = None,
    ) -> SimulationConfig:
        """
        Load configuration with inline override strings.
        
        Convenience method for loading with overrides without building a list.
        
        Args:
            *override_strings: Hydra override strings (e.g., "simulation_steps=200")
            environment: Environment name
            profile: Optional profile name
        
        Returns:
            SimulationConfig: Loaded configuration
        
        Example:
            ```python
            config = loader.load_config_with_overrides(
                "simulation_steps=200",
                "population.system_agents=50",
                environment="production"
            )
            ```
        """
        return self.load_config(
            environment=environment,
            profile=profile,
            overrides=list(override_strings) if override_strings else None,
        )

    def clear(self) -> None:
        """Clear Hydra initialization (useful for testing)."""
        GlobalHydra.instance().clear()
        self._initialized = False


# Global loader instance for convenience
_global_hydra_loader: Optional[HydraConfigLoader] = None


def get_global_hydra_loader() -> HydraConfigLoader:
    """Get or create the global Hydra config loader instance."""
    global _global_hydra_loader
    if _global_hydra_loader is None:
        _global_hydra_loader = HydraConfigLoader()
    return _global_hydra_loader


def load_hydra_config(
    environment: str = "development",
    profile: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> SimulationConfig:
    """
    Load configuration using Hydra (convenience function).
    
    This is a convenience function that uses the global Hydra loader.
    For more control, use HydraConfigLoader directly.
    
    Args:
        environment: Environment name
        profile: Optional profile name
        overrides: Optional list of override strings
    
    Returns:
        SimulationConfig: Loaded configuration
    
    Example:
        ```python
        from farm.config.hydra_loader import load_hydra_config
        
        config = load_hydra_config(
            environment="production",
            profile="benchmark",
            overrides=["simulation_steps=2000"]
        )
        ```
    """
    loader = get_global_hydra_loader()
    return loader.load_config(
        environment=environment,
        profile=profile,
        overrides=overrides,
    )
