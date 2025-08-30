"""
Registry for analysis modules.
This module provides a central registry for all analysis modules.
"""

import importlib
import os
from typing import Dict, List, Optional

from farm.analysis.base_module import AnalysisModule


class ModuleRegistry:
    """
    Registry for analysis modules.

    This class provides a central registry for all analysis modules.
    It allows modules to be registered and retrieved by name.
    """

    def __init__(self):
        """Initialize the module registry."""
        self._modules = {}

    def register_module(self, module: AnalysisModule) -> None:
        """
        Register an analysis module.

        Parameters
        ----------
        module : AnalysisModule
            The module to register
        """
        self._modules[module.name] = module

    def get_module(self, name: str) -> Optional[AnalysisModule]:
        """
        Get a module by name.

        Parameters
        ----------
        name : str
            Name of the module

        Returns
        -------
        Optional[AnalysisModule]
            The requested module, or None if not found
        """
        return self._modules.get(name)

    def get_module_names(self) -> List[str]:
        """
        Get a list of all registered module names.

        Returns
        -------
        List[str]
            List of module names
        """
        return list(self._modules.keys())

    def get_modules(self) -> Dict[str, AnalysisModule]:
        """
        Get all registered modules.

        Returns
        -------
        Dict[str, AnalysisModule]
            Dictionary mapping module names to modules
        """
        return self._modules


# Create a singleton instance
registry = ModuleRegistry()


def register_modules(config_env_var: str = "FARM_ANALYSIS_MODULES"):
    """
    Register available analysis modules.

    Discovery order:
    1) Environment variable FARM_ANALYSIS_MODULES as comma-separated import paths to module singletons
       e.g., "farm.analysis.dominance.module.dominance_module, farm.analysis.template.module.template_module"
    2) Fallback to built-in dominance module if env is not set.
    """
    module_paths = os.getenv(config_env_var, "").strip()
    if module_paths:
        for path in [p.strip() for p in module_paths.split(",") if p.strip()]:
            try:
                module_path, attr = path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                instance = getattr(mod, attr)
                if instance:
                    registry.register_module(instance)
            except Exception:
                continue
        return

    # Fallback registration
    try:
        from farm.analysis.dominance.module import dominance_module

        registry.register_module(dominance_module)
    except Exception:
        pass


def get_module(name: str) -> Optional[AnalysisModule]:
    """
    Get a module by name.

    Parameters
    ----------
    name : str
        Name of the module

    Returns
    -------
    Optional[AnalysisModule]
        The requested module, or None if not found
    """
    return registry.get_module(name)


def get_module_names() -> List[str]:
    """
    Get a list of all registered module names.

    Returns
    -------
    List[str]
        List of module names
    """
    return registry.get_module_names()


def get_modules() -> Dict[str, AnalysisModule]:
    """
    Get all registered modules.

    Returns
    -------
    Dict[str, AnalysisModule]
        Dictionary mapping module names to modules
    """
    return registry.get_modules()
