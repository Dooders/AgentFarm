"""
Registry for analysis modules.
This module provides a central registry for all analysis modules.
"""

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


def register_modules():
    """
    Register all available analysis modules.
    This function should be called once at application startup.
    """
    # Import modules here to avoid circular imports
    from farm.analysis.dominance.module import dominance_module

    # Register modules
    registry.register_module(dominance_module)

    # Add more modules as they become available
    # registry.register_module(reproduction_module)
    # registry.register_module(resource_module)
    # etc.


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
