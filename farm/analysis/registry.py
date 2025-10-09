"""
Registry for analysis modules.

Provides centralized registration and discovery of analysis modules.
"""

import importlib
from farm.utils.logging import get_logger
from typing import Any, Dict, List, Optional

from farm.analysis.protocols import AnalysisModule
from farm.analysis.exceptions import ModuleNotFoundError, ConfigurationError
from farm.core.services import IConfigService


logger = get_logger(__name__)


class ModuleRegistry:
    """
    Registry for analysis modules.
    
    Provides centralized registration and discovery of all available
    analysis modules in the system.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._modules: Dict[str, AnalysisModule] = {}
    
    def register(self, module: AnalysisModule) -> None:
        """Register an analysis module.
        
        Args:
            module: Module to register
            
        Raises:
            ConfigurationError: If module name conflicts with existing module
        """
        if module.name in self._modules:
            logger.warning(
                f"Module '{module.name}' already registered. Replacing with new instance."
            )
        
        self._modules[module.name] = module
        logger.info(f"Registered analysis module: {module.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a module by name.
        
        Args:
            name: Module name to unregister
        """
        if name in self._modules:
            del self._modules[name]
            logger.info(f"Unregistered analysis module: {name}")
    
    def get(self, name: str) -> AnalysisModule:
        """Get a module by name.
        
        Args:
            name: Module name
            
        Returns:
            The requested module
            
        Raises:
            ModuleNotFoundError: If module doesn't exist
        """
        if name not in self._modules:
            raise ModuleNotFoundError(name, self.get_module_names())
        return self._modules[name]
    
    def get_optional(self, name: str) -> Optional[AnalysisModule]:
        """Get a module by name, returning None if not found.
        
        Args:
            name: Module name
            
        Returns:
            Module or None if not found
        """
        return self._modules.get(name)
    
    def get_module_names(self) -> List[str]:
        """Get list of all registered module names.
        
        Returns:
            Sorted list of module names
        """
        return sorted(self._modules.keys())
    
    def get_all(self) -> Dict[str, AnalysisModule]:
        """Get all registered modules.
        
        Returns:
            Dictionary mapping module names to modules
        """
        return self._modules.copy()
    
    def clear(self) -> None:
        """Clear all registered modules."""
        self._modules.clear()
        logger.info("Cleared all registered modules")
    
    def list_modules(self) -> str:
        """Get formatted string listing all modules.
        
        Returns:
            Formatted string with module information
        """
        if not self._modules:
            return "No modules registered"
        
        lines = ["Registered Analysis Modules:", ""]
        for name in sorted(self._modules.keys()):
            module = self._modules[name]
            lines.append(f"  - {name}: {module.description}")
        
        return "\n".join(lines)


# Global registry instance
registry = ModuleRegistry()


def register_modules(
    config_env_var: str = "FARM_ANALYSIS_MODULES",
    *,
    config_service: IConfigService
) -> int:
    """Register analysis modules from configuration.
    
    Discovery order:
    1. Environment variable FARM_ANALYSIS_MODULES as comma-separated import paths
    2. Fallback to built-in modules if nothing configured
    
    Args:
        config_env_var: Environment variable name to check
        config_service: Configuration service for retrieving module paths
        
    Returns:
        Number of successfully registered modules
    """
    module_paths = config_service.get_analysis_module_paths(config_env_var)
    
    successfully_registered = 0
    
    for path in module_paths:
        try:
            module_path, attr = path.rsplit(".", 1)
        except ValueError:
            logger.warning(
                f"Invalid analysis module path (no attribute specified): {path}"
            )
            continue
        
        try:
            mod = importlib.import_module(module_path)
            instance = getattr(mod, attr)
        except (ImportError, AttributeError) as err:
            logger.warning(f"Failed to import analysis module '{path}': {err}")
            continue
        except Exception as err:
            logger.warning(
                f"Unexpected error loading analysis module '{path}': {err}"
            )
            continue
        
        # Check if instance matches AnalysisModule protocol
        if not _implements_analysis_module_protocol(instance):
            logger.warning(
                f"Configured object at '{path}' doesn't implement AnalysisModule protocol"
            )
            continue
        
        registry.register(instance)
        successfully_registered += 1
    
    # Fallback to built-in modules if nothing registered
    if successfully_registered == 0:
        logger.info("No modules configured, attempting to register built-in modules")
        successfully_registered = _register_builtin_modules()
    
    logger.info(f"Successfully registered {successfully_registered} analysis module(s)")
    return successfully_registered


def _implements_analysis_module_protocol(obj: Any) -> bool:
    """Check if object implements AnalysisModule protocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object implements the protocol
    """
    required_attrs = ['name', 'description', 'get_data_processor', 
                      'get_analysis_functions', 'get_function_groups']
    return all(hasattr(obj, attr) for attr in required_attrs)


def _register_builtin_modules() -> int:
    """Register built-in analysis modules.

    Returns:
        Number of successfully registered modules
    """
    # Organize built-in modules by category for better maintainability
    builtin_modules = [
        # Core analysis modules
        "farm.analysis.genesis.module.genesis_module",
        "farm.analysis.dominance.module.dominance_module",
        "farm.analysis.advantage.module.advantage_module",

        # Agent-focused analysis modules
        "farm.analysis.agents.module.agents_module",
        "farm.analysis.learning.module.learning_module",
        "farm.analysis.social_behavior.module.social_behavior_module",

        # Environment and resource analysis modules
        "farm.analysis.spatial.module.spatial_module",
        "farm.analysis.temporal.module.temporal_module",
        "farm.analysis.resources.module.resources_module",

        # Action and event analysis modules
        "farm.analysis.actions.module.actions_module",
        "farm.analysis.combat.module.combat_module",
        "farm.analysis.significant_events.module.significant_events_module",

        # Comparative and population analysis modules
        "farm.analysis.population.module.population_module",
        "farm.analysis.comparative.module.comparative_module",
    ]
    
    count = 0
    for path in builtin_modules:
        try:
            module_path, attr = path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            instance = getattr(mod, attr)
            
            if _implements_analysis_module_protocol(instance):
                registry.register(instance)
                count += 1
        except Exception as err:
            logger.debug(f"Couldn't register built-in module '{path}': {err}")
            continue
    
    return count


# Convenience functions
def get_module(name: str) -> AnalysisModule:
    """Get a module by name.
    
    Args:
        name: Module name
        
    Returns:
        The requested module
        
    Raises:
        ModuleNotFoundError: If module doesn't exist
    """
    return registry.get(name)


def get_module_names() -> List[str]:
    """Get list of all registered module names.
    
    Returns:
        Sorted list of module names
    """
    return registry.get_module_names()


def get_modules() -> Dict[str, AnalysisModule]:
    """Get all registered modules.
    
    Returns:
        Dictionary mapping module names to modules
    """
    return registry.get_all()


def list_modules() -> str:
    """Get formatted string listing all modules.
    
    Returns:
        Formatted string with module information
    """
    return registry.list_modules()
