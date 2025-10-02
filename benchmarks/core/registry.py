"""
Experiment registry and discovery utilities.

Provides a simple global registry mapping experiment slugs to experiment
classes and their metadata. Includes a decorator for registration and
helper methods for listing and instantiation with parameter validation.
"""

import importlib
import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from benchmarks.core.experiments import Experiment


@dataclass
class ExperimentInfo:
    """
    Metadata container for registered experiments.

    Attributes
    ----------
    slug : str
        Unique identifier for the experiment (used in specs and CLI)
    cls : Type[Experiment]
        The experiment class to instantiate
    summary : str
        Brief description of what the experiment does
    tags : List[str]
        Tags for categorizing and filtering experiments
    param_schema : Dict[str, Any]
        JSON Schema-like structure defining parameter validation and defaults
    """

    slug: str
    cls: Type[Experiment]
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    param_schema: Dict[str, Any] = field(default_factory=dict)


class ExperimentRegistry:
    """
    Global registry for experiment discovery and instantiation.

    The registry maintains a mapping from experiment slugs to their metadata and
    classes. It supports automatic discovery of experiments in packages and
    provides parameter validation and defaulting based on JSON Schema.
    """

    def __init__(self) -> None:
        self._slug_to_info: Dict[str, ExperimentInfo] = {}

    def register(
        self,
        slug: str,
        cls: Type[Experiment],
        summary: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        if slug in self._slug_to_info:
            raise ValueError(f"Experiment slug already registered: {slug}")
        info = ExperimentInfo(
            slug=slug,
            cls=cls,
            summary=summary or cls.__doc__ or "",
            tags=tags or [],
            param_schema=getattr(cls, "param_schema", {}) or {},
        )
        self._slug_to_info[slug] = info

    def list(self) -> List[ExperimentInfo]:
        return list(self._slug_to_info.values())

    def get(self, slug: str) -> ExperimentInfo:
        if slug not in self._slug_to_info:
            raise KeyError(f"Unknown experiment: {slug}")
        return self._slug_to_info[slug]

    def create(self, slug: str, params: Optional[Dict[str, Any]] = None) -> Experiment:
        info = self.get(slug)

        # Apply parameter defaults from JSON schema if available
        effective_params: Dict[str, Any] = {}
        schema = info.param_schema or {}
        for key, field in schema.get("properties", {}).items():
            if "default" in field:
                effective_params[key] = field["default"]

        # Override defaults with provided parameters
        if params:
            effective_params.update(params)

        # Validate that all required parameters are present
        for req in schema.get("required", []):
            if req not in effective_params:
                raise ValueError(
                    f"Missing required param '{req}' for experiment '{slug}'"
                )

        # Instantiate experiment with validated parameters
        return info.cls(**effective_params)

    def discover_package(self, package: str) -> None:
        """
        Import all modules in a package to trigger decorator registration.

        This method recursively imports all Python modules in the specified package,
        which causes any @register_experiment decorators to execute and register
        their experiments with this registry.

        Parameters
        ----------
        package : str
            Package name to discover (e.g., "benchmarks.implementations")
        """
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            return

        # Walk through all Python files in the package directory
        base_path = spec.submodule_search_locations[0]
        for root, _, files in os.walk(base_path):
            for f in files:
                # Only import .py files that aren't private modules
                if f.endswith(".py") and not f.startswith("_"):
                    # Convert file path to module name (e.g., "spatial/benchmark.py" -> "spatial.benchmark")
                    module_rel = os.path.relpath(os.path.join(root, f), base_path)
                    module_name = module_rel[:-3].replace(os.sep, ".")
                    # Import the module to trigger @register_experiment decorators
                    importlib.import_module(f"{package}.{module_name}")


REGISTRY = ExperimentRegistry()


def register_experiment(slug: str, summary: str = "", tags: Optional[List[str]] = None):
    """
    Decorator to register an Experiment subclass under a slug.

    This decorator automatically registers experiment classes with the global
    registry, making them discoverable by the CLI and spec system.

    Parameters
    ----------
    slug : str
        Unique identifier for the experiment (used in specs and CLI)
    summary : str, optional
        Brief description of what the experiment does
    tags : List[str], optional
        Tags for categorizing and filtering experiments

    Returns
    -------
    Callable
        Decorator function that registers the experiment class

    Example
    -------
    @register_experiment("my_benchmark", "Tests my system performance", ["performance"])
    class MyBenchmark(Experiment):
        def execute_once(self, context):
            # benchmark implementation
            pass
    """

    def _decorator(cls: Type[Experiment]) -> Type[Experiment]:
        if not inspect.isclass(cls) or not issubclass(cls, Experiment):
            raise TypeError(
                "@register_experiment can only decorate Experiment subclasses"
            )
        REGISTRY.register(slug=slug, cls=cls, summary=summary, tags=tags)
        return cls

    return _decorator
