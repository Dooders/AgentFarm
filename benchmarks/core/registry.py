from __future__ import annotations

"""
Experiment registry and discovery utilities.

Provides a simple global registry mapping experiment slugs to experiment
classes and their metadata. Includes a decorator for registration and
helper methods for listing and instantiation with parameter validation.
"""

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from benchmarks.core.experiments import Experiment


@dataclass
class ExperimentInfo:
    slug: str
    cls: Type[Experiment]
    summary: str = ""
    tags: List[str] = None  # type: ignore[assignment]
    param_schema: Dict[str, Any] = None  # type: ignore[assignment]


class ExperimentRegistry:
    def __init__(self) -> None:
        self._slug_to_info: Dict[str, ExperimentInfo] = {}

    def register(self, slug: str, cls: Type[Experiment], summary: str = "", tags: Optional[List[str]] = None) -> None:
        if slug in self._slug_to_info:
            raise ValueError(f"Experiment slug already registered: {slug}")
        info = ExperimentInfo(slug=slug, cls=cls, summary=summary or cls.__doc__ or "", tags=tags or [], param_schema=getattr(cls, "param_schema", {}) or {})
        self._slug_to_info[slug] = info

    def list(self) -> List[ExperimentInfo]:
        return list(self._slug_to_info.values())

    def get(self, slug: str) -> ExperimentInfo:
        if slug not in self._slug_to_info:
            raise KeyError(f"Unknown experiment: {slug}")
        return self._slug_to_info[slug]

    def create(self, slug: str, params: Optional[Dict[str, Any]] = None) -> Experiment:
        info = self.get(slug)
        # Basic param defaulting from schema if provided
        effective_params: Dict[str, Any] = {}
        schema = info.param_schema or {}
        for key, field in schema.get("properties", {}).items():
            if "default" in field:
                effective_params[key] = field["default"]
        if params:
            effective_params.update(params)
        # Required checks
        for req in schema.get("required", []):
            if req not in effective_params:
                raise ValueError(f"Missing required param '{req}' for experiment '{slug}'")
        return info.cls(effective_params)

    def discover_package(self, package: str) -> None:
        """Import all modules in a package to trigger decorator registration."""
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            return
        base_path = spec.submodule_search_locations[0]
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.endswith(".py") and not f.startswith("_"):
                    module_rel = os.path.relpath(os.path.join(root, f), base_path)
                    module_name = module_rel[:-3].replace(os.sep, ".")
                    importlib.import_module(f"{package}.{module_name}")


REGISTRY = ExperimentRegistry()


def register_experiment(slug: str, summary: str = "", tags: Optional[List[str]] = None):
    """Decorator to register an Experiment subclass under a slug."""

    def _decorator(cls: Type[Experiment]) -> Type[Experiment]:
        if not inspect.isclass(cls) or not issubclass(cls, Experiment):
            raise TypeError("@register_experiment can only decorate Experiment subclasses")
        REGISTRY.register(slug=slug, cls=cls, summary=summary, tags=tags)
        return cls

    return _decorator

