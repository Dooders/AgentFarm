"""
Configuration schema generation utilities.

This module provides functions to introspect configuration dataclasses and Pydantic models,
generating JSON schemas for UI generation, validation, and documentation purposes.
"""

import dataclasses
import datetime
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from pydantic import BaseModel, ValidationError

from farm.core.observations import ObservationConfig, StorageMode

from .config import RedisMemoryConfig, SimulationConfig, VisualizationConfig
from farm.core.agent.config import AgentComponentConfig


def _python_type_to_schema_type(annotation: Any) -> str:
    """Convert Python type annotation to JSON schema type string.

    Args:
        annotation: Python type annotation to convert

    Returns:
        str: JSON schema type ("integer", "number", "boolean", "string", "array", "object")
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list or origin is List:
        return "array"
    if origin is dict or origin is Dict:
        return "object"
    if origin is tuple or origin is Tuple:
        return "array"
    if origin is Optional:
        return _python_type_to_schema_type(args[0]) if args else "string"

    if annotation in (int, Optional[int]):
        return "integer"
    if annotation in (float, Optional[float]):
        return "number"
    if annotation in (bool, Optional[bool]):
        return "boolean"
    if annotation in (str, Optional[str]):
        return "string"

    # Enums
    try:
        if issubclass(annotation, Enum):
            return "string"
    except TypeError:
        pass

    # Fallback
    return "object"


def _enum_values(annotation: Any) -> Optional[List[Any]]:
    """Extract enum values from an Enum type annotation.

    Args:
        annotation: Type annotation that may be an Enum

    Returns:
        Optional[List[Any]]: List of enum values, or None if not an Enum
    """
    try:
        if issubclass(annotation, Enum):
            return [e.value for e in annotation]
    except TypeError:
        pass
    return None


def _get_default_from_instance(instance: Any, name: str) -> Any:
    """Safely get an attribute value from an instance.

    Args:
        instance: Object instance to get attribute from
        name: Attribute name to retrieve

    Returns:
        Any: Attribute value, or None if attribute doesn't exist or can't be accessed
    """
    try:
        return getattr(instance, name)
    except Exception:
        return None


def _dataclass_to_properties(
    dc_cls: type, known_enums: Optional[Dict[str, List[Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """Convert a dataclass to JSON schema properties dictionary.

    Args:
        dc_cls: Dataclass type to introspect
        known_enums: Optional mapping of field names to enum values for fields that should be treated as enums

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping field names to schema property definitions
    """
    props: Dict[str, Dict[str, Any]] = {}

    for f in fields(dc_cls):
        # Skip nested dataclass fields; these are split into their own sections
        if is_dataclass(f.type):
            continue

        # Determine default without instantiating the dataclass
        default_value = None
        if f.default is not dataclasses.MISSING:
            default_value = f.default
        elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:  # type: ignore[attr-defined]
            try:
                # Call the factory function to get the actual default value
                default_value = f.default_factory()  # type: ignore[misc]
            except Exception:
                # Factory function failed, use None as fallback
                default_value = None

        schema_entry: Dict[str, Any] = {
            "type": _python_type_to_schema_type(f.type),
            "default": default_value,
        }

        # Add enums if known by name
        if known_enums and f.name in known_enums:
            schema_entry["enum"] = list(known_enums[f.name])

        # Add enums if type itself is Enum
        enum_vals = _enum_values(f.type)
        if enum_vals:
            schema_entry["enum"] = enum_vals

        props[f.name] = schema_entry

    return props


def _pydantic_model_to_properties(
    model_cls: type[BaseModel],
) -> Dict[str, Dict[str, Any]]:
    """Convert a Pydantic model to JSON schema properties dictionary.

    Args:
        model_cls: Pydantic model type to introspect

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping field names to schema property definitions
    """
    schema = model_cls.model_json_schema()
    properties = schema.get("properties", {})
    result: Dict[str, Dict[str, Any]] = {}

    # Build a default instance to pull resolved defaults when missing
    default_instance = None
    try:
        default_instance = model_cls()  # type: ignore[call-arg]
    except (TypeError, ValueError, ValidationError):
        # Required args missing or validation-time errors; skip instance-based defaults
        default_instance = None

    for name, meta in properties.items():
        entry: Dict[str, Any] = {}
        # Normalize type
        if "type" in meta:
            entry["type"] = meta["type"]
        elif "anyOf" in meta:
            # Optional fields often appear as anyOf [type,null]
            types = [t.get("type") for t in meta["anyOf"] if isinstance(t, dict)]
            entry["type"] = next(
                (t for t in types if t != "null"), types[0] if types else "object"
            )
        else:
            entry["type"] = "object"

        # Defaults
        if "default" in meta:
            entry["default"] = meta["default"]
        elif default_instance is not None:
            entry["default"] = getattr(default_instance, name, None)

        # Description
        if "description" in meta:
            entry["description"] = meta["description"]

        # Enum
        if "enum" in meta:
            entry["enum"] = meta["enum"]

        # Ranges/constraints
        for key_src, key_dst in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("exclusiveMinimum", "exclusiveMinimum"),
            ("exclusiveMaximum", "exclusiveMaximum"),
        ):
            if key_src in meta:
                entry[key_dst] = meta[key_src]

        result[name] = entry

    return result


def generate_combined_config_schema() -> Dict[str, Any]:
    """Generate a combined configuration schema for the Electron explorer.

    Sections:
      - simulation: top-level simulation parameters (excludes nested sections)
      - visualization: nested visualization parameters
      - redis: nested Redis memory parameters
      - observation: observation system parameters
    """

    # Known enum-like choices present as strings in dataclasses
    # These fields use string values but should be treated as enums in the UI
    simulation_known_enums: Dict[str, List[Any]] = {
        "position_discretization_method": ["floor", "round", "ceil"],
        "db_pragma_profile": ["balanced", "performance", "safety", "memory"],
        "db_synchronous_mode": ["OFF", "NORMAL", "FULL"],
        "db_journal_mode": ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"],
        "device_preference": ["auto", "cpu", "cuda"],
    }

    sim_props = _dataclass_to_properties(
        SimulationConfig, known_enums=simulation_known_enums
    )
    # Remove nested sections from simulation section to avoid duplication
    # These are handled as separate top-level sections in the schema
    for nested in ("visualization", "redis", "observation"):
        sim_props.pop(nested, None)

    vis_props = _dataclass_to_properties(VisualizationConfig)
    redis_props = _dataclass_to_properties(RedisMemoryConfig)
    obs_props = _pydantic_model_to_properties(ObservationConfig)
    agent_props = _dataclass_to_properties(AgentComponentConfig)

    return {
        "version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "sections": {
            "simulation": {
                "title": "Simulation",
                "properties": sim_props,
            },
            "visualization": {
                "title": "Visualization",
                "properties": vis_props,
            },
            "redis": {
                "title": "Redis",
                "properties": redis_props,
            },
            "observation": {
                "title": "Observation",
                "properties": obs_props,
                "enums": {
                    "storage_mode": [e.value for e in StorageMode],
                },
            },
            "agent": {
                "title": "Agent Components",
                "properties": agent_props,
            },
        },
    }
