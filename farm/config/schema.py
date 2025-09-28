import dataclasses
import datetime
from dataclasses import is_dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from pydantic import BaseModel, ValidationError

from .config import (
    SimulationConfig,
    VisualizationConfig,
    RedisMemoryConfig,
)
from farm.core.observations import ObservationConfig, StorageMode


def _python_type_to_schema_type(annotation: Any) -> str:
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
    try:
        if issubclass(annotation, Enum):
            return [e.value for e in annotation]
    except TypeError:
        pass
    return None


def _get_default_from_instance(instance: Any, name: str) -> Any:
    try:
        return getattr(instance, name)
    except Exception:
        return None


def _dataclass_to_properties(dc_cls: type, known_enums: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Dict[str, Any]]:
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
                default_value = f.default_factory()  # type: ignore[misc]
            except Exception:
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


def _pydantic_model_to_properties(model_cls: type[BaseModel]) -> Dict[str, Dict[str, Any]]:
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
            entry["type"] = next((t for t in types if t != "null"), types[0] if types else "object")
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
    simulation_known_enums: Dict[str, List[Any]] = {
        "position_discretization_method": ["floor", "round", "ceil"],
        "db_pragma_profile": ["balanced", "performance", "safety", "memory"],
        "db_synchronous_mode": ["OFF", "NORMAL", "FULL"],
        "db_journal_mode": ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"],
        "device_preference": ["auto", "cpu", "cuda"],
    }

    sim_props = _dataclass_to_properties(SimulationConfig, known_enums=simulation_known_enums)
    # Remove nested sections from simulation section
    for nested in ("visualization", "redis", "observation"):
        sim_props.pop(nested, None)

    vis_props = _dataclass_to_properties(VisualizationConfig)
    redis_props = _dataclass_to_properties(RedisMemoryConfig)
    obs_props = _pydantic_model_to_properties(ObservationConfig)

    return {
        "version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
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
        },
    }

