# Configuration Schema Generation: Backend Support

## Summary

- **Schema generator**: `farm/config/schema.py` produces a combined, JSON-serializable schema for configuration sections.
- **Schema function**: `generate_combined_config_schema()` creates structured schema data for client applications.
- **No API endpoint**: Currently, schema generation exists as a utility function but is not exposed via REST API.

## What’s Included

- **Schema generator** (`farm/config/schema.py`):
  - Introspects these sources:
    - `SimulationConfig` (dataclass)
    - `VisualizationConfig` (dataclass)
    - `RedisMemoryConfig` (dataclass)
    - `ObservationConfig` (Pydantic model)
  - Emits a merged schema with:
    - Field `type` (integer, number, boolean, string, object, array)
    - `default` values
    - `enum` choices where known (e.g., storage mode, certain strings)
    - Range constraints for Pydantic fields when available (minimum/maximum)
  - Output is organized into four sections: `simulation`, `visualization`, `redis`, `observation`.

- **Schema Generation Function**:
  - `generate_combined_config_schema()` returns a payload like:

```json
{
  "version": 1,
  "generated_at": "2025-09-25T12:34:56Z",
  "sections": {
    "simulation": { "title": "Simulation", "properties": { /* ... */ } },
    "visualization": { "title": "Visualization", "properties": { /* ... */ } },
    "redis": { "title": "Redis", "properties": { /* ... */ } },
    "observation": {
      "title": "Observation",
      "properties": { /* Pydantic-derived with descriptions and ranges where present */ },
      "enums": { "storage_mode": ["hybrid", "dense"] }
    }
  }
}
```

## How Client Applications Can Use This

- **Generate schema**: Import and call `generate_combined_config_schema()` from `farm.config.schema` on app load or when refreshing settings.
- **Build UI**: For each section, iterate `properties` and render appropriate controls based on `type`, `default`, and `enum`.
- **Hints/validation**:
  - Use `enum` to render select inputs.
  - Use `minimum`/`maximum` when present to set numeric control ranges.
  - Defaults can seed initial field values.

## Usage Example

```python
from farm.config.schema import generate_combined_config_schema

# Get schema for building configuration UI
schema = generate_combined_config_schema()

# Access different sections
simulation_props = schema["sections"]["simulation"]["properties"]
visualization_props = schema["sections"]["visualization"]["properties"]
redis_props = schema["sections"]["redis"]["properties"]
observation_props = schema["sections"]["observation"]["properties"]
```

## Design Notes and Limitations

- Dataclass-based fields do not carry numeric range metadata; only defaults and known enums are emitted.
- Known string enums for `SimulationConfig` (e.g., `position_discretization_method`, `db_*` pragmas, `device_preference`) are provided via a curated list.
- `ObservationConfig` includes descriptions and constraints from Pydantic `Field` definitions.
- The generator emphasizes stability and JSON compatibility; complex Python types are surfaced as `object` with sensible defaults.

## Files Changed/Added

- Added: `farm/config/schema.py`

## Testing

- **Direct function testing**: The schema generation can be tested by importing and calling the function:

  ```python
  from farm.config.schema import generate_combined_config_schema
  schema = generate_combined_config_schema()
  assert "sections" in schema
  assert "simulation" in schema["sections"]
  ```

- **No API endpoint tests**: Since there's currently no REST API endpoint, there are no endpoint-specific tests.
