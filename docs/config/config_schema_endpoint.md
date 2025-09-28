## Configuration Schema API: Backend Support (Issue #356)

### Summary
- **Added schema generator**: `farm/core/config_schema.py` produces a combined, JSON-serializable schema for configuration sections.
- **New API endpoint**: `GET /config/schema` via `server/backend/app/routers/config.py` exposes the schema for client applications.
- **App wiring**: Router included in `server/backend/app/main.py`.
- **Basic test**: `tests/test_config_schema_endpoint.py` validates endpoint structure.

### What’s Included
- **Schema generator** (`farm/core/config_schema.py`):
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

- **API** (`server/backend/app/routers/config.py`):
  - `GET /config/schema` returns a payload like:

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

### How Client Applications Can Use This
- **Fetch schema**: Call `GET /config/schema` on app load or when refreshing settings.
- **Build UI**: For each section, iterate `properties` and render appropriate controls based on `type`, `default`, and `enum`.
- **Hints/validation**:
  - Use `enum` to render select inputs.
  - Use `minimum`/`maximum` when present to set numeric control ranges.
  - Defaults can seed initial field values.

### Design Notes and Limitations
- Dataclass-based fields do not carry numeric range metadata; only defaults and known enums are emitted.
- Known string enums for `SimulationConfig` (e.g., `position_discretization_method`, `db_*` pragmas, `device_preference`) are provided via a curated list.
- `ObservationConfig` includes descriptions and constraints from Pydantic `Field` definitions.
- The generator emphasizes stability and JSON compatibility; complex Python types are surfaced as `object` with sensible defaults.

### Files Changed/Added
- Added: `farm/core/config_schema.py`
- Added: `server/backend/app/routers/config.py`
- Updated: `server/backend/app/main.py` (included config router)
- Added: `tests/test_config_schema_endpoint.py`

### Testing
- Run tests (example):
  - Create a virtualenv and install dependencies, then `pytest -q`.
  - The test `tests/test_config_schema_endpoint.py` checks structure and a few representative fields.

