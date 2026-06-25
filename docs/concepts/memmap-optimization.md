## Memory-Mapped State Storage

### Overview

AgentFarm supports on-disk, memory-mapped storage for large dense state
representations using NumPy `memmap`. This dramatically reduces resident
memory usage for big worlds while still allowing fast localized window
reads.

A single configuration controls every memmap-backed structure:

| Memmap target | Toggle | Backing class |
|---------------|--------|---------------|
| Resource grid | `MemmapConfig.use_for_resources` | `ResourceManager` |
| OBSTACLES, TERRAIN_COST, VISIBILITY | `MemmapConfig.use_for_environmental` | `EnvironmentalGridManager` |
| DAMAGE_HEAT, TRAILS, ALLY_SIGNAL | `MemmapConfig.use_for_temporal` | `TemporalGridManager` |

All three managers share a generic engine,
`farm.core.memmap_manager.MemmapManager`, which owns memmap lifecycle
(create / window / flush / cleanup) and embeds the OS process id and
optional `simulation_id` in filenames so concurrent simulations on the
same host do not collide.

### Configuration

`MemmapConfig` lives at `SimulationConfig.memmap`:

```python
from farm.config import SimulationConfig, MemmapConfig

cfg = SimulationConfig(
    memmap=MemmapConfig(
        directory="/var/tmp/sims",   # None -> system temp
        dtype="float32",
        mode="w+",                    # "r+" reuses an existing file
        delete_on_close=False,        # remove .dat files on Environment.close()
        use_for_resources=True,
        use_for_environmental=True,
        use_for_temporal=True,
    ),
)
```

YAML / JSON configs use the same nested layout (`memmap:` block) and the
existing dot-notation parser also accepts `memmap.use_for_resources`,
`memmap.directory`, etc.

### What changed

- **`farm/core/memmap_manager.py`** – generic `MemmapManager` shared by
  every memmap-backed grid. Provides `create`, `get`, `get_window`,
  `flush`, `close`, `close_all`, plus convenience helpers for in-place
  updates and decay.
- **`farm/core/environment_grids.py`** – `EnvironmentalGridManager` holds
  `(H, W)` world layers for `OBSTACLES`, `TERRAIN_COST`, and
  `VISIBILITY`. Falls back to in-RAM ndarrays when memmap is disabled so
  callers do not branch on backend.
- **`farm/core/temporal_grids.py`** – `TemporalGridManager` holds
  `(H, W)` world grids for `DAMAGE_HEAT`, `TRAILS`, and `ALLY_SIGNAL`,
  with per-channel `apply_decay()` driven by the simulation's
  `ObservationConfig.gamma_*` factors.
- **`farm/core/resource_manager.py`** – refactored to compose
  `MemmapManager` instead of duplicating memmap logic.
- **`farm/core/environment.py`** – instantiates the new grid managers,
  exposes `set_environmental_layer()` / `deposit_temporal_events()`,
  threads windows from those grids into agent observations, and decays
  the temporal grids each `update()` tick. Temporal channels are passed
  to the observation pipeline via the dense `world_layers` slot rather
  than as sparse event lists, eliminating the
  dense → sparse → dense round-trip the original implementation paid in
  every `observe()` call.
- **`farm/core/observations.py`** – tracks `world_driven_channels` per
  tick so DYNAMIC channels backed by an externally-decayed world grid
  skip the per-agent decay step (the world grid already decayed at the
  world level).
- **`farm/core/channels.py`** – `TransientEventHandler` is now
  dual-mode: when a `world_layers[<channel_name>]` tensor is present it
  takes the same dense path as `WorldLayerHandler`; otherwise it falls
  back to the legacy sparse-event API.
- **`farm/config/config.py`** – adds `MemmapConfig` and wires it into
  `SimulationConfig.memmap` (with `to_dict` / `from_dict` support).

### Performance notes

Two micro-optimizations make the memmap path competitive with in-RAM
storage:

1. **Plain `ndarray` views** – `MemmapManager.create()` caches
   `np.asarray(memmap)` alongside the memmap. `get_window` slices the
   plain view, avoiding the per-call dispatch cost of the
   `numpy.memmap` subclass while still reusing the same underlying
   buffer for `flush()`.
2. **Fast in-bounds path** – `get_window` skips the `np.zeros`
   allocation and pad fill for the common case where the requested
   window is fully inside the grid (most agent observations away from
   the world edge). It just copies the slice into a fresh ndarray.

`TemporalGridManager` also tracks a sticky `has_any_data` flag per
channel. When no events have ever been deposited the per-tick window
read and dense channel write are short-circuited, so simulations that
never use a temporal channel pay nothing for it.

### Acceptance criteria mapping (issue #426)

| Criterion | Where verified |
|-----------|----------------|
| Environmental grids (OBSTACLES, TERRAIN_COST, VISIBILITY) use memmap when enabled | `EnvironmentalGridManager`, `tests/test_environmental_grids.py`, `tests/test_memmap_environment_integration.py` |
| Agent observations can use memmap-backed global layers | `Environment._make_environmental_layer_tensor`, `tests/test_memmap_environment_integration.py::test_set_environmental_layer_round_trip` |
| Temporal channel persistence (DAMAGE_HEAT, TRAILS, ALLY_SIGNAL) | `TemporalGridManager`, `Environment.deposit_temporal_events`, `tests/test_environmental_grids.py`, `tests/test_memmap_environment_integration.py::test_temporal_grid_decays_on_environment_update` |
| Performance ≤ 1.25× baseline latency for memmap operations | `scripts/validate_memmap_acceptance.py` reports per-grid microbenchmarks (resources, environmental, temporal) **and** an end-to-end `Environment.observe()` benchmark; all four are well under the 1.25× threshold (typically 0.98–1.05×) |
| Compatible with existing tensor operations | `Environment._np_window_to_tensor`, validation script tensor-compatibility checks |
| Memory usage scales with grid size, not loaded into RAM | Memmap files reported by validation script; tests assert `has_memmap` and on-disk file presence |
| Validation script covers all new memmap structures | `scripts/validate_memmap_acceptance.py` reports environmental and temporal acceptance lines |
| Cross-platform compatibility | `MemmapManager` only uses stdlib + numpy; filenames are sanitized for any FS |

### Multiprocess notes

Filenames embed `pid` and `simulation_id` so multiple concurrent
processes do not stomp on each other's files. If you need explicit
read-only sharing across processes, open consumers with `mode="r"` and
keep a single writer process.

### Run the validation script

```bash
python scripts/validate_memmap_acceptance.py
```

It reports PASS/WARN/FAIL per criterion (including the new
environmental and temporal grids) and an overall verdict
(`PASS`/`INCONCLUSIVE`/`FAIL`).
