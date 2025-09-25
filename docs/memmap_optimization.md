## Memory-Mapped State Storage (NumPy memmap)

### Overview

This change introduces on-disk, memory-mapped storage for large environment state representations using NumPy `memmap`. The initial target is the resource grid, enabling fast, localized window reads without loading full grids into RAM. This lays groundwork for additional strategies like sparse tensors and larger simulation scales.

### What Changed

- `farm/core/config.py`
  - New `SimulationConfig` options:
    - `use_memmap_resources: bool` — enable memmap-backed resource grid
    - `memmap_dir: Optional[str]` — directory for `.dat` files (defaults to system temp)
    - `memmap_dtype: str` — dtype string for memmap (default `float32`)
    - `memmap_mode: str` — file mode (`'w+'`, `'r+'`, `'c'`), default `'w+'`

- `farm/core/resource_manager.py`
  - Creates a memmapped 2D array of shape `(height, width)` when enabled.
  - Rebuilds the memmap after resource initialization and each update.
  - Provides `get_resource_window(y0, y1, x0, x1, normalize=True) -> np.ndarray` for fast local window reads with zero-padding at boundaries and optional normalization to `[0,1]` using `max_resource_amount`.
  - Cleans up via `cleanup_memmap(delete: bool)`. By default, files are flushed and retained on environment close for reuse.
  - Multiprocess-safety: filenames include PID and optional `simulation_id` to avoid collisions (e.g., `resources_<sim>_p12345_WidthxHeight.dat`).

- `farm/core/environment.py`
  - In `_get_observation`, if a memmap-backed grid is present, the resource channel window is sliced directly from memmap and converted to a torch tensor with the configured dtype/device.
  - Falls back to existing spatial queries and interpolation when memmap is disabled.
  - On `close()`, flushes the memmap file safely.

- Validation script: `scripts/validate_memmap_acceptance.py`
  - Builds baseline and memmap-enabled environments.
  - Measures RSS memory (Linux) to verify streaming behavior.
  - Benchmarks average local window access latency for both paths.
  - Confirms tensor compatibility by converting memmap windows to torch tensors.

### How to Enable

1. In your simulation config (YAML or programmatic), set:
   - `use_memmap_resources: true`
   - Optionally: `memmap_dir`, `memmap_dtype`, `memmap_mode`
2. Instantiate `Environment` with this `SimulationConfig`.
3. No agent or action changes are required.

### Acceptance Criteria Mapping

- States load/stream without full RAM usage
  - Memmap stores the resource grid on disk; window reads map pages on demand. The validation script reports memmap file size and RSS delta to demonstrate no full-grid RAM spike.

- Performance tests show no significant slowdown in access times
  - The script times `n` window reads using both the memmap path and the baseline spatial path and checks that memmap average latency ≤ 1.25× baseline.

- Compatible with existing tensor operations
  - Slices are converted to torch tensors in `_get_observation`; the script also converts a memmap slice to torch and sums it to ensure compatibility.

Run:

```
python scripts/validate_memmap_acceptance.py
```

It prints PASS/WARN/FAIL per criterion and an overall verdict.

### Benefits

- Reduced memory footprint for large grids by mapping state to disk instead of resident RAM.
- Scales to larger `width × height` without prohibitive memory usage.
- Maintains fast localized window access; avoids constructing full dense world tensors each step.
- Integrates seamlessly with existing observation/tensor pipelines.

### Multiprocess Notes

- Filenames include PID and optional `simulation_id` to avoid collisions when multiple processes run simultaneously on the same host.
- If you need multiple processes to share the exact same underlying file concurrently, open modes and locking strategies may need to be adapted (e.g., read-only `'r'` for consumers, single writer policy). Current default prioritizes isolation and safety over file sharing.

### Future Work

- Extend memmap to additional large state structures (e.g., agent-centric global layers).
- Optional sparse tensor backends for extremely sparse layers.
- Coordinated read-only sharing across processes with explicit file locks.
- Cross-platform memory metrics (non-Linux) in the validation script.

