### Spatial Index Demo — Notebook Guide

This guide explains the purpose and usage of the notebook at `notebooks/spatial_index_demo.ipynb`. It demonstrates the KD-tree based `SpatialIndex` used in AgentFarm for efficient proximity queries and validates expected behavior end-to-end.

### What this notebook shows

- **Initialization**: Creates random agents and resources in a 2D world, initializes `SpatialIndex`, and displays `get_stats()`.
- **Core queries**: Uses `get_nearby(position, radius)` and `get_nearest(position)` across registered indices and validates results.
- **Named index**: Registers an `enemies` index (subset of agents) and runs targeted queries against it.
- **Edge cases & rebuilds**: Covers zero/negative radius, relaxed bounds, empty indices behavior, and demonstrates dirty flag marking plus rebuilds.
- **Visualizations**: Plots the scene, highlights nearby and nearest results, and shows multiple radii comparisons.

### Where to find things

- **Notebook**: `notebooks/spatial_index_demo.ipynb`
- **Spatial index implementation**: `farm/core/spatial_index.py`

### How to run

1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   - Via Jupyter: `jupyter lab` or `jupyter notebook` and navigate to `notebooks/spatial_index_demo.ipynb`
   - In your IDE (e.g., VS Code/Cursor): open the file and run cells.

### Cell-by-cell tour

1. **Title**: Brief description of the demo.
2. **Imports and setup**: Imports `numpy`, `matplotlib`, and `SpatialIndex`; sets seeds for reproducibility.
3. **Helpers and data generation**:
   - Defines simple `Agent` and `Resource` dataclasses with `position` and `alive`.
   - Generates random agents/resources within bounds.
   - Initializes `SpatialIndex`, sets references, and updates the index.
4. **Visualization helpers**: `plot_scene(...)` for drawing agents, resources, query point, radius ring, and highlights.
5. **Core queries with validations**:
   - Chooses a random query point and radius.
   - Calls `get_nearby` and `get_nearest` across default indices (`agents`, `resources`).
   - Asserts all nearby items lie within the radius and nearest items have expected types.
   - Visualizes results.
6. **Named index (`enemies`)**:
   - Registers `enemies` as a subset of agents via `register_index(name="enemies", ...)`.
   - Marks positions dirty and updates the index.
   - Runs `get_nearby`/`get_nearest` restricted to `enemies` and visualizes.
7. **Edge cases and dirty/rebuild behavior**:
   - Zero/negative radius behavior.
   - Queries slightly outside bounds (relaxed margin) return proper keys.
   - Moves a subset of agents, marks dirty, updates, and checks `get_stats()` before/after.
   - Empties indices to verify empty behavior, then restores and validates counts.
8. **Additional visualization**: Compares results at multiple radii from the center of the map.

### Expected outputs and validations

- **`get_nearby`**: Returns a dict mapping each registered index name to a list of items within the radius. For zero/negative radius, returns `{}`.
- **`get_nearest`**: Returns a dict mapping each index name to the nearest item (or `None` if the index is empty).
- **Validations**: The notebook asserts distance constraints, presence of keys, and type expectations. Rebuild behavior clears the dirty flag after `update()`.

### Extending the demo

- **Custom indices**: You can register your own named indices over any item list or provider.
  ```python
  sindex.register_index(
      name="obstacles",
      data_reference=my_obstacles,                  # or use data_getter=callable
      position_getter=lambda o: o.position,
      filter_func=None,
  )
  sindex.mark_positions_dirty()
  sindex.update()
  nearby_obstacles = sindex.get_nearby(query, 12.0, index_names=["obstacles"])
  ```

### Troubleshooting

- **Missing dependencies**: Install with `pip install -r requirements.txt`. The demo needs `numpy`, `scipy`, and `matplotlib`.
- **No results shown**: Ensure cells are executed in order; re-run initialization cells if you modified the data.
- **Plots not showing**: Confirm `%matplotlib inline` (or your IDE’s plot backend) is active.

