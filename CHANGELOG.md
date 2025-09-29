## Unreleased

- Spatial module introduced at `farm.core.spatial`:
  - `SpatialIndex` (KD-tree orchestrator with batch updates)
  - `Quadtree`, `QuadtreeNode`
  - `SpatialHashGrid`
  - `DirtyRegionTracker`
- Deprecated `farm.core.spatial_index` (shim remains with DeprecationWarning). Migration:
  - Replace `from farm.core.spatial_index import SpatialIndex` with `from farm.core.spatial import SpatialIndex`
  - Replace `from farm.core.spatial_index import Quadtree, SpatialHashGrid` with `from farm.core.spatial import Quadtree, SpatialHashGrid`
  - Shim will be removed in a future minor release after two release cycles.
- Tests moved to `tests/spatial/` for organization; coverage preserved.

