"""Compatibility shim for spatial indexing (DEPRECATED).

Deprecated: This module remains for backward compatibility only.
The implementation has moved to `farm.core.spatial`.

Please update imports to:
    from farm.core.spatial import SpatialIndex, Quadtree, SpatialHashGrid

This shim will be removed in a future release.
"""

import warnings

warnings.warn(
    "farm.core.spatial_index is deprecated; use farm.core.spatial instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .spatial import (
    SpatialIndex,
    DirtyRegion,
    DirtyRegionTracker,
    SpatialHashGrid,
    Quadtree,
    QuadtreeNode,
)

__all__ = [
    "SpatialIndex",
    "DirtyRegion",
    "DirtyRegionTracker",
    "SpatialHashGrid",
    "Quadtree",
    "QuadtreeNode",
]

