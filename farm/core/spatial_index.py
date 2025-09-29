"""Compatibility shim for spatial indexing.

This module remains for backward compatibility. The implementation has been
moved to the `farm.core.spatial` package. Importing from here will continue to
work, but new code should prefer `from farm.core.spatial import SpatialIndex`.
"""

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

