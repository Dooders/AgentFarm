"""Spatial module exposing spatial indexing utilities and data structures.

This package contains:
- Dirty region tracking for batch updates
- Spatial hash grid for uniform-grid indexing
- Quadtree for hierarchical partitioning
- SpatialIndex orchestrating KD-tree, Quadtree, and Spatial Hash indices
- Priority constants for batch update ordering
"""

from .dirty_regions import DirtyRegion, DirtyRegionTracker
from .hash_grid import SpatialHashGrid
from .index import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    SpatialIndex,
)
from .quadtree import Quadtree, QuadtreeNode

__all__ = [
    "DirtyRegion",
    "DirtyRegionTracker",
    "SpatialHashGrid",
    "Quadtree",
    "QuadtreeNode",
    "SpatialIndex",
    "PRIORITY_LOW",
    "PRIORITY_NORMAL",
    "PRIORITY_HIGH",
    "PRIORITY_CRITICAL",
]
