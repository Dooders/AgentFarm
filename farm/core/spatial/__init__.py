"""Spatial module exposing spatial indexing utilities and data structures.

This package contains:
- Dirty region tracking for batch updates
- Spatial hash grid for uniform-grid indexing
- Quadtree for hierarchical partitioning
- SpatialIndex orchestrating KD-tree, Quadtree, and Spatial Hash indices
"""

from .dirty_regions import DirtyRegion, DirtyRegionTracker
from .hash_grid import SpatialHashGrid
from .quadtree import Quadtree, QuadtreeNode
from .index import SpatialIndex

__all__ = [
    "DirtyRegion",
    "DirtyRegionTracker",
    "SpatialHashGrid",
    "Quadtree",
    "QuadtreeNode",
    "SpatialIndex",
]

