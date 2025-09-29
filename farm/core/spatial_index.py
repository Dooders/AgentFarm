"""Spatial indexing system using KD-trees for efficient spatial queries.

This module provides a SpatialIndex class that manages KD-trees for agents and resources,
with optimized change detection and efficient spatial querying capabilities.
"""

import hashlib
import heapq
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Type aliases for better code clarity
Position = Tuple[float, float]
Bounds = Tuple[float, float, float, float]  # (x, y, width, height)
Entity = TypeVar("Entity")  # Generic entity type

DEFAULT_TARGET_CELLS = 20.0


@dataclass
class DirtyRegion:
    """Represents a dirty region that needs spatial index updates."""
    bounds: Tuple[float, float, float, float]  # (x, y, width, height)
    entity_type: str  # 'agent', 'resource', etc.
    priority: int = 0  # Higher priority regions updated first
    timestamp: float = 0.0  # When the region was marked dirty


class DirtyRegionTracker:
    """
    Tracks dirty regions for efficient batch spatial updates.
    
    This class manages regions that have changed and need spatial index updates,
    providing efficient batching and priority-based update scheduling.
    """
    
    def __init__(self, region_size: float = 50.0, max_regions: int = 1000):
        """
        Initialize the dirty region tracker.
        
        Parameters
        ----------
        region_size : float
            Size of each region for spatial partitioning
        max_regions : int
            Maximum number of regions to track before forcing cleanup
        """
        self.region_size = region_size
        self.max_regions = max_regions
        
        # Track dirty regions by entity type
        self._dirty_regions: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
        self._region_priorities: Dict[Tuple[int, int], int] = {}
        self._region_timestamps: Dict[Tuple[int, int], float] = {}
        
        # Batch update queue
        self._update_queue: deque = deque()
        self._batch_size = 10  # Process this many regions per batch
        
        # Performance metrics
        self._total_regions_marked = 0
        self._total_regions_updated = 0
        self._batch_count = 0
        
    def _world_to_region_coords(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to region coordinates."""
        x, y = position
        return int(x // self.region_size), int(y // self.region_size)
    
    def _region_to_world_bounds(self, region_coords: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Convert region coordinates to world bounds."""
        rx, ry = region_coords
        x = rx * self.region_size
        y = ry * self.region_size
        return (x, y, self.region_size, self.region_size)
    
    def mark_region_dirty(self, position: Tuple[float, float], entity_type: str, 
                         priority: int = 0, timestamp: Optional[float] = None) -> None:
        """
        Mark a region as dirty for the given entity type.
        
        Parameters
        ----------
        position : Tuple[float, float]
            World position that falls within the dirty region
        entity_type : str
            Type of entity ('agent', 'resource', etc.)
        priority : int
            Priority level (higher = more important)
        timestamp : float, optional
            Timestamp when region was marked dirty
        """
        if timestamp is None:
            import time
            timestamp = time.time()
            
        region_coords = self._world_to_region_coords(position)
        
        self._dirty_regions[entity_type].add(region_coords)
        self._region_priorities[region_coords] = max(
            self._region_priorities.get(region_coords, 0), priority
        )
        self._region_timestamps[region_coords] = timestamp
        
        self._total_regions_marked += 1
        
        # Cleanup if we have too many regions
        if len(self._region_priorities) > self.max_regions:
            self._cleanup_old_regions()
    
    def mark_region_dirty_batch(self, positions: List[Tuple[float, float]], 
                               entity_type: str, priority: int = 0) -> None:
        """
        Mark multiple regions as dirty in a single batch operation.
        
        Parameters
        ----------
        positions : List[Tuple[float, float]]
            List of world positions
        entity_type : str
            Type of entity
        priority : int
            Priority level for all regions
        """
        import time
        timestamp = time.time()
        
        for position in positions:
            self.mark_region_dirty(position, entity_type, priority, timestamp)
    
    def get_dirty_regions(self, entity_type: Optional[str] = None, 
                         max_count: Optional[int] = None) -> List[DirtyRegion]:
        """
        Get dirty regions that need updates, optionally filtered by entity type.
        
        Parameters
        ----------
        entity_type : str, optional
            Filter by entity type. If None, returns all dirty regions.
        max_count : int, optional
            Maximum number of regions to return
            
        Returns
        -------
        List[DirtyRegion]
            List of dirty regions sorted by priority (highest first)
        """
        dirty_regions = []
        
        # Collect regions
        if entity_type is not None:
            region_sets = {entity_type: self._dirty_regions.get(entity_type, set())}
        else:
            region_sets = self._dirty_regions
        
        for et, regions in region_sets.items():
            for region_coords in regions:
                bounds = self._region_to_world_bounds(region_coords)
                priority = self._region_priorities.get(region_coords, 0)
                timestamp = self._region_timestamps.get(region_coords, 0.0)
                
                dirty_regions.append(DirtyRegion(
                    bounds=bounds,
                    entity_type=et,
                    priority=priority,
                    timestamp=timestamp
                ))
        
        # Sort by priority (highest first), then by timestamp (oldest first)
        dirty_regions.sort(key=lambda r: (-r.priority, r.timestamp))
        
        if max_count is not None:
            dirty_regions = dirty_regions[:max_count]
            
        return dirty_regions
    
    def clear_region(self, region_coords: Tuple[int, int]) -> None:
        """Clear a specific region from dirty tracking."""
        for entity_type in self._dirty_regions:
            self._dirty_regions[entity_type].discard(region_coords)
        
        self._region_priorities.pop(region_coords, None)
        self._region_timestamps.pop(region_coords, None)
        self._total_regions_updated += 1
    
    def clear_regions(self, region_coords_list: List[Tuple[int, int]]) -> None:
        """Clear multiple regions from dirty tracking."""
        for region_coords in region_coords_list:
            self.clear_region(region_coords)
    
    def clear_all_regions(self) -> None:
        """Clear all dirty regions."""
        self._dirty_regions.clear()
        self._region_priorities.clear()
        self._region_timestamps.clear()
        self._update_queue.clear()
    
    def _cleanup_old_regions(self) -> None:
        """Remove oldest regions when we exceed max_regions limit."""
        if len(self._region_priorities) <= self.max_regions:
            return
            
        # Sort by timestamp and remove oldest regions
        sorted_regions = sorted(
            self._region_priorities.items(),
            key=lambda x: self._region_timestamps.get(x[0], 0.0)
        )
        
        regions_to_remove = len(self._region_priorities) - self.max_regions
        for region_coords, _ in sorted_regions[:regions_to_remove]:
            self.clear_region(region_coords)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about dirty region tracking."""
        total_dirty = sum(len(regions) for regions in self._dirty_regions.values())
        
        return {
            "total_dirty_regions": total_dirty,
            "regions_by_type": {et: len(regions) for et, regions in self._dirty_regions.items()},
            "total_regions_marked": self._total_regions_marked,
            "total_regions_updated": self._total_regions_updated,
            "batch_count": self._batch_count,
            "region_size": self.region_size,
            "max_regions": self.max_regions
        }


class SpatialHashGrid:
    """
    Uniform grid-based spatial hash for fast neighborhood queries.

    Partitions 2D space into fixed-size square cells. Entities are stored in
    buckets keyed by integer cell coordinates. Queries inspect only nearby
    buckets, keeping cost bounded regardless of global population size.

    Warning
    -------
    This implementation is not thread-safe. Concurrent access to insert(),
    remove(), move(), or query methods may result in data corruption or
    incorrect results. Use external synchronization if multi-threaded access
    is required.

    Notes
    -----
    - Cell size should be chosen based on typical query radii and entity density
    - Smaller cells provide better query performance but use more memory
    - Larger cells reduce memory usage but may increase query times
    - Default cell size calculation aims for ~sqrt(area)/20 cells per dimension
    """

    def __init__(self, cell_size: float, width: float, height: float):
        """
        Initialize the spatial hash grid.

        Parameters
        ----------
        cell_size : float
            Size of each square cell in the grid. Must be positive.
            Smaller values provide better spatial resolution but use more memory.
        width : float
            Width of the environment/space being indexed.
        height : float
            Height of the environment/space being indexed.

        Raises
        ------
        ValueError
            If cell_size is not positive.
        """
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self.width = float(width)
        self.height = float(height)
        # (ix, iy) -> List[(entity, (x, y))]
        self._buckets: Dict[Tuple[int, int], List[Tuple[Any, Position]]] = {}

    def _cell_coords(self, position: Position) -> Tuple[int, int]:
        x, y = position
        return int(x // self.cell_size), int(y // self.cell_size)

    def _bucket_keys_for_bounds(self, bounds: Bounds) -> List[Tuple[int, int]]:
        x, y, w, h = bounds
        if w <= 0 or h <= 0:
            return []
        x0 = int(x // self.cell_size)
        y0 = int(y // self.cell_size)
        x1 = int((x + w) // self.cell_size)
        y1 = int((y + h) // self.cell_size)
        keys: List[Tuple[int, int]] = []
        for iy in range(y0, y1 + 1):
            for ix in range(x0, x1 + 1):
                keys.append((ix, iy))
        return keys

    def insert(self, entity: Any, position: Position) -> None:
        """
        Insert an entity at the specified position.

        Parameters
        ----------
        entity : Any
            The entity to insert into the spatial index.
        position : Position
            The (x, y) coordinates where the entity is located.

        Notes
        -----
        If the entity is already present at this position, it will be added again.
        This allows multiple instances of the same entity at the same location.
        """
        key = self._cell_coords(position)
        self._buckets.setdefault(key, []).append((entity, position))

    def remove(self, entity: Any, position: Position) -> bool:
        """
        Remove an entity from the specified position.

        Parameters
        ----------
        entity : Any
            The entity to remove.
        position : Position
            The (x, y) coordinates where the entity should be located.

        Returns
        -------
        bool
            True if the entity was found and removed, False otherwise.

        Notes
        -----
        Only removes the first matching entity at the exact position.
        If multiple instances exist, only one is removed.
        """
        key = self._cell_coords(position)
        bucket = self._buckets.get(key)
        if not bucket:
            return False
        for i, (ent, _pos) in enumerate(bucket):
            if ent is entity:
                bucket.pop(i)
                if not bucket:
                    self._buckets.pop(key, None)
                return True
        return False

    def move(self, entity: Any, old_position: Position, new_position: Position) -> None:
        """
        Move an entity from one position to another.

        Parameters
        ----------
        entity : Any
            The entity to move.
        old_position : Position
            The current (x, y) coordinates of the entity.
        new_position : Position
            The new (x, y) coordinates for the entity.

        Notes
        -----
        This operation is optimized: if the entity stays within the same cell,
        only the position is updated. Otherwise, it's removed from the old cell
        and inserted into the new cell.
        """
        old_key = self._cell_coords(old_position)
        new_key = self._cell_coords(new_position)
        if old_key == new_key:
            bucket = self._buckets.get(old_key)
            if bucket:
                for i, (ent, _pos) in enumerate(bucket):
                    if ent is entity:
                        bucket[i] = (entity, new_position)
                        break
            return
        self.remove(entity, old_position)
        self.insert(entity, new_position)

    def query_radius(
        self, center: Position, radius: float
    ) -> List[Tuple[Any, Position]]:
        """
        Find all entities within a circular radius of a center point.

        Parameters
        ----------
        center : Position
            The (x, y) center point of the query circle.
        radius : float
            The radius of the query circle. Must be positive.

        Returns
        -------
        List[Tuple[Any, Position]]
            List of (entity, position) tuples for entities within the radius.
            Positions are exact entity locations, not cell centers.

        Notes
        -----
        This method uses a bounding box approximation for efficiency, then
        filters results to the exact circular radius. Performance is O(k)
        where k is the number of entities in the queried cells.
        """
        if radius <= 0:
            return []
        cx, cy = center
        bounds = (cx - radius, cy - radius, radius * 2, radius * 2)
        results: List[Tuple[Any, Position]] = []
        for key in self._bucket_keys_for_bounds(bounds):
            for entity, pos in self._buckets.get(key, []):
                dx = pos[0] - cx
                dy = pos[1] - cy
                if dx * dx + dy * dy <= radius * radius:
                    results.append((entity, pos))
        return results

    def query_range(self, bounds: Bounds) -> List[Tuple[Any, Position]]:
        """
        Find all entities within a rectangular bounds.

        Parameters
        ----------
        bounds : Bounds
            The rectangular bounds as (x, y, width, height).

        Returns
        -------
        List[Tuple[Any, Position]]
            List of (entity, position) tuples for entities within the bounds.

        Notes
        -----
        The bounds use half-open interval [x, x+width) x [y, y+height).
        Performance is O(k) where k is the number of entities in the queried cells.
        """
        x, y, w, h = bounds
        if w <= 0 or h <= 0:
            return []
        results: List[Tuple[Any, Position]] = []
        for key in self._bucket_keys_for_bounds(bounds):
            for entity, pos in self._buckets.get(key, []):
                px, py = pos
                if x <= px < x + w and y <= py < y + h:
                    results.append((entity, pos))
        return results

    def get_nearest(self, position: Position) -> Optional[Any]:
        """
        Find the nearest entity to the given position.

        This method performs a ring-based search starting from the cell containing
        the query position and expanding outward. It includes an early exit optimization
        when it can prove that no closer entity exists in outer rings.

        Parameters
        ----------
        position : Position
            The (x, y) query position.

        Returns
        -------
        Optional[Any]
            The nearest entity, or None if no entities exist in the index.

        Notes
        -----
        - Search starts with the cell containing the query position
        - Expands outward in Manhattan distance rings from the center cell
        - Uses early exit optimization based on minimum distance bounds
        - Time complexity is typically O(1) for dense populations, but can be
          O(n) in the worst case for sparse or empty regions

        Examples
        --------
        >>> grid = SpatialHashGrid(cell_size=10.0, width=100, height=100)
        >>> grid.insert(entity, (25, 25))
        >>> nearest = grid.get_nearest((20, 20))  # Returns entity
        """
        cx, cy = position
        ix, iy = self._cell_coords(position)
        best_entity: Optional[Any] = None
        best_dist_sq: float = float("inf")

        def consider_bucket(key: Tuple[int, int]) -> None:
            nonlocal best_entity, best_dist_sq
            for entity, pos in self._buckets.get(key, []):
                dx = pos[0] - cx
                dy = pos[1] - cy
                d2 = dx * dx + dy * dy
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_entity = entity

        # Always consider center cell first, but do not return early; a closer
        # entity may be in an adjacent cell near the boundary.
        consider_bucket((ix, iy))

        # Conservative ring cap: enough to cover the environment plus margin for entities
        # potentially outside bounds. Use the diagonal distance plus one dimension as margin.
        max_env_distance = math.sqrt(self.width**2 + self.height**2)
        margin = max(self.width, self.height)
        max_search_distance = max_env_distance + margin
        max_rings = int(math.ceil(max_search_distance / self.cell_size)) + 1
        for r in range(1, max_rings):
            for dx in range(-r, r + 1):
                consider_bucket((ix + dx, iy - r))
                consider_bucket((ix + dx, iy + r))
            for dy in range(-r + 1, r):
                consider_bucket((ix - r, iy + dy))
                consider_bucket((ix + r, iy + dy))
            # Early exit only when we can prove that no entity in the next ring
            # can be closer than the current best. The closest cell center in ring r+1
            # is at least r * cell_size away, but entities can be up to cell_size/sqrt(2)
            # closer than their cell centers, so we use a tighter bound.
            if best_entity is not None:
                # Distance to closest cell center in next ring minus max distance within cell
                cell_radius = self.cell_size / math.sqrt(2)
                min_dist_to_next_ring = max(0, r * self.cell_size - cell_radius)
                if best_dist_sq <= min_dist_to_next_ring * min_dist_to_next_ring:
                    return best_entity
        return best_entity


class QuadtreeNode:
    """
    A node in a quadtree for hierarchical spatial partitioning.

    Each node represents a rectangular region and can contain entities or be subdivided
    into four child quadrants. This enables efficient range queries and hierarchical operations.
    """

    def __init__(self, bounds: Tuple[float, float, float, float], capacity: int = 4):
        """
        Initialize a quadtree node.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (x, y, width, height) defining the rectangular region
        capacity : int
            Maximum entities before subdivision (default 4)
        """
        self.bounds = bounds  # (x, y, width, height)
        self.capacity = capacity
        self.entities: List[Any] = []
        self.children: Optional[List[QuadtreeNode]] = None
        self.is_divided = False

    def insert(self, entity: Any, position: Tuple[float, float]) -> bool:
        """
        Insert an entity into this node or its children.

        Parameters
        ----------
        entity : Any
            The entity to insert
        position : Tuple[float, float]
            (x, y) position of the entity

        Returns
        -------
        bool
            True if insertion was successful, False if entity is outside bounds
        """
        # Check if entity is within this node's bounds
        if not self._contains_point(position):
            return False

        # If not divided and not at capacity, add directly
        if not self.is_divided and len(self.entities) < self.capacity:
            self.entities.append((entity, position))
            return True

        # If not divided but at capacity, subdivide first
        if not self.is_divided:
            self._subdivide()

        # Insert into appropriate child quadrant
        for child in self.children:
            if child.insert(entity, position):
                return True

        # This shouldn't happen if bounds checking is correct
        return False

    def _subdivide(self) -> None:
        """Subdivide this node into four quadrants."""
        x, y, width, height = self.bounds
        half_width = width / 2
        half_height = height / 2

        # Create four child quadrants
        self.children = [
            # Northwest
            QuadtreeNode((x, y, half_width, half_height), self.capacity),
            # Northeast
            QuadtreeNode((x + half_width, y, half_width, half_height), self.capacity),
            # Southwest
            QuadtreeNode((x, y + half_height, half_width, half_height), self.capacity),
            # Southeast
            QuadtreeNode(
                (x + half_width, y + half_height, half_width, half_height),
                self.capacity,
            ),
        ]

        # Redistribute existing entities to children
        entities_to_redistribute = self.entities[:]
        self.entities.clear()

        for entity, position in entities_to_redistribute:
            inserted = False
            for child in self.children:
                if child.insert(entity, position):
                    inserted = True
                    break
            if not inserted:
                # If entity can't be inserted into children, keep it in parent
                self.entities.append((entity, position))

        self.is_divided = True

    def query_range(
        self, range_bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        """
        Query all entities within a rectangular range.

        Parameters
        ----------
        range_bounds : Tuple[float, float, float, float]
            (x, y, width, height) of the query rectangle

        Returns
        -------
        List[Tuple[Any, Tuple[float, float]]]
            List of (entity, position) tuples within the range
        """
        results = []

        # If range doesn't intersect this node, return empty
        if not self._intersects_range(range_bounds):
            return results

        # If this node is not divided, check all entities
        if not self.is_divided:
            for entity, position in self.entities:
                if self._point_in_range(position, range_bounds):
                    results.append((entity, position))
            return results

        # If divided, query children and also check any entities that remain in parent
        if self.children:
            for child in self.children:
                results.extend(child.query_range(range_bounds))

        # Check any entities that couldn't be subdivided
        for entity, position in self.entities:
            if self._point_in_range(position, range_bounds):
                results.append((entity, position))

        return results

    def query_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        """
        Query all entities within a circular radius.

        Parameters
        ----------
        center : Tuple[float, float]
            (x, y) center of the circle
        radius : float
            Radius of the circle

        Returns
        -------
        List[Tuple[Any, Tuple[float, float]]]
            List of (entity, position) tuples within the radius
        """
        results = []

        # Use bounding box for initial filtering
        x, y = center
        bbox = (x - radius, y - radius, radius * 2, radius * 2)

        # Get entities in bounding box
        bbox_entities = self.query_range(bbox)

        # Filter to circular radius
        for entity, position in bbox_entities:
            if self._distance(center, position) <= radius:
                results.append((entity, position))

        return results

    def remove(self, entity: Any, position: Tuple[float, float]) -> bool:
        """
        Remove an entity from this node or its children.

        Parameters
        ----------
        entity : Any
            The entity to remove
        position : Tuple[float, float]
            (x, y) position of the entity

        Returns
        -------
        bool
            True if entity was found and removed, False otherwise
        """
        # Check if position is within this node's bounds
        if not self._contains_point(position):
            return False

        # If not divided, check direct entities
        if not self.is_divided:
            for i, (ent, pos) in enumerate(self.entities):
                if ent is entity:
                    self.entities.pop(i)
                    return True
            return False

        # If divided, try to remove from children
        if self.children:
            for child in self.children:
                if child.remove(entity, position):
                    return True

        # Check entities that remain in parent
        for i, (ent, pos) in enumerate(self.entities):
            if ent is entity:
                self.entities.pop(i)
                return True

        return False

    def clear(self) -> None:
        """Clear all entities from this node and its children."""
        self.entities.clear()
        if self.children:
            for child in self.children:
                child.clear()
            self.children = None
        self.is_divided = False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this quadtree node."""
        entity_count = len(self.entities)
        if self.children:
            for child in self.children:
                entity_count += child.get_stats()["total_entities"]

        return {
            "bounds": self.bounds,
            "is_divided": self.is_divided,
            "local_entities": len(self.entities),
            "total_entities": entity_count,
            "children_count": len(self.children) if self.children else 0,
        }

    def _contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is within this node's bounds."""
        px, py = point
        x, y, width, height = self.bounds
        return x <= px < x + width and y <= py < y + height

    def _intersects_range(
        self, range_bounds: Tuple[float, float, float, float]
    ) -> bool:
        """Check if a rectangular range intersects this node's bounds."""
        rx, ry, rwidth, rheight = range_bounds
        nx, ny, nwidth, nheight = self.bounds

        return (
            rx < nx + nwidth
            and rx + rwidth > nx
            and ry < ny + nheight
            and ry + rheight > ny
        )

    def _point_in_range(
        self,
        point: Tuple[float, float],
        range_bounds: Tuple[float, float, float, float],
    ) -> bool:
        """Check if a point is within a rectangular range."""
        px, py = point
        rx, ry, rwidth, rheight = range_bounds
        return rx <= px < rx + rwidth and ry <= py < ry + rheight

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class Quadtree:
    """
    Quadtree implementation for efficient spatial partitioning and range queries.
    """

    def __init__(self, bounds: Tuple[float, float, float, float], capacity: int = 4):
        """
        Initialize a quadtree.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (x, y, width, height) defining the root region
        capacity : int
            Maximum entities per node before subdivision
        """
        self.root = QuadtreeNode(bounds, capacity)
        self.bounds = bounds
        self.capacity = capacity

    def insert(self, entity: Any, position: Tuple[float, float]) -> bool:
        """Insert an entity at the given position."""
        return self.root.insert(entity, position)

    def remove(self, entity: Any, position: Tuple[float, float]) -> bool:
        """Remove an entity from the given position."""
        return self.root.remove(entity, position)

    def query_range(
        self, bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        """Query all entities within a rectangular range."""
        return self.root.query_range(bounds)

    def query_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        """Query all entities within a circular radius."""
        return self.root.query_radius(center, radius)

    def clear(self) -> None:
        """Clear all entities from the quadtree."""
        self.root.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the quadtree."""
        return self.root.get_stats()


class SpatialIndex:
    """
    Efficient spatial indexing using KD-trees with optimized change detection and batch updates.

    This index maintains separate KD-trees for agents and resources and supports
    additional named indices. It uses multi-stage change detection (dirty flag,
    count deltas, and position hashing) to avoid unnecessary KD-tree rebuilds.
    Enhanced with batch spatial updates and dirty region tracking for improved performance.

    Capabilities:
    - O(log n) nearest/nearby queries across one or more indices
    - Named index registration for custom data sources
    - Relaxed bounds validation to tolerate edge cases
    - Deterministic caching of alive agents for query post-processing
    - Batch spatial updates with dirty region tracking
    - Region-based incremental updates for better performance
    """

    def __init__(
        self,
        width: float,
        height: float,
        index_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        index_data: Optional[Dict[str, Any]] = None,
        enable_batch_updates: bool = True,
        region_size: float = 50.0,
        max_batch_size: int = 100,
    ):
        """Initialize the spatial index.

        Parameters
        ----------
        width : float
            Width of the environment
        height : float
            Height of the environment
        index_configs : dict, optional
            Configuration for named indices
        index_data : dict, optional
            Initial data for named indices
        enable_batch_updates : bool
            Enable batch spatial updates with dirty region tracking
        region_size : float
            Size of regions for dirty region tracking
        max_batch_size : int
            Maximum number of position updates to batch together
        """
        self.width = width
        self.height = height
        self.enable_batch_updates = enable_batch_updates
        self.max_batch_size = max_batch_size

        # KD-tree attributes
        self.agent_kdtree: Optional[cKDTree] = None
        self.resource_kdtree: Optional[cKDTree] = None
        self.agent_positions: Optional[np.ndarray] = None
        self.resource_positions: Optional[np.ndarray] = None

        # Position change tracking for optimized updates
        self._positions_dirty: bool = True
        self._cached_counts: Optional[Tuple[int, int]] = None
        self._cached_hash: Optional[str] = None

        # Cached alive agents for efficient querying
        self._cached_alive_agents: Optional[List] = None

        # Reference to agent and resource lists
        self._agents: List = []
        self._resources: List = []

        # Named index registry for configurable indices
        # Each entry stores configuration and runtime state for that index
        self._named_indices: Dict[str, Dict[str, Any]] = {}

        # Optional initial indices supplied by caller
        # We defer registration until references are provided or explicit register_index is called
        self._initial_index_configs = index_configs or {}
        self._initial_index_data = index_data or {}

        # Batch update system
        if self.enable_batch_updates:
            self._dirty_region_tracker = DirtyRegionTracker(
                region_size=region_size,
                max_regions=max(1000, int((width * height) / (region_size * region_size)))
            )
            self._pending_position_updates: List[Tuple[Any, Tuple[float, float], Tuple[float, float]]] = []
            self._batch_update_enabled = True
        else:
            self._dirty_region_tracker = None
            self._pending_position_updates = []
            self._batch_update_enabled = False

        # Performance metrics for batch updates
        self._batch_update_stats = {
            "total_batch_updates": 0,
            "total_individual_updates": 0,
            "total_regions_processed": 0,
            "average_batch_size": 0.0,
            "last_batch_time": 0.0
        }

    def set_references(self, agents: List, resources: List) -> None:
        """Set references to agent and resource lists.

        Parameters
        ----------
        agents : List
            List of agents in the environment
        resources : List
            List of resources in the environment
        """
        self._agents = agents
        self._resources = resources

        # Basic type validation to catch incorrect references early
        try:
            if any(isinstance(a, (str, bytes)) for a in agents):
                logger.warning(
                    "SpatialIndex.set_references received agent IDs (strings) instead of agent objects. "
                    "This may cause empty agent indices. Ensure agent objects are passed."
                )
        except (TypeError, AttributeError):
            # Be tolerant of non-iterable or unusual inputs
            pass

        # Register default named indices mapping to agents/resources
        # Agents: filter only alive items
        self.register_index(
            name="agents",
            data_reference=self._agents,
            position_getter=lambda a: a.position,
            filter_func=lambda a: getattr(a, "alive", True),
        )
        # Resources: include all resources
        self.register_index(
            name="resources",
            data_reference=self._resources,
            position_getter=lambda r: r.position,
            filter_func=None,
        )

        # Register any user-supplied initial indices if provided at construction time
        for name, cfg in self._initial_index_configs.items():
            # Skip if already defined via defaults above
            if name in self._named_indices:
                continue
            data_ref_or_getter = self._initial_index_data.get(name)
            self.register_index(
                name=name,
                data_reference=(
                    data_ref_or_getter if isinstance(data_ref_or_getter, list) else None
                ),
                data_getter=(
                    data_ref_or_getter if callable(data_ref_or_getter) else None
                ),
                position_getter=cfg.get(
                    "position_getter", lambda x: getattr(x, "position", None)
                ),
                filter_func=cfg.get("filter_func", None),
            )

        # Changing references should trigger KD-tree rebuild on next query/update
        self.mark_positions_dirty()

    def mark_positions_dirty(self) -> None:
        """Mark that positions have changed and KD-trees need updating."""
        self._positions_dirty = True
        # Mark all named indices as dirty as well
        for idx in self._named_indices.values():
            idx["positions_dirty"] = True

    def add_position_update(self, entity: Any, old_position: Tuple[float, float], 
                           new_position: Tuple[float, float], entity_type: str = "agent",
                           priority: int = 0) -> None:
        """
        Add a position update to the batch queue for efficient processing.
        
        Parameters
        ----------
        entity : Any
            The entity whose position is being updated
        old_position : Tuple[float, float]
            The entity's old position
        new_position : Tuple[float, float]
            The entity's new position
        entity_type : str
            Type of entity ('agent', 'resource', etc.)
        priority : int
            Priority level for the update (higher = more important)
        """
        if not self._batch_update_enabled:
            # Fall back to immediate update
            self.update_entity_position(entity, old_position, new_position)
            return

        # Add to pending updates
        self._pending_position_updates.append((entity, old_position, new_position, entity_type, priority))
        
        # Mark regions as dirty
        if self._dirty_region_tracker:
            self._dirty_region_tracker.mark_region_dirty(old_position, entity_type, priority)
            self._dirty_region_tracker.mark_region_dirty(new_position, entity_type, priority)

        # Process batch if it's full
        if len(self._pending_position_updates) >= self.max_batch_size:
            self.process_batch_updates()

    def process_batch_updates(self, force: bool = False) -> None:
        """
        Process all pending position updates in a batch.
        
        Parameters
        ----------
        force : bool
            Force processing even if batch is not full
        """
        if not self._batch_update_enabled or not self._pending_position_updates:
            return

        if not force and len(self._pending_position_updates) < self.max_batch_size:
            return

        import time
        start_time = time.time()

        # Group updates by entity type for efficient processing
        updates_by_type = defaultdict(list)
        for entity, old_pos, new_pos, entity_type, priority in self._pending_position_updates:
            updates_by_type[entity_type].append((entity, old_pos, new_pos, priority))

        # Process each entity type
        regions_processed = 0
        for entity_type, updates in updates_by_type.items():
            # Get dirty regions for this entity type
            dirty_regions = self._dirty_region_tracker.get_dirty_regions(entity_type) if self._dirty_region_tracker else []
            
            # Process updates for this entity type
            for entity, old_pos, new_pos, priority in updates:
                self._process_single_position_update(entity, old_pos, new_pos, entity_type)
            
            # Clear processed regions
            if self._dirty_region_tracker and dirty_regions:
                region_coords_list = []
                for region in dirty_regions:
                    region_coords = self._dirty_region_tracker._world_to_region_coords(
                        (region.bounds[0], region.bounds[1])
                    )
                    region_coords_list.append(region_coords)
                self._dirty_region_tracker.clear_regions(region_coords_list)
                regions_processed += len(region_coords_list)

        # Clear pending updates
        batch_size = len(self._pending_position_updates)
        self._pending_position_updates.clear()

        # Update statistics
        end_time = time.time()
        self._batch_update_stats["total_batch_updates"] += 1
        self._batch_update_stats["total_individual_updates"] += batch_size
        self._batch_update_stats["total_regions_processed"] += regions_processed
        self._batch_update_stats["average_batch_size"] = (
            (self._batch_update_stats["average_batch_size"] * (self._batch_update_stats["total_batch_updates"] - 1) + batch_size) /
            self._batch_update_stats["total_batch_updates"]
        )
        self._batch_update_stats["last_batch_time"] = end_time - start_time

        logger.debug(
            "Processed batch update: %d entities, %d regions, %.3f seconds",
            batch_size, regions_processed, end_time - start_time
        )

    def _process_single_position_update(self, entity: Any, old_position: Tuple[float, float], 
                                       new_position: Tuple[float, float], entity_type: str) -> None:
        """
        Process a single position update efficiently.
        
        Parameters
        ----------
        entity : Any
            The entity whose position is being updated
        old_position : Tuple[float, float]
            The entity's old position
        new_position : Tuple[float, float]
            The entity's new position
        entity_type : str
            Type of entity
        """
        # Update named indices that support incremental updates
        for name, state in self._named_indices.items():
            index_type = state.get("index_type", "kdtree")

            if index_type == "quadtree" and state["quadtree"] is not None:
                # Efficient quadtree update
                state["quadtree"].remove(entity, old_position)
                state["quadtree"].insert(entity, new_position)
                
                # Update cached position data
                if state["positions"] is not None:
                    cached_items = state["cached_items"] or []
                    for i, cached_entity in enumerate(cached_items):
                        if cached_entity is entity and i < len(state["positions"]):
                            state["positions"][i] = new_position
                            break

            elif index_type == "spatial_hash" and state["spatial_hash"] is not None:
                # Efficient spatial hash update
                state["spatial_hash"].move(entity, old_position, new_position)
                if state["positions"] is not None:
                    cached_items = state["cached_items"] or []
                    for i, cached_entity in enumerate(cached_items):
                        if cached_entity is entity and i < len(state["positions"]):
                            state["positions"][i] = new_position
                            break

            elif index_type == "kdtree":
                # For KD-trees, we still need to mark as dirty for rebuild
                state["positions_dirty"] = True

        # Mark main positions as dirty for KD-tree rebuilds
        self._positions_dirty = True

    def get_batch_update_stats(self) -> Dict[str, Any]:
        """Get statistics about batch updates."""
        stats = dict(self._batch_update_stats)
        if self._dirty_region_tracker:
            stats.update(self._dirty_region_tracker.get_stats())
        return stats

    def enable_batch_updates(self, region_size: float = 50.0, max_batch_size: int = 100) -> None:
        """Enable batch updates with the specified configuration."""
        if not self._batch_update_enabled:
            self._dirty_region_tracker = DirtyRegionTracker(
                region_size=region_size,
                max_regions=max(1000, int((self.width * self.height) / (region_size * region_size)))
            )
            self._batch_update_enabled = True
            self.max_batch_size = max_batch_size
            logger.info("Batch updates enabled with region_size=%s, max_batch_size=%s", region_size, max_batch_size)

    def disable_batch_updates(self) -> None:
        """Disable batch updates and process any pending updates."""
        if self._batch_update_enabled:
            # Process any pending updates
            self.process_batch_updates(force=True)
            self._batch_update_enabled = False
            self._dirty_region_tracker = None
            logger.info("Batch updates disabled")

    def update(self) -> None:
        """Smart KD-tree update with optimized change detection and batch processing.

        Uses a multi-level optimization strategy:
        1. Process any pending batch updates first
        2. Dirty flag check (O(1)) - fastest for no-change scenarios
        3. Count-based check (O(1)) - catches structural changes
        4. Hash-based verification (O(n)) - ensures correctness
        """
        # Process any pending batch updates first
        if self._batch_update_enabled and self._pending_position_updates:
            self.process_batch_updates(force=True)

        # Fast path: if positions are not dirty, skip all expensive operations
        if not self._positions_dirty:
            # Only update named indices if they haven't been initialized yet
            if self.agent_kdtree is None or self.resource_kdtree is None:
                self._update_named_indices()
            return

        # Precompute alive agents once to avoid redundant computation
        alive_agents = [
            agent for agent in self._agents if getattr(agent, "alive", False)
        ]
        current_agent_count = len(alive_agents)

        # Count-based quick check for structural changes
        if self._counts_changed(current_agent_count):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return

        # Hash-based verification for position changes
        if self._hash_positions_changed(alive_agents):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return

        # No changes detected for default indices, but also ensure named indices are consistent
        self._update_named_indices()
        # Clear dirty flag
        self._positions_dirty = False

    def _counts_changed(self, current_agent_count: int) -> bool:
        """Check if agent or resource counts have changed (O(1) operation).

        Parameters
        ----------
        current_agent_count : int
            Precomputed count of alive agents

        Returns
        -------
        bool
            True if counts have changed, False otherwise
        """
        current_resource_count = len(self._resources)
        current_counts = (current_agent_count, current_resource_count)

        if self._cached_counts is None or self._cached_counts != current_counts:
            self._cached_counts = current_counts
            return True
        return False

    def _hash_positions_changed(self, alive_agents: List) -> bool:
        """Check if positions have changed using hash comparison (O(n) operation).

        This is the most expensive check but ensures correctness when
        counts match but positions have changed.

        Parameters
        ----------
        alive_agents : List
            Precomputed list of alive agents

        Returns
        -------
        bool
            True if positions have changed, False otherwise
        """
        # Build current positions for comparison
        # Filter out items without valid positions
        valid_alive_agents = [
            agent for agent in alive_agents if agent.position is not None
        ]
        valid_resources = [
            resource for resource in self._resources if resource.position is not None
        ]

        current_agent_positions = (
            np.array([agent.position for agent in valid_alive_agents])
            if valid_alive_agents
            else None
        )
        current_resource_positions = (
            np.array([resource.position for resource in valid_resources])
            if valid_resources
            else None
        )

        # Calculate hash of current agent positions
        if current_agent_positions is not None and len(current_agent_positions) > 0:
            agent_hash = hashlib.md5(current_agent_positions.tobytes()).hexdigest()
        else:
            agent_hash = "0"

        # Calculate hash of current resource positions
        if (
            current_resource_positions is not None
            and len(current_resource_positions) > 0
        ):
            resource_hash = hashlib.md5(
                current_resource_positions.tobytes()
            ).hexdigest()
        else:
            resource_hash = "0"

        current_hash = f"{agent_hash}:{resource_hash}"

        if self._cached_hash is None or self._cached_hash != current_hash:
            self._cached_hash = current_hash
            return True
        return False

    def _rebuild_kdtrees(self, alive_agents: List = None) -> None:
        """Rebuild KD-trees from current agent and resource positions.

        Parameters
        ----------
        alive_agents : List, optional
            Precomputed list of alive agents to avoid recomputation
        """
        # Update agent KD-tree
        if alive_agents is None:
            alive_agents = [
                agent for agent in self._agents if getattr(agent, "alive", False)
            ]

        # Filter out agents without valid positions
        alive_agents = [agent for agent in alive_agents if agent.position is not None]

        self._cached_alive_agents = (
            alive_agents  # Cache contains only alive agents for efficient queries
        )
        if alive_agents:
            self.agent_positions = np.array([agent.position for agent in alive_agents])
            self.agent_kdtree = cKDTree(self.agent_positions)
        else:
            self.agent_kdtree = None
            self.agent_positions = None

        # Update resource KD-tree
        # Filter out resources without valid positions
        valid_resources = [
            resource for resource in self._resources if resource.position is not None
        ]
        if valid_resources:
            self.resource_positions = np.array(
                [resource.position for resource in valid_resources]
            )
            self.resource_kdtree = cKDTree(self.resource_positions)
        else:
            self.resource_kdtree = None
            self.resource_positions = None

        # Also update any additional named indices after defaults are built
        self._update_named_indices()

    def register_index(
        self,
        name: str,
        data_reference: Optional[List[Any]] = None,
        position_getter: Optional[Callable[[Any], Tuple[float, float]]] = None,
        filter_func: Optional[Callable[[Any], bool]] = None,
        data_getter: Optional[Callable[[], List[Any]]] = None,
        index_type: str = "kdtree",
        cell_size: Optional[float] = None,
    ) -> None:
        """Register a configurable named index.

        Parameters
        ----------
        name : str
            Unique index name
        data_reference : list, optional
            Direct reference to a list of items for this index
        position_getter : callable, optional
            Function mapping an item -> (x, y)
        filter_func : callable, optional
            Predicate to include items (e.g., filter alive agents)
        data_getter : callable, optional
            Function returning the current list for this index
        index_type : str, optional
            Type of spatial index ("kdtree" or "quadtree", default "kdtree")
        """
        if position_getter is None:
            position_getter = lambda x: getattr(x, "position", None)

        self._named_indices[name] = {
            "data_reference": data_reference,
            "data_getter": data_getter,
            "position_getter": position_getter,
            "filter_func": filter_func,
            "index_type": index_type,
            "kdtree": None,
            "quadtree": None,
            "spatial_hash": None,
            "positions": None,
            "cached_items": None,
            "cached_count": None,
            "cached_hash": None,
            "positions_dirty": True,
            "cell_size": cell_size,
        }

    def _update_named_indices(self) -> None:
        """Update all registered named indices other than the built-in defaults."""
        for name, state in self._named_indices.items():
            # Skip: built-in defaults are already updated by _rebuild_kdtrees
            # but we still ensure their state mirrors the built-in structures
            if name == "agents":
                # Mirror built-in state for convenience
                state["kdtree"] = self.agent_kdtree
                state["positions"] = self.agent_positions
                state["cached_items"] = self._cached_alive_agents
                state["positions_dirty"] = False
                # Only compute expensive hash/counts if not already cached
                if state.get("cached_count") is None:
                    current_items = state["cached_items"] or []
                    # Filter out items without valid positions
                    valid_items = [
                        it
                        for it in current_items
                        if state["position_getter"](it) is not None
                    ]
                    current_positions = (
                        np.array([state["position_getter"](it) for it in valid_items])
                        if valid_items
                        else None
                    )
                    if current_positions is not None and len(current_positions) > 0:
                        curr_hash = hashlib.md5(current_positions.tobytes()).hexdigest()
                    else:
                        curr_hash = "0"
                    state["cached_count"] = len(valid_items)
                    state["cached_hash"] = curr_hash
                    # Update cached_items to only include items with valid positions
                    state["cached_items"] = valid_items
                continue
            if name == "resources":
                state["kdtree"] = self.resource_kdtree
                state["positions"] = self.resource_positions
                state["cached_items"] = self._resources
                state["positions_dirty"] = False
                # Only compute expensive hash/counts if not already cached
                if state.get("cached_count") is None:
                    current_items = state["cached_items"] or []
                    # Filter out items without valid positions
                    valid_items = [
                        it
                        for it in current_items
                        if state["position_getter"](it) is not None
                    ]
                    current_positions = (
                        np.array([state["position_getter"](it) for it in valid_items])
                        if valid_items
                        else None
                    )
                    if current_positions is not None and len(current_positions) > 0:
                        curr_hash = hashlib.md5(current_positions.tobytes()).hexdigest()
                    else:
                        curr_hash = "0"
                    state["cached_count"] = len(valid_items)
                    state["cached_hash"] = curr_hash
                    # Update cached_items to only include items with valid positions
                    state["cached_items"] = valid_items
                continue

            # For custom indices, rebuild if marked dirty
            if state.get("positions_dirty", True):
                self._rebuild_named_index(name)
                state["positions_dirty"] = False

    def _rebuild_named_index(self, name: str) -> None:
        """Rebuild spatial index for a specific named index (kdtree/quadtree/spatial_hash).

        For spatial hash indices, the `state` dictionary may include a `cell_size`
        parameter that controls the size of each grid cell. When not provided, a
        heuristic is applied based on environment dimensions to choose a reasonable
        default cell size.

        Args:
            name: The name of the index to rebuild.
        """
        state = self._named_indices[name]
        index_type = state.get("index_type", "kdtree")

        # Resolve items
        items = None
        if state["data_getter"] is not None:
            items = state["data_getter"]()
        elif state["data_reference"] is not None:
            items = state["data_reference"]
        else:
            items = []

        # Apply filter if provided
        if state["filter_func"] is not None:
            filtered_items = [it for it in items if state["filter_func"](it)]
        else:
            filtered_items = list(items)

        # Filter out items without valid positions
        valid_items = [
            it for it in filtered_items if state["position_getter"](it) is not None
        ]

        if index_type == "kdtree":
            # Build KD-tree
            if valid_items:
                positions = np.array(
                    [state["position_getter"](it) for it in valid_items]
                )
                kdtree = cKDTree(positions)
            else:
                positions = None
                kdtree = None

            state["cached_items"] = valid_items
            state["positions"] = positions
            state["kdtree"] = kdtree
            state["quadtree"] = None  # Clear quadtree if switching
            state["spatial_hash"] = None

        elif index_type == "quadtree":
            # Build Quadtree
            if valid_items:
                # Create quadtree with environment bounds
                bounds = (0, 0, self.width, self.height)
                quadtree = Quadtree(bounds, capacity=4)

                # Insert all valid items
                for item in valid_items:
                    position = state["position_getter"](item)
                    quadtree.insert(item, position)

                positions = np.array(
                    [state["position_getter"](it) for it in valid_items]
                )
            else:
                quadtree = None
                positions = None

            state["cached_items"] = valid_items
            state["positions"] = positions
            state["quadtree"] = quadtree
            state["kdtree"] = None  # Clear kdtree if switching
            state["spatial_hash"] = None

        elif index_type == "spatial_hash":
            # Build spatial hash grid
            if valid_items:
                cs = state.get("cell_size")
                if cs is None:
                    # Adaptive cell size based on environment dimensions
                    # Aim for roughly sqrt(area)/20 cells per dimension for good performance
                    # This scales with environment size and provides reasonable granularity
                    env_area = self.width * self.height
                    target_cells_per_dim = max(5.0, math.sqrt(env_area) / 20.0)
                    cell_w = max(self.width / target_cells_per_dim, 1.0)
                    cell_h = max(self.height / target_cells_per_dim, 1.0)
                    cs = float((cell_w + cell_h) / 2.0)
                grid = SpatialHashGrid(
                    cell_size=cs, width=self.width, height=self.height
                )
                for item in valid_items:
                    grid.insert(item, state["position_getter"](item))
                positions = np.array(
                    [state["position_getter"](it) for it in valid_items]
                )
            else:
                grid = None
                positions = None

            state["cached_items"] = valid_items
            state["positions"] = positions
            state["spatial_hash"] = grid
            state["kdtree"] = None
            state["quadtree"] = None

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        state["cached_count"] = len(valid_items)
        if positions is not None and len(positions) > 0:
            state["cached_hash"] = hashlib.md5(positions.tobytes()).hexdigest()
        else:
            state["cached_hash"] = "0"

    def get_nearby(
        self,
        position: Tuple[float, float],
        radius: float,
        index_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Generic nearby query across one or more named indices.

        Parameters
        ----------
        position : Tuple[float, float]
            Query position as (x, y) coordinates
        radius : float
            Search radius around the query position. Must be positive.
        index_names : List[str], optional
            Names of specific indices to search. If None, searches all registered indices.

        Returns
        -------
        Dict[str, List[Any]]
            Dictionary mapping index names to lists of nearby items within the radius.
            Each index name corresponds to a key in the returned dictionary, even if
            no items are found (in which case the list will be empty).

        Examples
        --------
        >>> # Search all indices for items within 10 units
        >>> nearby = spatial_index.get_nearby((5.0, 5.0), 10.0)
        >>> # Returns: {'agents': [agent1, agent2], 'resources': [resource1]}

        >>> # Search only specific indices
        >>> nearby = spatial_index.get_nearby((5.0, 5.0), 10.0, ['agents'])
        >>> # Returns: {'agents': [agent1, agent2]}

        Notes
        -----
        Only searches indices that have been registered via register_index() and
        contain valid KD-trees. Invalid or empty indices will return empty lists.
        """
        self.update()

        # Input validation (reuse same checks)
        if radius <= 0 or not self._is_valid_position(position):
            return {}

        names = index_names or list(self._named_indices.keys())
        results: Dict[str, List[Any]] = {}
        for name in names:
            state = self._named_indices.get(name)
            if state is None:
                results[name] = []
                continue

            index_type = state.get("index_type", "kdtree")

            if index_type == "kdtree":
                if state["kdtree"] is None:
                    results[name] = []
                    continue
                indices = state["kdtree"].query_ball_point(position, radius)
                cached_items = state["cached_items"] or []
                results[name] = [cached_items[i] for i in indices]

            elif index_type == "quadtree":
                if state["quadtree"] is None:
                    results[name] = []
                    continue
                # Use quadtree radius query
                entities_and_positions = state["quadtree"].query_radius(
                    position, radius
                )
                results[name] = [entity for entity, pos in entities_and_positions]

            elif index_type == "spatial_hash":
                if state["spatial_hash"] is None:
                    results[name] = []
                    continue
                entities_and_positions = state["spatial_hash"].query_radius(
                    position, radius
                )
                results[name] = [entity for entity, _ in entities_and_positions]

            else:
                results[name] = []
        return results

    def get_nearest(
        self, position: Tuple[float, float], index_names: Optional[List[str]] = None
    ) -> Dict[str, Optional[Any]]:
        """Generic nearest query across one or more named indices.

        Parameters
        ----------
        position : tuple of float
            The (x, y) coordinates to query for the nearest item.
        index_names : list of str, optional
            List of index names to query. If None, queries all registered indices.

        Returns
        -------
        dict of str to object or None
            A dictionary mapping each index name to the nearest item in that index,
            or None if the index is empty or not found.

        Examples
        --------
        Suppose you have two indices, "agents" and "resources":

        >>> idx = SpatialIndex(width=100, height=100, index_configs={
        ...     "agents": {"items": [(1, 2), (3, 4)]},
        ...     "resources": {"items": [(10, 10), (20, 20)]}
        ... })
        >>> idx.get_nearest((2, 3))
        {'agents': (1, 2), 'resources': (10, 10)}

        If you specify a subset of indices:

        >>> idx.get_nearest((2, 3), index_names=["resources"])
        {'resources': (10, 10)}

        If an index is empty or does not exist:

        >>> idx.get_nearest((2, 3), index_names=["nonexistent"])
        {'nonexistent': None}
        """
        self.update()
        if not self._is_valid_position(position):
            return {}

        names = index_names or list(self._named_indices.keys())
        results: Dict[str, Optional[Any]] = {}
        for name in names:
            state = self._named_indices.get(name)
            if state is None:
                results[name] = None
                continue

            index_type = state.get("index_type", "kdtree")

            if index_type == "kdtree":
                if state["kdtree"] is None or not state["cached_items"]:
                    results[name] = None
                    continue
                _, idx = state["kdtree"].query(position)
                results[name] = state["cached_items"][idx]

            elif index_type == "quadtree":
                if state["quadtree"] is None or not state["cached_items"]:
                    results[name] = None
                    continue
                # Use best-first search on quadtree for nearest neighbor
                results[name] = self._quadtree_nearest(state["quadtree"], position)

            elif index_type == "spatial_hash":
                if state["spatial_hash"] is None or not state["cached_items"]:
                    results[name] = None
                    continue
                results[name] = state["spatial_hash"].get_nearest(position)

            else:
                results[name] = None
        return results

    def _quadtree_nearest(
        self, quadtree: Quadtree, position: Tuple[float, float]
    ) -> Optional[Any]:
        """Find nearest entity in a quadtree using best-first search with pruning.

        Explores nodes ordered by the minimum possible distance from the query point
        to the node's bounds. Prunes subtrees whose min-distance exceeds the best
        entity distance found so far.
        """
        if quadtree is None or quadtree.root is None:
            return None

        def rect_min_distance_sq(
            point: Tuple[float, float], bounds: Tuple[float, float, float, float]
        ) -> float:
            px, py = point
            x, y, w, h = bounds
            cx = px if x <= px <= x + w else (x if px < x else x + w)
            cy = py if y <= py <= y + h else (y if py < y else y + h)
            dx = px - cx
            dy = py - cy
            return dx * dx + dy * dy

        def distance_sq(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            return dx * dx + dy * dy

        best_entity: Optional[Any] = None
        best_dist_sq: float = float("inf")

        # Priority queue of (min_possible_distance_sq, node)
        heap: List[Tuple[float, QuadtreeNode]] = []
        heapq.heappush(
            heap, (rect_min_distance_sq(position, quadtree.root.bounds), quadtree.root)
        )

        while heap:
            min_possible_sq, node = heapq.heappop(heap)

            # If the closest possible entity in this node cannot beat current best, prune
            if min_possible_sq >= best_dist_sq:
                break

            # Check entities stored at this node
            for entity, entity_pos in node.entities:
                d2 = distance_sq(position, entity_pos)
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_entity = entity

            # Explore children ordered by their min distance
            if node.children:
                for child in node.children:
                    child_min_sq = rect_min_distance_sq(position, child.bounds)
                    if child_min_sq < best_dist_sq:
                        heapq.heappush(heap, (child_min_sq, child))

        return best_entity

    def _is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid within the environment bounds.

        Uses relaxed bounds to allow positions slightly outside the strict boundaries,
        which is useful for edge cases and floating-point precision issues.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to check

        Returns
        -------
        bool
            True if position is within relaxed bounds, False otherwise
        """
        x, y = position
        # Allow positions within 1% margin outside bounds for edge cases
        margin_x = self.width * 0.01
        margin_y = self.height * 0.01
        return (-margin_x <= x <= self.width + margin_x) and (
            -margin_y <= y <= self.height + margin_y
        )

    def get_agent_count(self) -> int:
        """Get the number of alive agents.

        Returns
        -------
        int
            Number of alive agents
        """
        return len([a for a in self._agents if getattr(a, "alive", False)])

    def get_resource_count(self) -> int:
        """Get the number of resources.

        Returns
        -------
        int
            Number of resources
        """
        return len(self._resources)

    def is_dirty(self) -> bool:
        """Check if positions are dirty and need updating.

        Returns
        -------
        bool
            True if positions are dirty, False otherwise
        """
        return self._positions_dirty

    def update_entity_position(self, entity: Any, old_position: Tuple[float, float], new_position: Tuple[float, float]) -> None:
        """Update an entity's position in all spatial indices.

        This method now supports both immediate updates and batch updates depending on configuration.
        For batch updates, the position change is queued for efficient batch processing.
        For immediate updates, the position is updated directly in all indices.

        Parameters
        ----------
        entity : Any
            The entity whose position is being updated
        old_position : Tuple[float, float]
            The entity's old (x, y) position
        new_position : Tuple[float, float]
            The entity's new (x, y) position
        """
        # Determine entity type based on entity attributes or context
        entity_type = "agent"  # Default
        if hasattr(entity, 'resource_level') and not hasattr(entity, 'alive'):
            entity_type = "resource"
        elif hasattr(entity, 'alive'):
            entity_type = "agent"

        # Use batch updates if enabled
        if self._batch_update_enabled:
            self.add_position_update(entity, old_position, new_position, entity_type)
            return

        # Immediate update (original behavior)
        self._process_single_position_update(entity, old_position, new_position, entity_type)

    def force_rebuild(self) -> None:
        """Force a rebuild of the spatial indices regardless of change detection."""
        self._rebuild_kdtrees()
        self._positions_dirty = False

    def get_nearby_range(
        self,
        bounds: Tuple[float, float, float, float],
        index_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Query entities within a rectangular range (optimized for Quadtrees).

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (x, y, width, height) defining the rectangular query region
        index_names : List[str], optional
            Names of specific indices to search. If None, searches all quadtree indices.

        Returns
        -------
        Dict[str, List[Any]]
            Dictionary mapping index names to lists of entities within the range.
        """
        self.update()

        # Input validation
        x, y, width, height = bounds
        if width <= 0 or height <= 0:
            return {}

        names = index_names or list(self._named_indices.keys())
        results: Dict[str, List[Any]] = {}

        for name in names:
            state = self._named_indices.get(name)
            if state is None:
                results[name] = []
                continue

            index_type = state.get("index_type", "kdtree")

            if index_type == "quadtree" and state["quadtree"] is not None:
                # Use quadtree rectangular query
                entities_and_positions = state["quadtree"].query_range(bounds)
                results[name] = [entity for entity, pos in entities_and_positions]

            elif (
                index_type == "kdtree"
                and state["kdtree"] is not None
                and state["cached_items"]
            ):
                # For KD-trees, we use a different approach for rectangular queries
                # Since KD-trees don't have built-in rectangular query support,
                # we can use the existing radial query method with a large enough radius
                # to cover the rectangle, then filter to the actual rectangle
                cached_items = state["cached_items"]
                positions = state["positions"]

                if positions is not None:
                    # Calculate center and radius to cover the entire rectangle
                    center_x = x + width / 2
                    center_y = y + height / 2
                    # Use diagonal distance as radius to ensure coverage
                    radius = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

                    # Query with large radius
                    indices = state["kdtree"].query_ball_point(
                        (center_x, center_y), radius
                    )

                    # Filter to actual rectangular bounds
                    entities_in_range = []
                    for i in indices:
                        if i < len(positions):
                            px, py = positions[i]
                            if x <= px < x + width and y <= py < y + height:
                                entities_in_range.append(cached_items[i])
                    results[name] = entities_in_range
                else:
                    results[name] = []

            elif index_type == "spatial_hash" and state["spatial_hash"] is not None:
                entities_and_positions = state["spatial_hash"].query_range(bounds)
                results[name] = [entity for entity, _ in entities_and_positions]

            else:
                results[name] = []

        return results

    def get_quadtree_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics about a specific quadtree index.

        Parameters
        ----------
        index_name : str
            Name of the quadtree index

        Returns
        -------
        Dict[str, Any] or None
            Quadtree statistics if the index exists and is a quadtree, None otherwise
        """
        state = self._named_indices.get(index_name)
        if (
            state is None
            or state.get("index_type") != "quadtree"
            or state["quadtree"] is None
        ):
            return None

        return state["quadtree"].get_stats()

    def get_stats(self) -> dict:
        """Get statistics about the spatial index.

        Returns
        -------
        dict
            Dictionary containing spatial index statistics
        """
        stats = {
            "agent_count": self.get_agent_count(),
            "resource_count": self.get_resource_count(),
            "agent_kdtree_exists": self.agent_kdtree is not None,
            "resource_kdtree_exists": self.resource_kdtree is not None,
            "positions_dirty": self._positions_dirty,
            "cached_counts": self._cached_counts,
            "cached_hash": (
                self._cached_hash[:20] + "..." if self._cached_hash else None
            ),
        }

        # Add quadtree information for each index
        quadtree_info = {}
        for name, state in self._named_indices.items():
            if state.get("index_type") == "quadtree":
                quadtree_info[name] = {
                    "exists": state["quadtree"] is not None,
                    "total_entities": state.get("cached_count", 0),
                }

        if quadtree_info:
            stats["quadtree_indices"] = quadtree_info

        return stats
