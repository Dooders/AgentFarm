"""Dirty region tracking for efficient batch spatial updates."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


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

    def __init__(self, region_size: float = 50.0, max_regions: int = 1000, batch_size: int = 10):
        """
        Initialize the dirty region tracker.

        Parameters
        ----------
        region_size : float
            Size of each region for spatial partitioning
        max_regions : int
            Maximum number of regions to track before forcing cleanup
        batch_size : int
            Number of regions to process per batch (default: 10)
        """
        self.region_size = region_size
        self.max_regions = max_regions

        # Track dirty regions by entity type
        self._dirty_regions: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
        self._region_priorities: Dict[Tuple[int, int], int] = {}
        self._region_timestamps: Dict[Tuple[int, int], float] = {}

        # Batch update queue
        self._update_queue: deque = deque()
        self._batch_size = batch_size  # Process this many regions per batch

        # Performance metrics
        self._total_regions_marked = 0
        self._total_regions_updated = 0
        self._batch_count = 0

    def world_to_region_coords(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to region coordinates.

        Public wrapper to avoid external callers depending on private methods.
        """
        x, y = position
        return int(x // self.region_size), int(y // self.region_size)

    def _region_to_world_bounds(self, region_coords: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Convert region coordinates to world bounds."""
        rx, ry = region_coords
        x = rx * self.region_size
        y = ry * self.region_size
        return (x, y, self.region_size, self.region_size)

    def mark_region_dirty(
        self,
        position: Tuple[float, float],
        entity_type: str,
        priority: int = 0,
        timestamp: Optional[float] = None,
    ) -> None:
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
            timestamp = time.time()

        region_coords = self.world_to_region_coords(position)

        self._dirty_regions[entity_type].add(region_coords)
        self._region_priorities[region_coords] = max(
            self._region_priorities.get(region_coords, 0), priority
        )
        self._region_timestamps[region_coords] = timestamp

        self._total_regions_marked += 1

        # Cleanup if we have too many regions
        if len(self._region_priorities) > self.max_regions:
            self._cleanup_old_regions()

    def mark_region_dirty_batch(self, positions: List[Tuple[float, float]], entity_type: str, priority: int = 0) -> None:
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
        timestamp = time.time()

        for position in positions:
            self.mark_region_dirty(position, entity_type, priority, timestamp)

    def get_dirty_regions(self, entity_type: Optional[str] = None, max_count: Optional[int] = None) -> List[DirtyRegion]:
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
        dirty_regions: List[DirtyRegion] = []

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

                dirty_regions.append(
                    DirtyRegion(
                        bounds=bounds,
                        entity_type=et,
                        priority=priority,
                        timestamp=timestamp,
                    )
                )

        # Sort by priority (highest first), then by timestamp (oldest first)
        dirty_regions.sort(key=lambda r: (-r.priority, r.timestamp))

        if max_count is not None:
            dirty_regions = dirty_regions[: max_count]

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
            self._region_priorities.items(), key=lambda x: self._region_timestamps.get(x[0], 0.0)
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
            "max_regions": self.max_regions,
        }

