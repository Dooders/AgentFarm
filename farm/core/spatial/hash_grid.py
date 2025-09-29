"""Uniform grid-based spatial hash for fast neighborhood queries."""

import math
from typing import Any, Dict, List, Optional, Tuple

Position = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


class SpatialHashGrid:
    """
    Uniform grid-based spatial hash for fast neighborhood queries.

    Partitions 2D space into fixed-size square cells. Entities are stored in
    buckets keyed by integer cell coordinates. Queries inspect only nearby
    buckets, keeping cost bounded regardless of global population size.
    """

    def __init__(self, cell_size: float, width: float, height: float):
        """
        Initialize the spatial hash grid.

        Parameters
        ----------
        cell_size : float
            Size of each square cell in the grid. Must be positive.
        width : float
            Width of the environment/space being indexed.
        height : float
            Height of the environment/space being indexed.
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
        key = self._cell_coords(position)
        self._buckets.setdefault(key, []).append((entity, position))

    def remove(self, entity: Any, position: Position) -> bool:
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

    def query_radius(self, center: Position, radius: float) -> List[Tuple[Any, Position]]:
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

        consider_bucket((ix, iy))

        max_env_distance = math.sqrt(self.width ** 2 + self.height ** 2)
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
            if best_entity is not None:
                cell_radius = self.cell_size / math.sqrt(2)
                min_dist_to_next_ring = max(0, r * self.cell_size - cell_radius)
                if best_dist_sq <= min_dist_to_next_ring * min_dist_to_next_ring:
                    return best_entity
        return best_entity

