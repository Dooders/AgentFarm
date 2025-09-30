"""Quadtree implementation for efficient spatial partitioning and range queries."""

from typing import Any, Dict, List, Optional, Tuple


class QuadtreeNode:
    """
    A node in a quadtree for hierarchical spatial partitioning.

    Each node represents a rectangular region and can contain entities or be subdivided
    into four child quadrants. This enables efficient range queries and hierarchical operations.
    """

    def __init__(self, bounds: Tuple[float, float, float, float], capacity: int = 4):
        self.bounds = bounds  # (x, y, width, height)
        self.capacity = capacity
        self.entities: List[Any] = []
        self.children: Optional[List["QuadtreeNode"]] = None
        self.is_divided = False

    def insert(self, entity: Any, position: Tuple[float, float]) -> bool:
        if not self._contains_point(position):
            return False
        if not self.is_divided and len(self.entities) < self.capacity:
            self.entities.append((entity, position))
            return True
        if not self.is_divided:
            self._subdivide()
        for child in self.children or []:
            if child.insert(entity, position):
                return True
        return False

    def _subdivide(self) -> None:
        x, y, width, height = self.bounds
        half_width = width / 2
        half_height = height / 2
        # Subdivide the region into four equal quadrants
        self.children = [
            QuadtreeNode((x, y, half_width, half_height), self.capacity),
            QuadtreeNode((x + half_width, y, half_width, half_height), self.capacity),
            QuadtreeNode((x, y + half_height, half_width, half_height), self.capacity),
            QuadtreeNode(
                (x + half_width, y + half_height, half_width, half_height),
                self.capacity,
            ),
        ]
        entities_to_redistribute = self.entities[:]
        self.entities.clear()
        for entity, position in entities_to_redistribute:
            inserted = False
            for child in self.children:
                if child.insert(entity, position):
                    inserted = True
                    break
            if not inserted:
                self.entities.append((entity, position))
        self.is_divided = True

    def query_range(
        self, range_bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        results: List[Tuple[Any, Tuple[float, float]]] = []
        if not self._intersects_range(range_bounds):
            return results
        if not self.is_divided:
            for entity, position in self.entities:
                if self._point_in_range(position, range_bounds):
                    results.append((entity, position))
            return results
        if self.children:
            for child in self.children:
                results.extend(child.query_range(range_bounds))
        for entity, position in self.entities:
            if self._point_in_range(position, range_bounds):
                results.append((entity, position))
        return results

    def query_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        results: List[Tuple[Any, Tuple[float, float]]] = []
        x, y = center
        bbox = (x - radius, y - radius, radius * 2, radius * 2)
        bbox_entities = self.query_range(bbox)
        for entity, position in bbox_entities:
            if self._distance(center, position) <= radius:
                results.append((entity, position))
        return results

    def remove(self, entity: Any, position: Tuple[float, float]) -> bool:
        if not self._contains_point(position):
            return False
        if not self.is_divided:
            for i, (ent, pos) in enumerate(self.entities):
                if ent is entity:
                    self.entities.pop(i)
                    return True
            return False
        if self.children:
            for child in self.children:
                if child.remove(entity, position):
                    return True
        for i, (ent, pos) in enumerate(self.entities):
            if ent is entity:
                self.entities.pop(i)
                return True
        return False

    def clear(self) -> None:
        self.entities.clear()
        if self.children:
            for child in self.children:
                child.clear()
            self.children = None
        self.is_divided = False

    def get_stats(self) -> Dict[str, Any]:
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
        px, py = point
        x, y, width, height = self.bounds
        return x <= px < x + width and y <= py < y + height

    def _intersects_range(
        self, range_bounds: Tuple[float, float, float, float]
    ) -> bool:
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
        px, py = point
        rx, ry, rwidth, rheight = range_bounds
        return rx <= px < rx + rwidth and ry <= py < ry + rheight

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class Quadtree:
    """Quadtree wrapper exposing common operations."""

    def __init__(self, bounds: Tuple[float, float, float, float], capacity: int = 4):
        self.root = QuadtreeNode(bounds, capacity)
        self.bounds = bounds
        self.capacity = capacity

    def insert(self, entity: Any, position: Tuple[float, float]) -> bool:
        return self.root.insert(entity, position)

    def remove(self, entity: Any, position: Tuple[float, float]) -> bool:
        return self.root.remove(entity, position)

    def query_range(
        self, bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        return self.root.query_range(bounds)

    def query_radius(
        self, center: Tuple[float, float], radius: float
    ) -> List[Tuple[Any, Tuple[float, float]]]:
        return self.root.query_radius(center, radius)

    def clear(self) -> None:
        self.root.clear()

    def get_stats(self) -> Dict[str, Any]:
        return self.root.get_stats()
