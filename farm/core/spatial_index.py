"""Spatial indexing system using KD-trees for efficient spatial queries.

This module provides a SpatialIndex class that manages KD-trees for agents and resources,
with optimized change detection and efficient spatial querying capabilities.
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
import heapq

logger = logging.getLogger(__name__)


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
            QuadtreeNode((x + half_width, y + half_height, half_width, half_height), self.capacity),
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

    def query_range(self, range_bounds: Tuple[float, float, float, float]) -> List[Tuple[Any, Tuple[float, float]]]:
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

    def query_radius(self, center: Tuple[float, float], radius: float) -> List[Tuple[Any, Tuple[float, float]]]:
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

    def _intersects_range(self, range_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if a rectangular range intersects this node's bounds."""
        rx, ry, rwidth, rheight = range_bounds
        nx, ny, nwidth, nheight = self.bounds

        return (rx < nx + nwidth and rx + rwidth > nx and
                ry < ny + nheight and ry + rheight > ny)

    def _point_in_range(self, point: Tuple[float, float], range_bounds: Tuple[float, float, float, float]) -> bool:
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

    def query_range(self, bounds: Tuple[float, float, float, float]) -> List[Tuple[Any, Tuple[float, float]]]:
        """Query all entities within a rectangular range."""
        return self.root.query_range(bounds)

    def query_radius(self, center: Tuple[float, float], radius: float) -> List[Tuple[Any, Tuple[float, float]]]:
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
    Efficient spatial indexing using KD-trees with optimized change detection.

    This index maintains separate KD-trees for agents and resources and supports
    additional named indices. It uses multi-stage change detection (dirty flag,
    count deltas, and position hashing) to avoid unnecessary KD-tree rebuilds.

    Capabilities:
    - O(log n) nearest/nearby queries across one or more indices
    - Named index registration for custom data sources
    - Relaxed bounds validation to tolerate edge cases
    - Deterministic caching of alive agents for query post-processing
    """

    def __init__(
        self,
        width: float,
        height: float,
        index_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        index_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the spatial index.

        Parameters
        ----------
        width : float
            Width of the environment
        height : float
            Height of the environment
        """
        self.width = width
        self.height = height

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

    def update(self) -> None:
        """Smart KD-tree update with optimized change detection.

        Uses a multi-level optimization strategy:
        1. Dirty flag check (O(1)) - fastest for no-change scenarios
        2. Count-based check (O(1)) - catches structural changes
        3. Hash-based verification (O(n)) - ensures correctness
        """
        # Fast path: if positions are not dirty, skip all expensive operations
        if not self._positions_dirty:
            # Only update named indices if they haven't been initialized yet
            if self.agent_kdtree is None or self.resource_kdtree is None:
                self._update_named_indices()
            return

        # Precompute alive agents once to avoid redundant computation
        alive_agents = [agent for agent in self._agents if getattr(agent, "alive", False)]
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
            alive_agents = [agent for agent in self._agents if getattr(agent, "alive", False)]

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
            "positions": None,
            "cached_items": None,
            "cached_count": None,
            "cached_hash": None,
            "positions_dirty": True,
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
        """Rebuild spatial index (KD-tree or Quadtree) for a specific named index."""
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
                positions = np.array([state["position_getter"](it) for it in valid_items])
                kdtree = cKDTree(positions)
            else:
                positions = None
                kdtree = None

            state["cached_items"] = valid_items
            state["positions"] = positions
            state["kdtree"] = kdtree
            state["quadtree"] = None  # Clear quadtree if switching

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

                positions = np.array([state["position_getter"](it) for it in valid_items])
            else:
                quadtree = None
                positions = None

            state["cached_items"] = valid_items
            state["positions"] = positions
            state["quadtree"] = quadtree
            state["kdtree"] = None  # Clear kdtree if switching

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
                entities_and_positions = state["quadtree"].query_radius(position, radius)
                results[name] = [entity for entity, pos in entities_and_positions]

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

            else:
                results[name] = None
        return results

    def _quadtree_nearest(self, quadtree: Quadtree, position: Tuple[float, float]) -> Optional[Any]:
        """Find nearest entity in a quadtree using best-first search with pruning.

        Explores nodes ordered by the minimum possible distance from the query point
        to the node's bounds. Prunes subtrees whose min-distance exceeds the best
        entity distance found so far.
        """
        if quadtree is None or quadtree.root is None:
            return None

        def rect_min_distance_sq(point: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> float:
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
        heapq.heappush(heap, (rect_min_distance_sq(position, quadtree.root.bounds), quadtree.root))

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
        """Update an entity's position in all quadtree indices.

        For KD-tree indices, this will mark positions as dirty for rebuild on next query.
        For Quadtree indices, this will efficiently update the entity's position.

        Parameters
        ----------
        entity : Any
            The entity whose position is being updated
        old_position : Tuple[float, float]
            The entity's old (x, y) position
        new_position : Tuple[float, float]
            The entity's new (x, y) position
        """
        for name, state in self._named_indices.items():
            index_type = state.get("index_type", "kdtree")

            if index_type == "quadtree" and state["quadtree"] is not None:
                # Remove from old position and insert at new position
                state["quadtree"].remove(entity, old_position)
                state["quadtree"].insert(entity, new_position)

                # Update cached position data for change detection
                if state["positions"] is not None:
                    # Find and update the position in the cached positions array
                    cached_items = state["cached_items"] or []
                    for i, cached_entity in enumerate(cached_items):
                        if cached_entity is entity:
                            if i < len(state["positions"]):
                                state["positions"][i] = new_position
                            break

                # Mark as dirty to trigger hash recalculation
                state["positions_dirty"] = True

            elif index_type == "kdtree":
                # For KD-trees, just mark positions as dirty for rebuild
                state["positions_dirty"] = True

        # Also mark main positions as dirty
        self._positions_dirty = True

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

            elif index_type == "kdtree" and state["kdtree"] is not None and state["cached_items"]:
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
                    radius = ((width/2)**2 + (height/2)**2)**0.5

                    # Query with large radius
                    indices = state["kdtree"].query_ball_point((center_x, center_y), radius)

                    # Filter to actual rectangular bounds
                    entities_in_range = []
                    for i in indices:
                        if i < len(positions):
                            px, py = positions[i]
                            if (x <= px < x + width and y <= py < y + height):
                                entities_in_range.append(cached_items[i])
                    results[name] = entities_in_range
                else:
                    results[name] = []

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
        if state is None or state.get("index_type") != "quadtree" or state["quadtree"] is None:
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
