"""Spatial indexing system using KD-trees for efficient spatial queries.

This module provides a SpatialIndex class that manages KD-trees for agents and resources,
with optimized change detection and efficient spatial querying capabilities.
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class SpatialIndex:
    """Efficient spatial indexing using KD-trees with optimized change detection.

    This class manages separate KD-trees for agents and resources, providing
    O(log n) spatial queries with smart update strategies to minimize rebuilds.
    """

    def __init__(self, width: float, height: float, index_configs: Optional[Dict[str, Dict[str, Any]]] = None, index_data: Optional[Dict[str, Any]] = None):
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
                data_reference=data_ref_or_getter if isinstance(data_ref_or_getter, list) else None,
                data_getter=data_ref_or_getter if callable(data_ref_or_getter) else None,
                position_getter=cfg.get("position_getter", lambda x: getattr(x, "position", None)),
                filter_func=cfg.get("filter_func", None),
            )

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
        # Skip rebuild checks if positions are not dirty, but still ensure
        # named indices are initialized or refreshed as needed.
        if not self._positions_dirty:
            self._update_named_indices()
            return

        # Precompute alive agents once to avoid redundant computation
        alive_agents = [agent for agent in self._agents if agent.alive]
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
        current_agent_positions = np.array([agent.position for agent in alive_agents]) if alive_agents else None
        current_resource_positions = (
            np.array([resource.position for resource in self._resources]) if self._resources else None
        )

        # Calculate hash of current agent positions
        if current_agent_positions is not None and len(current_agent_positions) > 0:
            agent_hash = hashlib.md5(current_agent_positions.tobytes()).hexdigest()
        else:
            agent_hash = "0"

        # Calculate hash of current resource positions
        if current_resource_positions is not None and len(current_resource_positions) > 0:
            resource_hash = hashlib.md5(current_resource_positions.tobytes()).hexdigest()
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
            alive_agents = [agent for agent in self._agents if agent.alive]

        self._cached_alive_agents = alive_agents  # Cache contains only alive agents for efficient queries
        if alive_agents:
            self.agent_positions = np.array([agent.position for agent in alive_agents])
            self.agent_kdtree = cKDTree(self.agent_positions)
        else:
            self.agent_kdtree = None
            self.agent_positions = None

        # Update resource KD-tree
        if self._resources:
            self.resource_positions = np.array([resource.position for resource in self._resources])
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
        """
        if position_getter is None:
            position_getter = lambda x: getattr(x, "position", None)

        self._named_indices[name] = {
            "data_reference": data_reference,
            "data_getter": data_getter,
            "position_getter": position_getter,
            "filter_func": filter_func,
            "kdtree": None,
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
                # Compute hash and counts for consistency
                current_items = state["cached_items"] or []
                current_positions = (
                    np.array([state["position_getter"](it) for it in current_items])
                    if current_items
                    else None
                )
                if current_positions is not None and len(current_positions) > 0:
                    curr_hash = hashlib.md5(current_positions.tobytes()).hexdigest()
                else:
                    curr_hash = "0"
                state["cached_count"] = len(current_items)
                state["cached_hash"] = curr_hash
                continue
            if name == "resources":
                state["kdtree"] = self.resource_kdtree
                state["positions"] = self.resource_positions
                state["cached_items"] = self._resources
                state["positions_dirty"] = False
                current_items = state["cached_items"] or []
                current_positions = (
                    np.array([state["position_getter"](it) for it in current_items])
                    if current_items
                    else None
                )
                if current_positions is not None and len(current_positions) > 0:
                    curr_hash = hashlib.md5(current_positions.tobytes()).hexdigest()
                else:
                    curr_hash = "0"
                state["cached_count"] = len(current_items)
                state["cached_hash"] = curr_hash
                continue

            # For custom indices, rebuild if marked dirty
            if state.get("positions_dirty", True):
                self._rebuild_named_index(name)
                state["positions_dirty"] = False

    def _rebuild_named_index(self, name: str) -> None:
        """Rebuild KD-tree for a specific named index."""
        state = self._named_indices[name]
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

        # Build positions array
        if filtered_items:
            positions = np.array([state["position_getter"](it) for it in filtered_items])
            kdtree = cKDTree(positions)
        else:
            positions = None
            kdtree = None

        state["cached_items"] = filtered_items
        state["positions"] = positions
        state["kdtree"] = kdtree
        state["cached_count"] = len(filtered_items)
        if positions is not None and len(positions) > 0:
            state["cached_hash"] = hashlib.md5(positions.tobytes()).hexdigest()
        else:
            state["cached_hash"] = "0"

    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List:
        """Find all agents within radius of position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search around
        radius : float
            Search radius

        Returns
        -------
        list
            List of agents within radius
        """
        # Ensure KD-trees are up to date
        self.update()

        # Input validation
        if radius <= 0:
            return []
        if not self._is_valid_position(position):
            return []

        if self.agent_kdtree is None or self._cached_alive_agents is None:
            return []

        # Use cached alive agents for direct indexing
        indices = self.agent_kdtree.query_ball_point(position, radius)
        # Return agents directly since cache only contains alive agents
        return [self._cached_alive_agents[i] for i in indices]

    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List:
        """Find all resources within radius of position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search around
        radius : float
            Search radius

        Returns
        -------
        list
            List of resources within radius
        """
        # Ensure KD-trees are up to date
        self.update()

        # Input validation (same as get_nearby_agents)
        if radius <= 0:
            return []
        if not self._is_valid_position(position):
            return []

        if self.resource_kdtree is None:
            return []

        indices = self.resource_kdtree.query_ball_point(position, radius)
        return [self._resources[i] for i in indices]

    def get_nearby(
        self,
        position: Tuple[float, float],
        radius: float,
        index_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Generic nearby query across one or more named indices.

        By default, searches across all registered indices.
        """
        self.update()

        # Input validation (reuse same checks)
        if radius <= 0 or not self._is_valid_position(position):
            return {}

        names = index_names or list(self._named_indices.keys())
        results: Dict[str, List[Any]] = {}
        for name in names:
            state = self._named_indices.get(name)
            if state is None or state["kdtree"] is None:
                results[name] = []
                continue
            indices = state["kdtree"].query_ball_point(position, radius)
            cached_items = state["cached_items"] or []
            results[name] = [cached_items[i] for i in indices]
        return results

    def get_nearest_resource(self, position: Tuple[float, float]):
        """Find nearest resource to position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search from

        Returns
        -------
        Resource or None
            Nearest resource if any exist
        """
        # Ensure KD-trees are up to date
        self.update()

        # Input validation
        if not self._is_valid_position(position):
            return None

        if self.resource_kdtree is None:
            return None

        distance, index = self.resource_kdtree.query(position)
        return self._resources[index]

    def get_nearest(
        self, position: Tuple[float, float], index_names: Optional[List[str]] = None
    ) -> Dict[str, Optional[Any]]:
        """Generic nearest query across one or more named indices.

        Returns a mapping from index name to nearest item (or None).
        """
        self.update()
        if not self._is_valid_position(position):
            return {}

        names = index_names or list(self._named_indices.keys())
        results: Dict[str, Optional[Any]] = {}
        for name in names:
            state = self._named_indices.get(name)
            if state is None or state["kdtree"] is None or not state["cached_items"]:
                results[name] = None
                continue
            _, idx = state["kdtree"].query(position)
            results[name] = state["cached_items"][idx]
        return results

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
        return (-margin_x <= x <= self.width + margin_x) and (-margin_y <= y <= self.height + margin_y)

    def get_agent_count(self) -> int:
        """Get the number of alive agents.

        Returns
        -------
        int
            Number of alive agents
        """
        return len([a for a in self._agents if a.alive])

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

    def force_rebuild(self) -> None:
        """Force a rebuild of the KD-trees regardless of change detection."""
        self._rebuild_kdtrees()
        self._positions_dirty = False

    def get_stats(self) -> dict:
        """Get statistics about the spatial index.

        Returns
        -------
        dict
            Dictionary containing spatial index statistics
        """
        return {
            "agent_count": self.get_agent_count(),
            "resource_count": self.get_resource_count(),
            "agent_kdtree_exists": self.agent_kdtree is not None,
            "resource_kdtree_exists": self.resource_kdtree is not None,
            "positions_dirty": self._positions_dirty,
            "cached_counts": self._cached_counts,
            "cached_hash": (self._cached_hash[:20] + "..." if self._cached_hash else None),
        }
