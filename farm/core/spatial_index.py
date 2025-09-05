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
        
        # Selective entity tracking for mobile vs immobile entities
        self._mobile_agents: set = set()  # Track mobile agent IDs
        self._mobile_resources: set = set()  # Track mobile resource IDs
        self._static_agents_hash: Optional[str] = None  # One-time hash for static agents
        self._static_resources_hash: Optional[str] = None  # One-time hash for static resources
        self._mobile_only_mode: bool = False  # When True, only check mobile entities

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

    def mark_positions_dirty(self) -> None:
        """Mark that positions have changed and KD-trees need updating."""
        self._positions_dirty = True
        # Mark all named indices as dirty as well
        for idx in self._named_indices.values():
            idx["positions_dirty"] = True
    
    def register_mobile_agent(self, agent_id: str) -> None:
        """Register an agent as mobile (can change position).
        
        Mobile agents will be checked for position changes during updates.
        Static agents are only hashed once for better performance.
        
        Parameters
        ----------
        agent_id : str
            Unique identifier of the agent to mark as mobile
        """
        self._mobile_agents.add(agent_id)
        # Invalidate static hash since agent set changed
        self._static_agents_hash = None
        logger.debug(f"Agent {agent_id} registered as mobile")
    
    def register_mobile_resource(self, resource_id: str) -> None:
        """Register a resource as mobile (can change position).
        
        Mobile resources will be checked for position changes during updates.
        Static resources are only hashed once for better performance.
        
        Parameters
        ----------
        resource_id : str
            Unique identifier of the resource to mark as mobile
        """
        self._mobile_resources.add(resource_id)
        # Invalidate static hash since resource set changed
        self._static_resources_hash = None
        logger.debug(f"Resource {resource_id} registered as mobile")
    
    def unregister_mobile_agent(self, agent_id: str) -> None:
        """Remove an agent from mobile tracking.
        
        The agent will be treated as static (immobile) going forward.
        
        Parameters
        ----------
        agent_id : str
            Unique identifier of the agent to remove from mobile tracking
        """
        self._mobile_agents.discard(agent_id)
        # Invalidate static hash since agent set changed
        self._static_agents_hash = None
        logger.debug(f"Agent {agent_id} unregistered from mobile tracking")
    
    def unregister_mobile_resource(self, resource_id: str) -> None:
        """Remove a resource from mobile tracking.
        
        The resource will be treated as static (immobile) going forward.
        
        Parameters
        ----------
        resource_id : str
            Unique identifier of the resource to remove from mobile tracking
        """
        self._mobile_resources.discard(resource_id)
        # Invalidate static hash since resource set changed  
        self._static_resources_hash = None
        logger.debug(f"Resource {resource_id} unregistered from mobile tracking")
    
    def set_mobile_only_mode(self, enabled: bool) -> None:
        """Enable or disable mobile-only change detection mode.
        
        When enabled, only mobile entities are checked for position changes.
        Static entities are hashed once and assumed to never move.
        
        Parameters
        ----------
        enabled : bool
            True to enable mobile-only mode, False to check all entities
        """
        old_mode = self._mobile_only_mode
        self._mobile_only_mode = enabled
        if old_mode != enabled:
            # Mode changed, invalidate hashes to force recalculation
            self._cached_hash = None
            self._static_agents_hash = None
            self._static_resources_hash = None
            logger.info(f"Mobile-only mode {'enabled' if enabled else 'disabled'}")
    
    def get_mobile_entities_info(self) -> dict:
        """Get information about mobile vs static entity tracking.
        
        Returns
        -------
        dict
            Dictionary containing mobile/static entity counts and mode status
        """
        return {
            "mobile_agents_count": len(self._mobile_agents),
            "mobile_resources_count": len(self._mobile_resources),
            "mobile_only_mode": self._mobile_only_mode,
            "static_agents_hash_cached": self._static_agents_hash is not None,
            "static_resources_hash_cached": self._static_resources_hash is not None,
        }

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
        """Check if positions have changed using selective hash comparison.

        In mobile-only mode, this method only checks mobile entities for changes
        and uses cached hashes for static entities. This significantly improves
        performance when most entities are static.

        Parameters
        ----------
        alive_agents : List
            Precomputed list of alive agents

        Returns
        -------
        bool
            True if positions have changed, False otherwise
        """
        if self._mobile_only_mode:
            return self._selective_hash_positions_changed(alive_agents)
        else:
            return self._full_hash_positions_changed(alive_agents)
    
    def _selective_hash_positions_changed(self, alive_agents: List) -> bool:
        """Selective change detection that only checks mobile entities.
        
        Static entities are hashed once and cached. Mobile entities are
        checked on every update. This is much faster when most entities
        are static.
        
        Parameters
        ----------
        alive_agents : List
            Precomputed list of alive agents
            
        Returns
        -------
        bool
            True if any mobile entity positions have changed, False otherwise
        """
        # Separate mobile and static agents
        mobile_agents = []
        static_agents = []
        
        for agent in alive_agents:
            if agent.position is not None:
                agent_id = getattr(agent, 'agent_id', str(id(agent)))
                if agent_id in self._mobile_agents:
                    mobile_agents.append(agent)
                else:
                    static_agents.append(agent)
        
        # Separate mobile and static resources
        mobile_resources = []
        static_resources = []
        
        for resource in self._resources:
            if resource.position is not None:
                resource_id = getattr(resource, 'resource_id', str(id(resource)))
                if resource_id in self._mobile_resources:
                    mobile_resources.append(resource)
                else:
                    static_resources.append(resource)
        
        # Calculate hash for mobile entities (checked every time)
        mobile_agent_hash = self._calculate_positions_hash(mobile_agents)
        mobile_resource_hash = self._calculate_positions_hash(mobile_resources)
        
        # Calculate or reuse hash for static entities (cached)
        static_agent_hash = self._get_static_agents_hash(static_agents)
        static_resource_hash = self._get_static_resources_hash(static_resources)
        
        # Combine all hashes
        current_hash = f"{mobile_agent_hash}:{static_agent_hash}:{mobile_resource_hash}:{static_resource_hash}"
        
        if self._cached_hash is None or self._cached_hash != current_hash:
            self._cached_hash = current_hash
            return True
        return False
    
    def _full_hash_positions_changed(self, alive_agents: List) -> bool:
        """Traditional full hash comparison of all entities (fallback method).
        
        This is the original implementation that checks all entities regardless
        of their mobility status. Used when mobile-only mode is disabled.
        
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

        # Calculate hashes
        agent_hash = self._calculate_positions_hash(valid_alive_agents)
        resource_hash = self._calculate_positions_hash(valid_resources)
        current_hash = f"{agent_hash}:{resource_hash}"

        if self._cached_hash is None or self._cached_hash != current_hash:
            self._cached_hash = current_hash
            return True
        return False
    
    def _calculate_positions_hash(self, entities: List) -> str:
        """Calculate MD5 hash of entity positions.
        
        Parameters
        ----------
        entities : List
            List of entities with position attributes
            
        Returns
        -------
        str
            MD5 hash of positions or "0" if no entities
        """
        if not entities:
            return "0"
        
        positions = np.array([entity.position for entity in entities])
        return hashlib.md5(positions.tobytes()).hexdigest()
    
    def _get_static_agents_hash(self, static_agents: List) -> str:
        """Get or calculate hash for static agents.
        
        Parameters
        ----------
        static_agents : List
            List of static (immobile) agents
            
        Returns
        -------
        str
            Cached or newly calculated hash for static agents
        """
        if self._static_agents_hash is None:
            self._static_agents_hash = self._calculate_positions_hash(static_agents)
            logger.debug(f"Cached static agents hash: {self._static_agents_hash[:8]}...")
        return self._static_agents_hash
    
    def _get_static_resources_hash(self, static_resources: List) -> str:
        """Get or calculate hash for static resources.
        
        Parameters
        ----------
        static_resources : List
            List of static (immobile) resources
            
        Returns
        -------
        str
            Cached or newly calculated hash for static resources
        """
        if self._static_resources_hash is None:
            self._static_resources_hash = self._calculate_positions_hash(static_resources)
            logger.debug(f"Cached static resources hash: {self._static_resources_hash[:8]}...")
        return self._static_resources_hash

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

        # Filter out items without valid positions
        valid_items = [
            it for it in filtered_items if state["position_getter"](it) is not None
        ]

        # Build positions array
        if valid_items:
            positions = np.array([state["position_getter"](it) for it in valid_items])
            kdtree = cKDTree(positions)
        else:
            positions = None
            kdtree = None

        state["cached_items"] = valid_items
        state["positions"] = positions
        state["kdtree"] = kdtree
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
            if state is None or state["kdtree"] is None:
                results[name] = []
                continue
            indices = state["kdtree"].query_ball_point(position, radius)
            cached_items = state["cached_items"] or []
            results[name] = [cached_items[i] for i in indices]
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
            Dictionary containing spatial index statistics including selective tracking info
        """
        mobile_info = self.get_mobile_entities_info()
        
        return {
            "agent_count": self.get_agent_count(),
            "resource_count": self.get_resource_count(),
            "agent_kdtree_exists": self.agent_kdtree is not None,
            "resource_kdtree_exists": self.resource_kdtree is not None,
            "positions_dirty": self._positions_dirty,
            "cached_counts": self._cached_counts,
            "cached_hash": (
                self._cached_hash[:20] + "..." if self._cached_hash else None
            ),
            "mobile_only_mode": mobile_info["mobile_only_mode"],
            "mobile_agents_count": mobile_info["mobile_agents_count"],
            "mobile_resources_count": mobile_info["mobile_resources_count"],
            "static_agents_cached": mobile_info["static_agents_hash_cached"],
            "static_resources_cached": mobile_info["static_resources_hash_cached"],
        }
