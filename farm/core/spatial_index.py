"""Spatial indexing system using KD-trees for efficient spatial queries.

This module provides a SpatialIndex class that manages KD-trees for agents and resources,
with optimized change detection and efficient spatial querying capabilities.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class SpatialIndex:
    """Efficient spatial indexing using KD-trees with optimized change detection.

    This class manages separate KD-trees for agents and resources, providing
    O(log n) spatial queries with smart update strategies to minimize rebuilds.
    """

    def __init__(self, width: float, height: float, indices: Optional[List[str]] = None):
        """Initialize the spatial index.

        Parameters
        ----------
        width : float
            Width of the environment
        height : float
            Height of the environment
        indices : Optional[List[str]]
            Which indices to enable. Supported values: "agents", "resources", "obstacles".
            Defaults to ["agents", "resources"].
        """
        self.width = width
        self.height = height

        # Enabled index types
        enabled = set(indices or ["agents", "resources"])  # backward-compatible default
        self._use_agents: bool = "agents" in enabled
        self._use_resources: bool = "resources" in enabled
        self._use_obstacles: bool = "obstacles" in enabled

        # KD-tree attributes
        self.agent_kdtree: Optional[cKDTree] = None
        self.resource_kdtree: Optional[cKDTree] = None
        self.obstacle_kdtree: Optional[cKDTree] = None
        self.agent_positions: Optional[np.ndarray] = None
        self.resource_positions: Optional[np.ndarray] = None
        self.obstacle_positions: Optional[np.ndarray] = None

        # Position change tracking for optimized updates
        self._positions_dirty: bool = True
        # Backward-compatible count cache (agents, resources)
        self._cached_counts: Optional[Tuple[int, int]] = None
        # Extended cache for obstacles without breaking existing tests
        self._cached_obstacle_count: Optional[int] = None
        self._cached_hash: Optional[str] = None

        # Cached alive agents for efficient querying
        self._cached_alive_agents: Optional[List] = None

        # Reference to agent, resource, and obstacle lists
        self._agents: List = []
        self._resources: List = []
        self._obstacles: List = []

    def set_references(self, agents: List, resources: List, obstacles: Optional[List] = None) -> None:
        """Set references to agent and resource lists.

        Parameters
        ----------
        agents : List
            List of agents in the environment
        resources : List
            List of resources in the environment
        obstacles : List, optional
            List of obstacles in the environment (optional)
        """
        self._agents = agents
        self._resources = resources
        self._obstacles = obstacles or []
        # Mark dirty so indices can be rebuilt with new references
        self.mark_positions_dirty()

    def mark_positions_dirty(self) -> None:
        """Mark that positions have changed and KD-trees need updating."""
        self._positions_dirty = True

    def update(self) -> None:
        """Smart KD-tree update with optimized change detection.

        Uses a multi-level optimization strategy:
        1. Dirty flag check (O(1)) - fastest for no-change scenarios
        2. Count-based check (O(1)) - catches structural changes
        3. Hash-based verification (O(n)) - ensures correctness
        """
        # Skip update if positions are not dirty
        if not self._positions_dirty:
            return

        # Precompute alive agents once to avoid redundant computation
        alive_agents = [agent for agent in self._agents if getattr(agent, "alive", True)]
        current_agent_count = len(alive_agents)
        current_obstacle_count = len(self._obstacles) if self._use_obstacles else 0

        # Count-based quick check for structural changes
        if self._counts_changed(current_agent_count):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return

        # Check obstacle count changes only when obstacle indexing is enabled
        if self._use_obstacles and self._obstacle_count_changed(current_obstacle_count):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return

        # Hash-based verification for position changes
        if self._hash_positions_changed(alive_agents):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return

        # No changes detected, clear dirty flag
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
        current_obstacle_positions = (
            np.array([ob.position for ob in self._obstacles]) if (self._use_obstacles and self._obstacles) else None
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

        # Calculate hash of current obstacle positions (optional index)
        if self._use_obstacles:
            if current_obstacle_positions is not None and len(current_obstacle_positions) > 0:
                obstacle_hash = hashlib.md5(current_obstacle_positions.tobytes()).hexdigest()
            else:
                obstacle_hash = "0"
            current_hash = f"{agent_hash}:{resource_hash}:{obstacle_hash}"
        else:
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
            alive_agents = [agent for agent in self._agents if getattr(agent, "alive", True)]

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

        # Update obstacle KD-tree (optional index)
        if self._use_obstacles and self._obstacles:
            self.obstacle_positions = np.array([ob.position for ob in self._obstacles])
            self.obstacle_kdtree = cKDTree(self.obstacle_positions)
        else:
            self.obstacle_kdtree = None
            self.obstacle_positions = None

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

    def get_nearby_obstacles(self, position: Tuple[float, float], radius: float) -> List:
        """Find all obstacles within radius of position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search around
        radius : float
            Search radius

        Returns
        -------
        list
            List of obstacles within radius
        """
        # Ensure KD-trees are up to date
        self.update()

        # Input validation
        if radius <= 0:
            return []
        if not self._is_valid_position(position):
            return []

        if self.obstacle_kdtree is None:
            return []

        indices = self.obstacle_kdtree.query_ball_point(position, radius)
        return [self._obstacles[i] for i in indices]

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

    def get_nearest_obstacle(self, position: Tuple[float, float]):
        """Find nearest obstacle to position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search from

        Returns
        -------
        Obstacle or None
            Nearest obstacle if any exist
        """
        # Ensure KD-trees are up to date
        self.update()

        # Input validation
        if not self._is_valid_position(position):
            return None

        if self.obstacle_kdtree is None:
            return None

        distance, index = self.obstacle_kdtree.query(position)
        return self._obstacles[index]

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

    def get_obstacle_count(self) -> int:
        """Get the number of obstacles.

        Returns
        -------
        int
            Number of obstacles
        """
        return len(self._obstacles)

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

    def _obstacle_count_changed(self, current_obstacle_count: int) -> bool:
        """Check if obstacle count has changed without altering legacy count cache."""
        if self._cached_obstacle_count is None or self._cached_obstacle_count != current_obstacle_count:
            self._cached_obstacle_count = current_obstacle_count
            return True
        return False

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
            "obstacle_count": self.get_obstacle_count(),
            "agent_kdtree_exists": self.agent_kdtree is not None,
            "resource_kdtree_exists": self.resource_kdtree is not None,
            "obstacle_kdtree_exists": self.obstacle_kdtree is not None,
            "positions_dirty": self._positions_dirty,
            "cached_counts": self._cached_counts,
            "cached_hash": (self._cached_hash[:20] + "..." if self._cached_hash else None),
        }
