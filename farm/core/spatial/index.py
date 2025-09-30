"""SpatialIndex orchestrator for KD-tree, Quadtree, and Spatial Hash indices."""

import hashlib
import heapq
import logging
import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .dirty_regions import DirtyRegionTracker
from .hash_grid import SpatialHashGrid
from .quadtree import Quadtree, QuadtreeNode


logger = logging.getLogger(__name__)


class SpatialIndex:
    """
    Efficient spatial indexing using KD-trees with optimized change detection and batch updates.

    This index maintains separate KD-trees for agents and resources and supports
    additional named indices. It uses multi-stage change detection (dirty flag,
    count deltas, and position hashing) to avoid unnecessary KD-tree rebuilds.
    Enhanced with batch spatial updates and dirty region tracking for improved performance.
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
        self.width = width
        self.height = height
        self._initial_batch_updates_enabled = enable_batch_updates
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

        # References
        self._agents: List = []
        self._resources: List = []

        # Named indices
        self._named_indices: Dict[str, Dict[str, Any]] = {}
        self._initial_index_configs = index_configs or {}
        self._initial_index_data = index_data or {}

        # Batch update system
        if self._initial_batch_updates_enabled:
            self._dirty_region_tracker = DirtyRegionTracker(
                region_size=region_size,
                max_regions=max(
                    1000, int((width * height) / (region_size * region_size))
                ),
            )
            self._pending_position_updates: List[
                Tuple[Any, Tuple[float, float], Tuple[float, float], str, int]
            ] = []
            self._batch_update_enabled = True
        else:
            self._dirty_region_tracker = None
            self._pending_position_updates = []
            self._batch_update_enabled = False

        self._batch_update_stats = {
            "total_batch_updates": 0,
            "total_individual_updates": 0,
            "total_regions_processed": 0,
            "average_batch_size": 0.0,
            "last_batch_time": 0.0,
        }

    def set_references(self, agents: List, resources: List) -> None:
        self._agents = agents
        self._resources = resources

        try:
            if any(isinstance(a, (str, bytes)) for a in agents):
                logger.warning(
                    "SpatialIndex.set_references received agent IDs (strings) instead of agent objects. "
                    "This may cause empty agent indices. Ensure agent objects are passed."
                )
        except (TypeError, AttributeError):
            pass

        # Defaults
        self.register_index(
            name="agents",
            data_reference=self._agents,
            position_getter=lambda a: a.position,
            filter_func=lambda a: getattr(a, "alive", True),
        )
        self.register_index(
            name="resources",
            data_reference=self._resources,
            position_getter=lambda r: r.position,
            filter_func=None,
        )

        for name, cfg in self._initial_index_configs.items():
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
                index_type=cfg.get("index_type", "kdtree"),
                cell_size=cfg.get("cell_size"),
            )

        self.mark_positions_dirty()

    def mark_positions_dirty(self) -> None:
        self._positions_dirty = True
        for idx in self._named_indices.values():
            idx["positions_dirty"] = True

    def add_position_update(
        self,
        entity: Any,
        old_position: Tuple[float, float],
        new_position: Tuple[float, float],
        entity_type: str = "agent",
        priority: int = 0,
    ) -> None:
        if not self._batch_update_enabled:
            # Fall back to immediate update
            self.update_entity_position(entity, old_position, new_position)
            return

        # Add to pending updates
        self._pending_position_updates.append(
            (entity, old_position, new_position, entity_type, priority)
        )

        # Mark regions as dirty
        if self._dirty_region_tracker:
            self._dirty_region_tracker.mark_region_dirty(
                old_position, entity_type, priority
            )
            self._dirty_region_tracker.mark_region_dirty(
                new_position, entity_type, priority
            )

        # Process batch if it's full
        if len(self._pending_position_updates) >= self.max_batch_size:
            self.process_batch_updates()

    def process_batch_updates(self, force: bool = False) -> None:
        if not self._batch_update_enabled:
            return
        # If no queued objects, nothing to do
        if not self._pending_position_updates:
            return
        # If not forced, only process when queue reaches batch size
        if not force and len(self._pending_position_updates) < self.max_batch_size:
            return
        start_time = time.time()

        # Group updates by entity type for efficient processing
        updates_by_type = defaultdict(list)
        for (
            entity,
            old_pos,
            new_pos,
            entity_type,
            priority,
        ) in self._pending_position_updates:
            updates_by_type[entity_type].append((entity, old_pos, new_pos, priority))

        # Process each entity type
        regions_processed = 0
        for entity_type, updates in updates_by_type.items():
            # Get dirty regions for this entity type
            dirty_regions = (
                self._dirty_region_tracker.get_dirty_regions(entity_type)
                if self._dirty_region_tracker
                else []
            )

            # Process updates for this entity type
            for entity, old_pos, new_pos, _priority in updates:
                self._process_single_position_update(
                    entity, old_pos, new_pos, entity_type
                )

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
        total_batches = self._batch_update_stats["total_batch_updates"]
        prev_avg = self._batch_update_stats["average_batch_size"]
        if total_batches <= 1:
            new_avg = float(batch_size)
        else:
            new_avg = ((prev_avg * (total_batches - 1)) + batch_size) / float(
                total_batches
            )
        self._batch_update_stats["average_batch_size"] = new_avg
        self._batch_update_stats["last_batch_time"] = end_time - start_time

        logger.debug(
            "Processed batch update: %d entities, %d regions, %.3f seconds",
            batch_size,
            regions_processed,
            end_time - start_time,
        )

    def _process_single_position_update(
        self,
        entity: Any,
        old_position: Tuple[float, float],
        new_position: Tuple[float, float],
        entity_type: str,
    ) -> None:
        for _name, state in self._named_indices.items():
            index_type = state.get("index_type", "kdtree")
            if index_type == "quadtree" and state["quadtree"] is not None:
                state["quadtree"].remove(entity, old_position)
                state["quadtree"].insert(entity, new_position)
                if state["positions"] is not None:
                    cached_items = state["cached_items"] or []
                    for i, cached_entity in enumerate(cached_items):
                        if cached_entity is entity and i < len(state["positions"]):
                            state["positions"][i] = new_position
                            break
            elif index_type == "spatial_hash" and state["spatial_hash"] is not None:
                state["spatial_hash"].move(entity, old_position, new_position)
                if state["positions"] is not None:
                    cached_items = state["cached_items"] or []
                    for i, cached_entity in enumerate(cached_items):
                        if cached_entity is entity and i < len(state["positions"]):
                            state["positions"][i] = new_position
                            break
            elif index_type == "kdtree":
                state["positions_dirty"] = True
        self._positions_dirty = True

    def get_batch_update_stats(self) -> Dict[str, Any]:
        stats = dict(self._batch_update_stats)
        if self._dirty_region_tracker:
            stats.update(self._dirty_region_tracker.get_stats())
        return stats

    def enable_batch_updates(
        self, region_size: float = 50.0, max_batch_size: int = 100
    ) -> None:
        if not self._batch_update_enabled:
            self._dirty_region_tracker = DirtyRegionTracker(
                region_size=region_size,
                max_regions=max(
                    1000, int((self.width * self.height) / (region_size * region_size))
                ),
            )
            self._batch_update_enabled = True
            self.max_batch_size = max_batch_size
            logger.info(
                "Batch updates enabled with region_size=%s, max_batch_size=%s",
                region_size,
                max_batch_size,
            )

    def disable_batch_updates(self) -> None:
        if self._batch_update_enabled:
            self.process_batch_updates(force=True)
            self._batch_update_enabled = False
            self._dirty_region_tracker = None
            logger.info("Batch updates disabled")

    def update(self) -> None:
        if self._batch_update_enabled and self._pending_position_updates:
            self.process_batch_updates(force=True)
        if not self._positions_dirty:
            if self.agent_kdtree is None or self.resource_kdtree is None:
                self._update_named_indices()
            return
        alive_agents = [
            agent for agent in self._agents if getattr(agent, "alive", False)
        ]
        current_agent_count = len(alive_agents)
        if self._counts_changed(current_agent_count):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return
        if self._hash_positions_changed(alive_agents):
            self._rebuild_kdtrees(alive_agents)
            self._positions_dirty = False
            return
        self._update_named_indices()
        self._positions_dirty = False

    def _counts_changed(self, current_agent_count: int) -> bool:
        current_resource_count = len(self._resources)
        current_counts = (current_agent_count, current_resource_count)
        if self._cached_counts is None or self._cached_counts != current_counts:
            self._cached_counts = current_counts
            return True
        return False

    def _hash_positions_changed(self, alive_agents: List) -> bool:
        valid_alive_agents = [
            agent
            for agent in alive_agents
            if getattr(agent, "position", None) is not None
        ]
        valid_resources = [
            resource
            for resource in self._resources
            if getattr(resource, "position", None) is not None
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
        if current_agent_positions is not None and len(current_agent_positions) > 0:
            agent_hash = hashlib.md5(current_agent_positions.tobytes()).hexdigest()
        else:
            agent_hash = "0"
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
        if alive_agents is None:
            alive_agents = [
                agent for agent in self._agents if getattr(agent, "alive", False)
            ]
        alive_agents = [
            agent
            for agent in alive_agents
            if getattr(agent, "position", None) is not None
        ]
        self._cached_alive_agents = alive_agents
        if alive_agents:
            self.agent_positions = np.array([agent.position for agent in alive_agents])
            self.agent_kdtree = cKDTree(self.agent_positions)
        else:
            self.agent_kdtree = None
            self.agent_positions = None
        valid_resources = [
            resource
            for resource in self._resources
            if getattr(resource, "position", None) is not None
        ]
        if valid_resources:
            self.resource_positions = np.array(
                [resource.position for resource in valid_resources]
            )
            self.resource_kdtree = cKDTree(self.resource_positions)
        else:
            self.resource_kdtree = None
            self.resource_positions = None
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
        for name, state in self._named_indices.items():
            if name == "agents":
                state["kdtree"] = self.agent_kdtree
                state["positions"] = self.agent_positions
                state["cached_items"] = self._cached_alive_agents
                state["positions_dirty"] = False
                if state.get("cached_count") is None:
                    current_items = state["cached_items"] or []
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
                    state["cached_items"] = valid_items
                continue
            if name == "resources":
                state["kdtree"] = self.resource_kdtree
                state["positions"] = self.resource_positions
                state["cached_items"] = self._resources
                state["positions_dirty"] = False
                if state.get("cached_count") is None:
                    current_items = state["cached_items"] or []
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
                    state["cached_items"] = valid_items
                continue
            if state.get("positions_dirty", True):
                self._rebuild_named_index(name)
                state["positions_dirty"] = False

    def _rebuild_named_index(self, name: str) -> None:
        state = self._named_indices[name]
        index_type = state.get("index_type", "kdtree")
        items = None
        if state["data_getter"] is not None:
            items = state["data_getter"]()
        elif state["data_reference"] is not None:
            items = state["data_reference"]
        else:
            items = []
        if state["filter_func"] is not None:
            filtered_items = [it for it in items if state["filter_func"](it)]
        else:
            filtered_items = list(items)
        valid_items = [
            it for it in filtered_items if state["position_getter"](it) is not None
        ]

        if index_type == "kdtree":
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
            state["quadtree"] = None
            state["spatial_hash"] = None
        elif index_type == "quadtree":
            if valid_items:
                # Use exclusive bounds to avoid boundary issues with quadtree subdivision
                bounds = (0, 0, self.width - 1e-10, self.height - 1e-10)
                quadtree = Quadtree(bounds, capacity=4)
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
            state["kdtree"] = None
            state["spatial_hash"] = None
        elif index_type == "spatial_hash":
            if valid_items:
                cs = state.get("cell_size")
                if cs is None:
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
        self.update()
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
                entities_and_positions = state["quadtree"].query_radius(
                    position, radius
                )
                results[name] = [entity for entity, _ in entities_and_positions]
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
        heap: List[Tuple[float, QuadtreeNode]] = []
        heapq.heappush(
            heap, (rect_min_distance_sq(position, quadtree.root.bounds), quadtree.root)
        )
        while heap:
            min_possible_sq, node = heapq.heappop(heap)
            if min_possible_sq >= best_dist_sq:
                break
            for entity, entity_pos in node.entities:
                d2 = distance_sq(position, entity_pos)
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_entity = entity
            if node.children:
                for child in node.children:
                    child_min_sq = rect_min_distance_sq(position, child.bounds)
                    if child_min_sq < best_dist_sq:
                        heapq.heappush(heap, (child_min_sq, child))
        return best_entity

    def _is_valid_position(self, position: Tuple[float, float]) -> bool:
        x, y = position
        margin_x = self.width * 0.01
        margin_y = self.height * 0.01
        return (-margin_x <= x <= self.width + margin_x) and (
            -margin_y <= y <= self.height + margin_y
        )

    def get_agent_count(self) -> int:
        return len([a for a in self._agents if getattr(a, "alive", False)])

    def get_resource_count(self) -> int:
        return len(self._resources)

    def is_dirty(self) -> bool:
        return self._positions_dirty

    def update_entity_position(
        self,
        entity: Any,
        old_position: Tuple[float, float],
        new_position: Tuple[float, float],
    ) -> None:
        entity_type = "agent"
        try:
            resources_state = self._named_indices.get("resources")
            if resources_state and resources_state.get("cached_items"):
                if entity in resources_state["cached_items"]:
                    entity_type = "resource"
            else:
                if entity in self._resources:
                    entity_type = "resource"
        except Exception:
            entity_type = "agent"

        if self._batch_update_enabled:
            self.add_position_update(entity, old_position, new_position, entity_type)
            return
        self._process_single_position_update(
            entity, old_position, new_position, entity_type
        )

    def force_rebuild(self) -> None:
        self._rebuild_kdtrees()
        self._positions_dirty = False

    def get_nearby_range(
        self,
        bounds: Tuple[float, float, float, float],
        index_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        self.update()
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
                entities_and_positions = state["quadtree"].query_range(bounds)
                results[name] = [entity for entity, _ in entities_and_positions]
            elif (
                index_type == "kdtree"
                and state["kdtree"] is not None
                and state["cached_items"]
            ):
                cached_items = state["cached_items"]
                positions = state["positions"]
                if positions is not None:
                    center_x = x + width / 2
                    center_y = y + height / 2
                    radius = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5
                    indices = state["kdtree"].query_ball_point(
                        (center_x, center_y), radius
                    )
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
        state = self._named_indices.get(index_name)
        if (
            state is None
            or state.get("index_type") != "quadtree"
            or state["quadtree"] is None
        ):
            return None
        return state["quadtree"].get_stats()

    def get_stats(self) -> dict:
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
