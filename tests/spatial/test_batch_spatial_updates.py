"""
Tests for batch spatial updates with dirty region tracking.

This module tests the new batch spatial updates feature that improves performance
by only updating regions that have actually changed, rather than rebuilding
entire spatial indices on every position change.
"""

import pytest
import time
from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import Mock, patch
from typing import List, Tuple

from farm.core.spatial import SpatialIndex, DirtyRegionTracker, DirtyRegion
from farm.core.environment import Environment
from farm.config.config import SpatialIndexConfig, EnvironmentConfig, SimulationConfig


class TestDirtyRegionTracker:
    """Test the DirtyRegionTracker class."""

    def test_initialization(self):
        tracker = DirtyRegionTracker(region_size=25.0, max_regions=500)
        assert tracker.region_size == 25.0
        assert tracker.max_regions == 500
        assert tracker._total_regions_marked == 0
        assert tracker._total_regions_updated == 0

    def test_mark_region_dirty(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert dirty_regions[0].entity_type == "agent"
        assert dirty_regions[0].priority == 1

    def test_mark_region_dirty_batch(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        positions = [(10.0, 10.0), (60.0, 60.0), (110.0, 110.0)]
        tracker.mark_region_dirty_batch(positions, "resource", priority=2)
        dirty_regions = tracker.get_dirty_regions("resource")
        assert len(dirty_regions) == 3
        for region in dirty_regions:
            assert region.entity_type == "resource"
            assert region.priority == 2

    def test_region_coordinate_conversion(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        region_coords = tracker._world_to_region_coords((75.0, 125.0))
        assert region_coords == (1, 2)
        bounds = tracker._region_to_world_bounds((1, 2))
        assert bounds == (50.0, 100.0, 50.0, 50.0)

    def test_clear_region(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        region_coords = tracker._world_to_region_coords((25.0, 25.0))
        tracker.clear_region(region_coords)
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1

    def test_clear_all_regions(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "resource")
        tracker.clear_all_regions()
        assert len(tracker.get_dirty_regions()) == 0

    def test_priority_sorting(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        tracker.mark_region_dirty((75.0, 75.0), "agent", priority=3)
        tracker.mark_region_dirty((125.0, 125.0), "agent", priority=2)
        dirty_regions = tracker.get_dirty_regions("agent")
        priorities = [region.priority for region in dirty_regions]
        assert priorities == [3, 2, 1]

    def test_max_regions_cleanup(self):
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=3)
        for i in range(5):
            tracker.mark_region_dirty((i * 100.0, i * 100.0), "agent")
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) <= 3

    def test_get_stats(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "resource")
        stats = tracker.get_stats()
        assert stats["total_dirty_regions"] == 2
        assert stats["regions_by_type"]["agent"] == 1
        assert stats["regions_by_type"]["resource"] == 1
        assert stats["total_regions_marked"] == 2


class TestSpatialIndexBatchUpdates:
    def test_initialization_with_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            region_size=25.0,
            max_batch_size=50
        )
        assert spatial_index._initial_batch_updates_enabled is True
        assert spatial_index.max_batch_size == 50
        assert spatial_index._batch_update_enabled is True
        assert spatial_index._dirty_region_tracker is not None
        assert spatial_index._dirty_region_tracker.region_size == 25.0

    def test_initialization_without_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        assert spatial_index._initial_batch_updates_enabled is False
        assert spatial_index._batch_update_enabled is False
        assert spatial_index._dirty_region_tracker is None

    def test_add_position_update(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        entity = Mock()
        entity.position = (50.0, 50.0)
        spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent", priority=1
        )
        assert len(spatial_index._pending_position_updates) == 1

    def test_batch_processing_trigger(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=3
        )
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        assert len(spatial_index._pending_position_updates) == 0

    def test_process_batch_updates_force(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        entity = Mock()
        spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent"
        )
        spatial_index.process_batch_updates(force=True)
        assert len(spatial_index._pending_position_updates) == 0

    def test_batch_update_stats(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=2
        )
        entities = [Mock() for _ in range(2)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        stats = spatial_index.get_batch_update_stats()
        assert stats["total_batch_updates"] == 1
        assert stats["total_individual_updates"] == 2
        assert stats["average_batch_size"] == 2.0

    def test_enable_disable_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        spatial_index.enable_batch_updates(region_size=30.0, max_batch_size=20)
        assert spatial_index._batch_update_enabled is True
        assert spatial_index._dirty_region_tracker is not None
        assert spatial_index._dirty_region_tracker.region_size == 30.0
        assert spatial_index.max_batch_size == 20
        spatial_index.disable_batch_updates()
        assert spatial_index._batch_update_enabled is False
        assert spatial_index._dirty_region_tracker is None

    def test_update_entity_position_with_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        entity = Mock()
        entity.alive = True
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        assert len(spatial_index._pending_position_updates) == 1

    def test_update_entity_position_without_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        entity = Mock()
        entity.alive = True
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        assert len(spatial_index._pending_position_updates) == 0


class TestEnvironmentBatchUpdates:
    def test_environment_initialization_with_spatial_config(self):
        spatial_config = SpatialIndexConfig(
            enable_batch_updates=True,
            region_size=30.0,
            max_batch_size=25,
            enable_quadtree_indices=True
        )
        env_config = EnvironmentConfig(
            width=200,
            height=200,
            spatial_index=spatial_config
        )
        config = SimulationConfig()
        config.environment = env_config
        env = Environment(
            width=200,
            height=200,
            resource_distribution="uniform",
            config=config
        )
        assert env.spatial_index._batch_update_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 30.0
        assert env.spatial_index.max_batch_size == 25

    def test_environment_initialization_without_spatial_config(self):
        config = SimulationConfig()
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform",
            config=config
        )
        assert env.spatial_index._initial_batch_updates_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 50.0
        assert env.spatial_index.max_batch_size == 100

    def test_process_batch_spatial_updates(self):
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        entity = Mock()
        env.spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent"
        )
        env.process_batch_spatial_updates(force=True)
        assert len(env.spatial_index._pending_position_updates) == 0

    def test_get_spatial_performance_stats(self):
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        stats = env.get_spatial_performance_stats()
        assert "agent_count" in stats
        assert "resource_count" in stats
        assert "batch_updates" in stats
        assert "perception" in stats

    def test_enable_disable_batch_spatial_updates(self):
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        env.disable_batch_spatial_updates()
        assert env.spatial_index._batch_update_enabled is False
        env.enable_batch_spatial_updates(region_size=40.0, max_batch_size=30)
        assert env.spatial_index._batch_update_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 40.0
        assert env.spatial_index.max_batch_size == 30


class TestPerformanceImprovements:
    def test_region_based_efficiency(self):
        spatial_index = SpatialIndex(
            width=1000.0,
            height=1000.0,
            enable_batch_updates=True,
            region_size=100.0,
            max_batch_size=50
        )
        entities = [Mock() for _ in range(20)]
        for i in range(10):
            spatial_index.add_position_update(
                entities[i], (50.0, 50.0), (60.0, 60.0), "agent"
            )
        for i in range(10, 20):
            spatial_index.add_position_update(
                entities[i], (500.0, 500.0), (510.0, 510.0), "agent"
            )
        start_time = time.time()
        spatial_index.process_batch_updates(force=True)
        processing_time = time.time() - start_time
        assert processing_time < 0.1
        stats = spatial_index.get_batch_update_stats()
        assert stats["total_regions_processed"] > 0

    def test_reduces_update_overhead_in_dynamic_simulations(self):
        spatial_index = SpatialIndex(
            width=1000.0,
            height=1000.0,
            enable_batch_updates=True,
            region_size=50.0,
            max_batch_size=20
        )
        entities = [Mock() for _ in range(100)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity,
                (i * 5.0, i * 5.0),
                (i * 5.0 + 10.0, i * 5.0 + 10.0),
                "agent"
            )
        spatial_index.process_batch_updates(force=True)
        stats = spatial_index.get_batch_update_stats()
        assert stats["total_batch_updates"] > 0
        assert stats["total_individual_updates"] == 100
        assert stats["average_batch_size"] > 0
        assert stats["total_regions_processed"] > 0

