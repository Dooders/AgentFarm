"""
Tests for batch spatial updates with dirty region tracking.

This module tests the new batch spatial updates feature that improves performance
by only updating regions that have actually changed, rather than rebuilding
entire spatial indices on every position change.
"""

import time
from collections import defaultdict
from typing import List, Tuple
from unittest.mock import Mock, patch

import pytest

from farm.config.config import EnvironmentConfig, SimulationConfig, SpatialIndexConfig
from farm.core.environment import Environment
from farm.core.spatial import DirtyRegion, DirtyRegionTracker, SpatialIndex


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
        region_coords = tracker.world_to_region_coords((75.0, 125.0))
        assert region_coords == (1, 2)
        bounds = tracker._region_to_world_bounds((1, 2))
        assert bounds == (50.0, 100.0, 50.0, 50.0)

    def test_clear_region(self):
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        region_coords = tracker.world_to_region_coords((25.0, 25.0))
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

    def test_clear_regions_partial_processing(self):
        """Test that clear_regions works correctly with partial processing."""
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        tracker.mark_region_dirty((125.0, 125.0), "agent")

        # Clear only the first region
        region_coords = tracker.world_to_region_coords((25.0, 25.0))
        tracker.clear_region(region_coords)

        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 2

        # Clear another region
        region_coords = tracker.world_to_region_coords((75.0, 75.0))
        tracker.clear_region(region_coords)

        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1

    def test_clear_regions_by_coords_list(self):
        """Test clearing multiple regions by coordinates list."""
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        tracker.mark_region_dirty((125.0, 125.0), "agent")

        # Clear first and third regions
        coords_list = [
            tracker.world_to_region_coords((25.0, 25.0)),
            tracker.world_to_region_coords((125.0, 125.0)),
        ]
        tracker.clear_regions(coords_list)

        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        # The remaining region should be at (75.0, 75.0)
        expected_bounds = tracker._region_to_world_bounds(
            tracker.world_to_region_coords((75.0, 75.0))
        )
        assert dirty_regions[0].bounds == expected_bounds

    def test_clear_regions_nonexistent_coords(self):
        """Test clearing regions with coordinates that don't exist."""
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")

        # Try to clear a region that doesn't exist
        nonexistent_coords = (999, 999)
        tracker.clear_region(nonexistent_coords)

        # Original region should still exist
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1

    def test_clear_regions_empty_list(self):
        """Test clearing with an empty coordinates list."""
        tracker = DirtyRegionTracker(region_size=50.0)
        tracker.mark_region_dirty((25.0, 25.0), "agent")

        # Clear empty list
        tracker.clear_regions([])

        # Original region should still exist
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1

    def test_dirty_region_clearing_after_batch_processing(self):
        """Test that dirty regions are properly cleared after batch processing."""
        spatial_index = SpatialIndex(
            width=200.0,
            height=200.0,
            enable_batch_updates=True,
            region_size=50.0,
            max_batch_size=10,
        )

        # Add some position updates
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity,
                (i * 60.0, i * 60.0),
                (i * 60.0 + 10.0, i * 60.0 + 10.0),
                "agent",
            )

        # Check that regions are marked as dirty
        dirty_regions_before = spatial_index._dirty_region_tracker.get_dirty_regions(
            "agent"
        )
        assert len(dirty_regions_before) > 0

        # Process the batch
        spatial_index.process_batch_updates(force=True)

        # Check that regions are cleared
        dirty_regions_after = spatial_index._dirty_region_tracker.get_dirty_regions(
            "agent"
        )
        assert len(dirty_regions_after) == 0


class TestSpatialIndexBatchUpdates:
    def test_initialization_with_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            region_size=25.0,
            max_batch_size=50,
        )
        assert spatial_index._initial_batch_updates_enabled is True
        assert spatial_index.max_batch_size == 50
        assert spatial_index._batch_update_enabled is True
        assert spatial_index._dirty_region_tracker is not None
        assert spatial_index._dirty_region_tracker.region_size == 25.0

    def test_initialization_without_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=False
        )
        assert spatial_index._initial_batch_updates_enabled is False
        assert spatial_index._batch_update_enabled is False
        assert spatial_index._dirty_region_tracker is None

    def test_add_position_update(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=10
        )
        entity = Mock()
        entity.position = (50.0, 50.0)
        spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent", priority=1
        )
        assert len(spatial_index._pending_position_updates) == 1

    def test_batch_processing_trigger(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=3
        )
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        assert len(spatial_index._pending_position_updates) == 0

    def test_process_batch_updates_force(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=10
        )
        entity = Mock()
        spatial_index.add_position_update(entity, (25.0, 25.0), (75.0, 75.0), "agent")
        spatial_index.process_batch_updates(force=True)
        assert len(spatial_index._pending_position_updates) == 0

    def test_batch_update_stats(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=2
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
            width=100.0, height=100.0, enable_batch_updates=False
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
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=10
        )
        entity = Mock()
        entity.alive = True
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        assert len(spatial_index._pending_position_updates) == 1

    def test_update_entity_position_without_batch_updates(self):
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=False
        )
        entity = Mock()
        entity.alive = True
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        assert len(spatial_index._pending_position_updates) == 0

    def test_partial_batch_flushing_basic(self):
        """Test basic partial batch flushing functionality."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=10
        )
        entities = [Mock() for _ in range(5)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )

        # Process only 3 updates
        processed = spatial_index.flush_partial_updates(max_updates=3)
        assert processed == 3
        assert len(spatial_index._pending_position_updates) == 2

        # Process remaining updates
        processed = spatial_index.flush_partial_updates(max_updates=10)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 0

    def test_partial_batch_flushing_with_max_updates_larger_than_pending(self):
        """Test partial flushing when max_updates exceeds pending updates."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True
        )
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )

        # Try to process 10 updates when only 3 are pending
        processed = spatial_index.flush_partial_updates(max_updates=10)
        assert processed == 3
        assert len(spatial_index._pending_position_updates) == 0

    def test_partial_batch_flushing_zero_max_updates(self):
        """Test partial flushing with max_updates=0."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True
        )
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )

        processed = spatial_index.flush_partial_updates(max_updates=0)
        assert processed == 0
        assert len(spatial_index._pending_position_updates) == 3

    def test_partial_batch_flushing_disabled_batch_updates(self):
        """Test partial flushing when batch updates are disabled."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=False
        )
        entities = [Mock() for _ in range(3)]
        for i, entity in enumerate(entities):
            # Since batch updates are disabled, these should be processed immediately
            spatial_index.update_entity_position(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0)
            )

        processed = spatial_index.flush_partial_updates(max_updates=5)
        assert (
            processed == 0
        )  # No pending updates since they were processed immediately

    def test_partial_batch_flushing_no_pending_updates(self):
        """Test partial flushing when there are no pending updates."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True
        )

        processed = spatial_index.flush_partial_updates(max_updates=5)
        assert processed == 0
        assert len(spatial_index._pending_position_updates) == 0

    def test_process_batch_updates_return_value(self):
        """Test that process_batch_updates returns the correct number of processed updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10,  # Set high to avoid auto-flushing
        )
        entities = [Mock() for _ in range(5)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )

        # Test partial processing
        processed = spatial_index.process_batch_updates(force=True, max_updates=3)
        assert processed == 3
        assert len(spatial_index._pending_position_updates) == 2

        # Test processing remaining
        processed = spatial_index.process_batch_updates(force=True)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 0

    def test_partial_flushing_dirty_region_interaction(self):
        """Test that partial flushing correctly manages dirty regions."""
        spatial_index = SpatialIndex(
            width=200.0,
            height=200.0,
            enable_batch_updates=True,
            region_size=50.0,
            max_batch_size=10,
        )

        # Add updates that affect different regions
        entities = [Mock() for _ in range(4)]
        positions = [(25.0, 25.0), (75.0, 75.0), (125.0, 125.0), (175.0, 175.0)]
        for i, (entity, pos) in enumerate(zip(entities, positions)):
            spatial_index.add_position_update(
                entity, pos, (pos[0] + 10.0, pos[1] + 10.0), "agent"
            )

        # Check initial dirty regions
        initial_dirty = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        assert len(initial_dirty) == 4  # Each update affects a different region

        # Process only 2 updates
        processed = spatial_index.flush_partial_updates(max_updates=2)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 2

        # Check that only regions affected by the first 2 updates are cleared
        remaining_dirty = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        # Should have fewer dirty regions (some cleared, some remaining)
        assert len(remaining_dirty) < 4

        # Process remaining updates
        processed = spatial_index.flush_partial_updates(max_updates=10)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 0

        # All dirty regions should be cleared
        final_dirty = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        assert len(final_dirty) == 0

    def test_partial_flushing_mixed_entity_types(self):
        """Test partial flushing with mixed entity types."""
        spatial_index = SpatialIndex(
            width=200.0,
            height=200.0,
            enable_batch_updates=True,
            region_size=50.0,
            max_batch_size=10,
        )

        # Add agent and resource updates
        agents = [Mock() for _ in range(3)]
        resources = [Mock() for _ in range(2)]

        for i, agent in enumerate(agents):
            spatial_index.add_position_update(
                agent, (i * 50.0, i * 50.0), (i * 50.0 + 10.0, i * 50.0 + 10.0), "agent"
            )

        for i, resource in enumerate(resources):
            spatial_index.add_position_update(
                resource,
                (i * 75.0, i * 75.0),
                (i * 75.0 + 10.0, i * 75.0 + 10.0),
                "resource",
            )

        # Check initial state
        assert len(spatial_index._pending_position_updates) == 5
        agent_dirty_before = spatial_index._dirty_region_tracker.get_dirty_regions(
            "agent"
        )
        resource_dirty_before = spatial_index._dirty_region_tracker.get_dirty_regions(
            "resource"
        )
        assert len(agent_dirty_before) > 0
        assert len(resource_dirty_before) > 0

        # Process only 3 updates (should include both types)
        processed = spatial_index.flush_partial_updates(max_updates=3)
        assert processed == 3
        assert len(spatial_index._pending_position_updates) == 2

        # Process remaining updates
        processed = spatial_index.flush_partial_updates(max_updates=10)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 0

        # All dirty regions should be cleared
        agent_dirty_after = spatial_index._dirty_region_tracker.get_dirty_regions(
            "agent"
        )
        resource_dirty_after = spatial_index._dirty_region_tracker.get_dirty_regions(
            "resource"
        )
        assert len(agent_dirty_after) == 0
        assert len(resource_dirty_after) == 0

    def test_partial_flushing_statistics_accuracy(self):
        """Test that batch update statistics are accurate with partial flushing."""
        spatial_index = SpatialIndex(
            width=100.0, height=100.0, enable_batch_updates=True, max_batch_size=10
        )

        # Add 6 updates
        entities = [Mock() for _ in range(6)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 15.0, i * 15.0), "agent"
            )

        # Process 3 updates first
        processed1 = spatial_index.flush_partial_updates(max_updates=3)
        assert processed1 == 3

        stats1 = spatial_index.get_batch_update_stats()
        assert stats1["total_individual_updates"] == 3
        assert stats1["average_batch_size"] == 3.0
        assert stats1["total_batch_updates"] == 1

        # Process remaining 3 updates
        processed2 = spatial_index.flush_partial_updates(max_updates=3)
        assert processed2 == 3

        stats2 = spatial_index.get_batch_update_stats()
        assert stats2["total_individual_updates"] == 6
        assert stats2["average_batch_size"] == 3.0  # Average of 3 and 3
        assert stats2["total_batch_updates"] == 2

    def test_partial_flushing_with_overlapping_regions(self):
        """Test partial flushing when updates affect overlapping regions."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            region_size=30.0,  # Smaller regions for more overlap potential
            max_batch_size=10,
        )

        # Add updates where some positions are in the same region
        entities = [Mock() for _ in range(4)]
        # These positions are close enough to potentially overlap regions
        positions = [(10.0, 10.0), (15.0, 15.0), (60.0, 60.0), (65.0, 65.0)]
        for entity, pos in zip(entities, positions):
            spatial_index.add_position_update(
                entity, pos, (pos[0] + 5.0, pos[1] + 5.0), "agent"
            )

        # Process partial updates
        processed = spatial_index.flush_partial_updates(max_updates=2)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 2

        # Process remaining
        processed = spatial_index.flush_partial_updates(max_updates=10)
        assert processed == 2
        assert len(spatial_index._pending_position_updates) == 0

        # All regions should be cleared despite overlaps
        final_dirty = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        assert len(final_dirty) == 0


class TestEnvironmentBatchUpdates:
    @patch("farm.database.utilities.setup_db")
    def test_environment_initialization_with_spatial_config(self, mock_setup_db):
        mock_setup_db.return_value = Mock()
        spatial_config = SpatialIndexConfig(
            enable_batch_updates=True,
            region_size=30.0,
            max_batch_size=25,
            enable_quadtree_indices=True,
        )
        env_config = EnvironmentConfig(
            width=200, height=200, spatial_index=spatial_config
        )
        config = SimulationConfig()
        config.environment = env_config
        env = Environment(
            width=200, height=200, resource_distribution="uniform", config=config
        )
        assert env.spatial_index._batch_update_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 30.0
        assert env.spatial_index.max_batch_size == 25

    @patch("farm.database.utilities.setup_db")
    def test_environment_initialization_without_spatial_config(self, mock_setup_db):
        mock_setup_db.return_value = Mock()
        config = SimulationConfig()
        env = Environment(
            width=100, height=100, resource_distribution="uniform", config=config
        )
        assert env.spatial_index._initial_batch_updates_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 50.0
        assert env.spatial_index.max_batch_size == 100

    @patch("farm.database.utilities.setup_db")
    def test_process_batch_spatial_updates(self, mock_setup_db):
        mock_setup_db.return_value = Mock()
        env = Environment(width=100, height=100, resource_distribution="uniform")
        entity = Mock()
        env.spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent"
        )
        env.process_batch_spatial_updates(force=True)
        assert len(env.spatial_index._pending_position_updates) == 0

    @patch("farm.database.utilities.setup_db")
    def test_get_spatial_performance_stats(self, mock_setup_db):
        mock_setup_db.return_value = Mock()
        env = Environment(width=100, height=100, resource_distribution="uniform")
        stats = env.get_spatial_performance_stats()
        assert "agent_count" in stats
        assert "resource_count" in stats
        assert "batch_updates" in stats
        assert "perception" in stats

    @patch("farm.database.utilities.setup_db")
    def test_enable_disable_batch_spatial_updates(self, mock_setup_db):
        mock_setup_db.return_value = Mock()
        env = Environment(width=100, height=100, resource_distribution="uniform")
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
            max_batch_size=50,
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
            max_batch_size=20,
        )
        entities = [Mock() for _ in range(100)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 5.0, i * 5.0), (i * 5.0 + 10.0, i * 5.0 + 10.0), "agent"
            )
        spatial_index.process_batch_updates(force=True)
        stats = spatial_index.get_batch_update_stats()
        assert stats["total_batch_updates"] > 0
        assert stats["total_individual_updates"] == 100
        assert stats["average_batch_size"] > 0
        assert stats["total_regions_processed"] > 0
