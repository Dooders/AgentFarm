"""
Tests for batch spatial updates with dirty region tracking.

This module tests the new batch spatial updates feature that improves performance
by only updating regions that have actually changed, rather than rebuilding
entire spatial indices on every position change.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Tuple

from farm.core.spatial_index import SpatialIndex, DirtyRegionTracker, DirtyRegion
from farm.core.environment import Environment
from farm.config.config import SpatialIndexConfig, EnvironmentConfig, SimulationConfig


class TestDirtyRegionTracker:
    """Test the DirtyRegionTracker class."""

    def test_initialization(self):
        """Test DirtyRegionTracker initialization."""
        tracker = DirtyRegionTracker(region_size=25.0, max_regions=500)
        
        assert tracker.region_size == 25.0
        assert tracker.max_regions == 500
        assert tracker._total_regions_marked == 0
        assert tracker._total_regions_updated == 0

    def test_mark_region_dirty(self):
        """Test marking regions as dirty."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark a region as dirty
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        
        # Check that the region was marked
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert dirty_regions[0].entity_type == "agent"
        assert dirty_regions[0].priority == 1

    def test_mark_region_dirty_batch(self):
        """Test marking multiple regions as dirty in batch."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        positions = [(10.0, 10.0), (60.0, 60.0), (110.0, 110.0)]
        tracker.mark_region_dirty_batch(positions, "resource", priority=2)
        
        # Check that all regions were marked
        dirty_regions = tracker.get_dirty_regions("resource")
        assert len(dirty_regions) == 3
        
        # All should have the same priority
        for region in dirty_regions:
            assert region.entity_type == "resource"
            assert region.priority == 2

    def test_region_coordinate_conversion(self):
        """Test world to region coordinate conversion."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Test coordinate conversion
        region_coords = tracker._world_to_region_coords((75.0, 125.0))
        assert region_coords == (1, 2)
        
        # Test bounds conversion
        bounds = tracker._region_to_world_bounds((1, 2))
        assert bounds == (50.0, 100.0, 50.0, 50.0)

    def test_clear_region(self):
        """Test clearing dirty regions."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark regions as dirty
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        # Clear one region
        region_coords = tracker._world_to_region_coords((25.0, 25.0))
        tracker.clear_region(region_coords)
        
        # Check that only one region remains
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1

    def test_clear_all_regions(self):
        """Test clearing all dirty regions."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark multiple regions as dirty
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "resource")
        
        # Clear all regions
        tracker.clear_all_regions()
        
        # Check that no regions remain
        assert len(tracker.get_dirty_regions()) == 0

    def test_priority_sorting(self):
        """Test that dirty regions are sorted by priority."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark regions with different priorities
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        tracker.mark_region_dirty((75.0, 75.0), "agent", priority=3)
        tracker.mark_region_dirty((125.0, 125.0), "agent", priority=2)
        
        # Get dirty regions
        dirty_regions = tracker.get_dirty_regions("agent")
        
        # Check that they are sorted by priority (highest first)
        priorities = [region.priority for region in dirty_regions]
        assert priorities == [3, 2, 1]

    def test_max_regions_cleanup(self):
        """Test cleanup when max regions limit is exceeded."""
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=3)
        
        # Mark more regions than the limit
        for i in range(5):
            tracker.mark_region_dirty((i * 100.0, i * 100.0), "agent")
        
        # Check that cleanup occurred
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) <= 3

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark some regions
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "resource")
        
        # Get stats
        stats = tracker.get_stats()
        
        assert stats["total_dirty_regions"] == 2
        assert stats["regions_by_type"]["agent"] == 1
        assert stats["regions_by_type"]["resource"] == 1
        assert stats["total_regions_marked"] == 2


class TestSpatialIndexBatchUpdates:
    """Test the SpatialIndex batch update functionality."""

    def test_initialization_with_batch_updates(self):
        """Test SpatialIndex initialization with batch updates enabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            region_size=25.0,
            max_batch_size=50
        )
        
        assert spatial_index.enable_batch_updates is True
        assert spatial_index.max_batch_size == 50
        assert spatial_index._batch_update_enabled is True
        assert spatial_index._dirty_region_tracker is not None
        assert spatial_index._dirty_region_tracker.region_size == 25.0

    def test_initialization_without_batch_updates(self):
        """Test SpatialIndex initialization with batch updates disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        
        assert spatial_index.enable_batch_updates is False
        assert spatial_index._batch_update_enabled is False
        assert spatial_index._dirty_region_tracker is None

    def test_add_position_update(self):
        """Test adding position updates to batch queue."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        
        # Create mock entity
        entity = Mock()
        entity.position = (50.0, 50.0)
        
        # Add position update
        spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent", priority=1
        )
        
        # Check that update was added to queue
        assert len(spatial_index._pending_position_updates) == 1
        assert spatial_index._pending_position_updates[0][0] is entity
        assert spatial_index._pending_position_updates[0][1] == (25.0, 25.0)
        assert spatial_index._pending_position_updates[0][2] == (75.0, 75.0)
        assert spatial_index._pending_position_updates[0][3] == "agent"
        assert spatial_index._pending_position_updates[0][4] == 1

    def test_batch_processing_trigger(self):
        """Test that batch processing is triggered when batch is full."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=3
        )
        
        # Create mock entities
        entities = [Mock() for _ in range(3)]
        
        # Add updates to fill the batch
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        
        # Check that batch was processed (queue should be empty)
        assert len(spatial_index._pending_position_updates) == 0

    def test_process_batch_updates_force(self):
        """Test forcing batch update processing."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        
        # Add some updates
        entity = Mock()
        spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent"
        )
        
        # Force processing
        spatial_index.process_batch_updates(force=True)
        
        # Check that queue is empty
        assert len(spatial_index._pending_position_updates) == 0

    def test_batch_update_stats(self):
        """Test batch update statistics tracking."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=2
        )
        
        # Add updates to trigger batch processing
        entities = [Mock() for _ in range(2)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        
        # Get stats
        stats = spatial_index.get_batch_update_stats()
        
        assert stats["total_batch_updates"] == 1
        assert stats["total_individual_updates"] == 2
        assert stats["average_batch_size"] == 2.0

    def test_enable_disable_batch_updates(self):
        """Test enabling and disabling batch updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        
        # Enable batch updates
        spatial_index.enable_batch_updates(region_size=30.0, max_batch_size=20)
        
        assert spatial_index._batch_update_enabled is True
        assert spatial_index._dirty_region_tracker is not None
        assert spatial_index._dirty_region_tracker.region_size == 30.0
        assert spatial_index.max_batch_size == 20
        
        # Disable batch updates
        spatial_index.disable_batch_updates()
        
        assert spatial_index._batch_update_enabled is False
        assert spatial_index._dirty_region_tracker is None

    def test_update_entity_position_with_batch_updates(self):
        """Test update_entity_position with batch updates enabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10
        )
        
        # Create mock entity
        entity = Mock()
        entity.alive = True
        
        # Update position
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        
        # Check that update was added to batch queue
        assert len(spatial_index._pending_position_updates) == 1

    def test_update_entity_position_without_batch_updates(self):
        """Test update_entity_position with batch updates disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False
        )
        
        # Create mock entity
        entity = Mock()
        entity.alive = True
        
        # Update position
        spatial_index.update_entity_position(entity, (25.0, 25.0), (75.0, 75.0))
        
        # Check that update was processed immediately (no batch queue)
        assert len(spatial_index._pending_position_updates) == 0


class TestEnvironmentBatchUpdates:
    """Test Environment integration with batch spatial updates."""

    def test_environment_initialization_with_spatial_config(self):
        """Test Environment initialization with spatial index configuration."""
        # Create configuration
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
        
        # Create environment
        env = Environment(
            width=200,
            height=200,
            resource_distribution="uniform",
            config=config
        )
        
        # Check that spatial index was configured correctly
        assert env.spatial_index.enable_batch_updates is True
        assert env.spatial_index._dirty_region_tracker.region_size == 30.0
        assert env.spatial_index.max_batch_size == 25

    def test_environment_initialization_without_spatial_config(self):
        """Test Environment initialization with default spatial configuration."""
        config = SimulationConfig()
        
        # Create environment
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform",
            config=config
        )
        
        # Check that default batch updates are enabled
        assert env.spatial_index.enable_batch_updates is True
        assert env.spatial_index._dirty_region_tracker.region_size == 50.0
        assert env.spatial_index.max_batch_size == 100

    def test_process_batch_spatial_updates(self):
        """Test processing batch spatial updates from environment."""
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        
        # Add some position updates
        entity = Mock()
        env.spatial_index.add_position_update(
            entity, (25.0, 25.0), (75.0, 75.0), "agent"
        )
        
        # Process batch updates
        env.process_batch_spatial_updates(force=True)
        
        # Check that updates were processed
        assert len(env.spatial_index._pending_position_updates) == 0

    def test_get_spatial_performance_stats(self):
        """Test getting spatial performance statistics from environment."""
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        
        # Get stats
        stats = env.get_spatial_performance_stats()
        
        # Check that stats contain expected keys
        assert "agent_count" in stats
        assert "resource_count" in stats
        assert "batch_updates" in stats
        assert "perception" in stats

    def test_enable_disable_batch_spatial_updates(self):
        """Test enabling and disabling batch spatial updates from environment."""
        env = Environment(
            width=100,
            height=100,
            resource_distribution="uniform"
        )
        
        # Disable batch updates
        env.disable_batch_spatial_updates()
        assert env.spatial_index._batch_update_enabled is False
        
        # Enable batch updates
        env.enable_batch_spatial_updates(region_size=40.0, max_batch_size=30)
        assert env.spatial_index._batch_update_enabled is True
        assert env.spatial_index._dirty_region_tracker.region_size == 40.0
        assert env.spatial_index.max_batch_size == 30


class TestPerformanceImprovements:
    """Test performance improvements from batch spatial updates."""

    def test_batch_vs_individual_updates_performance(self):
        """Test that batch updates are more efficient than individual updates."""
        # Test with batch updates
        spatial_index_batch = SpatialIndex(
            width=1000.0,
            height=1000.0,
            enable_batch_updates=True,
            max_batch_size=100
        )
        
        # Test without batch updates
        spatial_index_individual = SpatialIndex(
            width=1000.0,
            height=1000.0,
            enable_batch_updates=False
        )
        
        # Create mock entities
        entities = [Mock() for _ in range(50)]
        
        # Time batch updates
        start_time = time.time()
        for i, entity in enumerate(entities):
            spatial_index_batch.add_position_update(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0), "agent"
            )
        spatial_index_batch.process_batch_updates(force=True)
        batch_time = time.time() - start_time
        
        # Time individual updates
        start_time = time.time()
        for i, entity in enumerate(entities):
            spatial_index_individual.update_entity_position(
                entity, (i * 10.0, i * 10.0), (i * 20.0, i * 20.0)
            )
        individual_time = time.time() - start_time
        
        # Batch updates should be faster (or at least not significantly slower)
        # Note: This test might be flaky due to timing variations, but it's useful
        # for detecting major performance regressions
        assert batch_time <= individual_time * 2  # Allow some tolerance

    def test_region_based_efficiency(self):
        """Test that region-based updates are more efficient than global updates."""
        spatial_index = SpatialIndex(
            width=1000.0,
            height=1000.0,
            enable_batch_updates=True,
            region_size=100.0,
            max_batch_size=50
        )
        
        # Add updates in different regions
        entities = [Mock() for _ in range(20)]
        
        # Updates in region 1
        for i in range(10):
            spatial_index.add_position_update(
                entities[i], (50.0, 50.0), (60.0, 60.0), "agent"
            )
        
        # Updates in region 2 (far away)
        for i in range(10, 20):
            spatial_index.add_position_update(
                entities[i], (500.0, 500.0), (510.0, 510.0), "agent"
            )
        
        # Process batch updates
        start_time = time.time()
        spatial_index.process_batch_updates(force=True)
        processing_time = time.time() - start_time
        
        # Check that processing was efficient
        assert processing_time < 0.1  # Should be very fast
        
        # Check that regions were tracked correctly
        stats = spatial_index.get_batch_update_stats()
        assert stats["total_regions_processed"] > 0


if __name__ == "__main__":
    pytest.main([__file__])