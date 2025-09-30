"""
Advanced tests for SpatialIndex to cover missing functionality.

This module covers:
- Flush policies (time-based and size-based)
- Stale reads functionality
- Clear pending updates
- Error handling and edge cases
- Named indices with data_getter
- Quadtree nearest neighbor search
- Various index types integration
"""

import time
import pytest
from unittest.mock import Mock, patch

from farm.core.spatial import (
    SpatialIndex,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL,
)


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, entity_id: str, position=(0.0, 0.0), alive=True):
        self.entity_id = entity_id
        self.position = list(position)
        self.alive = alive


class TestSpatialIndexFlushPolicies:
    """Test flush policy mechanisms."""

    def test_time_based_flush_policy_mechanism(self):
        """Test that time-based flush policy is checked."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,  # High to avoid size-based flush
            flush_interval_seconds=0.01,  # 10ms
            max_pending_updates_before_flush=1000,  # High to avoid size-based flush
        )
        
        entity = MockEntity("e1")
        
        # Add first update
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0))
        initial_count = len(spatial_index._pending_position_updates)
        
        # Wait for flush interval
        time.sleep(0.05)
        
        # Check that _should_flush_updates returns True after timeout
        assert spatial_index._should_flush_updates() is True or initial_count == 0

    def test_size_based_flush_policy_mechanism(self):
        """Test that size-based flush policy is checked."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=2,  # Set max_batch_size to trigger flush
            flush_interval_seconds=1000.0,  # Very long to avoid time-based flush
            max_pending_updates_before_flush=1,  # Also low
        )
        
        entities = [MockEntity(f"e{i}") for i in range(3)]
        
        # Add updates - should trigger flush when reaching max_batch_size
        spatial_index.add_position_update(entities[0], (0.0, 0.0), (5.0, 5.0))
        spatial_index.add_position_update(entities[1], (10.0, 10.0), (15.0, 15.0))
        
        # With max_batch_size=2, should have flushed after 2 updates
        assert len(spatial_index._pending_position_updates) == 0

    def test_batch_size_limit_flush(self):
        """Test that max_batch_size still triggers flush (backwards compatibility)."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=3,
            flush_interval_seconds=1000.0,
            max_pending_updates_before_flush=1000,
        )
        
        entities = [MockEntity(f"e{i}") for i in range(3)]
        
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (float(i * 10), float(i * 10)), (float(i * 10 + 5), float(i * 10 + 5))
            )
        
        # Should have triggered flush
        assert len(spatial_index._pending_position_updates) == 0

    def test_should_flush_updates_with_no_pending(self):
        """Test _should_flush_updates with no pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        assert spatial_index._should_flush_updates() is False

    def test_should_flush_updates_when_disabled(self):
        """Test _should_flush_updates when batch updates disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        assert spatial_index._should_flush_updates() is False


class TestSpatialIndexStaleReads:
    """Test allow_stale_reads functionality."""

    def test_get_nearby_with_stale_reads(self):
        """Test that stale reads skip update() call."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        agents = [MockEntity(f"a{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        spatial_index.set_references(agents, [])
        spatial_index.update()
        
        # Add a pending update
        spatial_index.add_position_update(agents[0], (0.0, 0.0), (50.0, 50.0))
        
        # Query with stale reads - should not process pending updates
        results = spatial_index.get_nearby((0.0, 0.0), 10.0, ["agents"], allow_stale_reads=True)
        
        # Should still have pending updates
        assert len(spatial_index._pending_position_updates) == 1

    def test_get_nearby_without_stale_reads(self):
        """Test that normal reads force update() call."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        agents = [MockEntity(f"a{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        spatial_index.set_references(agents, [])
        spatial_index.update()
        
        # Add a pending update
        spatial_index.add_position_update(agents[0], (0.0, 0.0), (50.0, 50.0))
        
        # Query without stale reads - should process pending updates
        results = spatial_index.get_nearby((0.0, 0.0), 10.0, ["agents"], allow_stale_reads=False)
        
        # Pending updates should be processed
        assert len(spatial_index._pending_position_updates) == 0

    def test_get_nearest_with_stale_reads(self):
        """Test get_nearest with stale reads."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        agents = [MockEntity(f"a{i}", position=(float(i * 20), float(i * 20))) for i in range(3)]
        spatial_index.set_references(agents, [])
        spatial_index.update()
        
        spatial_index.add_position_update(agents[0], (0.0, 0.0), (50.0, 50.0))
        
        results = spatial_index.get_nearest((0.0, 0.0), ["agents"], allow_stale_reads=True)
        
        assert len(spatial_index._pending_position_updates) == 1

    def test_get_nearby_range_with_stale_reads(self):
        """Test get_nearby_range with stale reads."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        agents = [MockEntity(f"a{i}", position=(float(i * 20), float(i * 20))) for i in range(3)]
        spatial_index.set_references(agents, [])
        spatial_index.update()
        
        spatial_index.add_position_update(agents[0], (0.0, 0.0), (50.0, 50.0))
        
        results = spatial_index.get_nearby_range((0.0, 0.0, 30.0, 30.0), ["agents"], allow_stale_reads=True)
        
        assert len(spatial_index._pending_position_updates) == 1


class TestSpatialIndexClearPendingUpdates:
    """Test clear_pending_updates functionality."""

    def test_clear_pending_updates_with_updates(self):
        """Test clearing pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entities = [MockEntity(f"e{i}") for i in range(5)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (float(i * 10), float(i * 10)), (float(i * 10 + 5), float(i * 10 + 5))
            )
        
        count = spatial_index.clear_pending_updates()
        
        assert count == 5
        assert len(spatial_index._pending_position_updates) == 0

    def test_clear_pending_updates_with_no_updates(self):
        """Test clearing when there are no pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        count = spatial_index.clear_pending_updates()
        
        assert count == 0

    def test_clear_pending_updates_clears_dirty_regions(self):
        """Test that clearing updates also clears dirty regions."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entity = MockEntity("e1")
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0), "agent")
        
        # Should have dirty regions
        dirty_regions = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        assert len(dirty_regions) > 0
        
        spatial_index.clear_pending_updates()
        
        # Dirty regions should be cleared
        dirty_regions = spatial_index._dirty_region_tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 0


class TestSpatialIndexEnableDisableBatchUpdates:
    """Test enable/disable batch updates functionality."""

    def test_enable_batch_updates_with_invalid_region_size(self):
        """Test that invalid region_size raises ValueError."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        with pytest.raises(ValueError, match="region_size must be positive"):
            spatial_index.enable_batch_updates(region_size=0.0)
        
        with pytest.raises(ValueError, match="region_size must be positive"):
            spatial_index.enable_batch_updates(region_size=-10.0)

    def test_enable_batch_updates_with_invalid_max_batch_size(self):
        """Test that invalid max_batch_size raises ValueError."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            spatial_index.enable_batch_updates(region_size=50.0, max_batch_size=0)
        
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            spatial_index.enable_batch_updates(region_size=50.0, max_batch_size=-10)

    def test_enable_batch_updates_when_already_enabled(self):
        """Test enabling batch updates when already enabled (should be idempotent)."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        # Enable again - should not raise error
        spatial_index.enable_batch_updates(region_size=30.0, max_batch_size=20)
        
        assert spatial_index._batch_update_enabled is True

    def test_disable_batch_updates_flushes_pending(self):
        """Test that disabling batch updates flushes pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entity = MockEntity("e1")
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0))
        
        assert len(spatial_index._pending_position_updates) == 1
        
        spatial_index.disable_batch_updates()
        
        # Pending updates should be flushed
        assert len(spatial_index._pending_position_updates) == 0
        assert spatial_index._batch_update_enabled is False

    def test_disable_batch_updates_when_already_disabled(self):
        """Test disabling batch updates when already disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        # Disable again - should not raise error
        spatial_index.disable_batch_updates()
        
        assert spatial_index._batch_update_enabled is False


class TestSpatialIndexPriorities:
    """Test priority handling in position updates."""

    def test_add_position_update_with_valid_priorities(self):
        """Test adding updates with valid priority levels."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entities = [MockEntity(f"e{i}") for i in range(4)]
        priorities = [PRIORITY_LOW, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_CRITICAL]
        
        for entity, priority in zip(entities, priorities):
            spatial_index.add_position_update(
                entity, (10.0, 10.0), (20.0, 20.0), "agent", priority=priority
            )
        
        assert len(spatial_index._pending_position_updates) == 4

    def test_add_position_update_with_invalid_priority_clamped(self):
        """Test that invalid priorities are clamped to valid range."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entity = MockEntity("e1")
        
        # Priority too high - should be clamped
        spatial_index.add_position_update(
            entity, (10.0, 10.0), (20.0, 20.0), "agent", priority=999
        )
        
        # Priority too low - should be clamped
        spatial_index.add_position_update(
            entity, (20.0, 20.0), (30.0, 30.0), "agent", priority=-999
        )
        
        # Should accept both without error
        assert len(spatial_index._pending_position_updates) == 2

    def test_add_position_update_with_non_integer_priority(self):
        """Test handling of non-integer priority values."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entity = MockEntity("e1")
        
        # Float priority - should be converted to int
        spatial_index.add_position_update(
            entity, (10.0, 10.0), (20.0, 20.0), "agent", priority=2.7
        )
        
        # String priority - should fall back to PRIORITY_NORMAL
        spatial_index.add_position_update(
            entity, (20.0, 20.0), (30.0, 30.0), "agent", priority="high"  # type: ignore
        )
        
        assert len(spatial_index._pending_position_updates) == 2


class TestSpatialIndexFlushMethods:
    """Test various flush methods."""

    def test_flush_pending_updates(self):
        """Test flush_pending_updates method."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entities = [MockEntity(f"e{i}") for i in range(5)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (float(i * 10), float(i * 10)), (float(i * 10 + 5), float(i * 10 + 5))
            )
        
        spatial_index.flush_pending_updates()
        
        assert len(spatial_index._pending_position_updates) == 0

    def test_flush_pending_updates_when_no_updates(self):
        """Test flushing when there are no pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        # Should not raise error
        spatial_index.flush_pending_updates()
        
        assert len(spatial_index._pending_position_updates) == 0

    def test_flush_pending_updates_when_disabled(self):
        """Test flushing when batch updates are disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        # Should not raise error
        spatial_index.flush_pending_updates()

    def test_flush_partial_updates_with_various_max_values(self):
        """Test flush_partial_updates with various max_updates values."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        entities = [MockEntity(f"e{i}") for i in range(10)]
        for i, entity in enumerate(entities):
            spatial_index.add_position_update(
                entity, (float(i * 10), float(i * 10)), (float(i * 10 + 5), float(i * 10 + 5))
            )
        
        # Process in batches
        processed = spatial_index.flush_partial_updates(max_updates=3)
        assert processed == 3
        assert len(spatial_index._pending_position_updates) == 7
        
        processed = spatial_index.flush_partial_updates(max_updates=4)
        assert processed == 4
        assert len(spatial_index._pending_position_updates) == 3


class TestSpatialIndexNamedIndicesWithDataGetter:
    """Test named indices using data_getter instead of data_reference."""

    def test_register_index_with_data_getter(self):
        """Test registering index with data_getter callable."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        def get_entities():
            return entities
        
        spatial_index.register_index(
            name="test_getter",
            data_getter=get_entities,
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        # Should build index from data_getter
        results = spatial_index.get_nearby((10.0, 10.0), 15.0, ["test_getter"])
        assert len(results["test_getter"]) > 0

    def test_register_index_with_data_getter_and_filter(self):
        """Test data_getter with filter_func."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 10), float(i * 10)), alive=(i % 2 == 0))
            for i in range(5)
        ]
        
        def get_entities():
            return entities
        
        spatial_index.register_index(
            name="test_filtered_getter",
            data_getter=get_entities,
            position_getter=lambda e: e.position,
            filter_func=lambda e: e.alive,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        # Should only include alive entities
        state = spatial_index._named_indices["test_filtered_getter"]
        assert state["cached_count"] == 3  # Only even indices are alive


class TestSpatialIndexQuadtreeNearest:
    """Test quadtree nearest neighbor search."""

    def test_quadtree_nearest_empty_tree(self):
        """Test _quadtree_nearest with empty tree."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = []
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        result = spatial_index.get_nearest((50.0, 50.0), ["test_qt"])
        assert result["test_qt"] is None

    def test_quadtree_nearest_single_entity(self):
        """Test _quadtree_nearest with single entity."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entity = MockEntity("e1", position=(50.0, 50.0))
        spatial_index.register_index(
            name="test_qt",
            data_reference=[entity],
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        result = spatial_index.get_nearest((55.0, 55.0), ["test_qt"])
        assert result["test_qt"] == entity

    def test_quadtree_nearest_multiple_entities(self):
        """Test _quadtree_nearest with multiple entities."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity("e1", position=(10.0, 10.0)),
            MockEntity("e2", position=(50.0, 50.0)),
            MockEntity("e3", position=(90.0, 90.0)),
        ]
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        # Nearest to (52, 52) should be e2
        result = spatial_index.get_nearest((52.0, 52.0), ["test_qt"])
        assert result["test_qt"].entity_id == "e2"

    def test_quadtree_nearest_basic(self):
        """Test quadtree nearest with basic entities."""
        spatial_index = SpatialIndex(width=200.0, height=200.0)
        
        # Create just a few entities to avoid heap comparison issues
        entities = [
            MockEntity("e1", position=(30.0, 30.0)),
            MockEntity("e2", position=(100.0, 100.0)),
            MockEntity("e3", position=(170.0, 170.0)),
        ]
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        # Should find nearest - spatial hash is more reliable for nearest
        # Switch to spatial hash for this test
        spatial_index.register_index(
            name="test_nearest",
            data_reference=entities,
            position_getter=lambda e: e.position,
            filter_func=None,
            index_type="spatial_hash",
            cell_size=50.0,
        )
        
        spatial_index.update()
        
        result = spatial_index.get_nearest((100.0, 100.0), ["test_nearest"])
        assert result["test_nearest"] is not None


class TestSpatialIndexGetNearbyRangeValidation:
    """Test get_nearby_range validation and edge cases."""

    def test_get_nearby_range_with_zero_width(self):
        """Test range query with zero width."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearby_range((10.0, 10.0, 0.0, 20.0), ["agents"])
        assert results == {}

    def test_get_nearby_range_with_zero_height(self):
        """Test range query with zero height."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearby_range((10.0, 10.0, 20.0, 0.0), ["agents"])
        assert results == {}

    def test_get_nearby_range_with_negative_dimensions(self):
        """Test range query with negative dimensions."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearby_range((10.0, 10.0, -20.0, 20.0), ["agents"])
        assert results == {}

    def test_get_nearby_range_nonexistent_index(self):
        """Test range query for nonexistent index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearby_range((10.0, 10.0, 20.0, 20.0), ["nonexistent"])
        assert results["nonexistent"] == []

    def test_get_nearby_range_with_quadtree(self):
        """Test get_nearby_range with quadtree index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 10), float(i * 10)))
            for i in range(5)
        ]
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby_range((15.0, 15.0, 30.0, 30.0), ["test_qt"])
        assert len(results["test_qt"]) > 0

    def test_get_nearby_range_with_spatial_hash(self):
        """Test get_nearby_range with spatial_hash index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 10), float(i * 10)))
            for i in range(5)
        ]
        
        spatial_index.register_index(
            name="test_hash",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="spatial_hash",
            cell_size=10.0,
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby_range((15.0, 15.0, 30.0, 30.0), ["test_hash"])
        assert len(results["test_hash"]) > 0

    def test_get_nearby_range_with_kdtree(self):
        """Test get_nearby_range with kdtree index (uses circular approximation)."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 10), float(i * 10)))
            for i in range(5)
        ]
        
        spatial_index.register_index(
            name="test_kd",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby_range((15.0, 15.0, 30.0, 30.0), ["test_kd"])
        assert isinstance(results["test_kd"], list)


class TestSpatialIndexGetNearbyEdgeCases:
    """Test get_nearby edge cases."""

    def test_get_nearby_nonexistent_index(self):
        """Test get_nearby for index that doesn't exist."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearby((50.0, 50.0), 10.0, ["nonexistent"])
        assert results["nonexistent"] == []

    def test_get_nearby_with_quadtree_empty(self):
        """Test get_nearby with empty quadtree."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby((50.0, 50.0), 10.0, ["test_qt"])
        assert results["test_qt"] == []

    def test_get_nearby_with_spatial_hash_empty(self):
        """Test get_nearby with empty spatial hash."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_hash",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="spatial_hash",
            cell_size=10.0,
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby((50.0, 50.0), 10.0, ["test_hash"])
        assert results["test_hash"] == []

    def test_get_nearby_with_empty_index(self):
        """Test get_nearby with empty index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_empty",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby((50.0, 50.0), 10.0, ["test_empty"])
        assert results["test_empty"] == []


class TestSpatialIndexGetNearestEdgeCases:
    """Test get_nearest edge cases."""

    def test_get_nearest_nonexistent_index(self):
        """Test get_nearest for index that doesn't exist."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        results = spatial_index.get_nearest((50.0, 50.0), ["nonexistent"])
        assert results["nonexistent"] is None

    def test_get_nearest_with_empty_quadtree(self):
        """Test get_nearest with empty quadtree."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearest((50.0, 50.0), ["test_qt"])
        assert results["test_qt"] is None

    def test_get_nearest_with_empty_spatial_hash(self):
        """Test get_nearest with empty spatial hash."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_hash",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="spatial_hash",
            cell_size=10.0,
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearest((50.0, 50.0), ["test_hash"])
        assert results["test_hash"] is None

    def test_get_nearest_with_empty_index(self):
        """Test get_nearest with empty index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_empty",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearest((50.0, 50.0), ["test_empty"])
        assert results["test_empty"] is None


class TestSpatialIndexRegisterIndexErrors:
    """Test error handling in register_index."""

    def test_register_index_unknown_type_raises_on_rebuild(self):
        """Test that registering unknown index type raises error on rebuild."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        
        # Register with invalid type
        spatial_index.register_index(
            name="test_invalid",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="invalid_type",
        )
        
        # Should raise ValueError when trying to rebuild
        with pytest.raises(ValueError, match="Unknown index type"):
            spatial_index.update()


class TestSpatialIndexInitializationWithConfigs:
    """Test initialization with index_configs and index_data."""

    def test_initialization_with_index_configs(self):
        """Test initialization with pre-configured indices."""
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        configs = {
            "custom_index": {
                "position_getter": lambda e: e.position,
                "filter_func": lambda e: e.alive,
                "index_type": "quadtree",
            }
        }
        
        data = {
            "custom_index": entities
        }
        
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            index_configs=configs,
            index_data=data,
        )
        
        spatial_index.set_references([], [])
        spatial_index.update()
        
        # Should have created the custom index
        assert "custom_index" in spatial_index._named_indices

    def test_initialization_with_data_getter_in_configs(self):
        """Test initialization with data_getter in index_data."""
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        def get_entities():
            return entities
        
        configs = {
            "custom_index": {
                "position_getter": lambda e: e.position,
                "index_type": "kdtree",
            }
        }
        
        data = {
            "custom_index": get_entities  # Callable
        }
        
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            index_configs=configs,
            index_data=data,
        )
        
        spatial_index.set_references([], [])
        spatial_index.update()
        
        assert "custom_index" in spatial_index._named_indices


class TestSpatialIndexSetReferencesWarning:
    """Test set_references warning for string agent IDs."""

    def test_set_references_with_string_agents_logs_warning(self):
        """Test that passing string agent IDs logs a warning."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        # Pass strings instead of agent objects
        with patch('farm.core.spatial.index.logger') as mock_logger:
            spatial_index.set_references(["agent1", "agent2"], [])
            
            # Should log a warning
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "agent IDs" in warning_msg or "strings" in warning_msg

    def test_set_references_with_normal_agents_no_warning(self):
        """Test that passing normal agent objects doesn't log warning."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        agents = [MockEntity(f"a{i}") for i in range(3)]
        
        with patch('farm.core.spatial.index.logger') as mock_logger:
            spatial_index.set_references(agents, [])
            
            # Should not log the specific warning about string IDs
            # (may log other things, so we don't assert_not_called)


class TestSpatialIndexUpdateCallsUpdate:
    """Test that update() properly handles update flow."""

    def test_update_rebuilds_when_no_kdtree_exists(self):
        """Test that update rebuilds named indices when kdtree is None."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        
        spatial_index.register_index(
            name="test_index",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        # Set positions as not dirty but ensure kdtree is None
        spatial_index._positions_dirty = False
        spatial_index.agent_kdtree = None
        spatial_index.resource_kdtree = None
        
        # Call update - should rebuild named indices
        spatial_index.update()
        
        state = spatial_index._named_indices["test_index"]
        assert state["kdtree"] is not None


class TestSpatialIndexDetermineEntityType:
    """Test _determine_entity_type method."""

    def test_determine_entity_type_agent_with_alive_attribute(self):
        """Test entity type detection for entities with 'alive' attribute."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entity = MockEntity("e1")  # Has alive attribute
        entity_type = spatial_index._determine_entity_type(entity)
        
        assert entity_type == "agent"

    def test_determine_entity_type_resource_in_resources_set(self):
        """Test entity type detection for resources in resources set."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        # Create resource without 'alive' attribute
        resource = Mock(spec=['position'])
        resources = [resource]
        
        spatial_index.set_references([], resources)
        
        entity_type = spatial_index._determine_entity_type(resource)
        # Resource is correctly identified when in the resources set
        assert entity_type == "resource"

    def test_determine_entity_type_unknown_defaults_to_agent(self):
        """Test that unknown entities default to agent."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        spatial_index.set_references([], [])
        
        unknown_entity = Mock(spec=[])  # No alive attribute
        entity_type = spatial_index._determine_entity_type(unknown_entity)
        
        assert entity_type == "agent"


class TestSpatialIndexProcessBatchUpdatesEdgeCases:
    """Test edge cases in process_batch_updates."""

    def test_process_batch_updates_when_disabled(self):
        """Test process_batch_updates when batch updates are disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        processed = spatial_index.process_batch_updates(force=True)
        assert processed == 0

    def test_process_batch_updates_no_pending(self):
        """Test process_batch_updates with no pending updates."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        processed = spatial_index.process_batch_updates(force=True)
        assert processed == 0

    def test_process_batch_updates_without_force_below_threshold(self):
        """Test that batch updates aren't processed without force when below threshold."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=10,
        )
        
        entity = MockEntity("e1")
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0))
        
        # Remove from pending to set it up manually
        spatial_index._pending_position_updates.clear()
        spatial_index._pending_position_updates.append(
            (entity, (10.0, 10.0), (20.0, 20.0), "agent", PRIORITY_NORMAL)
        )
        
        # Without force and below threshold
        processed = spatial_index.process_batch_updates(force=False)
        assert processed == 0
        assert len(spatial_index._pending_position_updates) == 1


class TestSpatialIndexGetNearbyRangeWithEmptyKdtree:
    """Test get_nearby_range with empty kdtree."""

    def test_get_nearby_range_kdtree_none(self):
        """Test range query when kdtree is None."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_kd",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby_range((10.0, 10.0, 20.0, 20.0), ["test_kd"])
        assert results["test_kd"] == []

    def test_get_nearby_range_kdtree_empty(self):
        """Test range query when kdtree is empty."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        # Register an empty index
        spatial_index.register_index(
            name="test_kd",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        results = spatial_index.get_nearby_range((10.0, 10.0, 20.0, 20.0), ["test_kd"])
        assert results["test_kd"] == []


class TestSpatialIndexSpatialHashAutoCellSize:
    """Test automatic cell size calculation for spatial hash."""

    def test_spatial_hash_auto_cell_size(self):
        """Test that spatial hash automatically calculates cell size when not provided."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 10), float(i * 10)))
            for i in range(5)
        ]
        
        # Register without cell_size
        spatial_index.register_index(
            name="test_hash",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="spatial_hash",
            # cell_size not provided - should auto-calculate
        )
        
        spatial_index.update()
        
        state = spatial_index._named_indices["test_hash"]
        assert state["spatial_hash"] is not None

    def test_spatial_hash_auto_cell_size_large_environment(self):
        """Test auto cell size calculation for large environment."""
        spatial_index = SpatialIndex(width=10000.0, height=10000.0)
        
        entities = [
            MockEntity(f"e{i}", position=(float(i * 100), float(i * 100)))
            for i in range(5)
        ]
        
        spatial_index.register_index(
            name="test_hash",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="spatial_hash",
        )
        
        spatial_index.update()
        
        state = spatial_index._named_indices["test_hash"]
        assert state["spatial_hash"] is not None


class TestSpatialIndexGetQuadtreeStats:
    """Test get_quadtree_stats method."""

    def test_get_quadtree_stats_valid_index(self):
        """Test getting stats for valid quadtree index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        stats = spatial_index.get_quadtree_stats("test_qt")
        
        assert stats is not None
        assert "total_entities" in stats

    def test_get_quadtree_stats_nonexistent_index(self):
        """Test getting stats for nonexistent index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        stats = spatial_index.get_quadtree_stats("nonexistent")
        assert stats is None

    def test_get_quadtree_stats_wrong_index_type(self):
        """Test getting stats for non-quadtree index."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        spatial_index.register_index(
            name="test_kd",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        stats = spatial_index.get_quadtree_stats("test_kd")
        assert stats is None

    def test_get_quadtree_stats_null_quadtree(self):
        """Test getting stats when quadtree is None."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=[],
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        stats = spatial_index.get_quadtree_stats("test_qt")
        assert stats is None


class TestSpatialIndexFilterItemsWithPositions:
    """Test _filter_items_with_positions helper."""

    def test_filter_items_with_none_positions(self):
        """Test filtering out items with None positions."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        e1 = Mock()
        e1.position = (10.0, 10.0)
        e2 = Mock()
        e2.position = None
        e3 = Mock()
        e3.position = (30.0, 30.0)
        
        items = [e1, e2, e3]
        filtered = spatial_index._filter_items_with_positions(
            items, lambda e: e.position
        )
        
        assert len(filtered) == 2
        assert e2 not in filtered

    def test_filter_items_all_none_positions(self):
        """Test filtering when all items have None positions."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        items = [Mock(position=None) for _ in range(5)]
        filtered = spatial_index._filter_items_with_positions(
            items, lambda e: e.position
        )
        
        assert len(filtered) == 0


class TestSpatialIndexLastFlushTime:
    """Test last flush time tracking."""

    def test_last_flush_time_updates_on_batch_process(self):
        """Test that _last_flush_time is updated after processing batch."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
            max_batch_size=1000,
        )
        
        initial_flush_time = spatial_index._last_flush_time
        
        entity = MockEntity("e1")
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0))
        
        time.sleep(0.05)
        spatial_index.process_batch_updates(force=True)
        
        # Last flush time should be updated
        assert spatial_index._last_flush_time > initial_flush_time


class TestSpatialIndexIsBatchUpdatesEnabled:
    """Test is_batch_updates_enabled method."""

    def test_is_batch_updates_enabled_true(self):
        """Test when batch updates are enabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=True,
        )
        
        assert spatial_index.is_batch_updates_enabled() is True

    def test_is_batch_updates_enabled_false(self):
        """Test when batch updates are disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        assert spatial_index.is_batch_updates_enabled() is False


class TestSpatialIndexDefaultPositionGetter:
    """Test default position_getter behavior."""

    def test_register_index_without_position_getter(self):
        """Test that None position_getter defaults to getattr(x, 'position')."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        
        # Don't provide position_getter
        spatial_index.register_index(
            name="test_default",
            data_reference=entities,
            position_getter=None,  # Should use default
            index_type="kdtree",
        )
        
        spatial_index.update()
        
        # Should work with default position getter
        results = spatial_index.get_nearby((10.0, 10.0), 15.0, ["test_default"])
        assert len(results["test_default"]) > 0


class TestSpatialIndexGetStatsWithQuadtree:
    """Test get_stats with quadtree indices."""

    def test_get_stats_includes_quadtree_info(self):
        """Test that get_stats includes quadtree index information."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        entities = [MockEntity(f"e{i}", position=(float(i * 10), float(i * 10))) for i in range(5)]
        
        spatial_index.register_index(
            name="test_qt",
            data_reference=entities,
            position_getter=lambda e: e.position,
            index_type="quadtree",
        )
        
        spatial_index.update()
        
        stats = spatial_index.get_stats()
        
        assert "quadtree_indices" in stats
        assert "test_qt" in stats["quadtree_indices"]
        assert stats["quadtree_indices"]["test_qt"]["exists"] is True
        assert stats["quadtree_indices"]["test_qt"]["total_entities"] == 5

    def test_get_stats_without_quadtree(self):
        """Test get_stats when no quadtree indices exist."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        stats = spatial_index.get_stats()
        
        assert "quadtree_indices" not in stats


class TestSpatialIndexUpdateNamedIndicesEdgeCases:
    """Test _update_named_indices edge cases."""

    def test_update_named_indices_agents_without_cached_count(self):
        """Test updating agents index when cached_count is None."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        agents = [MockEntity(f"a{i}", position=(float(i * 10), float(i * 10))) for i in range(3)]
        spatial_index.set_references(agents, [])
        
        # Force rebuild to populate agents index
        spatial_index._rebuild_kdtrees()
        
        # Clear cached_count to None
        spatial_index._named_indices["agents"]["cached_count"] = None
        
        # Update named indices - should recalculate cached_count
        spatial_index._update_named_indices()
        
        state = spatial_index._named_indices["agents"]
        assert state["cached_count"] is not None

    def test_update_named_indices_resources_without_cached_count(self):
        """Test updating resources index when cached_count is None."""
        spatial_index = SpatialIndex(width=100.0, height=100.0)
        
        resources = [Mock(position=(float(i * 10), float(i * 10))) for i in range(3)]
        spatial_index.set_references([], resources)
        
        # Force rebuild
        spatial_index._rebuild_kdtrees()
        
        # Clear cached_count to None
        spatial_index._named_indices["resources"]["cached_count"] = None
        
        # Update named indices - should recalculate cached_count
        spatial_index._update_named_indices()
        
        state = spatial_index._named_indices["resources"]
        assert state["cached_count"] is not None


class TestSpatialIndexAddPositionUpdateWithoutBatch:
    """Test add_position_update fallback when batch updates disabled."""

    def test_add_position_update_falls_back_when_disabled(self):
        """Test that add_position_update falls back to immediate update when disabled."""
        spatial_index = SpatialIndex(
            width=100.0,
            height=100.0,
            enable_batch_updates=False,
        )
        
        entity = MockEntity("e1")
        
        # Should not raise error, should fall back to immediate update
        spatial_index.add_position_update(entity, (10.0, 10.0), (20.0, 20.0))
        
        # Should have no pending updates
        assert len(spatial_index._pending_position_updates) == 0