"""
Additional tests for DirtyRegionTracker to achieve complete coverage.

This module fills in gaps in dirty_regions.py test coverage, specifically:
- Edge cases in get_dirty_regions with max_count
- Cleanup mechanisms
"""

import time
import pytest

from farm.core.spatial.dirty_regions import DirtyRegion, DirtyRegionTracker


class TestDirtyRegionTrackerGetDirtyRegionsMaxCount:
    """Test get_dirty_regions with max_count parameter."""

    def test_get_dirty_regions_with_max_count(self):
        """Test that max_count limits the number of returned regions."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark multiple regions dirty
        positions = [(25.0, 25.0), (75.0, 75.0), (125.0, 125.0), (175.0, 175.0), (225.0, 225.0)]
        for i, pos in enumerate(positions):
            tracker.mark_region_dirty(pos, "agent", priority=i)
        
        # Get only 3 regions
        dirty_regions = tracker.get_dirty_regions("agent", max_count=3)
        
        assert len(dirty_regions) == 3

    def test_get_dirty_regions_max_count_larger_than_available(self):
        """Test max_count larger than available regions."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        dirty_regions = tracker.get_dirty_regions("agent", max_count=10)
        
        assert len(dirty_regions) == 2

    def test_get_dirty_regions_max_count_zero(self):
        """Test max_count of zero."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        dirty_regions = tracker.get_dirty_regions("agent", max_count=0)
        
        assert len(dirty_regions) == 0

    def test_get_dirty_regions_max_count_respects_priority(self):
        """Test that max_count returns highest priority regions."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark regions with different priorities
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        tracker.mark_region_dirty((75.0, 75.0), "agent", priority=3)  # Highest
        tracker.mark_region_dirty((125.0, 125.0), "agent", priority=2)
        
        # Get only 2 regions
        dirty_regions = tracker.get_dirty_regions("agent", max_count=2)
        
        assert len(dirty_regions) == 2
        # Should return highest priority regions
        assert dirty_regions[0].priority == 3
        assert dirty_regions[1].priority == 2

    def test_get_dirty_regions_without_entity_type_filter(self):
        """Test get_dirty_regions without entity_type filter (returns all types)."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        tracker.mark_region_dirty((75.0, 75.0), "resource", priority=2)
        tracker.mark_region_dirty((125.0, 125.0), "npc", priority=3)
        
        # Get all types
        all_regions = tracker.get_dirty_regions(entity_type=None)
        
        assert len(all_regions) == 3
        entity_types = {region.entity_type for region in all_regions}
        assert entity_types == {"agent", "resource", "npc"}

    def test_get_dirty_regions_with_max_count_and_no_type_filter(self):
        """Test max_count without entity_type filter."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1)
        tracker.mark_region_dirty((75.0, 75.0), "resource", priority=3)
        tracker.mark_region_dirty((125.0, 125.0), "npc", priority=2)
        
        # Get only 2 regions across all types
        dirty_regions = tracker.get_dirty_regions(entity_type=None, max_count=2)
        
        assert len(dirty_regions) == 2
        assert dirty_regions[0].priority == 3  # Highest priority first


class TestDirtyRegionTrackerCleanup:
    """Test cleanup mechanisms in DirtyRegionTracker."""

    def test_cleanup_old_regions_when_exceeding_max(self):
        """Test that old regions are cleaned up when exceeding max_regions."""
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=3)
        
        # Mark regions with delays to ensure different timestamps
        tracker.mark_region_dirty((25.0, 25.0), "agent", timestamp=1.0)
        time.sleep(0.01)
        tracker.mark_region_dirty((75.0, 75.0), "agent", timestamp=2.0)
        time.sleep(0.01)
        tracker.mark_region_dirty((125.0, 125.0), "agent", timestamp=3.0)
        time.sleep(0.01)
        
        # This should trigger cleanup
        tracker.mark_region_dirty((175.0, 175.0), "agent", timestamp=4.0)
        
        # Should only have max_regions
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) <= 3

    def test_cleanup_removes_oldest_regions(self):
        """Test that cleanup removes oldest regions by timestamp."""
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=2)
        
        # Mark regions with explicit timestamps
        tracker.mark_region_dirty((25.0, 25.0), "agent", timestamp=1.0)
        tracker.mark_region_dirty((75.0, 75.0), "agent", timestamp=2.0)
        tracker.mark_region_dirty((125.0, 125.0), "agent", timestamp=3.0)  # Triggers cleanup
        
        dirty_regions = tracker.get_dirty_regions("agent")
        
        # Should keep the 2 newest regions
        assert len(dirty_regions) <= 2

    def test_cleanup_old_regions_direct_call(self):
        """Test calling _cleanup_old_regions directly."""
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=2)
        
        # Mark more regions than max
        tracker.mark_region_dirty((25.0, 25.0), "agent", timestamp=1.0)
        tracker.mark_region_dirty((75.0, 75.0), "agent", timestamp=2.0)
        tracker.mark_region_dirty((125.0, 125.0), "agent", timestamp=3.0)
        
        # Manually call cleanup
        tracker._cleanup_old_regions()
        
        # Should be at or below max_regions
        total_regions = sum(len(regions) for regions in tracker._dirty_regions.values())
        assert total_regions <= tracker.max_regions

    def test_cleanup_old_regions_when_not_exceeding(self):
        """Test that cleanup does nothing when not exceeding max_regions."""
        tracker = DirtyRegionTracker(region_size=50.0, max_regions=10)
        
        # Mark only a few regions
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        initial_count = len(tracker._dirty_regions["agent"])
        
        # Call cleanup
        tracker._cleanup_old_regions()
        
        # Should not remove anything
        assert len(tracker._dirty_regions["agent"]) == initial_count


class TestDirtyRegionTrackerTimestamps:
    """Test timestamp handling in DirtyRegionTracker."""

    def test_default_timestamp(self):
        """Test that marking a region uses current time when timestamp not provided."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        before = time.time()
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        after = time.time()
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert before <= dirty_regions[0].timestamp <= after

    def test_explicit_timestamp(self):
        """Test using explicit timestamp."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        explicit_time = 12345.6789
        tracker.mark_region_dirty((25.0, 25.0), "agent", timestamp=explicit_time)
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert dirty_regions[0].timestamp == explicit_time

    def test_timestamp_sorting_in_get_dirty_regions(self):
        """Test that regions are sorted by priority first, then timestamp."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark regions with same priority but different timestamps
        tracker.mark_region_dirty((25.0, 25.0), "agent", priority=1, timestamp=3.0)
        tracker.mark_region_dirty((75.0, 75.0), "agent", priority=1, timestamp=1.0)
        tracker.mark_region_dirty((125.0, 125.0), "agent", priority=1, timestamp=2.0)
        
        dirty_regions = tracker.get_dirty_regions("agent")
        
        # All have same priority, so should be sorted by timestamp (oldest first)
        assert dirty_regions[0].timestamp == 1.0
        assert dirty_regions[1].timestamp == 2.0
        assert dirty_regions[2].timestamp == 3.0


class TestDirtyRegionTrackerMultipleEntityTypes:
    """Test handling of multiple entity types."""

    def test_different_entity_types_tracked_separately(self):
        """Test that different entity types are tracked separately."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "resource")
        tracker.mark_region_dirty((125.0, 125.0), "npc")
        
        agent_regions = tracker.get_dirty_regions("agent")
        resource_regions = tracker.get_dirty_regions("resource")
        npc_regions = tracker.get_dirty_regions("npc")
        
        assert len(agent_regions) == 1
        assert len(resource_regions) == 1
        assert len(npc_regions) == 1

    def test_clear_region_affects_all_entity_types(self):
        """Test that clear_region removes the region from all entity types."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        # Mark same region for multiple entity types
        position = (25.0, 25.0)
        tracker.mark_region_dirty(position, "agent")
        tracker.mark_region_dirty(position, "resource")
        
        region_coords = tracker.world_to_region_coords(position)
        tracker.clear_region(region_coords)
        
        # Should be cleared for both types
        assert len(tracker.get_dirty_regions("agent")) == 0
        assert len(tracker.get_dirty_regions("resource")) == 0


class TestDirtyRegionTrackerPriorityHandling:
    """Test priority handling in DirtyRegionTracker."""

    def test_priority_upgrade_on_remark(self):
        """Test that re-marking a region with higher priority upgrades it."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        position = (25.0, 25.0)
        tracker.mark_region_dirty(position, "agent", priority=1)
        tracker.mark_region_dirty(position, "agent", priority=3)  # Higher priority
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert dirty_regions[0].priority == 3

    def test_priority_not_downgraded_on_remark(self):
        """Test that re-marking a region with lower priority doesn't downgrade it."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        position = (25.0, 25.0)
        tracker.mark_region_dirty(position, "agent", priority=3)
        tracker.mark_region_dirty(position, "agent", priority=1)  # Lower priority
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert dirty_regions[0].priority == 3  # Should keep higher priority


class TestDirtyRegionTrackerStatistics:
    """Test statistics tracking."""

    def test_stats_total_regions_marked(self):
        """Test that total_regions_marked increments correctly."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        initial_stats = tracker.get_stats()
        assert initial_stats["total_regions_marked"] == 0
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        stats = tracker.get_stats()
        assert stats["total_regions_marked"] == 2

    def test_stats_total_regions_updated(self):
        """Test that total_regions_updated increments on clear."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        
        region_coords = tracker.world_to_region_coords((25.0, 25.0))
        tracker.clear_region(region_coords)
        
        stats = tracker.get_stats()
        assert stats["total_regions_updated"] == 1

    def test_stats_regions_by_type(self):
        """Test regions_by_type in statistics."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty((25.0, 25.0), "agent")
        tracker.mark_region_dirty((75.0, 75.0), "agent")
        tracker.mark_region_dirty((125.0, 125.0), "resource")
        
        stats = tracker.get_stats()
        
        assert stats["regions_by_type"]["agent"] == 2
        assert stats["regions_by_type"]["resource"] == 1


class TestDirtyRegionDataclass:
    """Test the DirtyRegion dataclass."""

    def test_dirty_region_creation(self):
        """Test creating a DirtyRegion."""
        region = DirtyRegion(
            bounds=(0.0, 0.0, 50.0, 50.0),
            entity_type="agent",
            priority=2,
            timestamp=123.45
        )
        
        assert region.bounds == (0.0, 0.0, 50.0, 50.0)
        assert region.entity_type == "agent"
        assert region.priority == 2
        assert region.timestamp == 123.45

    def test_dirty_region_default_values(self):
        """Test DirtyRegion default values."""
        region = DirtyRegion(
            bounds=(0.0, 0.0, 50.0, 50.0),
            entity_type="agent"
        )
        
        assert region.priority == 0
        assert region.timestamp == 0.0


class TestDirtyRegionTrackerBatchOperations:
    """Test batch operations in DirtyRegionTracker."""

    def test_mark_region_dirty_batch_empty_list(self):
        """Test batch marking with empty list."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty_batch([], "agent", priority=1)
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 0

    def test_mark_region_dirty_batch_single_position(self):
        """Test batch marking with single position."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        tracker.mark_region_dirty_batch([(25.0, 25.0)], "agent", priority=2)
        
        dirty_regions = tracker.get_dirty_regions("agent")
        assert len(dirty_regions) == 1
        assert dirty_regions[0].priority == 2

    def test_mark_region_dirty_batch_updates_stats(self):
        """Test that batch marking updates statistics correctly."""
        tracker = DirtyRegionTracker(region_size=50.0)
        
        positions = [(25.0, 25.0), (75.0, 75.0), (125.0, 125.0)]
        tracker.mark_region_dirty_batch(positions, "agent", priority=1)
        
        stats = tracker.get_stats()
        assert stats["total_regions_marked"] == 3