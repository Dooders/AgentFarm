"""
Comprehensive tests for the SpatialHashGrid class.

This module tests the uniform grid-based spatial hash implementation including:
- Basic insert, remove, and move operations
- Radius and range queries
- Nearest neighbor search
- Edge cases and error handling
- Performance characteristics
"""

import math
import pytest
from typing import Any, Tuple

from farm.core.spatial.hash_grid import SpatialHashGrid


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, entity_id: str):
        self.entity_id = entity_id

    def __repr__(self):
        return f"MockEntity({self.entity_id})"


class TestSpatialHashGridInitialization:
    """Test SpatialHashGrid initialization and validation."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        assert grid.cell_size == 10.0
        assert grid.width == 100.0
        assert grid.height == 100.0
        assert len(grid._buckets) == 0

    def test_initialization_with_small_cell_size(self):
        """Test initialization with very small cell size."""
        grid = SpatialHashGrid(cell_size=0.1, width=100.0, height=100.0)
        assert grid.cell_size == 0.1

    def test_initialization_with_large_environment(self):
        """Test initialization with large environment."""
        grid = SpatialHashGrid(cell_size=50.0, width=10000.0, height=10000.0)
        assert grid.width == 10000.0
        assert grid.height == 10000.0

    def test_invalid_cell_size_zero(self):
        """Test that zero cell size raises ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            SpatialHashGrid(cell_size=0.0, width=100.0, height=100.0)

    def test_invalid_cell_size_negative(self):
        """Test that negative cell size raises ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            SpatialHashGrid(cell_size=-10.0, width=100.0, height=100.0)


class TestSpatialHashGridCellCoordinates:
    """Test cell coordinate calculations."""

    def test_cell_coords_origin(self):
        """Test cell coordinates at origin."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        coords = grid._cell_coords((0.0, 0.0))
        assert coords == (0, 0)

    def test_cell_coords_positive(self):
        """Test cell coordinates in positive quadrant."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        coords = grid._cell_coords((15.0, 25.0))
        assert coords == (1, 2)

    def test_cell_coords_boundaries(self):
        """Test cell coordinates at cell boundaries."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        coords1 = grid._cell_coords((10.0, 10.0))
        coords2 = grid._cell_coords((9.99, 9.99))
        assert coords1 == (1, 1)
        assert coords2 == (0, 0)

    def test_cell_coords_negative(self):
        """Test cell coordinates with negative positions."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        coords = grid._cell_coords((-15.0, -25.0))
        assert coords == (-2, -3)  # Floor division behavior


class TestSpatialHashGridInsertRemove:
    """Test insert and remove operations."""

    def test_insert_single_entity(self):
        """Test inserting a single entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        grid.insert(entity, (5.0, 5.0))

        key = grid._cell_coords((5.0, 5.0))
        assert key in grid._buckets
        assert len(grid._buckets[key]) == 1
        assert grid._buckets[key][0] == (entity, (5.0, 5.0))

    def test_insert_multiple_entities_same_cell(self):
        """Test inserting multiple entities into the same cell."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        grid.insert(e1, (5.0, 5.0))
        grid.insert(e2, (7.0, 7.0))

        key = grid._cell_coords((5.0, 5.0))
        assert len(grid._buckets[key]) == 2

    def test_insert_multiple_entities_different_cells(self):
        """Test inserting entities into different cells."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        grid.insert(e1, (5.0, 5.0))
        grid.insert(e2, (15.0, 25.0))

        assert len(grid._buckets) == 2

    def test_remove_existing_entity(self):
        """Test removing an existing entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        position = (5.0, 5.0)

        grid.insert(entity, position)
        result = grid.remove(entity, position)

        assert result is True
        key = grid._cell_coords(position)
        assert key not in grid._buckets  # Bucket should be cleaned up

    def test_remove_nonexistent_entity(self):
        """Test removing an entity that doesn't exist."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        result = grid.remove(entity, (5.0, 5.0))
        assert result is False

    def test_remove_from_empty_bucket(self):
        """Test removing from a cell that has no bucket."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        result = grid.remove(entity, (5.0, 5.0))
        assert result is False

    def test_remove_wrong_position(self):
        """Test removing entity from wrong position."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (5.0, 5.0))
        result = grid.remove(entity, (15.0, 15.0))  # Wrong cell

        assert result is False
        # Entity should still be in original cell
        key = grid._cell_coords((5.0, 5.0))
        assert key in grid._buckets

    def test_remove_cleans_up_empty_bucket(self):
        """Test that removing last entity cleans up the bucket."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        grid.insert(e1, (5.0, 5.0))
        grid.insert(e2, (5.0, 5.0))

        key = grid._cell_coords((5.0, 5.0))
        assert len(grid._buckets[key]) == 2

        grid.remove(e1, (5.0, 5.0))
        assert len(grid._buckets[key]) == 1

        grid.remove(e2, (5.0, 5.0))
        assert key not in grid._buckets  # Bucket cleaned up


class TestSpatialHashGridMove:
    """Test move operations."""

    def test_move_within_same_cell(self):
        """Test moving entity within the same cell."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (5.0, 5.0))
        grid.move(entity, (5.0, 5.0), (7.0, 7.0))

        key = grid._cell_coords((7.0, 7.0))
        assert len(grid._buckets[key]) == 1
        assert grid._buckets[key][0] == (entity, (7.0, 7.0))

    def test_move_to_different_cell(self):
        """Test moving entity to a different cell."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (5.0, 5.0))
        grid.move(entity, (5.0, 5.0), (15.0, 25.0))

        old_key = grid._cell_coords((5.0, 5.0))
        new_key = grid._cell_coords((15.0, 25.0))

        assert old_key not in grid._buckets  # Old bucket cleaned up
        assert new_key in grid._buckets
        assert grid._buckets[new_key][0] == (entity, (15.0, 25.0))

    def test_move_within_cell_updates_position(self):
        """Test that moving within a cell updates the stored position."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (5.0, 5.0))
        grid.move(entity, (5.0, 5.0), (8.0, 8.0))

        key = grid._cell_coords((8.0, 8.0))
        bucket = grid._buckets[key]
        assert bucket[0][1] == (8.0, 8.0)

    def test_move_nonexistent_entity(self):
        """Test moving an entity that doesn't exist (should insert)."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        # Move should work even if entity isn't in old position
        grid.move(entity, (5.0, 5.0), (15.0, 15.0))

        new_key = grid._cell_coords((15.0, 15.0))
        assert new_key in grid._buckets


class TestSpatialHashGridQueryRadius:
    """Test radius query operations."""

    def test_query_radius_empty_grid(self):
        """Test radius query on empty grid."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        results = grid.query_radius((50.0, 50.0), 10.0)
        assert results == []

    def test_query_radius_negative(self):
        """Test radius query with negative radius."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        results = grid.query_radius((50.0, 50.0), -5.0)
        assert results == []

    def test_query_radius_zero(self):
        """Test radius query with zero radius."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        results = grid.query_radius((50.0, 50.0), 0.0)
        assert results == []

    def test_query_radius_single_entity(self):
        """Test radius query finding single entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        grid.insert(entity, position)
        results = grid.query_radius((50.0, 50.0), 5.0)

        assert len(results) == 1
        assert results[0] == (entity, position)

    def test_query_radius_multiple_entities(self):
        """Test radius query finding multiple entities."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entities = [MockEntity(f"e{i}") for i in range(5)]
        positions = [(50.0, 50.0), (52.0, 52.0), (55.0, 55.0), (70.0, 70.0), (90.0, 90.0)]

        for entity, pos in zip(entities, positions):
            grid.insert(entity, pos)

        results = grid.query_radius((50.0, 50.0), 10.0)
        result_entities = [entity for entity, _ in results]

        # First three entities should be within radius
        assert len(results) >= 3
        assert entities[0] in result_entities
        assert entities[1] in result_entities

    def test_query_radius_exact_boundary(self):
        """Test radius query at exact boundary distance."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        # Place entity exactly 10 units away
        grid.insert(entity, (60.0, 50.0))

        # Query with radius 10 should include it
        results = grid.query_radius((50.0, 50.0), 10.0)
        assert len(results) == 1

    def test_query_radius_multiple_cells(self):
        """Test radius query spanning multiple cells."""
        grid = SpatialHashGrid(cell_size=5.0, width=100.0, height=100.0)
        entities = []

        # Create a grid of entities
        for x in range(0, 30, 5):
            for y in range(0, 30, 5):
                entity = MockEntity(f"e_{x}_{y}")
                grid.insert(entity, (float(x), float(y)))
                entities.append(entity)

        # Query should span multiple cells
        results = grid.query_radius((15.0, 15.0), 12.0)
        assert len(results) > 0


class TestSpatialHashGridQueryRange:
    """Test range query operations."""

    def test_query_range_empty_grid(self):
        """Test range query on empty grid."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        results = grid.query_range((10.0, 10.0, 20.0, 20.0))
        assert results == []

    def test_query_range_zero_width(self):
        """Test range query with zero width."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        grid.insert(entity, (50.0, 50.0))

        results = grid.query_range((45.0, 45.0, 0.0, 10.0))
        assert results == []

    def test_query_range_zero_height(self):
        """Test range query with zero height."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        grid.insert(entity, (50.0, 50.0))

        results = grid.query_range((45.0, 45.0, 10.0, 0.0))
        assert results == []

    def test_query_range_negative_dimensions(self):
        """Test range query with negative dimensions."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        grid.insert(entity, (50.0, 50.0))

        results = grid.query_range((45.0, 45.0, -10.0, 10.0))
        assert results == []

    def test_query_range_single_entity(self):
        """Test range query finding single entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        grid.insert(entity, position)
        results = grid.query_range((45.0, 45.0, 10.0, 10.0))

        assert len(results) == 1
        assert results[0] == (entity, position)

    def test_query_range_multiple_entities(self):
        """Test range query finding multiple entities."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entities = [MockEntity(f"e{i}") for i in range(5)]
        positions = [(10.0, 10.0), (12.0, 12.0), (15.0, 15.0), (50.0, 50.0), (90.0, 90.0)]

        for entity, pos in zip(entities, positions):
            grid.insert(entity, pos)

        results = grid.query_range((5.0, 5.0, 15.0, 15.0))
        result_entities = [entity for entity, _ in results]

        # First three entities should be in range
        assert entities[0] in result_entities
        assert entities[1] in result_entities
        assert entities[2] in result_entities
        assert entities[3] not in result_entities

    def test_query_range_boundary_inclusive(self):
        """Test that range query uses inclusive lower bounds and exclusive upper bounds."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        # Entity at lower boundary (should be included)
        grid.insert(e1, (10.0, 10.0))
        # Entity at upper boundary (should be excluded)
        grid.insert(e2, (20.0, 20.0))

        results = grid.query_range((10.0, 10.0, 10.0, 10.0))
        result_entities = [entity for entity, _ in results]

        assert e1 in result_entities
        assert e2 not in result_entities

    def test_query_range_spanning_multiple_cells(self):
        """Test range query spanning multiple cells."""
        grid = SpatialHashGrid(cell_size=5.0, width=100.0, height=100.0)

        # Create entities in different cells
        entities = []
        for x in range(0, 30, 5):
            for y in range(0, 30, 5):
                entity = MockEntity(f"e_{x}_{y}")
                grid.insert(entity, (float(x), float(y)))
                entities.append(entity)

        # Query a large range
        results = grid.query_range((5.0, 5.0, 20.0, 20.0))
        assert len(results) > 0


class TestSpatialHashGridGetNearest:
    """Test nearest neighbor search."""

    def test_get_nearest_empty_grid(self):
        """Test get_nearest on empty grid."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        result = grid.get_nearest((50.0, 50.0))
        assert result is None

    def test_get_nearest_single_entity(self):
        """Test get_nearest with single entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (50.0, 50.0))
        result = grid.get_nearest((55.0, 55.0))

        assert result == entity

    def test_get_nearest_same_position(self):
        """Test get_nearest when query position is same as entity."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        grid.insert(entity, position)
        result = grid.get_nearest(position)

        assert result == entity

    def test_get_nearest_multiple_entities(self):
        """Test get_nearest with multiple entities."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")
        e3 = MockEntity("e3")

        grid.insert(e1, (10.0, 10.0))
        grid.insert(e2, (50.0, 50.0))
        grid.insert(e3, (90.0, 90.0))

        # Nearest to (51, 51) should be e2
        result = grid.get_nearest((51.0, 51.0))
        assert result == e2

    def test_get_nearest_in_different_cells(self):
        """Test get_nearest when entities are in different cells."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        grid.insert(e1, (5.0, 5.0))
        grid.insert(e2, (25.0, 25.0))

        result = grid.get_nearest((20.0, 20.0))
        assert result == e2  # e2 is closer

    def test_get_nearest_expands_search(self):
        """Test that get_nearest expands search to find distant entities."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        # Place entity far from query point
        grid.insert(entity, (90.0, 90.0))
        result = grid.get_nearest((10.0, 10.0))

        assert result == entity  # Should find it despite distance

    def test_get_nearest_early_termination(self):
        """Test that get_nearest terminates early when best candidate found."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        # Place one entity very close, another far
        grid.insert(e1, (50.0, 50.0))
        grid.insert(e2, (90.0, 90.0))

        result = grid.get_nearest((51.0, 51.0))
        assert result == e1

    def test_get_nearest_with_multiple_candidates_same_cell(self):
        """Test get_nearest with multiple entities in the same cell."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        e1 = MockEntity("e1")
        e2 = MockEntity("e2")
        e3 = MockEntity("e3")

        grid.insert(e1, (50.0, 50.0))
        grid.insert(e2, (52.0, 52.0))
        grid.insert(e3, (58.0, 58.0))

        result = grid.get_nearest((51.0, 51.0))
        assert result in [e1, e2]  # Either could be nearest depending on exact distance


class TestSpatialHashGridBucketKeys:
    """Test bucket key generation for bounds."""

    def test_bucket_keys_for_bounds_single_cell(self):
        """Test bucket keys for bounds within a single cell."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        keys = grid._bucket_keys_for_bounds((5.0, 5.0, 5.0, 5.0))
        assert (0, 0) in keys

    def test_bucket_keys_for_bounds_multiple_cells(self):
        """Test bucket keys for bounds spanning multiple cells."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        keys = grid._bucket_keys_for_bounds((5.0, 5.0, 20.0, 20.0))

        # Should span cells (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
        assert len(keys) >= 4
        assert (0, 0) in keys
        assert (1, 1) in keys

    def test_bucket_keys_for_bounds_zero_width(self):
        """Test bucket keys with zero width."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        keys = grid._bucket_keys_for_bounds((5.0, 5.0, 0.0, 10.0))
        assert keys == []

    def test_bucket_keys_for_bounds_zero_height(self):
        """Test bucket keys with zero height."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        keys = grid._bucket_keys_for_bounds((5.0, 5.0, 10.0, 0.0))
        assert keys == []

    def test_bucket_keys_for_bounds_negative_dimensions(self):
        """Test bucket keys with negative dimensions."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        keys = grid._bucket_keys_for_bounds((5.0, 5.0, -10.0, 10.0))
        assert keys == []


class TestSpatialHashGridEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_cell_size(self):
        """Test grid with very small cell size."""
        grid = SpatialHashGrid(cell_size=0.01, width=10.0, height=10.0)
        entity = MockEntity("e1")

        grid.insert(entity, (5.0, 5.0))
        results = grid.query_radius((5.0, 5.0), 1.0)

        assert len(results) == 1

    def test_very_large_cell_size(self):
        """Test grid with very large cell size."""
        grid = SpatialHashGrid(cell_size=1000.0, width=100.0, height=100.0)
        entities = [MockEntity(f"e{i}") for i in range(3)]

        for i, entity in enumerate(entities):
            grid.insert(entity, (float(i * 20), float(i * 20)))

        # All entities should be in the same cell
        results = grid.query_radius((50.0, 50.0), 100.0)
        assert len(results) == 3

    def test_positions_at_negative_coordinates(self):
        """Test handling of negative coordinates."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (-5.0, -5.0))
        results = grid.query_radius((-5.0, -5.0), 2.0)

        assert len(results) == 1

    def test_positions_outside_environment_bounds(self):
        """Test handling positions outside environment bounds."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        # Grid should still work with positions outside nominal bounds
        grid.insert(entity, (150.0, 150.0))
        results = grid.query_radius((150.0, 150.0), 10.0)

        assert len(results) == 1

    def test_many_entities_same_position(self):
        """Test handling many entities at the exact same position."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        position = (50.0, 50.0)
        entities = [MockEntity(f"e{i}") for i in range(100)]

        for entity in entities:
            grid.insert(entity, position)

        results = grid.query_radius(position, 1.0)
        assert len(results) == 100

    def test_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        # Insert with high precision
        position = (50.123456789, 50.987654321)
        grid.insert(entity, position)

        # Query with slightly different precision
        results = grid.query_radius((50.123456789, 50.987654321), 0.1)
        assert len(results) == 1


class TestSpatialHashGridIntegration:
    """Integration tests for combined operations."""

    def test_insert_query_remove_sequence(self):
        """Test a sequence of insert, query, and remove operations."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        # Insert
        grid.insert(entity, position)
        assert len(grid.query_radius(position, 5.0)) == 1

        # Remove
        grid.remove(entity, position)
        assert len(grid.query_radius(position, 5.0)) == 0

    def test_move_and_query_sequence(self):
        """Test moving entities and querying."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entity = MockEntity("e1")

        grid.insert(entity, (10.0, 10.0))
        grid.move(entity, (10.0, 10.0), (50.0, 50.0))

        # Should not find at old position
        old_results = grid.query_radius((10.0, 10.0), 5.0)
        assert len(old_results) == 0

        # Should find at new position
        new_results = grid.query_radius((50.0, 50.0), 5.0)
        assert len(new_results) == 1

    def test_multiple_operations_mixed(self):
        """Test mixed operations (insert, move, remove, query)."""
        grid = SpatialHashGrid(cell_size=10.0, width=100.0, height=100.0)
        entities = [MockEntity(f"e{i}") for i in range(10)]

        # Insert entities
        for i, entity in enumerate(entities[:5]):
            grid.insert(entity, (float(i * 10), float(i * 10)))

        # Move some entities
        grid.move(entities[0], (0.0, 0.0), (50.0, 50.0))

        # Remove some entities
        grid.remove(entities[1], (10.0, 10.0))

        # Add more entities
        for i, entity in enumerate(entities[5:]):
            grid.insert(entity, (float((i + 5) * 10), float((i + 5) * 10)))

        # Query should return correct results
        results = grid.query_radius((50.0, 50.0), 20.0)
        assert len(results) > 0

    def test_stress_many_entities(self):
        """Test with many entities to verify performance."""
        grid = SpatialHashGrid(cell_size=20.0, width=1000.0, height=1000.0)
        entities = [MockEntity(f"e{i}") for i in range(1000)]

        # Insert many entities
        import random
        random.seed(42)
        for entity in entities:
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            grid.insert(entity, (x, y))

        # Query should still work efficiently
        results = grid.query_radius((500.0, 500.0), 50.0)
        assert len(results) > 0

        # Get nearest should still work
        nearest = grid.get_nearest((500.0, 500.0))
        assert nearest is not None


class TestSpatialHashGridPerformance:
    """Test performance characteristics."""

    def test_query_performance_scales_with_density(self):
        """Test that query performance depends on local density, not total entities."""
        import time

        grid = SpatialHashGrid(cell_size=20.0, width=1000.0, height=1000.0)

        # Add entities far from query point
        for i in range(1000):
            grid.insert(MockEntity(f"far_{i}"), (900.0, 900.0))

        # Add a few entities near query point
        for i in range(10):
            grid.insert(MockEntity(f"near_{i}"), (50.0 + i, 50.0 + i))

        # Query near the sparse area should be fast
        start = time.time()
        results = grid.query_radius((50.0, 50.0), 30.0)
        elapsed = time.time() - start

        # Should complete quickly despite 1000+ total entities
        assert elapsed < 0.1
        assert len(results) > 0

    def test_consistent_results_regardless_of_cell_size(self):
        """Test that query results are consistent across different cell sizes."""
        positions = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (50.0, 50.0)]
        entities = [MockEntity(f"e{i}") for i in range(len(positions))]

        # Test with different cell sizes
        for cell_size in [5.0, 10.0, 20.0, 50.0]:
            grid = SpatialHashGrid(cell_size=cell_size, width=100.0, height=100.0)

            for entity, pos in zip(entities, positions):
                grid.insert(entity, pos)

            # Query should return same entities regardless of cell size
            results = grid.query_radius((25.0, 25.0), 15.0)
            result_ids = {entity.entity_id for entity, _ in results}

            # Should find entities at (20, 20) and (30, 30)
            assert "e1" in result_ids or "e2" in result_ids
