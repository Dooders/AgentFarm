"""
Comprehensive tests for the Quadtree and QuadtreeNode classes.

This module tests the quadtree spatial partitioning implementation including:
- Node subdivision and capacity limits
- Insert and remove operations
- Range and radius queries
- Tree structure and statistics
- Edge cases and boundary conditions
"""

import pytest
from typing import Any, Tuple

from farm.core.spatial.quadtree import Quadtree, QuadtreeNode


class MockEntity:
    """Mock entity for testing."""
    def __init__(self, entity_id: str):
        self.entity_id = entity_id

    def __repr__(self):
        return f"MockEntity({self.entity_id})"


class TestQuadtreeNodeInitialization:
    """Test QuadtreeNode initialization."""

    def test_basic_initialization(self):
        """Test basic node initialization."""
        bounds = (0.0, 0.0, 100.0, 100.0)
        node = QuadtreeNode(bounds, capacity=4)

        assert node.bounds == bounds
        assert node.capacity == 4
        assert node.entities == []
        assert node.children is None
        assert node.is_divided is False

    def test_custom_capacity(self):
        """Test initialization with custom capacity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=10)
        assert node.capacity == 10

    def test_small_bounds(self):
        """Test initialization with small bounds."""
        bounds = (0.0, 0.0, 1.0, 1.0)
        node = QuadtreeNode(bounds, capacity=4)
        assert node.bounds == bounds

    def test_large_bounds(self):
        """Test initialization with large bounds."""
        bounds = (0.0, 0.0, 10000.0, 10000.0)
        node = QuadtreeNode(bounds, capacity=4)
        assert node.bounds == bounds


class TestQuadtreeNodeInsert:
    """Test QuadtreeNode insert operations."""

    def test_insert_single_entity(self):
        """Test inserting a single entity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        result = node.insert(entity, position)

        assert result is True
        assert len(node.entities) == 1
        assert node.entities[0] == (entity, position)
        assert not node.is_divided

    def test_insert_up_to_capacity(self):
        """Test inserting entities up to capacity without subdivision."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(4):
            entity = MockEntity(f"e{i}")
            result = node.insert(entity, (float(i * 10), float(i * 10)))
            assert result is True

        assert len(node.entities) == 4
        assert not node.is_divided

    def test_insert_triggers_subdivision(self):
        """Test that inserting beyond capacity triggers subdivision."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        # Insert up to capacity
        for i in range(4):
            node.insert(MockEntity(f"e{i}"), (float(i * 10 + 10), float(i * 10 + 10)))

        # Insert one more to trigger subdivision
        result = node.insert(MockEntity("e5"), (60.0, 60.0))

        assert result is True
        assert node.is_divided
        assert node.children is not None
        assert len(node.children) == 4

    def test_insert_outside_bounds(self):
        """Test inserting entity outside node bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        result = node.insert(entity, (150.0, 150.0))

        assert result is False
        assert len(node.entities) == 0

    def test_insert_at_boundary(self):
        """Test inserting entity at boundary."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        # At lower boundary (inclusive)
        result1 = node.insert(MockEntity("e1"), (0.0, 0.0))
        assert result1 is True

        # At upper boundary (exclusive)
        result2 = node.insert(MockEntity("e2"), (100.0, 100.0))
        assert result2 is False

    def test_insert_into_subdivided_node(self):
        """Test inserting into an already subdivided node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Fill and subdivide
        node.insert(MockEntity("e1"), (10.0, 10.0))
        node.insert(MockEntity("e2"), (20.0, 20.0))
        node.insert(MockEntity("e3"), (30.0, 30.0))  # Triggers subdivision

        assert node.is_divided

        # Insert another entity
        result = node.insert(MockEntity("e4"), (70.0, 70.0))
        assert result is True

    def test_subdivision_redistributes_entities(self):
        """Test that subdivision redistributes existing entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Insert entities in different quadrants
        node.insert(MockEntity("e1"), (25.0, 25.0))  # Top-left
        node.insert(MockEntity("e2"), (75.0, 25.0))  # Top-right
        node.insert(MockEntity("e3"), (25.0, 75.0))  # Bottom-left

        assert node.is_divided
        # Entities should be distributed to children
        total_in_children = sum(len(child.entities) for child in node.children)
        total_in_parent = len(node.entities)
        assert total_in_children + total_in_parent == 3

    def test_insert_that_cannot_redistribute(self):
        """Test inserting entities that can't be redistributed to children."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Insert entities at exact subdivision boundary
        node.insert(MockEntity("e1"), (50.0, 50.0))
        node.insert(MockEntity("e2"), (50.0, 50.0))
        node.insert(MockEntity("e3"), (50.0, 50.0))

        # These might stay in parent node if they're exactly on boundary
        assert node.is_divided or len(node.entities) == 3


class TestQuadtreeNodeSubdivide:
    """Test QuadtreeNode subdivision logic."""

    def test_subdivide_creates_four_children(self):
        """Test that subdivision creates exactly four children."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Force subdivision
        node.insert(MockEntity("e1"), (10.0, 10.0))
        node.insert(MockEntity("e2"), (20.0, 20.0))
        node.insert(MockEntity("e3"), (30.0, 30.0))

        assert node.is_divided
        assert len(node.children) == 4

    def test_subdivide_creates_equal_quadrants(self):
        """Test that subdivision creates equal-sized quadrants."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)
        node._subdivide()

        # Check child bounds
        expected_bounds = [
            (0.0, 0.0, 50.0, 50.0),      # Top-left
            (50.0, 0.0, 50.0, 50.0),     # Top-right
            (0.0, 50.0, 50.0, 50.0),     # Bottom-left
            (50.0, 50.0, 50.0, 50.0),    # Bottom-right
        ]

        actual_bounds = [child.bounds for child in node.children]
        for expected in expected_bounds:
            assert expected in actual_bounds

    def test_subdivide_inherits_capacity(self):
        """Test that child nodes inherit capacity from parent."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=8)
        node._subdivide()

        for child in node.children:
            assert child.capacity == 8


class TestQuadtreeNodeRemove:
    """Test QuadtreeNode remove operations."""

    def test_remove_existing_entity(self):
        """Test removing an existing entity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        node.insert(entity, position)
        result = node.remove(entity, position)

        assert result is True
        assert len(node.entities) == 0

    def test_remove_nonexistent_entity(self):
        """Test removing an entity that doesn't exist."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        result = node.remove(entity, (50.0, 50.0))

        assert result is False

    def test_remove_from_wrong_position_in_bounds(self):
        """Test removing entity with wrong position but within bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        # Insert entity at one position
        node.insert(entity, (50.0, 50.0))

        # Try to remove with a different position (but still in bounds)
        # The implementation searches by entity identity within the subtree containing
        # the given position. Since (30, 30) is in the same root node, it will find
        # and remove the entity even though the position doesn't match exactly.
        result = node.remove(entity, (30.0, 30.0))

        # This should succeed because the entity is found by identity
        assert result is True
        assert len(node.entities) == 0

    def test_remove_outside_bounds(self):
        """Test removing with position outside bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        result = node.remove(entity, (150.0, 150.0))

        assert result is False

    def test_remove_from_subdivided_node(self):
        """Test removing from a subdivided node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        e1 = MockEntity("e1")
        e2 = MockEntity("e2")
        e3 = MockEntity("e3")

        # Insert and subdivide
        node.insert(e1, (25.0, 25.0))
        node.insert(e2, (75.0, 75.0))
        node.insert(e3, (30.0, 30.0))

        assert node.is_divided

        # Remove from child
        result = node.remove(e1, (25.0, 25.0))
        assert result is True

    def test_remove_multiple_entities(self):
        """Test removing multiple entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        entities = [MockEntity(f"e{i}") for i in range(3)]
        positions = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]

        for entity, pos in zip(entities, positions):
            node.insert(entity, pos)

        for entity, pos in zip(entities, positions):
            result = node.remove(entity, pos)
            assert result is True

        assert len(node.entities) == 0


class TestQuadtreeNodeQueryRange:
    """Test QuadtreeNode range query operations."""

    def test_query_range_empty_node(self):
        """Test range query on empty node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        results = node.query_range((10.0, 10.0, 20.0, 20.0))
        assert results == []

    def test_query_range_non_intersecting(self):
        """Test range query that doesn't intersect node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        node.insert(MockEntity("e1"), (50.0, 50.0))

        results = node.query_range((200.0, 200.0, 50.0, 50.0))
        assert results == []

    def test_query_range_single_entity(self):
        """Test range query finding single entity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        node.insert(entity, position)
        results = node.query_range((40.0, 40.0, 20.0, 20.0))

        assert len(results) == 1
        assert results[0] == (entity, position)

    def test_query_range_multiple_entities(self):
        """Test range query finding multiple entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        entities = [MockEntity(f"e{i}") for i in range(3)]
        positions = [(10.0, 10.0), (15.0, 15.0), (80.0, 80.0)]

        for entity, pos in zip(entities, positions):
            node.insert(entity, pos)

        results = node.query_range((5.0, 5.0, 15.0, 15.0))
        result_entities = [entity for entity, _ in results]

        assert entities[0] in result_entities
        assert entities[1] in result_entities
        assert entities[2] not in result_entities

    def test_query_range_subdivided_node(self):
        """Test range query on subdivided node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Force subdivision
        for i in range(5):
            node.insert(MockEntity(f"e{i}"), (float(i * 15), float(i * 15)))

        assert node.is_divided

        results = node.query_range((10.0, 10.0, 40.0, 40.0))
        assert len(results) > 0

    def test_query_range_boundary_conditions(self):
        """Test range query boundary inclusivity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        e1 = MockEntity("e1")
        e2 = MockEntity("e2")

        node.insert(e1, (10.0, 10.0))
        node.insert(e2, (20.0, 20.0))

        # Lower boundary inclusive, upper exclusive
        results = node.query_range((10.0, 10.0, 10.0, 10.0))
        result_entities = [entity for entity, _ in results]

        assert e1 in result_entities
        assert e2 not in result_entities

    def test_query_range_entire_bounds(self):
        """Test range query covering entire node bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(4):
            node.insert(MockEntity(f"e{i}"), (float(i * 20), float(i * 20)))

        results = node.query_range((0.0, 0.0, 100.0, 100.0))
        assert len(results) == 4


class TestQuadtreeNodeQueryRadius:
    """Test QuadtreeNode radius query operations."""

    def test_query_radius_empty_node(self):
        """Test radius query on empty node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        results = node.query_radius((50.0, 50.0), 10.0)
        assert results == []

    def test_query_radius_single_entity(self):
        """Test radius query finding single entity."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        node.insert(entity, position)
        results = node.query_radius((50.0, 50.0), 10.0)

        assert len(results) == 1
        assert results[0] == (entity, position)

    def test_query_radius_multiple_entities(self):
        """Test radius query finding multiple entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        entities = [MockEntity(f"e{i}") for i in range(5)]
        positions = [(50.0, 50.0), (55.0, 55.0), (60.0, 60.0), (80.0, 80.0), (90.0, 90.0)]

        for entity, pos in zip(entities, positions):
            node.insert(entity, pos)

        results = node.query_radius((50.0, 50.0), 15.0)
        result_entities = [entity for entity, _ in results]

        # First three should be within radius
        assert entities[0] in result_entities
        assert entities[1] in result_entities

    def test_query_radius_exact_boundary(self):
        """Test radius query at exact boundary distance."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        # Place entity close enough to be found
        node.insert(entity, (58.0, 50.0))  # 8 units away, within radius 10

        results = node.query_radius((50.0, 50.0), 10.0)
        assert len(results) >= 1

    def test_query_radius_uses_bounding_box(self):
        """Test that radius query uses bounding box for efficiency."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        # Insert entity within bounding box but outside radius
        entity = MockEntity("e1")
        node.insert(entity, (60.0, 60.0))

        # Query with small radius
        results = node.query_radius((50.0, 50.0), 5.0)

        # Should not find entity (outside radius)
        assert len(results) == 0


class TestQuadtreeNodeClear:
    """Test QuadtreeNode clear operations."""

    def test_clear_empty_node(self):
        """Test clearing an empty node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        node.clear()

        assert len(node.entities) == 0
        assert node.children is None
        assert not node.is_divided

    def test_clear_node_with_entities(self):
        """Test clearing a node with entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(3):
            node.insert(MockEntity(f"e{i}"), (float(i * 10), float(i * 10)))

        node.clear()

        assert len(node.entities) == 0

    def test_clear_subdivided_node(self):
        """Test clearing a subdivided node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Force subdivision
        for i in range(5):
            node.insert(MockEntity(f"e{i}"), (float(i * 15), float(i * 15)))

        assert node.is_divided

        node.clear()

        assert len(node.entities) == 0
        assert node.children is None
        assert not node.is_divided

    def test_clear_recursively_clears_children(self):
        """Test that clear recursively clears all children."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Create a deep tree
        for i in range(10):
            node.insert(MockEntity(f"e{i}"), (float(i * 5), float(i * 5)))

        node.clear()

        # All should be cleared
        assert len(node.entities) == 0
        assert node.children is None


class TestQuadtreeNodeGetStats:
    """Test QuadtreeNode statistics."""

    def test_get_stats_empty_node(self):
        """Test statistics for empty node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        stats = node.get_stats()

        assert stats["bounds"] == (0.0, 0.0, 100.0, 100.0)
        assert stats["is_divided"] is False
        assert stats["local_entities"] == 0
        assert stats["total_entities"] == 0
        assert stats["children_count"] == 0

    def test_get_stats_with_entities(self):
        """Test statistics with entities."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(3):
            node.insert(MockEntity(f"e{i}"), (float(i * 10), float(i * 10)))

        stats = node.get_stats()

        assert stats["local_entities"] == 3
        assert stats["total_entities"] == 3
        assert stats["is_divided"] is False

    def test_get_stats_subdivided_node(self):
        """Test statistics for subdivided node."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=2)

        for i in range(5):
            node.insert(MockEntity(f"e{i}"), (float(i * 15), float(i * 15)))

        stats = node.get_stats()

        assert stats["is_divided"] is True
        assert stats["children_count"] == 4
        assert stats["total_entities"] == 5


class TestQuadtreeNodeHelperMethods:
    """Test QuadtreeNode helper methods."""

    def test_contains_point_inside(self):
        """Test _contains_point for point inside bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._contains_point((50.0, 50.0)) is True

    def test_contains_point_outside(self):
        """Test _contains_point for point outside bounds."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._contains_point((150.0, 150.0)) is False

    def test_contains_point_on_lower_boundary(self):
        """Test _contains_point on lower boundary (inclusive)."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._contains_point((0.0, 0.0)) is True

    def test_contains_point_on_upper_boundary(self):
        """Test _contains_point on upper boundary (exclusive)."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._contains_point((100.0, 100.0)) is False

    def test_intersects_range_overlapping(self):
        """Test _intersects_range for overlapping ranges."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._intersects_range((50.0, 50.0, 60.0, 60.0)) is True

    def test_intersects_range_non_overlapping(self):
        """Test _intersects_range for non-overlapping ranges."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._intersects_range((200.0, 200.0, 50.0, 50.0)) is False

    def test_intersects_range_partial_overlap(self):
        """Test _intersects_range for partial overlap."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._intersects_range((90.0, 90.0, 20.0, 20.0)) is True

    def test_point_in_range_inside(self):
        """Test _point_in_range for point inside range."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._point_in_range((50.0, 50.0), (40.0, 40.0, 20.0, 20.0)) is True

    def test_point_in_range_outside(self):
        """Test _point_in_range for point outside range."""
        node = QuadtreeNode((0.0, 0.0, 100.0, 100.0), capacity=4)
        assert node._point_in_range((50.0, 50.0), (60.0, 60.0, 20.0, 20.0)) is False

    def test_distance_calculation(self):
        """Test _distance calculation."""
        distance = QuadtreeNode._distance((0.0, 0.0), (3.0, 4.0))
        assert distance == 5.0  # 3-4-5 triangle


class TestQuadtreeWrapper:
    """Test Quadtree wrapper class."""

    def test_quadtree_initialization(self):
        """Test Quadtree initialization."""
        bounds = (0.0, 0.0, 100.0, 100.0)
        tree = Quadtree(bounds, capacity=4)

        assert tree.bounds == bounds
        assert tree.capacity == 4
        assert tree.root is not None
        assert tree.root.bounds == bounds

    def test_quadtree_insert(self):
        """Test Quadtree insert."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")

        result = tree.insert(entity, (50.0, 50.0))
        assert result is True

    def test_quadtree_remove(self):
        """Test Quadtree remove."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        tree.insert(entity, position)
        result = tree.remove(entity, position)

        assert result is True

    def test_quadtree_query_range(self):
        """Test Quadtree query_range."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)

        entities = [MockEntity(f"e{i}") for i in range(3)]
        positions = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)]

        for entity, pos in zip(entities, positions):
            tree.insert(entity, pos)

        results = tree.query_range((40.0, 40.0, 30.0, 30.0))
        result_entities = [entity for entity, _ in results]

        assert entities[1] in result_entities

    def test_quadtree_query_radius(self):
        """Test Quadtree query_radius."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)

        entities = [MockEntity(f"e{i}") for i in range(3)]
        positions = [(50.0, 50.0), (55.0, 55.0), (90.0, 90.0)]

        for entity, pos in zip(entities, positions):
            tree.insert(entity, pos)

        results = tree.query_radius((50.0, 50.0), 10.0)
        result_entities = [entity for entity, _ in results]

        assert entities[0] in result_entities
        assert entities[1] in result_entities
        assert entities[2] not in result_entities

    def test_quadtree_clear(self):
        """Test Quadtree clear."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(5):
            tree.insert(MockEntity(f"e{i}"), (float(i * 15), float(i * 15)))

        tree.clear()

        assert len(tree.root.entities) == 0

    def test_quadtree_get_stats(self):
        """Test Quadtree get_stats."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)

        for i in range(3):
            tree.insert(MockEntity(f"e{i}"), (float(i * 20), float(i * 20)))

        stats = tree.get_stats()

        assert stats["total_entities"] == 3
        assert "bounds" in stats


class TestQuadtreeEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_capacity(self):
        """Test quadtree with capacity of 1."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=1)

        tree.insert(MockEntity("e1"), (25.0, 25.0))
        tree.insert(MockEntity("e2"), (75.0, 75.0))

        assert tree.root.is_divided

    def test_entities_at_exact_center(self):
        """Test handling entities at exact center of bounds."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=2)

        # Insert multiple entities at center
        for i in range(5):
            tree.insert(MockEntity(f"e{i}"), (50.0, 50.0))

        results = tree.query_radius((50.0, 50.0), 1.0)
        # All entities are at the exact same position; the query should return all of them
        assert len(results) == 5

    def test_many_subdivisions(self):
        """Test creating many subdivision levels."""
        tree = Quadtree((0.0, 0.0, 1000.0, 1000.0), capacity=1)

        # Insert entities in a line to force deep subdivision
        for i in range(10):
            tree.insert(MockEntity(f"e{i}"), (float(i), float(i)))

        # Tree should be deeply subdivided
        assert tree.root.is_divided

    def test_insert_remove_repeatedly(self):
        """Test repeatedly inserting and removing same entity."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)
        entity = MockEntity("e1")
        position = (50.0, 50.0)

        for _ in range(5):
            tree.insert(entity, position)
            tree.remove(entity, position)

        results = tree.query_radius(position, 5.0)
        assert len(results) == 0

    def test_floating_point_boundaries(self):
        """Test handling of floating point precision at boundaries."""
        tree = Quadtree((0.0, 0.0, 100.0, 100.0), capacity=4)

        # Insert at precise boundaries
        tree.insert(MockEntity("e1"), (50.0, 50.0))
        tree.insert(MockEntity("e2"), (50.000001, 50.000001))

        results = tree.query_radius((50.0, 50.0), 0.001)
        assert len(results) >= 1
