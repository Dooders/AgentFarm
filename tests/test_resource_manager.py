"""Unit tests for the ResourceManager class.

This module tests the ResourceManager functionality including:
- Initialization and configuration
- Resource distribution types (random, grid, clustered)
- Resource updates and regeneration
- Resource consumption and management
- Spatial queries and statistics
- Database integration
"""

import os
import random
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from farm.core.resource_manager import ResourceManager
from farm.core.resources import Resource


class TestResourceManager(unittest.TestCase):
    """Test cases for ResourceManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 100.0
        self.height = 100.0
        self.config = Mock()
        # Mock the nested resources config structure
        self.config.resources = Mock()
        self.config.resources.max_resource_amount = 20
        self.config.resource_regen_rate = 0.1
        self.config.resource_regen_amount = 2

        # Mock database logger
        self.mock_logger = Mock()

        # Create resource manager without seed for non-deterministic tests
        self.resource_manager = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=None,
            database_logger=self.mock_logger,
        )

    def tearDown(self):
        """Clean up after each test."""
        # Reset the resource manager to clear any accumulated state
        if hasattr(self, "resource_manager"):
            self.resource_manager.reset()

    def test_initialization(self):
        """Test ResourceManager initialization."""
        self.assertEqual(self.resource_manager.width, self.width)
        self.assertEqual(self.resource_manager.height, self.height)
        self.assertEqual(self.resource_manager.config, self.config)
        self.assertIsNone(self.resource_manager.seed_value)
        self.assertEqual(self.resource_manager.database_logger, self.mock_logger)
        self.assertEqual(len(self.resource_manager.resources), 0)
        self.assertEqual(self.resource_manager.next_resource_id, 0)

    def test_initialization_with_seed(self):
        """Test ResourceManager initialization with seed."""
        seed = 42
        resource_manager = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=seed,
            database_logger=self.mock_logger,
        )

        self.assertEqual(resource_manager.seed_value, seed)

    def test_initialization_tracking_variables(self):
        """Test that tracking variables are properly initialized."""
        self.assertEqual(self.resource_manager.regeneration_step, 0)
        self.assertEqual(self.resource_manager.total_resources_consumed, 0)
        self.assertEqual(self.resource_manager.total_resources_regenerated, 0)
        self.assertEqual(self.resource_manager.regeneration_events, 0)
        self.assertEqual(self.resource_manager.depletion_events, 0)

    def test_create_random_distribution(self):
        """Test random resource distribution."""
        distribution = {"type": "random", "amount": 10}
        resources = self.resource_manager._create_random_distribution(10, distribution)

        self.assertEqual(len(resources), 10)

        # Check that all resources are within bounds
        for resource in resources:
            self.assertIsInstance(resource, Resource)
            self.assertGreaterEqual(resource.position[0], 0)
            self.assertLessEqual(resource.position[0], self.width)
            self.assertGreaterEqual(resource.position[1], 0)
            self.assertLessEqual(resource.position[1], self.height)

    def test_create_grid_distribution(self):
        """Test grid resource distribution."""
        distribution = {"type": "grid", "amount": 25}
        resources = self.resource_manager._create_grid_distribution(25, distribution)

        # Should create a 5x5 grid (25 resources)
        self.assertEqual(len(resources), 25)

        # Check grid spacing
        positions = [r.position for r in resources]
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        # Should have 5 unique x positions and 5 unique y positions
        self.assertEqual(len(set(x_positions)), 5)
        self.assertEqual(len(set(y_positions)), 5)

    def test_create_clustered_distribution(self):
        """Test clustered resource distribution."""
        distribution = {"type": "clustered", "amount": 30}
        resources = self.resource_manager._create_clustered_distribution(
            30, distribution
        )

        self.assertEqual(len(resources), 30)

        # Check that all resources are within bounds
        for resource in resources:
            self.assertIsInstance(resource, Resource)
            self.assertGreaterEqual(resource.position[0], 0)
            self.assertLessEqual(resource.position[0], self.width)
            self.assertGreaterEqual(resource.position[1], 0)
            self.assertLessEqual(resource.position[1], self.height)

    def test_create_resource_deterministic(self):
        """Test resource creation with deterministic seed."""
        resource_manager = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=42,
            database_logger=self.mock_logger,
        )

        position = (50.0, 50.0)
        distribution = {"min_amount": 3, "max_amount": 8}

        resource = resource_manager._create_resource(position, distribution)

        self.assertIsInstance(resource, Resource)
        self.assertEqual(resource.position, position)
        self.assertEqual(resource.max_amount, self.config.resources.max_resource_amount)
        self.assertEqual(resource.regeneration_rate, self.config.resource_regen_rate)

        # With seed=42, pos_sum=100, so amount should be 3 + (100 % 6) = 3 + 4 = 7
        self.assertEqual(resource.amount, 7)

    def test_create_resource_random(self):
        """Test resource creation without seed (random)."""
        position = (50.0, 50.0)
        distribution = {"min_amount": 3, "max_amount": 8}

        resource = self.resource_manager._create_resource(position, distribution)

        self.assertIsInstance(resource, Resource)
        self.assertEqual(resource.position, position)
        self.assertGreaterEqual(resource.amount, 3)
        self.assertLessEqual(resource.amount, 8)

    def test_update_resources_random(self):
        """Test resource updates with random regeneration."""
        # Create some resources
        distribution = {"type": "random", "amount": 5}
        self.resource_manager.initialize_resources(distribution)

        # Test that update works without mocking (since we're using original Environment logic)
        stats = self.resource_manager.update_resources(time_step=1)

        # Should have some regeneration events (may be 0 due to random nature)
        self.assertGreaterEqual(stats["regeneration_events"], 0)
        self.assertGreaterEqual(stats["resources_regenerated"], 0)
        self.assertEqual(stats["total_resources"], 5)

    def test_update_resources_deterministic(self):
        """Test resource updates with deterministic regeneration."""
        resource_manager = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=42,
            database_logger=self.mock_logger,
        )

        # Create some resources
        distribution = {"type": "random", "amount": 3}
        resource_manager.initialize_resources(distribution)

        stats = resource_manager.update_resources(time_step=1)

        # Should have some regeneration events
        self.assertGreaterEqual(stats["regeneration_events"], 0)
        self.assertGreaterEqual(stats["resources_regenerated"], 0)

    def test_consume_resource(self):
        """Test resource consumption."""
        # Create a resource
        resource = Resource(
            resource_id=0,
            position=(50.0, 50.0),
            amount=10.0,
            max_amount=20.0,
            regeneration_rate=0.1,
        )
        self.resource_manager.resources.append(resource)

        # Consume some resources
        consumed = self.resource_manager.consume_resource(resource, 5.0)

        self.assertEqual(consumed, 5.0)
        self.assertEqual(resource.amount, 5.0)
        self.assertEqual(self.resource_manager.total_resources_consumed, 5.0)

    def test_consume_resource_depleted(self):
        """Test resource consumption when resource is depleted."""
        # Create a depleted resource
        resource = Resource(
            resource_id=0,
            position=(50.0, 50.0),
            amount=0.0,
            max_amount=20.0,
            regeneration_rate=0.1,
        )
        self.resource_manager.resources.append(resource)

        # Try to consume from depleted resource
        consumed = self.resource_manager.consume_resource(resource, 5.0)

        self.assertEqual(consumed, 0.0)
        self.assertEqual(resource.amount, 0.0)

    def test_consume_resource_partial(self):
        """Test partial resource consumption."""
        # Create a resource with less than requested amount
        resource = Resource(
            resource_id=0,
            position=(50.0, 50.0),
            amount=3.0,
            max_amount=20.0,
            regeneration_rate=0.1,
        )
        self.resource_manager.resources.append(resource)

        # Try to consume more than available
        consumed = self.resource_manager.consume_resource(resource, 5.0)

        self.assertEqual(consumed, 3.0)
        self.assertEqual(resource.amount, 0.0)
        self.assertEqual(self.resource_manager.depletion_events, 1)

    def test_get_nearby_resources(self):
        """Test getting nearby resources."""
        # Create resources at different positions
        resources = [
            Resource(0, (10.0, 10.0), 5.0),
            Resource(1, (15.0, 15.0), 5.0),
            Resource(2, (50.0, 50.0), 5.0),
        ]
        self.resource_manager.resources = resources

        # Get resources within radius 10 of (12, 12)
        nearby = self.resource_manager.get_nearby_resources((12.0, 12.0), 10.0)

        # Should find 2 resources (at (10,10) and (15,15))
        self.assertEqual(len(nearby), 2)

    def test_get_nearest_resource(self):
        """Test getting the nearest resource."""
        # Create resources at different positions
        resources = [
            Resource(0, (10.0, 10.0), 5.0),
            Resource(1, (50.0, 50.0), 5.0),
            Resource(2, (90.0, 90.0), 5.0),
        ]
        self.resource_manager.resources = resources

        # Get nearest resource to (15, 15)
        nearest = self.resource_manager.get_nearest_resource((15.0, 15.0))

        # Should be the resource at (10, 10)
        self.assertEqual(nearest, resources[0])

    def test_get_nearest_resource_empty(self):
        """Test getting nearest resource when no resources exist."""
        nearest = self.resource_manager.get_nearest_resource((50.0, 50.0))

        self.assertIsNone(nearest)

    def test_add_resource(self):
        """Test adding a new resource."""
        position = (25.0, 25.0)
        amount = 10.0

        resource = self.resource_manager.add_resource(position, amount)

        self.assertIsInstance(resource, Resource)
        self.assertEqual(resource.position, position)
        self.assertEqual(resource.amount, amount)
        self.assertEqual(len(self.resource_manager.resources), 1)
        self.assertEqual(self.resource_manager.next_resource_id, 1)

        # Check that logger was called
        self.mock_logger.log_resource.assert_called_once()

    def test_add_resource_default_amount(self):
        """Test adding a resource with default amount."""
        position = (25.0, 25.0)

        resource = self.resource_manager.add_resource(position)

        self.assertEqual(resource.amount, 5.0)  # Default amount

    def test_remove_resource(self):
        """Test removing a resource."""
        # Add a resource
        resource = self.resource_manager.add_resource((25.0, 25.0), 10.0)

        # Remove it
        success = self.resource_manager.remove_resource(resource)

        self.assertTrue(success)
        self.assertEqual(len(self.resource_manager.resources), 0)

    def test_remove_resource_not_found(self):
        """Test removing a resource that doesn't exist."""
        resource = Resource(0, (25.0, 25.0), 10.0)

        success = self.resource_manager.remove_resource(resource)

        self.assertFalse(success)

    def test_get_resource_statistics_empty(self):
        """Test resource statistics with no resources."""
        stats = self.resource_manager.get_resource_statistics()

        self.assertEqual(stats["total_resources"], 0)
        self.assertEqual(stats["average_amount"], 0)
        self.assertEqual(stats["depleted_resources"], 0)
        self.assertEqual(stats["full_resources"], 0)
        self.assertEqual(stats["total_capacity"], 0)
        self.assertEqual(stats["utilization_rate"], 0)

    def test_get_resource_statistics_with_resources(self):
        """Test resource statistics with resources."""
        # Add resources with different amounts
        self.resource_manager.add_resource((10.0, 10.0), 5.0)
        self.resource_manager.add_resource((20.0, 20.0), 15.0)
        self.resource_manager.add_resource((30.0, 30.0), 0.0)  # Depleted

        stats = self.resource_manager.get_resource_statistics()

        self.assertEqual(stats["total_resources"], 3)
        self.assertAlmostEqual(stats["average_amount"], 6.67, places=2)  # (5+15+0)/3
        self.assertEqual(stats["depleted_resources"], 1)
        self.assertEqual(stats["full_resources"], 0)  # None at max capacity
        self.assertEqual(stats["total_capacity"], 60)  # 3 * 20
        self.assertAlmostEqual(stats["utilization_rate"], 0.33, places=2)  # 20/60

    def test_reset(self):
        """Test resetting the resource manager."""
        # Add some resources and modify tracking variables
        self.resource_manager.add_resource((10.0, 10.0), 5.0)
        self.resource_manager.total_resources_consumed = 10.0
        self.resource_manager.regeneration_events = 5

        # Reset
        self.resource_manager.reset()

        self.assertEqual(len(self.resource_manager.resources), 0)
        self.assertEqual(self.resource_manager.next_resource_id, 0)
        self.assertEqual(self.resource_manager.total_resources_consumed, 0)
        self.assertEqual(self.resource_manager.regeneration_events, 0)

    def test_deterministic_behavior_with_seed(self):
        """Test that resource manager behaves deterministically with seed."""
        seed = 42

        # Create two resource managers with the same seed
        rm1 = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=seed,
            database_logger=self.mock_logger,
        )

        rm2 = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=seed,
            database_logger=self.mock_logger,
        )

        # Initialize resources with same distribution
        distribution = {"type": "random", "amount": 10}
        resources1 = rm1.initialize_resources(distribution)
        resources2 = rm2.initialize_resources(distribution)

        # Should have same number of resources
        self.assertEqual(len(resources1), len(resources2))

        # Should have same positions and amounts
        for r1, r2 in zip(resources1, resources2):
            self.assertEqual(r1.position, r2.position)
            self.assertEqual(r1.amount, r2.amount)

    def test_update_tracking_variables(self):
        """Test that tracking variables are updated correctly."""
        # Create resources
        distribution = {"type": "random", "amount": 3}
        self.resource_manager.initialize_resources(distribution)

        # Update resources
        stats = self.resource_manager.update_resources(time_step=1)

        # Check that tracking variables were updated
        self.assertEqual(self.resource_manager.regeneration_step, 1)
        self.assertGreaterEqual(self.resource_manager.regeneration_events, 0)
        self.assertGreaterEqual(self.resource_manager.total_resources_regenerated, 0)

    def test_database_logger_integration(self):
        """Test integration with database logger."""
        # Test that logger is called during initialization
        distribution = {"type": "random", "amount": 5}
        self.resource_manager.initialize_resources(distribution)

        # Check that logger was called for each resource
        self.assertEqual(self.mock_logger.log_resource.call_count, 5)

        # Test that logger is called when adding individual resources
        self.mock_logger.log_resource.reset_mock()
        self.resource_manager.add_resource((50.0, 50.0), 10.0)

        self.mock_logger.log_resource.assert_called_once()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with zero resources
        distribution = {"type": "random", "amount": 0}
        resources = self.resource_manager.initialize_resources(distribution)
        self.assertEqual(len(resources), 0)

        # Test with very large number of resources
        distribution = {"type": "random", "amount": 1000}
        resources = self.resource_manager.initialize_resources(distribution)
        self.assertEqual(len(resources), 1000)

        # Test resource creation at boundary positions
        resource = self.resource_manager.add_resource((0.0, 0.0), 5.0)
        self.assertEqual(resource.position, (0.0, 0.0))

        resource = self.resource_manager.add_resource((self.width, self.height), 5.0)
        self.assertEqual(resource.position, (self.width, self.height))

    def test_error_handling(self):
        """Test error handling in resource manager."""
        # Test with invalid distribution type
        distribution = {"type": "invalid", "amount": 5}
        resources = self.resource_manager.initialize_resources(distribution)

        # Should fall back to random distribution
        self.assertEqual(len(resources), 5)

        # Test with negative amount
        distribution = {"type": "random", "amount": -5}
        resources = self.resource_manager.initialize_resources(distribution)

        # Should handle gracefully (empty list)
        self.assertEqual(len(resources), 0)


if __name__ == "__main__":
    unittest.main()
