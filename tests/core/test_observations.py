"""Comprehensive tests for observations module.

Tests AgentObservation, SparsePoints, channel operations, and utility functions.
"""

import unittest

import pytest
import torch

from farm.core.observations import (
    AgentObservation,
    ObservationConfig,
    SparsePoints,
    create_observation_tensor,
    crop_local,
    make_disk_mask,
)


class TestSparsePoints(unittest.TestCase):
    """Test SparsePoints class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.dtype = torch.float32

    def test_initialization(self):
        """Test SparsePoints initialization."""
        sparse = SparsePoints(self.device, self.dtype)

        self.assertEqual(sparse.device, self.device)
        self.assertEqual(sparse.dtype, self.dtype)
        self.assertEqual(len(sparse), 0)

    def test_add_point(self):
        """Test adding a single point."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_point(5, 10, 1.0)

        self.assertEqual(len(sparse), 1)

    def test_add_points(self):
        """Test adding multiple points."""
        sparse = SparsePoints(self.device, self.dtype)
        points = [(1, 2, 0.5), (3, 4, 0.7), (5, 6, 0.9)]
        sparse.add_points(points)

        self.assertEqual(len(sparse), 3)

    def test_add_points_empty(self):
        """Test adding empty points list."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_points([])

        self.assertEqual(len(sparse), 0)

    def test_decay(self):
        """Test decay operation."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_point(5, 10, 1.0)
        sparse.decay(0.9)

        self.assertAlmostEqual(sparse.values[0].item(), 0.9, places=5)

    def test_decay_with_pruning(self):
        """Test decay with pruning."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_point(5, 10, 0.01)
        sparse.decay(0.1, prune_eps=0.05)

        # Value should be pruned (0.01 * 0.1 = 0.001 < 0.05)
        self.assertEqual(len(sparse), 0)

    def test_apply_to_dense_max(self):
        """Test applying sparse points to dense tensor with max reduction."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_points([(2, 3, 0.5), (2, 3, 0.8)])  # Duplicate indices

        dense = torch.zeros(13, 13, dtype=self.dtype, device=self.device)
        sparse.apply_to_dense(dense, reduction="max")

        self.assertAlmostEqual(dense[2, 3].item(), 0.8, places=5)  # Max value

    def test_apply_to_dense_sum(self):
        """Test applying sparse points with sum reduction."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_points([(2, 3, 0.5), (2, 3, 0.3)])

        dense = torch.zeros(13, 13, dtype=self.dtype, device=self.device)
        sparse.apply_to_dense(dense, reduction="sum")

        self.assertAlmostEqual(dense[2, 3].item(), 0.8, places=5)

    def test_apply_to_dense_out_of_bounds(self):
        """Test applying points with out-of-bounds indices."""
        sparse = SparsePoints(self.device, self.dtype)
        sparse.add_points([(20, 20, 1.0), (5, 5, 0.5)])  # One out of bounds

        dense = torch.zeros(13, 13, dtype=self.dtype, device=self.device)
        sparse.apply_to_dense(dense, reduction="max")

        self.assertEqual(dense[5, 5].item(), 0.5)
        # Out of bounds index (20, 20) should be filtered and not accessible
        # Verify that only valid indices were written
        self.assertEqual(dense.sum().item(), 0.5)

    def test_apply_to_dense_empty(self):
        """Test applying empty sparse points."""
        sparse = SparsePoints(self.device, self.dtype)
        dense = torch.zeros(13, 13, dtype=self.dtype, device=self.device)

        sparse.apply_to_dense(dense, reduction="max")

        # Should not raise and dense should remain zeros
        self.assertTrue(torch.all(dense == 0))


class TestCreateObservationTensor(unittest.TestCase):
    """Test create_observation_tensor factory function."""

    def test_create_zeros(self):
        """Test creating zeros tensor."""
        tensor = create_observation_tensor(13, 13)

        self.assertEqual(tensor.shape, (13, 13, 13))
        self.assertTrue(torch.all(tensor == 0))

    def test_create_random(self):
        """Test creating random tensor."""
        tensor = create_observation_tensor(13, 13, initialization="random")

        self.assertEqual(tensor.shape, (13, 13, 13))
        self.assertFalse(torch.all(tensor == 0))

    def test_create_random_custom_range(self):
        """Test creating random tensor with custom range."""
        tensor = create_observation_tensor(
            13, 13, initialization="random", random_min=-0.1, random_max=0.1
        )

        self.assertEqual(tensor.shape, (13, 13, 13))
        self.assertTrue(torch.all(tensor >= -0.1))
        self.assertTrue(torch.all(tensor < 0.1))

    def test_create_invalid_initialization(self):
        """Test invalid initialization method."""
        with self.assertRaises(ValueError):
            create_observation_tensor(13, 13, initialization="invalid")


class TestObservationConfig(unittest.TestCase):
    """Test ObservationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservationConfig()

        self.assertEqual(config.R, 6)
        self.assertEqual(config.gamma_trail, 0.90)
        self.assertEqual(config.gamma_dmg, 0.85)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ObservationConfig(R=10, gamma_trail=0.95)

        self.assertEqual(config.R, 10)
        self.assertEqual(config.gamma_trail, 0.95)

    def test_validation(self):
        """Test configuration validation."""
        with self.assertRaises(Exception):  # Pydantic validation error
            ObservationConfig(R=-1)  # R must be > 0


class TestAgentObservation(unittest.TestCase):
    """Test AgentObservation class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ObservationConfig(R=6)
        self.agent_obs = AgentObservation(self.config)

    def test_initialization(self):
        """Test AgentObservation initialization."""
        self.assertIsNotNone(self.agent_obs)
        self.assertEqual(self.agent_obs.config.R, 6)

    def test_perceive_world(self):
        """Test perceive_world updates observation."""
        # Create mock world layers
        world_layers = {
            "resources": torch.zeros(100, 100),
            "obstacles": torch.zeros(100, 100),
        }

        self.agent_obs.perceive_world(
            world_layers, agent_world_pos=(50, 50), self_hp01=0.8
        )

        # Should not raise
        self.assertIsNotNone(self.agent_obs)

    def test_tensor(self):
        """Test tensor generation."""
        tensor = self.agent_obs.tensor()

        self.assertIsInstance(tensor, torch.Tensor)
        # Shape should be (num_channels, 2R+1, 2R+1)
        expected_size = 2 * self.config.R + 1
        self.assertEqual(tensor.shape[1], expected_size)
        self.assertEqual(tensor.shape[2], expected_size)

    def test_get_metrics(self):
        """Test metrics collection."""
        metrics = self.agent_obs.get_metrics()

        self.assertIsInstance(metrics, dict)
        self.assertIn("dense_bytes", metrics)
        self.assertIn("sparse_points", metrics)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_crop_local(self):
        """Test crop_local function."""
        # Create a large tensor
        large_tensor = torch.randn(100, 100)

        # Crop around center (note: crop_local uses R parameter, not radius)
        cropped = crop_local(large_tensor, center=(50, 50), R=5)

        self.assertEqual(cropped.shape, (11, 11))  # 2*5+1

    def test_crop_local_edge_case(self):
        """Test crop_local at edge of tensor."""
        large_tensor = torch.randn(100, 100)

        # Crop near edge (note: crop_local uses R parameter, not radius)
        cropped = crop_local(large_tensor, center=(5, 5), R=5)

        # Should handle edge case gracefully
        self.assertIsNotNone(cropped)

    def test_make_disk_mask(self):
        """Test make_disk_mask creates correct mask."""
        # Note: make_disk_mask signature is make_disk_mask(size, R, ...)
        mask = make_disk_mask(size=13, R=5)

        self.assertEqual(mask.shape, (13, 13))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask <= 1))

        # Center should be 1
        self.assertEqual(mask[6, 6].item(), 1.0)

    def test_make_disk_mask_radius_zero(self):
        """Test make_disk_mask with zero radius."""
        # Note: make_disk_mask signature is make_disk_mask(size, R, ...)
        mask = make_disk_mask(size=13, R=0)

        # Only center should be 1
        self.assertEqual(mask[6, 6].item(), 1.0)
        # Others should be 0
        self.assertEqual(mask[0, 0].item(), 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestObservationsGPU(unittest.TestCase):
    """Test observations with GPU tensors."""

    def test_sparse_points_gpu(self):
        """Test SparsePoints on GPU."""
        device = "cuda"
        sparse = SparsePoints(device, torch.float32)
        sparse.add_point(5, 10, 1.0)

        self.assertEqual(sparse.device, device)
        self.assertEqual(len(sparse), 1)

    def test_apply_to_dense_gpu(self):
        """Test applying sparse points to GPU tensor."""
        device = "cuda"
        sparse = SparsePoints(device, torch.float32)
        sparse.add_point(5, 10, 1.0)

        dense = torch.zeros(13, 13, dtype=torch.float32, device=device)
        sparse.apply_to_dense(dense, reduction="max")

        self.assertEqual(dense[5, 10].item(), 1.0)


if __name__ == "__main__":
    unittest.main()

