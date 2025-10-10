"""
Unit tests for the observations module.

This module tests the agent observation system including:
- Channel enumeration
- ObservationConfig validation
- Utility functions (crop_local, crop_local_stack, make_disk_mask) for local cropping
- AgentObservation class and all its methods
"""

from typing import Dict, List, Tuple
import unittest

import numpy as np
import pytest
import torch

from farm.core.channels import NUM_CHANNELS, Channel
from farm.core.observations import (
    AgentObservation,
    ObservationConfig,
    SparsePoints,
    crop_local,
    crop_local_stack,
    make_disk_mask,
    rotate_coordinates,
    crop_local_rotated,
)


class TestChannel:
    """Test the Channel enumeration."""

    def test_channel_values(self):
        """Test that channels have correct integer values."""
        assert Channel.SELF_HP == 0
        assert Channel.ALLIES_HP == 1
        assert Channel.ENEMIES_HP == 2
        assert Channel.RESOURCES == 3
        assert Channel.OBSTACLES == 4

    def test_num_channels(self):
        """Test that NUM_CHANNELS is correctly calculated."""
        assert NUM_CHANNELS == 13  # 13 channels defined


class TestObservationConfig:
    """Test the ObservationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservationConfig()
        assert config.R == 6
        assert config.gamma_trail == 0.90
        assert config.gamma_dmg == 0.85
        assert config.gamma_sig == 0.92
        assert config.gamma_known == 0.98
        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.fov_radius == 6

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ObservationConfig(
            R=10,
            gamma_trail=0.95,
            gamma_dmg=0.80,
            gamma_sig=0.90,
            gamma_known=0.99,
            device="cuda",
            dtype="float64",
            fov_radius=8,
        )
        assert config.R == 10
        assert config.gamma_trail == 0.95
        assert config.gamma_dmg == 0.80
        assert config.gamma_sig == 0.90
        assert config.gamma_known == 0.99
        assert config.device == "cuda"
        assert config.dtype == torch.float64
        assert config.fov_radius == 8

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Test R > 0
        with pytest.raises(ValueError):
            ObservationConfig(R=0)

        with pytest.raises(ValueError):
            ObservationConfig(R=-1)

        # Test gamma values in [0, 1]
        with pytest.raises(ValueError):
            ObservationConfig(gamma_trail=1.1)

        with pytest.raises(ValueError):
            ObservationConfig(gamma_dmg=-0.1)

        # Test fov_radius > 0
        with pytest.raises(ValueError):
            ObservationConfig(fov_radius=0)

    def test_torch_dtype_property(self):
        """Test the torch_dtype property."""
        config = ObservationConfig(dtype="float32")
        assert config.torch_dtype == torch.float32

        config = ObservationConfig(dtype="float64")
        assert config.torch_dtype == torch.float64

    def test_validate_dtype(self):
        """Test dtype validation through field validation."""
        # Test that string dtypes are converted to torch dtypes
        config = ObservationConfig(dtype="float32")
        assert config.dtype == torch.float32

        config = ObservationConfig(dtype="float64")
        assert config.dtype == torch.float64


class TestCropLocal:
    """Test the crop_local function."""

    def test_basic_crop(self):
        """Test basic cropping functionality."""
        # Create a 10x10 grid with known values
        grid = torch.zeros(10, 10)
        grid[3:7, 3:7] = 1.0  # Create a 4x4 square of 1s

        # Crop around center (5, 5) with radius 2
        crop = crop_local(grid, center=(5, 5), R=2)

        assert crop.shape == (5, 5)  # 2*2 + 1 = 5
        assert crop[2, 2] == 1.0  # Center should be 1
        # The 4x4 square of 1s should be visible in the crop
        assert (
            crop[0, 0] == 1.0
        )  # Top-left corner of the 4x4 square (maps to world[3,3])
        assert (
            crop[3, 3] == 1.0
        )  # Bottom-right corner of the 4x4 square (maps to world[6,6])
        assert crop[4, 4] == 0.0  # Outside the 4x4 square (maps to world[7,7])

    def test_crop_at_edge(self):
        """Test cropping when center is near grid edge."""
        grid = torch.ones(10, 10)

        # Crop at edge (0, 0) with radius 3
        crop = crop_local(grid, center=(0, 0), R=3)

        assert crop.shape == (7, 7)  # 2*3 + 1 = 7
        # Should be padded with zeros for out-of-bounds areas
        assert crop[3, 3] == 1.0  # Center of crop should be original (0,0) position
        assert crop[0, 0] == 0.0  # Top-left should be padded (maps to world[-3,-3])
        assert crop[6, 6] == 1.0  # Bottom-right maps to world[3,3] which contains 1.0

    def test_crop_at_corner(self):
        """Test cropping when center is at grid corner."""
        grid = torch.ones(5, 5)

        # Crop at corner (4, 4) with radius 2
        crop = crop_local(grid, center=(4, 4), R=2)

        assert crop.shape == (5, 5)
        assert crop[2, 2] == 1.0  # Center of crop should be original corner
        assert crop[0, 0] == 1.0  # Maps to world[2,2] which contains 1.0
        assert crop[4, 4] == 0.0  # Maps to world[6,6] which is out of bounds, so padded

    def test_custom_pad_value(self):
        """Test cropping with custom padding value."""
        grid = torch.zeros(10, 10)
        grid[5, 5] = 1.0

        # Crop with custom pad value
        crop = crop_local(grid, center=(0, 0), R=3, pad_val=0.5)

        assert crop.shape == (7, 7)
        # Check that padded areas have custom value
        assert (
            crop[0, 0] == 0.5
        )  # Top-left corner should be padded (maps to world[-3,-3])
        assert (
            crop[6, 6] == 0.0
        )  # Bottom-right corner maps to world[3,3] which contains 0.0

    def test_large_radius(self):
        """Test cropping with radius larger than grid."""
        grid = torch.ones(3, 3)

        # Crop with radius larger than grid
        crop = crop_local(grid, center=(1, 1), R=5)

        assert crop.shape == (11, 11)  # 2*5 + 1 = 11
        # Most should be padded, only center 3x3 should be original values
        assert crop[5, 5] == 1.0  # Center of original grid
        assert crop[0, 0] == 0.0  # Padded corner


class TestCropLocalStack:
    """Test the crop_local_stack function."""

    def test_multi_channel_crop(self):
        """Test cropping multi-channel tensor."""
        # Create 3-channel 10x10 grid
        gridC = torch.zeros(3, 10, 10)
        gridC[0, 3:7, 3:7] = 1.0  # Channel 0: 4x4 square
        gridC[1, 4:6, 4:6] = 2.0  # Channel 1: 2x2 square
        gridC[2, 5, 5] = 3.0  # Channel 2: single pixel

        # Crop around center (5, 5) with radius 2
        crop = crop_local_stack(gridC, center=(5, 5), R=2)

        assert crop.shape == (3, 5, 5)  # 3 channels, 5x5 spatial
        assert crop[0, 2, 2] == 1.0  # Channel 0 center
        assert crop[1, 2, 2] == 2.0  # Channel 1 center
        assert crop[2, 2, 2] == 3.0  # Channel 2 center

    def test_edge_cropping_multi_channel(self):
        """Test multi-channel cropping at edge."""
        gridC = torch.ones(2, 5, 5)

        # Crop at edge
        crop = crop_local_stack(gridC, center=(0, 0), R=2)

        assert crop.shape == (2, 5, 5)
        assert crop[0, 2, 2] == 1.0  # Center should be original value
        assert crop[0, 0, 0] == 0.0  # Corner should be padded
        assert crop[1, 2, 2] == 1.0  # Same for second channel


class TestMakeDiskMask:
    """Test the make_disk_mask function."""

    def test_basic_disk_mask(self):
        """Test basic disk mask creation."""
        mask = make_disk_mask(size=7, R=3)

        assert mask.shape == (7, 7)
        assert mask[3, 3] == 1.0  # Center should be 1
        assert mask[0, 0] == 0.0  # Corner should be 0
        # Check that radius 3 includes pixels within distance 3
        assert mask[3, 6] == 1.0  # Distance 3 from center (3,3) to (3,6) is 3
        assert mask[3, 2] == 1.0  # Inside radius should be 1

    def test_small_radius(self):
        """Test disk mask with small radius."""
        mask = make_disk_mask(size=5, R=1)

        assert mask.shape == (5, 5)
        assert mask[2, 2] == 1.0  # Center
        assert mask[1, 2] == 1.0  # Adjacent (distance 1)
        assert mask[0, 2] == 0.0  # Outside radius (distance 2)

    def test_custom_device_and_dtype(self):
        """Test disk mask with custom device and dtype."""
        if torch.cuda.is_available():
            mask = make_disk_mask(size=5, R=2, device="cuda", dtype=torch.float64)
            assert mask.device.type == "cuda"
            assert mask.dtype == torch.float64

    def test_radius_at_boundary(self):
        """Test disk mask with radius at grid boundary."""
        mask = make_disk_mask(size=7, R=3)

        # Check that radius 3 includes exactly the right pixels
        # Center at (3, 3), radius 3 should include pixels within distance 3
        assert mask[3, 3] == 1.0  # Center
        assert mask[3, 6] == 1.0  # Distance 3 from center (inclusive)
        assert mask[3, 5] == 1.0  # Distance 2 from center


class TestRotationHelpers:
    """Tests for rotation math and rotated cropping."""

    def test_rotate_coordinates(self):
        cy, cx = 10.0, 10.0
        # Point directly above center (north)
        y, x = 9.0, 10.0
        # Rotate by 90 degrees clockwise -> should move to east of center
        ry, rx = rotate_coordinates(y, x, 90.0, cy, cx)
        assert pytest.approx(ry, rel=1e-5, abs=1e-5) == 10.0
        assert pytest.approx(rx, rel=1e-5, abs=1e-5) == 11.0

    def test_crop_local_rotated_basic(self):
        # World grid 9x9 with a single value east of center
        grid = torch.zeros(9, 9)
        cy = cx = 4
        grid[cy, cx + 1] = 1.0
        # Crop with R=2 at center, orientation=90 (facing east)
        crop = crop_local_rotated(grid, center=(cy, cx), R=2, orientation=90.0)
        # In rotated view, east should appear "up" at (R-1, R)
        assert crop.shape == (5, 5)
        assert crop[1, 2] == pytest.approx(1.0, abs=1e-5)


class TestAgentObservation:
    """Test the AgentObservation class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ObservationConfig(R=3, device="cpu", dtype="float32")

    @pytest.fixture
    def agent_obs(self, config):
        """Create an AgentObservation instance."""
        return AgentObservation(config)

    def test_initialization(self, agent_obs, config):
        """Test AgentObservation initialization."""
        assert agent_obs.config == config
        assert agent_obs.tensor().shape == (NUM_CHANNELS, 7, 7)  # 2*3 + 1 = 7
        assert agent_obs.tensor().device.type == "cpu"
        assert agent_obs.tensor().dtype == torch.float32
        assert torch.all(agent_obs.tensor() == 0.0)

    def test_decay_dynamics(self, agent_obs):
        """Test decay dynamics application."""
        # Set some initial values
        agent_obs.tensor()[Channel.TRAILS] = 1.0
        agent_obs.tensor()[Channel.DAMAGE_HEAT] = 1.0
        agent_obs.tensor()[Channel.ALLY_SIGNAL] = 1.0
        agent_obs.tensor()[Channel.KNOWN_EMPTY] = 1.0

        # Apply decay
        agent_obs.decay_dynamics()

        # Check that values have been decayed
        assert torch.all(agent_obs.tensor()[Channel.TRAILS] == 0.90)
        assert torch.all(agent_obs.tensor()[Channel.DAMAGE_HEAT] == 0.85)
        assert torch.all(agent_obs.tensor()[Channel.ALLY_SIGNAL] == 0.92)
        assert torch.all(agent_obs.tensor()[Channel.KNOWN_EMPTY] == 0.98)

    def test_clear_instant(self, agent_obs):
        """Test clearing of instantaneous channels."""
        # Set some values in all channels
        agent_obs.tensor().fill_(1.0)

        # Clear instant channels
        agent_obs.clear_instant()

        # Check that instant channels are cleared
        assert torch.all(agent_obs.tensor()[Channel.SELF_HP] == 0.0)
        assert torch.all(agent_obs.tensor()[Channel.ALLIES_HP] == 0.0)
        assert torch.all(agent_obs.tensor()[Channel.ENEMIES_HP] == 0.0)
        assert torch.all(agent_obs.tensor()[Channel.RESOURCES] == 0.0)
        assert torch.all(agent_obs.tensor()[Channel.OBSTACLES] == 0.0)

        # Dynamic channels should remain unchanged
        assert torch.all(agent_obs.tensor()[Channel.TRAILS] == 1.0)
        assert torch.all(agent_obs.tensor()[Channel.DAMAGE_HEAT] == 1.0)
        assert torch.all(agent_obs.tensor()[Channel.ALLY_SIGNAL] == 1.0)
        assert torch.all(agent_obs.tensor()[Channel.KNOWN_EMPTY] == 1.0)



    def test_update_known_empty(self, agent_obs):
        """Test updating known empty cells."""
        # Set up visibility and entity presence
        agent_obs.tensor()[Channel.VISIBILITY] = 1.0  # All visible
        agent_obs.tensor()[Channel.ALLIES_HP, 3, 4] = 0.5  # Some entity
        agent_obs.tensor()[Channel.ENEMIES_HP, 4, 3] = 0.3  # Some entity

        agent_obs.update_known_empty()

        # Cells with entities should not be marked as known empty
        assert agent_obs.tensor()[Channel.KNOWN_EMPTY, 3, 4] == 0.0
        assert agent_obs.tensor()[Channel.KNOWN_EMPTY, 4, 3] == 0.0

        # Empty visible cells should be marked as known empty
        assert agent_obs.tensor()[Channel.KNOWN_EMPTY, 3, 3] == 1.0

    def test_perceive_world_basic(self, agent_obs):
        """Test basic world perception."""
        # Create simple world layers
        world_layers = {
            "RESOURCES": torch.zeros(10, 10),
            "OBSTACLES": torch.zeros(10, 10),
        }
        world_layers["RESOURCES"][5, 5] = 1.0
        world_layers["OBSTACLES"][6, 6] = 1.0

        # Test basic perception
        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(5, 5),
            self_hp01=0.8,
            allies=[],
            enemies=[],
            goal_world_pos=None,
        )

        # Check that self HP is written
        assert agent_obs.tensor()[Channel.SELF_HP, 3, 3] == 0.8

        # Check that world layers are cropped correctly
        assert agent_obs.tensor()[Channel.RESOURCES, 3, 3] == 1.0
        assert agent_obs.tensor()[Channel.OBSTACLES, 4, 4] == 1.0

    def test_perceive_world_with_orientation_entities(self, agent_obs):
        """Entities rotate so facing direction is up."""
        world_layers = {"RESOURCES": torch.zeros(10, 10)}
        ay, ax = 5, 5
        # Ally to the east in world
        allies = [(ay, ax + 1, 0.9)]
        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(ay, ax),
            self_hp01=1.0,
            allies=allies,
            enemies=[],
            goal_world_pos=None,
            agent_orientation=90.0,  # facing east
        )
        R = agent_obs.config.R
        # East should map to "up" (y=R-1, x=R)
        assert agent_obs.tensor()[Channel.ALLIES_HP, R - 1, R] == pytest.approx(0.9)

    def test_perceive_world_with_orientation_world_layer(self, agent_obs):
        """World layers rotate according to agent orientation."""
        # Place a resource to the east of agent
        grid = torch.zeros(10, 10)
        ay, ax = 5, 5
        grid[ay, ax + 1] = 1.0
        world_layers = {"RESOURCES": grid, "OBSTACLES": torch.zeros(10, 10)}
        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(ay, ax),
            self_hp01=1.0,
            allies=[],
            enemies=[],
            goal_world_pos=None,
            agent_orientation=90.0,  # facing east
        )
        R = agent_obs.config.R
        # Resource east should appear at up position in local crop
        assert agent_obs.tensor()[Channel.RESOURCES, R - 1, R] == pytest.approx(1.0)

    def test_perceive_world_with_entities(self, agent_obs):
        """Test world perception with allies and enemies."""
        world_layers = {"RESOURCES": torch.zeros(10, 10)}

        allies = [(4, 5, 0.9), (6, 5, 0.7)]  # (y, x, hp)
        enemies = [(5, 4, 0.6)]  # (y, x, hp)

        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(5, 5),
            self_hp01=0.8,
            allies=allies,
            enemies=enemies,
            goal_world_pos=None,
        )

        # Check that allies are written correctly (relative to center)
        assert (
            agent_obs.tensor()[Channel.ALLIES_HP, 2, 3] == 0.9
        )  # (4, 5) -> (-1, 0)
        assert agent_obs.tensor()[Channel.ALLIES_HP, 4, 3] == 0.7  # (6, 5) -> (1, 0)

        # Check that enemies are written correctly
        assert (
            agent_obs.tensor()[Channel.ENEMIES_HP, 3, 2] == 0.6
        )  # (5, 4) -> (0, -1)

    def test_perceive_world_with_goal(self, agent_obs):
        """Test world perception with goal."""
        world_layers = {"RESOURCES": torch.zeros(10, 10)}

        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(5, 5),
            self_hp01=0.8,
            allies=[],
            enemies=[],
            goal_world_pos=(7, 7),  # Goal at (7, 7)
        )

        # Check that goal is written correctly (relative to agent at 5,5)
        assert agent_obs.tensor()[Channel.GOAL, 5, 5] == 1.0  # (7-5, 7-5) -> (2, 2)

    def test_perceive_world_with_transient_events(self, agent_obs):
        """Test world perception with transient events."""
        world_layers = {"RESOURCES": torch.zeros(10, 10)}

        recent_damage = [(4, 5, 0.5)]  # (y, x, intensity)
        ally_signals = [(6, 5, 0.8)]  # (y, x, intensity)
        trails = [(5, 4, 0.3)]  # (y, x, intensity)

        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(5, 5),
            self_hp01=0.8,
            allies=[],
            enemies=[],
            goal_world_pos=None,
            recent_damage_world=recent_damage,
            ally_signals_world=ally_signals,
            trails_world_points=trails,
        )

        # Check that transient events are written
        assert agent_obs.tensor()[Channel.DAMAGE_HEAT, 2, 3] == 0.5
        assert agent_obs.tensor()[Channel.ALLY_SIGNAL, 4, 3] == 0.8
        assert agent_obs.tensor()[Channel.TRAILS, 3, 2] == 0.3

    def test_tensor_method(self, agent_obs):
        """Test the tensor method."""
        # Set some values
        agent_obs.tensor()[Channel.SELF_HP, 3, 3] = 0.8

        # Get tensor
        tensor = agent_obs.tensor()

        # Should return the same tensor
        assert torch.equal(tensor, agent_obs.tensor())
        assert tensor.shape == (NUM_CHANNELS, 7, 7)


class TestIntegration:
    """Integration tests for the observation system."""

    def test_full_observation_cycle(self):
        """Test a complete observation cycle."""
        config = ObservationConfig(R=4, fov_radius=3)
        agent_obs = AgentObservation(config)

        # Create world state
        world_layers = {
            "RESOURCES": torch.zeros(20, 20),
            "OBSTACLES": torch.zeros(20, 20),
        }
        world_layers["RESOURCES"][10, 10] = 1.0
        world_layers["OBSTACLES"][12, 12] = 1.0

        # Perform observation
        agent_obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(10, 10),
            self_hp01=0.9,
            allies=[(9, 10, 0.8), (11, 10, 0.7)],
            enemies=[(10, 9, 0.6)],
            goal_world_pos=(15, 15),
            recent_damage_world=[(9, 9, 0.5)],
            ally_signals_world=[(11, 11, 0.8)],
            trails_world_points=[(10, 11, 0.3)],
        )

        # Verify observation tensor
        obs_tensor = agent_obs.tensor()
        assert obs_tensor.shape == (NUM_CHANNELS, 9, 9)  # 2*4 + 1 = 9

        # Check key values
        assert obs_tensor[Channel.SELF_HP, 4, 4] == 0.9  # Center
        assert obs_tensor[Channel.RESOURCES, 4, 4] == 1.0  # Resources at center
        assert obs_tensor[Channel.OBSTACLES, 6, 6] == 1.0  # Obstacles offset
        assert obs_tensor[Channel.ALLIES_HP, 3, 4] == 0.8  # Ally above
        assert obs_tensor[Channel.ALLIES_HP, 5, 4] == 0.7  # Ally below
        assert obs_tensor[Channel.ENEMIES_HP, 4, 3] == 0.6  # Enemy left
        # Goal at (15-10, 15-10) = (5, 5) relative to center (4, 4) = (9, 9) but is out of bounds
        # So no goal should be written (goal only written if within bounds)
        assert (
            torch.sum(obs_tensor[Channel.GOAL]) == 0.0
        )  # No goal written since out of bounds

    def test_observation_decay_over_time(self):
        """Test that observations decay properly over multiple cycles."""
        config = ObservationConfig(R=3, gamma_trail=0.5, gamma_dmg=0.7)
        agent_obs = AgentObservation(config)

        # Set initial values
        agent_obs.tensor()[Channel.TRAILS] = 1.0
        agent_obs.tensor()[Channel.DAMAGE_HEAT] = 1.0

        # Apply decay multiple times
        for _ in range(3):
            agent_obs.decay_dynamics()

        # Check decayed values
        expected_trail = 1.0 * (0.5**3)
        expected_dmg = 1.0 * (0.7**3)

        assert torch.allclose(
            agent_obs.tensor()[Channel.TRAILS], torch.tensor(expected_trail)
        )
        assert torch.allclose(
            agent_obs.tensor()[Channel.DAMAGE_HEAT], torch.tensor(expected_dmg)
        )


if __name__ == "__main__":
    pytest.main([__file__])


class TestSparseReductionModes:
    """Tests for different reduction modes in SparsePoints application."""

    def test_max_reduction_scatter_backend(self):
        config = ObservationConfig(R=1, default_point_reduction="max", sparse_backend="scatter")
        obs = AgentObservation(config)
        # Two points at the same location (1,1), expect max
        obs._store_sparse_points(Channel.ALLIES_HP, [(1, 1, 0.4), (1, 1, 0.7)], accumulate=False)
        tensor = obs.tensor()
        assert tensor[Channel.ALLIES_HP, 1, 1] == pytest.approx(0.7)

    def test_sum_reduction_scatter_backend(self):
        config = ObservationConfig(R=1, default_point_reduction="sum", sparse_backend="scatter")
        obs = AgentObservation(config)
        obs._store_sparse_points(Channel.ALLIES_HP, [(1, 1, 0.4), (1, 1, 0.7)], accumulate=False)
        tensor = obs.tensor()
        assert tensor[Channel.ALLIES_HP, 1, 1].item() == pytest.approx(1.1, rel=1e-6, abs=1e-6)

    def test_overwrite_reduction_scatter_backend(self):
        config = ObservationConfig(R=1, default_point_reduction="overwrite", sparse_backend="scatter")
        obs = AgentObservation(config)
        obs._store_sparse_points(Channel.ALLIES_HP, [(1, 1, 0.4), (1, 1, 0.7)], accumulate=False)
        tensor = obs.tensor()
        # Last value should win
        assert tensor[Channel.ALLIES_HP, 1, 1] == pytest.approx(0.7)


class TestSparseCOOBackend:
    """Tests for COO backend behavior, especially sum reduction efficiency."""

    def test_sum_reduction_coo_backend(self):
        config = ObservationConfig(R=1, default_point_reduction="sum", sparse_backend="coo")
        obs = AgentObservation(config)
        obs._store_sparse_points(Channel.ENEMIES_HP, [(1, 1, 0.25), (1, 1, 0.75)], accumulate=False)
        tensor = obs.tensor()
        assert tensor[Channel.ENEMIES_HP, 1, 1].item() == pytest.approx(1.0, rel=1e-6, abs=1e-6)


class TestSparseMetrics:
    """Tests for sparse/dense metrics exposure and reasonable values."""

    def test_sparse_and_dense_metrics_exposed(self):
        config = ObservationConfig(R=1, default_point_reduction="max", sparse_backend="scatter", enable_metrics=True)
        obs = AgentObservation(config)
        obs._store_sparse_points(Channel.ALLY_SIGNAL, [(0, 0, 0.5), (0, 0, 0.6)], accumulate=False)
        # Trigger build
        _ = obs.tensor()
        metrics = obs.get_metrics()
        # Check required keys present
        for key in [
            "dense_bytes",
            "sparse_points",
            "sparse_logical_bytes",
            "memory_reduction_percent",
            "cache_hits",
            "cache_misses",
            "dense_rebuilds",
            "dense_rebuild_time_s_total",
            "sparse_apply_calls",
            "sparse_apply_time_s_total",
        ]:
            assert key in metrics
        # Basic sanity checks
        assert metrics["sparse_points"] >= 2
        assert metrics["dense_rebuilds"] >= 1
        assert metrics["sparse_apply_calls"] >= 1


class TestGridSparsification:
    """Tests for grid sparsification to SparsePoints with accuracy checks."""

    def test_grid_sparsification_accuracy_and_memory(self):
        # Configure to encourage sparsification
        config = ObservationConfig(
            R=2,
            dtype="float32",
            grid_sparsify_enabled=True,
            grid_sparsify_threshold=0.25,  # sparsify when <25% non-zero
            grid_zero_epsilon=1e-12,
        )
        obs = AgentObservation(config)

        # Build a sparse local grid (S=2R+1=5)
        S = 2 * config.R + 1
        grid = torch.zeros(S, S, dtype=config.torch_dtype)
        grid[0, 0] = 1.0
        grid[S - 1, S // 2] = 0.5

        # Store into a full-grid channel (e.g., RESOURCES)
        channel_idx = Channel.RESOURCES
        obs._store_sparse_grid(channel_idx, grid)

        # Expect sparsification to SparsePoints backend
        assert channel_idx in obs.sparse_channels
        assert isinstance(obs.sparse_channels[channel_idx], SparsePoints)

        # Dense reconstruction should match original grid exactly
        dense = obs.tensor()
        assert torch.allclose(dense[channel_idx], grid, atol=0.0, rtol=0.0)

        # Memory estimate should indicate reduction relative to dense baseline
        metrics = obs.get_metrics()
        assert metrics["sparse_logical_bytes"] < metrics["dense_bytes"]


class TestObservationFrequency(unittest.TestCase):
    """Test that observations are called the correct number of times during simulation."""

    def setUp(self):
        """Set up test environment with minimal configuration."""
        import tempfile
        from farm.config import SimulationConfig
        from farm.config.config import EnvironmentConfig, PopulationConfig, ResourceConfig
        from farm.core.environment import Environment
        from farm.core.agent import AgentFactory

        self.test_dir = tempfile.mkdtemp()
        self.db_path = f"{self.test_dir}/test.db"

        # Create minimal config for testing
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            population=PopulationConfig(system_agents=2, independent_agents=1, control_agents=0),
            resources=ResourceConfig(initial_resources=10),
            max_steps=5,  # Small number of steps for testing
        )

        # Create environment
        self.env = Environment(
            width=50,
            height=50,
            resource_distribution={"amount": 10},
            config=self.config,
            db_path=self.db_path,
        )

        # Add agents
        self.agents = []
        factory = AgentFactory(spatial_service=self.env.spatial_service)
        for i in range(3):  # 2 system + 1 independent = 3 total
            agent = factory.create_default_agent(
                agent_id=self.env.get_next_agent_id(),
                position=(10 + i * 5, 10 + i * 5),
                initial_resources=5,
            )
            self.agents.append(agent)
            self.env.add_agent(agent)

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'env') and self.env:
            self.env.cleanup()

        import shutil
        if hasattr(self, 'test_dir') and self.test_dir:
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_observation_frequency_per_step_per_agent(self):
        """Test that perceive_world is called correctly: once per agent during reset (caching) + once per agent per step.

        This validates that the observation system doesn't have excessive redundant calls.
        The reset calls are expected in AEC environments for caching observations.
        For N agents and S steps: expect N + (N*S) total calls.
        """
        # Counter to track observation calls with context
        observation_calls = []

        # Get the AgentObservation class and store original method
        from farm.core.observations import AgentObservation
        original_perceive_world = AgentObservation.perceive_world

        def tracking_perceive_world(self_obs, *args, **kwargs):
            # Track when and for which agent the call is made
            agent_pos = kwargs.get('agent_world_pos', 'unknown')
            call_info = {
                'agent_pos': agent_pos,
                'call_stack': []  # Could add more context if needed
            }
            observation_calls.append(call_info)
            # Call the original method
            return original_perceive_world(self_obs, *args, **kwargs)

        # Replace with tracking version
        AgentObservation.perceive_world = tracking_perceive_world

        try:
            # Reset environment to starting state
            reset_start_count = len(observation_calls)
            self.env.reset()
            reset_calls = len(observation_calls) - reset_start_count

            # Run simulation for known number of full cycles (steps)
            num_full_cycles = 3  # Complete cycles where all agents act
            num_agents = len(self.env.agents)
            initial_agent_selection = self.env.agent_selection

            step_start_count = len(observation_calls)
            steps_taken = 0
            cycles_completed = 0

            # Continue until we've completed the desired number of cycles
            while cycles_completed < num_full_cycles and not self.env.terminations.get(self.env.agent_selection, False):
                # Take a step for the current agent
                action = None  # No action
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps_taken += 1

                # Check if we've completed a full cycle (all agents have acted)
                if self.env.agent_selection == initial_agent_selection:
                    cycles_completed += 1

                # Break if episode ended
                if terminated or truncated:
                    break

            step_calls = len(observation_calls) - step_start_count
            total_calls = len(observation_calls)

            # Expected: reset_calls + (num_agents * cycles_completed)
            expected_step_calls = num_agents * cycles_completed
            expected_total_calls = reset_calls + expected_step_calls

            print(f"Reset calls: {reset_calls}, Step calls: {step_calls}, Total calls: {total_calls}")
            print(f"Expected: {reset_calls} + ({num_agents} × {cycles_completed}) = {expected_total_calls}")

            # Check that we have the right number of calls during steps
            self.assertEqual(
                step_calls,
                expected_step_calls,
                f"Expected {expected_step_calls} observation calls during steps "
                f"({num_agents} agents × {cycles_completed} cycles), "
                f"but got {step_calls}."
            )

            # Check total calls
            self.assertEqual(
                total_calls,
                expected_total_calls,
                f"Expected {expected_total_calls} total observation calls "
                f"({reset_calls} during reset + {expected_step_calls} during steps), "
                f"but got {total_calls}."
            )

            # Verify that reset calls equal number of agents
            self.assertEqual(
                reset_calls,
                num_agents,
                f"Expected {num_agents} observation calls during reset (one per agent), "
                f"but got {reset_calls}."
            )

        finally:
            # Restore original method
            AgentObservation.perceive_world = original_perceive_world
