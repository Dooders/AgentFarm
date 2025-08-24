"""
Unit tests for the observations module.

This module tests the agent observation system including:
- Channel enumeration
- ObservationConfig validation
- Utility functions (crop_egocentric, crop_egocentric_stack, make_disk_mask)
- AgentObservation class and all its methods
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple

from farm.core.observations import (
    Channel,
    NUM_CHANNELS,
    ObservationConfig,
    crop_egocentric,
    crop_egocentric_stack,
    make_disk_mask,
    AgentObservation,
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
        assert NUM_CHANNELS == 12  # 12 channels defined


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
        """Test dtype validation."""
        config = ObservationConfig()
        assert config.validate_dtype("float32") == torch.float32
        assert config.validate_dtype("float64") == torch.float64
        assert config.validate_dtype(torch.float32) == torch.float32


class TestCropEgocentric:
    """Test the crop_egocentric function."""

    def test_basic_crop(self):
        """Test basic cropping functionality."""
        # Create a 10x10 grid with known values
        grid = torch.zeros(10, 10)
        grid[3:7, 3:7] = 1.0  # Create a 4x4 square of 1s
        
        # Crop around center (5, 5) with radius 2
        crop = crop_egocentric(grid, center=(5, 5), R=2)
        
        assert crop.shape == (5, 5)  # 2*2 + 1 = 5
        assert crop[2, 2] == 1.0  # Center should be 1
        # The 4x4 square of 1s should be visible in the crop
        assert crop[0, 0] == 1.0  # Top-left corner of the 4x4 square (maps to world[3,3])
        assert crop[3, 3] == 1.0  # Bottom-right corner of the 4x4 square (maps to world[6,6])
        assert crop[4, 4] == 0.0  # Outside the 4x4 square (maps to world[7,7])

    def test_crop_at_edge(self):
        """Test cropping when center is near grid edge."""
        grid = torch.ones(10, 10)
        
        # Crop at edge (0, 0) with radius 3
        crop = crop_egocentric(grid, center=(0, 0), R=3)
        
        assert crop.shape == (7, 7)  # 2*3 + 1 = 7
        # Should be padded with zeros for out-of-bounds areas
        assert crop[3, 3] == 1.0  # Center of crop should be original (0,0) position
        assert crop[0, 0] == 0.0  # Top-left should be padded (maps to world[-3,-3])
        assert crop[6, 6] == 1.0  # Bottom-right maps to world[3,3] which contains 1.0

    def test_crop_at_corner(self):
        """Test cropping when center is at grid corner."""
        grid = torch.ones(5, 5)
        
        # Crop at corner (4, 4) with radius 2
        crop = crop_egocentric(grid, center=(4, 4), R=2)
        
        assert crop.shape == (5, 5)
        assert crop[2, 2] == 1.0  # Center of crop should be original corner
        assert crop[0, 0] == 1.0  # Maps to world[2,2] which contains 1.0
        assert crop[4, 4] == 0.0  # Maps to world[6,6] which is out of bounds, so padded

    def test_custom_pad_value(self):
        """Test cropping with custom padding value."""
        grid = torch.zeros(10, 10)
        grid[5, 5] = 1.0
        
        # Crop with custom pad value
        crop = crop_egocentric(grid, center=(0, 0), R=3, pad_val=0.5)
        
        assert crop.shape == (7, 7)
        # Check that padded areas have custom value
        assert crop[0, 0] == 0.5  # Top-left corner should be padded (maps to world[-3,-3])
        assert crop[6, 6] == 0.0  # Bottom-right corner maps to world[3,3] which contains 0.0

    def test_large_radius(self):
        """Test cropping with radius larger than grid."""
        grid = torch.ones(3, 3)
        
        # Crop with radius larger than grid
        crop = crop_egocentric(grid, center=(1, 1), R=5)
        
        assert crop.shape == (11, 11)  # 2*5 + 1 = 11
        # Most should be padded, only center 3x3 should be original values
        assert crop[5, 5] == 1.0  # Center of original grid
        assert crop[0, 0] == 0.0  # Padded corner


class TestCropEgocentricStack:
    """Test the crop_egocentric_stack function."""

    def test_multi_channel_crop(self):
        """Test cropping multi-channel tensor."""
        # Create 3-channel 10x10 grid
        gridC = torch.zeros(3, 10, 10)
        gridC[0, 3:7, 3:7] = 1.0  # Channel 0: 4x4 square
        gridC[1, 4:6, 4:6] = 2.0  # Channel 1: 2x2 square
        gridC[2, 5, 5] = 3.0      # Channel 2: single pixel
        
        # Crop around center (5, 5) with radius 2
        crop = crop_egocentric_stack(gridC, center=(5, 5), R=2)
        
        assert crop.shape == (3, 5, 5)  # 3 channels, 5x5 spatial
        assert crop[0, 2, 2] == 1.0  # Channel 0 center
        assert crop[1, 2, 2] == 2.0  # Channel 1 center
        assert crop[2, 2, 2] == 3.0  # Channel 2 center

    def test_edge_cropping_multi_channel(self):
        """Test multi-channel cropping at edge."""
        gridC = torch.ones(2, 5, 5)
        
        # Crop at edge
        crop = crop_egocentric_stack(gridC, center=(0, 0), R=2)
        
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
        assert agent_obs.observation.shape == (NUM_CHANNELS, 7, 7)  # 2*3 + 1 = 7
        assert agent_obs.observation.device.type == "cpu"
        assert agent_obs.observation.dtype == torch.float32
        assert torch.all(agent_obs.observation == 0.0)

    def test_decay_dynamics(self, agent_obs):
        """Test decay dynamics application."""
        # Set some initial values
        agent_obs.observation[Channel.TRAILS] = 1.0
        agent_obs.observation[Channel.DAMAGE_HEAT] = 1.0
        agent_obs.observation[Channel.ALLY_SIGNAL] = 1.0
        agent_obs.observation[Channel.KNOWN_EMPTY] = 1.0
        
        # Apply decay
        agent_obs.decay_dynamics()
        
        # Check that values have been decayed
        assert torch.all(agent_obs.observation[Channel.TRAILS] == 0.90)
        assert torch.all(agent_obs.observation[Channel.DAMAGE_HEAT] == 0.85)
        assert torch.all(agent_obs.observation[Channel.ALLY_SIGNAL] == 0.92)
        assert torch.all(agent_obs.observation[Channel.KNOWN_EMPTY] == 0.98)

    def test_clear_instant(self, agent_obs):
        """Test clearing of instantaneous channels."""
        # Set some values in all channels
        agent_obs.observation.fill_(1.0)
        
        # Clear instant channels
        agent_obs.clear_instant()
        
        # Check that instant channels are cleared
        assert torch.all(agent_obs.observation[Channel.SELF_HP] == 0.0)
        assert torch.all(agent_obs.observation[Channel.ALLIES_HP] == 0.0)
        assert torch.all(agent_obs.observation[Channel.ENEMIES_HP] == 0.0)
        assert torch.all(agent_obs.observation[Channel.RESOURCES] == 0.0)
        assert torch.all(agent_obs.observation[Channel.OBSTACLES] == 0.0)
        
        # Dynamic channels should remain unchanged
        assert torch.all(agent_obs.observation[Channel.TRAILS] == 1.0)
        assert torch.all(agent_obs.observation[Channel.DAMAGE_HEAT] == 1.0)
        assert torch.all(agent_obs.observation[Channel.ALLY_SIGNAL] == 1.0)
        assert torch.all(agent_obs.observation[Channel.KNOWN_EMPTY] == 1.0)

    def test_write_visibility(self, agent_obs):
        """Test visibility mask writing."""
        agent_obs.write_visibility()
        
        # Check that visibility channel has been written
        vis_channel = agent_obs.observation[Channel.VISIBILITY]
        assert vis_channel[3, 3] == 1.0  # Center should be visible
        assert vis_channel[0, 0] == 0.0  # Corner should not be visible

    def test_write_self(self, agent_obs):
        """Test writing self health information."""
        agent_obs.write_self(0.75)
        
        # Check that health is written to center pixel
        assert agent_obs.observation[Channel.SELF_HP, 3, 3] == 0.75
        # Other pixels should remain 0
        assert agent_obs.observation[Channel.SELF_HP, 0, 0] == 0.0

    def test_write_points_with_values(self, agent_obs):
        """Test writing values to specific points."""
        rel_points = [(0, 1), (1, 0), (-1, 0)]  # Relative to center
        values = [0.5, 0.7, 0.3]
        
        agent_obs.write_points_with_values("ALLIES_HP", rel_points, values)
        
        # Check that values are written correctly
        assert agent_obs.observation[Channel.ALLIES_HP, 3, 4] == 0.5  # (0, 1)
        assert agent_obs.observation[Channel.ALLIES_HP, 4, 3] == 0.7  # (1, 0)
        assert agent_obs.observation[Channel.ALLIES_HP, 2, 3] == 0.3  # (-1, 0)

    def test_write_points_outside_window(self, agent_obs):
        """Test writing points outside observation window."""
        rel_points = [(10, 10), (-10, -10)]  # Outside window
        values = [1.0, 1.0]
        
        # Should not raise error, just ignore out-of-window points
        agent_obs.write_points_with_values("ALLIES_HP", rel_points, values)
        
        # All values should remain 0
        assert torch.all(agent_obs.observation[Channel.ALLIES_HP] == 0.0)

    def test_write_points_collision(self, agent_obs):
        """Test writing multiple values to same position."""
        rel_points = [(0, 0), (0, 0)]  # Same position
        values = [0.3, 0.7]
        
        agent_obs.write_points_with_values("ALLIES_HP", rel_points, values)
        
        # Should use maximum value
        assert agent_obs.observation[Channel.ALLIES_HP, 3, 3] == 0.7

    def test_write_binary_points(self, agent_obs):
        """Test writing binary points."""
        rel_points = [(0, 1), (1, 0)]
        
        agent_obs.write_binary_points("ENEMIES_HP", rel_points, value=0.8)
        
        # Check that all points have the same value
        assert agent_obs.observation[Channel.ENEMIES_HP, 3, 4] == 0.8
        assert agent_obs.observation[Channel.ENEMIES_HP, 4, 3] == 0.8

    def test_write_goal(self, agent_obs):
        """Test writing goal position."""
        rel_goal = (1, 2)  # Relative to center
        
        agent_obs.write_goal(rel_goal)
        
        # Check that goal is written correctly
        assert agent_obs.observation[Channel.GOAL, 4, 5] == 1.0  # (3+1, 3+2)

    def test_write_goal_none(self, agent_obs):
        """Test writing None goal."""
        # Should not raise error
        agent_obs.write_goal(None)
        
        # Goal channel should remain unchanged
        assert torch.all(agent_obs.observation[Channel.GOAL] == 0.0)

    def test_write_goal_outside_window(self, agent_obs):
        """Test writing goal outside observation window."""
        rel_goal = (10, 10)  # Outside window
        
        # Should not raise error, just ignore
        agent_obs.write_goal(rel_goal)
        
        # Goal channel should remain unchanged
        assert torch.all(agent_obs.observation[Channel.GOAL] == 0.0)

    def test_update_known_empty(self, agent_obs):
        """Test updating known empty cells."""
        # Set up visibility and entity presence
        agent_obs.observation[Channel.VISIBILITY] = 1.0  # All visible
        agent_obs.observation[Channel.ALLIES_HP, 3, 4] = 0.5  # Some entity
        agent_obs.observation[Channel.ENEMIES_HP, 4, 3] = 0.3  # Some entity
        
        agent_obs.update_known_empty()
        
        # Cells with entities should not be marked as known empty
        assert agent_obs.observation[Channel.KNOWN_EMPTY, 3, 4] == 0.0
        assert agent_obs.observation[Channel.KNOWN_EMPTY, 4, 3] == 0.0
        
        # Empty visible cells should be marked as known empty
        assert agent_obs.observation[Channel.KNOWN_EMPTY, 3, 3] == 1.0

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
        assert agent_obs.observation[Channel.SELF_HP, 3, 3] == 0.8
        
        # Check that world layers are cropped correctly
        assert agent_obs.observation[Channel.RESOURCES, 3, 3] == 1.0
        assert agent_obs.observation[Channel.OBSTACLES, 4, 4] == 1.0

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
        assert agent_obs.observation[Channel.ALLIES_HP, 2, 3] == 0.9  # (4, 5) -> (-1, 0)
        assert agent_obs.observation[Channel.ALLIES_HP, 4, 3] == 0.7  # (6, 5) -> (1, 0)
        
        # Check that enemies are written correctly
        assert agent_obs.observation[Channel.ENEMIES_HP, 3, 2] == 0.6  # (5, 4) -> (0, -1)

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
        assert agent_obs.observation[Channel.GOAL, 5, 5] == 1.0  # (7-5, 7-5) -> (2, 2)

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
        assert agent_obs.observation[Channel.DAMAGE_HEAT, 2, 3] == 0.5
        assert agent_obs.observation[Channel.ALLY_SIGNAL, 4, 3] == 0.8
        assert agent_obs.observation[Channel.TRAILS, 3, 2] == 0.3

    def test_tensor_method(self, agent_obs):
        """Test the tensor method."""
        # Set some values
        agent_obs.observation[Channel.SELF_HP, 3, 3] = 0.8
        
        # Get tensor
        tensor = agent_obs.tensor()
        
        # Should return the same tensor
        assert torch.equal(tensor, agent_obs.observation)
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
        # So no goal should be written (write_goal only writes if within bounds)
        assert torch.sum(obs_tensor[Channel.GOAL]) == 0.0  # No goal written since out of bounds

    def test_observation_decay_over_time(self):
        """Test that observations decay properly over multiple cycles."""
        config = ObservationConfig(R=3, gamma_trail=0.5, gamma_dmg=0.7)
        agent_obs = AgentObservation(config)
        
        # Set initial values
        agent_obs.observation[Channel.TRAILS] = 1.0
        agent_obs.observation[Channel.DAMAGE_HEAT] = 1.0
        
        # Apply decay multiple times
        for _ in range(3):
            agent_obs.decay_dynamics()
        
        # Check decayed values
        expected_trail = 1.0 * (0.5 ** 3)
        expected_dmg = 1.0 * (0.7 ** 3)
        
        assert torch.allclose(agent_obs.observation[Channel.TRAILS], torch.tensor(expected_trail))
        assert torch.allclose(agent_obs.observation[Channel.DAMAGE_HEAT], torch.tensor(expected_dmg))


if __name__ == "__main__":
    pytest.main([__file__])
