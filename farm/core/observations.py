"""
Agent Observation System for Multi-Agent Simulations

This module provides a comprehensive observation system for agents in grid-based
multi-agent simulations. It implements egocentric perception with multiple channels
for different types of environmental and social information.

The observation system is designed around the concept of an agent-centered view,
where each agent maintains a local observation buffer that represents their perception
of the world from their own perspective. This includes both instantaneous observations
(current state) and dynamic observations that persist and decay over time.

Key Components:
    - Channel: Enumeration of observation channels (self HP, allies, enemies, etc.)
    - ObservationConfig: Configuration class for observation parameters
    - AgentObservation: Main class managing an agent's observation buffer
    - Utility functions for egocentric cropping and mask generation

Observation Channels:
    The system supports multiple channels of information:
    - SELF_HP: Agent's own health (center pixel only)
    - ALLIES_HP: Health of visible allies
    - ENEMIES_HP: Health of visible enemies
    - RESOURCES: Resource locations and quantities
    - OBSTACLES: Obstacle and terrain information
    - TERRAIN_COST: Movement cost of terrain
    - VISIBILITY: Field-of-view mask
    - KNOWN_EMPTY: Previously observed empty cells
    - DAMAGE_HEAT: Recent damage events (decays over time)
    - TRAILS: Agent movement trails (decays over time)
    - ALLY_SIGNAL: Communication signals from allies (decays over time)
    - GOAL: Current goal or waypoint location

Egocentric View:
    Each agent's observation is a square crop centered on their position with
    shape (NUM_CHANNELS, 2R+1, 2R+1) where R is the observation radius.
    The center pixel (R, R) represents the agent's own position.

Dynamic Decay:
    Certain channels (trails, damage heat, signals, known empty) decay over time
    using configurable gamma factors to simulate the gradual fading of information.

Usage:
    # Create configuration
    config = ObservationConfig(R=6, fov_radius=5)

    # Initialize agent observation
    agent_obs = AgentObservation(config)

    # Update observation with world state
    agent_obs.perceive_world(
        world_layers={"RESOURCES": resource_grid, "OBSTACLES": obstacle_grid},
        agent_world_pos=(50, 50),
        self_hp01=0.8,
        allies=[(48, 50, 0.9), (52, 50, 0.7)],
        enemies=[(45, 45, 0.6)],
        goal_world_pos=(60, 60)
    )

    # Get observation tensor for neural network input
    observation_tensor = agent_obs.tensor()
"""

# observations.py
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator


class Channel(IntEnum):
    """Observation channels for the agent's perception system."""

    SELF_HP = 0  # 1
    ALLIES_HP = 1  # 2
    ENEMIES_HP = 2  # 3
    RESOURCES = 3  # 4
    OBSTACLES = 4  # 5 (binary or passability)
    TERRAIN_COST = 5  # 6 (0..1)
    VISIBILITY = 6  # 7 (1=visible now)
    KNOWN_EMPTY = 7  # 8 (decays)
    DAMAGE_HEAT = 8  # 9 (decays)
    TRAILS = 9  # 10 (decays)
    ALLY_SIGNAL = 10  # 11 (decays)
    GOAL = 11  # 12 (waypoint projection)


NUM_CHANNELS = len(Channel)


class ObservationConfig(BaseModel):
    """Configuration for agent observation system."""

    R: int = Field(default=6, gt=0, description="Radius -> window size = 2R+1")
    gamma_trail: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Decay rate for trails"
    )
    gamma_dmg: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Decay rate for damage heat"
    )
    gamma_sig: float = Field(
        default=0.92, ge=0.0, le=1.0, description="Decay rate for ally signals"
    )
    gamma_known: float = Field(
        default=0.98, ge=0.0, le=1.0, description="Decay rate for known empty cells"
    )
    device: str = Field(default="cpu", description="Device for tensor operations")
    dtype: str = Field(default="float32", description="PyTorch dtype as string")
    fov_radius: int = Field(
        default=6, gt=0, description="Field of view radius (simple disk)"
    )

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v):
        """Convert string dtype to torch.dtype."""
        if isinstance(v, str):
            return getattr(torch, v)
        return v

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype for tensor operations."""
        if isinstance(self.dtype, str):
            return getattr(torch, self.dtype)
        return self.dtype

    model_config = {"arbitrary_types_allowed": True}


def crop_egocentric(
    grid: torch.Tensor,  # (H, W)
    center: Tuple[int, int],  # (y, x) in world coords
    R: int,
    pad_val: float = 0.0,
) -> torch.Tensor:
    """
    Extract a square crop from a 2D grid centered at the specified position.

    This function creates an egocentric view by extracting a (2R+1) x (2R+1) square
    region centered at the given coordinates. When the crop extends beyond the grid
    boundaries, the out-of-bounds areas are filled with the specified padding value.

    Args:
        grid: 2D tensor of shape (H, W) representing the world grid
        center: Tuple of (y, x) coordinates specifying the center of the crop
        R: Radius of the crop (the crop will be (2R+1) x (2R+1) pixels)
        pad_val: Value to use for padding when crop extends beyond grid boundaries

    Returns:
        torch.Tensor: Square crop of shape (2R+1, 2R+1) centered at the specified position

    Example:
        >>> grid = torch.randn(100, 100)
        >>> crop = crop_egocentric(grid, center=(50, 50), R=5)
        >>> crop.shape
        torch.Size([11, 11])
    """
    H, W = grid.shape[-2:]
    y, x = center
    size = 2 * R + 1

    # Compute world-space box
    y0, y1 = y - R, y + R + 1
    x0, x1 = x - R, x + R + 1

    # Amount of padding needed
    top_pad = max(0, -y0)
    left_pad = max(0, -x0)
    bot_pad = max(0, y1 - H)
    right_pad = max(0, x1 - W)

    # Clamp to valid slice
    ys0, ys1 = max(0, y0), min(H, y1)
    xs0, xs1 = max(0, x0), min(W, x1)

    patch = grid[ys0:ys1, xs0:xs1]

    if top_pad or left_pad or bot_pad or right_pad:
        patch = F.pad(
            patch,
            (left_pad, right_pad, top_pad, bot_pad),
            mode="constant",
            value=pad_val,
        )
    # Safety: hard enforce exact size
    if patch.shape[-2:] != (size, size):
        patch = patch[:size, :size]  # should be exact already
    return patch


def crop_egocentric_stack(
    gridC: torch.Tensor,  # (C, H, W)
    center: Tuple[int, int],
    R: int,
    pad_val: float = 0.0,
) -> torch.Tensor:
    """
    Extract square crops from a multi-channel 2D grid centered at the specified position.

    This function applies crop_egocentric to each channel of a multi-channel tensor,
    creating an egocentric view across all channels. The result is a stacked tensor
    where each channel contains the corresponding cropped region.

    Args:
        gridC: Multi-channel 2D tensor of shape (C, H, W) representing the world grid
        center: Tuple of (y, x) coordinates specifying the center of the crop
        R: Radius of the crop (the crop will be (2R+1) x (2R+1) pixels)
        pad_val: Value to use for padding when crop extends beyond grid boundaries

    Returns:
        torch.Tensor: Multi-channel crop of shape (C, 2R+1, 2R+1) centered at the specified position

    Example:
        >>> gridC = torch.randn(3, 100, 100)  # 3 channels
        >>> crop = crop_egocentric_stack(gridC, center=(50, 50), R=5)
        >>> crop.shape
        torch.Size([3, 11, 11])
    """
    return torch.stack(
        [crop_egocentric(gridC[c], center, R, pad_val) for c in range(gridC.shape[0])],
        dim=0,
    )


def make_disk_mask(
    size: int, R: int, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    Create a circular mask with specified radius centered in a square grid.

    This function generates a 2D tensor where pixels within the specified radius
    from the center are set to 1.0, and pixels outside the radius are set to 0.0.
    The mask is useful for creating field-of-view or visibility masks.

    Args:
        size: Size of the square grid (size x size)
        R: Radius of the disk (inclusive). Must be <= size//2 for proper centering
        device: PyTorch device for tensor creation
        dtype: PyTorch data type for the output tensor

    Returns:
        torch.Tensor: Square mask of shape (size, size) with 1.0 inside radius R

    Example:
        >>> mask = make_disk_mask(size=7, R=3)
        >>> mask.shape
        torch.Size([7, 7])
        >>> mask[3, 3]  # center pixel
        tensor(1.)
    """
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    cy = cx = size // 2
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (dist2 <= (R * R)).to(dtype)


class AgentObservation:
    """
    Manages an agent's egocentric observation buffer with multiple perception channels.

    This class maintains a multi-channel observation tensor that represents the agent's
    view of the world from its perspective. It handles both instantaneous observations
    (like current HP, visible entities) and dynamic observations that decay over time
    (like damage heat, trails, signals).

    The observation tensor has shape (NUM_CHANNELS, 2R+1, 2R+1) where R is the
    observation radius. The center pixel (R, R) represents the agent's position.

    Attributes:
        config: Configuration object containing observation parameters
        observation: Multi-channel observation tensor of shape (NUM_CHANNELS, 2R+1, 2R+1)

    Methods:
        decay_dynamics: Apply decay factors to dynamic observation channels.
        clear_instant: Clear all instantaneous observation channels.
        write_visibility: Write the field-of-view visibility mask to the observation tensor.
        write_self: Write the agent's own health information to the center of the observation.
        write_points_with_values: Write values to specific points in a channel relative to the agent's position.
        write_binary_points: Write a constant value to multiple points in a channel.
        write_goal: Write the goal position to the observation tensor.
        update_known_empty: Update the known empty cells based on current visibility and entity presence.
        perceive_world: Perform a complete observation update for one simulation tick.
        tensor: Get the current observation tensor.

    Example:
        >>> config = ObservationConfig(R=6, fov_radius=5)
        >>> agent_obs = AgentObservation(config)
        >>> agent_obs.perceive_world(
        ...     world_layers={"RESOURCES": resource_grid, "OBSTACLES": obstacle_grid},
        ...     agent_world_pos=(50, 50),
        ...     self_hp01=0.8,
        ...     allies=[(48, 50, 0.9), (52, 50, 0.7)],
        ...     enemies=[(45, 45, 0.6)],
        ...     goal_world_pos=(60, 60)
        ... )
        >>> observation_tensor = agent_obs.tensor()
    """

    def __init__(self, config: ObservationConfig):
        self.config = config
        S = 2 * config.R + 1
        self.observation = torch.zeros(
            NUM_CHANNELS, S, S, device=config.device, dtype=config.torch_dtype
        )

    def decay_dynamics(self):
        """
        Apply decay factors to dynamic observation channels.

        This method multiplies each dynamic channel by its corresponding decay rate
        (gamma value) to simulate the gradual fading of transient information over time.
        Dynamic channels include trails, damage heat, ally signals, and known empty cells.
        """
        self.observation[Channel.TRAILS] *= self.config.gamma_trail
        self.observation[Channel.DAMAGE_HEAT] *= self.config.gamma_dmg
        self.observation[Channel.ALLY_SIGNAL] *= self.config.gamma_sig
        self.observation[Channel.KNOWN_EMPTY] *= self.config.gamma_known

    def clear_instant(self):
        """
        Clear all instantaneous observation channels.

        This method resets all channels that are overwritten each tick with fresh
        world data. Instantaneous channels represent current state information that
        doesn't persist between ticks, such as current HP, visible entities, and
        immediate environmental features.
        """
        # Clear channels that are overwritten each tick from fresh world reads
        for channel in (
            Channel.SELF_HP,
            Channel.ALLIES_HP,
            Channel.ENEMIES_HP,
            Channel.RESOURCES,
            Channel.OBSTACLES,
            Channel.TERRAIN_COST,
            Channel.VISIBILITY,
            Channel.GOAL,
        ):
            self.observation[channel].zero_()

    def write_visibility(self):
        """
        Write the field-of-view visibility mask to the observation tensor.

        This method creates a circular visibility mask based on the field-of-view radius
        and writes it to the VISIBILITY channel. The mask indicates which cells are
        currently visible to the agent within its observation range.
        """
        S = 2 * self.config.R + 1
        vis = make_disk_mask(
            S,
            min(self.config.fov_radius, self.config.R),
            device=self.config.device,
            dtype=self.config.torch_dtype,
        )
        self.observation[Channel.VISIBILITY] = vis

    # ------------------------
    # Writers (egocentric)
    # ------------------------
    def write_self(self, hp01: float):
        """
        Write the agent's own health information to the center of the observation.

        Args:
            hp01: Health value normalized to [0, 1] range, where 1.0 is full health
        """
        # center pixel only
        R = self.config.R
        self.observation[Channel.SELF_HP, R, R] = float(hp01)

    def write_points_with_values(
        self, ch_name: str, rel_points: List[Tuple[int, int]], values: List[float]
    ):
        """
        Write values to specific points in a channel relative to the agent's position.

        This method writes values to the observation tensor at specified relative positions.
        Points outside the observation window are ignored. If multiple values are written
        to the same position, the maximum value is retained.

        Args:
            ch_name: Name of the channel to write to (must be a valid Channel enum value)
            rel_points: List of (dy, dx) tuples representing positions relative to center
                       where negative dy is up and negative dx is left
            values: List of values to write, assumed to be in [0, 1] range
        """
        ch = getattr(Channel, ch_name)
        R = self.config.R
        for (dy, dx), val in zip(rel_points, values):
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                self.observation[ch, y, x] = max(
                    self.observation[ch, y, x].item(), float(val)
                )

    def write_binary_points(
        self, ch_name: str, rel_points: List[Tuple[int, int]], value: float = 1.0
    ):
        """
        Write a constant value to multiple points in a channel.

        This is a convenience method that writes the same value to all specified points.
        It's equivalent to calling write_points_with_values with a list of identical values.

        Args:
            ch_name: Name of the channel to write to (must be a valid Channel enum value)
            rel_points: List of (dy, dx) tuples representing positions relative to center
            value: Value to write to all points (default: 1.0)
        """
        self.write_points_with_values(ch_name, rel_points, [value] * len(rel_points))

    def write_goal(self, rel_goal: Optional[Tuple[int, int]]):
        """
        Write the goal position to the observation tensor.

        Args:
            rel_goal: Tuple of (dy, dx) representing goal position relative to agent center,
                     or None if no goal is set
        """
        if rel_goal is None:
            return
        R = self.config.R
        gy = R + rel_goal[0]
        gx = R + rel_goal[1]
        if 0 <= gy < 2 * R + 1 and 0 <= gx < 2 * R + 1:
            self.observation[Channel.GOAL, gy, gx] = 1.0

    def update_known_empty(self):
        """
        Update the known empty cells based on current visibility and entity presence.

        This method identifies cells that are currently visible and contain no entities
        (allies, enemies, resources, or obstacles) and marks them as known empty.
        The known empty information persists across ticks and decays over time.
        """
        # known_empty = known_empty ∪ (visible & empty_of_all_entities)
        visible = self.observation[Channel.VISIBILITY]
        # Cells considered "non-empty" if any of these layers have mass at that cell:
        entity_like = (
            self.observation[Channel.ALLIES_HP]
            + self.observation[Channel.ENEMIES_HP]
            + self.observation[Channel.RESOURCES]
            + self.observation[Channel.OBSTACLES]
        )
        empty_visible = (visible > 0.5) & (entity_like <= 1e-6)
        self.observation[Channel.KNOWN_EMPTY][empty_visible] = 1.0

    # ------------------------
    # World → egocentric pass
    # ------------------------
    def perceive_world(
        self,
        world_layers: Dict[
            str, torch.Tensor
        ],  # world tensors: ("RESOURCES","OBSTACLES","TERRAIN_COST", optional others) shape (H,W), 0..1
        agent_world_pos: Tuple[int, int],  # (y,x) in world coordinates
        self_hp01: float,
        allies: List[Tuple[int, int, float]],  # (y,x,hp01)
        enemies: List[Tuple[int, int, float]],  # (y,x,hp01)
        goal_world_pos: Optional[Tuple[int, int]],  # (y,x) or None
        recent_damage_world: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # events within last tick
        ally_signals_world: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # (y,x,intensity01)
        trails_world_points: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # you can also auto-add from seen agents
    ):
        """
        Perform a complete observation update for one simulation tick.

        This method orchestrates the full observation update process, converting world
        state into the agent's egocentric view. The update follows this sequence:

        1. Apply decay to dynamic channels (trails, damage heat, signals, known empty)
        2. Clear all instantaneous channels
        3. Write field-of-view visibility mask
        4. Write current state information (self HP, allies, enemies, world layers)
        5. Write transient events (damage, trails, ally signals)
        6. Update known empty cells based on visibility and entity presence
        7. Write goal position if specified

        Args:
            world_layers: Dictionary mapping layer names to 2D tensors (H, W) with values in [0, 1]
            agent_world_pos: Agent's position in world coordinates (y, x)
            self_hp01: Agent's health normalized to [0, 1]
            allies: List of (y, x, hp01) tuples for visible allies
            enemies: List of (y, x, hp01) tuples for visible enemies
            goal_world_pos: Goal position in world coordinates, or None
            recent_damage_world: List of (y, x, intensity) tuples for recent damage events
            ally_signals_world: List of (y, x, intensity01) tuples for ally signals
            trails_world_points: List of (y, x, intensity) tuples for agent trails
        """
        self.decay_dynamics()
        self.clear_instant()
        self.write_visibility()

        # Self
        self.write_self(self_hp01)

        # World layers (resources/obstacles/terrain): convert world → egocentric crop, mask by VISIBILITY if desired
        R = self.config.R
        for name in ("RESOURCES", "OBSTACLES", "TERRAIN_COST"):
            if name in world_layers:
                crop = crop_egocentric(
                    world_layers[name], agent_world_pos, R, pad_val=0.0
                )
                # OPTIONAL: enforce strict partial observability now
                # crop *= self.obs[Channel.VISIBILITY]
                self.observation[getattr(Channel, name)].copy_(crop)

        # Allies / Enemies projected as relative offsets
        ay, ax = agent_world_pos

        def to_rel(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            return [(py - ay, px - ax) for (py, px) in points]

        if allies:
            rel_xy = to_rel([(y, x) for (y, x, _) in allies])
            vals = [float(h) for (_, _, h) in allies]
            self.write_points_with_values("ALLIES_HP", rel_xy, vals)

        if enemies:
            rel_xy = to_rel([(y, x) for (y, x, _) in enemies])
            vals = [float(h) for (_, _, h) in enemies]
            self.write_points_with_values("ENEMIES_HP", rel_xy, vals)

        # Damage heat (transient)
        if recent_damage_world:
            rel_xy = to_rel([(y, x) for (y, x, _) in recent_damage_world])
            vals = [float(v) for (_, _, v) in recent_damage_world]
            self.write_points_with_values("DAMAGE_HEAT", rel_xy, vals)

        # Ally signals
        if ally_signals_world:
            rel_xy = to_rel([(y, x) for (y, x, _) in ally_signals_world])
            vals = [float(v) for (_, _, v) in ally_signals_world]
            self.write_points_with_values("ALLY_SIGNAL", rel_xy, vals)

        # Trails
        if trails_world_points:
            rel_xy = to_rel([(y, x) for (y, x, _) in trails_world_points])
            vals = [float(v) for (_, _, v) in trails_world_points]
            self.write_points_with_values("TRAILS", rel_xy, vals)

        # Known-empty update (uses VISIBILITY + entity presence)
        self.update_known_empty()

        # Goal projection
        if goal_world_pos is not None:
            gy, gx = goal_world_pos
            self.write_goal((gy - ay, gx - ax))

    def tensor(self) -> torch.Tensor:
        """
        Get the current observation tensor.

        Returns:
            torch.Tensor: Multi-channel observation tensor of shape (NUM_CHANNELS, 2R+1, 2R+1)
                         representing the agent's current view of the world
        """
        return self.observation
