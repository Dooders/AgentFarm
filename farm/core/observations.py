"""
Agent Observation System for Multi-Agent Simulations

This module provides a comprehensive observation system for agents in grid-based
multi-agent simulations. It implements local perception with multiple channels
for different types of environmental and social information.

The observation system is designed around the concept of an agent-centered view,
where each agent maintains a local observation buffer that represents their perception
of the world from their own perspective. This includes both instantaneous observations
(current state) and dynamic observations that persist and decay over time.

Key Components:
    - Channel: Enumeration of observation channels (self HP, allies, enemies, etc.)
    - ObservationConfig: Configuration class for observation parameters
    - AgentObservation: Main class managing an agent's observation buffer
    - Utility functions for local cropping and mask generation

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

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator

from farm.core.channels import NUM_CHANNELS, Channel, get_channel_registry
from farm.core.spatial_index import SpatialIndex

logger = logging.getLogger(__name__)


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


def crop_local(
    grid: torch.Tensor,  # (H, W)
    center: Tuple[int, int],  # (y, x) in world coords
    R: int,
    pad_val: float = 0.0,
) -> torch.Tensor:
    """
    Extract a square crop from a 2D grid centered at the specified position.

    This function creates a local view by extracting a (2R+1) x (2R+1) square
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
        >>> crop = crop_local(grid, center=(50, 50), R=5)
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


def crop_local_stack(
    gridC: torch.Tensor,  # (C, H, W)
    center: Tuple[int, int],
    R: int,
    pad_val: float = 0.0,
) -> torch.Tensor:
    """
    Extract square crops from a multi-channel 2D grid centered at the specified position.

    This function applies crop_local to each channel of a multi-channel tensor,
    creating a local view across all channels. The result is a stacked tensor
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
        >>> crop = crop_local_stack(gridC, center=(50, 50), R=5)
        >>> crop.shape
        torch.Size([3, 11, 11])
    """
    return torch.stack(
        [crop_local(gridC[c], center, R, pad_val) for c in range(gridC.shape[0])],
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
    Manages an agent's local observation buffer with multiple perception channels.

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
        self.registry = get_channel_registry()
        S = 2 * config.R + 1
        self.observation = torch.zeros(
            self.registry.num_channels,
            S,
            S,
            device=config.device,
            dtype=config.torch_dtype,
        )

    def _compute_entities_from_spatial_index(
        self,
        spatial_index: Optional["SpatialIndex"],
        agent_object: Optional[object],
        agent_world_pos: Tuple[int, int],
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """Compute allies and enemies using the provided spatial index.

        Returns (allies, enemies) lists of tuples (y, x, hp01).
        """
        if spatial_index is None or agent_object is None:
            return ([], [])

        ay, ax = agent_world_pos
        query_position_xy = (float(ax), float(ay))
        radius = float(self.config.fov_radius)

        try:
            nearby_agents = spatial_index.get_nearby_agents(query_position_xy, radius)
        except Exception as e:  # pragma: no cover - defensive only
            logger.exception(
                "Error in spatial_index.get_nearby_agents; defaulting to empty list"
            )
            nearby_agents = []

        computed_allies: List[Tuple[int, int, float]] = []
        computed_enemies: List[Tuple[int, int, float]] = []
        agent_type = getattr(agent_object, "agent_type", type(agent_object).__name__)
        for other in nearby_agents:
            if other is agent_object or not getattr(other, "alive", False):
                continue

            position = getattr(other, "position", None)
            if (
                position is None
                or not isinstance(position, (list, tuple))
                or len(position) < 2
            ):
                # Skip invalid positions
                continue

            # Use configurable discretization method for consistent mapping to grid cells
            # This avoids overlap issues and preserves sub-grid precision
            discretization_method = getattr(
                self.config, "position_discretization_method", "floor"
            )
            if discretization_method == "round":
                nx = int(round(position[0]))
                ny = int(round(position[1]))
            elif discretization_method == "ceil":
                nx = int(math.ceil(position[0]))
                ny = int(math.ceil(position[1]))
            else:  # "floor" (default)
                nx = int(math.floor(position[0]))
                ny = int(math.floor(position[1]))

            starting_health = getattr(other, "starting_health", 0)
            current_health = getattr(other, "current_health", 0)
            hp01 = (
                float(current_health) / float(starting_health)
                if starting_health and starting_health > 0
                else 0.0
            )

            other_agent_type = getattr(other, "agent_type", type(other).__name__)

            # Consider agents of the same type as allies, different types as enemies
            if other_agent_type == agent_type:
                computed_allies.append((ny, nx, hp01))
            else:
                computed_enemies.append((ny, nx, hp01))

        return (computed_allies, computed_enemies)

    def decay_dynamics(self):
        """
        Apply decay factors to dynamic observation channels.

        This method multiplies each dynamic channel by its corresponding decay rate
        (gamma value) to simulate the gradual fading of transient information over time.
        Dynamic channels include trails, damage heat, ally signals, and known empty cells.
        """
        self.registry.apply_decay(self.observation, self.config)

    def clear_instant(self):
        """
        Clear all instantaneous observation channels.

        This method resets all channels that are overwritten each tick with fresh
        world data. Instantaneous channels represent current state information that
        doesn't persist between ticks, such as current HP, visible entities, and
        immediate environmental features.
        """
        self.registry.clear_instant(self.observation)

    def update_known_empty(self):
        """
        Update the known empty cells based on current visibility and entity presence.

        This method identifies cells that are currently visible and contain no entities
        (allies, enemies, resources, or obstacles) and marks them as known empty.
        The known empty information persists across ticks and decays over time.
        """
        # known_empty = known_empty ∪ (visible & empty_of_all_entities)
        try:
            visibility_idx = self.registry.get_index("VISIBILITY")
            known_empty_idx = self.registry.get_index("KNOWN_EMPTY")
            allies_idx = self.registry.get_index("ALLIES_HP")
            enemies_idx = self.registry.get_index("ENEMIES_HP")
            resources_idx = self.registry.get_index("RESOURCES")
            obstacles_idx = self.registry.get_index("OBSTACLES")

            visible = self.observation[visibility_idx]
            # Cells considered "non-empty" if any of these layers have mass at that cell:
            entity_like = (
                self.observation[allies_idx]
                + self.observation[enemies_idx]
                + self.observation[resources_idx]
                + self.observation[obstacles_idx]
            )
            empty_visible = (visible > 0.5) & (entity_like <= 1e-6)
            self.observation[known_empty_idx][empty_visible] = 1.0
        except KeyError:
            # If any required channels are missing, skip the update
            pass

    # ------------------------
    # World → local pass
    # ------------------------
    def perceive_world(
        self,
        world_layers: Dict[
            str, torch.Tensor
        ],  # world tensors: ("RESOURCES","OBSTACLES","TERRAIN_COST", optional others) shape (H,W), 0..1
        agent_world_pos: Tuple[int, int],  # (y,x) in world coordinates
        self_hp01: float,
        allies: Optional[List[Tuple[int, int, float]]] = None,  # (y,x,hp01)
        enemies: Optional[List[Tuple[int, int, float]]] = None,  # (y,x,hp01)
        goal_world_pos: Optional[Tuple[int, int]] = None,  # (y,x) or None
        recent_damage_world: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # events within last tick
        ally_signals_world: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # (y,x,intensity01)
        trails_world_points: Optional[
            List[Tuple[int, int, float]]
        ] = None,  # you can also auto-add from seen agents
        spatial_index: Optional["SpatialIndex"] = None,
        agent_object: Optional[object] = None,
        **kwargs,  # Additional data for custom channels
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
            spatial_index: Optional spatial index to derive entities efficiently
            agent_object: Optional agent instance used to determine ally/enemy types
        """
        self.decay_dynamics()
        self.clear_instant()

        # Optionally derive allies/enemies from spatial index if not provided
        final_allies = allies
        final_enemies = enemies
        if (
            (final_allies is None or final_enemies is None)
            and spatial_index is not None
            and agent_object is not None
        ):
            computed_allies, computed_enemies = (
                self._compute_entities_from_spatial_index(
                    spatial_index, agent_object, agent_world_pos
                )
            )
            if final_allies is None:
                final_allies = computed_allies
            if final_enemies is None:
                final_enemies = computed_enemies

        # Process all channels using their registered handlers
        kwargs_for_handlers = {
            "world_layers": world_layers,
            "self_hp01": self_hp01,
            "allies": final_allies or [],
            "enemies": final_enemies or [],
            "goal_world_pos": goal_world_pos,
            "recent_damage_world": recent_damage_world,
            "ally_signals_world": ally_signals_world,
            "trails_world_points": trails_world_points,
            "spatial_index": spatial_index,
            "agent_object": agent_object,
            **kwargs,  # Include any additional custom channel data
        }

        for name, handler in self.registry.get_all_handlers().items():
            channel_idx = self.registry.get_index(name)
            handler.process(
                self.observation,
                channel_idx,
                self.config,
                agent_world_pos,
                **kwargs_for_handlers,
            )

        # Known-empty update (special case that depends on other channels)
        self.update_known_empty()

    # ------------------------
    # Backward compatibility methods
    # ------------------------
    def write_visibility(self):
        """Write the field-of-view visibility mask (backward compatibility)."""
        handler = self.registry.get_handler("VISIBILITY")
        channel_idx = self.registry.get_index("VISIBILITY")
        handler.process(self.observation, channel_idx, self.config, (0, 0))

    def write_self(self, hp01: float):
        """Write agent's own health information (backward compatibility)."""
        handler = self.registry.get_handler("SELF_HP")
        channel_idx = self.registry.get_index("SELF_HP")
        handler.process(
            self.observation, channel_idx, self.config, (0, 0), self_hp01=hp01
        )

    def write_points_with_values(
        self, ch_name: str, rel_points: List[Tuple[int, int]], values: List[float]
    ):
        """Write values to specific points (backward compatibility)."""
        try:
            ch_idx = self.registry.get_index(ch_name)
        except KeyError:
            # Fallback to Channel enum for backward compatibility
            ch_idx = getattr(Channel, ch_name)

        R = self.config.R
        for (dy, dx), val in zip(rel_points, values):
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                self.observation[ch_idx, y, x] = max(
                    self.observation[ch_idx, y, x].item(), float(val)
                )

    def write_binary_points(
        self, ch_name: str, rel_points: List[Tuple[int, int]], value: float = 1.0
    ):
        """Write constant value to multiple points (backward compatibility)."""
        self.write_points_with_values(ch_name, rel_points, [value] * len(rel_points))

    def write_goal(self, rel_goal: Optional[Tuple[int, int]]):
        """Write goal position (backward compatibility)."""
        if rel_goal is None:
            return
        handler = self.registry.get_handler("GOAL")
        channel_idx = self.registry.get_index("GOAL")
        # Convert relative position to world position for handler
        # This is a bit of a hack since we don't know the actual agent world pos
        handler.process(
            self.observation,
            channel_idx,
            self.config,
            (0, 0),
            goal_world_pos=rel_goal,  # Pass relative as if it were world pos
        )

    def tensor(self) -> torch.Tensor:
        """
        Get the current observation tensor.

        Returns:
            torch.Tensor: Multi-channel observation tensor of shape (num_channels, 2R+1, 2R+1)
                         representing the agent's current view of the world
        """
        return self.observation
