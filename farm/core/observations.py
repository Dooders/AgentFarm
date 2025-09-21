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
    - create_observation_tensor: Factory function for creating observation tensors with
      zeros or random initialization
    - Channel: Dynamic channel system with extensible handlers
    - ObservationConfig: Configuration class for observation parameters
    - AgentObservation: Main class managing an agent's observation buffer
    - ChannelRegistry: Dynamic registry for managing observation channels
    - ChannelHandler: Abstract base class for custom channel implementations
    - Utility functions for local cropping and mask generation

Observation Channels:
    The system supports multiple channels of information through a dynamic registry:
    - SELF_HP: Agent's own health (center pixel only)
    - ALLIES_HP: Health of visible allies
    - ENEMIES_HP: Health of visible enemies
    - RESOURCES: Resource locations and quantities
    - OBSTACLES: Obstacle and terrain information
    - TERRAIN_COST: Movement cost of terrain
    - VISIBILITY: Field-of-view mask
    - KNOWN_EMPTY: Previously observed empty cells (decays over time)
    - DAMAGE_HEAT: Recent damage events (decays over time)
    - TRAILS: Agent movement trails (decays over time)
    - ALLY_SIGNAL: Communication signals from allies (decays over time)
    - GOAL: Current goal or waypoint location
    - LANDMARKS: Permanent landmarks and waypoints (persistent)

Egocentric View:
    Each agent's observation is a square crop centered on their position with
    shape (num_channels, 2R+1, 2R+1) where R is the observation radius and
    num_channels is determined by the active channel registry.
    The center pixel (R, R) represents the agent's own position.

Dynamic Decay:
    Certain channels (trails, damage heat, signals, known empty) decay over time
    using configurable gamma factors to simulate the gradual fading of information.

Channel Behavior Types:
    - INSTANT: Overwritten each tick with fresh data
    - DYNAMIC: Persists across ticks and decays over time
    - PERSISTENT: Persists indefinitely until explicitly cleared

Usage:
    # Create configuration
    config = ObservationConfig(R=6, fov_radius=5)

    # Initialize agent observation with zeros (default)
    agent_obs = AgentObservation(config)

    # Or initialize with random values
    config_random = ObservationConfig(
        R=6,
        fov_radius=5,
        initialization="random",
        random_min=0.0,
        random_max=0.1
    )
    agent_obs_random = AgentObservation(config_random)

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

    # Factory function can also be used directly
    from farm.core.observations import create_observation_tensor
    zeros_obs = create_observation_tensor(13, 13)  # 13 channels, 13x13 size
    random_obs = create_observation_tensor(13, 13, initialization="random")
"""

# observations.py
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator

from farm.core.channels import ChannelBehavior, get_channel_registry
from farm.core.observation_render import ObservationRenderer
from farm.core.spatial_index import SpatialIndex

logger = logging.getLogger(__name__)


def create_observation_tensor(
    num_channels: int,
    size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    initialization: str = "zeros",
    random_min: float = 0.0,
    random_max: float = 1.0,
) -> torch.Tensor:
    """
    Factory function to create observation tensors with different initialization methods.

    Args:
        num_channels: Number of observation channels
        size: Size of the square observation window (size x size)
        device: PyTorch device for tensor creation
        dtype: PyTorch data type for the tensor
        initialization: Initialization method - "zeros" or "random"
        random_min: Minimum value for random initialization (inclusive)
        random_max: Maximum value for random initialization (exclusive)

    Returns:
        torch.Tensor: Initialized observation tensor of shape (num_channels, size, size)

    Examples:
        # Create zeros tensor (default)
        obs_zeros = create_observation_tensor(13, 13)

        # Create random tensor
        obs_random = create_observation_tensor(13, 13, initialization="random")

        # Create random tensor with custom range
        obs_custom = create_observation_tensor(
            13, 13,
            initialization="random",
            random_min=-0.1,
            random_max=0.1
        )
    """
    if initialization == "zeros":
        return torch.zeros(num_channels, size, size, device=device, dtype=dtype)
    elif initialization == "random":
        rand_tensor = (
            torch.rand(num_channels, size, size, device=device, dtype=dtype)
            * (random_max - random_min)
            + random_min
        )
        return rand_tensor
    else:
        raise ValueError(
            f"Unknown initialization method: {initialization}. Must be 'zeros' or 'random'"
        )


class ObservationConfig(BaseModel):
    """Configuration for agent observation system.

    The configuration supports different tensor initialization methods:
    - 'zeros': Initialize with zeros (default, maintains backward compatibility)
    - 'random': Initialize with random values between random_min and random_max
    """

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
    initialization: str = Field(
        default="zeros",
        description="Observation tensor initialization method ('zeros' or 'random')",
    )
    storage_mode: str = Field(
        default="hybrid",
        description="Storage mode: 'hybrid' (sparse + lazy dense) or 'dense' baseline",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Collect cache/memory metrics for benchmarking",
    )
    random_min: float = Field(
        default=0.0, description="Minimum value for random initialization"
    )
    random_max: float = Field(
        default=1.0, description="Maximum value for random initialization"
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

    def get_local_observation_size(self) -> Tuple[int, int]:
        """Get the size of the local observation window (2R+1, 2R+1)."""
        size = 2 * self.R + 1
        return (size, size)

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

    The coordinate system uses (y, x) format where y is the row (vertical) coordinate
    and x is the column (horizontal) coordinate. The center position refers to the
    pixel that will become the center of the cropped region.

    Args:
        grid: 2D tensor of shape (H, W) representing the world grid
        center: Tuple of (y, x) coordinates specifying the center of the crop.
               y is the row coordinate, x is the column coordinate.
        R: Radius of the crop (the crop will be (2R+1) x (2R+1) pixels).
           The total crop size is 2*R + 1 in both dimensions.
        pad_val: Value to use for padding when crop extends beyond grid boundaries.
                Default is 0.0.

    Returns:
        torch.Tensor: Square crop of shape (2R+1, 2R+1) centered at the specified position.
                     The returned tensor has the same dtype as the input grid.

    Example:
        >>> grid = torch.randn(100, 100)
        >>> crop = crop_local(grid, center=(50, 50), R=5)
        >>> crop.shape
        torch.Size([11, 11])
        >>> # The crop covers grid positions [45:56, 45:56] centered at (50, 50)
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
        gridC: Multi-channel 2D tensor of shape (C, H, W) representing the world grid.
              C is the number of channels, H and W are the spatial dimensions.
        center: Tuple of (y, x) coordinates specifying the center of the crop.
               y is the row coordinate, x is the column coordinate.
        R: Radius of the crop (the crop will be (2R+1) x (2R+1) pixels per channel).
           The total crop size is 2*R + 1 in both spatial dimensions.
        pad_val: Value to use for padding when crop extends beyond grid boundaries.
                Default is 0.0.

    Returns:
        torch.Tensor: Multi-channel crop of shape (C, 2R+1, 2R+1) centered at the specified position.
                     The returned tensor has the same dtype as the input grid.

    Example:
        >>> gridC = torch.randn(3, 100, 100)  # 3 channels
        >>> crop = crop_local_stack(gridC, center=(50, 50), R=5)
        >>> crop.shape
        torch.Size([3, 11, 11])
        >>> # Each of the 3 channels is cropped to 11x11 pixels centered at (50, 50)
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

    The center of the disk is at coordinates (size//2, size//2). The radius is
    measured using Euclidean distance, and pixels are included if their distance
    from the center is less than or equal to R.

    Args:
        size: Size of the square grid (size x size). Must be odd for perfect centering.
        R: Radius of the disk (inclusive). Pixels with distance <= R from center
           are set to 1.0. For perfect centering, R should be <= size//2.
        device: PyTorch device for tensor creation. Default is "cpu".
        dtype: PyTorch data type for the output tensor. Default is torch.float32.

    Returns:
        torch.Tensor: Square mask of shape (size, size) with 1.0 inside radius R
                     and 0.0 outside. Values are of the specified dtype.

    Example:
        >>> mask = make_disk_mask(size=7, R=3)
        >>> mask.shape
        torch.Size([7, 7])
        >>> mask[3, 3]  # center pixel (at size//2 = 3)
        tensor(1.)
        >>> # All pixels within Euclidean distance 3 of center are 1.0
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
    AgentFarm Perception System - Hybrid Sparse/Dense Observation Management.

    OVERVIEW:
    ---------
    This class implements a multi-channel observation system that balances
    memory efficiency with computational performance using a hybrid approach:

    - SPARSE STORAGE: Only non-zero values are stored until needed
    - LAZY DENSE CONSTRUCTION: Full tensors are built on-demand for NN processing
    - CHANNEL-SPECIFIC OPTIMIZATION: Strategies vary by channel type

    OBSERVATION FORMAT:
    ------------------
    - Shape: (num_channels, 2R+1, 2R+1) where R = observation radius
    - Center: (R, R) represents the agent's current position
    - Channels: Provided by the dynamic channel registry (see ChannelRegistry)

    CHANNEL TYPES & STORAGE STRATEGIES:
    ----------------------------------
    INSTANT (overwritten each tick):
      SELF_HP, ALLIES_HP, ENEMIES_HP, GOAL, VISIBILITY, RESOURCES, OBSTACLES, TERRAIN_COST
    DYNAMIC (persists with decay):
      KNOWN_EMPTY, DAMAGE_HEAT, TRAILS, ALLY_SIGNAL
    PERSISTENT (never auto-cleared):
      LANDMARKS

    SPATIAL INTEGRATION:
    -------------------
    - Optional coupling with SpatialIndex for efficient proximity queries
    - Automatic world→local coordinate mapping
    - Bilinear interpolation support for continuous positions

    PERFORMANCE CHARACTERISTICS:
    ---------------------------
    - Memory: O(active_entities) for sparse storage, dense only when needed
    - Computation: Dense tensors provided for NN processing when requested

    Attributes:
        config: ObservationConfig with radius, decay rates, device settings
        registry: ChannelRegistry managing channel registration and indexing
        sparse_channels: Dict[channel_idx -> sparse_data] for memory efficiency
        dense_cache: Optional[torch.Tensor] built lazily for NN processing
        cache_dirty: Boolean flag for cache invalidation

    Example:
        >>> config = ObservationConfig(R=6, fov_radius=5)
        >>> agent_obs = AgentObservation(config)
        >>> agent_obs.perceive_world(world_layers=..., agent_world_pos=(50, 50), self_hp01=0.8)
        >>> observation = agent_obs.tensor()  # (num_channels, 2R+1, 2R+1)
    """

    def __init__(self, config: ObservationConfig):
        self.config = config
        self.registry = get_channel_registry()
        S = 2 * config.R + 1

        # Sparse storage: only allocate when needed
        self.sparse_channels = {}  # {channel_idx: sparse_data}
        self.dense_cache = None  # Lazy dense tensor
        self.cache_dirty = True  # Whether we need to rebuild dense

        # Metrics
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "dense_rebuilds": 0,
            "dense_rebuild_time_s_total": 0.0,
            "sparse_points_count": 0,
            "sparse_points_per_channel": {},
        }

        # Pre-allocate dense tensor only if initialization is non-zero
        if config.initialization != "zeros":
            self.dense_cache = create_observation_tensor(
                num_channels=self.registry.num_channels,
                size=S,
                device=config.device,
                dtype=config.torch_dtype,
                initialization=config.initialization,
                random_min=config.random_min,
                random_max=config.random_max,
            )
            self.cache_dirty = False

        # Dense baseline allocates zero tensor upfront
        if getattr(config, "storage_mode", "hybrid") == "dense" and self.dense_cache is None:
            self.dense_cache = torch.zeros(
                self.registry.num_channels, S, S, device=self.config.device, dtype=self.config.torch_dtype
            )
            self.cache_dirty = False

    def _store_sparse_point(self, channel_idx: int, y: int, x: int, value: float):
        """Store a single point value in sparse format."""
        if getattr(self.config, "storage_mode", "hybrid") == "dense":
            # Write directly into dense cache
            if self.dense_cache is None:
                S = 2 * self.config.R + 1
                self.dense_cache = torch.zeros(
                    self.registry.num_channels, S, S, device=self.config.device, dtype=self.config.torch_dtype
                )
            H, W = self.dense_cache.shape[-2:]
            if 0 <= y < H and 0 <= x < W:
                self.dense_cache[channel_idx, y, x] = value
            if self.config.enable_metrics:
                self._metrics["sparse_points_count"] += 1
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += 1
            return

        if channel_idx not in self.sparse_channels:
            self.sparse_channels[channel_idx] = {}
        self.sparse_channels[channel_idx][(y, x)] = value
        self.cache_dirty = True
        if self.config.enable_metrics:
            self._metrics["sparse_points_count"] += 1
            self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
            self._metrics["sparse_points_per_channel"][channel_idx] += 1

    def _store_sparse_points(
        self,
        channel_idx: int,
        points: List[Tuple[int, int, float]],
        accumulate: bool = True,
    ):
        """Store multiple point values in sparse format."""
        if getattr(self.config, "storage_mode", "hybrid") == "dense":
            if self.dense_cache is None:
                S = 2 * self.config.R + 1
                self.dense_cache = torch.zeros(
                    self.registry.num_channels, S, S, device=self.config.device, dtype=self.config.torch_dtype
                )
            H, W = self.dense_cache.shape[-2:]
            for y, x, value in points:
                if 0 <= y < H and 0 <= x < W:
                    if accumulate:
                        self.dense_cache[channel_idx, y, x] = max(
                            float(self.dense_cache[channel_idx, y, x].item()), float(value)
                        )
                    else:
                        self.dense_cache[channel_idx, y, x] = value
            if self.config.enable_metrics:
                inc = len(points)
                self._metrics["sparse_points_count"] += inc
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += inc
            return

        if channel_idx not in self.sparse_channels:
            self.sparse_channels[channel_idx] = {}

        channel_data = self.sparse_channels[channel_idx]
        for y, x, value in points:
            if accumulate:
                channel_data[(y, x)] = max(channel_data.get((y, x), 0.0), value)
            else:
                channel_data[(y, x)] = value
        self.cache_dirty = True
        if self.config.enable_metrics:
            inc = len(points)
            self._metrics["sparse_points_count"] += inc
            self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
            self._metrics["sparse_points_per_channel"][channel_idx] += inc

    def _store_sparse_grid(self, channel_idx: int, grid: torch.Tensor):
        """Store a full grid (for dense channels like VISIBILITY, RESOURCES)."""
        if getattr(self.config, "storage_mode", "hybrid") == "dense":
            if self.dense_cache is None:
                S = 2 * self.config.R + 1
                self.dense_cache = torch.zeros(
                    self.registry.num_channels, S, S, device=self.config.device, dtype=self.config.torch_dtype
                )
            self.dense_cache[channel_idx].copy_(grid)
            if self.config.enable_metrics:
                try:
                    nz = int((grid != 0).sum().item())
                except Exception:
                    nz = 0
                self._metrics["sparse_points_count"] += nz
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += nz
            return

        self.sparse_channels[channel_idx] = grid  # Store as dense for these
        self.cache_dirty = True
        if self.config.enable_metrics:
            try:
                nz = int((grid != 0).sum().item())
            except Exception:
                nz = 0
            self._metrics["sparse_points_count"] += nz
            self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
            self._metrics["sparse_points_per_channel"][channel_idx] += nz

    def _clear_sparse_channel(self, channel_idx: int):
        """Clear all data for a sparse channel."""
        if channel_idx in self.sparse_channels:
            del self.sparse_channels[channel_idx]
            self.cache_dirty = True
        else:
            # No sparse data exists, clear the dense tensor directly
            if self.dense_cache is not None:
                self.dense_cache[channel_idx].zero_()

    def _decay_sparse_channel(self, channel_idx: int, decay_factor: float):
        """Apply decay to a sparse channel."""
        if channel_idx in self.sparse_channels:
            channel_data = self.sparse_channels[channel_idx]
            if isinstance(channel_data, dict):
                # Sparse points: decay each value
                keys_to_remove = []
                for pos, value in channel_data.items():
                    new_value = value * decay_factor
                    if abs(new_value) < 1e-6:  # Remove effectively zero values
                        keys_to_remove.append(pos)
                    else:
                        channel_data[pos] = new_value
                for pos in keys_to_remove:
                    del channel_data[pos]
            else:
                # Dense grid: decay the tensor
                channel_data *= decay_factor
            self.cache_dirty = True
        else:
            # No sparse data exists, decay the dense tensor directly
            if self.dense_cache is not None:
                self.dense_cache[channel_idx] *= decay_factor

    def _build_dense_tensor(self) -> torch.Tensor:
        """Build dense tensor from sparse data on-demand."""
        # Dense baseline: always a cache hit
        if getattr(self.config, "storage_mode", "hybrid") == "dense":
            if self.config.enable_metrics:
                self._metrics["cache_hits"] += 1
            if self.dense_cache is not None:
                return self.dense_cache
            S = 2 * self.config.R + 1
            self.dense_cache = torch.zeros(
                self.registry.num_channels, S, S, device=self.config.device, dtype=self.config.torch_dtype
            )
            return self.dense_cache

        if not self.cache_dirty and self.dense_cache is not None:
            if self.config.enable_metrics:
                self._metrics["cache_hits"] += 1
            return self.dense_cache

        S = 2 * self.config.R + 1
        num_channels = self.registry.num_channels

        # Create dense tensor
        if self.dense_cache is None:
            self.dense_cache = torch.zeros(
                num_channels,
                S,
                S,
                device=self.config.device,
                dtype=self.config.torch_dtype,
            )

        # Clear existing values and record rebuild time
        t0 = None
        if self.config.enable_metrics:
            import time as _time
            t0 = _time.perf_counter()
        self.dense_cache.zero_()

        # Populate from sparse data
        for channel_idx, channel_data in self.sparse_channels.items():
            if isinstance(channel_data, dict):
                # Sparse points
                for (y, x), value in channel_data.items():
                    if 0 <= y < S and 0 <= x < S:
                        self.dense_cache[channel_idx, y, x] = value
            else:
                # Dense grid (VISIBILITY, RESOURCES, etc.)
                self.dense_cache[channel_idx] = channel_data

        self.cache_dirty = False
        if self.config.enable_metrics:
            import time as _time
            t1 = _time.perf_counter()
            self._metrics["cache_misses"] += 1
            self._metrics["dense_rebuilds"] += 1
            if t0 is not None:
                self._metrics["dense_rebuild_time_s_total"] += max(0.0, t1 - t0)
        return self.dense_cache

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
            nearby = spatial_index.get_nearby(query_position_xy, radius, ["agents"])
            nearby_agents = nearby.get("agents", [])
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ):  # pragma: no cover - defensive only
            logger.exception(
                "Error in spatial_index.get_nearby; defaulting to empty list"
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

        Uses sparse-aware decay that removes effectively zero values to maintain sparsity.
        """
        # Apply decay to each dynamic channel using handler-specific logic
        for name, handler in self.registry.get_all_handlers().items():
            if handler.behavior == ChannelBehavior.DYNAMIC:
                channel_idx = self.registry.get_index(name)
                handler.decay(self, channel_idx, self.config)

    def clear_instant(self):
        """
        Clear all instantaneous observation channels.

        This method resets all channels that are overwritten each tick with fresh
        world data. Instantaneous channels represent current state information that
        doesn't persist between ticks, such as current HP, visible entities, and
        immediate environmental features.

        Uses sparse-aware clearing to maintain memory efficiency.
        """
        # Clear all instantaneous channels using handler-specific logic
        for name, handler in self.registry.get_all_handlers().items():
            if handler.behavior == ChannelBehavior.INSTANT:
                channel_idx = self.registry.get_index(name)
                handler.clear(self, channel_idx)

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

            # Cache the tensor to avoid multiple dense tensor constructions
            obs_tensor = self.tensor()
            visible = obs_tensor[visibility_idx]
            # Cells considered "non-empty" if any of these layers have mass at that cell:
            entity_like = (
                obs_tensor[allies_idx]
                + obs_tensor[enemies_idx]
                + obs_tensor[resources_idx]
                + obs_tensor[obstacles_idx]
            )
            empty_visible = (visible > 0.5) & (entity_like <= 1e-6)
            obs_tensor[known_empty_idx][empty_visible] = 1.0
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
        3. Process all registered channels using their handlers
        4. Update known empty cells based on visibility and entity presence

        The method integrates with the dynamic channel system, automatically processing
        all registered channels through their respective handlers. Custom channels can
        be added by registering new ChannelHandler implementations.

        Args:
            world_layers: Dictionary mapping layer names to 2D tensors (H, W) with values in [0, 1].
                         Expected keys include "RESOURCES", "OBSTACLES", "TERRAIN_COST".
            agent_world_pos: Agent's position in world coordinates (y, x).
            self_hp01: Agent's health normalized to [0, 1].
            allies: List of (y, x, hp01) tuples for visible allies. If None and spatial_index
                   is provided, will be computed automatically.
            enemies: List of (y, x, hp01) tuples for visible enemies. If None and spatial_index
                    is provided, will be computed automatically.
            goal_world_pos: Goal position in world coordinates (y, x), or None.
            recent_damage_world: List of (y, x, intensity) tuples for recent damage events
                                that occurred within the last tick.
            ally_signals_world: List of (y, x, intensity01) tuples for ally communication signals.
            trails_world_points: List of (y, x, intensity) tuples for agent movement trails.
            spatial_index: Optional spatial index for efficient entity queries. If provided
                          along with agent_object, can automatically derive allies and enemies.
            agent_object: Optional agent instance for spatial index queries. Used to determine
                         ally/enemy relationships based on agent types.
            **kwargs: Additional keyword arguments passed to custom channel handlers.
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
                self,  # Pass the AgentObservation instance, not the tensor
                channel_idx,
                self.config,
                agent_world_pos,
                **kwargs_for_handlers,
            )

        # Known-empty update (special case that depends on other channels)
        self.update_known_empty()

    def tensor(self) -> torch.Tensor:
        """
        Get the current observation tensor.

        Builds dense tensor from sparse data on-demand for neural network processing.
        Uses caching to avoid unnecessary reconstruction.

        Returns:
            torch.Tensor: Multi-channel observation tensor of shape (num_channels, 2R+1, 2R+1)
                         representing the agent's current view of the world
        """
        return self._build_dense_tensor()

    # ------------------------
    # Rendering and interactive export
    # ------------------------
    def render(
        self,
        mode: str = "overlay",
        size: int = 256,
        palette: Optional[dict] = None,
        grid: bool = False,
        legend: bool = False,
        background: str = "#111213",
        draw_center: bool = True,
        return_type: str = "pil",
    ):
        """Render the multichannel observation as a minimalist image.

        Args:
            mode: "overlay" to blend channels, or "gallery" to tile per-channel views
            size: Target longer-side size in pixels (nearest-neighbor for crisp cells)
            palette: Optional mapping of channel name -> {color,cmap,alpha}
            grid: Draw subtle grid lines (disabled by default)
            legend: Reserved; currently no-ops in static image
            background: Background color hex string
            draw_center: Draw crosshair at center
            return_type: "pil" | "numpy" | "bytes"

        Returns:
            PIL.Image | numpy.ndarray | bytes
        """
        # Build channel names ordered by channel index
        handlers = self.registry.get_all_handlers()
        ordered = sorted(handlers.keys(), key=lambda n: self.registry.get_index(n))
        return ObservationRenderer.render(
            self.observation,
            ordered,
            mode=mode,
            size=size,
            palette=palette,
            grid=grid,
            legend=legend,
            background=background,
            draw_center=draw_center,
            return_type=return_type,
        )

    def to_interactive_json(self) -> Dict:
        """Export observation and metadata for the interactive HTML viewer."""
        handlers = self.registry.get_all_handlers()
        ordered = sorted(handlers.keys(), key=lambda n: self.registry.get_index(n))
        meta = {"R": int(self.config.R)}
        return ObservationRenderer.to_interactive_json(self.observation, ordered, meta)

    def render_interactive_html(
        self,
        outfile: Optional[str] = None,
        title: str = "Observation Viewer",
        background: str = "#0b0c0f",
        initial_scale: int = 16,
        palette: Optional[dict] = None,
    ) -> str:
        """Generate a self-contained interactive HTML viewer.

        Args:
            outfile: Optional path to write HTML file. If None, returns the HTML string only
            title: Page title
            background: Background color hex string
            initial_scale: Initial integer pixel size per cell
            palette: Optional mapping of channel name -> {color,cmap,alpha}

        Returns:
            HTML string (and writes to outfile if provided)
        """
        handlers = self.registry.get_all_handlers()
        ordered = sorted(handlers.keys(), key=lambda n: self.registry.get_index(n))
        return ObservationRenderer.render_interactive_html(
            self.observation,
            ordered,
            outfile=outfile,
            title=title,
            background=background,
            palette=palette,
            initial_scale=initial_scale,
        )

    def get_metrics(self) -> Dict:
        """Return cache/memory metrics and estimates for this observation."""
        dtype_size = torch.tensor(0, dtype=self.config.torch_dtype).element_size()
        S = 2 * self.config.R + 1
        channels = self.registry.num_channels
        dense_bytes = int(channels * S * S * dtype_size)

        # Count sparse logical points
        sparse_points = 0
        for _, channel_data in self.sparse_channels.items():
            if isinstance(channel_data, dict):
                sparse_points += len(channel_data)
            else:
                try:
                    sparse_points += int((channel_data != 0).sum().item())
                except Exception:
                    pass
        # Include points counted during dense baseline writes
        if getattr(self.config, "storage_mode", "hybrid") == "dense":
            sparse_points = max(sparse_points, int(self._metrics.get("sparse_points_count", 0)))

        # Approximate sparse logical memory: value + y + x (floatsize + 2*int32)
        sparse_logical_bytes = int(sparse_points * (dtype_size + 4 + 4))

        cache_hits = int(self._metrics.get("cache_hits", 0))
        cache_misses = int(self._metrics.get("cache_misses", 0))
        total = cache_hits + cache_misses
        hit_rate = (cache_hits / total) if total > 0 else 1.0
        reduction = 1.0 - (sparse_logical_bytes / dense_bytes) if dense_bytes > 0 else 0.0

        return {
            "dense_bytes": dense_bytes,
            "sparse_points": sparse_points,
            "sparse_logical_bytes": sparse_logical_bytes,
            "memory_reduction_percent": max(0.0, reduction * 100.0),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
            "dense_rebuilds": int(self._metrics.get("dense_rebuilds", 0)),
            "dense_rebuild_time_s_total": float(self._metrics.get("dense_rebuild_time_s_total", 0.0)),
        }
