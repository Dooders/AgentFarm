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
    - create_observation_tensor: Factory for creating observation tensors
    - ObservationConfig: Configuration for observation parameters and storage
    - AgentObservation: Manages an agent's observation buffer
    - SparsePoints: Tensor-backed sparse point storage (y, x -> value)
    - ChannelRegistry/ChannelHandler: Dynamic channel system APIs
    - Utility functions: local cropping and disk mask generation

Observation Channels:
    The system supports multiple channels via a dynamic registry:
    - SELF_HP, ALLIES_HP, ENEMIES_HP, RESOURCES, OBSTACLES, TERRAIN_COST,
      VISIBILITY, KNOWN_EMPTY, DAMAGE_HEAT, TRAILS, ALLY_SIGNAL, GOAL, LANDMARKS

Egocentric View:
    Each observation is a square crop centered on the agent with
    shape (num_channels, 2R+1, 2R+1), center at (R, R).

Dynamic Decay:
    Dynamic channels (trails, damage heat, signals, known empty) decay using
    configurable gamma factors.

Storage & Performance:
    - HYBRID mode stores point-sparse channels as `SparsePoints` until needed,
      building dense tensors on-demand. This reduces Python dict overhead and
      improves GPU transfer efficiency.
    - DENSE mode writes directly to a dense tensor.

Config Highlights (ObservationConfig):
    - storage_mode: HYBRID (default) | DENSE
    - sparse_backend: "scatter" (default) | "coo"
    - default_point_reduction: "max" (default) | "sum" | "overwrite"
    - channel_reduction_overrides: per-channel reduction by name

Metrics (AgentObservation.get_metrics):
    - dense_bytes, sparse_points, sparse_logical_bytes, memory_reduction_percent
    - cache_hits/misses, dense_rebuilds, dense_rebuild_time_s_total
    - sparse_apply_calls, sparse_apply_time_s_total

Usage:
    # Create configuration
    config = ObservationConfig(R=6, fov_radius=5, sparse_backend="scatter",
                               default_point_reduction="max")

    # Initialize agent observation with zeros (default)
    agent_obs = AgentObservation(config)

    # Update observation with world state
    agent_obs.perceive_world(world_layers, agent_world_pos=(50, 50), self_hp01=0.8)

    # Access observation tensor and metrics
    observation_tensor = agent_obs.tensor()
    metrics = agent_obs.get_metrics()

    # Factory function can also be used directly
    from farm.core.observations import create_observation_tensor
    zeros_obs = create_observation_tensor(13, 13)
    random_obs = create_observation_tensor(13, 13, initialization="random")
"""

# observations.py
from __future__ import annotations

import math
import time as _time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)
from pydantic import BaseModel, Field, field_validator

from farm.core.channels import ChannelBehavior, get_channel_registry
from farm.core.observation_render import ObservationRenderer
from farm.core.spatial import SpatialIndex


class SparsePoints:
    """Tensor-backed sparse point storage for (y, x) -> value entries.

    Stores indices as a (2, N) int64 tensor [rows: (y_indices, x_indices)] and
    values as a (N,) tensor with the observation dtype. This representation is
    compact, GPU-friendly, and avoids Python dict overhead.

    Notes:
        - Duplicated indices are allowed during incremental builds. At dense
          reconstruction time, duplicates are reduced via max by default.
        - Out-of-bounds indices are filtered during dense reconstruction.
    """

    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.indices = torch.empty(2, 0, dtype=torch.long, device=self.device)
        self.values = torch.empty(0, dtype=self.dtype, device=self.device)
        # metrics
        self._apply_calls = 0
        self._apply_time_s_total = 0.0

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def add_points(self, points: List[Tuple[int, int, float]]) -> None:
        """Append a batch of (y, x, value) points."""
        if not points:
            return
        ys = torch.tensor([p[0] for p in points], dtype=torch.long, device=self.device)
        xs = torch.tensor([p[1] for p in points], dtype=torch.long, device=self.device)
        vals = torch.tensor(
            [float(p[2]) for p in points], dtype=self.dtype, device=self.device
        )

        self.indices = torch.cat([self.indices, torch.stack([ys, xs], dim=0)], dim=1)
        self.values = torch.cat([self.values, vals], dim=0)

    def add_point(self, y: int, x: int, value: float) -> None:
        self.add_points([(y, x, value)])

    def decay(self, decay_factor: float, prune_eps: float = 1e-6) -> None:
        if len(self) == 0:
            return
        self.values = self.values * decay_factor
        if prune_eps is not None and prune_eps > 0:
            mask = self.values.abs() >= prune_eps
            if not torch.all(mask):
                self.indices = self.indices[:, mask]
                self.values = self.values[mask]

    @torch.no_grad()
    def apply_to_dense(
        self,
        dense_plane: torch.Tensor,
        reduction: str = "max",
        backend: str = "scatter",
    ) -> None:
        """Write sparse points into a dense plane using a specified strategy.

        Args:
            dense_plane: Tensor of shape (S, S) on the target device and dtype.
                Updated in-place.
            reduction: Strategy to combine values when multiple points map to the
                same (y, x):
                - "max": keep the maximum value per index; good for presence/one-hot maps
                - "sum": sum all contributions; good for accumulated intensity
                - "overwrite": last write wins (order-dependent, non-deterministic with duplicates)
            backend: Implementation used to materialize points:
                - "scatter": uses torch.scatter_reduce_ if available; fast on GPU for "max" and "overwrite";
                  supports "sum" via scatter or index_add as fallback.
                - "coo": builds a sparse COO tensor; efficient for "sum" when many duplicates; can materialize
                  and overwrite the dense plane; "max" is emulated.

        Trade-offs:
            - Prefer backend="scatter" for GPU-accelerated amax/sum when duplicates are modest.
            - Prefer backend="coo" for heavy duplicate indices with reduction="sum".
            - "overwrite" is order-dependent; prefer "max"/"sum" for reproducibility.

        Notes:
            Out-of-bounds indices are filtered prior to writing.
        """
        if len(self) == 0:
            return
        t0: float = _time.perf_counter()
        S = dense_plane.shape[-1]
        ys = self.indices[0]
        xs = self.indices[1]

        # Filter out-of-bounds before scattering
        valid = (ys >= 0) & (ys < S) & (xs >= 0) & (xs < S)
        if not torch.any(valid):
            self._apply_calls += 1
            return
        ys = ys[valid]
        xs = xs[valid]
        vals = self.values[valid].to(dense_plane.dtype)

        # If data device differs, move temporarily to match dense
        if ys.device != dense_plane.device:
            ys = ys.to(dense_plane.device)
            xs = xs.to(dense_plane.device)
            vals = vals.to(dense_plane.device)

        flat_idx = ys * S + xs
        flat = dense_plane.view(-1)

        # COO backend (primarily for sum, can emulate max via scatter-reduce beforehand)
        if backend == "coo":
            try:
                # Build sparse COO (coalescing sums duplicates by sum)
                coo = torch.sparse_coo_tensor(
                    torch.stack([ys, xs], dim=0),
                    vals,
                    size=(S, S),
                    device=dense_plane.device,
                    dtype=dense_plane.dtype,
                )
                coo = coo.coalesce()
                if reduction == "sum":
                    dense_plane.add_(coo.to_dense())
                elif reduction == "overwrite":
                    dense_plane.copy_(coo.to_dense())
                elif reduction == "max":
                    # compute per-index max via scatter-reduce then materialize
                    tmp = torch.zeros_like(flat)
                    if hasattr(torch.Tensor, "scatter_reduce_"):
                        tmp.scatter_reduce_(
                            0, flat_idx, vals, reduce="amax", include_self=False
                        )
                    else:
                        self._segment_max_(tmp, flat_idx, vals)
                    dense_plane.copy_(tmp.view(S, S))
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
            finally:
                self._apply_calls += 1
                self._apply_time_s_total += max(0.0, _time.perf_counter() - t0)
            return

        # Scatter backend (default)
        if reduction == "max":
            if hasattr(torch.Tensor, "scatter_reduce_"):
                flat.scatter_reduce_(
                    0, flat_idx, vals, reduce="amax", include_self=False
                )
            else:
                # Fallback: segment max
                self._segment_max_(flat, flat_idx, vals)
        elif reduction == "sum":
            if hasattr(torch.Tensor, "scatter_reduce_"):
                flat.scatter_reduce_(0, flat_idx, vals, reduce="sum", include_self=True)
            else:
                # Fallback: index_add
                flat.index_add_(0, flat_idx, vals)
        elif reduction == "overwrite":
            # Overwrite semantics: last wins (order-dependent)
            flat.index_put_((flat_idx,), vals, accumulate=False)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        self._apply_calls += 1
        self._apply_time_s_total += max(0.0, _time.perf_counter() - t0)

    @staticmethod
    def _segment_max_(
        flat: torch.Tensor, flat_idx: torch.Tensor, vals: torch.Tensor
    ) -> None:
        """Compute segment-wise max over flat indices and write into flat tensor in-place.

        This CPU-friendly fallback sorts indices, finds contiguous segments of identical
        indices, computes per-segment maxima, and updates the dense flat tensor.
        """
        order = torch.argsort(flat_idx)
        flat_idx_sorted = flat_idx[order]
        vals_sorted = vals[order]
        is_new = torch.ones_like(flat_idx_sorted, dtype=torch.bool)
        is_new[1:] = flat_idx_sorted[1:] != flat_idx_sorted[:-1]
        start_positions = torch.nonzero(is_new, as_tuple=False).flatten()
        end_positions = torch.cat(
            [
                start_positions[1:],
                torch.tensor([len(vals_sorted)], device=start_positions.device),
            ]
        )
        for start, end in zip(start_positions.tolist(), end_positions.tolist()):
            t_idx = int(flat_idx_sorted[start].item())
            segment_max = torch.max(vals_sorted[start:end])
            flat[t_idx] = torch.maximum(flat[t_idx], segment_max)


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


class StorageMode(Enum):
    """
    Enum specifying storage modes for observation tensors.

    The storage mode determines how observation data is managed internally,
    balancing memory efficiency with computational performance.

    HYBRID: Uses a combination of sparse and dense storage to optimize memory usage and performance.
      - Stores only non-zero values in sparse format until needed
      - Builds full dense tensors on-demand for neural network processing
      - Uses lazy evaluation with caching to avoid unnecessary reconstruction
      - Ideal for large observation spaces with sparse updates
      - Provides O(active_entities) memory complexity for sparse storage

    DENSE: Uses a fully dense tensor for storage throughout the observation lifecycle.
      - Allocates full dense tensor upfront and writes directly to it
      - No sparse storage or lazy construction overhead
      - May be faster for small or fully populated observation spaces
      - Can consume more memory for large, mostly empty spaces
      - Provides O(total_observation_space) memory complexity

    Use HYBRID when working with large observation spaces with sparse updates.
    Use DENSE when the observation space is small or densely populated.
    """

    HYBRID = "hybrid"
    DENSE = "dense"


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
    high_frequency_channels: List[str] = Field(
        default_factory=list,
        description="Channel names to prebuild densely for frequent access",
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
    storage_mode: StorageMode = Field(
        default=StorageMode.HYBRID,
        description="Storage mode: HYBRID (sparse + lazy dense) or DENSE baseline",
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
    sparse_backend: str = Field(
        default="scatter",
        description="Sparse apply backend: 'scatter' or 'coo'",
    )
    default_point_reduction: str = Field(
        default="max",
        description="Default reduction for point-sparse channels: 'max' | 'sum' | 'overwrite'",
    )
    channel_reduction_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-channel reduction overrides by channel name",
    )
    # Grid sparsification for full-grid channels (e.g., RESOURCES/OBSTACLES/TERRAIN_COST)
    grid_sparsify_enabled: bool = Field(
        default=True,
        description="If True, store full-grid channels as SparsePoints when density is low",
    )
    grid_sparsify_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sparsify grids when non-zero density is below this fraction",
    )
    grid_zero_epsilon: float = Field(
        default=1e-12,
        ge=0.0,
        description="Values with absolute magnitude <= eps are treated as zero for sparsification",
    )
    grid_sparse_reduction: str = Field(
        default="overwrite",
        description="Reduction to use when applying sparsified grids back to dense: 'max'|'sum'|'overwrite'",
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


def rotate_coordinates(
    y: float,
    x: float,
    angle_deg: float,
    center_y: float,
    center_x: float,
) -> Tuple[float, float]:
    """
    Rotate a coordinate (y, x) around a center by angle_deg (degrees, clockwise positive).

    Returns (y_rot, x_rot) as floats.
    """
    if angle_deg % 360 == 0:
        return y, x

    # Convert to offsets from center (dx = x - cx, dy = y - cy)
    dy = float(y) - float(center_y)
    dx = float(x) - float(center_x)

    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)

    # Standard 2D rotation with image coordinates (y down, x right)
    # Using conventional rotation formulas on (dx, dy)
    # x' = dx*cos(a) - dy*sin(a)
    # y' = dx*sin(a) + dy*cos(a)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    return (center_y + dy_rot, center_x + dx_rot)


def _to_normalized_grid(x: torch.Tensor, size: int) -> torch.Tensor:
    """Convert absolute pixel coordinate x (0..size-1) to normalized grid [-1,1] (align_corners=False)."""
    return (2.0 * (x + 0.5) / float(size)) - 1.0


def crop_local_rotated(
    grid: torch.Tensor,  # (H, W)
    center: Tuple[int, int],  # (y, x) in world coords
    R: int,
    orientation: float = 0.0,  # degrees clockwise
    pad_val: float = 0.0,
) -> torch.Tensor:
    """
    Extract a rotated square crop from a 2D grid centered at the specified position.

    Orientation is interpreted as the agent's facing direction in degrees clockwise from north.
    The returned crop is aligned so that the agent's facing direction is "up" in the crop.
    """
    if orientation % 360 == 0:
        return crop_local(grid, center, R, pad_val)

    H, W = grid.shape[-2:]
    cy, cx = center
    S = 2 * R + 1

    # Build output-relative offsets
    device = grid.device
    dtype = grid.dtype
    oy = torch.arange(S, device=device, dtype=torch.float32) - float(R)
    ox = torch.arange(S, device=device, dtype=torch.float32) - float(R)
    yy, xx = torch.meshgrid(oy, ox, indexing="ij")  # (S,S)

    # Map each output offset (yy,xx) to world offsets by rotating by +orientation
    a = math.radians(float(orientation))
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    dx_world = xx * cos_a - yy * sin_a
    dy_world = xx * sin_a + yy * cos_a

    y_world = dy_world + float(cy)
    x_world = dx_world + float(cx)

    # Normalize for grid_sample (x,y order)
    x_norm = _to_normalized_grid(x_world, W)
    y_norm = _to_normalized_grid(y_world, H)
    sample_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1,S,S,2)

    # Prepare input
    inp = grid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    out = F.grid_sample(
        inp,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    crop = out.squeeze(0).squeeze(0)
    if pad_val != 0.0:
        # Replace out-of-bounds sampled regions with pad_val using normalized grid bounds
        x_norm = sample_grid[0, :, :, 0]
        y_norm = sample_grid[0, :, :, 1]
        oob_mask = (x_norm.abs() > 1) | (y_norm.abs() > 1)
        crop = crop.clone()
        crop[oob_mask] = pad_val
    if dtype != torch.float32:
        crop = crop.to(dtype=dtype)
    return crop


def rotate_local_grid(
    grid: torch.Tensor, angle_deg: float, pad_val: float = 0.0
) -> torch.Tensor:
    """
    Rotate a local SxS grid around its center so that the agent's facing direction
    becomes "up". Positive angle rotates the world clockwise; output equals input rotated by -angle.
    """
    if angle_deg % 360 == 0:
        return grid

    S = grid.shape[-1]
    device = grid.device
    dtype = grid.dtype

    # Build normalized sampling grid rotating by +angle to obtain output = input rotated by -angle
    a = math.radians(float(angle_deg))
    cos_a = math.cos(a)
    sin_a = math.sin(a)

    # Create a regular grid of output coordinates in normalized space [-1,1]
    coords = torch.linspace(-1.0, 1.0, steps=S, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (S,S)

    # Map output normalized coords -> input normalized coords via rotation matrix
    x_in = xx * cos_a - yy * sin_a
    y_in = xx * sin_a + yy * cos_a
    sample_grid = torch.stack([x_in, y_in], dim=-1).unsqueeze(0)  # (1,S,S,2)

    inp = grid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
    out = F.grid_sample(
        inp,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    rot = out.squeeze(0).squeeze(0)
    if pad_val != 0.0:
        # Create a mask of out-of-bounds (padded) regions by sampling ones
        ones_inp = torch.ones_like(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask_out = (
            F.grid_sample(
                ones_inp,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask = mask_out < 1e-6
        rot = rot.clone()
        rot[mask] = pad_val
    if dtype != torch.float32:
        rot = rot.to(dtype=dtype)
    return rot


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
        self.sparse_channels = (
            {}
        )  # {channel_idx: SparsePoints for point-sparse data, torch.Tensor for dense grids}
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
            "sparse_apply_calls": 0,
            "sparse_apply_time_s_total": 0.0,
            "sparse_apply_calls_per_channel": {},
            "sparse_apply_time_s_total_per_channel": {},
            "grid_population_ops": 0,
            "vectorized_point_assign_ops": 0,
            "prebuilt_channel_copies": 0,
            "prebuilt_channels_active": 0,
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
        self._dense_storage = (
            getattr(config, "storage_mode", StorageMode.HYBRID) == StorageMode.DENSE
        )
        if self._dense_storage and self.dense_cache is None:
            self.dense_cache = torch.zeros(
                self.registry.num_channels,
                S,
                S,
                device=self.config.device,
                dtype=self.config.torch_dtype,
            )
            self.cache_dirty = False

        # High-frequency channel prebuild support
        high_freq_names: Set[str] = set(config.high_frequency_channels or [])
        self._high_freq_indices: Set[int] = set()
        for name in high_freq_names:
            try:
                idx = self.registry.get_index(name)
                self._high_freq_indices.add(int(idx))
            except KeyError:
                # Unknown channel name; ignore
                pass
        # Prebuilt per-channel dense slices for high-frequency channels
        self._prebuilt_dense: Dict[int, torch.Tensor] = {}
        if self._high_freq_indices:
            # Pre-allocate zero tensors lazily per channel when first used
            self._metrics["prebuilt_channels_active"] = len(self._high_freq_indices)

    def _ensure_dense_cache(self) -> None:
        """Ensure dense cache is initialized and has correct size."""
        S = 2 * self.config.R + 1
        num_channels = self.registry.num_channels
        expected_shape = (int(num_channels), int(S), int(S))
        if self.dense_cache is None:
            # Initialize without toggling cache_dirty; callers control rebuild semantics
            self.dense_cache = torch.zeros(
                expected_shape[0],
                expected_shape[1],
                expected_shape[2],
                device=self.config.device,
                dtype=self.config.torch_dtype,
            )
            return
        current_shape = tuple(int(d) for d in self.dense_cache.shape)
        if current_shape != expected_shape:
            self.dense_cache = torch.zeros(
                expected_shape[0],
                expected_shape[1],
                expected_shape[2],
                device=self.config.device,
                dtype=self.config.torch_dtype,
            )
            # Only mark dirty when we actually changed size
            self.cache_dirty = True

    def _store_sparse_point(self, channel_idx: int, y: int, x: int, value: float):
        """Store a single point value in sparse format."""
        if self._dense_storage:
            # Write directly into dense cache
            self._ensure_dense_cache()
            H, W = self.dense_cache.shape[-2:]
            if 0 <= y < H and 0 <= x < W:
                self.dense_cache[channel_idx, y, x] = value
            if self.config.enable_metrics:
                self._metrics["sparse_points_count"] += 1
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += 1
            return

        # If channel is marked high-frequency, update prebuilt dense slice instead
        if channel_idx in self._high_freq_indices:
            S = 2 * self.config.R + 1
            if channel_idx not in self._prebuilt_dense:
                self._prebuilt_dense[channel_idx] = torch.zeros(
                    S, S, device=self.config.device, dtype=self.config.torch_dtype
                )
            if 0 <= y < S and 0 <= x < S:
                self._prebuilt_dense[channel_idx][y, x] = value
            # Keep sparse mirror minimal to avoid rebuild cost
            self.sparse_channels.pop(channel_idx, None)
            self.cache_dirty = True
            if self.config.enable_metrics:
                self._metrics["sparse_points_count"] += 1
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += 1
            return

        # Non-high-frequency channels
        existing = self.sparse_channels.get(channel_idx)
        # If a dense grid is already stored for this channel, write directly
        if isinstance(existing, torch.Tensor):
            S = 2 * self.config.R + 1
            if 0 <= y < S and 0 <= x < S:
                existing[y, x] = value
            self.cache_dirty = True
            if self.config.enable_metrics:
                self._metrics["sparse_points_count"] += 1
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += 1
            return

        # Ensure SparsePoints backend for point storage
        if not isinstance(existing, SparsePoints):
            self.sparse_channels[channel_idx] = SparsePoints(
                self.config.device, self.config.torch_dtype
            )
        sp: SparsePoints = self.sparse_channels[channel_idx]
        sp.add_point(y, x, float(value))
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
        if self._dense_storage:
            self._ensure_dense_cache()
            H, W = self.dense_cache.shape[-2:]
            for y, x, value in points:
                if 0 <= y < H and 0 <= x < W:
                    if accumulate:
                        self.dense_cache[channel_idx, y, x] = max(
                            float(self.dense_cache[channel_idx, y, x].item()),
                            float(value),
                        )
                    else:
                        self.dense_cache[channel_idx, y, x] = value
            if self.config.enable_metrics:
                inc = len(points)
                self._metrics["sparse_points_count"] += inc
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += inc
            return

        if channel_idx in self._high_freq_indices:
            S = 2 * self.config.R + 1
            if channel_idx not in self._prebuilt_dense:
                self._prebuilt_dense[channel_idx] = torch.zeros(
                    S, S, device=self.config.device, dtype=self.config.torch_dtype
                )
            # Vectorized index put for prebuilt slice
            if points:
                ys, xs, vals = zip(*points)
                ys_t = torch.as_tensor(ys, device=self.config.device, dtype=torch.long)
                xs_t = torch.as_tensor(xs, device=self.config.device, dtype=torch.long)
                vals_t = torch.as_tensor(
                    vals, device=self.config.device, dtype=self.config.torch_dtype
                )
                # Clamp valid indices
                mask = (ys_t >= 0) & (ys_t < S) & (xs_t >= 0) & (xs_t < S)
                if mask.any().item():
                    ys_t = ys_t[mask]
                    xs_t = xs_t[mask]
                    vals_t = vals_t[mask]
                    if accumulate:
                        current = self._prebuilt_dense[channel_idx][ys_t, xs_t]
                        updated = torch.maximum(current, vals_t)
                        self._prebuilt_dense[channel_idx][ys_t, xs_t] = updated
                    else:
                        self._prebuilt_dense[channel_idx][ys_t, xs_t] = vals_t
            # Drop sparse mirror
            self.sparse_channels.pop(channel_idx, None)
            self.cache_dirty = True
            if self.config.enable_metrics:
                inc = len(points)
                self._metrics["sparse_points_count"] += inc
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += inc
            return

        # Non-high-frequency path
        existing = self.sparse_channels.get(channel_idx)
        # If dense grid stored, write directly with accumulate semantics
        if isinstance(existing, torch.Tensor):
            S = 2 * self.config.R + 1
            if points:
                ys, xs, vals = zip(*points)
                ys_t = torch.as_tensor(ys, device=self.config.device, dtype=torch.long)
                xs_t = torch.as_tensor(xs, device=self.config.device, dtype=torch.long)
                vals_t = torch.as_tensor(
                    vals, device=self.config.device, dtype=self.config.torch_dtype
                )
                mask = (ys_t >= 0) & (ys_t < S) & (xs_t >= 0) & (xs_t < S)
                if mask.any().item():
                    ys_t = ys_t[mask]
                    xs_t = xs_t[mask]
                    vals_t = vals_t[mask]
                    if accumulate:
                        current = existing[ys_t, xs_t]
                        updated = torch.maximum(current, vals_t)
                        existing[ys_t, xs_t] = updated
                    else:
                        existing[ys_t, xs_t] = vals_t
            self.cache_dirty = True
            if self.config.enable_metrics:
                inc = len(points)
                self._metrics["sparse_points_count"] += inc
                self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
                self._metrics["sparse_points_per_channel"][channel_idx] += inc
            return

        # Ensure SparsePoints backend for point storage
        if not isinstance(existing, SparsePoints):
            self.sparse_channels[channel_idx] = SparsePoints(
                self.config.device, self.config.torch_dtype
            )
        sp: SparsePoints = self.sparse_channels[channel_idx]
        sp.add_points(points)

        self.cache_dirty = True
        if self.config.enable_metrics:
            inc = len(points)
            self._metrics["sparse_points_count"] += inc
            self._metrics["sparse_points_per_channel"].setdefault(channel_idx, 0)
            self._metrics["sparse_points_per_channel"][channel_idx] += inc

    def _store_sparse_grid(self, channel_idx: int, grid: torch.Tensor):
        """Store a full grid (for dense channels like VISIBILITY, RESOURCES)."""
        if self._dense_storage:
            self._ensure_dense_cache()
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

        if channel_idx in self._high_freq_indices:
            # Maintain prebuilt slice
            self._prebuilt_dense[channel_idx] = grid.to(
                device=self.config.device, dtype=self.config.torch_dtype
            )
            # Remove sparse mirror to avoid duplicate work
            self.sparse_channels.pop(channel_idx, None)
        else:
            # Optionally compress sparse grids to point representation
            should_sparsify = bool(getattr(self.config, "grid_sparsify_enabled", True))
            store_obj = None
            if should_sparsify:
                S = 2 * int(self.config.R) + 1
                total = max(1, S * S)
                eps = float(getattr(self.config, "grid_zero_epsilon", 1e-12))
                # Count non-zero entries (treat small values as zeros)
                nnz = int((grid.abs() > eps).sum().item())
                density = nnz / float(total)
                threshold = float(getattr(self.config, "grid_sparsify_threshold", 0.5))
                if nnz > 0 and density < threshold:
                    # Build SparsePoints from non-zero cells
                    nz_idx = torch.nonzero(grid.abs() > eps, as_tuple=False)
                    sp = SparsePoints(self.config.device, self.config.torch_dtype)
                    # Ensure correct device/dtype
                    sp.indices = nz_idx.t().to(
                        device=self.config.device, dtype=torch.long
                    )
                    sp.values = grid[nz_idx[:, 0], nz_idx[:, 1]].to(
                        device=self.config.device, dtype=self.config.torch_dtype
                    )
                    # Tag sparsified grids with reduction hint from config
                    try:
                        sp._reduction = str(
                            getattr(self.config, "grid_sparse_reduction", "overwrite")
                        )
                    except Exception:
                        sp._reduction = "overwrite"
                    store_obj = sp
            # Fallback to dense grid storage
            if store_obj is None:
                store_obj = grid
            self.sparse_channels[channel_idx] = store_obj
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
        # Also clear any prebuilt slice
        if channel_idx in self._prebuilt_dense:
            self._prebuilt_dense[channel_idx].zero_()

    def _decay_sparse_channel(self, channel_idx: int, decay_factor: float):
        """Apply decay to a sparse channel."""
        if channel_idx in self._prebuilt_dense:
            # Apply decay directly to prebuilt slice
            self._prebuilt_dense[channel_idx] *= decay_factor
            self.cache_dirty = True
            return
        if channel_idx in self.sparse_channels:
            channel_data = self.sparse_channels[channel_idx]
            if isinstance(channel_data, SparsePoints):
                channel_data.decay(decay_factor)
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
        if self._dense_storage:
            if self.config.enable_metrics:
                self._metrics["cache_hits"] += 1
            if self.dense_cache is not None:
                return self.dense_cache
            self._ensure_dense_cache()
            return self.dense_cache

        if not self.cache_dirty and self.dense_cache is not None:
            if self.config.enable_metrics:
                self._metrics["cache_hits"] += 1
            return self.dense_cache

        S = 2 * self.config.R + 1
        num_channels = self.registry.num_channels

        # Create dense tensor
        if self.dense_cache is None:
            self._ensure_dense_cache()

        # Clear existing values and record rebuild time
        t0 = None
        if self.config.enable_metrics:
            t0 = _time.perf_counter()
        self.dense_cache.zero_()

        # 1) Copy prebuilt high-frequency channels
        if self._prebuilt_dense:
            for channel_idx, grid in self._prebuilt_dense.items():
                if grid is not None:
                    self.dense_cache[int(channel_idx)].copy_(grid)
                    if self.config.enable_metrics:
                        self._metrics["prebuilt_channel_copies"] += 1
                        self._metrics["grid_population_ops"] += 1

        # 2) Populate remaining from sparse data
        for channel_idx, channel_data in self.sparse_channels.items():

            # Skip if this channel is handled by prebuilt
            if (
                channel_idx in self._high_freq_indices
                and channel_idx in self._prebuilt_dense
            ):
                continue
            if isinstance(channel_data, dict):
                # Vectorized sparse points assignment
                if channel_data:
                    coords = list(channel_data.keys())
                    if coords:
                        ys, xs = zip(*coords)
                        vals = [channel_data[(y, x)] for (y, x) in coords]
                        ys_t = torch.as_tensor(
                            ys, device=self.config.device, dtype=torch.long
                        )
                        xs_t = torch.as_tensor(
                            xs, device=self.config.device, dtype=torch.long
                        )
                        vals_t = torch.as_tensor(
                            vals,
                            device=self.config.device,
                            dtype=self.config.torch_dtype,
                        )
                        # Clamp valid indices
                        mask = (ys_t >= 0) & (ys_t < S) & (xs_t >= 0) & (xs_t < S)
                        if mask.any().item():
                            ys_t = ys_t[mask]
                            xs_t = xs_t[mask]
                            vals_t = vals_t[mask]
                            self.dense_cache[channel_idx, ys_t, xs_t] = vals_t
                            if self.config.enable_metrics:
                                self._metrics["vectorized_point_assign_ops"] += 1

            elif isinstance(channel_data, SparsePoints):
                # Sparse points
                channel_plane = self.dense_cache[channel_idx]
                # Determine reduction per channel
                # Prefer per-object reduction if provided (e.g., sparsified full grids)
                reduction = getattr(channel_data, "_reduction", None)
                if not reduction:
                    handlers = self.registry.get_all_handlers()
                    channel_name = None
                    for name, _handler in handlers.items():
                        if self.registry.get_index(name) == channel_idx:
                            channel_name = name
                            break
                    reduction = self.config.channel_reduction_overrides.get(
                        channel_name or "", self.config.default_point_reduction
                    )
                backend = self.config.sparse_backend
                before_calls = channel_data._apply_calls
                before_time = channel_data._apply_time_s_total
                channel_data.apply_to_dense(
                    channel_plane, reduction=reduction, backend=backend
                )
                # accumulate metrics
                delta_calls = channel_data._apply_calls - before_calls
                delta_time = channel_data._apply_time_s_total - before_time
                self._metrics["sparse_apply_calls"] += int(delta_calls)
                self._metrics["sparse_apply_time_s_total"] += float(delta_time)
                self._metrics["sparse_apply_calls_per_channel"].setdefault(
                    channel_idx, 0
                )
                self._metrics["sparse_apply_time_s_total_per_channel"].setdefault(
                    channel_idx, 0.0
                )
                self._metrics["sparse_apply_calls_per_channel"][channel_idx] += int(
                    delta_calls
                )
                self._metrics["sparse_apply_time_s_total_per_channel"][
                    channel_idx
                ] += float(delta_time)

            else:
                # Dense grid (VISIBILITY, RESOURCES, etc.)
                self.dense_cache[channel_idx] = channel_data
                if self.config.enable_metrics:
                    self._metrics["grid_population_ops"] += 1

        self.cache_dirty = False
        if self.config.enable_metrics:
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
        agent_orientation: float = 0.0,
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
            agent_orientation: Agent's facing orientation in degrees (clockwise). 0=north/up.
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
            "agent_orientation": float(agent_orientation or 0.0),
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
        for channel_idx, channel_data in self.sparse_channels.items():
            if isinstance(channel_data, SparsePoints):
                sparse_points += len(channel_data)
            else:
                try:
                    sparse_points += int((channel_data != 0).sum().item())
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.warning(
                        "sparse_points_count_failed",
                        channel=channel_idx,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    continue
        # Include points counted during dense baseline writes
        if self._dense_storage:
            sparse_points = max(
                sparse_points, int(self._metrics.get("sparse_points_count", 0))
            )

        # Approximate sparse logical memory: value + y + x (floatsize + 2*int32)
        sparse_logical_bytes = int(sparse_points * (dtype_size + 4 + 4))

        cache_hits = int(self._metrics.get("cache_hits", 0))
        cache_misses = int(self._metrics.get("cache_misses", 0))
        total = cache_hits + cache_misses
        hit_rate = (cache_hits / total) if total > 0 else 1.0
        reduction = (
            1.0 - (sparse_logical_bytes / dense_bytes) if dense_bytes > 0 else 0.0
        )

        return {
            "dense_bytes": dense_bytes,
            "sparse_points": sparse_points,
            "sparse_logical_bytes": sparse_logical_bytes,
            "memory_reduction_percent": max(0.0, reduction * 100.0),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
            "dense_rebuilds": int(self._metrics.get("dense_rebuilds", 0)),
            "dense_rebuild_time_s_total": float(
                self._metrics.get("dense_rebuild_time_s_total", 0.0)
            ),
            "grid_population_ops": int(self._metrics.get("grid_population_ops", 0)),
            "vectorized_point_assign_ops": int(
                self._metrics.get("vectorized_point_assign_ops", 0)
            ),
            "prebuilt_channel_copies": int(
                self._metrics.get("prebuilt_channel_copies", 0)
            ),
            "prebuilt_channels_active": int(
                self._metrics.get("prebuilt_channels_active", 0)
            ),
            "sparse_apply_calls": int(self._metrics.get("sparse_apply_calls", 0)),
            "sparse_apply_time_s_total": float(
                self._metrics.get("sparse_apply_time_s_total", 0.0)
            ),
        }
