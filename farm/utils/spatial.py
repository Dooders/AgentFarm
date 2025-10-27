"""Spatial utility functions for AgentFarm.

This module provides mathematical and spatial computation utilities including
bilinear interpolation, distance calculations, and other spatial operations.
"""

import math
from typing import Tuple

import torch


def bilinear_distribute_value(
    position: Tuple[float, float],
    value: float,
    grid: torch.Tensor,
    grid_size: Tuple[int, int],
) -> None:
    """
    Distribute a value across grid cells using bilinear interpolation.

    This preserves continuous position information by distributing values
    across the four nearest grid cells based on the fractional position components.

    Args:
        position: (x, y) continuous coordinates
        value: Value to distribute
        grid: Target grid tensor of shape (H, W)
        grid_size: (width, height) of the grid
    """
    x, y = position
    width, height = grid_size

    # Get the four nearest grid cells
    x_floor = int(math.floor(x))
    y_floor = int(math.floor(y))

    # Ensure we don't go out of bounds first
    x_floor = max(0, min(x_floor, width - 1))
    y_floor = max(0, min(y_floor, height - 1))

    # Calculate ceil positions after clamping floor
    x_ceil = min(x_floor + 1, width - 1)
    y_ceil = min(y_floor + 1, height - 1)

    # Calculate interpolation weights using clamped floor positions
    x_frac = x - x_floor
    y_frac = y - y_floor

    # Bilinear interpolation weights
    w00 = (1 - x_frac) * (1 - y_frac)  # bottom-left
    w01 = (1 - x_frac) * y_frac  # top-left
    w10 = x_frac * (1 - y_frac)  # bottom-right
    w11 = x_frac * y_frac  # top-right

    # Distribute the value
    grid[y_floor, x_floor] += value * w00
    grid[y_ceil, x_floor] += value * w01
    grid[y_floor, x_ceil] += value * w10
    grid[y_ceil, x_ceil] += value * w11
