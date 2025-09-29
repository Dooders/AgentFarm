import math
from typing import Tuple


def discretize_position_continuous(
    position: Tuple[float, float], grid_size: Tuple[int, int], method: str = "floor"
) -> Tuple[int, int]:
    """
    Convert continuous position to discrete grid coordinates using specified method.

    Args:
        position: (x, y) continuous coordinates
        grid_size: (width, height) of the grid
        method: Discretization method - "floor", "round", or "ceil"

    Returns:
        (x_idx, y_idx) discrete grid coordinates
    """
    x, y = position
    width, height = grid_size

    if method == "round":
        x_idx = max(0, min(int(round(x)), width - 1))
        y_idx = max(0, min(int(round(y)), height - 1))
    elif method == "ceil":
        x_idx = max(0, min(int(math.ceil(x)), width - 1))
        y_idx = max(0, min(int(math.ceil(y)), height - 1))
    else:  # "floor" (default)
        x_idx = max(0, min(int(math.floor(x)), width - 1))
        y_idx = max(0, min(int(math.floor(y)), height - 1))

    return x_idx, y_idx

