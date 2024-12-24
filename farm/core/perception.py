from enum import IntEnum
from typing import Optional

import numpy as np
import torch


class PerceptionContent(IntEnum):
    """Enumeration of possible cell contents in agent perception."""

    EMPTY = 0
    RESOURCE = 1
    AGENT = 2
    OBSTACLE = 3


class PerceptionData:
    """Wrapper for perception grid data with semantic meaning and tensor conversion.

    Attributes:
        grid (np.ndarray): 2D numpy array storing perception values
        size (int): Size of the perception grid (assumes square)
        radius (int): Perception radius (derived from size)
    """

    def __init__(self, grid: np.ndarray):
        """Initialize perception data with a grid.

        Args:
            grid (np.ndarray): 2D numpy array of perception values
        """
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError("Grid must be a 2D numpy array")

        self.grid = grid
        self.size = grid.shape[0]
        self.radius = (self.size - 1) // 2

        # Validate grid contents
        valid_values = set(item.value for item in PerceptionContent)
        if not set(np.unique(grid)).issubset(valid_values):
            raise ValueError(
                f"Grid contains invalid values. Must be one of {valid_values}"
            )

    def __getitem__(self, key):
        return self.grid[key]

    def __array__(self) -> np.ndarray:
        return self.grid

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert perception grid to flattened tensor for neural network input.

        Args:
            device: Optional torch device to place tensor on

        Returns:
            torch.Tensor: 1D tensor of perception values
        """
        tensor = torch.from_numpy(self.grid).float()
        if device:
            tensor = tensor.to(device)
        return tensor.flatten()

    def count_content(self, content: PerceptionContent) -> int:
        """Count occurrences of specific content type in perception.

        Args:
            content (PerceptionContent): Content type to count

        Returns:
            int: Number of cells containing specified content
        """
        return np.sum(self.grid == content.value)

    def get_content_positions(
        self, content: PerceptionContent
    ) -> list[tuple[int, int]]:
        """Get grid positions of specific content type.

        Args:
            content (PerceptionContent): Content type to locate

        Returns:
            list[tuple[int, int]]: List of (row, col) positions
        """
        positions = np.where(self.grid == content.value)
        return list(zip(positions[0], positions[1]))
