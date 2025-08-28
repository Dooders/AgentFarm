"""Base classes for reinforcement learning algorithms.

This module provides abstract base classes and interfaces for integrating
reinforcement learning algorithms with the AgentFarm action system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch

from .base import ActionAlgorithm


class RLAlgorithm(ActionAlgorithm, ABC):
    """Abstract base class for reinforcement learning algorithms.

    This class extends ActionAlgorithm with RL-specific methods for experience
    replay, training on batches of experiences, and model management. It provides
    a unified interface for integrating various RL algorithms (PPO, SAC, DQN, etc.)
    with the AgentFarm action selection system.
    """

    def __init__(self, num_actions: int, **kwargs: Any) -> None:
        """Initialize the RL algorithm.

        Args:
            num_actions: Number of possible actions
            **kwargs: Algorithm-specific parameters
        """
        super().__init__(num_actions=num_actions, **kwargs)
        self._step_count = 0

        # Abstract attributes that concrete implementations should define
        self.replay_buffer: Any = None
        self.batch_size: int = 32

    @abstractmethod
    def store_experience(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        **kwargs: Any,
    ) -> None:
        """Store a single experience in the replay buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended
            **kwargs: Additional experience data (e.g., log_prob for PPO)
        """
        pass

    @abstractmethod
    def train_on_batch(self, batch: Any, **kwargs: Any) -> Dict[str, float]:
        """Train the algorithm on a batch of experiences.

        Args:
            batch: Batch of experiences (format depends on algorithm)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics (e.g., loss, value_loss, etc.)
        """
        pass

    @abstractmethod
    def should_train(self) -> bool:
        """Check if the algorithm should train on the current step.

        Returns:
            True if training should occur, False otherwise
        """
        pass

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state for saving/checkpointing.

        Returns:
            Dictionary containing model state and training progress
        """
        pass

    @abstractmethod
    def load_model_state(self, state: Dict[str, Any]) -> None:
        """Load a saved model state.

        Args:
            state: Model state dictionary from get_model_state()
        """
        pass

    def update_step_count(self) -> None:
        """Increment the step counter."""
        self._step_count += 1

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count


class ExperienceReplayBuffer(ABC):
    """Abstract base class for experience replay buffers."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        pass

    @abstractmethod
    def append(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        **kwargs: Any,
    ) -> None:
        """Add an experience to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Sample a batch of experiences from the buffer."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        pass


class SimpleReplayBuffer(ExperienceReplayBuffer):
    """Simple FIFO experience replay buffer."""

    def __init__(self, max_size: int = 10000):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer: list = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def append(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        **kwargs: Any,
    ) -> None:
        """Add an experience to the buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            **kwargs,
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> Dict[str, Union[list, np.ndarray]]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer ({len(self.buffer)}) for batch size {batch_size}"
            )

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch: Dict[str, Union[list, np.ndarray]] = {
            key: [] for key in self.buffer[0].keys()
        }

        for idx in indices:
            for key, value in self.buffer[idx].items():
                # At this point, all values are still lists
                cast(list, batch[key]).append(value)

        # Convert lists to appropriate arrays
        for key in batch:
            if key in ["state", "next_state"]:
                batch[key] = np.array(batch[key])
            elif key in ["action", "done"]:
                batch[key] = np.array(batch[key])
            elif key == "reward":
                batch[key] = np.array(batch[key], dtype=np.float32)
            else:
                # Keep other keys as lists or convert to numpy arrays
                try:
                    batch[key] = np.array(batch[key])
                except:
                    pass  # Keep as list if conversion fails

        return batch

    def clear(self) -> None:
        """Clear all experiences."""
        self.buffer.clear()
        self.position = 0
