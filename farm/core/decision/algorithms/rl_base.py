"""Base classes for reinforcement learning algorithms.

This module provides abstract base classes and interfaces for integrating
reinforcement learning algorithms with the AgentFarm action system.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import torch

from .base import ActionAlgorithm

logger = logging.getLogger(__name__)


class RLAlgorithm(ActionAlgorithm, ABC):
    """Abstract base class for reinforcement learning algorithms.

    This class extends ActionAlgorithm with RL-specific methods for experience
    replay, training on batches of experiences, and model management. It provides
    a unified interface for integrating various RL algorithms (PPO, SAC, DQN, etc.)
    with the AgentFarm action selection system.

    Note: This class remains abstract as it doesn't implement ActionAlgorithm methods.
    Concrete RL implementations should inherit from this class and implement all
    required abstract methods.
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
    """Abstract base class for experience replay buffers.

    This class defines the interface for experience replay functionality.
    Concrete implementations like SimpleReplayBuffer should inherit from this class
    and implement all required abstract methods.
    """

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
                except Exception:
                    pass  # Keep as list if conversion fails

        return batch

    def clear(self) -> None:
        """Clear all experiences."""
        self.buffer.clear()
        self.position = 0


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """Experience replay buffer with Prioritized Experience Replay (PER).

    Implements the proportional variant of PER from Schaul et al. (2015)
    (https://arxiv.org/abs/1511.05952).

    Transitions are sampled with probability proportional to ``priority^alpha``.
    Importance-sampling (IS) weights are returned to correct for the introduced bias.
    The IS exponent ``beta`` can be annealed toward 1.0 over training to remove bias
    gradually.

    When ``replay_strategy`` is ``"uniform"`` the buffer degrades gracefully to
    uniform sampling (identical behaviour to ``SimpleReplayBuffer``) without
    returning indices or IS weights.

    Args:
        max_size: Maximum number of experiences to store.
        alpha: Exponent that controls how much prioritisation is used
            (0 = uniform, 1 = full prioritisation).
        beta_start: Initial exponent for IS weight correction (0 = no correction,
            1 = full correction).
        beta_end: Final value of ``beta`` reached after ``beta_steps`` updates.
        beta_steps: Number of ``update_beta`` calls over which ``beta`` anneals
            from ``beta_start`` to ``beta_end``.
        epsilon: Small constant added to priorities to ensure every transition
            has a non-zero sampling probability.
        replay_strategy: ``"prioritized"`` (default) or ``"uniform"`` for
            ablation / debugging.
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        epsilon: float = 1e-6,
        replay_strategy: Literal["prioritized", "uniform"] = "prioritized",
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")
        if not 0.0 <= beta_end <= 1.0:
            raise ValueError(f"beta_end must be in [0, 1], got {beta_end}")
        if beta_steps <= 0:
            raise ValueError(f"beta_steps must be positive, got {beta_steps}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if replay_strategy not in ("prioritized", "uniform"):
            raise ValueError(
                f"replay_strategy must be 'prioritized' or 'uniform', got {replay_strategy!r}"
            )

        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.replay_strategy = replay_strategy

        self.buffer: List[Dict[str, Any]] = []
        self.priorities: np.ndarray = np.zeros(max_size, dtype=np.float64)
        self.position = 0
        self._beta_step_count = 0

    # ------------------------------------------------------------------
    # ExperienceReplayBuffer interface
    # ------------------------------------------------------------------

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
        """Add an experience to the buffer.

        New transitions are assigned the current maximum priority (or 1.0 if
        the buffer is empty) so that they are sampled at least once.
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            **kwargs,
        }

        max_priority = float(self.priorities[: len(self.buffer)].max()) if self.buffer else 1.0

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of experiences.

        Both strategies (``"prioritized"`` and ``"uniform"``) always include the
        following extra keys in the returned batch:

        * ``"indices"`` – 1-D integer array of sampled buffer indices, needed
          to call :meth:`update_priorities` after computing TD errors.
        * ``"is_weights"`` – 1-D float32 array of importance-sampling weights
          (normalised to ``[0, 1]``), to be multiplied into the loss.

        When ``replay_strategy`` is ``"uniform"``, ``"is_weights"`` contains all
        1.0 values (no bias correction needed).  When ``replay_strategy`` is
        ``"prioritized"``, weights are computed as
        ``(N * P(i))^{-beta} / max_j(w_j)`` to down-weight over-represented
        high-priority transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch dictionary as described above.

        Raises:
            ValueError: If the buffer contains fewer experiences than
                ``batch_size``.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer ({len(self.buffer)}) for batch size {batch_size}"
            )

        n = len(self.buffer)
        if self.replay_strategy == "uniform":
            indices = np.random.choice(n, batch_size, replace=False)
            is_weights = np.ones(batch_size, dtype=np.float32)
        else:
            priorities = self.priorities[:n]
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(n, batch_size, replace=False, p=probs)
            # IS weights: w_i = (N * P(i))^{-beta} / max_j w_j
            weights = (n * probs[indices]) ** (-self.beta)
            is_weights = (weights / weights.max()).astype(np.float32)

        batch: Dict[str, Any] = {key: [] for key in self.buffer[0].keys()}
        for idx in indices:
            for key, value in self.buffer[idx].items():
                cast(list, batch[key]).append(value)

        # Convert lists to arrays (mirrors SimpleReplayBuffer behaviour)
        for key in list(batch.keys()):
            if key in ("state", "next_state"):
                batch[key] = np.array(batch[key])
            elif key in ("action", "done"):
                batch[key] = np.array(batch[key])
            elif key == "reward":
                batch[key] = np.array(batch[key], dtype=np.float32)
            else:
                try:
                    batch[key] = np.array(batch[key])
                except Exception:
                    pass  # Keep as list if conversion fails

        batch["indices"] = indices
        batch["is_weights"] = is_weights
        return batch

    def clear(self) -> None:
        """Clear all experiences and reset priorities."""
        self.buffer.clear()
        self.priorities[:] = 0.0
        self.position = 0
        self._beta_step_count = 0
        self.beta = self.beta_start

    # ------------------------------------------------------------------
    # PER-specific API
    # ------------------------------------------------------------------

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update transition priorities from TD errors.

        Should be called after each training step using the ``"indices"`` key
        returned by :meth:`sample`.

        Args:
            indices: 1-D integer array of buffer indices (from the ``"indices"``
                key returned by :meth:`sample`).
            td_errors: 1-D array of absolute TD errors for the corresponding
                transitions. Scalar values and shapes that broadcast against
                ``indices`` are accepted.
        """
        indices_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices_arr.size == 0:
            return

        td_errors_arr = np.asarray(td_errors, dtype=np.float64)
        try:
            td_errors_arr = np.broadcast_to(td_errors_arr, indices_arr.shape)
        except ValueError as exc:
            raise ValueError(
                f"td_errors shape {td_errors_arr.shape} is not broadcastable to "
                f"indices shape {indices_arr.shape}"
            ) from exc

        new_priorities = np.abs(td_errors_arr) + self.epsilon
        for idx, priority in zip(indices_arr, new_priorities):
            self.priorities[idx] = float(priority)

    def update_beta(self) -> float:
        """Anneal ``beta`` by one step toward ``beta_end``.

        Call this once per training iteration (not per gradient step).

        Returns:
            The updated ``beta`` value.
        """
        self._beta_step_count += 1
        fraction = min(1.0, self._beta_step_count / self.beta_steps)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        return self.beta

    def diagnostics(self) -> Dict[str, float]:
        """Return a snapshot of internal priority statistics for logging.

        Returns:
            Dictionary with keys: ``priority_min``, ``priority_max``,
            ``priority_mean``, ``beta``, ``buffer_size``.
        """
        n = len(self.buffer)
        if n == 0:
            return {
                "priority_min": 0.0,
                "priority_max": 0.0,
                "priority_mean": 0.0,
                "beta": self.beta,
                "buffer_size": 0,
            }
        priorities = self.priorities[:n]
        return {
            "priority_min": float(priorities.min()),
            "priority_max": float(priorities.max()),
            "priority_mean": float(priorities.mean()),
            "beta": self.beta,
            "buffer_size": n,
        }
