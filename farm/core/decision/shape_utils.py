"""Shared observation-shape normalization for the decision pipeline.

Both :class:`farm.core.decision.decision.DecisionModule` and
:class:`farm.core.decision.algorithms.tianshou.TianshouWrapper` need to coerce
an incoming observation (1D vector, raw spatial tensor, already-batched
tensor, etc.) into a single canonical batched-numpy form before calling the
underlying policy network.  Keeping that logic in one place prevents the two
sites from drifting on edge cases like ``(C, H, W)`` vs ``(N, C, H, W)`` or
flat ``(D,)`` vs already-batched ``(1, D)``.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np


def _to_float32_numpy(state: Any) -> np.ndarray:
    """Coerce ``state`` to a ``float32`` numpy array without unnecessary copies.

    Handles torch tensors lazily so this helper can be imported without
    requiring torch to be installed at module-load time.
    """
    if isinstance(state, np.ndarray):
        return state.astype(np.float32, copy=False)
    try:
        import torch  # local import keeps torch optional at import time

        if isinstance(state, torch.Tensor):
            return state.detach().cpu().numpy().astype(np.float32, copy=False)
    except ImportError:
        pass
    return np.asarray(state, dtype=np.float32)


def batch_observation(
    state: Any,
    observation_shape: Sequence[int] | Tuple[int, ...],
) -> np.ndarray:
    """Return ``state`` reshaped to ``(1, *observation_shape)``-style batched form.

    Args:
        state: Raw observation as a numpy array, torch tensor, or array-like.
        observation_shape: Per-sample shape the policy network expects.

    Returns:
        Float32 numpy array with a leading batch dimension.  The function is
        intentionally tolerant of common input shapes:

        * 1D vector matching ``prod(observation_shape)`` reshapes into the
          full multi-dimensional layout when ``observation_shape`` has rank
          > 1.
        * 2D input matching ``observation_shape`` exactly gets a batch axis
          prepended.
        * Already-batched inputs (leading singleton dim) pass through.
    """
    state_np = _to_float32_numpy(state)
    obs_shape = tuple(observation_shape)
    target_size = int(np.prod(obs_shape)) if obs_shape else state_np.size

    if state_np.ndim == 1:
        if state_np.size == target_size and len(obs_shape) > 1:
            return state_np.reshape(1, *obs_shape)
        return state_np.reshape(1, -1)
    if state_np.ndim == 2:
        if state_np.shape == obs_shape:
            return state_np[np.newaxis, ...]
        if state_np.size == target_size:
            return state_np.reshape(1, *obs_shape)
        if state_np.shape[0] == 1:
            return state_np
        return state_np.reshape(1, -1)
    if state_np.ndim == len(obs_shape):
        return state_np[np.newaxis, ...]
    if state_np.ndim == len(obs_shape) + 1:
        return state_np
    return state_np.reshape(1, -1)
