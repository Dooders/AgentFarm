"""Utilities for train/holdout splits and domain-shift state perturbations.

For publication-grade generalisation claims, evaluation must be performed on
data that was **not** used for training or calibration.  This module provides:

- :func:`split_replay_buffer` – split a flat ``(N, D)`` NumPy array into two
  non-overlapping subsets (in-distribution / holdout).
- :func:`apply_gaussian_noise` – add i.i.d. Gaussian noise to every feature
  to simulate sensor jitter or a mild distribution shift.
- :func:`apply_input_scaling` – multiply all features by a scalar factor to
  simulate a calibration change or units mismatch.
- :func:`make_shifted_states` – convenience factory that dispatches on a
  ``shift_type`` key (``"gaussian_noise"`` or ``"input_scaling"``).

Typical usage
-------------
::

    import numpy as np
    from farm.core.decision.training.holdout_utils import (
        split_replay_buffer,
        apply_gaussian_noise,
        apply_input_scaling,
        make_shifted_states,
    )

    all_states = np.load("data/replay_states.npy")

    # 80 / 20 in-distribution / holdout split
    id_states, holdout_states = split_replay_buffer(all_states, holdout_fraction=0.2, seed=42)

    # Domain-shift variants of the holdout set
    noisy_states  = apply_gaussian_noise(holdout_states, std=0.1, seed=0)
    scaled_states = apply_input_scaling(holdout_states, scale_factor=2.0)

    # Or use the factory:
    shifted = make_shifted_states(holdout_states, "gaussian_noise", std=0.1, seed=0)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "SHIFT_TYPES",
    "apply_gaussian_noise",
    "apply_input_scaling",
    "make_shifted_states",
    "split_replay_buffer",
]

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _require_2d_float32(states: np.ndarray, name: str = "states") -> np.ndarray:
    """Return *states* as a float32 2-D array; raise :class:`ValueError` if invalid."""
    arr = np.asarray(states, dtype="float32")
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array with shape (N, input_dim); got shape {arr.shape!r}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty; got 0 rows.")
    return arr


# ---------------------------------------------------------------------------
# Train / holdout split
# ---------------------------------------------------------------------------


def split_replay_buffer(
    states: np.ndarray,
    holdout_fraction: float = 0.2,
    *,
    seed: int | None = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split *states* into two non-overlapping subsets.

    The buffer is randomly shuffled (seeded for reproducibility) before
    splitting so that the holdout set is an i.i.d. draw from the same
    distribution as the training set.

    Parameters
    ----------
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
        Non-float32 inputs are cast automatically.
    holdout_fraction:
        Fraction of rows reserved for the holdout / test set.  Must be in
        ``(0, 1)``.  The remaining ``1 - holdout_fraction`` rows form the
        in-distribution (training / calibration) split.
    seed:
        Optional integer seed for the shuffle RNG.  Pass the same seed on
        every run to obtain identical splits.
    shuffle:
        When ``True`` (default), shuffle *states* before splitting.  Set to
        ``False`` if rows are already in random order and you want to preserve
        temporal adjacency within each split.

    Returns
    -------
    id_states : np.ndarray
        In-distribution subset of shape ``(N_train, input_dim)``.
    holdout_states : np.ndarray
        Holdout subset of shape ``(N_holdout, input_dim)``.

    Raises
    ------
    ValueError
        If *states* is not a non-empty 2-D array, or if *holdout_fraction* is
        not strictly in ``(0, 1)``, or if either split would be empty.
    """
    arr = _require_2d_float32(states)
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(
            f"holdout_fraction must be in (0, 1); got {holdout_fraction!r}"
        )

    n_total = arr.shape[0]
    n_holdout = max(1, int(round(n_total * holdout_fraction)))
    n_train = n_total - n_holdout

    if n_train == 0:
        raise ValueError(
            f"All {n_total} row(s) would be allocated to the holdout set with "
            f"holdout_fraction={holdout_fraction!r}.  Provide more data or reduce the fraction."
        )
    if n_holdout == 0:  # pragma: no cover – guarded by the max(1, …) above
        raise ValueError(
            f"holdout_fraction={holdout_fraction!r} is too small for {n_total} rows."
        )

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
    else:
        indices = np.arange(n_total)

    id_states = arr[indices[:n_train]]
    holdout_states = arr[indices[n_train:]]
    return id_states, holdout_states


# ---------------------------------------------------------------------------
# Domain-shift perturbations
# ---------------------------------------------------------------------------


def apply_gaussian_noise(
    states: np.ndarray,
    std: float = 0.1,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a copy of *states* with i.i.d. Gaussian noise added to each element.

    This simulates sensor jitter, observation noise, or mild distribution
    shift between training and deployment environments.

    Parameters
    ----------
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
    std:
        Standard deviation of the Gaussian noise.  Must be non-negative.
        ``std=0`` returns an unmodified copy.
    seed:
        Optional integer seed for the noise RNG.  Use the same seed across
        experiments for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, input_dim)`` with ``dtype=float32``.

    Raises
    ------
    ValueError
        If *states* is not a non-empty 2-D array, or if *std* is negative.
    """
    arr = _require_2d_float32(states)
    if std < 0.0:
        raise ValueError(f"std must be non-negative; got {std!r}")
    if std == 0.0:
        return arr.copy()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(arr.shape).astype("float32") * float(std)
    return arr + noise


def apply_input_scaling(
    states: np.ndarray,
    scale_factor: float = 2.0,
) -> np.ndarray:
    """Return a copy of *states* with every element multiplied by *scale_factor*.

    This simulates a calibration change or units mismatch between the
    environment used for training and the deployment environment.

    Parameters
    ----------
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
    scale_factor:
        Scalar multiplier.  Must be finite.  ``scale_factor=1.0`` returns an
        unmodified copy.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, input_dim)`` with ``dtype=float32``.

    Raises
    ------
    ValueError
        If *states* is not a non-empty 2-D array, or if *scale_factor* is not
        a finite number.
    """
    arr = _require_2d_float32(states)
    sf = float(scale_factor)
    if not np.isfinite(sf):
        raise ValueError(f"scale_factor must be finite; got {scale_factor!r}")
    return (arr * sf).astype("float32")


# ---------------------------------------------------------------------------
# Factory dispatcher
# ---------------------------------------------------------------------------

#: Supported shift types for :func:`make_shifted_states`.
SHIFT_TYPES = ("gaussian_noise", "input_scaling")


def make_shifted_states(
    states: np.ndarray,
    shift_type: str,
    **kwargs,
) -> np.ndarray:
    """Apply a named domain-shift perturbation to *states*.

    This convenience factory dispatches to :func:`apply_gaussian_noise` or
    :func:`apply_input_scaling` based on *shift_type*.

    Parameters
    ----------
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
    shift_type:
        One of ``"gaussian_noise"`` or ``"input_scaling"``.
    **kwargs:
        Forwarded to the underlying function.  For ``"gaussian_noise"``:
        ``std`` (default ``0.1``) and ``seed`` (default ``None``).
        For ``"input_scaling"``: ``scale_factor`` (default ``2.0``).

    Returns
    -------
    np.ndarray
        Perturbed state array with the same shape and ``dtype=float32``.

    Raises
    ------
    ValueError
        If *shift_type* is not one of the supported values.

    Examples
    --------
    >>> shifted = make_shifted_states(states, "gaussian_noise", std=0.2, seed=99)
    >>> scaled  = make_shifted_states(states, "input_scaling", scale_factor=0.5)
    """
    if shift_type == "gaussian_noise":
        return apply_gaussian_noise(states, **kwargs)
    if shift_type == "input_scaling":
        return apply_input_scaling(states, **kwargs)
    raise ValueError(
        f"Unknown shift_type {shift_type!r}. "
        f"Supported values: {list(SHIFT_TYPES)}"
    )
