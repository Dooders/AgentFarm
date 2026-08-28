"""Post-election budget allocation by the winning steward."""

from __future__ import annotations

import numpy as np

BLEND_DIRECTED = 0.72
BLEND_PLATFORM = 0.28


def _as_simplex(vector: np.ndarray) -> np.ndarray | None:
    """Clip to the nonnegative orthant and renormalize; ``None`` if mass is 0."""
    clipped = np.clip(vector, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return None
    return clipped / total


def allocate(
    benefits: np.ndarray,
    supporters: np.ndarray,
    winner_platform: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Allocate the unit budget across projects.

    ``lam = 1`` targets the mean benefit direction of electoral supporters only;
    ``lam = 0`` targets the mean benefit direction of everyone. The directed
    mix (already a probability vector) is then blended with the winner's
    platform after that platform is clipped and renormalized onto the simplex,
    so the 0.72 / 0.28 weights are a convex combination of matched units.
    A winner with zero supporters uses the everyone-direction only. A platform
    with no positive mass is dropped (directed-only).
    """
    if not (np.isfinite(lam) and 0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be a finite value in [0, 1], got {lam!r}")
    dir_all = benefits.mean(axis=0)
    if supporters.any():
        dir_supporters = benefits[supporters].mean(axis=0)
        directed = lam * dir_supporters + (1.0 - lam) * dir_all
    else:
        directed = dir_all
    platform = _as_simplex(winner_platform)
    if platform is None:
        raw = directed
    else:
        raw = BLEND_DIRECTED * directed + BLEND_PLATFORM * platform
    return raw / raw.sum()
