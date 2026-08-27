"""Post-election budget allocation by the winning steward."""

import numpy as np

BLEND_DIRECTED = 0.72
BLEND_PLATFORM = 0.28


def allocate(
    benefits: np.ndarray,
    supporters: np.ndarray,
    winner_platform: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Allocate the unit budget across projects.

    ``lam = 1`` targets the mean benefit direction of electoral supporters only;
    ``lam = 0`` targets the mean benefit direction of everyone. The directed
    allocation is then blended with the winner's (clipped) platform and
    normalized to sum to 1. A winner with zero supporters uses the
    everyone-direction only.
    """
    dir_all = benefits.mean(axis=0)
    if supporters.any():
        dir_supporters = benefits[supporters].mean(axis=0)
        raw = lam * dir_supporters + (1.0 - lam) * dir_all
    else:
        raw = dir_all
    raw = BLEND_DIRECTED * raw + BLEND_PLATFORM * np.clip(winner_platform, 0.0, None)
    return raw / raw.sum()
