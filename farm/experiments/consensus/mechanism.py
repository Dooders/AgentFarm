"""Incentive mechanisms that make λ (or platforms) chosen, not merely drawn.

The default one-shot cell draws λ independently of anything voters see and
runs a single sincere election. That cell can compare selection *rules* with
exogenous types. It cannot test theories in which loyalty is a choice.

This module adds a first-class re-election cell: after the first-round
election fixes the winner and the supporter mask, the incumbent chooses λ
on a grid to maximize the mean utility of a random observation sample plus
a weight on loyal targeting. Party vs individual then differ through the
re-election constraint (larger vs smaller winning coalition), not through
an optional rank-coupling flag.

Theories this cell *can* speak to
---------------------------------
- Re-election / accountability: winners shade λ toward the observed sample
  when the sample is not the same as their electoral coalition.
- Coalition-size channel: a larger winning bloc (party) can keep a higher λ
  without collapsing observed-sample utility as quickly as a small plurality
  winner.

Theories this cell *cannot* distinguish
---------------------------------------
- Citizen-candidate entry (Osborne–Slivinski, Besley–Coate): platforms and
  the candidate set are still drawn, not chosen at a cost.
- Core-voter vs swing-voter targeting (Cox–McCubbins vs Lindbeck–Weibull):
  there is no explicit swing set; the observation sample is a uniform
  random subset, not an equilibrium targeting rule.
- Campaigning, repeated play beyond one re-election, or party primaries.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.consensus.allocation import allocate

MECHANISMS = ("oneshot", "reelection")

REELECTION_OBSERVE_FRAC = 0.40
REELECTION_LOYALTY_WEIGHT = 0.15
LAMBDA_GRID = np.linspace(0.0, 1.0, 21)


def choose_lambda_reelection(
    benefits: np.ndarray,
    supporters: np.ndarray,
    platform: np.ndarray,
    observe_mask: np.ndarray,
    loyalty_weight: float = REELECTION_LOYALTY_WEIGHT,
    grid: np.ndarray = LAMBDA_GRID,
) -> float:
    """Choose λ to maximize re-election rate plus a weight on loyal targeting.

    An observer retains the incumbent if their utility is at least as high as
    under the everyone-direction (λ = 0) allocation with the same platform.
    The incumbent then maximizes ``retain_rate + loyalty_weight · λ``.

    A larger electoral coalition (party) overlaps more with a random
    observation sample, so it can keep a higher λ without losing as many
    retainers. The observation sample is shared across paradigms within a
    trial so pairing stays valid.
    """
    if not observe_mask.any():
        observe_mask = np.ones(benefits.shape[0], dtype=bool)
    everyone = benefits @ allocate(benefits, supporters, platform, 0.0)
    benchmark = everyone[observe_mask]
    best_lam = 0.0
    best_score = -np.inf
    for lam in grid:
        utility = (benefits @ allocate(benefits, supporters, platform, float(lam)))[observe_mask]
        retain_rate = float((utility >= benchmark - 1e-12).mean())
        score = retain_rate + loyalty_weight * float(lam)
        if score > best_score:
            best_lam = float(lam)
            best_score = score
    return best_lam


def draw_observe_mask(rng: np.random.Generator, n_voters: int, frac: float = REELECTION_OBSERVE_FRAC) -> np.ndarray:
    """Random subset of voters who observe their utility before the second vote."""
    return rng.random(n_voters) < frac
