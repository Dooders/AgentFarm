"""Selection paradigms (treatments): how the steward is elected.

Every paradigm returns an :class:`ElectionResult` with the winning candidate
index and a boolean supporters mask over voters. Ties resolve to the lowest
index, which is deterministic under a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from farm.experiments.consensus.population import Candidates, Population

PARADIGMS = ("party", "individual", "score", "latent_match")
CONSTRAINED_PARADIGM = "constrained_individual"

MAX_SCORE = 10.0


@dataclass(frozen=True)
class ElectionResult:
    winner: int
    supporters: np.ndarray  # (N,) bool


def _distances(points: np.ndarray, platforms: np.ndarray) -> np.ndarray:
    """Euclidean distances between each point (row) and each platform: (N, M)."""
    return np.linalg.norm(points[:, None, :] - platforms[None, :, :], axis=2)


def _nearest_candidate(population: Population, candidates: Candidates) -> np.ndarray:
    return _distances(population.prefs, candidates.platforms).argmin(axis=1)


def _party_platforms(population: Population) -> np.ndarray:
    """Two party brands at cluster means.

    Populations with at least two generator clusters use the mean preferences
    of the two largest clusters. A one-cluster population has no natural party
    split, so voters are divided by the sign of their first principal
    component and the halves' means become the brands.
    """
    n_clusters = population.cluster_centers.shape[0]
    if n_clusters >= 2:
        sizes = np.bincount(population.cluster_ids, minlength=n_clusters)
        top_two = np.argsort(sizes)[::-1][:2]
        return np.stack([population.prefs[population.cluster_ids == c].mean(axis=0) for c in top_two])

    centered = population.prefs - population.prefs.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[np.argmax(np.abs(axis))] < 0:  # fix SVD sign ambiguity for determinism
        axis = -axis
    side = centered @ axis >= 0
    return np.stack([population.prefs[side].mean(axis=0), population.prefs[~side].mean(axis=0)])


def party(population: Population, candidates: Candidates) -> ElectionResult:
    """Two parties at cluster means; nearest-party voting; parties nominate loyally."""
    platforms = _party_platforms(population)
    party_votes = _distances(population.prefs, platforms).argmin(axis=1)
    nominee_of = _distances(platforms, candidates.platforms).argmin(axis=1)

    votes_party_0 = int((party_votes == 0).sum())
    winning_party = 0 if votes_party_0 >= population.n_voters - votes_party_0 else 1
    supporters = party_votes == winning_party
    return ElectionResult(winner=int(nominee_of[winning_party]), supporters=supporters)


def individual(population: Population, candidates: Candidates) -> ElectionResult:
    """No party labels: nearest-candidate plurality."""
    nearest = _nearest_candidate(population, candidates)
    counts = np.bincount(nearest, minlength=candidates.n_candidates)
    winner = int(counts.argmax())
    return ElectionResult(winner=winner, supporters=nearest == winner)


def score(population: Population, candidates: Candidates) -> ElectionResult:
    """Each voter scores every candidate 0-10 from inverted distance; highest mean wins."""
    dists = _distances(population.prefs, candidates.platforms)
    d_min = dists.min(axis=1, keepdims=True)
    d_range = np.maximum(dists.max(axis=1, keepdims=True) - d_min, 1e-12)
    scores = MAX_SCORE * (1.0 - (dists - d_min) / d_range)
    winner = int(scores.mean(axis=0).argmax())
    supporters = scores.argmax(axis=1) == winner
    return ElectionResult(winner=winner, supporters=supporters)


def latent_match(population: Population, candidates: Candidates) -> ElectionResult:
    """Elect the candidate closest to the mean of all privately submitted preferences."""
    target = population.prefs.mean(axis=0)
    winner = int(np.linalg.norm(candidates.platforms - target, axis=1).argmin())
    supporters = _nearest_candidate(population, candidates) == winner
    return ElectionResult(winner=winner, supporters=supporters)


#: constrained_individual shares the individual election; only allocation differs.
ELECTION_RULES: dict[str, Callable[[Population, Candidates], ElectionResult]] = {
    "party": party,
    "individual": individual,
    "score": score,
    "latent_match": latent_match,
    CONSTRAINED_PARADIGM: individual,
}


def run_election(name: str, population: Population, candidates: Candidates) -> ElectionResult:
    try:
        rule = ELECTION_RULES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown paradigm {name!r}; expected one of {tuple(ELECTION_RULES)}") from exc
    return rule(population, candidates)
