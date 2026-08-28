"""Selection paradigms (treatments): how the steward is elected.

Every paradigm returns an :class:`ElectionResult` with the winning candidate
index and a boolean supporters mask over voters. Ties resolve to the lowest
index, which is deterministic under a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from farm.experiments.consensus.population import Candidates, Population, pca_split

PARADIGMS = ("party", "individual", "score", "latent_match")
CONSTRAINED_PARADIGM = "constrained_individual"
SELECTION_PARADIGMS = PARADIGMS
BASELINE_PARADIGMS = ("random_winner", "utilitarian", "egalitarian")

MAX_SCORE = 10.0
VOTING_MODES = ("sincere", "abandon_trailing")


@dataclass(frozen=True)
class ElectionResult:
    winner: int
    supporters: np.ndarray  # (N,) bool
    ballots: np.ndarray | None = None  # (N,) int candidate / party index when defined


def _distances(points: np.ndarray, platforms: np.ndarray) -> np.ndarray:
    """Euclidean distances between each point (row) and each platform: (N, M)."""
    return np.linalg.norm(points[:, None, :] - platforms[None, :, :], axis=2)


def nearest_candidate(population: Population, candidates: Candidates) -> np.ndarray:
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

    side = pca_split(population.prefs).astype(bool)
    return np.stack([population.prefs[side].mean(axis=0), population.prefs[~side].mean(axis=0)])


def party(population: Population, candidates: Candidates) -> ElectionResult:
    """Two parties at cluster means; nearest-party voting; parties nominate loyally."""
    platforms = _party_platforms(population)
    party_votes = _distances(population.prefs, platforms).argmin(axis=1)
    nominee_of = _distances(platforms, candidates.platforms).argmin(axis=1)

    votes_party_0 = int((party_votes == 0).sum())
    winning_party = 0 if votes_party_0 >= population.n_voters - votes_party_0 else 1
    supporters = party_votes == winning_party
    return ElectionResult(winner=int(nominee_of[winning_party]), supporters=supporters, ballots=party_votes)


def individual(population: Population, candidates: Candidates) -> ElectionResult:
    """No party labels: nearest-candidate plurality (sincere)."""
    nearest = nearest_candidate(population, candidates)
    counts = np.bincount(nearest, minlength=candidates.n_candidates)
    winner = int(counts.argmax())
    return ElectionResult(winner=winner, supporters=nearest == winner, ballots=nearest)


def individual_abandon_trailing(population: Population, candidates: Candidates) -> ElectionResult:
    """Plurality with a Duverger-style 'abandon trailing candidates' heuristic.

    First preferences identify the top two candidates. Every voter whose nearest
    candidate is outside that pair switches to the closer of the two. Sincere
    plurality remains the default; this is an explicit non-sincere option.
    """
    nearest = nearest_candidate(population, candidates)
    counts = np.bincount(nearest, minlength=candidates.n_candidates)
    if candidates.n_candidates == 1:
        return ElectionResult(winner=0, supporters=np.ones(population.n_voters, dtype=bool), ballots=nearest)
    top_two = np.argsort(counts)[-2:]
    dists = _distances(population.prefs, candidates.platforms[top_two])
    switched = top_two[dists.argmin(axis=1)]
    choice = np.where(np.isin(nearest, top_two), nearest, switched)
    winner = int(np.bincount(choice, minlength=candidates.n_candidates).argmax())
    return ElectionResult(winner=winner, supporters=choice == winner, ballots=choice)


def score(population: Population, candidates: Candidates) -> ElectionResult:
    """Each voter scores every candidate 0-10 from inverted distance; highest mean wins."""
    dists = _distances(population.prefs, candidates.platforms)
    d_min = dists.min(axis=1, keepdims=True)
    d_range = np.maximum(dists.max(axis=1, keepdims=True) - d_min, 1e-12)
    scores = MAX_SCORE * (1.0 - (dists - d_min) / d_range)
    winner = int(scores.mean(axis=0).argmax())
    ballots = scores.argmax(axis=1)
    supporters = ballots == winner
    return ElectionResult(winner=winner, supporters=supporters, ballots=ballots)


def latent_match(population: Population, candidates: Candidates) -> ElectionResult:
    """Elect the candidate closest to the mean of all privately submitted preferences."""
    target = population.prefs.mean(axis=0)
    winner = int(np.linalg.norm(candidates.platforms - target, axis=1).argmin())
    nearest = nearest_candidate(population, candidates)
    return ElectionResult(winner=winner, supporters=nearest == winner, ballots=nearest)


#: constrained_individual shares the individual election; only allocation differs.
ELECTION_RULES: dict[str, Callable[[Population, Candidates], ElectionResult]] = {
    "party": party,
    "individual": individual,
    "score": score,
    "latent_match": latent_match,
    CONSTRAINED_PARADIGM: individual,
}


def run_election(
    name: str,
    population: Population,
    candidates: Candidates,
    voting: str = "sincere",
) -> ElectionResult:
    if voting not in VOTING_MODES:
        raise ValueError(f"Unknown voting mode {voting!r}; expected one of {VOTING_MODES}")
    if name in ("individual", CONSTRAINED_PARADIGM) and voting == "abandon_trailing":
        return individual_abandon_trailing(population, candidates)
    try:
        rule = ELECTION_RULES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown paradigm {name!r}; expected one of {tuple(ELECTION_RULES)}") from exc
    return rule(population, candidates)
