"""Voter and candidate population generation for the consensus experiment.

Voters carry a latent preference vector ``prefs[i] in R^5`` (one coordinate per
project) and a nonnegative benefit vector ``benefits[i]`` whose rows sum to 1.
Utility of an allocation ``a`` for voter ``i`` is ``benefits[i] @ a``.

Project semantics are encoded in the cluster centers and in a per-voter
periphery adjustment:

- ``core_services``: moderate positive weight for every cluster, so everyone
  derives a small benefit.
- ``coalition_club``: pork for the first (majority) bloc.
- ``outgroup_repair``: pork for the second (minority) bloc, i.e. the people
  least like a majority-bloc winner.
- ``prestige_project``: weak broad value for every cluster.
- ``buffer_reserve``: low base weight, raised for voters far from the
  population center. Peripheral voters are the likeliest electoral losers
  under any rule, so this project acts as insurance for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

PROJECTS: tuple[str, ...] = (
    "core_services",
    "coalition_club",
    "outgroup_repair",
    "prestige_project",
    "buffer_reserve",
)
N_PROJECTS = len(PROJECTS)

_CENTER_A = np.array([0.55, 0.95, -0.35, 0.05, -0.15])
_CENTER_B = np.array([0.55, -0.35, 0.95, 0.05, -0.15])
_CENTER_C = np.array([0.55, 0.30, 0.30, 0.05, -0.15])

#: population type -> tuple of (cluster center, cluster weight)
_CLUSTER_SPECS = {
    "one_cluster": (((_CENTER_A + _CENTER_B) / 2.0, 1.0),),
    "two_cluster": ((_CENTER_A, 0.5), (_CENTER_B, 0.5)),
    "three_cluster": ((_CENTER_A, 0.4), (_CENTER_B, 0.4), (_CENTER_C, 0.2)),
    "rural_town": ((_CENTER_A, 0.7), (_CENTER_B, 0.3)),
}

POPULATION_TYPES: tuple[str, ...] = tuple(_CLUSTER_SPECS)

PREF_NOISE_SCALE = 0.35
BENEFIT_TEMPERATURE = 0.55
BUFFER_PERIPHERY_WEIGHT = 0.45
LAMBDA_BETA_A = 2.2
LAMBDA_BETA_B = 2.2


@dataclass(frozen=True)
class Population:
    """Voters of one trial: latent preferences, benefit shares, cluster labels."""

    prefs: np.ndarray  # (N, 5) latent preferences
    benefits: np.ndarray  # (N, 5) nonnegative, rows sum to 1
    cluster_ids: np.ndarray  # (N,) generator cluster label per voter
    cluster_centers: np.ndarray  # (K, 5)

    @property
    def n_voters(self) -> int:
        return self.prefs.shape[0]


@dataclass(frozen=True)
class Candidates:
    """Candidates of one trial: platforms and loyalty traits."""

    platforms: np.ndarray  # (M, 5)
    lam: np.ndarray  # (M,) loyalty trait in [0, 1]

    @property
    def n_candidates(self) -> int:
        return self.platforms.shape[0]


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _cluster_assignments(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    """Assign cluster ids with deterministic counts matching the weights.

    Exact proportions (rather than multinomial draws) keep the two_cluster
    population truly 50/50, which the party-paradigm invariants rely on.
    """
    counts = np.floor(weights * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    ids = np.repeat(np.arange(len(weights)), counts)
    return rng.permutation(ids)


def generate_population(rng: np.random.Generator, n_voters: int, population_type: str) -> Population:
    """Draw one voter population of the requested type."""
    if population_type not in _CLUSTER_SPECS:
        raise ValueError(f"Unknown population type {population_type!r}; expected one of {POPULATION_TYPES}")
    spec = _CLUSTER_SPECS[population_type]
    centers = np.stack([center for center, _ in spec])
    weights = np.array([weight for _, weight in spec])

    cluster_ids = _cluster_assignments(rng, n_voters, weights)
    prefs = centers[cluster_ids] + rng.normal(0.0, PREF_NOISE_SCALE, size=(n_voters, N_PROJECTS))

    # Buffer reserve appeals to peripheral voters (far from the population center).
    offsets = prefs - prefs.mean(axis=0)
    distance = np.linalg.norm(offsets, axis=1)
    periphery = distance / max(distance.max(), 1e-12)
    prefs = prefs.copy()
    prefs[:, PROJECTS.index("buffer_reserve")] += BUFFER_PERIPHERY_WEIGHT * periphery

    benefits = _softmax(prefs, BENEFIT_TEMPERATURE)
    return Population(prefs=prefs, benefits=benefits, cluster_ids=cluster_ids, cluster_centers=centers)


def generate_candidates(
    rng: np.random.Generator,
    n_candidates: int,
    population: Population,
    lambda_correlated: bool = False,
) -> Candidates:
    """Draw candidates whose platforms come from the same cluster structure as voters.

    By default the loyalty trait lambda ~ Beta(2.2, 2.2) is independent of the
    platform. With ``lambda_correlated=True`` the same Beta marginal is kept but
    lambda is assigned comonotonically with platform extremity (distance from
    the mean voter preference), tying high loyalty to cluster-extreme platforms.
    """
    n_clusters = population.cluster_centers.shape[0]
    cluster_sizes = np.bincount(population.cluster_ids, minlength=n_clusters)
    weights = cluster_sizes / cluster_sizes.sum()

    candidate_clusters = rng.choice(n_clusters, size=n_candidates, p=weights)
    platforms = population.cluster_centers[candidate_clusters] + rng.normal(
        0.0, PREF_NOISE_SCALE, size=(n_candidates, N_PROJECTS)
    )
    lam = rng.beta(LAMBDA_BETA_A, LAMBDA_BETA_B, size=n_candidates)

    if lambda_correlated:
        extremity = np.linalg.norm(platforms - population.prefs.mean(axis=0), axis=1)
        lam = np.sort(lam)[np.argsort(np.argsort(extremity))]

    return Candidates(platforms=platforms, lam=lam)
