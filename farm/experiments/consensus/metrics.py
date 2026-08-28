"""Per-trial metric computation.

Official artifacts store aggregates and the winner's allocation. Synthetic
ballots, supporter masks, and cluster ids are written under ``private/``
when persistence is enabled; they are not part of the stamped record.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from farm.experiments.consensus.allocation import allocate
from farm.experiments.consensus.paradigms import ElectionResult
from farm.experiments.consensus.population import PROJECTS, Candidates, Population, partition_ids

ALLOCATION_COLUMNS = tuple(f"alloc_{name}" for name in PROJECTS)
MAX_GROUPS = 3

ENDOGENOUS_COLUMNS = (
    "total_welfare",
    "supporter_welfare",
    "loser_welfare",
    "gap",
    "lambda_winner",
    "lambda_effective",
    "loser_share",
)
FIXED_COLUMNS = (
    "majority_welfare",
    "minority_welfare",
    *(f"cluster_{k}_welfare" for k in range(MAX_GROUPS)),
)
TAIL_COLUMNS = (
    "min_utility",
    "p10_utility",
    "gini_utility",
)
NORM_COLUMNS = (
    "total_welfare_norm",
    "minority_welfare_norm",
)
WELFARE_COLUMNS = ENDOGENOUS_COLUMNS + FIXED_COLUMNS + TAIL_COLUMNS + NORM_COLUMNS


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a nonnegative (or shifted) utility vector."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        x = x - x.min()
    x = np.sort(x)
    total = float(x.sum())
    if total <= 0.0:
        return 0.0
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * total)) - (n + 1) / n)


def utilitarian_allocation(benefits: np.ndarray) -> np.ndarray:
    """Simplex allocation maximizing mean utility: the mean benefit direction.

    Documented family: directed-only (no platform blend). ``E[u] = dir_all @ a``
    is maximized on the simplex at ``a = dir_all``.
    """
    direction = benefits.mean(axis=0)
    return direction / direction.sum()


def egalitarian_allocation(benefits: np.ndarray) -> np.ndarray:
    """Maximin allocation: maximize the worst-off voter's utility.

    Small LP (P variables). Falls back to the utilitarian direction if the
    solver fails.
    """
    n_voters, n_projects = benefits.shape
    cost = np.zeros(n_projects + 1)
    cost[-1] = -1.0
    upper = np.hstack([-benefits, np.ones((n_voters, 1))])
    equality = np.zeros((1, n_projects + 1))
    equality[0, :n_projects] = 1.0
    bounds = [(0.0, None)] * n_projects + [(None, None)]
    result = linprog(
        cost,
        A_ub=upper,
        b_ub=np.zeros(n_voters),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return utilitarian_allocation(benefits)
    alloc = np.clip(result.x[:n_projects], 0.0, None)
    return alloc / alloc.sum()


def _norm(value: float, floor: float, ceiling: float) -> float:
    denom = ceiling - floor
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return (value - floor) / denom


def _group_welfare(utilities: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    majority_id, minority_id = _majority_minority(group_ids)
    row: dict[str, float] = {
        "majority_welfare": float(utilities[group_ids == majority_id].mean()),
        "minority_welfare": float(utilities[group_ids == minority_id].mean()),
    }
    for k in range(MAX_GROUPS):
        mask = group_ids == k
        row[f"cluster_{k}_welfare"] = float(utilities[mask].mean()) if mask.any() else float("nan")
    return row


def _majority_minority(group_ids: np.ndarray) -> tuple[int, int]:
    counts = np.bincount(group_ids)
    order = np.argsort(counts)[::-1]
    majority = int(order[0])
    minority = int(order[1]) if order.size > 1 else majority
    return majority, minority


def welfare_from_allocation(
    paradigm: str,
    population: Population,
    alloc: np.ndarray,
    supporters: np.ndarray | None,
    lambda_winner: float,
    lambda_effective: float,
    winner: int,
    random_total: float | None = None,
    random_minority: float | None = None,
    opt_total: float | None = None,
    opt_minority: float | None = None,
) -> dict[str, float]:
    """Compute the full metric row from an already-chosen allocation."""
    utilities = population.benefits @ alloc
    groups = partition_ids(population)
    if supporters is None or supporters.size == 0:
        supporter_welfare = float("nan")
        loser_welfare = float("nan")
        loser_share = float("nan")
        gap = float("nan")
    else:
        losers = ~supporters
        supporter_welfare = float(utilities[supporters].mean()) if supporters.any() else float("nan")
        loser_welfare = float(utilities[losers].mean()) if losers.any() else float("nan")
        loser_share = float(losers.mean())
        gap = supporter_welfare - loser_welfare

    total = float(utilities.mean())
    groups_row = _group_welfare(utilities, groups)
    row: dict[str, float] = {
        "paradigm": paradigm,
        "winner": int(winner),
        "total_welfare": total,
        "supporter_welfare": supporter_welfare,
        "loser_welfare": loser_welfare,
        "gap": gap,
        "lambda_winner": float(lambda_winner),
        "lambda_effective": float(lambda_effective),
        "loser_share": loser_share,
        **groups_row,
        "min_utility": float(utilities.min()),
        "p10_utility": float(np.quantile(utilities, 0.10)),
        "gini_utility": gini(utilities),
        "total_welfare_norm": (
            _norm(total, random_total, opt_total) if random_total is not None and opt_total is not None else float("nan")
        ),
        "minority_welfare_norm": (
            _norm(groups_row["minority_welfare"], random_minority, opt_minority)
            if random_minority is not None and opt_minority is not None
            else float("nan")
        ),
    }
    row.update({col: float(value) for col, value in zip(ALLOCATION_COLUMNS, alloc)})
    return row


def evaluate_trial(
    paradigm: str,
    population: Population,
    candidates: Candidates,
    election: ElectionResult,
    lambda_cap: float | None = None,
    lambda_override: float | None = None,
    random_total: float | None = None,
    random_minority: float | None = None,
    opt_total: float | None = None,
    opt_minority: float | None = None,
) -> dict[str, float]:
    """Run the winner's allocation and compute welfare metrics for one trial.

    ``lambda_winner`` is the loyalty used as the scientific outcome (drawn
    type in the one-shot cell; chosen λ under re-election).
    ``lambda_effective`` is what the allocation actually used (capped for the
    constrained paradigm).
    """
    drawn = float(candidates.lam[election.winner])
    lambda_winner = drawn if lambda_override is None else float(lambda_override)
    lambda_effective = lambda_winner if lambda_cap is None else min(lambda_winner, lambda_cap)

    alloc = allocate(
        population.benefits,
        election.supporters,
        candidates.platforms[election.winner],
        lambda_effective,
    )
    return welfare_from_allocation(
        paradigm,
        population,
        alloc,
        election.supporters,
        lambda_winner=lambda_winner,
        lambda_effective=lambda_effective,
        winner=election.winner,
        random_total=random_total,
        random_minority=random_minority,
        opt_total=opt_total,
        opt_minority=opt_minority,
    )
