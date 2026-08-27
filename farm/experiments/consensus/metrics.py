"""Per-trial metric computation.

Only aggregates and the winner's allocation are recorded; individual ballots
and voter-level utilities never leave this module.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.consensus.allocation import allocate
from farm.experiments.consensus.paradigms import ElectionResult
from farm.experiments.consensus.population import PROJECTS, Candidates, Population

ALLOCATION_COLUMNS = tuple(f"alloc_{name}" for name in PROJECTS)

WELFARE_COLUMNS = (
    "total_welfare",
    "supporter_welfare",
    "loser_welfare",
    "gap",
    "lambda_winner",
    "lambda_effective",
    "loser_share",
)


def evaluate_trial(
    paradigm: str,
    population: Population,
    candidates: Candidates,
    election: ElectionResult,
    lambda_cap: float | None = None,
) -> dict[str, float]:
    """Run the winner's allocation and compute welfare metrics for one trial.

    ``lambda_winner`` is always the selected candidate's raw loyalty trait;
    ``lambda_effective`` is what the allocation actually used (capped for the
    constrained paradigm).
    """
    lambda_winner = float(candidates.lam[election.winner])
    lambda_effective = lambda_winner if lambda_cap is None else min(lambda_winner, lambda_cap)

    alloc = allocate(
        population.benefits,
        election.supporters,
        candidates.platforms[election.winner],
        lambda_effective,
    )
    utilities = population.benefits @ alloc

    supporters = election.supporters
    losers = ~supporters
    supporter_welfare = float(utilities[supporters].mean()) if supporters.any() else float("nan")
    loser_welfare = float(utilities[losers].mean()) if losers.any() else float("nan")

    row: dict[str, float] = {
        "paradigm": paradigm,
        "winner": int(election.winner),
        "total_welfare": float(utilities.mean()),
        "supporter_welfare": supporter_welfare,
        "loser_welfare": loser_welfare,
        "gap": supporter_welfare - loser_welfare,
        "lambda_winner": lambda_winner,
        "lambda_effective": lambda_effective,
        "loser_share": float(losers.mean()),
    }
    row.update({col: float(value) for col, value in zip(ALLOCATION_COLUMNS, alloc)})
    return row
