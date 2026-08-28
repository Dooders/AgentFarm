"""Thin wrapper around ``farm.experiments.consensus`` for the catalog / notary path.

The implementation lives in ``farm.experiments.consensus``. This runner calls
``run_trials`` so catalog users and ``scripts/run_consensus_paradigms_experiment.py``
hit the same code as ``run_experiment.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from farm.experiments.consensus.allocation import allocate
from farm.experiments.consensus.experiment import ExperimentConfig, run_trials
from farm.experiments.consensus.paradigms import CONSTRAINED_PARADIGM, PARADIGMS, run_election
from farm.experiments.consensus.population import generate_candidates, generate_population
from farm.provenance.notary import notarize_run_dir

# Selection treatments only. constrained_individual is opt-in via include_constrained.
SELECTION_PARADIGMS = PARADIGMS


@dataclass
class TrialRow:
    paradigm: str
    seed: int
    winner: int
    total_welfare: float
    supporter_welfare: float
    loser_welfare: float
    lambda_winner: float
    loser_share: float


def _allocate(winner_plat, winner_loy, benefits, supporter_mask):
    """Delegate to the package allocator (normalized platform blend)."""
    return allocate(benefits, supporter_mask, winner_plat, float(winner_loy))


def run_once(paradigm: str, n_voters: int, n_cand: int, seed: int, lambda_cap: float = 0.25) -> TrialRow:
    """One paradigm on one seeded draw, using the package generator and rules."""
    if paradigm not in (*PARADIGMS, CONSTRAINED_PARADIGM):
        raise ValueError(paradigm)
    rng = np.random.default_rng([seed, 0, n_cand, 0])
    population = generate_population(rng, n_voters, "two_cluster")
    candidates = generate_candidates(rng, n_cand, population)
    election = run_election(paradigm, population, candidates)
    used_lambda = float(candidates.lam[election.winner])
    if paradigm == CONSTRAINED_PARADIGM:
        used_lambda = min(used_lambda, lambda_cap)
    alloc = allocate(population.benefits, election.supporters, candidates.platforms[election.winner], used_lambda)
    utility = population.benefits @ alloc
    losers = ~election.supporters
    return TrialRow(
        paradigm=paradigm,
        seed=seed,
        winner=int(election.winner),
        total_welfare=float(utility.mean()),
        supporter_welfare=float(utility[election.supporters].mean()) if election.supporters.any() else float("nan"),
        loser_welfare=float(utility[losers].mean()) if losers.any() else float("nan"),
        lambda_winner=used_lambda,
        loser_share=float(losers.mean()),
    )


class ConsensusParadigmsExperiment:
    def __init__(self, output_dir: str | Path = "experiments/consensus_paradigms"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        trials: int = 50,
        voters: int = 200,
        candidates: int = 8,
        paradigms: Iterable[str] | None = None,
        notarize: bool = True,
        include_constrained: bool = False,
    ) -> Path:
        requested = tuple(paradigms) if paradigms is not None else PARADIGMS
        include_constrained = include_constrained or CONSTRAINED_PARADIGM in requested
        config = ExperimentConfig(
            trials=trials,
            voters=voters,
            candidates=candidates,
            population="two_cluster",
            seed=0,
            include_constrained=include_constrained,
            persist_ballots=False,
        )
        frame = run_trials(config)
        if paradigms is not None:
            frame = frame[frame["paradigm"].isin(requested)]

        trials_path = self.results_dir / "trials.csv"
        frame.to_csv(trials_path, index=False)

        metric_cols = [c for c in ("total_welfare", "supporter_welfare", "loser_welfare", "gap", "lambda_winner", "loser_share") if c in frame.columns]
        summary = frame.groupby("paradigm", sort=False)[metric_cols].mean().reset_index()
        summary_path = self.results_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)

        meta = {
            "trials": trials,
            "voters": voters,
            "candidates": candidates,
            "paradigms": list(requested),
            "implementation": "farm.experiments.consensus.experiment.run_trials",
        }
        (self.results_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        if notarize:
            official = summary.to_dict(orient="records")
            notarize_run_dir(
                self.results_dir,
                runner="consensus_paradigms",
                config=meta,
                official_record={"summary": official},
            )
        return summary_path
