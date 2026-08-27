"""Experiment orchestration: trials, summaries, and artifact writing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from farm.experiments.consensus.metrics import ALLOCATION_COLUMNS, WELFARE_COLUMNS, evaluate_trial
from farm.experiments.consensus.paradigms import CONSTRAINED_PARADIGM, PARADIGMS, run_election
from farm.experiments.consensus.population import (
    POPULATION_TYPES,
    generate_candidates,
    generate_population,
)

GROUP_COLUMNS = ["population", "n_candidates", "paradigm"]
DEFAULT_LAMBDA_CAP = 0.25


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters of one experiment cell (population type x candidate count)."""

    trials: int = 250
    voters: int = 400
    candidates: int = 8
    population: str = "two_cluster"
    seed: int = 0
    include_constrained: bool = False
    lambda_cap: float = DEFAULT_LAMBDA_CAP
    lambda_correlated: bool = False

    def paradigms(self) -> Sequence[str]:
        names = list(PARADIGMS)
        if self.include_constrained:
            names.append(CONSTRAINED_PARADIGM)
        return names


@dataclass(frozen=True)
class SweepConfig:
    """Cartesian sweep over population types and candidate counts."""

    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    populations: Sequence[str] = POPULATION_TYPES
    candidate_counts: Sequence[int] = (8,)

    def cells(self) -> list[ExperimentConfig]:
        return [
            replace(self.base, population=population, candidates=n_candidates)
            for population in self.populations
            for n_candidates in self.candidate_counts
        ]


def _trial_rng(config: ExperimentConfig, trial: int) -> np.random.Generator:
    """Independent, reproducible stream per (seed, population, candidates, trial)."""
    population_code = POPULATION_TYPES.index(config.population)
    return np.random.default_rng([config.seed, population_code, config.candidates, trial])


def run_trials(config: ExperimentConfig) -> pd.DataFrame:
    """Run all trials of one cell; one output row per (paradigm x trial).

    Each trial draws one voter population and one candidate slate shared by all
    paradigms, so treatments differ only in the selection rule.
    """
    rows = []
    for trial in range(config.trials):
        rng = _trial_rng(config, trial)
        population = generate_population(rng, config.voters, config.population)
        candidates = generate_candidates(rng, config.candidates, population, config.lambda_correlated)

        for paradigm in config.paradigms():
            election = run_election(paradigm, population, candidates)
            cap = config.lambda_cap if paradigm == CONSTRAINED_PARADIGM else None
            row = evaluate_trial(paradigm, population, candidates, election, lambda_cap=cap)
            row.update(
                trial=trial,
                seed=config.seed,
                population=config.population,
                n_voters=config.voters,
                n_candidates=config.candidates,
            )
            rows.append(row)

    columns = [
        "population",
        "n_voters",
        "n_candidates",
        "paradigm",
        "trial",
        "seed",
        "winner",
        *WELFARE_COLUMNS,
        *ALLOCATION_COLUMNS,
    ]
    return pd.DataFrame(rows)[columns]


def run_sweep(sweep: SweepConfig) -> pd.DataFrame:
    return pd.concat([run_trials(cell) for cell in sweep.cells()], ignore_index=True)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    """Means and stds of the welfare metrics by (population, candidates, paradigm)."""
    grouped = trials.groupby(GROUP_COLUMNS, sort=False)[list(WELFARE_COLUMNS)]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def allocation_means(trials: pd.DataFrame) -> pd.DataFrame:
    """Mean winner allocation per project by (population, candidates, paradigm)."""
    grouped = trials.groupby(GROUP_COLUMNS, sort=False)[list(ALLOCATION_COLUMNS)]
    return grouped.mean().reset_index()


def write_outputs(trials: pd.DataFrame, out_dir: Path, manifest: dict) -> None:
    """Write trials.csv, summary.csv, allocation_means.csv, manifest.json, figures, REPORT.md."""
    from farm.experiments.consensus.plots import write_figures
    from farm.experiments.consensus.report import write_report

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials)
    allocations = allocation_means(trials)

    trials.to_csv(out_dir / "trials.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    allocations.to_csv(out_dir / "allocation_means.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_figures(trials, out_dir / "figures")
    write_report(trials, summary, allocations, manifest, out_dir / "REPORT.md")


def config_manifest(config: ExperimentConfig, command: str) -> dict:
    return {"command": command, "config": asdict(config)}


def sweep_manifest(sweep: SweepConfig, command: str) -> dict:
    return {
        "command": command,
        "config": asdict(sweep.base),
        "populations": list(sweep.populations),
        "candidate_counts": list(sweep.candidate_counts),
    }
