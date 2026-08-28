"""Experiment orchestration: trials, summaries, and artifact writing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from farm.experiments.consensus.contrasts import paired_contrasts
from farm.experiments.consensus.mechanism import MECHANISMS, choose_lambda_reelection, draw_observe_mask
from farm.experiments.consensus.metrics import (
    ALLOCATION_COLUMNS,
    WELFARE_COLUMNS,
    egalitarian_allocation,
    evaluate_trial,
    utilitarian_allocation,
    welfare_from_allocation,
)
from farm.experiments.consensus.paradigms import (
    CONSTRAINED_PARADIGM,
    PARADIGMS,
    VOTING_MODES,
    ElectionResult,
    nearest_candidate,
    run_election,
)
from farm.experiments.consensus.population import (
    POPULATION_TYPES,
    generate_candidates,
    generate_population,
    partition_ids,
)

GROUP_COLUMNS = ["population", "n_candidates", "paradigm"]
DEFAULT_LAMBDA_CAP = 0.25
AUDIT_DIRNAME = "private"


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
    persist_ballots: bool = True
    mechanism: str = "oneshot"
    voting: str = "sincere"

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if self.candidates < 1:
            raise ValueError(f"candidates must be >= 1, got {self.candidates}")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}, got {self.mechanism!r}")
        if self.voting not in VOTING_MODES:
            raise ValueError(f"voting must be one of {VOTING_MODES}, got {self.voting!r}")

    def paradigms(self) -> Sequence[str]:
        names = list(PARADIGMS)
        if self.include_constrained:
            names.append(CONSTRAINED_PARADIGM)
        return names

    @property
    def include_lambda_primary(self) -> bool:
        """λ_winner is a primary endpoint only when selection can see or choose it."""
        return bool(self.lambda_correlated or self.mechanism == "reelection")

    @property
    def primary_question(self) -> str:
        if self.mechanism == "reelection":
            return "reelection_incentives"
        if self.lambda_correlated:
            return "lambda_selection_robustness"
        return "ballot_format_fixed_partition"


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


@dataclass
class AuditArtifacts:
    """Synthetic ballots / supporter masks held off the official record."""

    cluster_ids: np.ndarray  # (T, N)
    supporters: dict[str, np.ndarray]  # paradigm -> (T, N) bool
    ballots: dict[str, np.ndarray]  # paradigm -> (T, N) int
    group_ids: np.ndarray  # (T, N) fixed partition labels

    @property
    def n_trials(self) -> int:
        return int(self.cluster_ids.shape[0])


@dataclass
class TrialRun:
    trials: pd.DataFrame
    audit: AuditArtifacts


def _trial_rng(config: ExperimentConfig, trial: int) -> np.random.Generator:
    """Independent, reproducible stream per (seed, population, candidates, trial)."""
    population_code = POPULATION_TYPES.index(config.population)
    return np.random.default_rng([config.seed, population_code, config.candidates, trial])


def _baseline_rows(
    population,
    candidates,
    random_total: float,
    random_minority: float,
    opt_total: float,
    opt_minority: float,
    random_winner: int,
) -> list[dict]:
    nearest = nearest_candidate(population, candidates)
    random_election = ElectionResult(winner=int(random_winner), supporters=nearest == random_winner, ballots=nearest)
    random_row = evaluate_trial(
        "random_winner",
        population,
        candidates,
        random_election,
        random_total=random_total,
        random_minority=random_minority,
        opt_total=opt_total,
        opt_minority=opt_minority,
    )
    util_alloc = utilitarian_allocation(population.benefits)
    egal_alloc = egalitarian_allocation(population.benefits)
    util_row = welfare_from_allocation(
        "utilitarian",
        population,
        util_alloc,
        supporters=None,
        lambda_winner=float("nan"),
        lambda_effective=float("nan"),
        winner=-1,
        random_total=random_total,
        random_minority=random_minority,
        opt_total=opt_total,
        opt_minority=opt_minority,
    )
    egal_row = welfare_from_allocation(
        "egalitarian",
        population,
        egal_alloc,
        supporters=None,
        lambda_winner=float("nan"),
        lambda_effective=float("nan"),
        winner=-1,
        random_total=random_total,
        random_minority=random_minority,
        opt_total=opt_total,
        opt_minority=opt_minority,
    )
    return [random_row, util_row, egal_row]


def _trial_brackets(population, candidates, rng: np.random.Generator):
    nearest = nearest_candidate(population, candidates)
    random_winner = int(rng.integers(0, candidates.n_candidates))
    random_election = ElectionResult(winner=random_winner, supporters=nearest == random_winner)
    random_row = evaluate_trial("random_winner", population, candidates, random_election)
    util_alloc = utilitarian_allocation(population.benefits)
    util_row = welfare_from_allocation(
        "utilitarian",
        population,
        util_alloc,
        supporters=None,
        lambda_winner=float("nan"),
        lambda_effective=float("nan"),
        winner=-1,
    )
    return random_winner, random_row["total_welfare"], random_row["minority_welfare"], util_row["total_welfare"], util_row["minority_welfare"]


def run_cell(config: ExperimentConfig) -> TrialRun:
    """Run all trials of one cell and collect official rows plus local audit arrays."""
    rows: list[dict] = []
    cluster_ids = np.empty((config.trials, config.voters), dtype=int)
    group_ids = np.empty((config.trials, config.voters), dtype=int)
    supporters = {name: np.empty((config.trials, config.voters), dtype=bool) for name in config.paradigms()}
    ballots = {name: np.empty((config.trials, config.voters), dtype=int) for name in config.paradigms()}

    for trial in range(config.trials):
        rng = _trial_rng(config, trial)
        population = generate_population(rng, config.voters, config.population)
        candidates = generate_candidates(rng, config.candidates, population, config.lambda_correlated)
        observe_mask = draw_observe_mask(rng, config.voters) if config.mechanism == "reelection" else None
        random_winner, random_total, random_minority, opt_total, opt_minority = _trial_brackets(
            population, candidates, rng
        )

        cluster_ids[trial] = population.cluster_ids
        group_ids[trial] = partition_ids(population)

        for paradigm in config.paradigms():
            election = run_election(paradigm, population, candidates, voting=config.voting)
            cap = config.lambda_cap if paradigm == CONSTRAINED_PARADIGM else None
            override = None
            if config.mechanism == "reelection" and observe_mask is not None:
                override = choose_lambda_reelection(
                    population.benefits,
                    election.supporters,
                    candidates.platforms[election.winner],
                    observe_mask,
                )
            row = evaluate_trial(
                paradigm,
                population,
                candidates,
                election,
                lambda_cap=cap,
                lambda_override=override,
                random_total=random_total,
                random_minority=random_minority,
                opt_total=opt_total,
                opt_minority=opt_minority,
            )
            row.update(
                trial=trial,
                seed=config.seed,
                population=config.population,
                n_voters=config.voters,
                n_candidates=config.candidates,
            )
            rows.append(row)
            supporters[paradigm][trial] = election.supporters
            if election.ballots is None:
                ballots[paradigm][trial] = -1
            else:
                ballots[paradigm][trial] = election.ballots

        for row in _baseline_rows(
            population,
            candidates,
            random_total,
            random_minority,
            opt_total,
            opt_minority,
            random_winner,
        ):
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
    return TrialRun(
        trials=pd.DataFrame(rows)[columns],
        audit=AuditArtifacts(
            cluster_ids=cluster_ids,
            supporters=supporters,
            ballots=ballots,
            group_ids=group_ids,
        ),
    )


def run_trials(config: ExperimentConfig) -> pd.DataFrame:
    """Run all trials of one cell; one output row per (paradigm x trial), plus baselines.

    Each trial draws one voter population and one candidate slate shared by all
    paradigms, so treatments differ only in the selection rule (and, when
    enabled, in the incentive mechanism applied to that rule's winner).
    """
    return run_cell(config).trials


def _concat_audits(audits: Sequence[AuditArtifacts]) -> AuditArtifacts:
    """Stack per-cell audit arrays in the same cell-major order as ``run_sweep``."""
    supporters: dict[str, list[np.ndarray]] = {}
    ballots: dict[str, list[np.ndarray]] = {}
    for audit in audits:
        for name, array in audit.supporters.items():
            supporters.setdefault(name, []).append(array)
        for name, array in audit.ballots.items():
            ballots.setdefault(name, []).append(array)
    return AuditArtifacts(
        cluster_ids=np.concatenate([audit.cluster_ids for audit in audits], axis=0),
        group_ids=np.concatenate([audit.group_ids for audit in audits], axis=0),
        supporters={name: np.concatenate(chunks, axis=0) for name, chunks in supporters.items()},
        ballots={name: np.concatenate(chunks, axis=0) for name, chunks in ballots.items()},
    )


def run_sweep(sweep: SweepConfig) -> TrialRun:
    """Run every sweep cell and concatenate official rows plus audit arrays."""
    runs = [run_cell(cell) for cell in sweep.cells()]
    return TrialRun(
        trials=pd.concat([run.trials for run in runs], ignore_index=True),
        audit=_concat_audits([run.audit for run in runs]),
    )


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    """Means and stds of the welfare metrics by (population, candidates, paradigm)."""
    present = [c for c in WELFARE_COLUMNS if c in trials.columns]
    grouped = trials.groupby(GROUP_COLUMNS, sort=False)[present]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def allocation_means(trials: pd.DataFrame) -> pd.DataFrame:
    """Mean winner allocation per project by (population, candidates, paradigm)."""
    grouped = trials.groupby(GROUP_COLUMNS, sort=False)[list(ALLOCATION_COLUMNS)]
    return grouped.mean().reset_index()


def write_audit(audit: AuditArtifacts, out_dir: Path) -> None:
    """Write supporter masks and cluster ids under ``private/`` (not notarized)."""
    private = out_dir / AUDIT_DIRNAME
    private.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cluster_ids": audit.cluster_ids,
        "group_ids": audit.group_ids,
    }
    for name, array in audit.supporters.items():
        payload[f"supporters_{name}"] = array
    for name, array in audit.ballots.items():
        payload[f"ballots_{name}"] = array
    np.savez_compressed(private / "ballots.npz", **payload)
    (private / "README.md").write_text(
        "Local audit artifacts for the consensus experiment.\n\n"
        "- `ballots.npz` holds `cluster_ids`, `group_ids`, per-paradigm supporter\n"
        "  masks (`supporters_<paradigm>`), and per-paradigm ballot indices\n"
        "  (`ballots_<paradigm>`).\n"
        "- Official / notarized record is `summary.csv` plus `trials.csv`\n"
        "  aggregates only. Do not stamp this folder (FarmNotary policy #983).\n"
        "- These are synthetic voters. Discarding them is not a privacy claim;\n"
        "  it is a split between the official record and a local audit trail.\n"
    )


def write_outputs(
    trials: pd.DataFrame,
    out_dir: Path,
    run_config: dict,
    audit: AuditArtifacts | None = None,
    persist_ballots: bool | None = None,
) -> None:
    """Write trials, summary, allocations, contrasts, run_config, figures, REPORT.md.

    The run config is written as run_config.json, not manifest.json: the
    manifest.json name is reserved for the FarmNotary artifact manifest.
    Synthetic ballots go under ``private/`` and are not part of the official
    record.
    """
    from farm.experiments.consensus.plots import write_figures
    from farm.experiments.consensus.report import write_report

    out_dir.mkdir(parents=True, exist_ok=True)
    config = run_config.get("config", {})
    include_lambda = bool(config.get("lambda_correlated") or config.get("mechanism") == "reelection")
    summary = summarize(trials)
    allocations = allocation_means(trials)
    contrasts = paired_contrasts(trials, include_lambda_primary=include_lambda)

    trials.to_csv(out_dir / "trials.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    allocations.to_csv(out_dir / "allocation_means.csv", index=False)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    write_figures(trials, out_dir / "figures")
    write_report(trials, summary, allocations, run_config, out_dir / "REPORT.md", contrasts=contrasts)

    should_persist = persist_ballots if persist_ballots is not None else bool(config.get("persist_ballots", True))
    if should_persist and audit is not None:
        write_audit(audit, out_dir)


def config_manifest(config: ExperimentConfig, command: str) -> dict:
    payload = asdict(config)
    payload["primary_question"] = config.primary_question
    payload["lambda_primary"] = config.include_lambda_primary
    return {"command": command, "config": payload}


def sweep_manifest(sweep: SweepConfig, command: str) -> dict:
    payload = asdict(sweep.base)
    payload["primary_question"] = sweep.base.primary_question
    payload["lambda_primary"] = sweep.base.include_lambda_primary
    return {
        "command": command,
        "config": payload,
        "populations": list(sweep.populations),
        "candidate_counts": list(sweep.candidate_counts),
    }
