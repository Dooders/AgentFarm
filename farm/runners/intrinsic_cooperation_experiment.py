"""Intrinsic cooperation experiment runner.

Builds on the intrinsic-evolution world (agents gather, reproduce with
genetic inheritance, and each offspring learns with its own brain) and asks:

    **When agents can bond into cooperative units -- and both the terms of
    those units and the taste for them are heritable -- does cooperation
    emerge, which institutions does selection favour, and what does it do to
    the population?**

Four arms share every seed and every other setting:

- ``baseline``    -- the unmodified world: eight actions, no cooperation loci.
- ``placebo``     -- the ``bond``/``leave`` actions and the cooperation loci
  exist but ``bond`` never forms a unit. This separates the cost of carrying
  two extra actions and twelve extra genes (a larger policy output, extra
  mutation load, wasted actions) from the effect of the institution itself.
- ``cooperation`` -- units form, pool, pay out, insure and enforce.
- ``cooperation_no_motive`` -- the same, with the motive-driven membership
  reward switched off, so the institution has to pay for itself through
  pooling, surplus, aid and enforcement alone.

Per arm the runner records population, births, deaths and lifespans, the
action mix (including ``bond`` and ``leave``), the policies' own probability
of choosing ``bond`` (before innate priors), the population means of the
cooperation loci and their change over the run, and the cooperation
service's own telemetry (coverage, units, failure modes). With several
paired replicates it reports seed-paired contrasts with Wilcoxon signed-rank
tests and a Holm correction over the registered family (see
``docs/research/experiments/cooperation/SpatialDesign.md``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import statistics
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from farm.config import SimulationConfig
from farm.core.cooperation import CooperationConfig
from farm.core.cooperation.actions import COOPERATION_ACTION_NAMES, cooperation_service_of
from farm.core.cooperation.genes import COOPERATION_GENE_NAMES, SETTING_GENE_NAMES, gene_values
from farm.core.hyperparameter_chromosome import (
    INTRINSIC_REWARD_GENE_NAMES,
    LEARNING_GENE_NAMES,
    BoundaryMode,
    CrossoverMode,
    MutationMode,
)
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode
from farm.core.simulation import run_simulation
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy
from farm.utils.logging import get_logger

logger = get_logger(__name__)

#: Loci pinned for the run when ``freeze_platform_genes`` is on: Chromosome A
#: (learning hyperparameters, including the network and replay sizes) and
#: Chromosome C (reward weights).
FROZEN_PLATFORM_GENES: frozenset[str] = frozenset(LEARNING_GENE_NAMES) | frozenset(INTRINSIC_REWARD_GENE_NAMES)


def git_commit() -> str | None:
    """The repository's HEAD commit, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


#: Every arm the runner knows. ``cooperation_zero_synergy`` is optional: the
#: working institution with ``synergy=0``, so pooling creates no surplus and
#: any population effect must come from insurance, aid and enforcement alone.
ARM_NAMES: tuple[str, ...] = (
    "baseline",
    "placebo",
    "cooperation",
    "cooperation_no_motive",
    "cooperation_zero_synergy",
)
#: The registered four.
DEFAULT_ARMS: tuple[str, ...] = ("baseline", "placebo", "cooperation", "cooperation_no_motive")
BASELINE_ARM: str = "baseline"

#: Seed-paired contrasts, ``name -> (baseline arm, treatment arm)``.
ARM_CONTRASTS: dict[str, tuple[str, str]] = {
    "cooperation_minus_placebo": ("placebo", "cooperation"),
    "no_motive_minus_placebo": ("placebo", "cooperation_no_motive"),
    "cooperation_minus_no_motive": ("cooperation_no_motive", "cooperation"),
    "cooperation_minus_baseline": ("baseline", "cooperation"),
    "placebo_minus_baseline": ("baseline", "placebo"),
    "zero_synergy_minus_placebo": ("placebo", "cooperation_zero_synergy"),
    "cooperation_minus_zero_synergy": ("cooperation_zero_synergy", "cooperation"),
}

TRACKED_ACTIONS: tuple[str, ...] = (
    "move",
    "gather",
    "share",
    "attack",
    "defend",
    "reproduce",
    "pass",
    "communicate",
) + COOPERATION_ACTION_NAMES

#: Settings loci in the confirmatory family. The other three (symmetry,
#: exclusivity, max_members) are reported as exploratory so the Holm family
#: is not diluted by loci with no directional prediction.
CONFIRMATORY_SETTING_LOCI: tuple[str, ...] = ("coop_scope", "coop_exit_cost", "coop_recognition")
EXPLORATORY_SETTING_LOCI: tuple[str, ...] = tuple(g for g in SETTING_GENE_NAMES if g not in CONFIRMATORY_SETTING_LOCI)


def _selection_on(name: str) -> tuple[str, str, str, str]:
    return (
        f"change_{name}",
        "cooperation_minus_placebo",
        "two-sided",
        (
            f"Selection acts on {name[5:]}: the run-long change in the population mean of {name} differs from "
            "its neutral drift in the placebo."
        ),
    )


#: The registered hypotheses: ``key -> (metric, contrast, direction, statement)``.
#: All are seed-paired tests of ``cooperation - placebo`` across replicates,
#: Holm-corrected as one family.
HYPOTHESES: dict[str, tuple[str, str, str, str]] = {
    "S2": (
        "policy_bond_prob_late",
        "cooperation_minus_placebo",
        "greater",
        (
            "The brain learns to bond: in the last quarter of the run the policies' own probability of choosing "
            "bond (before innate priors) is higher in the cooperation arm than in the placebo, where it never pays."
        ),
    ),
    "S3": (
        "mean_population",
        "cooperation_minus_placebo",
        "greater",
        "Cooperation sustains a larger population than the same world without working units.",
    ),
    "S3b": (
        "mean_lifespan_at_death",
        "cooperation_minus_placebo",
        "greater",
        (
            "Cooperation lengthens lives: agents that die in the cooperation arm lived longer than in the placebo. "
            "Unlike population, lifespan has no ceiling when the population sits at its cap."
        ),
    ),
    **{f"S4_{name[5:]}": _selection_on(name) for name in CONFIRMATORY_SETTING_LOCI},
    "S5": (
        "change_expected_exit_payment",
        "cooperation_minus_placebo",
        "greater",
        (
            "Selection favours binding terms: the population mean of recognition x exit cost (the enforced exit "
            "payment per unit of the exit-cost base) rises more than under neutral drift."
        ),
    ),
}

#: Manipulation checks reported beside the family (not corrected with it).
MANIPULATION_CHECKS: dict[str, tuple[str, str, str, str]] = {
    "M1": (
        "mean_coverage",
        "cooperation_minus_placebo",
        "greater",
        "Units form at all: coverage in the cooperation arm exceeds the placebo's, which is zero by construction.",
    ),
    "M2": (
        "mean_population",
        "placebo_minus_baseline",
        "two-sided",
        "The cost of carrying two extra actions and twelve extra loci without a working institution.",
    ),
}

#: Positive controls for learning: one-sample tests of an arm's own metric
#: against zero across seeds (``key -> (metric, arm, direction, statement)``).
#: If these fail, ``S2`` cannot be read as a statement about learning.
LEARNING_CHECKS: dict[str, tuple[str, str, str, str]] = {
    "L1": (
        "policy_gather_lift_late",
        "baseline",
        "greater",
        (
            "The brains learn at all: late in the run the policies' own probability of gather (before priors) "
            "exceeds uniform, in the unmodified world."
        ),
    ),
    "L2": (
        "gather_share_over_prior_late",
        "baseline",
        "greater",
        "Learning shows in behaviour: the late gather share exceeds what the innate priors alone would give.",
    ),
}

#: Exploratory contrasts: reported uncorrected and flagged, never part of the family.
EXPLORATORY: dict[str, tuple[str, str, str, str]] = {
    **{f"X4_{name[5:]}": _selection_on(name) for name in EXPLORATORY_SETTING_LOCI},
    "X_starvation": (
        "starvation_deaths",
        "cooperation_minus_placebo",
        "less",
        "Insurance works: fewer agents starve to death in the cooperation arm.",
    ),
    "X_births": (
        "total_births",
        "cooperation_minus_placebo",
        "greater",
        "Cooperation raises turnover: more births over the run than in the placebo.",
    ),
    "X_zero_synergy_population": (
        "mean_population",
        "zero_synergy_minus_placebo",
        "greater",
        "Without the synergy subsidy the institution still sustains a larger population than the placebo.",
    ),
    "X_synergy_share": (
        "mean_population",
        "cooperation_minus_zero_synergy",
        "greater",
        "How much of the population effect is the 20% surplus created by pooling.",
    ),
}


def _wilcoxon(diff: np.ndarray, alternative: str) -> float:
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    return float(scipy_stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue)


def _bootstrap_ci(diff: np.ndarray, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if diff.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _run_replicate_worker(
    base_config: SimulationConfig,
    config: IntrinsicCooperationExperimentConfig,
    index: int,
    seed: int,
) -> ReplicateResult:
    """Spawned worker: one seed, all arms. Thread pools stay pinned to 1."""
    for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(thread_var, "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    logger.info("intrinsic_cooperation_replicate_start", replicate=index, seed=seed, worker=os.getpid())
    return IntrinsicCooperationExperiment(base_config, config)._run_replicate(index, seed)


def holm(results: list[dict[str, Any]], alpha: float = 0.05) -> None:
    """Holm step-down correction in place over ``results[i]["p_value"]``."""
    order = sorted(range(len(results)), key=lambda i: results[i]["p_value"])
    m = len(results)
    running = 0.0
    stop = False
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i]["p_value"])
        running = max(running, adj)
        results[i]["p_holm"] = running
        if not stop and results[i]["p_value"] <= alpha / (m - rank):
            results[i]["significant"] = True
        else:
            stop = True
            results[i]["significant"] = False


# ----------------------------------------------------------- aggregation
# These work on the per-replicate summary dicts that the summary JSON stores
# (``{"index", "seed", <arm>: ArmResult.summary(), ...}``), so replicates run
# in separate processes (see ``scripts/merge_intrinsic_cooperation_runs.py``)
# are aggregated and tested exactly as replicates run in one process.


def replicate_summary(index: int, seed: int, arm_summaries: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """One replicate's record in the summary JSON."""
    return {"index": index, "seed": seed, **dict(arm_summaries)}


def scalar_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """The scalar metrics of one arm summary that enter the paired contrasts."""
    metrics: dict[str, float] = {
        "mean_population": float(summary["mean_population"]),
        "final_population": float(summary["final_population"]),
        "peak_population": float(summary["peak_population"]),
        "total_births": float(summary["total_births"]),
        "total_deaths": float(summary["total_deaths"]),
        "mean_coverage": float(summary["mean_coverage"]),
        "final_coverage": float(summary["final_coverage"]),
        "mean_units_active": float(summary["mean_units_active"]),
        "units_formed": float(summary["units_formed"]),
        "mean_unit_lifetime": float(summary["mean_unit_lifetime"]),
        "bond_share_late": float(summary["bond_share_late"]),
        "policy_bond_prob_late": float(summary["policy_bond_prob_late"]),
        "acting_bond_prob_late": float(summary.get("acting_bond_prob_late", float("nan"))),
        "bond_rank_late": float(summary.get("bond_rank_late", float("nan"))),
        "policy_gather_prob_late": float(summary.get("policy_gather_prob_late", float("nan"))),
        "policy_gather_lift_late": float(summary.get("policy_gather_lift_late", float("nan"))),
        "gather_share_over_prior_late": float(summary.get("gather_share_over_prior_late", float("nan"))),
        "policy_peak_late": float(summary.get("policy_peak_late", float("nan"))),
        "mean_lifespan_at_death": float(summary["mean_lifespan_at_death"]),
        "starvation_deaths": float(summary.get("starvation_deaths", float("nan"))),
        "combat_deaths": float(summary.get("combat_deaths", float("nan"))),
        "max_generation": float(summary["max_generation"]),
        "change_expected_exit_payment": float(summary["change_expected_exit_payment"]),
        "forfeits_per_enforced_exit": float(summary.get("forfeits_per_enforced_exit", float("nan"))),
    }
    for action, share in summary["mean_action_share"].items():
        metrics[f"action_share[{action}]"] = float(share)
    for gene, value in summary["gene_mean_end"].items():
        metrics[f"mean_{gene}_end"] = float(value)
    for gene, value in summary["gene_mean_change"].items():
        metrics[f"change_{gene}"] = float(value)
    return metrics


def metric_arrays(
    replicate_summaries: Sequence[Mapping[str, Any]], arms: Sequence[str]
) -> dict[str, dict[str, np.ndarray]]:
    """``arm -> metric -> array over replicates`` (replicate order preserved)."""
    arms = list(arms)
    per_arm = {arm: [scalar_metrics(r[arm]) for r in replicate_summaries] for arm in arms}
    names = list(per_arm[arms[0]][0].keys())
    return {arm: {n: np.array([m[n] for m in per_arm[arm]], dtype=float) for n in names} for arm in arms}


def aggregate_summaries(replicate_summaries: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    """Per-arm descriptives and every seed-paired contrast, for every scalar metric."""
    arms = list(arms)
    arrays = metric_arrays(replicate_summaries, arms)
    names = list(arrays[arms[0]].keys())

    def describe(v: np.ndarray) -> dict[str, float]:
        v = v[~np.isnan(v)]
        if v.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "sem": float("nan")}
        std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        return {"mean": float(np.mean(v)), "std": std, "sem": std / np.sqrt(v.size) if v.size > 1 else 0.0}

    paired: dict[str, dict[str, Any]] = {}
    for contrast, (b, t) in ARM_CONTRASTS.items():
        if b not in arms or t not in arms:
            continue
        paired[contrast] = {}
        for n in names:
            diff = arrays[t][n] - arrays[b][n]
            diff = diff[~np.isnan(diff)]
            lo, hi = _bootstrap_ci(diff)
            paired[contrast][n] = {
                "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "positive_seeds": int((diff > 0).sum()),
                "n": int(diff.size),
                "p_two_sided": _wilcoxon(diff, "two-sided"),
            }
    return {
        "num_replicates": len(replicate_summaries),
        "seeds": [int(r["seed"]) for r in replicate_summaries],
        "per_arm": {arm: {n: describe(arrays[arm][n]) for n in names} for arm in arms},
        "paired_deltas": paired,
    }


def _run_tests(arrays: dict[str, dict[str, np.ndarray]], family: Mapping[str, tuple]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key, (metric, contrast, direction, statement) in family.items():
        b, t = ARM_CONTRASTS[contrast]
        if b not in arrays or t not in arrays or metric not in arrays[t]:
            continue
        diff = arrays[t][metric] - arrays[b][metric]
        diff = diff[~np.isnan(diff)]
        lo, hi = _bootstrap_ci(diff)
        results.append(
            {
                "key": key,
                "statement": statement,
                "metric": metric,
                "contrast": contrast,
                "direction": direction,
                "n": int(diff.size),
                "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "positive_seeds": int((diff > 0).sum()),
                "p_value": _wilcoxon(diff, direction),
            }
        )
    return results


def _run_one_sample_tests(
    arrays: dict[str, dict[str, np.ndarray]], family: Mapping[str, tuple]
) -> list[dict[str, Any]]:
    """Tests of one arm's metric against zero across replicates."""
    results: list[dict[str, Any]] = []
    for key, (metric, arm, direction, statement) in family.items():
        if arm not in arrays or metric not in arrays[arm]:
            continue
        values = arrays[arm][metric]
        values = values[~np.isnan(values)]
        lo, hi = _bootstrap_ci(values)
        results.append(
            {
                "key": key,
                "statement": statement,
                "metric": metric,
                "contrast": f"{arm} vs 0",
                "direction": direction,
                "n": int(values.size),
                "mean_diff": float(np.mean(values)) if values.size else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "positive_seeds": int((values > 0).sum()),
                "p_value": _wilcoxon(values, direction),
            }
        )
    return results


#: Flags that mark a test entry as outside the Holm family.
OUT_OF_FAMILY_FLAGS: tuple[str, ...] = ("manipulation_check", "learning_check", "exploratory")


def hypothesis_kind(entry: Mapping[str, Any]) -> str:
    """``"family"`` or the out-of-family flag an entry from :func:`test_hypotheses` carries."""
    return next((flag for flag in OUT_OF_FAMILY_FLAGS if entry.get(flag)), "family")


def expected_test_keys(arms: Sequence[str]) -> set[str]:
    """The keys :func:`test_hypotheses` reports when exactly ``arms`` were run."""
    present = set(arms)

    def has(contrast: str) -> bool:
        b, t = ARM_CONTRASTS[contrast]
        return b in present and t in present

    keys = {k for k, (_, c, _, _) in HYPOTHESES.items() if has(c)}
    keys |= {k for k, (_, c, _, _) in MANIPULATION_CHECKS.items() if has(c)}
    keys |= {k for k, (_, arm, _, _) in LEARNING_CHECKS.items() if arm in present}
    keys |= {k for k, (_, c, _, _) in EXPLORATORY.items() if has(c)}
    return keys


def _uncorrected(tests: list[dict[str, Any]], alpha: float, flag: str) -> list[dict[str, Any]]:
    for test in tests:
        test["p_holm"] = None
        test["significant"] = test["p_value"] <= alpha
        test[flag] = True
    return tests


def test_hypotheses(
    replicate_summaries: Sequence[Mapping[str, Any]], arms: Sequence[str], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """The registered family (Holm-corrected), then the uncorrected checks and exploratory contrasts.

    Entries outside the family are flagged ``manipulation_check``,
    ``learning_check`` or ``exploratory``.
    """
    arrays = metric_arrays(replicate_summaries, arms)
    results = _run_tests(arrays, HYPOTHESES)
    holm(results, alpha)
    return (
        results
        + _uncorrected(_run_tests(arrays, MANIPULATION_CHECKS), alpha, "manipulation_check")
        + _uncorrected(_run_one_sample_tests(arrays, LEARNING_CHECKS), alpha, "learning_check")
        + _uncorrected(_run_tests(arrays, EXPLORATORY), alpha, "exploratory")
    )


@dataclass(frozen=True)
class IntrinsicCooperationExperimentConfig:
    """Configuration for :class:`IntrinsicCooperationExperiment`."""

    num_steps: int = 3000
    seed: int = 101
    num_replicates: int = 1
    output_dir: str = "experiments/intrinsic_cooperation"
    record_interval: int = 1
    arms: tuple[str, ...] = DEFAULT_ARMS

    # Cooperation settings shared by the placebo and cooperation arms. The
    # enforced exit forfeit is a fraction of what the unit paid the leaver,
    # not of its whole wealth, so a random ``leave`` cannot cost more than
    # the unit was ever worth to it.
    cooperation: CooperationConfig = field(
        default_factory=lambda: CooperationConfig(enabled=True, exit_cost_base="received")
    )

    # Per-reproduction inheritance. Crossover is on so a member's partner can
    # be its co-parent (CooperationConfig.partner_coparent).
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.REFLECT
    crossover_enabled: bool = True
    crossover_mode: CrossoverMode = CrossoverMode.UNIFORM
    coparent_max_radius: float | None = 10.0
    # Density-dependent reproduction cost. ``medium`` regulates the
    # population below the cap, so population and turnover can differ
    # between arms instead of every arm sitting on the ceiling.
    selection_pressure: Any = "medium"

    # Startup conditions (same knobs as the intrinsic-goals experiment).
    initial_agent_resource_level: float | None = 12.0
    initial_resource_count: int | None = 60
    # The development environment caps the population at 50, which pins
    # every arm against the ceiling and leaves selection nothing to act on;
    # density-dependent reproduction cost (selection_pressure) regulates instead.
    max_population: int | None = 300

    # Which loci vary. The cooperation loci (Chromosome D) are drawn
    # uniformly per founder by the service and mutate at every birth. The
    # platform's own initial-diversity layer is off (``None``): it would
    # scramble the learning hyperparameters (Chromosome A) and the reward
    # weights (Chromosome C) too, which leaves brains unable to learn and
    # makes the membership reward worth a different amount to every agent.
    # With ``freeze_platform_genes`` those two chromosomes also stay at their
    # defaults across generations; the action priors (Chromosome B) still
    # evolve, as they do in the intrinsic-evolution world.
    initial_diversity_scale: float | None = None
    freeze_platform_genes: bool = True
    # Temperature of the policies' softmax over Q before innate priors are
    # applied. At 1.0 the per-step Q gaps are too small to move the action
    # mix off the priors; below 1 the learned preference is expressed.
    policy_temperature: float = 0.1

    # Resume: an arm whose ``arm_result.json`` exists is loaded, not re-run.
    resume: bool = True

    alpha: float = 0.05
    #: Independent seeds can run concurrently. Each worker still runs the four
    #: arms of its seed in series. 1 keeps the historical sequential path.
    jobs: int = 1

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.record_interval <= 0:
            raise ValueError("record_interval must be positive.")
        if self.num_replicates <= 0:
            raise ValueError("num_replicates must be positive.")
        if self.jobs < 1:
            raise ValueError("jobs must be at least 1.")
        unknown = set(self.arms) - set(ARM_NAMES)
        if unknown or not self.arms:
            raise ValueError(f"arms must be drawn from {ARM_NAMES}, got {self.arms!r}")
        if not self.cooperation.enabled:
            raise ValueError("cooperation.enabled must be True (the baseline arm disables it itself).")
        if self.policy_temperature <= 0.0:
            raise ValueError("policy_temperature must be positive.")
        if self.initial_diversity_scale is not None and self.initial_diversity_scale < 0.0:
            raise ValueError("initial_diversity_scale must be non-negative or None.")
        object.__setattr__(self, "mutation_mode", MutationMode(self.mutation_mode))
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))
        object.__setattr__(self, "crossover_mode", CrossoverMode(self.crossover_mode))

    def to_dict(self) -> dict[str, Any]:
        """The run configuration as stored in the summary JSON."""
        return {
            "num_steps": self.num_steps,
            "seed": self.seed,
            "num_replicates": self.num_replicates,
            "jobs": self.jobs,
            "arms": list(self.arms),
            "cooperation": self.cooperation.to_dict(),
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.mutation_scale,
            "mutation_mode": self.mutation_mode.value,
            "boundary_mode": self.boundary_mode.value,
            "crossover_enabled": self.crossover_enabled,
            "coparent_max_radius": self.coparent_max_radius,
            "selection_pressure": self.selection_pressure,
            "initial_agent_resource_level": self.initial_agent_resource_level,
            "initial_resource_count": self.initial_resource_count,
            "max_population": self.max_population,
            "initial_diversity_scale": self.initial_diversity_scale,
            "freeze_platform_genes": self.freeze_platform_genes,
            "policy_temperature": self.policy_temperature,
            "alpha": self.alpha,
        }


@dataclass
class ArmResult:
    """Per-arm telemetry collected while running one simulation."""

    arm: str
    steps: list[int] = field(default_factory=list)
    population: list[int] = field(default_factory=list)
    births: list[int] = field(default_factory=list)
    deaths: list[int] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    units_active: list[int] = field(default_factory=list)
    action_mix: dict[str, list[float]] = field(default_factory=dict)
    gene_means: dict[str, list[float]] = field(default_factory=dict)
    #: Per step, the mean over alive learning agents of their policy's own
    #: probability of ``bond`` (before priors); NaN when the action is absent.
    policy_bond_prob: list[float] = field(default_factory=list)
    #: Per step, the mean probability of ``bond`` in the distribution agents
    #: actually sampled from (policy x priors x mask), and the mean rank of
    #: ``bond`` in the policy's own distribution (1 = most preferred).
    acting_bond_prob: list[float] = field(default_factory=list)
    bond_rank: list[float] = field(default_factory=list)
    #: Learning diagnostics: the policies' own probability of ``gather``, the
    #: innate prior share of ``gather`` (normalised action weights), and the
    #: mean peak of the policy distribution (1/n when nothing was learned).
    policy_gather_prob: list[float] = field(default_factory=list)
    prior_gather_share: list[float] = field(default_factory=list)
    policy_peak: list[float] = field(default_factory=list)
    #: Per step, the mean over alive agents of ``recognition x exit_cost`` loci.
    expected_exit_payment: list[float] = field(default_factory=list)
    #: Lifespan (steps) of every agent that died during the run, deaths by
    #: cause, and the maximum generation seen.
    lifespans: list[int] = field(default_factory=list)
    death_causes: dict[str, int] = field(default_factory=dict)
    max_generation: int = 0
    final_population: int = 0
    num_actions: int = 0
    cooperation_summary: dict[str, Any] = field(default_factory=dict)

    def _late(self, values: Sequence[float]) -> list[float]:
        """The last quarter of a series (at least its last point)."""
        n = len(values)
        return list(values[n - max(1, n // 4) :]) if n else []

    def _late_mean(self, values: Sequence[float]) -> float:
        late = [x for x in self._late(values) if not np.isnan(x)]
        return statistics.mean(late) if late else float("nan")

    def summary(self) -> dict[str, Any]:
        mean_pop = statistics.mean(self.population) if self.population else 0.0
        action_share = {a: (statistics.mean(v) if v else 0.0) for a, v in self.action_mix.items()}
        action_share_late = {a: (statistics.mean(self._late(v)) if v else 0.0) for a, v in self.action_mix.items()}
        gene_start = {g: (v[0] if v else float("nan")) for g, v in self.gene_means.items()}
        gene_end = {g: (v[-1] if v else float("nan")) for g, v in self.gene_means.items()}
        gene_change = {g: gene_end[g] - gene_start[g] for g in self.gene_means}
        eep = [x for x in self.expected_exit_payment if not np.isnan(x)]
        pbp_late = [x for x in self._late(self.policy_bond_prob) if not np.isnan(x)]
        failure = self.cooperation_summary.get("failure_modes", {}) if self.cooperation_summary else {}
        counters = self.cooperation_summary.get("counters", {}) if self.cooperation_summary else {}
        enforced = float(counters.get("exits_enforced", 0) or 0)
        uniform = 1.0 / self.num_actions if self.num_actions else float("nan")
        policy_gather_late = self._late_mean(self.policy_gather_prob)
        prior_gather_late = self._late_mean(self.prior_gather_share)
        return {
            "arm": self.arm,
            "final_population": self.final_population,
            "mean_population": mean_pop,
            "peak_population": max(self.population) if self.population else 0,
            "total_births": sum(self.births),
            "total_deaths": sum(self.deaths),
            "starvation_deaths": int(self.death_causes.get("starvation", 0)),
            "combat_deaths": int(self.death_causes.get("combat", 0)),
            "death_causes": dict(self.death_causes),
            "mean_coverage": statistics.mean(self.coverage) if self.coverage else 0.0,
            "final_coverage": self.coverage[-1] if self.coverage else 0.0,
            "mean_units_active": statistics.mean(self.units_active) if self.units_active else 0.0,
            "mean_action_share": action_share,
            "late_action_share": action_share_late,
            "bond_share_late": action_share_late.get("bond", 0.0),
            "policy_bond_prob_late": statistics.mean(pbp_late) if pbp_late else 0.0,
            "acting_bond_prob_late": self._late_mean(self.acting_bond_prob),
            "bond_rank_late": self._late_mean(self.bond_rank),
            "num_actions": self.num_actions,
            "policy_gather_prob_late": policy_gather_late,
            "policy_gather_lift_late": policy_gather_late - uniform,
            "prior_gather_share_late": prior_gather_late,
            "gather_share_over_prior_late": action_share_late.get("gather", 0.0) - prior_gather_late,
            "policy_peak_late": self._late_mean(self.policy_peak),
            "mean_lifespan_at_death": statistics.mean(self.lifespans) if self.lifespans else 0.0,
            "max_generation": self.max_generation,
            "gene_mean_start": gene_start,
            "gene_mean_end": gene_end,
            "gene_mean_change": gene_change,
            "expected_exit_payment_start": eep[0] if eep else float("nan"),
            "expected_exit_payment_end": eep[-1] if eep else float("nan"),
            "change_expected_exit_payment": (eep[-1] - eep[0]) if eep else float("nan"),
            "units_formed": self.cooperation_summary.get("units_formed", 0),
            "mean_unit_lifetime": self.cooperation_summary.get("mean_unit_lifetime", 0.0),
            "forfeits_total": float(counters.get("forfeits", 0.0) or 0.0),
            "forfeits_per_enforced_exit": (
                float(counters.get("forfeits", 0.0) or 0.0) / enforced if enforced else float("nan")
            ),
            "failure_modes": failure,
            "counters": counters,
        }

    # ------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this result (summary plus time series)."""
        series = {name: getattr(self, name) for name in _ARM_SERIES}
        return {
            "arm": self.arm,
            "summary": self.summary(),
            "series": series,
            "action_mix": self.action_mix,
            "gene_means": self.gene_means,
            "lifespans": self.lifespans,
            "death_causes": self.death_causes,
            "max_generation": self.max_generation,
            "final_population": self.final_population,
            "num_actions": self.num_actions,
            "cooperation_summary": self.cooperation_summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArmResult:
        result = cls(arm=str(data["arm"]))
        for name in _ARM_SERIES:
            setattr(result, name, list(data.get("series", {}).get(name, [])))
        result.action_mix = {k: list(v) for k, v in data.get("action_mix", {}).items()}
        result.gene_means = {k: list(v) for k, v in data.get("gene_means", {}).items()}
        result.lifespans = list(data.get("lifespans", []))
        result.death_causes = dict(data.get("death_causes", {}))
        result.max_generation = int(data.get("max_generation", 0))
        result.final_population = int(data.get("final_population", 0))
        result.num_actions = int(data.get("num_actions", 0))
        result.cooperation_summary = dict(data.get("cooperation_summary", {}))
        return result


_ARM_SERIES: tuple[str, ...] = (
    "steps",
    "population",
    "births",
    "deaths",
    "coverage",
    "units_active",
    "policy_bond_prob",
    "acting_bond_prob",
    "bond_rank",
    "policy_gather_prob",
    "prior_gather_share",
    "policy_peak",
    "expected_exit_payment",
)

ARM_RESULT_FILE = "arm_result.json"
REPLICATE_SUMMARY_FILE = "replicate_summary.json"


@dataclass
class ReplicateResult:
    index: int
    seed: int
    arms: dict[str, ArmResult]


@dataclass
class IntrinsicCooperationResult:
    replicates: list[ReplicateResult]
    aggregate: dict[str, Any] | None = None
    hypotheses: list[dict[str, Any]] | None = None
    summary_path: str | None = None
    figure_path: str | None = None

    @property
    def arms(self) -> dict[str, ArmResult]:
        return self.replicates[0].arms


class IntrinsicCooperationExperiment:
    """Run the baseline / placebo / cooperation comparison and persist artifacts."""

    def __init__(self, base_config: SimulationConfig, config: IntrinsicCooperationExperimentConfig | None = None):
        self.base_config = base_config
        self.config = config or IntrinsicCooperationExperimentConfig()

    # ------------------------------------------------------------ public
    def run(self) -> IntrinsicCooperationResult:
        os.makedirs(self.config.output_dir, exist_ok=True)
        replicates = self._run_replicates()

        aggregate = self._aggregate(replicates) if self.config.num_replicates > 1 else None
        hypotheses = self._test_hypotheses(replicates) if self.config.num_replicates > 1 else None
        result = IntrinsicCooperationResult(replicates=replicates, aggregate=aggregate, hypotheses=hypotheses)

        payload = {
            "config": {**self.config.to_dict(), "git_commit": git_commit()},
            "replicates": self._replicate_summaries(replicates),
            "aggregate": aggregate,
            "hypotheses": hypotheses,
        }
        result.summary_path = os.path.join(self.config.output_dir, "intrinsic_cooperation_summary.json")
        with open(result.summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        result.figure_path = self._maybe_plot(replicates)
        logger.info("intrinsic_cooperation_complete", summary_path=result.summary_path)
        return result

    # --------------------------------------------------------- internals
    def _replicate_dir(self, index: int) -> str:
        if self.config.num_replicates > 1:
            return os.path.join(self.config.output_dir, f"rep_{index:02d}")
        return self.config.output_dir

    def _run_replicates(self) -> list[ReplicateResult]:
        planned = [(index, self.config.seed + index) for index in range(self.config.num_replicates)]
        workers = min(self.config.jobs, len(planned))
        if workers <= 1:
            replicates: list[ReplicateResult] = []
            for index, seed in planned:
                logger.info("intrinsic_cooperation_replicate_start", replicate=index, seed=seed)
                replicates.append(self._run_replicate(index, seed))
                self._write_progress(completed=len(replicates), total=len(planned), seed=seed)
            return replicates

        logger.info(
            "intrinsic_cooperation_parallel_start",
            jobs=workers,
            num_replicates=len(planned),
        )
        ordered: list[ReplicateResult | None] = [None] * len(planned)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {
                pool.submit(_run_replicate_worker, self.base_config, self.config, index, seed): (index, seed)
                for index, seed in planned
            }
            finished = 0
            for future in as_completed(futures):
                index, seed = futures[future]
                ordered[index] = future.result()
                finished += 1
                logger.info(
                    "intrinsic_cooperation_replicate_done",
                    replicate=index,
                    seed=seed,
                    completed=finished,
                    total=len(planned),
                )
                self._write_progress(completed=finished, total=len(planned), seed=seed)
        missing = [index for index, result in enumerate(ordered) if result is None]
        if missing:
            raise RuntimeError(f"replicate workers returned no result for indices {missing}")
        return [result for result in ordered if result is not None]

    def _write_progress(self, completed: int, total: int, seed: int) -> None:
        path = os.path.join(self.config.output_dir, "progress.json")
        payload = {
            "completed_replicates": completed,
            "total_replicates": total,
            "last_seed": seed,
            "updated_at": time.time(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _run_replicate(self, index: int, seed: int) -> ReplicateResult:
        rep_dir = self._replicate_dir(index)
        os.makedirs(rep_dir, exist_ok=True)
        arms = {arm: self._load_or_run_arm(arm, seed, rep_dir) for arm in self.config.arms}
        replicate = ReplicateResult(index=index, seed=seed, arms=arms)
        # Written as soon as the replicate finishes, so a run that dies later
        # still leaves every completed seed's summary behind.
        with open(os.path.join(rep_dir, REPLICATE_SUMMARY_FILE), "w", encoding="utf-8") as handle:
            json.dump(self._replicate_summaries([replicate])[0], handle, indent=2, default=str)
        return replicate

    def _load_or_run_arm(self, arm: str, seed: int, rep_dir: str) -> ArmResult:
        path = os.path.join(rep_dir, arm, ARM_RESULT_FILE)
        if self.config.resume and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                result = ArmResult.from_dict(json.load(handle))
            logger.info("intrinsic_cooperation_arm_resumed", arm=arm, seed=seed, path=path)
            return result
        result = self._run_arm(arm, seed, rep_dir)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, default=str)
        return result

    def _build_run_config(self, arm: str, seed: int, out_dir: str) -> SimulationConfig:
        run_config = self.base_config.copy()
        scale = self.config.initial_diversity_scale
        if scale:
            run_config.initial_diversity = InitialDiversityConfig(
                mode=SeedingMode.INDEPENDENT_MUTATION,
                mutation_rate=1.0,
                mutation_scale=scale,
                mutation_mode=self.config.mutation_mode,
                boundary_mode=self.config.boundary_mode,
                seed=seed,
            )
        else:
            run_config.initial_diversity = InitialDiversityConfig(mode=SeedingMode.NONE)
        run_config.learning.policy_temperature = float(self.config.policy_temperature)
        if self.config.initial_agent_resource_level is not None:
            run_config.agent_behavior.initial_resource_level = float(self.config.initial_agent_resource_level)
        if self.config.initial_resource_count is not None:
            run_config.resources.initial_resources = int(self.config.initial_resource_count)
        if self.config.max_population is not None:
            run_config.population.max_population = int(self.config.max_population)
        coop = self.config.cooperation
        if arm == "baseline":
            run_config.cooperation = CooperationConfig(enabled=False)
        else:
            run_config.cooperation = CooperationConfig(
                **{
                    **coop.to_dict(),
                    "enabled": True,
                    "mode": "placebo" if arm == "placebo" else "active",
                    "membership_reward_scale": 0.0 if arm == "cooperation_no_motive" else coop.membership_reward_scale,
                    "synergy": 0.0 if arm == "cooperation_zero_synergy" else coop.synergy,
                    "seed": seed if coop.seed is None else coop.seed,
                    "output_dir": out_dir,
                    "record_interval": self.config.record_interval,
                }
            )
        return run_config

    @staticmethod
    def _freeze_platform_genes(agent: Any) -> None:
        """Pin every locus outside Chromosome B and the cooperation loci.

        Children inherit the flags, so the learning hyperparameters and the
        reward weights stay at their defaults for the whole run while the
        action priors and the cooperation loci evolve.
        """
        chromosome = getattr(agent, "hyperparameter_chromosome", None)
        if chromosome is None:
            return
        frozen = type(chromosome)(
            genes=tuple(
                dataclasses.replace(g, evolvable=False) if g.name in FROZEN_PLATFORM_GENES else g
                for g in chromosome.genes
            )
        )
        agent.hyperparameter_chromosome = frozen

    def _make_policy(self, seed: int) -> IntrinsicEvolutionPolicy:
        return IntrinsicEvolutionPolicy(
            enabled=True,
            mutation_rate=self.config.mutation_rate,
            mutation_scale=self.config.mutation_scale,
            mutation_mode=self.config.mutation_mode,
            boundary_mode=self.config.boundary_mode,
            crossover_enabled=self.config.crossover_enabled,
            crossover_mode=self.config.crossover_mode,
            coparent_max_radius=self.config.coparent_max_radius,
            selection_pressure=self.config.selection_pressure,
            seed=seed,
        )

    def _run_arm(self, arm: str, seed: int, rep_dir: str) -> ArmResult:
        logger.info("intrinsic_cooperation_arm_start", arm=arm, seed=seed)
        path = os.path.join(rep_dir, arm)
        os.makedirs(path, exist_ok=True)
        run_config = self._build_run_config(arm, seed, path)
        policy = self._make_policy(seed)
        result = ArmResult(arm=arm)
        for action in TRACKED_ACTIONS:
            result.action_mix[action] = []
        for gene in COOPERATION_GENE_NAMES:
            result.gene_means[gene] = []
        # Previous step's alive agents by id, kept as objects so the cause of
        # death can be read off an agent after the environment dropped it.
        prev: dict[str, Any] = {}
        birth_step: dict[str, int] = {}

        def _on_environment_ready(environment: Any) -> None:
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = random.Random(seed)
            founders = list(environment.alive_agent_objects)
            if self.config.freeze_platform_genes:
                for agent in founders:
                    self._freeze_platform_genes(agent)
            prev.update({a.agent_id: a for a in founders})
            birth_step.update(dict.fromkeys(prev, 0))
            result.num_actions = max((len(getattr(a, "actions", None) or []) for a in founders), default=0)

        def _on_step_end(environment: Any, step: int) -> None:
            alive = list(environment.alive_agent_objects)
            current = {a.agent_id: a for a in alive}
            born = [agent_id for agent_id in current if agent_id not in prev]
            died = [agent_id for agent_id in prev if agent_id not in current]
            for agent_id in born:
                birth_step[agent_id] = step
            for agent_id in died:
                result.lifespans.append(step - birth_step.pop(agent_id, step))
                cause = getattr(prev[agent_id], "death_cause", None) or "unknown"
                result.death_causes[cause] = result.death_causes.get(cause, 0) + 1
            prev.clear()
            prev.update(current)
            if step % self.config.record_interval != 0:
                return
            result.steps.append(step)
            result.population.append(len(alive))
            result.births.append(len(born))
            result.deaths.append(len(died))
            result.max_generation = max([result.max_generation] + [int(getattr(a, "generation", 0)) for a in alive])
            self._record_action_mix(result, alive)
            self._record_gene_means(result, alive)
            self._record_policy_telemetry(result, alive)
            service = cooperation_service_of(environment)
            if service is not None and service.timeseries:
                row = service.timeseries[-1]
                result.coverage.append(float(row["coverage"]))
                result.units_active.append(int(row["units_active"]))
            else:
                result.coverage.append(0.0)
                result.units_active.append(0)

        environment = run_simulation(
            num_steps=self.config.num_steps,
            config=run_config,
            path=path,
            save_config=True,
            seed=seed,
            disable_console_logging=True,
            on_environment_ready=_on_environment_ready,
            on_step_end=_on_step_end,
        )
        result.final_population = len(list(environment.alive_agent_objects))
        service = cooperation_service_of(environment)
        if service is not None:
            service.close(path)
            result.cooperation_summary = service.summary()
        return result

    @staticmethod
    def _record_action_mix(result: ArmResult, alive: Sequence[Any]) -> None:
        counts: Counter = Counter(getattr(a, "last_action_name", None) for a in alive)
        counts.pop(None, None)
        denom = float(sum(counts.values())) or 1.0
        for action in result.action_mix:
            result.action_mix[action].append(counts.get(action, 0) / denom)

    @staticmethod
    def _record_gene_means(result: ArmResult, alive: Sequence[Any]) -> None:
        per_gene: dict[str, list[float]] = {g: [] for g in COOPERATION_GENE_NAMES}
        eep: list[float] = []
        for agent in alive:
            chromosome = getattr(agent, "hyperparameter_chromosome", None)
            if chromosome is None or chromosome.get_gene("coop_scope") is None:
                continue
            values = gene_values(chromosome)
            for g in COOPERATION_GENE_NAMES:
                per_gene[g].append(values[g])
            eep.append(values["coop_recognition"] * values["coop_exit_cost"])
        for g in COOPERATION_GENE_NAMES:
            vals = per_gene[g]
            result.gene_means[g].append(statistics.mean(vals) if vals else float("nan"))
        result.expected_exit_payment.append(statistics.mean(eep) if eep else float("nan"))

    @staticmethod
    def _record_policy_telemetry(result: ArmResult, alive: Sequence[Any]) -> None:
        """Per-step means of what the policies prefer versus what they are born preferring.

        From each agent's last decision: the policy's own probability of
        ``bond`` and of ``gather`` (before priors), the probability of ``bond``
        in the distribution it actually sampled from, the rank of ``bond`` in
        the policy's own ordering, the policy's peak probability, and the
        innate prior share of ``gather`` (normalised action weights).
        """
        policy_bond: list[float] = []
        acting_bond: list[float] = []
        bond_rank: list[float] = []
        policy_gather: list[float] = []
        prior_gather: list[float] = []
        policy_peak: list[float] = []
        for agent in alive:
            actions = getattr(agent, "actions", None) or []
            names = [a.name for a in actions]
            if not names:
                continue
            weights = np.array([max(0.0, float(getattr(a, "weight", 0.0))) for a in actions], dtype=float)
            if "gather" in names and weights.sum() > 0:
                prior_gather.append(float(weights[names.index("gather")] / weights.sum()))
            module = getattr(getattr(agent, "behavior", None), "decision_module", None)
            dist = getattr(module, "last_policy_probs", None)
            if dist is None or len(dist) != len(names):
                continue
            dist = np.asarray(dist, dtype=float)
            policy_peak.append(float(dist.max()))
            if "gather" in names:
                policy_gather.append(float(dist[names.index("gather")]))
            if "bond" in names:
                i = names.index("bond")
                policy_bond.append(float(dist[i]))
                bond_rank.append(float(1 + int((dist > dist[i]).sum())))
                acting = getattr(module, "last_action_probs", None)
                if acting is not None and len(acting) == len(names):
                    acting_bond.append(float(acting[i]))

        def _mean(values: list[float]) -> float:
            return statistics.mean(values) if values else float("nan")

        result.policy_bond_prob.append(_mean(policy_bond))
        result.acting_bond_prob.append(_mean(acting_bond))
        result.bond_rank.append(_mean(bond_rank))
        result.policy_gather_prob.append(_mean(policy_gather))
        result.prior_gather_share.append(_mean(prior_gather))
        result.policy_peak.append(_mean(policy_peak))

    # ------------------------------------------------------- aggregation
    @staticmethod
    def _replicate_summaries(replicates: Sequence[ReplicateResult]) -> list[dict[str, Any]]:
        return [replicate_summary(r.index, r.seed, {arm: r.arms[arm].summary() for arm in r.arms}) for r in replicates]

    def _aggregate(self, replicates: Sequence[ReplicateResult]) -> dict[str, Any]:
        return aggregate_summaries(self._replicate_summaries(replicates), self.config.arms)

    def _test_hypotheses(self, replicates: Sequence[ReplicateResult]) -> list[dict[str, Any]]:
        return test_hypotheses(self._replicate_summaries(replicates), self.config.arms, self.config.alpha)

    # ------------------------------------------------------------- plots
    def _maybe_plot(self, replicates: Sequence[ReplicateResult]) -> str | None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        colors = {
            "baseline": "#7a7a7a",
            "placebo": "#c08a2e",
            "cooperation": "#2b6cb0",
            "cooperation_no_motive": "#7fb3e6",
            "cooperation_zero_synergy": "#4a9d7f",
        }
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_pop, ax_cov, ax_bond, ax_genes = axes.ravel()
        for arm in self.config.arms:
            series = [r.arms[arm] for r in replicates]
            length = min(len(s.population) for s in series)
            if length == 0:
                continue
            steps = series[0].steps[:length]
            pop = np.mean([s.population[:length] for s in series], axis=0)
            ax_pop.plot(steps, pop, color=colors[arm], label=arm)
            cov = np.mean([s.coverage[:length] for s in series], axis=0)
            ax_cov.plot(steps, cov, color=colors[arm], label=arm)
            if arm != "baseline":
                bond = np.nanmean([s.policy_bond_prob[:length] for s in series], axis=0)
                ax_bond.plot(steps, bond, color=colors[arm], label=arm)
                for gene, style in (("coop_exit_cost", "-"), ("coop_scope", "--"), ("coop_recognition", ":")):
                    vals = np.nanmean([s.gene_means[gene][:length] for s in series], axis=0)
                    ax_genes.plot(steps, vals, style, color=colors[arm], label=f"{arm}: {gene[5:]}")
        ax_pop.set_title("Population")
        ax_cov.set_title("Share of agents in a unit")
        ax_bond.set_title("Policies' own probability of bond (before priors)")
        ax_genes.set_title("Population mean of settings loci")
        for ax in axes.ravel():
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(self.config.output_dir, "intrinsic_cooperation_comparison.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


__all__ = [
    "ARM_CONTRASTS",
    "ARM_NAMES",
    "ARM_RESULT_FILE",
    "CONFIRMATORY_SETTING_LOCI",
    "DEFAULT_ARMS",
    "EXPLORATORY",
    "EXPLORATORY_SETTING_LOCI",
    "FROZEN_PLATFORM_GENES",
    "HYPOTHESES",
    "LEARNING_CHECKS",
    "MANIPULATION_CHECKS",
    "OUT_OF_FAMILY_FLAGS",
    "REPLICATE_SUMMARY_FILE",
    "TRACKED_ACTIONS",
    "ArmResult",
    "IntrinsicCooperationExperiment",
    "IntrinsicCooperationExperimentConfig",
    "IntrinsicCooperationResult",
    "ReplicateResult",
    "aggregate_summaries",
    "expected_test_keys",
    "git_commit",
    "holm",
    "hypothesis_kind",
    "metric_arrays",
    "replicate_summary",
    "scalar_metrics",
    "test_hypotheses",
]
