"""Intrinsic cooperation experiment runner.

Builds on the intrinsic-evolution world (agents gather, reproduce with
genetic inheritance, and each offspring learns with its own brain) and asks:

    **When agents can bond into cooperative units -- and both the terms of
    those units and the taste for them are heritable -- does cooperation
    emerge, which institutions does selection favour, and what does it do to
    the population?**

Three arms share every seed and every other setting:

- ``baseline``    -- the unmodified world: eight actions, no cooperation loci.
- ``placebo``     -- the ``bond``/``leave`` actions and the cooperation loci
  exist but ``bond`` never forms a unit. This separates the cost of carrying
  two extra actions and twelve extra genes (a larger policy output, extra
  mutation load, wasted actions) from the effect of the institution itself.
- ``cooperation`` -- units form, pool, pay out, insure and enforce.

Per arm the runner records population, births and deaths, the action mix
(including ``bond`` and ``leave``), the population means of the cooperation
loci, and the cooperation service's own telemetry (coverage, units, failure
modes). With several paired replicates it reports seed-paired contrasts with
Wilcoxon signed-rank tests and a Holm correction over the registered family
(see ``docs/research/experiments/cooperation/SpatialDesign.md``).
"""

from __future__ import annotations

import json
import os
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from farm.config import SimulationConfig
from farm.core.cooperation import CooperationConfig
from farm.core.cooperation.actions import COOPERATION_ACTION_NAMES, cooperation_service_of
from farm.core.cooperation.genes import COOPERATION_GENE_NAMES, gene_values
from farm.core.hyperparameter_chromosome import BoundaryMode, CrossoverMode, MutationMode
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode
from farm.core.simulation import run_simulation
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy
from farm.utils.logging import get_logger

logger = get_logger(__name__)

ARM_NAMES: tuple[str, ...] = ("baseline", "placebo", "cooperation")
BASELINE_ARM: str = "baseline"

#: Seed-paired contrasts, ``name -> (baseline arm, treatment arm)``.
ARM_CONTRASTS: dict[str, tuple[str, str]] = {
    "cooperation_minus_placebo": ("placebo", "cooperation"),
    "cooperation_minus_baseline": ("baseline", "cooperation"),
    "placebo_minus_baseline": ("baseline", "placebo"),
}

TRACKED_ACTIONS: tuple[str, ...] = (
    "move",
    "gather",
    "share",
    "attack",
    "defend",
    "reproduce",
    "pass",
) + COOPERATION_ACTION_NAMES

#: The registered one-sided hypotheses: ``key -> (metric, contrast, direction, statement)``.
#: Tested across replicates with a Wilcoxon signed-rank test, Holm-corrected as one family.
HYPOTHESES: dict[str, tuple[str, str, str, str]] = {
    "S1": (
        "mean_coverage",
        "cooperation_minus_placebo",
        "greater",
        "Cooperation takes hold: the share of agents in a unit exceeds the placebo's (which is zero by construction).",
    ),
    "S2": (
        "bond_share_late",
        "cooperation_minus_placebo",
        "greater",
        (
            "The brain learns to bond: in the last quarter of the run, bond is chosen more often in the cooperation arm "
            "than in the placebo, where it never pays."
        ),
    ),
    "S3": (
        "mean_population",
        "cooperation_minus_placebo",
        "greater",
        "Cooperation sustains a larger population than the same world without working units.",
    ),
    "S4": (
        "coop_gene_drift",
        "cooperation_minus_placebo",
        "greater",
        (
            "Selection acts on the institution: the settings loci move further from their founding means in the "
            "cooperation arm than under neutral drift in the placebo."
        ),
    ),
    "S5": (
        "mean_coop_exit_cost_end",
        "cooperation_minus_placebo",
        "greater",
        "Selection favours binding terms: the population mean exit-cost locus ends higher than under neutral drift.",
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


@dataclass(frozen=True)
class IntrinsicCooperationExperimentConfig:
    """Configuration for :class:`IntrinsicCooperationExperiment`."""

    num_steps: int = 1000
    seed: int = 101
    num_replicates: int = 1
    output_dir: str = "experiments/intrinsic_cooperation"
    record_interval: int = 1
    arms: tuple[str, ...] = ARM_NAMES

    # Cooperation settings shared by the placebo and cooperation arms.
    cooperation: CooperationConfig = field(default_factory=lambda: CooperationConfig(enabled=True))

    # Per-reproduction inheritance. Crossover is on so a member's partner can
    # be its co-parent (CooperationConfig.partner_coparent).
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.REFLECT
    crossover_enabled: bool = True
    crossover_mode: CrossoverMode = CrossoverMode.UNIFORM
    coparent_max_radius: float | None = 10.0
    selection_pressure: Any = "low"

    # Startup conditions (same knobs as the intrinsic-goals experiment).
    initial_agent_resource_level: float | None = 12.0
    initial_resource_count: int | None = 60
    max_population: int | None = None
    initial_diversity_scale: float = 0.2

    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.record_interval <= 0:
            raise ValueError("record_interval must be positive.")
        if self.num_replicates <= 0:
            raise ValueError("num_replicates must be positive.")
        unknown = set(self.arms) - set(ARM_NAMES)
        if unknown or not self.arms:
            raise ValueError(f"arms must be drawn from {ARM_NAMES}, got {self.arms!r}")
        if not self.cooperation.enabled:
            raise ValueError("cooperation.enabled must be True (the baseline arm disables it itself).")
        object.__setattr__(self, "mutation_mode", MutationMode(self.mutation_mode))
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))
        object.__setattr__(self, "crossover_mode", CrossoverMode(self.crossover_mode))


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
    final_population: int = 0
    cooperation_summary: dict[str, Any] = field(default_factory=dict)

    def _late(self, values: Sequence[float]) -> list[float]:
        n = len(values)
        return list(values[max(0, n - n // 4) :]) if n else []

    def summary(self) -> dict[str, Any]:
        mean_pop = statistics.mean(self.population) if self.population else 0.0
        action_share = {a: (statistics.mean(v) if v else 0.0) for a, v in self.action_mix.items()}
        action_share_late = {a: (statistics.mean(self._late(v)) if v else 0.0) for a, v in self.action_mix.items()}
        gene_start = {g: (v[0] if v else float("nan")) for g, v in self.gene_means.items()}
        gene_end = {g: (v[-1] if v else float("nan")) for g, v in self.gene_means.items()}
        drift = [abs(gene_end[g] - gene_start[g]) for g in self.gene_means if g.startswith("coop_")]
        drift = [d for d in drift if not np.isnan(d)]
        failure = self.cooperation_summary.get("failure_modes", {}) if self.cooperation_summary else {}
        return {
            "arm": self.arm,
            "final_population": self.final_population,
            "mean_population": mean_pop,
            "peak_population": max(self.population) if self.population else 0,
            "total_births": sum(self.births),
            "total_deaths": sum(self.deaths),
            "mean_coverage": statistics.mean(self.coverage) if self.coverage else 0.0,
            "final_coverage": self.coverage[-1] if self.coverage else 0.0,
            "mean_units_active": statistics.mean(self.units_active) if self.units_active else 0.0,
            "mean_action_share": action_share,
            "late_action_share": action_share_late,
            "bond_share_late": action_share_late.get("bond", 0.0),
            "gene_mean_start": gene_start,
            "gene_mean_end": gene_end,
            "coop_gene_drift": statistics.mean(drift) if drift else 0.0,
            "units_formed": self.cooperation_summary.get("units_formed", 0),
            "mean_unit_lifetime": self.cooperation_summary.get("mean_unit_lifetime", 0.0),
            "failure_modes": failure,
            "counters": self.cooperation_summary.get("counters", {}),
        }


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
        replicates: list[ReplicateResult] = []
        for index in range(self.config.num_replicates):
            seed = self.config.seed + index
            logger.info("intrinsic_cooperation_replicate_start", replicate=index, seed=seed)
            replicates.append(self._run_replicate(index, seed))

        aggregate = self._aggregate(replicates) if self.config.num_replicates > 1 else None
        hypotheses = self._test_hypotheses(replicates) if self.config.num_replicates > 1 else None
        result = IntrinsicCooperationResult(replicates=replicates, aggregate=aggregate, hypotheses=hypotheses)

        payload = {
            "config": {
                "num_steps": self.config.num_steps,
                "seed": self.config.seed,
                "num_replicates": self.config.num_replicates,
                "arms": list(self.config.arms),
                "cooperation": self.config.cooperation.to_dict(),
                "mutation_rate": self.config.mutation_rate,
                "mutation_scale": self.config.mutation_scale,
                "crossover_enabled": self.config.crossover_enabled,
                "coparent_max_radius": self.config.coparent_max_radius,
                "selection_pressure": self.config.selection_pressure,
                "max_population": self.config.max_population,
                "alpha": self.config.alpha,
            },
            "replicates": [
                {"index": r.index, "seed": r.seed, **{arm: r.arms[arm].summary() for arm in r.arms}} for r in replicates
            ],
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
    def _run_replicate(self, index: int, seed: int) -> ReplicateResult:
        rep_dir = (
            os.path.join(self.config.output_dir, f"rep_{index:02d}")
            if self.config.num_replicates > 1
            else self.config.output_dir
        )
        os.makedirs(rep_dir, exist_ok=True)
        arms = {arm: self._run_arm(arm, seed, rep_dir) for arm in self.config.arms}
        return ReplicateResult(index=index, seed=seed, arms=arms)

    def _build_run_config(self, arm: str, seed: int, out_dir: str) -> SimulationConfig:
        run_config = self.base_config.copy()
        run_config.initial_diversity = InitialDiversityConfig(
            mode=SeedingMode.INDEPENDENT_MUTATION,
            mutation_rate=1.0,
            mutation_scale=self.config.initial_diversity_scale,
            mutation_mode=self.config.mutation_mode,
            boundary_mode=self.config.boundary_mode,
            seed=seed,
        )
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
                    "seed": seed if coop.seed is None else coop.seed,
                    "output_dir": out_dir,
                    "record_interval": self.config.record_interval,
                }
            )
        return run_config

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
        prev_ids: set = set()

        def _on_environment_ready(environment: Any) -> None:
            nonlocal prev_ids
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = random.Random(seed)
            prev_ids = {a.agent_id for a in environment.alive_agent_objects}

        def _on_step_end(environment: Any, step: int) -> None:
            nonlocal prev_ids
            if step % self.config.record_interval != 0:
                return
            alive = list(environment.alive_agent_objects)
            current = {a.agent_id for a in alive}
            result.steps.append(step)
            result.population.append(len(alive))
            result.births.append(len(current - prev_ids))
            result.deaths.append(len(prev_ids - current))
            prev_ids = current
            self._record_action_mix(result, alive)
            self._record_gene_means(result, alive)
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
        for agent in alive:
            chromosome = getattr(agent, "hyperparameter_chromosome", None)
            if chromosome is None or chromosome.get_gene("coop_scope") is None:
                continue
            values = gene_values(chromosome)
            for g in COOPERATION_GENE_NAMES:
                per_gene[g].append(values[g])
        for g in COOPERATION_GENE_NAMES:
            vals = per_gene[g]
            result.gene_means[g].append(statistics.mean(vals) if vals else float("nan"))

    # ------------------------------------------------------- aggregation
    @staticmethod
    def _scalar_metrics(summary: dict[str, Any]) -> dict[str, float]:
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
            "coop_gene_drift": float(summary["coop_gene_drift"]),
        }
        for action, share in summary["mean_action_share"].items():
            metrics[f"action_share[{action}]"] = float(share)
        for gene, value in summary["gene_mean_end"].items():
            metrics[f"mean_{gene}_end"] = float(value)
        return metrics

    def _metric_arrays(self, replicates: Sequence[ReplicateResult]) -> dict[str, dict[str, np.ndarray]]:
        arms = list(self.config.arms)
        per_arm = {arm: [self._scalar_metrics(r.arms[arm].summary()) for r in replicates] for arm in arms}
        names = list(per_arm[arms[0]][0].keys())
        return {arm: {n: np.array([m[n] for m in per_arm[arm]], dtype=float) for n in names} for arm in arms}

    def _aggregate(self, replicates: Sequence[ReplicateResult]) -> dict[str, Any]:
        arrays = self._metric_arrays(replicates)
        arms = list(self.config.arms)
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
            "num_replicates": len(replicates),
            "seeds": [r.seed for r in replicates],
            "per_arm": {arm: {n: describe(arrays[arm][n]) for n in names} for arm in arms},
            "paired_deltas": paired,
        }

    def _test_hypotheses(self, replicates: Sequence[ReplicateResult]) -> list[dict[str, Any]]:
        arrays = self._metric_arrays(replicates)
        results: list[dict[str, Any]] = []
        for key, (metric, contrast, direction, statement) in HYPOTHESES.items():
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
        holm(results, self.config.alpha)
        return results

    # ------------------------------------------------------------- plots
    def _maybe_plot(self, replicates: Sequence[ReplicateResult]) -> str | None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        colors = {"baseline": "#7a7a7a", "placebo": "#c08a2e", "cooperation": "#2b6cb0"}
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
            bond = np.mean([s.action_mix["bond"][:length] for s in series], axis=0)
            ax_bond.plot(steps, bond, color=colors[arm], label=arm)
            if arm != "baseline":
                for gene, style in (("coop_exit_cost", "-"), ("coop_scope", "--"), ("coop_recognition", ":")):
                    vals = np.nanmean([s.gene_means[gene][:length] for s in series], axis=0)
                    ax_genes.plot(steps, vals, style, color=colors[arm], label=f"{arm}: {gene[5:]}")
        ax_pop.set_title("Population")
        ax_cov.set_title("Share of agents in a unit")
        ax_bond.set_title("Share of agents choosing bond")
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
    "HYPOTHESES",
    "TRACKED_ACTIONS",
    "ArmResult",
    "IntrinsicCooperationExperiment",
    "IntrinsicCooperationExperimentConfig",
    "IntrinsicCooperationResult",
    "ReplicateResult",
    "holm",
]
