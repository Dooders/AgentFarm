"""Intrinsic-goals experiment runner.

This runner explores a specific question *inside* the intrinsic-evolution
framework:

    **What happens to a simulation when every agent has a different
    reinforcement-learning goal (loss/reward function)?**

Each agent's per-step RL reward is parameterized by the *Chromosome C* genes
(``reward_*``) defined in :mod:`farm.core.hyperparameter_chromosome` and consumed
by :meth:`farm.core.agent.core.AgentCore._calculate_reward`.  Those genes are
heritable and mutate on reproduction just like every other gene, so a diverse
set of goals seeded at *t=0* co-evolves with survival.

The experiment runs two arms with *identical* seeds and configuration so the
only difference is the agents' objectives:

- ``uniform``  — every agent shares the default reward function (the historical
  AgentFarm reward).  This is the control.
- ``unique``   — every initial agent is given an independently sampled reward
  function, so each one optimizes for a genuinely different goal.  Offspring
  inherit and mutate their parent's goal.

For each arm the runner records per-step population, action-mix, and goal-gene
trajectories, then writes a JSON summary and (when matplotlib is available) a
comparison figure so the behavioural divergence is easy to see.

Compare with :mod:`farm.runners.intrinsic_evolution_experiment`, which evolves
*all* genes (learning hyperparameters + action priors + goals) at once; this
runner isolates the goal genes to make their effect legible.
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
from farm.core.hyperparameter_chromosome import (
    INTRINSIC_REWARD_GENE_NAMES,
    BoundaryMode,
    HyperparameterChromosome,
    MutationMode,
    reward_weights_from_chromosome,
)
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode
from farm.core.simulation import run_simulation
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Action names tracked for the behavioural action-mix telemetry. Only these
# actions are emitted in the summary output; any other runtime action names are
# ignored by the action-mix series.
TRACKED_ACTIONS: Tuple[str, ...] = (
    "move",
    "gather",
    "share",
    "attack",
    "reproduce",
    "defend",
    "pass",
)


def sample_goal_chromosome(
    base_chromosome: HyperparameterChromosome,
    rng: random.Random,
    gene_names: Sequence[str] = INTRINSIC_REWARD_GENE_NAMES,
) -> HyperparameterChromosome:
    """Return a copy of *base_chromosome* with randomized intrinsic-goal genes.

    Each named gene is drawn independently and uniformly from its own
    ``[min_value, max_value]`` bounds, yielding a unique reward function while
    leaving every non-goal gene (learning hyperparameters, action priors)
    untouched.
    """
    overrides: Dict[str, float] = {}
    for name in gene_names:
        gene = base_chromosome.get_gene(name)
        if gene is None:
            continue
        overrides[name] = rng.uniform(gene.min_value, gene.max_value)
    if not overrides:
        return base_chromosome
    return base_chromosome.with_overrides(overrides)


def assign_unique_goals(
    environment: Any,
    rng: random.Random,
    gene_names: Sequence[str] = INTRINSIC_REWARD_GENE_NAMES,
) -> int:
    """Give every alive agent in *environment* an independently sampled goal.

    Returns the number of agents whose goal was reassigned.  The reward genes
    are read straight off the chromosome by ``AgentCore._calculate_reward`` and
    are not projected into ``DecisionConfig``, so simply replacing
    ``agent.hyperparameter_chromosome`` is sufficient for the new goal to take
    effect on the next step.
    """
    count = 0
    for agent in list(environment.alive_agent_objects):
        chromosome = getattr(agent, "hyperparameter_chromosome", None)
        if chromosome is None:
            continue
        agent.hyperparameter_chromosome = sample_goal_chromosome(
            chromosome, rng, gene_names
        )
        count += 1
    return count


@dataclass(frozen=True)
class IntrinsicGoalsExperimentConfig:
    """Configuration for :class:`IntrinsicGoalsExperiment`."""

    num_steps: int = 600
    seed: int = 42
    output_dir: str = "experiments/intrinsic_goals"
    record_interval: int = 1

    # Number of paired replicates.  Replicate ``r`` runs both arms with
    # ``seed + r`` so the two arms stay matched while the ensemble samples
    # run-to-run stochasticity.  ``num_replicates > 1`` unlocks the aggregate
    # statistics (per-arm mean/std and paired t-tests on the unique-vs-uniform
    # deltas) that are required to actually *assess* the effect.
    num_replicates: int = 1

    # Genes that define an agent's goal and are diversified in the ``unique`` arm.
    goal_genes: Tuple[str, ...] = INTRINSIC_REWARD_GENE_NAMES

    # Per-reproduction mutation applied to ALL genes (so goals keep drifting and
    # selection can act on them).  Reflect boundary keeps mutated goals off the
    # absorbing edges of their range.
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.REFLECT

    # Density-dependent reproduction cost preset ("none"/"low"/"medium"/"high"
    # or a float in [0, 1]).  A little pressure makes selection on goals matter.
    selection_pressure: Any = "low"

    # Startup conditions tuned for a population that survives long enough to
    # observe goal-driven divergence (the default dev config is boom/bust).
    initial_agent_resource_level: Optional[float] = 12.0
    initial_resource_count: Optional[int] = 60

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.record_interval <= 0:
            raise ValueError("record_interval must be positive.")
        if self.num_replicates <= 0:
            raise ValueError("num_replicates must be positive.")
        object.__setattr__(self, "mutation_mode", MutationMode(self.mutation_mode))
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))


@dataclass
class ArmResult:
    """Per-arm telemetry collected while running one simulation."""

    arm: str
    steps: List[int] = field(default_factory=list)
    population: List[int] = field(default_factory=list)
    births: List[int] = field(default_factory=list)
    deaths: List[int] = field(default_factory=list)
    # action_mix[action] -> list of per-step fractions of alive agents whose
    # most recent action was `action`.
    action_mix: Dict[str, List[float]] = field(default_factory=dict)
    # gene_means[gene] -> list of per-step population mean of that goal gene.
    gene_means: Dict[str, List[float]] = field(default_factory=dict)
    final_population: int = 0
    goal_diversity_start: Dict[str, float] = field(default_factory=dict)
    goal_diversity_end: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of this arm."""
        mean_pop = statistics.mean(self.population) if self.population else 0.0
        peak_pop = max(self.population) if self.population else 0
        action_share = {
            action: (statistics.mean(values) if values else 0.0)
            for action, values in self.action_mix.items()
        }
        return {
            "arm": self.arm,
            "final_population": self.final_population,
            "mean_population": mean_pop,
            "peak_population": peak_pop,
            "total_births": sum(self.births),
            "total_deaths": sum(self.deaths),
            "mean_action_share": action_share,
            "goal_gene_mean_start": {
                gene: (values[0] if values else 0.0)
                for gene, values in self.gene_means.items()
            },
            "goal_gene_mean_end": {
                gene: (values[-1] if values else 0.0)
                for gene, values in self.gene_means.items()
            },
            "goal_diversity_start": self.goal_diversity_start,
            "goal_diversity_end": self.goal_diversity_end,
        }


@dataclass
class ReplicateResult:
    """Both arms of a single paired replicate (one shared seed)."""

    index: int
    seed: int
    uniform: ArmResult
    unique: ArmResult
    comparison: Dict[str, Any]


@dataclass
class IntrinsicGoalsResult:
    """Combined result of the experiment.

    ``uniform``/``unique``/``comparison`` mirror the first replicate so a
    single-replicate run behaves exactly as before.  ``replicates`` holds every
    paired run and ``aggregate`` holds the cross-replicate statistics (present
    only when ``num_replicates > 1``).
    """

    uniform: ArmResult
    unique: ArmResult
    comparison: Dict[str, Any]
    replicates: List[ReplicateResult] = field(default_factory=list)
    aggregate: Optional[Dict[str, Any]] = None
    summary_path: Optional[str] = None
    figure_path: Optional[str] = None


class IntrinsicGoalsExperiment:
    """Run the uniform-vs-unique-goals comparison and persist artifacts."""

    def __init__(
        self,
        base_config: SimulationConfig,
        config: Optional[IntrinsicGoalsExperimentConfig] = None,
    ) -> None:
        self.base_config = base_config
        self.config = config or IntrinsicGoalsExperimentConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> IntrinsicGoalsResult:
        """Run every paired replicate and write the summary + figure(s)."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        replicates: List[ReplicateResult] = []
        for index in range(self.config.num_replicates):
            seed = self.config.seed + index
            logger.info(
                "intrinsic_goals_replicate_start",
                replicate=index,
                seed=seed,
                total=self.config.num_replicates,
            )
            replicates.append(self._run_replicate(index, seed))

        first = replicates[0]
        aggregate = (
            self._aggregate(replicates) if self.config.num_replicates > 1 else None
        )

        result = IntrinsicGoalsResult(
            uniform=first.uniform,
            unique=first.unique,
            comparison=first.comparison,
            replicates=replicates,
            aggregate=aggregate,
        )

        summary_payload = {
            "config": {
                "num_steps": self.config.num_steps,
                "seed": self.config.seed,
                "num_replicates": self.config.num_replicates,
                "goal_genes": list(self.config.goal_genes),
                "mutation_rate": self.config.mutation_rate,
                "mutation_scale": self.config.mutation_scale,
                "selection_pressure": self.config.selection_pressure,
            },
            "replicates": [
                {
                    "index": rep.index,
                    "seed": rep.seed,
                    "uniform": rep.uniform.summary(),
                    "unique": rep.unique.summary(),
                    "comparison": rep.comparison,
                }
                for rep in replicates
            ],
            "aggregate": aggregate,
            # Top-level convenience mirror of the first replicate.
            "uniform": first.uniform.summary(),
            "unique": first.unique.summary(),
            "comparison": first.comparison,
        }
        result.summary_path = os.path.join(
            self.config.output_dir, "intrinsic_goals_summary.json"
        )
        with open(result.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)

        if self.config.num_replicates > 1:
            result.figure_path = self._maybe_plot_aggregate(replicates, aggregate)
        else:
            result.figure_path = self._maybe_plot(first.uniform, first.unique)

        logger.info(
            "intrinsic_goals_experiment_complete",
            summary_path=result.summary_path,
            figure_path=result.figure_path,
            num_replicates=self.config.num_replicates,
        )
        return result

    def _run_replicate(self, index: int, seed: int) -> ReplicateResult:
        """Run both arms with a single shared seed and pair their results."""
        if self.config.num_replicates > 1:
            rep_dir = os.path.join(self.config.output_dir, f"rep_{index:02d}")
        else:
            rep_dir = self.config.output_dir
        os.makedirs(rep_dir, exist_ok=True)

        uniform = self._run_arm("uniform", False, seed, rep_dir)
        unique = self._run_arm("unique", True, seed, rep_dir)
        comparison = self._build_comparison(uniform, unique)
        return ReplicateResult(
            index=index,
            seed=seed,
            uniform=uniform,
            unique=unique,
            comparison=comparison,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_run_config(self) -> SimulationConfig:
        run_config = self.base_config.copy()
        # Isolate the manipulated variable: do NOT diversify any genes at
        # startup.  In the ``unique`` arm we diversify *only* the goal genes
        # ourselves in the environment-ready hook.
        run_config.initial_diversity = InitialDiversityConfig(mode=SeedingMode.NONE)

        if self.config.initial_agent_resource_level is not None:
            run_config.agent_behavior.initial_resource_level = float(
                self.config.initial_agent_resource_level
            )
        if self.config.initial_resource_count is not None:
            run_config.resources.initial_resources = int(
                self.config.initial_resource_count
            )
        return run_config

    def _make_policy(self, seed: int) -> IntrinsicEvolutionPolicy:
        return IntrinsicEvolutionPolicy(
            enabled=True,
            mutation_rate=self.config.mutation_rate,
            mutation_scale=self.config.mutation_scale,
            mutation_mode=self.config.mutation_mode,
            boundary_mode=self.config.boundary_mode,
            selection_pressure=self.config.selection_pressure,
            seed=seed,
        )

    def _run_arm(
        self, arm_name: str, diversify_goals: bool, seed: int, out_dir: str
    ) -> ArmResult:
        logger.info(
            "intrinsic_goals_arm_start",
            arm=arm_name,
            diversify=diversify_goals,
            seed=seed,
        )
        run_config = self._build_run_config()
        policy = self._make_policy(seed)
        # A dedicated RNG for goal sampling keeps the two arms aligned on the
        # shared simulation seed while still giving reproducible goals.
        goal_rng = random.Random(seed + 1)

        arm = ArmResult(arm=arm_name)
        for action in TRACKED_ACTIONS:
            arm.action_mix[action] = []
        for gene in self.config.goal_genes:
            arm.gene_means[gene] = []

        prev_ids: set = set()

        def _on_environment_ready(environment: Any) -> None:
            nonlocal prev_ids
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = random.Random(seed)
            if diversify_goals:
                n = assign_unique_goals(environment, goal_rng, self.config.goal_genes)
                logger.info("intrinsic_goals_assigned", arm=arm_name, agents=n)
            prev_ids = {a.agent_id for a in environment.alive_agent_objects}
            arm.goal_diversity_start = self._goal_diversity(environment)

        def _on_step_end(environment: Any, step: int) -> None:
            nonlocal prev_ids
            if step % self.config.record_interval != 0:
                return
            alive = list(environment.alive_agent_objects)
            current_ids = {a.agent_id for a in alive}
            births = len(current_ids - prev_ids)
            deaths = len(prev_ids - current_ids)
            prev_ids = current_ids

            arm.steps.append(step)
            arm.population.append(len(alive))
            arm.births.append(births)
            arm.deaths.append(deaths)

            self._record_action_mix(arm, alive)
            self._record_gene_means(arm, alive)

        path = os.path.join(out_dir, arm_name)
        os.makedirs(path, exist_ok=True)
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

        arm.final_population = len(list(environment.alive_agent_objects))
        arm.goal_diversity_end = self._goal_diversity(environment)
        return arm

    @staticmethod
    def _record_action_mix(arm: ArmResult, alive: Sequence[Any]) -> None:
        counts: Counter = Counter()
        for agent in alive:
            name = getattr(agent, "last_action_name", None)
            if name is None:
                continue
            counts[name] += 1
        denom = float(sum(counts.values())) if counts else 1.0
        for action in arm.action_mix:
            arm.action_mix[action].append(counts.get(action, 0) / denom)

    def _record_gene_means(self, arm: ArmResult, alive: Sequence[Any]) -> None:
        per_gene: Dict[str, List[float]] = {gene: [] for gene in self.config.goal_genes}
        for agent in alive:
            chromosome = getattr(agent, "hyperparameter_chromosome", None)
            if chromosome is None:
                continue
            weights = reward_weights_from_chromosome(chromosome)
            for gene in self.config.goal_genes:
                if gene in weights:
                    per_gene[gene].append(weights[gene])
        for gene in self.config.goal_genes:
            values = per_gene[gene]
            arm.gene_means[gene].append(statistics.mean(values) if values else 0.0)

    def _goal_diversity(self, environment: Any) -> Dict[str, float]:
        """Population standard deviation of each goal gene (a diversity proxy)."""
        per_gene: Dict[str, List[float]] = {gene: [] for gene in self.config.goal_genes}
        for agent in list(environment.alive_agent_objects):
            chromosome = getattr(agent, "hyperparameter_chromosome", None)
            if chromosome is None:
                continue
            weights = reward_weights_from_chromosome(chromosome)
            for gene in self.config.goal_genes:
                if gene in weights:
                    per_gene[gene].append(weights[gene])
        return {
            gene: (statistics.pstdev(values) if len(values) > 1 else 0.0)
            for gene, values in per_gene.items()
        }

    def _build_comparison(self, uniform: ArmResult, unique: ArmResult) -> Dict[str, Any]:
        u_sum = uniform.summary()
        q_sum = unique.summary()
        action_share_delta = {
            action: q_sum["mean_action_share"].get(action, 0.0)
            - u_sum["mean_action_share"].get(action, 0.0)
            for action in uniform.action_mix
        }
        return {
            "final_population_delta": q_sum["final_population"]
            - u_sum["final_population"],
            "mean_population_delta": q_sum["mean_population"] - u_sum["mean_population"],
            "total_births_delta": q_sum["total_births"] - u_sum["total_births"],
            "total_deaths_delta": q_sum["total_deaths"] - u_sum["total_deaths"],
            "action_share_delta_unique_minus_uniform": action_share_delta,
            "start_goal_diversity": {
                "uniform": u_sum["goal_diversity_start"],
                "unique": q_sum["goal_diversity_start"],
            },
            "end_goal_diversity": {
                "uniform": u_sum["goal_diversity_end"],
                "unique": q_sum["goal_diversity_end"],
            },
        }

    # ------------------------------------------------------------------
    # Aggregation across replicates
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
        """Flatten one arm summary into the scalar metrics we aggregate."""
        metrics: Dict[str, float] = {
            "mean_population": float(summary["mean_population"]),
            "final_population": float(summary["final_population"]),
            "peak_population": float(summary["peak_population"]),
            "total_births": float(summary["total_births"]),
            "total_deaths": float(summary["total_deaths"]),
        }
        for action, share in summary["mean_action_share"].items():
            metrics[f"action_share[{action}]"] = float(share)
        metrics["goal_diversity_end_sum"] = float(
            sum(summary["goal_diversity_end"].values())
        )
        return metrics

    def _aggregate(self, replicates: Sequence[ReplicateResult]) -> Dict[str, Any]:
        """Compute per-arm descriptive stats and paired unique-vs-uniform tests.

        For every scalar metric we report each arm's mean/std/sem across
        replicates, and a *paired* analysis of the per-seed delta
        (unique - uniform): mean, std, sem, a 95% CI, and a two-sided paired
        t-test (``scipy.stats.ttest_rel``).  Pairing removes shared
        seed-level variance, which is the whole reason the two arms share a
        seed.
        """
        uniform_metrics = [
            self._scalar_metrics(rep.uniform.summary()) for rep in replicates
        ]
        unique_metrics = [
            self._scalar_metrics(rep.unique.summary()) for rep in replicates
        ]
        metric_names = list(uniform_metrics[0].keys())
        n = len(replicates)

        def _describe(values: np.ndarray) -> Dict[str, float]:
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            sem = std / np.sqrt(values.size) if values.size > 1 else 0.0
            return {
                "mean": float(np.mean(values)),
                "std": std,
                "sem": float(sem),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        per_arm: Dict[str, Dict[str, Any]] = {"uniform": {}, "unique": {}}
        paired: Dict[str, Any] = {}
        for name in metric_names:
            u_vals = np.array([m[name] for m in uniform_metrics], dtype=float)
            q_vals = np.array([m[name] for m in unique_metrics], dtype=float)
            per_arm["uniform"][name] = _describe(u_vals)
            per_arm["unique"][name] = _describe(q_vals)

            deltas = q_vals - u_vals
            d_mean = float(np.mean(deltas))
            d_std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
            d_sem = d_std / np.sqrt(n) if n > 1 else 0.0
            if n > 1 and d_sem > 0.0:
                t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
                ci = [d_mean - t_crit * d_sem, d_mean + t_crit * d_sem]
                t_res = scipy_stats.ttest_rel(q_vals, u_vals)
                t_stat = float(t_res.statistic)
                p_value = float(t_res.pvalue)
                # Cohen's dz for paired samples.
                cohen_dz = d_mean / d_std if d_std > 0 else 0.0
            else:
                ci = [d_mean, d_mean]
                t_stat = float("nan")
                p_value = float("nan")
                cohen_dz = 0.0
            paired[name] = {
                "delta_mean": d_mean,
                "delta_std": d_std,
                "delta_sem": float(d_sem),
                "ci95": ci,
                "t_stat": t_stat,
                "p_value": p_value,
                "cohen_dz": cohen_dz,
                "significant_p05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                "n": n,
            }

        return {
            "num_replicates": n,
            "seeds": [rep.seed for rep in replicates],
            "per_arm": per_arm,
            "paired_deltas": paired,
        }

    def _maybe_plot(self, uniform: ArmResult, unique: ArmResult) -> Optional[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - plotting is optional
            logger.warning("intrinsic_goals_plot_skipped", reason=str(exc))
            return None

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        ax = axes[0][0]
        ax.plot(uniform.steps, uniform.population, label="uniform goals", color="#1f77b4")
        ax.plot(unique.steps, unique.population, label="unique goals", color="#d62728")
        ax.set_title("Population over time")
        ax.set_xlabel("step")
        ax.set_ylabel("alive agents")
        ax.legend()

        # Mean action share per arm (bar comparison).
        ax = axes[0][1]
        actions = list(TRACKED_ACTIONS)
        u_share = [
            statistics.mean(uniform.action_mix[a]) if uniform.action_mix[a] else 0.0
            for a in actions
        ]
        q_share = [
            statistics.mean(unique.action_mix[a]) if unique.action_mix[a] else 0.0
            for a in actions
        ]
        x = range(len(actions))
        ax.bar([i - 0.2 for i in x], u_share, width=0.4, label="uniform", color="#1f77b4")
        ax.bar([i + 0.2 for i in x], q_share, width=0.4, label="unique", color="#d62728")
        ax.set_xticks(list(x))
        ax.set_xticklabels(actions, rotation=45, ha="right")
        ax.set_title("Mean action mix")
        ax.set_ylabel("fraction of agents")
        ax.legend()

        # Goal-gene drift (unique arm): how the population mean of each goal
        # gene moves under selection.
        ax = axes[1][0]
        for gene in self.config.goal_genes:
            series = unique.gene_means.get(gene, [])
            if series:
                ax.plot(unique.steps, series, label=gene.replace("reward_", ""))
        ax.set_title("Goal-gene population mean (unique arm)")
        ax.set_xlabel("step")
        ax.set_ylabel("mean weight")
        ax.legend(fontsize=7, ncol=2)

        # Goal diversity start vs end (unique arm): std-dev per gene.
        ax = axes[1][1]
        genes = list(self.config.goal_genes)
        start = [unique.goal_diversity_start.get(g, 0.0) for g in genes]
        end = [unique.goal_diversity_end.get(g, 0.0) for g in genes]
        y = range(len(genes))
        ax.barh([i + 0.2 for i in y], start, height=0.4, label="start", color="#2ca02c")
        ax.barh([i - 0.2 for i in y], end, height=0.4, label="end", color="#9467bd")
        ax.set_yticks(list(y))
        ax.set_yticklabels([g.replace("reward_", "") for g in genes], fontsize=7)
        ax.set_title("Goal diversity (std) start vs end (unique arm)")
        ax.set_xlabel("population std of gene")
        ax.legend()

        fig.suptitle("Intrinsic goals: uniform vs unique reward functions", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        figure_path = os.path.join(
            self.config.output_dir, "intrinsic_goals_comparison.png"
        )
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)
        return figure_path

    def _maybe_plot_aggregate(
        self, replicates: Sequence[ReplicateResult], aggregate: Dict[str, Any]
    ) -> Optional[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - plotting is optional
            logger.warning("intrinsic_goals_plot_skipped", reason=str(exc))
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1) Population over time: mean +/- std band across replicates.
        ax = axes[0][0]
        max_len = max(
            len(pop)
            for r in replicates
            for pop in (r.uniform.population, r.unique.population)
        )
        steps = None
        for r in replicates:
            for arm in (r.uniform, r.unique):
                if len(arm.population) == max_len:
                    steps = arm.steps
                    break
            if steps is not None:
                break
        u_stack = np.zeros((len(replicates), max_len), dtype=float)
        q_stack = np.zeros((len(replicates), max_len), dtype=float)
        for i, r in enumerate(replicates):
            u_stack[i, : len(r.uniform.population)] = r.uniform.population
            q_stack[i, : len(r.unique.population)] = r.unique.population
        for stack, label, color in (
            (u_stack, "uniform", "#1f77b4"),
            (q_stack, "unique", "#d62728"),
        ):
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(steps, mean, label=f"{label} (mean)", color=color)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(f"Population over time (n={len(replicates)} seeds, mean ± std)")
        ax.set_xlabel("step")
        ax.set_ylabel("alive agents")
        ax.legend()

        # 2) Mean action share per arm with std error bars across replicates.
        ax = axes[0][1]
        actions = list(TRACKED_ACTIONS)
        per_arm = aggregate["per_arm"]
        u_mean = [per_arm["uniform"][f"action_share[{a}]"]["mean"] for a in actions]
        u_std = [per_arm["uniform"][f"action_share[{a}]"]["std"] for a in actions]
        q_mean = [per_arm["unique"][f"action_share[{a}]"]["mean"] for a in actions]
        q_std = [per_arm["unique"][f"action_share[{a}]"]["std"] for a in actions]
        x = np.arange(len(actions))
        ax.bar(x - 0.2, u_mean, yerr=u_std, width=0.4, label="uniform",
               color="#1f77b4", capsize=3)
        ax.bar(x + 0.2, q_mean, yerr=q_std, width=0.4, label="unique",
               color="#d62728", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(actions, rotation=45, ha="right")
        ax.set_title("Mean action mix (± std across seeds)")
        ax.set_ylabel("fraction of agents")
        ax.legend()

        # 3) Paired deltas (unique - uniform) with 95% CI for key metrics.
        ax = axes[1][0]
        key_metrics = [
            "mean_population",
            "final_population",
            "peak_population",
            "total_births",
            "total_deaths",
        ]
        paired = aggregate["paired_deltas"]
        means = [paired[m]["delta_mean"] for m in key_metrics]
        errs = [
            (paired[m]["ci95"][1] - paired[m]["ci95"][0]) / 2.0 for m in key_metrics
        ]
        y = np.arange(len(key_metrics))
        colors = [
            "#2ca02c" if paired[m]["significant_p05"] else "#7f7f7f"
            for m in key_metrics
        ]
        ax.barh(y, means, xerr=errs, color=colors, capsize=4)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(key_metrics)
        ax.set_title("Paired delta (unique − uniform), 95% CI\ngreen = p<0.05")
        ax.set_xlabel("Δ (unique − uniform)")

        # 4) Per-action-share paired deltas with 95% CI.
        ax = axes[1][1]
        means = [paired[f"action_share[{a}]"]["delta_mean"] for a in actions]
        errs = [
            (
                paired[f"action_share[{a}]"]["ci95"][1]
                - paired[f"action_share[{a}]"]["ci95"][0]
            )
            / 2.0
            for a in actions
        ]
        colors = [
            "#2ca02c" if paired[f"action_share[{a}]"]["significant_p05"] else "#7f7f7f"
            for a in actions
        ]
        y = np.arange(len(actions))
        ax.barh(y, means, xerr=errs, color=colors, capsize=4)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(actions)
        ax.set_title("Action-share Δ (unique − uniform), 95% CI\ngreen = p<0.05")
        ax.set_xlabel("Δ fraction of agents")

        fig.suptitle(
            f"Intrinsic goals: uniform vs unique ({len(replicates)} paired seeds)",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figure_path = os.path.join(
            self.config.output_dir, "intrinsic_goals_aggregate.png"
        )
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)
        return figure_path
