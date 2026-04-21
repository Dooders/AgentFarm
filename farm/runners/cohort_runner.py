"""Multi-seed cohort runner for statistically-robust evolution experiments.

Runs the same :class:`~farm.runners.EvolutionExperiment` configuration
over *N* different random seeds and aggregates the per-seed results into a
single :class:`CohortAggregateResult`.  This lets callers compare
strategies with appropriate statistical confidence rather than relying on
a single run.

Typical usage::

    from farm.config import SimulationConfig
    from farm.runners import EvolutionExperimentConfig, EvolutionFitnessMetric
    from farm.runners.cohort_runner import CohortRunner

    base_config = SimulationConfig.from_centralized_config(environment="development")
    template = EvolutionExperimentConfig(
        num_generations=8,
        population_size=10,
        num_steps_per_candidate=80,
        seed=None,  # overridden per seed by CohortRunner
    )
    runner = CohortRunner(
        base_config=base_config,
        experiment_config_template=template,
        seeds=[1, 2, 3],
        output_dir="experiments/cohort_run",
    )
    aggregate = runner.run()
    print(aggregate.best_fitness_mean, aggregate.best_fitness_std)
"""

from __future__ import annotations

import csv
import enum
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    chromosome_from_learning_config,
    default_hyperparameter_chromosome,
)
from farm.runners.evolution_experiment import (
    EvolutionExperiment,
    EvolutionExperimentConfig,
    EvolutionExperimentResult,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CohortSeedResult:
    """Per-seed result captured within a cohort run.

    Attributes
    ----------
    seed:
        The random seed used for this run.
    best_fitness:
        Best fitness value observed across all generations.
    num_generations_completed:
        Number of generations that ran (may be less than the budget when
        ``early_stop`` is active).
    converged:
        ``True`` when a convergence criterion was satisfied.
    convergence_reason:
        String value from :class:`~farm.runners.ConvergenceReason`, or
        ``None`` when convergence checking was disabled.
    generation_of_convergence:
        0-based generation index at which convergence was declared (or the
        last generation when the budget was exhausted with checking enabled).
        ``None`` when convergence checking was disabled.
    elapsed_seconds:
        Wall-clock seconds for this seed's run.
    lower_bound_occupancy:
        Fraction of generations in which the best chromosome's
        ``learning_rate`` gene was at (or within 1 % of) its minimum
        boundary.  ``None`` when no generation summaries were recorded.
    """

    seed: int
    best_fitness: float
    num_generations_completed: int
    converged: bool
    convergence_reason: Optional[str]
    generation_of_convergence: Optional[int]
    elapsed_seconds: float
    lower_bound_occupancy: Optional[float]


@dataclass
class CohortAggregateResult:
    """Aggregated statistics from a multi-seed cohort run.

    Schema
    ------
    The following keys appear in both the JSON and CSV artifacts produced by
    :meth:`CohortRunner._persist`:

    ``num_seeds``
        Total number of seeds executed.
    ``seeds``
        Ordered list of seed values used.
    ``best_fitness_mean``, ``best_fitness_std``, ``best_fitness_min``, ``best_fitness_max``
        Cross-seed statistics for the per-run best fitness.
    ``convergence_rate``
        Fraction (0–1) of seed runs that satisfied a convergence criterion.
    ``convergence_reason_counts``
        Mapping of convergence reason → count across seeds.
    ``mean_generation_of_convergence``
        Mean 0-based generation index at which convergence was first
        declared (only across runs where ``converged`` is ``True``).
        ``None`` when no run converged.
    ``std_generation_of_convergence``
        Standard deviation of the same (``None`` for 0 or 1 converged
        runs).
    ``lower_bound_occupancy_mean``, ``lower_bound_occupancy_std``
        Cross-seed mean/std of per-seed lower-bound occupancy.  ``None``
        when no seed produced generation summaries.
    ``mean_elapsed_seconds``
        Average wall-clock time per seed.
    ``total_elapsed_seconds``
        Total wall-clock time for the whole cohort.
    ``seed_results``
        Per-seed detail as a list of :class:`CohortSeedResult` mappings.
    ``config``
        Serialised :class:`~farm.runners.EvolutionExperimentConfig`
        template (seed field reflects the template value, *not* individual
        overrides).
    """

    config: Dict[str, Any]
    num_seeds: int
    seeds: List[int]
    seed_results: List[CohortSeedResult]

    # Cross-seed fitness statistics
    best_fitness_mean: float
    best_fitness_std: float
    best_fitness_min: float
    best_fitness_max: float

    # Convergence statistics
    convergence_rate: float
    convergence_reason_counts: Dict[str, int]
    mean_generation_of_convergence: Optional[float]
    std_generation_of_convergence: Optional[float]

    # Lower-bound occupancy
    lower_bound_occupancy_mean: Optional[float]
    lower_bound_occupancy_std: Optional[float]

    # Timing
    mean_elapsed_seconds: float
    total_elapsed_seconds: float


def _lower_bound_occupancy(
    result: EvolutionExperimentResult,
    *,
    learning_rate_lower_bound: Optional[float],
) -> Optional[float]:
    """Return fraction of generations where the best chromosome hit the lower boundary.

    A generation is considered to have *lower-bound occupancy* when the
    best candidate's ``learning_rate`` gene value is within 1 % of the
    configured lower boundary.  The 1 % tolerance handles floating-point
    noise near the exact boundary value.

    Returns ``None`` when no generation summaries are available or when the
    learning-rate lower boundary is unavailable.
    """
    if learning_rate_lower_bound is None:
        return None

    summaries = result.generation_summaries
    if not summaries:
        return None

    count = 0
    for summary in summaries:
        lr_stats = summary.gene_statistics.get("learning_rate")
        if lr_stats is None:
            continue
        best_lr = summary.best_chromosome.get("learning_rate")
        if best_lr is None:
            continue
        # Use a small relative tolerance around the configured lower bound.
        threshold = max(abs(learning_rate_lower_bound) * 0.01, 1e-9)
        if abs(best_lr - learning_rate_lower_bound) <= threshold:
            count += 1

    return count / len(summaries)


def _resolve_learning_rate_lower_bound(base_config: SimulationConfig) -> Optional[float]:
    """Resolve configured learning-rate lower bound from simulation config."""
    chromosome = default_hyperparameter_chromosome()
    learning_config = getattr(base_config, "learning", None)
    if learning_config is not None:
        try:
            chromosome = chromosome_from_learning_config(learning_config)
        except (TypeError, ValueError):
            # Fallback keeps test doubles and partial configs robust.
            chromosome = default_hyperparameter_chromosome()
    gene = chromosome.get_gene("learning_rate")
    if gene is None:
        return None
    return gene.min_value


def _safe_stdev(values: List[float]) -> Optional[float]:
    """Return population standard deviation, or ``None`` for fewer than 2 values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


class CohortRunner:
    """Run an :class:`~farm.runners.EvolutionExperiment` over multiple seeds.

    Parameters
    ----------
    base_config:
        Simulation configuration shared by all seed runs.
    experiment_config_template:
        Experiment configuration template.  The ``seed`` field is
        overridden for each run; all other fields are shared.
    seeds:
        Explicit list of integer seeds to use.  Must be non-empty.
    output_dir:
        Root directory for artifacts.  A ``seed_<N>/`` sub-directory is
        created for each run, and the aggregate artifacts are written to
        the root.  When ``None`` no artifacts are persisted.
    """

    def __init__(
        self,
        base_config: SimulationConfig,
        experiment_config_template: EvolutionExperimentConfig,
        seeds: List[int],
        output_dir: Optional[str] = None,
    ) -> None:
        if not seeds:
            raise ValueError("seeds must be a non-empty list.")
        self.base_config = base_config
        self.experiment_config_template = experiment_config_template
        self.seeds = list(seeds)
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> CohortAggregateResult:
        """Execute one evolution run per seed and return aggregated results.

        Each seed run writes its per-generation and lineage artifacts to
        ``<output_dir>/seed_<N>/`` (when *output_dir* is set).  The
        cohort-level JSON and CSV summaries are written to *output_dir*
        after all seeds complete.
        """
        logger.info(
            "cohort_run_start",
            num_seeds=len(self.seeds),
            seeds=self.seeds,
            output_dir=self.output_dir,
        )

        cohort_start = time.time()
        seed_results: List[CohortSeedResult] = []
        learning_rate_lower_bound = _resolve_learning_rate_lower_bound(self.base_config)

        for seed in self.seeds:
            seed_output_dir = (
                os.path.join(self.output_dir, f"seed_{seed}")
                if self.output_dir
                else None
            )
            seed_config = _replace_seed(self.experiment_config_template, seed, seed_output_dir)

            logger.info("cohort_seed_start", seed=seed, output_dir=seed_output_dir)
            seed_start = time.time()
            result = EvolutionExperiment(self.base_config, seed_config).run()
            elapsed = time.time() - seed_start

            occupancy = _lower_bound_occupancy(
                result,
                learning_rate_lower_bound=learning_rate_lower_bound,
            )
            best_fitness = result.best_candidate.fitness

            seed_result = CohortSeedResult(
                seed=seed,
                best_fitness=best_fitness,
                num_generations_completed=len(result.generation_summaries),
                converged=result.converged,
                convergence_reason=result.convergence_reason,
                generation_of_convergence=result.generation_of_convergence,
                elapsed_seconds=round(elapsed, 3),
                lower_bound_occupancy=occupancy,
            )
            seed_results.append(seed_result)
            logger.info(
                "cohort_seed_complete",
                seed=seed,
                best_fitness=best_fitness,
                converged=result.converged,
                convergence_reason=result.convergence_reason,
                elapsed_seconds=seed_result.elapsed_seconds,
            )

        total_elapsed = time.time() - cohort_start
        aggregate = self._aggregate(seed_results, total_elapsed)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self._persist(aggregate)

        logger.info(
            "cohort_run_complete",
            num_seeds=len(self.seeds),
            best_fitness_mean=aggregate.best_fitness_mean,
            best_fitness_std=aggregate.best_fitness_std,
            convergence_rate=aggregate.convergence_rate,
            total_elapsed_seconds=round(total_elapsed, 3),
        )
        return aggregate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        seed_results: List[CohortSeedResult],
        total_elapsed: float,
    ) -> CohortAggregateResult:
        fitnesses = [sr.best_fitness for sr in seed_results]
        n = len(fitnesses)
        mean_fitness = sum(fitnesses) / n
        std_fitness = _safe_stdev(fitnesses) or 0.0

        convergence_reason_counts: Dict[str, int] = {}
        gen_of_convergence_values: List[float] = []
        for sr in seed_results:
            if sr.convergence_reason:
                convergence_reason_counts[sr.convergence_reason] = (
                    convergence_reason_counts.get(sr.convergence_reason, 0) + 1
                )
            if sr.converged and sr.generation_of_convergence is not None:
                gen_of_convergence_values.append(float(sr.generation_of_convergence))

        convergence_rate = sum(1 for sr in seed_results if sr.converged) / n

        mean_gen_convergence: Optional[float] = None
        std_gen_convergence: Optional[float] = None
        if gen_of_convergence_values:
            mean_gen_convergence = sum(gen_of_convergence_values) / len(gen_of_convergence_values)
            std_gen_convergence = _safe_stdev(gen_of_convergence_values)

        lb_occ_values = [sr.lower_bound_occupancy for sr in seed_results if sr.lower_bound_occupancy is not None]
        lb_occ_mean: Optional[float] = (
            sum(lb_occ_values) / len(lb_occ_values) if lb_occ_values else None
        )
        lb_occ_std: Optional[float] = _safe_stdev(lb_occ_values) if lb_occ_values else None

        elapsed_values = [sr.elapsed_seconds for sr in seed_results]
        mean_elapsed = sum(elapsed_values) / len(elapsed_values)

        return CohortAggregateResult(
            config=_serialize_experiment_config(self.experiment_config_template),
            num_seeds=n,
            seeds=list(self.seeds),
            seed_results=seed_results,
            best_fitness_mean=round(mean_fitness, 6),
            best_fitness_std=round(std_fitness, 6),
            best_fitness_min=round(min(fitnesses), 6),
            best_fitness_max=round(max(fitnesses), 6),
            convergence_rate=round(convergence_rate, 4),
            convergence_reason_counts=convergence_reason_counts,
            mean_generation_of_convergence=(
                round(mean_gen_convergence, 3) if mean_gen_convergence is not None else None
            ),
            std_generation_of_convergence=(
                round(std_gen_convergence, 3) if std_gen_convergence is not None else None
            ),
            lower_bound_occupancy_mean=(
                round(lb_occ_mean, 4) if lb_occ_mean is not None else None
            ),
            lower_bound_occupancy_std=(
                round(lb_occ_std, 4) if lb_occ_std is not None else None
            ),
            mean_elapsed_seconds=round(mean_elapsed, 3),
            total_elapsed_seconds=round(total_elapsed, 3),
        )

    def _persist(self, aggregate: CohortAggregateResult) -> None:
        """Write JSON and CSV aggregate artifacts to *output_dir*."""
        assert self.output_dir is not None  # guarded by caller

        json_path = os.path.join(self.output_dir, "cohort_aggregate.json")
        csv_path = os.path.join(self.output_dir, "cohort_aggregate.csv")

        # --- JSON artifact ---
        payload: Dict[str, Any] = {
            "config": aggregate.config,
            "num_seeds": aggregate.num_seeds,
            "seeds": aggregate.seeds,
            "best_fitness_mean": aggregate.best_fitness_mean,
            "best_fitness_std": aggregate.best_fitness_std,
            "best_fitness_min": aggregate.best_fitness_min,
            "best_fitness_max": aggregate.best_fitness_max,
            "convergence_rate": aggregate.convergence_rate,
            "convergence_reason_counts": aggregate.convergence_reason_counts,
            "mean_generation_of_convergence": aggregate.mean_generation_of_convergence,
            "std_generation_of_convergence": aggregate.std_generation_of_convergence,
            "lower_bound_occupancy_mean": aggregate.lower_bound_occupancy_mean,
            "lower_bound_occupancy_std": aggregate.lower_bound_occupancy_std,
            "mean_elapsed_seconds": aggregate.mean_elapsed_seconds,
            "total_elapsed_seconds": aggregate.total_elapsed_seconds,
            "seed_results": [asdict(sr) for sr in aggregate.seed_results],
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        # --- CSV artifact (one row per seed) ---
        fieldnames = [
            "seed",
            "best_fitness",
            "num_generations_completed",
            "converged",
            "convergence_reason",
            "generation_of_convergence",
            "elapsed_seconds",
            "lower_bound_occupancy",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for sr in aggregate.seed_results:
                writer.writerow(asdict(sr))

        logger.info(
            "cohort_aggregate_persisted",
            output_dir=self.output_dir,
            json_path=json_path,
            csv_path=csv_path,
            num_seeds=aggregate.num_seeds,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replace_seed(
    template: EvolutionExperimentConfig,
    seed: int,
    output_dir: Optional[str],
) -> EvolutionExperimentConfig:
    """Return a copy of *template* with ``seed`` and ``output_dir`` replaced."""
    # EvolutionExperimentConfig is a frozen dataclass; use keyword reconstruction.
    import dataclasses

    return dataclasses.replace(template, seed=seed, output_dir=output_dir)


def _serialize_experiment_config(config: EvolutionExperimentConfig) -> Dict[str, Any]:
    """Return a JSON-serialisable dict from an :class:`EvolutionExperimentConfig`.

    Uses a recursive field walk instead of ``dataclasses.asdict`` to avoid
    deep-copy failures on ``MappingProxyType`` members inside
    :class:`~farm.runners.AdaptiveMutationConfig`.
    """
    import dataclasses

    def _coerce(obj: Any) -> Any:
        # Enum → its .value string (only for real Enum instances, not arbitrary
        # objects that happen to expose a `.value` attribute).
        if isinstance(obj, enum.Enum):
            return obj.value
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _coerce(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if hasattr(obj, "items"):
            # dict-like (including MappingProxyType)
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        return obj

    return _coerce(config)
