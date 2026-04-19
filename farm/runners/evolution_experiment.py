"""Evolution experiment runner for multi-generation hyperparameter search."""

from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from farm.config import SimulationConfig
from farm.core.genome import Genome, RouletteSelectionConfig, TournamentSelectionConfig
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    BoundaryPenaltyConfig,
    CrossoverMode,
    HyperparameterChromosome,
    MutationMode,
    apply_chromosome_to_learning_config,
    chromosome_from_learning_config,
    compute_boundary_penalty,
    crossover_chromosomes,
    mutate_chromosome,
)
from farm.core.simulation import run_simulation
from farm.runners.adaptive_mutation import (
    AdaptiveMutationConfig,
    AdaptiveMutationController,
    compute_normalized_diversity,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class EvolutionFitnessMetric(str, Enum):
    """Supported built-in fitness metrics for evolution experiments."""

    FINAL_POPULATION = "final_population"
    TOTAL_BIRTHS = "total_births"
    FINAL_RESOURCES = "final_resources"


class EvolutionSelectionMethod(str, Enum):
    """Parent selection strategy."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"


class ConvergenceReason(str, Enum):
    """Reason a run was declared converged or its budget was exhausted.

    ``FITNESS_PLATEAU``
        The best fitness did not improve by more than the configured threshold
        over the trailing fitness window.

    ``DIVERSITY_COLLAPSE``
        The mean normalized gene diversity stayed below the configured
        threshold for the required number of consecutive generations.

    ``BUDGET_EXHAUSTED``
        The run completed all configured generations without any convergence
        criterion being satisfied.  Returned only when
        :attr:`ConvergenceCriteria.enabled` is ``True`` and neither fitness
        plateau nor diversity collapse was detected.
    """

    FITNESS_PLATEAU = "fitness_plateau"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class ConvergenceCriteria:
    """Configurable convergence stopping criteria for evolution experiments.

    When ``enabled`` is ``False`` (default), convergence checking is skipped
    entirely and the run always uses the full generation budget, preserving
    existing behavior.  When ``True``:

    - **Fitness plateau**: declares convergence when the best fitness fails
      to improve by more than ``fitness_threshold`` absolute units over the
      trailing ``fitness_window`` generations.
    - **Diversity collapse**: declares convergence when the mean normalized
      gene diversity remains below ``diversity_threshold`` for
      ``diversity_window`` consecutive generations.

    Checks are suppressed until at least ``min_generations`` full generations
    have been evaluated.

    When ``early_stop`` is ``True`` (default), the run halts as soon as a
    criterion is met.  When ``False``, the run continues for all configured
    generations but the result is still annotated with the first convergence
    event detected.
    """

    enabled: bool = False
    fitness_window: int = 5
    fitness_threshold: float = 1e-4
    diversity_window: int = 3
    diversity_threshold: float = 0.01
    min_generations: int = 1
    early_stop: bool = True

    def __post_init__(self) -> None:
        if self.fitness_window < 1:
            raise ValueError("fitness_window must be at least 1.")
        if self.fitness_threshold < 0.0:
            raise ValueError("fitness_threshold must be non-negative.")
        if self.diversity_window < 1:
            raise ValueError("diversity_window must be at least 1.")
        if self.diversity_threshold < 0.0:
            raise ValueError("diversity_threshold must be non-negative.")
        if self.min_generations < 0:
            raise ValueError("min_generations must be non-negative.")


@dataclass(frozen=True)
class EvolutionExperimentConfig:
    """Configuration for generation-based hyperparameter evolution."""

    num_generations: int = 3
    population_size: int = 6
    num_steps_per_candidate: int = 50
    mutation_rate: float = 0.25
    mutation_scale: float = 0.2
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.CLAMP
    boundary_penalty: BoundaryPenaltyConfig = field(default_factory=BoundaryPenaltyConfig)
    crossover_mode: CrossoverMode = CrossoverMode.UNIFORM
    blend_alpha: float = 0.5
    num_crossover_points: int = 2
    selection_method: EvolutionSelectionMethod = EvolutionSelectionMethod.TOURNAMENT
    tournament_size: int = 3
    elitism_count: int = 1
    fitness_metric: EvolutionFitnessMetric = EvolutionFitnessMetric.FINAL_POPULATION
    adaptive_mutation: AdaptiveMutationConfig = field(default_factory=AdaptiveMutationConfig)
    convergence_criteria: ConvergenceCriteria = field(default_factory=ConvergenceCriteria)
    seed: Optional[int] = None
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_generations < 1:
            raise ValueError("num_generations must be at least 1.")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.num_steps_per_candidate < 1:
            raise ValueError("num_steps_per_candidate must be at least 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative.")
        # Validate enum coercion for string-friendly construction.
        BoundaryMode(self.boundary_mode)
        CrossoverMode(self.crossover_mode)
        if self.blend_alpha < 0.0:
            raise ValueError("blend_alpha must be non-negative.")
        if self.num_crossover_points < 1:
            raise ValueError("num_crossover_points must be at least 1.")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1.")
        if self.elitism_count < 0:
            raise ValueError("elitism_count must be non-negative.")
        if self.elitism_count >= self.population_size:
            raise ValueError("elitism_count must be smaller than population_size.")


@dataclass(frozen=True)
class EvolutionCandidate:
    """Single chromosome candidate in the evolving population."""

    candidate_id: str
    generation: int
    chromosome: HyperparameterChromosome
    parent_ids: Tuple[str, str]


@dataclass(frozen=True)
class EvolutionCandidateEvaluation:
    """Evaluation result for a single candidate in one generation."""

    candidate_id: str
    generation: int
    fitness: float
    learning_rate: float
    chromosome_values: Dict[str, float]
    parent_ids: Tuple[str, str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EvolutionGenerationSummary:
    """Aggregate stats for one evolved generation.

    The ``mutation_rate_used`` / ``mutation_scale_used`` /
    ``mutation_*_multiplier`` / ``adaptive_event`` fields describe the
    mutation parameters that **produced this generation's children**.  For
    generation 0 these are ``None`` because the initial population is seeded
    via :meth:`EvolutionExperiment._initialize_population` (which uses
    ``mutation_rate=1.0`` to spread seed candidates) rather than via the
    adaptive controller.  ``diversity`` is measured **on this generation**
    and therefore is recorded for every generation.
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    min_fitness: float
    best_candidate_id: str
    gene_statistics: Dict[str, Dict[str, float]]
    best_chromosome: Dict[str, float]
    mutation_rate_used: Optional[float] = None
    mutation_scale_used: Optional[float] = None
    mutation_rate_multiplier: Optional[float] = None
    mutation_scale_multiplier: Optional[float] = None
    diversity: Optional[float] = None
    adaptive_event: str = "initial_seeding"


@dataclass
class EvolutionExperimentResult:
    """Container with generation summaries and per-candidate lineage.

    ``converged`` is ``True`` when a convergence criterion (fitness plateau or
    diversity collapse) was satisfied.  It remains ``False`` when convergence
    checking is disabled or when the run exhausted its generation budget
    without triggering a criterion.

    ``convergence_reason`` holds a :class:`ConvergenceReason` value (as a
    string) when convergence checking is enabled.  It is ``"budget_exhausted"``
    when all generations completed without a criterion being met, and ``None``
    when convergence checking is disabled.

    ``generation_of_convergence`` is the 0-based generation index at which the
    criterion was first satisfied.  When the budget is exhausted it is set to
    the index of the last completed generation.  ``None`` when disabled.
    """

    generation_summaries: List[EvolutionGenerationSummary]
    evaluations: List[EvolutionCandidateEvaluation]
    best_candidate: EvolutionCandidateEvaluation
    converged: bool = False
    convergence_reason: Optional[str] = None
    generation_of_convergence: Optional[int] = None


FitnessEvaluator = Callable[
    [EvolutionCandidate, SimulationConfig, int, int],
    Tuple[float, Dict[str, Any]],
]


@dataclass(frozen=True)
class _ProducedWith:
    """Mutation parameters that produced a generation's population.

    All four numeric fields are ``None`` for the initial population because
    seeding bypasses the adaptive controller.  ``event`` is a short tag
    matching :attr:`AdaptiveMutationController.last_event` from the
    observation that yielded these parameters.
    """

    rate: Optional[float]
    scale: Optional[float]
    rate_multiplier: Optional[float]
    scale_multiplier: Optional[float]
    event: str

    @classmethod
    def initial(cls) -> "_ProducedWith":
        return cls(rate=None, scale=None, rate_multiplier=None, scale_multiplier=None, event="initial_seeding")


class EvolutionExperiment:
    """Run multi-generation evolution over hyperparameter chromosomes."""

    def __init__(self, base_config: SimulationConfig, config: EvolutionExperimentConfig):
        self.base_config = base_config
        self.config = config
        self._initial_chromosome = chromosome_from_learning_config(base_config.learning)
        self._selection_call_count = 0
        self._run_rng: Optional[random.Random] = None

    def run(self, fitness_evaluator: Optional[FitnessEvaluator] = None) -> EvolutionExperimentResult:
        """Run the configured number of generations and return tracked lineage."""
        # Keep seeded runs reproducible even when the same experiment instance is reused.
        self._selection_call_count = 0
        self._run_rng = random.Random(self.config.seed) if self.config.seed is not None else random.Random()
        evaluator = fitness_evaluator or self._default_fitness_evaluator
        controller = AdaptiveMutationController(self.config.adaptive_mutation)
        population = self._initialize_population()
        generation_summaries: List[EvolutionGenerationSummary] = []
        evaluations: List[EvolutionCandidateEvaluation] = []

        # Tracks the mutation parameters that produced the *current* population.
        # `None` for generation 0 since the initial population is seeded by
        # `_initialize_population`, not by the adaptive controller.
        produced_with: _ProducedWith = _ProducedWith.initial()

        # Convergence tracking: histories are used by _check_convergence.
        best_fitness_history: List[float] = []
        diversity_history: List[Optional[float]] = []
        converged = False
        convergence_reason: Optional[str] = None
        generation_of_convergence: Optional[int] = None

        for generation in range(self.config.num_generations):
            generation_evals = self._evaluate_generation(generation, population, evaluator)
            evaluations.extend(generation_evals)
            gene_statistics = self._build_gene_statistics(generation_evals)
            diversity = self._compute_diversity(gene_statistics)
            best_fitness = max(evaluation.fitness for evaluation in generation_evals)

            summary = self._build_generation_summary(
                generation,
                generation_evals,
                gene_statistics=gene_statistics,
                diversity=diversity,
                produced_with=produced_with,
            )
            generation_summaries.append(summary)
            logger.info(
                "evolution_generation_completed",
                generation=generation,
                best_fitness=summary.best_fitness,
                mean_fitness=summary.mean_fitness,
                mutation_rate_used=produced_with.rate,
                mutation_scale_used=produced_with.scale,
                mutation_rate_multiplier=produced_with.rate_multiplier,
                mutation_scale_multiplier=produced_with.scale_multiplier,
                diversity=diversity,
                adaptive_event=produced_with.event,
            )

            # Update convergence histories and check criteria.
            best_fitness_history.append(best_fitness)
            diversity_history.append(diversity)
            if not converged:
                reason = self._check_convergence(generation, best_fitness_history, diversity_history)
                if reason is not None:
                    converged = True
                    convergence_reason = reason.value
                    generation_of_convergence = generation
                    logger.info(
                        "evolution_converged",
                        generation=generation,
                        reason=reason.value,
                        early_stop=self.config.convergence_criteria.early_stop,
                    )
                    if self.config.convergence_criteria.early_stop:
                        break

            controller.observe(best_fitness=best_fitness, diversity=diversity)
            next_rate = controller.effective_rate(self.config.mutation_rate)
            next_scale = controller.effective_scale(self.config.mutation_scale)
            population = self._next_generation(
                generation,
                generation_evals,
                effective_rate=next_rate,
                effective_scale=next_scale,
                per_gene_rate_multipliers=controller.per_gene_rate_multipliers(),
                per_gene_scale_multipliers=controller.per_gene_scale_multipliers(),
            )
            # Carry forward into next iteration: these are the params that
            # produced the population we just generated.
            produced_with = _ProducedWith(
                rate=next_rate,
                scale=next_scale,
                rate_multiplier=controller.rate_multiplier,
                scale_multiplier=controller.scale_multiplier,
                event=controller.last_event,
            )

        # When convergence checking is enabled but no criterion was met during
        # the run, annotate the result as budget-exhausted.
        if self.config.convergence_criteria.enabled and not converged and generation_summaries:
            convergence_reason = ConvergenceReason.BUDGET_EXHAUSTED.value
            generation_of_convergence = len(generation_summaries) - 1

        best_candidate = max(evaluations, key=lambda item: item.fitness)
        result = EvolutionExperimentResult(
            generation_summaries=generation_summaries,
            evaluations=evaluations,
            best_candidate=best_candidate,
            converged=converged,
            convergence_reason=convergence_reason,
            generation_of_convergence=generation_of_convergence,
        )
        self._persist_results(result)
        return result

    def _initialize_population(self) -> List[EvolutionCandidate]:
        population: List[EvolutionCandidate] = []
        for idx in range(self.config.population_size):
            chromosome = self._initial_chromosome
            if idx > 0:
                chromosome = mutate_chromosome(
                    chromosome,
                    mutation_rate=1.0,
                    mutation_scale=self.config.mutation_scale,
                    mutation_mode=self.config.mutation_mode,
                    boundary_mode=self.config.boundary_mode,
                    rng=self._run_rng,
                )
            population.append(
                EvolutionCandidate(
                    candidate_id=f"g0_c{idx}",
                    generation=0,
                    chromosome=chromosome,
                    parent_ids=("seed", "seed"),
                )
            )
        return population

    def _evaluate_generation(
        self,
        generation: int,
        population: List[EvolutionCandidate],
        evaluator: FitnessEvaluator,
    ) -> List[EvolutionCandidateEvaluation]:
        generation_evals: List[EvolutionCandidateEvaluation] = []
        for idx, candidate in enumerate(population):
            candidate_config = self._config_for_chromosome(candidate.chromosome)
            raw_fitness, metadata = evaluator(candidate, candidate_config, generation, idx)
            boundary_penalty = compute_boundary_penalty(
                candidate.chromosome,
                self.config.boundary_penalty,
            )
            adjusted_fitness = float(raw_fitness) - boundary_penalty
            metadata_with_lineage = dict(metadata)
            metadata_with_lineage["raw_fitness"] = float(raw_fitness)
            metadata_with_lineage["boundary_penalty"] = boundary_penalty
            metadata_with_lineage["chromosome"] = candidate.chromosome
            generation_evals.append(
                EvolutionCandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    generation=generation,
                    fitness=adjusted_fitness,
                    learning_rate=candidate.chromosome.get_value("learning_rate"),
                    chromosome_values=self._serialize_chromosome_values(candidate.chromosome),
                    parent_ids=candidate.parent_ids,
                    metadata=metadata_with_lineage,
                )
            )
        return generation_evals

    def _next_generation(
        self,
        generation: int,
        generation_evals: List[EvolutionCandidateEvaluation],
        *,
        effective_rate: Optional[float] = None,
        effective_scale: Optional[float] = None,
        per_gene_rate_multipliers: Optional[Mapping[str, float]] = None,
        per_gene_scale_multipliers: Optional[Mapping[str, float]] = None,
    ) -> List[EvolutionCandidate]:
        ranked = sorted(generation_evals, key=lambda item: item.fitness, reverse=True)
        next_population: List[EvolutionCandidate] = []

        for elite_idx in range(self.config.elitism_count):
            elite_eval = ranked[elite_idx]
            next_population.append(
                EvolutionCandidate(
                    candidate_id=f"g{generation + 1}_elite{elite_idx}",
                    generation=generation + 1,
                    chromosome=elite_eval.metadata["chromosome"],
                    parent_ids=(elite_eval.candidate_id, elite_eval.candidate_id),
                )
            )

        resolved_rate = effective_rate if effective_rate is not None else self.config.mutation_rate
        resolved_scale = effective_scale if effective_scale is not None else self.config.mutation_scale

        while len(next_population) < self.config.population_size:
            parent_a, parent_b = self._select_parents(generation_evals)
            child_chromosome = crossover_chromosomes(
                parent_a.metadata["chromosome"],
                parent_b.metadata["chromosome"],
                mode=self.config.crossover_mode,
                include_fixed=False,
                blend_alpha=self.config.blend_alpha,
                num_crossover_points=self.config.num_crossover_points,
                rng=self._run_rng,
            )
            child_chromosome = mutate_chromosome(
                child_chromosome,
                mutation_rate=resolved_rate,
                mutation_scale=resolved_scale,
                mutation_mode=self.config.mutation_mode,
                boundary_mode=self.config.boundary_mode,
                per_gene_rate_multipliers=per_gene_rate_multipliers,
                per_gene_scale_multipliers=per_gene_scale_multipliers,
                rng=self._run_rng,
            )
            child_idx = len(next_population)
            next_population.append(
                EvolutionCandidate(
                    candidate_id=f"g{generation + 1}_c{child_idx}",
                    generation=generation + 1,
                    chromosome=child_chromosome,
                    parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
                )
            )
        return next_population

    def _select_parents(
        self, generation_evals: List[EvolutionCandidateEvaluation]
    ) -> Tuple[EvolutionCandidateEvaluation, EvolutionCandidateEvaluation]:
        population = generation_evals
        fitnesses = [evaluation.fitness for evaluation in generation_evals]
        if self.config.selection_method is EvolutionSelectionMethod.ROULETTE:
            seed = (
                self.config.seed + self._selection_call_count
                if self.config.seed is not None
                else None
            )
            parent_indices = Genome.roulette_selection(
                population,
                fitnesses,
                RouletteSelectionConfig(
                    num_selected=2,
                    seed=seed,
                    return_indices=True,
                    shift_negative_fitness=True,
                ),
            )
        else:
            seed = (
                self.config.seed + self._selection_call_count
                if self.config.seed is not None
                else None
            )
            parent_indices = Genome.tournament_selection(
                population,
                fitnesses,
                TournamentSelectionConfig(
                    num_selected=2,
                    tournament_size=self.config.tournament_size,
                    seed=seed,
                    return_indices=True,
                ),
            )
        self._selection_call_count += 1
        return population[parent_indices[0]], population[parent_indices[1]]

    def _build_generation_summary(
        self,
        generation: int,
        generation_evals: List[EvolutionCandidateEvaluation],
        *,
        gene_statistics: Optional[Dict[str, Dict[str, float]]] = None,
        diversity: Optional[float] = None,
        produced_with: Optional[_ProducedWith] = None,
    ) -> EvolutionGenerationSummary:
        best = max(generation_evals, key=lambda item: item.fitness)
        fitness_values = [evaluation.fitness for evaluation in generation_evals]
        resolved_gene_statistics = (
            gene_statistics if gene_statistics is not None else self._build_gene_statistics(generation_evals)
        )
        best_chromosome = self._serialize_chromosome_values(best.metadata["chromosome"])
        produced = produced_with or _ProducedWith.initial()
        return EvolutionGenerationSummary(
            generation=generation,
            best_fitness=max(fitness_values),
            mean_fitness=statistics.mean(fitness_values),
            min_fitness=min(fitness_values),
            best_candidate_id=best.candidate_id,
            gene_statistics=resolved_gene_statistics,
            best_chromosome=best_chromosome,
            mutation_rate_used=produced.rate,
            mutation_scale_used=produced.scale,
            mutation_rate_multiplier=produced.rate_multiplier,
            mutation_scale_multiplier=produced.scale_multiplier,
            diversity=diversity,
            adaptive_event=produced.event,
        )

    def _compute_diversity(
        self,
        gene_statistics: Dict[str, Dict[str, float]],
    ) -> Optional[float]:
        """Compute mean normalized gene-std diversity over evolvable genes.

        Returns ``None`` when no evolvable gene has a non-zero span, so
        callers can decide whether to record the measure in telemetry.
        """
        evolvable_names: List[str] = []
        gene_bounds: Dict[str, Tuple[float, float]] = {}
        for gene in self._initial_chromosome.genes:
            if gene.evolvable and gene.max_value > gene.min_value:
                evolvable_names.append(gene.name)
                gene_bounds[gene.name] = (gene.min_value, gene.max_value)
        if not evolvable_names:
            return None
        return compute_normalized_diversity(gene_statistics, evolvable_names, gene_bounds)

    def _check_convergence(
        self,
        generation: int,
        best_fitness_history: List[float],
        diversity_history: List[Optional[float]],
    ) -> Optional[ConvergenceReason]:
        """Return a :class:`ConvergenceReason` if convergence is detected.

        Returns ``None`` when convergence checking is disabled, when fewer
        than ``min_generations`` have completed, or when no criterion is
        satisfied.  The caller is responsible for appending to both history
        lists *before* calling this method so that the current generation is
        included in the check.
        """
        criteria = self.config.convergence_criteria
        if not criteria.enabled:
            return None
        if generation < criteria.min_generations:
            return None

        # Fitness plateau: current best has not improved over the trailing window.
        if len(best_fitness_history) >= criteria.fitness_window + 1:
            window = best_fitness_history[-(criteria.fitness_window + 1):]
            improvement = window[-1] - max(window[:-1])
            if improvement <= criteria.fitness_threshold:
                return ConvergenceReason.FITNESS_PLATEAU

        # Diversity collapse: enough consecutive non-None diversity readings
        # are all below the threshold.
        recent_non_none = [d for d in diversity_history[-criteria.diversity_window:] if d is not None]
        if len(recent_non_none) >= criteria.diversity_window and all(
            d <= criteria.diversity_threshold for d in recent_non_none
        ):
            return ConvergenceReason.DIVERSITY_COLLAPSE

        return None

    def _build_gene_statistics(
        self,
        generation_evals: List[EvolutionCandidateEvaluation],
    ) -> Dict[str, Dict[str, float]]:
        if not generation_evals:
            return {}

        gene_values: Dict[str, List[float]] = {}
        for evaluation in generation_evals:
            chromosome = evaluation.metadata["chromosome"]
            for gene in chromosome.genes:
                gene_values.setdefault(gene.name, []).append(gene.value)

        gene_statistics: Dict[str, Dict[str, float]] = {}
        for gene_name, values in gene_values.items():
            gene_statistics[gene_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        return gene_statistics

    def _serialize_chromosome_values(self, chromosome: HyperparameterChromosome) -> Dict[str, float]:
        """Return a stable gene-name/value mapping for artifact serialization."""
        return {gene.name: gene.value for gene in chromosome.genes}

    def _config_for_chromosome(self, chromosome: HyperparameterChromosome) -> SimulationConfig:
        config_copy = self.base_config.copy()
        config_copy.learning = apply_chromosome_to_learning_config(
            config_copy.learning,
            chromosome,
        )
        return config_copy

    def _default_fitness_evaluator(
        self,
        candidate: EvolutionCandidate,
        candidate_config: SimulationConfig,
        generation: int,
        member_index: int,
    ) -> Tuple[float, Dict[str, Any]]:
        run_dir = self.config.output_dir
        if run_dir:
            run_dir = os.path.join(
                run_dir,
                f"generation_{generation}",
                candidate.candidate_id,
            )
        env = run_simulation(
            num_steps=self.config.num_steps_per_candidate,
            config=candidate_config,
            path=run_dir,
            save_config=False,
            seed=(
                (self.config.seed + generation * self.config.population_size + member_index)
                if self.config.seed is not None
                else None
            ),
        )
        fitness, metadata = self._fitness_from_environment(env)
        metadata["chromosome"] = candidate.chromosome
        return fitness, metadata

    def _fitness_from_environment(self, environment: Any) -> Tuple[float, Dict[str, Any]]:
        if self.config.fitness_metric is EvolutionFitnessMetric.TOTAL_BIRTHS:
            births = (
                getattr(environment.metrics_tracker.cumulative_metrics, "total_births", 0)
                if getattr(environment, "metrics_tracker", None)
                and getattr(environment.metrics_tracker, "cumulative_metrics", None)
                else 0
            )
            return float(births), {"total_births": births}
        if self.config.fitness_metric is EvolutionFitnessMetric.FINAL_RESOURCES:
            return float(environment.cached_total_resources), {
                "final_resources": environment.cached_total_resources,
            }
        return float(len(environment.agents)), {"final_population": len(environment.agents)}

    def _persist_results(self, result: EvolutionExperimentResult) -> None:
        if not self.config.output_dir:
            return
        os.makedirs(self.config.output_dir, exist_ok=True)
        summaries_path = os.path.join(self.config.output_dir, "evolution_generation_summaries.json")
        lineage_path = os.path.join(self.config.output_dir, "evolution_lineage.json")
        metadata_path = os.path.join(self.config.output_dir, "evolution_metadata.json")

        with open(summaries_path, "w", encoding="utf-8") as summaries_file:
            json.dump([asdict(summary) for summary in result.generation_summaries], summaries_file, indent=2)

        serialized_evaluations: List[Dict[str, Any]] = []
        for evaluation in result.evaluations:
            payload: Dict[str, Any] = {
                "candidate_id": evaluation.candidate_id,
                "generation": evaluation.generation,
                "fitness": evaluation.fitness,
                "learning_rate": evaluation.learning_rate,
                "chromosome": evaluation.chromosome_values,
                "parent_ids": list(evaluation.parent_ids),
                "metadata": {
                    key: value
                    for key, value in evaluation.metadata.items()
                    if key != "chromosome"
                },
            }
            serialized_evaluations.append(payload)

        with open(lineage_path, "w", encoding="utf-8") as lineage_file:
            json.dump(serialized_evaluations, lineage_file, indent=2)

        convergence_metadata: Dict[str, Any] = {
            "converged": result.converged,
            "convergence_reason": result.convergence_reason,
            "generation_of_convergence": result.generation_of_convergence,
            "num_generations_completed": len(result.generation_summaries),
        }
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(convergence_metadata, metadata_file, indent=2)

        logger.info(
            "evolution_experiment_persisted",
            output_dir=self.config.output_dir,
            summaries_path=summaries_path,
            lineage_path=lineage_path,
            metadata_path=metadata_path,
            num_generations=self.config.num_generations,
            converged=result.converged,
            convergence_reason=result.convergence_reason,
        )
