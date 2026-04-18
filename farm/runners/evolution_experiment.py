"""Evolution experiment runner for multi-generation hyperparameter search."""

from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from farm.config import SimulationConfig
from farm.core.genome import Genome, RouletteSelectionConfig, TournamentSelectionConfig
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    BoundaryPenaltyConfig,
    compute_boundary_penalty,
    CrossoverMode,
    HyperparameterChromosome,
    MutationMode,
    apply_chromosome_to_learning_config,
    chromosome_from_learning_config,
    crossover_chromosomes,
    mutate_chromosome,
)
from farm.core.simulation import run_simulation
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


@dataclass(frozen=True)
class EvolutionExperimentConfig:
    """Configuration for generation-based hyperparameter evolution."""

    num_generations: int = 3
    population_size: int = 6
    num_steps_per_candidate: int = 50
    mutation_rate: float = 0.25
    mutation_scale: float = 0.2
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    crossover_mode: CrossoverMode = CrossoverMode.UNIFORM
    selection_method: EvolutionSelectionMethod = EvolutionSelectionMethod.TOURNAMENT
    tournament_size: int = 3
    elitism_count: int = 1
    fitness_metric: EvolutionFitnessMetric = EvolutionFitnessMetric.FINAL_POPULATION
    boundary_mode: BoundaryMode = BoundaryMode.CLAMP
    boundary_penalty_config: Optional[BoundaryPenaltyConfig] = None
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
    parent_ids: Tuple[str, str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EvolutionGenerationSummary:
    """Aggregate stats for one evolved generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    min_fitness: float
    best_candidate_id: str
    gene_statistics: Dict[str, Dict[str, float]]
    best_chromosome: Dict[str, float]


@dataclass
class EvolutionExperimentResult:
    """Container with generation summaries and per-candidate lineage."""

    generation_summaries: List[EvolutionGenerationSummary]
    evaluations: List[EvolutionCandidateEvaluation]
    best_candidate: EvolutionCandidateEvaluation


FitnessEvaluator = Callable[
    [EvolutionCandidate, SimulationConfig, int, int],
    Tuple[float, Dict[str, Any]],
]


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
        population = self._initialize_population()
        generation_summaries: List[EvolutionGenerationSummary] = []
        evaluations: List[EvolutionCandidateEvaluation] = []

        for generation in range(self.config.num_generations):
            generation_evals = self._evaluate_generation(generation, population, evaluator)
            evaluations.extend(generation_evals)
            generation_summaries.append(self._build_generation_summary(generation, generation_evals))
            population = self._next_generation(generation, generation_evals)

        best_candidate = max(evaluations, key=lambda item: item.fitness)
        result = EvolutionExperimentResult(
            generation_summaries=generation_summaries,
            evaluations=evaluations,
            best_candidate=best_candidate,
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
            fitness, metadata = evaluator(candidate, candidate_config, generation, idx)
            penalty_cfg = self.config.boundary_penalty_config
            if penalty_cfg is not None and penalty_cfg.enabled:
                penalty = compute_boundary_penalty(candidate.chromosome, penalty_cfg)
            else:
                penalty = 0.0
            adjusted_fitness = float(fitness) - penalty
            metadata_with_lineage = dict(metadata)
            metadata_with_lineage["chromosome"] = candidate.chromosome
            if penalty > 0.0:
                metadata_with_lineage["boundary_penalty"] = penalty
            generation_evals.append(
                EvolutionCandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    generation=generation,
                    fitness=adjusted_fitness,
                    learning_rate=candidate.chromosome.get_value("learning_rate"),
                    parent_ids=candidate.parent_ids,
                    metadata=metadata_with_lineage,
                )
            )
        return generation_evals

    def _next_generation(
        self,
        generation: int,
        generation_evals: List[EvolutionCandidateEvaluation],
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

        while len(next_population) < self.config.population_size:
            parent_a, parent_b = self._select_parents(generation_evals)
            child_chromosome = crossover_chromosomes(
                parent_a.metadata["chromosome"],
                parent_b.metadata["chromosome"],
                mode=self.config.crossover_mode,
                include_fixed=False,
                rng=self._run_rng,
            )
            child_chromosome = mutate_chromosome(
                child_chromosome,
                mutation_rate=self.config.mutation_rate,
                mutation_scale=self.config.mutation_scale,
                mutation_mode=self.config.mutation_mode,
                boundary_mode=self.config.boundary_mode,
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
        self, generation: int, generation_evals: List[EvolutionCandidateEvaluation]
    ) -> EvolutionGenerationSummary:
        best = max(generation_evals, key=lambda item: item.fitness)
        fitness_values = [evaluation.fitness for evaluation in generation_evals]
        gene_statistics = self._build_gene_statistics(generation_evals)
        best_chromosome = {
            gene.name: gene.value
            for gene in best.metadata["chromosome"].genes
        }
        return EvolutionGenerationSummary(
            generation=generation,
            best_fitness=max(fitness_values),
            mean_fitness=statistics.mean(fitness_values),
            min_fitness=min(fitness_values),
            best_candidate_id=best.candidate_id,
            gene_statistics=gene_statistics,
            best_chromosome=best_chromosome,
        )

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

        with open(summaries_path, "w", encoding="utf-8") as summaries_file:
            json.dump([asdict(summary) for summary in result.generation_summaries], summaries_file, indent=2)

        serialized_evaluations: List[Dict[str, Any]] = []
        for evaluation in result.evaluations:
            payload: Dict[str, Any] = {
                "candidate_id": evaluation.candidate_id,
                "generation": evaluation.generation,
                "fitness": evaluation.fitness,
                "learning_rate": evaluation.learning_rate,
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

        logger.info(
            "evolution_experiment_persisted",
            output_dir=self.config.output_dir,
            summaries_path=summaries_path,
            lineage_path=lineage_path,
            num_generations=self.config.num_generations,
        )
