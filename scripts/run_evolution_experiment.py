#!/usr/bin/env python3
"""CLI entrypoint for multi-generation hyperparameter evolution experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running directly from repo root without installing the package
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.runners import (  # noqa: E402
    EvolutionExperiment,
    EvolutionExperimentConfig,
    EvolutionFitnessMetric,
    EvolutionSelectionMethod,
)
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    BoundaryPenaltyConfig,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-generation hyperparameter evolution experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "production", "testing"],
        help="Centralized config environment.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["benchmark", "simulation", "research"],
        help="Optional centralized config profile.",
    )
    parser.add_argument("--generations", type=int, default=2, help="Number of generations to run.")
    parser.add_argument("--population-size", type=int, default=4, help="Candidates per generation.")
    parser.add_argument(
        "--steps-per-candidate",
        type=int,
        default=20,
        help="Simulation steps used to evaluate each candidate.",
    )
    parser.add_argument(
        "--fitness-metric",
        type=str,
        default=EvolutionFitnessMetric.FINAL_POPULATION.value,
        choices=[metric.value for metric in EvolutionFitnessMetric],
        help="Built-in fitness metric used for parent selection.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=EvolutionSelectionMethod.TOURNAMENT.value,
        choices=[method.value for method in EvolutionSelectionMethod],
        help="Parent selection strategy.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Mutation probability per gene.")
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.2,
        help="Mutation magnitude for mutated genes.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament bracket size when tournament selection is used.",
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=BoundaryMode.CLAMP.value,
        choices=[mode.value for mode in BoundaryMode],
        help="Boundary strategy after mutation overshoots gene bounds.",
    )
    parser.add_argument(
        "--boundary-penalty-enabled",
        action="store_true",
        help="Enable soft near-boundary fitness penalty.",
    )
    parser.add_argument(
        "--boundary-penalty-strength",
        type=float,
        default=0.01,
        help="Max per-gene penalty at exact boundary when penalty is enabled.",
    )
    parser.add_argument(
        "--boundary-penalty-threshold",
        type=float,
        default=0.05,
        help="Near-boundary zone width as fraction of gene range (0, 0.5].",
    )
    parser.add_argument("--elitism-count", type=int, default=1, help="Top candidates copied to next generation.")
    parser.add_argument("--seed", type=int, default=42, help="Global deterministic seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/evolution",
        help="Directory where lineage and summaries are persisted.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Structured logging level.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(environment=args.environment, log_dir="logs", log_level=args.log_level, disable_console=False)
    logger = get_logger(__name__)

    logger.info(
        "evolution_experiment_cli_start",
        environment=args.environment,
        profile=args.profile,
        generations=args.generations,
        population_size=args.population_size,
        steps_per_candidate=args.steps_per_candidate,
        fitness_metric=args.fitness_metric,
        selection_method=args.selection_method,
        boundary_mode=args.boundary_mode,
        boundary_penalty_enabled=args.boundary_penalty_enabled,
        boundary_penalty_strength=args.boundary_penalty_strength,
        boundary_penalty_threshold=args.boundary_penalty_threshold,
        output_dir=args.output_dir,
    )

    try:
        base_config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile,
        )

        experiment_config = EvolutionExperimentConfig(
            num_generations=args.generations,
            population_size=args.population_size,
            num_steps_per_candidate=args.steps_per_candidate,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            boundary_mode=BoundaryMode(args.boundary_mode),
            boundary_penalty=BoundaryPenaltyConfig(
                enabled=args.boundary_penalty_enabled,
                penalty_strength=args.boundary_penalty_strength,
                near_boundary_threshold=args.boundary_penalty_threshold,
            ),
            selection_method=EvolutionSelectionMethod(args.selection_method),
            tournament_size=args.tournament_size,
            elitism_count=args.elitism_count,
            fitness_metric=EvolutionFitnessMetric(args.fitness_metric),
            seed=args.seed,
            output_dir=args.output_dir,
        )

        start = time.time()
        result = EvolutionExperiment(base_config, experiment_config).run()
        elapsed = time.time() - start

        summary = {
            "elapsed_seconds": round(elapsed, 3),
            "num_generations": len(result.generation_summaries),
            "num_evaluations": len(result.evaluations),
            "best_candidate_id": result.best_candidate.candidate_id,
            "best_fitness": result.best_candidate.fitness,
            "best_learning_rate": result.best_candidate.learning_rate,
            "best_parent_ids": list(result.best_candidate.parent_ids),
            "boundary_mode": args.boundary_mode,
            "boundary_penalty_enabled": args.boundary_penalty_enabled,
            "boundary_penalty_strength": args.boundary_penalty_strength,
            "boundary_penalty_threshold": args.boundary_penalty_threshold,
            "output_dir": args.output_dir,
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety guard
        logger.error(
            "evolution_experiment_cli_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
        )
        print(f"Evolution experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
