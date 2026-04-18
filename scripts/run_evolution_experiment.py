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
    AdaptiveMutationConfig,
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


def _parse_per_gene_multipliers(raw: str | None, *, label: str) -> dict[str, float]:
    """Parse a comma-separated ``gene=value`` string into a multiplier dict.

    Returns an empty dict for ``None`` or empty input.  Raises ``ValueError``
    on malformed entries; ``argparse`` will surface this as a CLI error.
    """
    if not raw:
        return {}
    multipliers: dict[str, float] = {}
    for entry in raw.split(","):
        token = entry.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"{label} entry '{token}' must be of the form 'gene_name=value'."
            )
        name, _, value_str = token.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"{label} entry '{token}' has an empty gene name.")
        try:
            multipliers[name] = float(value_str)
        except ValueError as exc:
            raise ValueError(
                f"{label} entry '{token}' has a non-numeric value."
            ) from exc
    return multipliers


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
    parser.add_argument(
        "--adaptive-mutation",
        action="store_true",
        help="Enable adaptive mutation rate/scale based on fitness progress and diversity.",
    )
    parser.add_argument(
        "--adaptive-stall-window",
        type=int,
        default=3,
        help="Generations to look back when detecting a fitness stall.",
    )
    parser.add_argument(
        "--adaptive-improve-threshold",
        type=float,
        default=1e-6,
        help="Minimum best-fitness improvement over the window that counts as progress.",
    )
    parser.add_argument(
        "--adaptive-stall-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to mutation rate/scale when the search stalls.",
    )
    parser.add_argument(
        "--adaptive-improve-multiplier",
        type=float,
        default=0.8,
        help="Multiplier applied to mutation rate/scale when fitness is improving.",
    )
    parser.add_argument(
        "--adaptive-diversity-threshold",
        type=float,
        default=0.05,
        help="Normalized diversity at or below which mutation is boosted.",
    )
    parser.add_argument(
        "--adaptive-diversity-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to mutation rate/scale when diversity collapses.",
    )
    parser.add_argument(
        "--adaptive-disable-fitness",
        action="store_true",
        help="When --adaptive-mutation is set, skip the fitness-progress adaptation rule.",
    )
    parser.add_argument(
        "--adaptive-disable-diversity",
        action="store_true",
        help="When --adaptive-mutation is set, skip the diversity-collapse adaptation rule.",
    )
    parser.add_argument(
        "--adaptive-per-gene-rate",
        type=str,
        default=None,
        help=(
            "Comma-separated per-gene mutation-rate multipliers, e.g. "
            "'learning_rate=0.5,gamma=2.0'.  Each value must be non-negative."
        ),
    )
    parser.add_argument(
        "--adaptive-per-gene-scale",
        type=str,
        default=None,
        help=(
            "Comma-separated per-gene mutation-scale multipliers, e.g. "
            "'learning_rate=0.5,gamma=2.0'.  Each value must be non-negative."
        ),
    )
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
        adaptive_mutation=args.adaptive_mutation,
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
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=args.adaptive_mutation,
                use_fitness_adaptation=not args.adaptive_disable_fitness,
                use_diversity_adaptation=not args.adaptive_disable_diversity,
                stall_window=args.adaptive_stall_window,
                improvement_threshold=args.adaptive_improve_threshold,
                stall_multiplier=args.adaptive_stall_multiplier,
                improve_multiplier=args.adaptive_improve_multiplier,
                diversity_threshold=args.adaptive_diversity_threshold,
                diversity_multiplier=args.adaptive_diversity_multiplier,
                per_gene_rate_multipliers=_parse_per_gene_multipliers(
                    args.adaptive_per_gene_rate, label="--adaptive-per-gene-rate"
                ),
                per_gene_scale_multipliers=_parse_per_gene_multipliers(
                    args.adaptive_per_gene_scale, label="--adaptive-per-gene-scale"
                ),
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

        last_summary = result.generation_summaries[-1] if result.generation_summaries else None
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
            "adaptive_mutation_enabled": args.adaptive_mutation,
            "final_mutation_rate_multiplier": (
                last_summary.mutation_rate_multiplier if last_summary else None
            ),
            "final_mutation_scale_multiplier": (
                last_summary.mutation_scale_multiplier if last_summary else None
            ),
            "final_adaptive_event": last_summary.adaptive_event if last_summary else None,
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
