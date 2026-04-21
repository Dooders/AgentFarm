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
    ConvergenceCriteria,
    EvolutionExperiment,
    EvolutionExperimentConfig,
    EvolutionFitnessMetric,
    EvolutionSelectionMethod,
)
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    BoundaryPenaltyConfig,
    CrossoverMode,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402

from scripts.evolution_experiment_cli import (  # noqa: E402
    EvolutionExperimentHelpFormatter,
    add_evolution_convergence_arguments,
    add_evolution_training_arguments,
    get_presets,
    parse_per_gene_multipliers,
)

PRESETS = get_presets(
    evolution_selection_method=EvolutionSelectionMethod,
    boundary_mode=BoundaryMode,
)


def _parse_per_gene_multipliers(raw: str | None, *, label: str) -> dict[str, float]:
    """Backward-compatible alias for tests/imports using the old helper name."""
    return parse_per_gene_multipliers(raw, label=label)


def _build_parser() -> argparse.ArgumentParser:
    """Return a fully-configured argument parser (without actually parsing)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-generation hyperparameter evolution experiments.\n\n"
            "Named presets (--preset) provide opinionated defaults that reflect current\n"
            "best-performing configurations.  Any explicit CLI flag still overrides the\n"
            "preset value.\n\n"
            "Available presets:\n"
            "  stable_hyper_evo  tournament selection + reflect boundary + adaptive\n"
            "                    mutation (rate 0.20, scale 0.15).  Prevents lower-bound\n"
            "                    collapse and diversity collapse seen with bare defaults."
        ),
        formatter_class=EvolutionExperimentHelpFormatter,
    )
    add_evolution_training_arguments(
        parser,
        presets=PRESETS,
        preset_help=(
            "Load a named configuration preset.  Preset values act as defaults; any "
            "explicit CLI flag still takes priority.  Available: %(choices)s."
        ),
        evolution_fitness_metric=EvolutionFitnessMetric,
        evolution_selection_method=EvolutionSelectionMethod,
        boundary_mode=BoundaryMode,
        crossover_mode=CrossoverMode,
    )
    add_evolution_convergence_arguments(parser)
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
    return parser


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments, applying any named preset as baseline defaults.

    Two-pass approach: the first parse discovers ``--preset``; if set, the
    preset values are installed as argparse defaults before the final parse so
    that any explicit flag the user supplied still wins over the preset.
    """
    parser = _build_parser()
    # First pass: discover --preset without failing on unknown/remaining args.
    preset_discovery_args, _ = parser.parse_known_args()
    if preset_discovery_args.preset is not None:
        # preset is already validated by choices=, so this lookup always succeeds.
        parser.set_defaults(**PRESETS[preset_discovery_args.preset])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(environment=args.environment, log_dir="logs", log_level=args.log_level, disable_console=False)
    logger = get_logger(__name__)

    logger.info(
        "evolution_experiment_cli_start",
        preset=args.preset,
        environment=args.environment,
        profile=args.profile,
        generations=args.generations,
        population_size=args.population_size,
        steps_per_candidate=args.steps_per_candidate,
        fitness_metric=args.fitness_metric,
        selection_method=args.selection_method,
        boundary_mode=args.boundary_mode,
        interior_bias_fraction=args.interior_bias_fraction,
        boundary_penalty_enabled=args.boundary_penalty_enabled,
        boundary_penalty_strength=args.boundary_penalty_strength,
        boundary_penalty_threshold=args.boundary_penalty_threshold,
        crossover_mode=args.crossover_mode,
        blend_alpha=args.blend_alpha,
        num_crossover_points=args.num_crossover_points,
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
            interior_bias_fraction=args.interior_bias_fraction,
            boundary_penalty=BoundaryPenaltyConfig(
                enabled=args.boundary_penalty_enabled,
                penalty_strength=args.boundary_penalty_strength,
                near_boundary_threshold=args.boundary_penalty_threshold,
            ),
            crossover_mode=CrossoverMode(args.crossover_mode),
            blend_alpha=args.blend_alpha,
            num_crossover_points=args.num_crossover_points,
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
                max_step_multiplier=args.adaptive_max_step_multiplier,
                use_default_per_gene_multipliers=args.adaptive_default_per_gene,
                per_gene_rate_multipliers=parse_per_gene_multipliers(
                    args.adaptive_per_gene_rate, label="--adaptive-per-gene-rate"
                ),
                per_gene_scale_multipliers=parse_per_gene_multipliers(
                    args.adaptive_per_gene_scale, label="--adaptive-per-gene-scale"
                ),
            ),
            convergence_criteria=ConvergenceCriteria(
                enabled=args.convergence_enabled,
                fitness_window=args.convergence_fitness_window,
                fitness_threshold=args.convergence_fitness_threshold,
                diversity_window=args.convergence_diversity_window,
                diversity_threshold=args.convergence_diversity_threshold,
                min_generations=args.convergence_min_generations,
                early_stop=not args.convergence_no_early_stop,
            ),
            selection_method=EvolutionSelectionMethod(args.selection_method),
            tournament_size=args.tournament_size,
            elitism_count=args.elitism_count,
            fitness_metric=EvolutionFitnessMetric(args.fitness_metric),
            seed=args.seed,
            output_dir=args.output_dir,
        )

        # Persist the resolved configuration before running so the manifest is
        # available even if the experiment is interrupted.
        manifest: dict[str, object] = {
            "script": "scripts/run_evolution_experiment.py",
            "preset": args.preset,
            "environment": args.environment,
            "profile": args.profile,
            "generations": args.generations,
            "population_size": args.population_size,
            "steps_per_candidate": args.steps_per_candidate,
            "fitness_metric": args.fitness_metric,
            "selection_method": args.selection_method,
            "mutation_rate": args.mutation_rate,
            "mutation_scale": args.mutation_scale,
            "tournament_size": args.tournament_size,
            "boundary_mode": args.boundary_mode,
            "interior_bias_fraction": args.interior_bias_fraction,
            "boundary_penalty_enabled": args.boundary_penalty_enabled,
            "boundary_penalty_strength": args.boundary_penalty_strength,
            "boundary_penalty_threshold": args.boundary_penalty_threshold,
            "crossover_mode": args.crossover_mode,
            "blend_alpha": args.blend_alpha,
            "num_crossover_points": args.num_crossover_points,
            "elitism_count": args.elitism_count,
            "adaptive_mutation": args.adaptive_mutation,
            "convergence_enabled": args.convergence_enabled,
            "convergence_fitness_window": args.convergence_fitness_window,
            "convergence_fitness_threshold": args.convergence_fitness_threshold,
            "convergence_diversity_window": args.convergence_diversity_window,
            "convergence_diversity_threshold": args.convergence_diversity_threshold,
            "convergence_min_generations": args.convergence_min_generations,
            "convergence_early_stop": not args.convergence_no_early_stop,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
        manifest_path = os.path.join(args.output_dir, "run_manifest.json")
        with open(manifest_path, "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        logger.info("evolution_experiment_manifest_written", path=manifest_path)

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
            "crossover_mode": args.crossover_mode,
            "blend_alpha": args.blend_alpha,
            "num_crossover_points": args.num_crossover_points,
            "adaptive_mutation_enabled": args.adaptive_mutation,
            "final_mutation_rate_multiplier": (
                last_summary.mutation_rate_multiplier if last_summary else None
            ),
            "final_mutation_scale_multiplier": (
                last_summary.mutation_scale_multiplier if last_summary else None
            ),
            "final_adaptive_event": last_summary.adaptive_event if last_summary else None,
            "converged": result.converged,
            "convergence_reason": result.convergence_reason,
            "generation_of_convergence": result.generation_of_convergence,
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
