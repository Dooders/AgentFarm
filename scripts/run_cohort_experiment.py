#!/usr/bin/env python3
"""CLI entrypoint for multi-seed cohort evolution experiments.

Runs the same :class:`~farm.runners.EvolutionExperiment` configuration
over *N* random seeds and writes aggregate JSON/CSV artifacts to
``--output-dir``.  All evolution flags mirror :mod:`run_evolution_experiment`.

Example::

    source venv/bin/activate
    python scripts/run_cohort_experiment.py \\
      --preset stable_hyper_evo \\
      --generations 8 \\
      --population-size 10 \\
      --steps-per-candidate 80 \\
      --num-seeds 5 \\
      --base-seed 0 \\
      --output-dir experiments/cohort_smoke

Artifacts written to ``--output-dir``:

* ``cohort_manifest.json``   – resolved configuration snapshot (pre-run)
* ``cohort_aggregate.json``  – per-seed and aggregate statistics
* ``cohort_aggregate.csv``   – per-seed rows (notebook-ready)
* ``seed_<N>/``              – per-seed evolution artifacts (same layout as
                               :mod:`run_evolution_experiment`)
"""

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
    CohortRunner,
    ConvergenceCriteria,
    EvolutionExperimentConfig,
    EvolutionFitnessMetric,
    EvolutionSelectionMethod,
)
from farm.runners.cohort_runner import _serialize_experiment_config  # noqa: E402
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


def _build_parser() -> argparse.ArgumentParser:
    """Return a fully-configured argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed cohort hyperparameter evolution experiments.\n\n"
            "Executes the same evolution config over N seeds and writes aggregate\n"
            "JSON/CSV artifacts to --output-dir for confidence-aware comparison.\n\n"
            "Available presets:\n"
            "  stable_hyper_evo  tournament selection + reflect boundary + adaptive\n"
            "                    mutation (rate 0.20, scale 0.15)."
        ),
        formatter_class=EvolutionExperimentHelpFormatter,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of seeds to run in the cohort.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help=(
            "Base seed value.  Seeds are derived as "
            "[base_seed, base_seed+1, ..., base_seed+num_seeds-1]."
        ),
    )
    add_evolution_training_arguments(
        parser,
        presets=PRESETS,
        preset_help="Load a named configuration preset.",
        evolution_fitness_metric=EvolutionFitnessMetric,
        evolution_selection_method=EvolutionSelectionMethod,
        boundary_mode=BoundaryMode,
        crossover_mode=CrossoverMode,
        generations_help="Number of generations per seed.",
        fitness_metric_help="Built-in fitness metric.",
    )
    add_evolution_convergence_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/cohort",
        help="Root directory for cohort artifacts.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments, applying any named preset as baseline defaults."""
    parser = _build_parser()
    preset_discovery_args, _ = parser.parse_known_args()
    if preset_discovery_args.preset is not None:
        parser.set_defaults(**PRESETS[preset_discovery_args.preset])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(environment=args.environment, log_dir="logs", log_level=args.log_level, disable_console=False)
    logger = get_logger(__name__)

    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))

    logger.info(
        "cohort_experiment_cli_start",
        preset=args.preset,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        seeds=seeds,
        environment=args.environment,
        generations=args.generations,
        population_size=args.population_size,
        steps_per_candidate=args.steps_per_candidate,
        interior_bias_fraction=args.interior_bias_fraction,
        output_dir=args.output_dir,
    )

    try:
        base_config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile,
        )

        experiment_config_template = EvolutionExperimentConfig(
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
            # seed overridden per-run by CohortRunner
            seed=None,
            output_dir=None,
        )

        # Persist cohort manifest before running.
        manifest: dict[str, object] = {
            "script": "scripts/run_cohort_experiment.py",
            "preset": args.preset,
            "environment": args.environment,
            "profile": args.profile,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "seeds": seeds,
            "interior_bias_fraction": args.interior_bias_fraction,
            "experiment_config": _serialize_experiment_config(experiment_config_template),
        }
        manifest_path = os.path.join(args.output_dir, "cohort_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        logger.info("cohort_manifest_written", path=manifest_path)

        start = time.time()
        runner = CohortRunner(
            base_config=base_config,
            experiment_config_template=experiment_config_template,
            seeds=seeds,
            output_dir=args.output_dir,
        )
        aggregate = runner.run()
        elapsed = time.time() - start

        summary = {
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
            "total_elapsed_seconds": round(elapsed, 3),
            "output_dir": args.output_dir,
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety guard
        logger.error(
            "cohort_experiment_cli_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
        )
        print(f"Cohort experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
