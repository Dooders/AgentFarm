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

# ---------------------------------------------------------------------------
# Named presets (mirror run_evolution_experiment.py)
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, object]] = {
    "stable_hyper_evo": {
        "selection_method": EvolutionSelectionMethod.TOURNAMENT.value,
        "boundary_mode": BoundaryMode.REFLECT.value,
        "mutation_rate": 0.20,
        "mutation_scale": 0.15,
        "adaptive_mutation": True,
        "tournament_size": 3,
        "elitism_count": 1,
    },
}


def _parse_per_gene_multipliers(raw: str | None, *, label: str) -> dict[str, float]:
    """Parse a comma-separated ``gene=value`` string into a multiplier dict."""
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


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Formatter that preserves raw description layout and also shows defaults."""


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
        formatter_class=_HelpFormatter,
    )
    # ------------------------------------------------------------------
    # Cohort-specific flags
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Shared preset / environment flags
    # ------------------------------------------------------------------
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(PRESETS),
        help="Load a named configuration preset.",
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
    parser.add_argument("--generations", type=int, default=2, help="Number of generations per seed.")
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
        help="Built-in fitness metric.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=EvolutionSelectionMethod.TOURNAMENT.value,
        choices=[method.value for method in EvolutionSelectionMethod],
        help="Parent selection strategy.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Mutation probability per gene.")
    parser.add_argument("--mutation-scale", type=float, default=0.2, help="Mutation magnitude for mutated genes.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament bracket size.")
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=BoundaryMode.CLAMP.value,
        choices=[mode.value for mode in BoundaryMode],
        help="Boundary strategy after mutation overshoots gene bounds.",
    )
    parser.add_argument("--boundary-penalty-enabled", action="store_true", help="Enable soft near-boundary fitness penalty.")
    parser.add_argument("--boundary-penalty-strength", type=float, default=0.01)
    parser.add_argument("--boundary-penalty-threshold", type=float, default=0.05)
    parser.add_argument(
        "--crossover-mode",
        type=str,
        default=CrossoverMode.UNIFORM.value,
        choices=[mode.value for mode in CrossoverMode],
        help="Crossover operator.",
    )
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--num-crossover-points", type=int, default=2)
    parser.add_argument("--elitism-count", type=int, default=1)
    parser.add_argument(
        "--adaptive-mutation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive mutation.",
    )
    parser.add_argument("--adaptive-stall-window", type=int, default=3)
    parser.add_argument("--adaptive-improve-threshold", type=float, default=1e-6)
    parser.add_argument("--adaptive-stall-multiplier", type=float, default=1.5)
    parser.add_argument("--adaptive-improve-multiplier", type=float, default=0.8)
    parser.add_argument("--adaptive-diversity-threshold", type=float, default=0.05)
    parser.add_argument("--adaptive-diversity-multiplier", type=float, default=1.5)
    parser.add_argument("--adaptive-disable-fitness", action="store_true")
    parser.add_argument("--adaptive-disable-diversity", action="store_true")
    parser.add_argument("--adaptive-max-step-multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive-default-per-gene", action="store_true")
    parser.add_argument("--adaptive-per-gene-rate", type=str, default=None)
    parser.add_argument("--adaptive-per-gene-scale", type=str, default=None)
    # ------------------------------------------------------------------
    # Convergence criteria
    # ------------------------------------------------------------------
    parser.add_argument("--convergence-enabled", action="store_true")
    parser.add_argument("--convergence-fitness-window", type=int, default=5)
    parser.add_argument("--convergence-fitness-threshold", type=float, default=1e-4)
    parser.add_argument("--convergence-diversity-window", type=int, default=3)
    parser.add_argument("--convergence-diversity-threshold", type=float, default=0.01)
    parser.add_argument("--convergence-min-generations", type=int, default=1)
    parser.add_argument("--convergence-no-early-stop", action="store_true")
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
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
                per_gene_rate_multipliers=_parse_per_gene_multipliers(
                    args.adaptive_per_gene_rate, label="--adaptive-per-gene-rate"
                ),
                per_gene_scale_multipliers=_parse_per_gene_multipliers(
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
            "experiment_config": _serialize_experiment_config(experiment_config_template),
        }
        manifest_path = os.path.join(args.output_dir, "cohort_manifest.json")
        with open(manifest_path, "w") as manifest_file:
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
