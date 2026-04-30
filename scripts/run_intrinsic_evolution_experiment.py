#!/usr/bin/env python3
"""CLI entrypoint for the in-situ ``IntrinsicEvolutionExperiment`` runner.

This is the intrinsic-evolution analogue of
``scripts/run_evolution_experiment.py``.  It runs a *single* simulation in
which every agent carries its own
:class:`~farm.core.hyperparameter_chromosome.HyperparameterChromosome`;
selection emerges from the shared resource environment rather than from an
outer-loop GA.

The script writes:

- ``intrinsic_gene_trajectory.jsonl`` (per-step aggregate gene stats)
- ``intrinsic_gene_snapshots.jsonl`` (full per-agent chromosomes every
  ``--snapshot-interval`` steps)
- ``intrinsic_evolution_metadata.json`` (final summary + resolved policy)
- ``cluster_lineage.jsonl`` when ``--enable-speciation`` is set
- ``run_manifest.json`` (CLI args + resolved seed for reproducibility)
- whatever the underlying ``run_simulation`` writes (config dump, sqlite db)

Use ``scripts/analyze_intrinsic_evolution.py`` afterwards to produce
visualisations from the artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    CrossoverMode,
    MutationMode,
)
from farm.core.initial_diversity import (  # noqa: E402
    InitialDiversityConfig,
    SeedingMode,
)
from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import (  # noqa: E402
    InitialConditionsConfig,
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an intrinsic (in-situ) hyperparameter-evolution experiment.\n\n"
            "Selection emerges from a single simulation: each agent carries\n"
            "its own HyperparameterChromosome, offspring inherit + mutate it\n"
            "(optionally with crossover from a co-parent), and lineages that\n"
            "fail to reproduce die out."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/intrinsic_evolution",
    )

    # Mutation / boundary
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.1)
    parser.add_argument(
        "--mutation-mode",
        type=str,
        default=MutationMode.GAUSSIAN.value,
        choices=[m.value for m in MutationMode],
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=BoundaryMode.REFLECT.value,
        choices=[m.value for m in BoundaryMode],
    )
    parser.add_argument("--interior-bias-fraction", type=float, default=1e-3)

    # Initial conditions (startup state overrides; see InitialConditionsConfig).
    parser.add_argument(
        "--initial-conditions-profile",
        type=str,
        default=None,
        choices=["stable", "stress", "exploratory", "legacy"],
        help=(
            "Preset for initial simulation state. "
            "'stable' (default) reduces early boom-bust by giving agents and the "
            "environment more starting resources. 'legacy' reproduces pre-feature "
            "behavior exactly."
        ),
    )
    parser.add_argument(
        "--initial-agent-resource-level",
        type=float,
        default=None,
        help="Override: starting resource level for each agent (overrides profile).",
    )
    parser.add_argument(
        "--initial-resource-count",
        type=int,
        default=None,
        help="Override: number of resource nodes at simulation start (overrides profile).",
    )
    parser.add_argument(
        "--resource-regen-rate",
        type=float,
        default=None,
        help="Override: resource regeneration probability per step (overrides profile).",
    )
    parser.add_argument(
        "--resource-regen-amount",
        type=int,
        default=None,
        help="Override: resource amount regenerated per node per step (overrides profile).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Extra steps run before telemetry collection begins. "
            "Gene-trajectory snapshots are suppressed during the warmup phase."
        ),
    )
    parser.add_argument(
        "--transient-window",
        type=int,
        default=50,
        help="Number of initial steps used to compute startup-transient metrics.",
    )

    # Initial diversity seeding (platform-wide; see farm/core/initial_diversity.py).
    # The CLI default for --initial-diversity-mode is "independent_mutation".
    # Explicitly passing "none" disables seeding, so no default initial-diversity
    # configuration is installed and the starting population is a monoculture.
    parser.add_argument(
        "--initial-diversity-mode",
        type=str,
        default=SeedingMode.INDEPENDENT_MUTATION.value,
        choices=[m.value for m in SeedingMode],
        help="Initial genotype diversity strategy applied before the loop starts.",
    )
    parser.add_argument("--initial-diversity-mutation-rate", type=float, default=1.0)
    parser.add_argument("--initial-diversity-mutation-scale", type=float, default=0.25)
    parser.add_argument("--initial-diversity-min-distance", type=float, default=0.05)
    parser.add_argument("--initial-diversity-max-retries", type=int, default=32)

    # Crossover
    parser.add_argument(
        "--crossover",
        action="store_true",
        help="Enable crossover with a co-parent before mutation.",
    )
    parser.add_argument(
        "--crossover-mode",
        type=str,
        default=CrossoverMode.UNIFORM.value,
        choices=[m.value for m in CrossoverMode],
    )
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--num-crossover-points", type=int, default=2)
    parser.add_argument(
        "--coparent-strategy",
        type=str,
        default="nearest_alive_same_type",
        choices=["nearest_alive_same_type", "random_alive_same_type"],
    )
    parser.add_argument("--coparent-max-radius", type=float, default=None)

    # Selection pressure (preset name or float)
    parser.add_argument(
        "--selection-pressure",
        type=str,
        default="low",
        help="Preset ('none','low','medium','high') or float in [0,1].",
    )

    # Speciation tracking
    parser.add_argument("--enable-speciation", action="store_true", default=True)
    parser.add_argument("--no-speciation", dest="enable_speciation", action="store_false")
    parser.add_argument(
        "--speciation-algorithm",
        type=str,
        default="gmm",
        choices=["gmm", "dbscan"],
    )
    parser.add_argument("--speciation-max-k", type=int, default=4)
    parser.add_argument(
        "--speciation-scaler",
        type=str,
        default="none",
        choices=["none", "standard", "robust"],
        help="Feature scaling applied before speciation clustering. Default: none.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _parse_selection_pressure(raw: str):
    """Coerce the selection-pressure CLI string to either a preset or a float."""
    if raw is None:
        return None
    if raw.lower() in {"none", "low", "medium", "high"}:
        return raw.lower()
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(
            f"--selection-pressure must be a preset or float in [0,1]; got {raw!r}"
        ) from exc


def _build_run(args: argparse.Namespace) -> IntrinsicEvolutionExperiment:
    base_config = SimulationConfig.from_centralized_config(
        environment=args.environment,
        profile=args.profile,
    )

    base_config.initial_diversity = InitialDiversityConfig(
        mode=SeedingMode(args.initial_diversity_mode),
        mutation_rate=args.initial_diversity_mutation_rate,
        mutation_scale=args.initial_diversity_mutation_scale,
        mutation_mode=MutationMode(args.mutation_mode),
        boundary_mode=BoundaryMode(args.boundary_mode),
        interior_bias_fraction=args.interior_bias_fraction,
        max_retries_per_agent=args.initial_diversity_max_retries,
        min_distance=args.initial_diversity_min_distance,
        seed=args.seed,
    )

    policy = IntrinsicEvolutionPolicy(
        enabled=True,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode(args.mutation_mode),
        boundary_mode=BoundaryMode(args.boundary_mode),
        interior_bias_fraction=args.interior_bias_fraction,
        crossover_enabled=args.crossover,
        crossover_mode=CrossoverMode(args.crossover_mode),
        blend_alpha=args.blend_alpha,
        num_crossover_points=args.num_crossover_points,
        coparent_strategy=args.coparent_strategy,
        coparent_max_radius=args.coparent_max_radius,
        selection_pressure=_parse_selection_pressure(args.selection_pressure),
        seed=args.seed,
    )

    initial_conditions_profile = args.initial_conditions_profile or "stable"
    initial_conditions = InitialConditionsConfig(
        profile=initial_conditions_profile,
        initial_agent_resource_level=args.initial_agent_resource_level,
        initial_resource_count=args.initial_resource_count,
        resource_regen_rate=args.resource_regen_rate,
        resource_regen_amount=args.resource_regen_amount,
        warmup_steps=args.warmup_steps,
        transient_window=args.transient_window,
    )

    config = IntrinsicEvolutionExperimentConfig(
        num_steps=args.num_steps,
        snapshot_interval=args.snapshot_interval,
        install_default_initial_diversity=(
            SeedingMode(args.initial_diversity_mode) is not SeedingMode.NONE
        ),
        initial_conditions=initial_conditions,
        policy=policy,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return IntrinsicEvolutionExperiment(base_config, config)


def main() -> int:
    args = _build_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = get_logger(__name__)
    if args.initial_conditions_profile is None:
        logger.warning(
            "intrinsic_evolution_cli_default_initial_conditions_profile",
            profile="stable",
            note=(
                "No --initial-conditions-profile provided; defaulting to 'stable'. "
                "Pass --initial-conditions-profile legacy to match older runs."
            ),
        )

    manifest = {
        "script": "scripts/run_intrinsic_evolution_experiment.py",
        "args": vars(args),
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    logger.info(
        "intrinsic_evolution_cli_start",
        environment=args.environment,
        num_steps=args.num_steps,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        crossover=args.crossover,
        selection_pressure=args.selection_pressure,
        enable_speciation=args.enable_speciation,
    )

    # The runner internally constructs its own GeneTrajectoryLogger without a
    # speciation toggle, so when the user opts in we monkey-patch
    # GeneTrajectoryLogger so the runner picks up our extended logger.  This
    # keeps the runner code unchanged while still exposing speciation here.
    if args.enable_speciation:
        import farm.runners.intrinsic_evolution_experiment as runner_mod

        original_logger_cls = runner_mod.GeneTrajectoryLogger

        def _make_logger(*pos, **kw):
            return original_logger_cls(
                *pos,
                enable_speciation=True,
                speciation_algorithm=args.speciation_algorithm,
                speciation_max_k=args.speciation_max_k,
                speciation_seed=args.seed,
                speciation_scaler=args.speciation_scaler,
                **kw,
            )

        runner_mod.GeneTrajectoryLogger = _make_logger  # type: ignore[assignment]
        try:
            start = time.time()
            result = _build_run(args).run()
            elapsed = time.time() - start
        finally:
            runner_mod.GeneTrajectoryLogger = original_logger_cls
    else:
        start = time.time()
        result = _build_run(args).run()
        elapsed = time.time() - start

    summary = {
        "elapsed_seconds": round(elapsed, 3),
        "num_steps_completed": result.num_steps_completed,
        "final_population": result.final_population,
        "final_gene_statistics": result.final_gene_statistics,
        "startup_transient_metrics": result.startup_transient_metrics,
        "output_dir": args.output_dir,
    }
    print(json.dumps(summary, indent=2))

    # Touch the underlying GeneTrajectoryLogger to make sure speciation flag
    # is not silently dropped: report it in the summary file too.
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
