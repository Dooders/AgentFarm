#!/usr/bin/env python3
"""Seed sweep for stable resource profiles (conservative / balanced / buffered).

Runs the IntrinsicEvolutionExperiment across multiple seeds per sub-profile
to bracket the speciation-trajectory direction and learning_rate shift
established in the single-seed stable-profile comparison.

The sub-profile overrides applied on top of the ``stable`` preset are
defined once in :data:`farm.runners.intrinsic_evolution_experiment.STABLE_SUB_PROFILES`.

Outputs
-------
Per-run artifacts are written to::

    {output_dir}/stable_{profile}/seed_{seed}/

A machine-readable manifest covering all completed runs is written to::

    {output_dir}/sweep_manifest.json

After the sweep, run ``scripts/analyze_stable_profile_seed_sweep.py
--sweep-dir {output_dir}`` to aggregate results into summary stats and
comparison plots.

Example
-------
::

    # 4 seeds, all three profiles
    python scripts/run_stable_profile_seed_sweep.py \\
        --seeds 42 7 19 101 \\
        --output-dir experiments/stable_profile_sweep

    # Single profile, 8 seeds
    python scripts/run_stable_profile_seed_sweep.py \\
        --profiles conservative \\
        --seeds 42 7 19 101 137 256 512 999 \\
        --output-dir experiments/stable_profile_sweep

    # Print the planned matrix without running anything
    python scripts/run_stable_profile_seed_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    CrossoverMode,
    MutationMode,
)
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode  # noqa: E402
from scripts._warmstart_cli import (  # noqa: E402
    add_warmstart_tuning_arguments,
    warmstart_tuning_kwargs,
)
from farm.runners.intrinsic_evolution_experiment import (  # noqa: E402
    STABLE_SUB_PROFILES,
    InitialConditionsConfig,
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
    SpeciationConfig,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402

DEFAULT_SEEDS: List[int] = [42, 7, 19, 101, 137, 256]
DEFAULT_PROFILES: List[str] = ["conservative", "balanced", "buffered"]

CROSSOVER_MODES: List[str] = [m.value for m in CrossoverMode]
COPARENT_STRATEGIES: List[str] = [
    "nearest_alive_same_type",
    "random_alive_same_type",
]
INHERITANCE_MODES: List[str] = ["baldwinian", "lamarckian", "p2", "p3", "p4"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed sweep for stable resource profiles.\n\n"
            "Runs IntrinsicEvolutionExperiment for each (profile, seed) pair "
            "and writes per-run artifacts under {output_dir}/stable_{profile}/seed_{seed}/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        choices=list(STABLE_SUB_PROFILES),
        metavar="PROFILE",
        help="Sub-profiles to run (conservative, balanced, buffered).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="SEED",
        help="Seeds to sweep over.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/stable_profile_sweep",
        help="Base directory for all run artifacts.",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--selection-pressure", type=str, default="low")
    parser.add_argument(
        "--inheritance-mode",
        type=str,
        default="baldwinian",
        choices=INHERITANCE_MODES,
        help=(
            "Policy inheritance mode for offspring. "
            "'baldwinian' (default): cold-start. "
            "'lamarckian' (P1): weights only. "
            "'p2': weights + plasticity damping. "
            "'p3': weights + optimizer state + replay slice. "
            "'p4': gated/blended transfer."
        ),
    )
    add_warmstart_tuning_arguments(parser)
    parser.add_argument("--initial-diversity-mutation-rate", type=float, default=1.0)
    parser.add_argument("--initial-diversity-mutation-scale", type=float, default=0.25)
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the entire sweep on the first run failure.",
    )
    parser.add_argument(
        "--disk-database",
        action="store_true",
        help=(
            "Use disk-backed SQLite instead of :memory: for each run (recommended for long "
            "horizons). Writes simulation_<id>.db under each seed output directory."
        ),
    )
    parser.add_argument(
        "--crossover-enabled",
        action="store_true",
        help=(
            "Enable in-situ chromosome crossover during reproduction. When set, each "
            "reproduction picks a co-parent and runs the configured crossover operator "
            "before mutation. When unset (default), reproduction inherits the parent's "
            "chromosome unchanged (modulo mutation)."
        ),
    )
    parser.add_argument(
        "--crossover-mode",
        type=str,
        default="uniform",
        choices=CROSSOVER_MODES,
        help="Crossover operator (only used when --crossover-enabled is set).",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="BLX-alpha for blend crossover (only used when --crossover-mode=blend).",
    )
    parser.add_argument(
        "--num-crossover-points",
        type=int,
        default=2,
        help=(
            "Number of pivots for multi-point crossover "
            "(only used when --crossover-mode=multi_point)."
        ),
    )
    parser.add_argument(
        "--coparent-strategy",
        type=str,
        default="nearest_alive_same_type",
        choices=COPARENT_STRATEGIES,
        help="How to pick the co-parent when crossover is enabled.",
    )
    parser.add_argument(
        "--coparent-max-radius",
        type=float,
        default=None,
        help="Spatial cap on the co-parent search (None = unbounded).",
    )
    parser.add_argument(
        "--allow-cross-type-pollination",
        action="store_true",
        help=(
            "Allow co-parents of any agent_type. When unset (default), only "
            "same-type agents are eligible."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip per-(profile, seed) runs whose output directory already contains "
            "a completed metadata file with num_steps_completed >= --num-steps."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned (profile, seed) matrix and resolved overrides, then exit.",
    )
    return parser


def _build_run(profile: str, seed: int, args: argparse.Namespace, run_dir: Path) -> IntrinsicEvolutionExperiment:
    """Construct (but do not execute) a single (profile, seed) experiment."""
    overrides = STABLE_SUB_PROFILES[profile]

    base_config = SimulationConfig.from_centralized_config(environment=args.environment)
    if getattr(args, "disk_database", False):
        base_config.database.use_in_memory_db = False
        base_config.database.persist_db_on_completion = True

    base_config.initial_diversity = InitialDiversityConfig(
        mode=SeedingMode.INDEPENDENT_MUTATION,
        mutation_rate=args.initial_diversity_mutation_rate,
        mutation_scale=args.initial_diversity_mutation_scale,
        mutation_mode=MutationMode.GAUSSIAN,
        boundary_mode=BoundaryMode.REFLECT,
        seed=seed,
    )

    policy = IntrinsicEvolutionPolicy(
        enabled=True,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode.GAUSSIAN,
        boundary_mode=BoundaryMode.REFLECT,
        crossover_enabled=bool(getattr(args, "crossover_enabled", False)),
        crossover_mode=CrossoverMode(getattr(args, "crossover_mode", "uniform")),
        blend_alpha=float(getattr(args, "blend_alpha", 0.5)),
        num_crossover_points=int(getattr(args, "num_crossover_points", 2)),
        coparent_strategy=getattr(args, "coparent_strategy", "nearest_alive_same_type"),
        coparent_max_radius=getattr(args, "coparent_max_radius", None),
        allow_cross_type_pollination=bool(
            getattr(args, "allow_cross_type_pollination", False)
        ),
        selection_pressure=args.selection_pressure,
        inheritance_mode=str(getattr(args, "inheritance_mode", "baldwinian")),
        **warmstart_tuning_kwargs(args),
        seed=seed,
    )

    initial_conditions = InitialConditionsConfig(
        profile="stable",
        warmup_steps=args.warmup_steps,
        **overrides,
    )

    speciation = SpeciationConfig(
        enabled=True,
        algorithm="gmm",
        max_k=4,
        seed=seed,
        scaler="none",
    )

    exp_config = IntrinsicEvolutionExperimentConfig(
        num_steps=args.num_steps,
        snapshot_interval=args.snapshot_interval,
        install_default_initial_diversity=True,
        initial_conditions=initial_conditions,
        policy=policy,
        speciation=speciation,
        output_dir=str(run_dir),
        seed=seed,
    )
    return IntrinsicEvolutionExperiment(base_config, exp_config)


def _maybe_resume_skip(
    run_dir: Path,
    args: argparse.Namespace,
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """If ``args.resume`` and prior metadata shows a full run, return a completed record.

    Otherwise return ``None`` so the caller should execute the simulation.
    """
    if not getattr(args, "resume", False):
        return None
    meta_path = run_dir / "intrinsic_evolution_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        completed = meta.get("num_steps_completed")
        if completed is None:
            return None
        if int(completed) < int(args.num_steps):
            return None
        record.update(
            status="skipped_done",
            elapsed_seconds=0.0,
            num_steps_completed=int(completed),
            final_population=meta.get("final_population"),
            error=None,
        )
        return record
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _run_one(
    profile: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    logger: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute a single (profile, seed) experiment and return its manifest record."""
    run_dir = output_dir / f"stable_{profile}" / f"seed_{seed}"

    record: Dict[str, Any] = {
        "profile": profile,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "pending",
        "elapsed_seconds": None,
        "num_steps_completed": None,
        "final_population": None,
        "error": None,
    }

    skipped = _maybe_resume_skip(run_dir, args, record)
    if skipped is not None:
        logger.info(
            "seed_sweep_run_skipped_resume",
            profile=profile,
            seed=seed,
            run_dir=str(run_dir),
            num_steps_completed=record.get("num_steps_completed"),
        )
        return True, record

    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("seed_sweep_run_start", profile=profile, seed=seed, run_dir=str(run_dir))

    try:
        experiment = _build_run(profile, seed, args, run_dir)
        t0 = time.time()
        result = experiment.run()
        elapsed = time.time() - t0
        record.update(
            status="ok",
            elapsed_seconds=round(elapsed, 3),
            num_steps_completed=result.num_steps_completed,
            final_population=result.final_population,
        )
        logger.info(
            "seed_sweep_run_ok",
            profile=profile,
            seed=seed,
            elapsed=round(elapsed, 1),
            steps=result.num_steps_completed,
            population=result.final_population,
        )
        return True, record
    except Exception as exc:
        record.update(status="error", error=str(exc))
        logger.error("seed_sweep_run_failed", profile=profile, seed=seed, error=str(exc))
        traceback.print_exc(file=sys.stderr)
        return False, record


def _crossover_settings_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Snapshot of the resolved crossover-related CLI args for the manifest."""
    return {
        "crossover_enabled": bool(getattr(args, "crossover_enabled", False)),
        "crossover_mode": str(getattr(args, "crossover_mode", "uniform")),
        "blend_alpha": float(getattr(args, "blend_alpha", 0.5)),
        "num_crossover_points": int(getattr(args, "num_crossover_points", 2)),
        "coparent_strategy": str(
            getattr(args, "coparent_strategy", "nearest_alive_same_type")
        ),
        "coparent_max_radius": getattr(args, "coparent_max_radius", None),
        "allow_cross_type_pollination": bool(
            getattr(args, "allow_cross_type_pollination", False)
        ),
    }


def _inheritance_settings_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Snapshot of inheritance mode related args for the manifest."""
    return {
        "inheritance_mode": str(getattr(args, "inheritance_mode", "baldwinian")),
        **warmstart_tuning_kwargs(args),
    }


def _print_dry_run_plan(args: argparse.Namespace, output_dir: Path) -> None:
    print("Stable-profile seed sweep — DRY RUN")
    print(f"  output_dir : {output_dir}")
    print(f"  profiles   : {args.profiles}")
    print(f"  seeds      : {args.seeds}")
    print(f"  num_steps  : {args.num_steps} (warmup {args.warmup_steps}, "
          f"snapshot/{args.snapshot_interval})")
    print(f"  disk_db    : {getattr(args, 'disk_database', False)}")
    print(f"  inheritance: {_inheritance_settings_dict(args)}")
    print(f"  crossover  : {_crossover_settings_dict(args)}")
    print(f"  total runs : {len(args.profiles) * len(args.seeds)}")
    print()
    print("Resolved sub-profile overrides:")
    for profile in args.profiles:
        overrides = STABLE_SUB_PROFILES[profile]
        print(f"  stable_{profile}: {overrides}")


def _execute_sweep(
    args: argparse.Namespace, output_dir: Path, logger: Any
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Run every (profile, seed) pair, honoring --fail-fast.

    Returns (records, n_ok, n_fail).  Extracted from main() so the
    abort-on-failure flow can be tested without replicating it inline.
    """
    sweep_records: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for profile in args.profiles:
        for seed in args.seeds:
            success, record = _run_one(profile, seed, args, output_dir, logger)
            sweep_records.append(record)
            if success:
                n_ok += 1
                continue
            n_fail += 1
            if args.fail_fast:
                print("Aborting sweep (--fail-fast).", file=sys.stderr)
                return sweep_records, n_ok, n_fail
    return sweep_records, n_ok, n_fail


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.dry_run:
        _print_dry_run_plan(args, output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = get_logger(__name__)

    total = len(args.profiles) * len(args.seeds)
    print(
        f"Stable-profile seed sweep: {len(args.profiles)} profile(s) × "
        f"{len(args.seeds)} seed(s) = {total} run(s)"
    )
    print(f"  profiles : {args.profiles}")
    print(f"  seeds    : {args.seeds}")
    print(f"  output   : {output_dir}")

    sweep_records, n_ok, n_fail = _execute_sweep(args, output_dir, logger)

    manifest = {
        "sweep_type": "stable_profile_seed_sweep",
        "profiles": args.profiles,
        "seeds": args.seeds,
        "disk_database": getattr(args, "disk_database", False),
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale,
        "selection_pressure": args.selection_pressure,
        "inheritance": _inheritance_settings_dict(args),
        "crossover": _crossover_settings_dict(args),
        "sub_profile_overrides": {p: STABLE_SUB_PROFILES[p] for p in args.profiles},
        "runs": sweep_records,
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    manifest_path = output_dir / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nSweep complete: {n_ok}/{total} runs succeeded.")
    print(f"Manifest: {manifest_path}")
    if n_fail:
        print(f"  {n_fail} run(s) failed — see manifest for details.")
    print(
        f"\nNext step: python scripts/analyze_stable_profile_seed_sweep.py "
        f"--sweep-dir {output_dir}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
