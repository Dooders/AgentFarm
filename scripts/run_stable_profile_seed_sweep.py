#!/usr/bin/env python3
"""Seed sweep for stable resource profiles (conservative / balanced / buffered).

Runs the IntrinsicEvolutionExperiment across multiple seeds per sub-profile
to bracket the speciation-trajectory direction and learning_rate shift
established in the single-seed stable-profile comparison.

Sub-profile overrides (on top of the 'stable' preset):

  conservative: initial_agent_resource_level=8,  initial_resource_count=32,
                resource_regen_rate=0.14, resource_regen_amount=3
  balanced:     initial_agent_resource_level=10, initial_resource_count=34,
                resource_regen_rate=0.15, resource_regen_amount=3
  buffered:     initial_agent_resource_level=12, initial_resource_count=36,
                resource_regen_rate=0.16, resource_regen_amount=3

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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import BoundaryMode, MutationMode  # noqa: E402
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import (  # noqa: E402
    InitialConditionsConfig,
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402

# ── Sub-profile definitions ───────────────────────────────────────────────────
# These are manual overrides applied on top of the 'stable' InitialConditions
# preset, matching the three variants in stable_profile_comparison.md.

STABLE_SUB_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "initial_agent_resource_level": 8,
        "initial_resource_count": 32,
        "resource_regen_rate": 0.14,
        "resource_regen_amount": 3,
    },
    "balanced": {
        "initial_agent_resource_level": 10,
        "initial_resource_count": 34,
        "resource_regen_rate": 0.15,
        "resource_regen_amount": 3,
    },
    "buffered": {
        "initial_agent_resource_level": 12,
        "initial_resource_count": 36,
        "resource_regen_rate": 0.16,
        "resource_regen_amount": 3,
    },
}

DEFAULT_SEEDS: List[int] = [42, 7, 19, 101, 137, 256]
DEFAULT_PROFILES: List[str] = ["conservative", "balanced", "buffered"]


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
    # Fixed-policy options (matching the original stable-profile comparison)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--selection-pressure", type=str, default="low")
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
    return parser


def _run_one(
    profile: str,
    seed: int,
    args: argparse.Namespace,
    logger: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute a single (profile, seed) experiment.

    Returns (success, record) where record contains run metadata suitable for
    embedding in the sweep manifest.
    """
    overrides = STABLE_SUB_PROFILES[profile]
    run_dir = os.path.join(args.output_dir, f"stable_{profile}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info(
        "seed_sweep_run_start",
        profile=profile,
        seed=seed,
        run_dir=run_dir,
    )

    base_config = SimulationConfig.from_centralized_config(environment=args.environment)
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
        crossover_enabled=False,
        selection_pressure=args.selection_pressure,
        seed=seed,
    )

    initial_conditions = InitialConditionsConfig(
        profile="stable",
        warmup_steps=args.warmup_steps,
        **overrides,
    )

    exp_config = IntrinsicEvolutionExperimentConfig(
        num_steps=args.num_steps,
        snapshot_interval=args.snapshot_interval,
        install_default_initial_diversity=True,
        initial_conditions=initial_conditions,
        policy=policy,
        output_dir=run_dir,
        seed=seed,
    )

    record: Dict[str, Any] = {
        "profile": profile,
        "seed": seed,
        "run_dir": run_dir,
        "status": "pending",
        "elapsed_seconds": None,
        "num_steps_completed": None,
        "final_population": None,
        "error": None,
    }

    import farm.runners.intrinsic_evolution_experiment as runner_mod
    original_logger_cls = runner_mod.GeneTrajectoryLogger

    def _make_speciation_logger(*pos, **kw):
        return original_logger_cls(
            *pos,
            enable_speciation=True,
            speciation_algorithm="gmm",
            speciation_max_k=4,
            speciation_seed=seed,
            speciation_scaler="none",
            **kw,
        )

    runner_mod.GeneTrajectoryLogger = _make_speciation_logger  # type: ignore[assignment]
    try:
        t0 = time.time()
        result = IntrinsicEvolutionExperiment(base_config, exp_config).run()
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
        record.update(
            status="error",
            error=str(exc),
        )
        logger.error(
            "seed_sweep_run_failed",
            profile=profile,
            seed=seed,
            error=str(exc),
        )
        traceback.print_exc(file=sys.stderr)
        return False, record
    finally:
        runner_mod.GeneTrajectoryLogger = original_logger_cls


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

    total = len(args.profiles) * len(args.seeds)
    print(
        f"Stable-profile seed sweep: {len(args.profiles)} profile(s) × "
        f"{len(args.seeds)} seed(s) = {total} run(s)"
    )
    print(f"  profiles : {args.profiles}")
    print(f"  seeds    : {args.seeds}")
    print(f"  output   : {args.output_dir}")

    sweep_records: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    for profile in args.profiles:
        for seed in args.seeds:
            success, record = _run_one(profile, seed, args, logger)
            sweep_records.append(record)
            if success:
                n_ok += 1
            else:
                n_fail += 1
                if args.fail_fast:
                    print("Aborting sweep (--fail-fast).", file=sys.stderr)
                    break
        else:
            continue
        break

    manifest = {
        "sweep_type": "stable_profile_seed_sweep",
        "profiles": args.profiles,
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale,
        "selection_pressure": args.selection_pressure,
        "sub_profile_overrides": {p: STABLE_SUB_PROFILES[p] for p in args.profiles},
        "runs": sweep_records,
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    manifest_path = os.path.join(args.output_dir, "sweep_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nSweep complete: {n_ok}/{total} runs succeeded.")
    print(f"Manifest: {manifest_path}")
    if n_fail:
        print(f"  {n_fail} run(s) failed — see manifest for details.")
    print(
        f"\nNext step: python scripts/analyze_stable_profile_seed_sweep.py "
        f"--sweep-dir {args.output_dir}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
