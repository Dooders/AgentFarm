#!/usr/bin/env python3
"""Long-horizon intrinsic-evolution runs for the balanced stable resource profile.

Implements experiment #1 from the transition-regime wiki: same six seeds as the
published stable-profile sweep, but many more logged steps after warmup, to see
whether balanced-profile outcome dispersion narrows (slow convergence) or
persists (bimodal regime).

Wiki: https://github.com/Dooders/AgentFarm/wiki/Balanced-Profile-as-a-Transition-Regime

Outputs mirror ``run_stable_profile_seed_sweep.py``::

    {output_dir}/stable_balanced/seed_{seed}/
    {output_dir}/sweep_manifest.json

Analyze with::

    python scripts/analyze_balanced_long_horizon.py --sweep-dir {output_dir}

Example
-------
::

    python scripts/run_balanced_long_horizon_experiment.py \\
        --output-dir experiments/balanced_long_horizon

    python scripts/run_balanced_long_horizon_experiment.py \\
        --num-steps 3000 --resume \\
        --output-dir experiments/balanced_long_horizon_3k

    # Opt into :memory: DB (development default); long horizon uses disk DB by default.
    python scripts/run_balanced_long_horizon_experiment.py --memory-db ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.runners.intrinsic_evolution_experiment import STABLE_SUB_PROFILES  # noqa: E402
from farm.utils.logging import configure_logging, get_logger  # noqa: E402

from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    DEFAULT_SEEDS,
    _execute_sweep,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Long-horizon seed sweep for the balanced stable profile only "
            "(IntrinsicEvolutionExperiment). Writes stable_balanced/seed_*/ "
            "under the output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        metavar="SEED",
        help="Seeds to sweep (same list as the published 6-seed sweep).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/balanced_long_horizon",
        help="Base directory for all run artifacts.",
    )
    parser.add_argument("--num-steps", type=int, default=5000)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip a seed when intrinsic_evolution_metadata.json exists and "
            "num_steps_completed >= configured num_steps."
        ),
    )
    parser.add_argument(
        "--memory-db",
        action="store_true",
        help=(
            "Use the environment's default DB mode (:memory: under development). "
            "Default for this script is disk-backed SQLite to reduce RAM pressure."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned seed matrix and resolved overrides, then exit.",
    )
    return parser


def _print_dry_run_plan(args: argparse.Namespace, output_dir: Path) -> None:
    profile = "balanced"
    print("Balanced long-horizon experiment — DRY RUN")
    print(f"  output_dir : {output_dir}")
    print(f"  profile    : {profile} (fixed)")
    print(f"  seeds      : {args.seeds}")
    print(
        f"  num_steps  : {args.num_steps} (warmup {args.warmup_steps}, "
        f"snapshot every {args.snapshot_interval})"
    )
    print(f"  resume     : {args.resume}")
    print(f"  disk_db    : {not args.memory_db}")
    print(f"  total runs : {len(args.seeds)}")
    print()
    print("Resolved sub-profile overrides:")
    print(f"  stable_{profile}: {STABLE_SUB_PROFILES[profile]}")


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    args.profiles = ["balanced"]
    args.disk_database = not args.memory_db

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

    total = len(args.seeds)
    print(
        f"Balanced long-horizon: profile=balanced × {len(args.seeds)} seed(s) = {total} run(s)"
    )
    print(f"  seeds       : {args.seeds}")
    print(f"  num_steps   : {args.num_steps}")
    print(f"  resume      : {args.resume}")
    print(f"  disk_db     : {args.disk_database}")
    print(f"  output      : {output_dir}")

    sweep_records, n_ok, n_fail = _execute_sweep(args, output_dir, logger)

    manifest = {
        "sweep_type": "balanced_long_horizon",
        "profiles": args.profiles,
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale,
        "selection_pressure": args.selection_pressure,
        "resume": args.resume,
        "disk_database": args.disk_database,
        "sub_profile_overrides": {p: STABLE_SUB_PROFILES[p] for p in args.profiles},
        "runs": sweep_records,
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    manifest_path = output_dir / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nSweep complete: {n_ok}/{total} runs succeeded (ok or skipped).")
    print(f"Manifest: {manifest_path}")
    if n_fail:
        print(f"  {n_fail} run(s) failed — see manifest for details.")
    print(
        f"\nNext step: python scripts/analyze_balanced_long_horizon.py "
        f"--sweep-dir {output_dir}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
