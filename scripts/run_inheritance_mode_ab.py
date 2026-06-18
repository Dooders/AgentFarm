#!/usr/bin/env python3
"""Inheritance-mode A/B orchestrator for intrinsic evolution (Issue #849/#903).

Runs matched seed sweeps across inheritance arms:

- ``baldwinian``: offspring start with a fresh policy (default/baseline).
- ``lamarckian``: offspring attempt policy warm-start from parent (P1).
- ``p2``: weights inherited with plasticity damping (lower child LR/ε).
- ``p3``: weights + optimizer state + bounded replay slice transferred.
- ``p4``: gated/blended transfer (fitness gate + θ blend).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.core.policy_inheritance import (  # noqa: E402
    P2_PLASTICITY_DAMPING,
    P3_REPLAY_BUFFER_LIMIT,
    P4_BLEND_ALPHA,
    P4_FITNESS_GATE_MIN_RESOURCES,
)
from scripts import analyze_stable_profile_seed_sweep as analyzer_mod  # noqa: E402
from scripts import run_stable_profile_seed_sweep as runner_mod  # noqa: E402
from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    DEFAULT_PROFILES,
    DEFAULT_SEEDS,
)

ARM_PRESETS: Dict[str, Dict[str, Any]] = {
    "baldwinian": {"inheritance_mode": "baldwinian"},
    "lamarckian": {"inheritance_mode": "lamarckian"},
    "p2": {"inheritance_mode": "p2"},
    "p3": {"inheritance_mode": "p3"},
    "p4": {"inheritance_mode": "p4"},
}
DEFAULT_ARMS: List[str] = ["baldwinian", "lamarckian"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate inheritance-mode matched A/B sweeps "
            "using the stable-profile intrinsic evolution runner. "
            "Supports baldwinian (baseline), lamarckian (P1), p2, p3, and p4 arms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=DEFAULT_ARMS,
        choices=list(ARM_PRESETS),
        metavar="ARM",
        help=(
            "Inheritance arms to run. "
            "Choices: baldwinian (baseline), lamarckian (P1), p2, p3, p4. "
            "Default: baldwinian lamarckian."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/inheritance_ab",
        help="Base directory for all arm artifacts.",
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        metavar="PROFILE",
        help="Stable sub-profiles to sweep (forwarded to the runner).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="SEED",
        help="Seeds to sweep (paired across arms).",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--selection-pressure", type=str, default="low")
    parser.add_argument(
        "--warmstart-plasticity-damping",
        type=float,
        default=P2_PLASTICITY_DAMPING,
        help="P2 damping factor in (0, 1] for child LR/epsilon.",
    )
    parser.add_argument(
        "--warmstart-replay-buffer-limit",
        type=int,
        default=P3_REPLAY_BUFFER_LIMIT,
        help="P3 cap (>= 1) on replay transitions transferred to the child.",
    )
    parser.add_argument(
        "--warmstart-blend-alpha",
        type=float,
        default=P4_BLEND_ALPHA,
        help="P4 blend coefficient in [0, 1] weighting the parent's policy.",
    )
    parser.add_argument(
        "--warmstart-fitness-gate-min-resources",
        type=float,
        default=P4_FITNESS_GATE_MIN_RESOURCES,
        help="P4 minimum parent resource level (>= 0) to clear the fitness gate.",
    )
    parser.add_argument("--initial-diversity-mutation-rate", type=float, default=1.0)
    parser.add_argument("--initial-diversity-mutation-scale", type=float, default=0.25)
    parser.add_argument(
        "--disk-database",
        action="store_true",
        help="Use disk-backed SQLite for each run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the rerun on the first run failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip per-run work that already has a completed metadata file.",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip per-arm analyze_stable_profile_seed_sweep.py invocation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned arm × profile × seed matrix and exit.",
    )
    return parser


def _build_runner_args(
    args: argparse.Namespace, arm: str, arm_output_dir: Path
) -> argparse.Namespace:
    preset = ARM_PRESETS[arm]
    return argparse.Namespace(
        environment=args.environment,
        profiles=list(args.profiles),
        seeds=list(args.seeds),
        output_dir=str(arm_output_dir),
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        snapshot_interval=args.snapshot_interval,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        selection_pressure=args.selection_pressure,
        inheritance_mode=preset["inheritance_mode"],
        warmstart_plasticity_damping=getattr(
            args, "warmstart_plasticity_damping", P2_PLASTICITY_DAMPING
        ),
        warmstart_replay_buffer_limit=getattr(
            args, "warmstart_replay_buffer_limit", P3_REPLAY_BUFFER_LIMIT
        ),
        warmstart_blend_alpha=getattr(
            args, "warmstart_blend_alpha", P4_BLEND_ALPHA
        ),
        warmstart_fitness_gate_min_resources=getattr(
            args,
            "warmstart_fitness_gate_min_resources",
            P4_FITNESS_GATE_MIN_RESOURCES,
        ),
        initial_diversity_mutation_rate=args.initial_diversity_mutation_rate,
        initial_diversity_mutation_scale=args.initial_diversity_mutation_scale,
        log_level=args.log_level,
        fail_fast=args.fail_fast,
        disk_database=args.disk_database,
        resume=args.resume,
        dry_run=False,
        crossover_enabled=False,
        crossover_mode="uniform",
        blend_alpha=0.5,
        num_crossover_points=2,
        coparent_strategy="nearest_alive_same_type",
        coparent_max_radius=None,
        allow_cross_type_pollination=False,
    )


def _run_arm(
    args: argparse.Namespace,
    arm: str,
    output_dir: Path,
    logger: Any,
) -> Dict[str, Any]:
    arm_output_dir = output_dir / arm
    arm_output_dir.mkdir(parents=True, exist_ok=True)
    runner_args = _build_runner_args(args, arm, arm_output_dir)

    print(
        f"\n=== Arm '{arm}' "
        f"(inheritance_mode={runner_args.inheritance_mode}) ==="
    )
    print(f"  output : {arm_output_dir}")
    print(f"  runs   : {len(runner_args.profiles) * len(runner_args.seeds)}")

    t0 = time.time()
    records, n_ok, n_fail = runner_mod._execute_sweep(
        runner_args, arm_output_dir, logger
    )
    elapsed = time.time() - t0

    arm_manifest = {
        "arm": arm,
        "output_dir": str(arm_output_dir),
        "inheritance": runner_mod._inheritance_settings_dict(runner_args),
        "profiles": runner_args.profiles,
        "seeds": runner_args.seeds,
        "num_steps": runner_args.num_steps,
        "warmup_steps": runner_args.warmup_steps,
        "snapshot_interval": runner_args.snapshot_interval,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_seconds": round(elapsed, 3),
        "runs": records,
    }
    arm_manifest_path = arm_output_dir / "sweep_manifest.json"
    with arm_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(arm_manifest, fh, indent=2, default=str)
    print(
        f"  done   : {n_ok}/{n_ok + n_fail} runs ok in {elapsed:.1f}s "
        f"(manifest: {arm_manifest_path})"
    )
    return arm_manifest


def _analyze_arm(arm_dir: Path) -> bool:
    print(f"\n--- Aggregating arm at {arm_dir} ---")
    argv = ["analyze_stable_profile_seed_sweep.py", "--sweep-dir", str(arm_dir)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = analyzer_mod.main()
    finally:
        sys.argv = old_argv
    return rc == 0


def _print_dry_run(args: argparse.Namespace, output_dir: Path) -> None:
    total = len(args.arms) * len(args.profiles) * len(args.seeds)
    print("Inheritance A/B rerun - DRY RUN")
    print(f"  output_dir : {output_dir}")
    print(f"  arms       : {args.arms}")
    print(f"  profiles   : {args.profiles}")
    print(f"  seeds      : {args.seeds}")
    print(f"  total runs : {total}")
    print()
    for arm in args.arms:
        preset = ARM_PRESETS[arm]
        print(f"  {arm}: {preset}")


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.dry_run:
        _print_dry_run(args, output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    runner_mod.configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = runner_mod.get_logger(__name__)

    overall: Dict[str, Any] = {
        "rerun_type": "inheritance_mode_ab",
        "output_dir": str(output_dir),
        "arms": args.arms,
        "profiles": args.profiles,
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "arm_manifests": {},
    }

    total_ok = 0
    total_fail = 0
    for arm in args.arms:
        arm_manifest = _run_arm(args, arm, output_dir, logger)
        overall["arm_manifests"][arm] = {
            "output_dir": arm_manifest["output_dir"],
            "inheritance": arm_manifest["inheritance"],
            "n_ok": arm_manifest["n_ok"],
            "n_fail": arm_manifest["n_fail"],
            "elapsed_seconds": arm_manifest["elapsed_seconds"],
        }
        total_ok += arm_manifest["n_ok"]
        total_fail += arm_manifest["n_fail"]
        if arm_manifest["n_fail"] and args.fail_fast:
            print("Aborting rerun (--fail-fast).", file=sys.stderr)
            break

    manifest_path = output_dir / "inheritance_ab_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2, default=str)

    print(f"\nInheritance A/B rerun complete: {total_ok} ok, {total_fail} failed.")
    print(f"Manifest: {manifest_path}")

    if not args.skip_analyze:
        for arm in args.arms:
            arm_dir = output_dir / arm
            if arm_dir.is_dir():
                ok = _analyze_arm(arm_dir)
                if not ok:
                    print(f"Per-arm aggregation failed for {arm_dir}", file=sys.stderr)

    baseline_arm = args.arms[0]
    treatment_arms = args.arms[1:]
    if treatment_arms:
        compare_lines = [
            "\nNext step: python scripts/compare_inheritance_arms.py \\",
            f"    --baseline-dir {output_dir / baseline_arm} \\",
            f"    --baseline-label {baseline_arm} \\",
        ]
        compare_lines.extend(
            f"    --treatment-dir {output_dir / arm} \\" for arm in treatment_arms
        )
        compare_lines.append(
            "    --arm-labels " + " ".join(treatment_arms) + " \\"
        )
        compare_lines.append(f"    --output-dir {output_dir / 'aggregate'}")
        print("\n".join(compare_lines))

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
