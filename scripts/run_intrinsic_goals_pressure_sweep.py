#!/usr/bin/env python3
"""Selection-pressure sweep for the intrinsic-goals experiment.

Runs :class:`~farm.runners.intrinsic_goals_experiment.IntrinsicGoalsExperiment`
once per selection-pressure preset (``low``/``medium``/``high`` by default),
holding every other knob fixed, so the population-suppression effect from the
unique-vs-uniform goals comparison can be compared *across* pressure levels
(issue #892).

Each pressure level writes the usual per-arm artifacts plus
``intrinsic_goals_summary.json`` and ``intrinsic_goals_aggregate.png`` under::

    {output_dir}/intrinsic_goals_sweep_{pressure}/

A machine-readable manifest covering every completed pressure level is written
to ``{output_dir}/pressure_sweep_manifest.json``.

After the sweep, run ``scripts/analyze_intrinsic_goals_pressure_sweep.py
--sweep-dir {output_dir}`` to combine the three summaries into a single
cross-pressure comparison (table + figure).

Example::

    python scripts/run_intrinsic_goals_pressure_sweep.py \
        --pressures low medium high \
        --num-steps 600 --seed 42 --num-replicates 20 \
        --output-dir experiments

    # Print the planned matrix without running anything
    python scripts/run_intrinsic_goals_pressure_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    MutationMode,
)
from farm.runners.intrinsic_goals_experiment import (  # noqa: E402
    IntrinsicGoalsExperiment,
    IntrinsicGoalsExperimentConfig,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402

DEFAULT_PRESSURES: List[str] = ["low", "medium", "high"]

# Directory name for a given pressure level, e.g. "intrinsic_goals_sweep_low".
SWEEP_DIR_TEMPLATE = "intrinsic_goals_sweep_{pressure}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the intrinsic-goals experiment across selection-pressure "
            "presets (low/medium/high) with everything else held fixed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--pressures",
        nargs="+",
        default=DEFAULT_PRESSURES,
        metavar="PRESSURE",
        help="Selection-pressure presets to sweep (e.g. low medium high).",
    )
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-replicates", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help=(
            "Base directory; each pressure writes to "
            "{output_dir}/intrinsic_goals_sweep_{pressure}/."
        ),
    )
    parser.add_argument("--record-interval", type=int, default=1)

    parser.add_argument("--mutation-rate", type=float, default=0.1)
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
    parser.add_argument(
        "--initial-agent-resource-level",
        type=float,
        default=12.0,
        help="Starting resource level for each agent (stability knob).",
    )
    parser.add_argument(
        "--initial-resource-count",
        type=int,
        default=60,
        help="Number of resource nodes at simulation start (stability knob).",
    )
    parser.add_argument(
        "--max-population",
        type=int,
        default=None,
        help=(
            "Override the hard population cap (population.max_population). The "
            "development environment caps at 50, which pins both arms against "
            "the ceiling; raise it (e.g. 3000) to restore the boom/bust regime."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the entire sweep on the first pressure-level failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip a pressure level whose output directory already contains a "
            "completed intrinsic_goals_summary.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pressure levels and resolved config, then exit.",
    )
    return parser


def _run_dir(output_dir: str, pressure: str) -> str:
    return os.path.join(output_dir, SWEEP_DIR_TEMPLATE.format(pressure=pressure))


def _build_config(
    pressure: str, run_dir: str, args: argparse.Namespace
) -> IntrinsicGoalsExperimentConfig:
    return IntrinsicGoalsExperimentConfig(
        num_steps=args.num_steps,
        seed=args.seed,
        num_replicates=args.num_replicates,
        output_dir=run_dir,
        record_interval=args.record_interval,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode(args.mutation_mode),
        boundary_mode=BoundaryMode(args.boundary_mode),
        selection_pressure=pressure,
        initial_agent_resource_level=args.initial_agent_resource_level,
        initial_resource_count=args.initial_resource_count,
        max_population=args.max_population,
    )


def _summary_is_complete(run_dir: str) -> bool:
    """True when a readable intrinsic_goals_summary.json already exists."""
    summary_path = os.path.join(run_dir, "intrinsic_goals_summary.json")
    if not os.path.isfile(summary_path):
        return False
    try:
        with open(summary_path, encoding="utf-8") as handle:
            json.load(handle)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _run_one(
    pressure: str,
    base_config: SimulationConfig,
    args: argparse.Namespace,
    logger: Any,
) -> Tuple[bool, Dict[str, Any]]:
    run_dir = _run_dir(args.output_dir, pressure)
    record: Dict[str, Any] = {
        "pressure": pressure,
        "run_dir": run_dir,
        "status": "pending",
        "elapsed_seconds": None,
        "summary_path": None,
        "error": None,
    }

    if args.resume and _summary_is_complete(run_dir):
        record.update(
            status="skipped_done",
            elapsed_seconds=0.0,
            summary_path=os.path.join(run_dir, "intrinsic_goals_summary.json"),
        )
        logger.info("pressure_sweep_run_skipped_resume", pressure=pressure, run_dir=run_dir)
        return True, record

    os.makedirs(run_dir, exist_ok=True)
    logger.info("pressure_sweep_run_start", pressure=pressure, run_dir=run_dir)

    try:
        config = _build_config(pressure, run_dir, args)
        t0 = time.time()
        result = IntrinsicGoalsExperiment(base_config, config).run()
        elapsed = time.time() - t0
        record.update(
            status="ok",
            elapsed_seconds=round(elapsed, 3),
            summary_path=result.summary_path,
        )
        logger.info(
            "pressure_sweep_run_ok",
            pressure=pressure,
            elapsed=round(elapsed, 1),
            summary_path=result.summary_path,
        )
        return True, record
    except Exception as exc:  # noqa: BLE001 - record and continue unless --fail-fast
        record.update(status="error", error=str(exc))
        logger.error("pressure_sweep_run_failed", pressure=pressure, error=str(exc))
        traceback.print_exc(file=sys.stderr)
        return False, record


def _print_dry_run_plan(args: argparse.Namespace) -> None:
    print("Intrinsic-goals pressure sweep — DRY RUN")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  pressures     : {args.pressures}")
    print(f"  num_steps     : {args.num_steps}")
    print(f"  seed          : {args.seed}")
    print(f"  num_replicates: {args.num_replicates}")
    print(f"  mutation      : rate={args.mutation_rate} scale={args.mutation_scale}")
    print(f"  max_population: {args.max_population if args.max_population is not None else 'env default'}")
    print(f"  total runs    : {len(args.pressures)}")
    print()
    print("Per-pressure output directories:")
    for pressure in args.pressures:
        print(f"  {pressure:>6}: {_run_dir(args.output_dir, pressure)}")


def main() -> int:
    args = _build_parser().parse_args()

    if args.dry_run:
        _print_dry_run_plan(args)
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = get_logger(__name__)

    base_config = SimulationConfig.from_centralized_config(
        environment=args.environment,
        profile=args.profile,
    )

    total = len(args.pressures)
    print(f"Intrinsic-goals pressure sweep: {total} pressure level(s)")
    print(f"  pressures : {args.pressures}")
    print(f"  output    : {args.output_dir}")

    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for pressure in args.pressures:
        success, record = _run_one(pressure, base_config, args, logger)
        records.append(record)
        if success:
            n_ok += 1
            continue
        n_fail += 1
        if args.fail_fast:
            print("Aborting sweep (--fail-fast).", file=sys.stderr)
            break

    manifest = {
        "sweep_type": "intrinsic_goals_pressure_sweep",
        "pressures": args.pressures,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "num_replicates": args.num_replicates,
        "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale,
        "mutation_mode": args.mutation_mode,
        "boundary_mode": args.boundary_mode,
        "initial_agent_resource_level": args.initial_agent_resource_level,
        "initial_resource_count": args.initial_resource_count,
        "max_population": args.max_population,
        "runs": records,
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    manifest_path = os.path.join(args.output_dir, "pressure_sweep_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nSweep complete: {n_ok}/{total} pressure level(s) succeeded.")
    print(f"Manifest: {manifest_path}")
    if n_fail:
        print(f"  {n_fail} pressure level(s) failed — see manifest for details.")
    print(
        "\nNext step: python scripts/analyze_intrinsic_goals_pressure_sweep.py "
        f"--sweep-dir {args.output_dir}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
