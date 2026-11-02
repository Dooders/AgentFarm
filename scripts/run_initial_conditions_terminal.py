#!/usr/bin/env python3
"""Sweep founder and offspring endowments on the intrinsic-cooperation baseline.

The food budget stays pinned (60 resource nodes, medium selection pressure,
population cap 300, pilot learning defaults). Only two startup knobs move:

- ``initial_agent_resource_level`` (founders): 5, 8, 12, 20
- ``offspring_initial_resources``: 5, 8

Eight cells, six seeds (42, 7, 19, 101, 137, 256), baseline arm only.
Each (cell, seed) is one directory under the output root. The child process
is ``scripts/run_intrinsic_cooperation_experiment.py --arms baseline``.

::

    python scripts/run_initial_conditions_terminal.py \\
        --output-dir experiments/initial_conditions_terminal

    python scripts/run_initial_conditions_terminal.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

FOUNDER_LEVELS: tuple[int, ...] = (5, 8, 12, 20)
OFFSPRING_LEVELS: tuple[int, ...] = (5, 8)
DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 19, 101, 137, 256)

INITIAL_RESOURCE_COUNT = 60
MAX_POPULATION = 300
SELECTION_PRESSURE = "medium"
POLICY_TEMPERATURE = 0.1
DEFAULT_NUM_STEPS = 3000
ARM = "baseline"

CELL_PLAN_FILE = "cell.json"
ARM_RESULT_RELATIVE = Path(ARM) / "arm_result.json"
SWEEP_MANIFEST = "sweep_manifest.json"


@dataclass(frozen=True)
class Cell:
    """One registered (founder, offspring, seed) run."""

    founder: int
    offspring: int
    seed: int
    output_dir: Path
    num_steps: int
    environment: str

    @property
    def run_dir(self) -> Path:
        return (
            self.output_dir
            / f"founder_{self.founder}"
            / f"offspring_{self.offspring}"
            / f"seed_{self.seed}"
        )

    def plan(self) -> dict[str, Any]:
        """Factor levels and the pinned budget. Compared on resume."""
        return {
            "arms": [ARM],
            "environment": self.environment,
            "founder_initial_resources": self.founder,
            "freeze_platform_genes": True,
            "initial_resource_count": INITIAL_RESOURCE_COUNT,
            "max_population": MAX_POPULATION,
            "num_steps": self.num_steps,
            "offspring_initial_resources": self.offspring,
            "policy_temperature": POLICY_TEMPERATURE,
            "seed": self.seed,
            "selection_pressure": SELECTION_PRESSURE,
        }


def iter_cells(
    output_dir: Path,
    founders: Sequence[int] = FOUNDER_LEVELS,
    offsprings: Sequence[int] = OFFSPRING_LEVELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_steps: int = DEFAULT_NUM_STEPS,
    environment: str = "development",
) -> list[Cell]:
    """The registered grid, founders then offspring then seeds."""
    return [
        Cell(
            founder=int(founder),
            offspring=int(offspring),
            seed=int(seed),
            output_dir=output_dir,
            num_steps=int(num_steps),
            environment=environment,
        )
        for founder in founders
        for offspring in offsprings
        for seed in seeds
    ]


def build_cell_command(cell: Cell, script: Path, python: str | None = None) -> list[str]:
    """Argv for one baseline run. The food budget is not a parameter."""
    return [
        python or sys.executable,
        str(script),
        "--arms",
        ARM,
        "--environment",
        cell.environment,
        "--num-steps",
        str(cell.num_steps),
        "--seed",
        str(cell.seed),
        "--num-replicates",
        "1",
        "--jobs",
        "1",
        "--initial-agent-resource-level",
        str(cell.founder),
        "--offspring-initial-resources",
        str(cell.offspring),
        "--initial-resource-count",
        str(INITIAL_RESOURCE_COUNT),
        "--max-population",
        str(MAX_POPULATION),
        "--selection-pressure",
        SELECTION_PRESSURE,
        "--policy-temperature",
        str(POLICY_TEMPERATURE),
        "--output-dir",
        str(cell.run_dir),
    ]


def cell_is_complete(cell: Cell) -> bool:
    """True when this directory already holds a finished run of the same plan."""
    plan_path = cell.run_dir / CELL_PLAN_FILE
    arm_path = cell.run_dir / ARM_RESULT_RELATIVE
    if not plan_path.is_file() or not arm_path.is_file():
        return False
    try:
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return stored == cell.plan()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def execute_cell(cell: Cell, script: Path) -> dict[str, Any]:
    """Run one cell, or skip it when the same plan already finished.

    An incomplete directory is removed first. The cooperation runner only
    resumes from ``arm_result.json``, so a leftover database would be a second
    simulation in the same folder.
    """
    started = time.time()
    label = f"founder={cell.founder} offspring={cell.offspring} seed={cell.seed}"
    if cell_is_complete(cell):
        print(f"skip {label}", flush=True)
        return {**cell.plan(), "status": "skipped", "elapsed_seconds": 0.0, "returncode": 0}

    if cell.run_dir.exists():
        shutil.rmtree(cell.run_dir)
    cell.run_dir.mkdir(parents=True)
    _write_json(cell.run_dir / CELL_PLAN_FILE, cell.plan())

    command = build_cell_command(cell, script)
    log_path = cell.run_dir / "driver.log"
    print(f"start {label}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=script.parent.parent,
            env=_child_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = round(time.time() - started, 1)
    arm_path = cell.run_dir / ARM_RESULT_RELATIVE
    status = "ok" if completed.returncode == 0 and arm_path.is_file() else "failed"
    print(f"done {label} status={status} elapsed_s={elapsed}", flush=True)
    return {
        **cell.plan(),
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "run_dir": str(cell.run_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/initial_conditions_terminal"))
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--founder-levels", type=int, nargs="+", default=list(FOUNDER_LEVELS))
    parser.add_argument("--offspring-levels", type=int, nargs="+", default=list(OFFSPRING_LEVELS))
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent cells. Each cell is one baseline simulation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the child commands and exit.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_steps <= 0:
        raise SystemExit("--num-steps must be positive.")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1.")
    if any(level < 0 for level in args.founder_levels):
        raise SystemExit("--founder-levels must be non-negative.")
    if any(level < 0 for level in args.offspring_levels):
        raise SystemExit("--offspring-levels must be non-negative.")
    if not args.seeds or not args.founder_levels or not args.offspring_levels:
        raise SystemExit("The sweep needs at least one founder level, offspring level, and seed.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    args.output_dir = args.output_dir.resolve()
    script = Path(__file__).resolve().parent / "run_intrinsic_cooperation_experiment.py"
    cells = iter_cells(
        output_dir=args.output_dir,
        founders=args.founder_levels,
        offsprings=args.offspring_levels,
        seeds=args.seeds,
        num_steps=args.num_steps,
        environment=args.environment,
    )
    if args.dry_run:
        for cell in cells:
            print(" ".join(build_cell_command(cell, script)))
        print(f"{len(cells)} cells", file=sys.stderr)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if args.jobs == 1:
        for cell in cells:
            records.append(execute_cell(cell, script))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(execute_cell, cell, script) for cell in cells]
            for future in as_completed(futures):
                records.append(future.result())

    failures = [record for record in records if record["status"] == "failed"]
    _write_json(
        args.output_dir / SWEEP_MANIFEST,
        {
            "cells_planned": len(cells),
            "cells_recorded": len(records),
            "failures": len(failures),
            "pinned": {
                "initial_resource_count": INITIAL_RESOURCE_COUNT,
                "max_population": MAX_POPULATION,
                "selection_pressure": SELECTION_PRESSURE,
                "policy_temperature": POLICY_TEMPERATURE,
                "arms": [ARM],
            },
            "records": sorted(
                records,
                key=lambda row: (
                    row["founder_initial_resources"],
                    row["offspring_initial_resources"],
                    row["seed"],
                ),
            ),
        },
    )
    if failures:
        print(f"{len(failures)} cell(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
