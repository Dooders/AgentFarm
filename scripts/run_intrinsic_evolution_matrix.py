#!/usr/bin/env python3
"""Fan out the intrinsic-evolution research matrix across CPU cores.

This orchestrator implements the confirmatory factorial described in
``docs/research/experiments/intrinsic_evolution/research_plan_population_matrix.md``:
a replicated sweep over three axes — selection pressure, gene flow, and
population size (effective population, ``Ne``) — with many seeds per cell.

It does **not** re-implement any simulation logic.  Each (cell, seed) job is a
subprocess call to :mod:`scripts.run_stable_profile_seed_sweep`, which builds and
runs a single :class:`~farm.runners.intrinsic_evolution_experiment.IntrinsicEvolutionExperiment`.
Running one seed per subprocess keeps every job independent and single-threaded,
which is what makes ``--jobs`` (one process per vCPU) and ``--resume`` behave
predictably on a Spot VM.

Axes
----
- ``--pressures``: density-dependent reproduction-cost presets. ``none`` is the
  drift-only null used to test whether selection (not drift) shapes outcomes.
- ``--gene-flow``: ``mutation`` (inherit parent chromosome, mutated) vs
  ``crossover`` (pick a co-parent and cross before mutating).
- ``--populations``: a named ``(environment, approx max_population)`` pair. The
  population axis is driven by the environment profile because population is
  bounded by grid size and ``max_population``, not by resource supply — see the
  plan doc and ``farm/config/environments/*.yaml``.

Layout
------
Per-cell artifacts land under::

    {output_dir}/pop-{pop}__pressure-{pressure}__geneflow-{geneflow}/
        stable_balanced/seed_{seed}/...

A top-level ``matrix_manifest.json`` records every job's resolved command,
status, and wall time.

Examples
--------
Print the planned matrix and an (upper-bound) cost estimate without running::

    python scripts/run_intrinsic_evolution_matrix.py --dry-run

Fast smoke check (tiny run, not scientific evidence)::

    python scripts/run_intrinsic_evolution_matrix.py \\
        --populations dev --pressures low --gene-flow mutation \\
        --seeds 42 --num-steps 5 --warmup-steps 0 --snapshot-interval 5 \\
        --jobs 1 --output-dir /tmp/intrinsic_matrix_smoke

Full Phase-2 confirmatory matrix (12 cells x 8 seeds = 96 runs)::

    python scripts/run_intrinsic_evolution_matrix.py \\
        --populations sim research \\
        --pressures none low high \\
        --gene-flow mutation crossover \\
        --seeds 42 7 19 101 137 256 512 999 \\
        --num-steps 10000 --warmup-steps 200 --snapshot-interval 200 \\
        --output-dir experiments/intrinsic_matrix

Aggregate each cell afterwards with::

    python scripts/analyze_stable_profile_seed_sweep.py --sweep-dir <cell_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_SWEEP_SCRIPT = _REPO_ROOT / "scripts" / "run_stable_profile_seed_sweep.py"

# Thread-pool env vars pinned to 1 for every child: these are many tiny
# per-agent DQNs, so oversubscribed BLAS pools slow the whole matrix down.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Named population levels. Each maps to a (environment profile, approximate
# max_population) pair. The max_population figure is only used to produce an
# upper-bound cost estimate in --dry-run; the true standing population is set
# by the simulation and is typically well below the cap.
POPULATION_LEVELS: dict[str, tuple[str, int]] = {
    "dev": ("development", 50),
    "sim": ("simulation", 300),
    "research": ("research", 500),
    "production": ("production", 1000),
}

PRESSURES: tuple[str, ...] = ("none", "low", "high")
GENE_FLOWS: tuple[str, ...] = ("mutation", "crossover")

DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 19, 101, 137, 256, 512, 999)


@dataclass(frozen=True)
class Job:
    """A single (cell, seed) unit of work."""

    population: str
    pressure: str
    gene_flow: str
    seed: int

    @property
    def cell_name(self) -> str:
        return f"pop-{self.population}__pressure-{self.pressure}__geneflow-{self.gene_flow}"


@dataclass
class JobResult:
    """Outcome of running a single :class:`Job`."""

    job: dict[str, object]
    cell_dir: str
    command: list[str]
    status: str
    returncode: int | None
    elapsed_seconds: float | None


def build_matrix(
    populations: list[str],
    pressures: list[str],
    gene_flows: list[str],
    seeds: list[int],
) -> list[Job]:
    """Cartesian product of the four axes, in a stable, readable order."""
    jobs: list[Job] = []
    for population in populations:
        for pressure in pressures:
            for gene_flow in gene_flows:
                for seed in seeds:
                    jobs.append(Job(population, pressure, gene_flow, seed))
    return jobs


def build_command(job: Job, args: argparse.Namespace, cell_dir: Path) -> list[str]:
    """Resolve the seed-sweep subprocess command for a single job."""
    environment, _ = POPULATION_LEVELS[job.population]
    command: list[str] = [
        sys.executable,
        str(_SEED_SWEEP_SCRIPT),
        "--environment",
        environment,
        "--profiles",
        "balanced",
        "--seeds",
        str(job.seed),
        "--selection-pressure",
        job.pressure,
        "--num-steps",
        str(args.num_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--snapshot-interval",
        str(args.snapshot_interval),
        "--output-dir",
        str(cell_dir),
        "--log-level",
        args.log_level,
    ]
    if job.gene_flow == "crossover":
        command.append("--crossover-enabled")
    if args.disk_database:
        command.append("--disk-database")
    if args.resume:
        command.append("--resume")
    return command


def estimate_core_hours(job: Job, args: argparse.Namespace) -> float:
    """Upper-bound single-run cost estimate in core-hours.

    Uses ``sec_per_agent_step`` x (max_population) x (num_steps + warmup). The
    max_population cap is an over-estimate of the true standing population, so
    this is deliberately conservative. Re-measure with a Phase-1 pilot before
    trusting the total for scheduling.
    """
    _, approx_max_pop = POPULATION_LEVELS[job.population]
    total_steps = args.num_steps + args.warmup_steps
    seconds = args.sec_per_agent_step * approx_max_pop * total_steps
    return seconds / 3600.0


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in _THREAD_VARS:
        env.setdefault(var, "1")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def run_job(job: Job, args: argparse.Namespace) -> JobResult:
    """Execute one (cell, seed) job as a subprocess and capture its outcome."""
    cell_dir = Path(args.output_dir) / job.cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(job, args, cell_dir)

    log_path = cell_dir / f"seed_{job.seed}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            env=_child_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - t0
    status = "ok" if completed.returncode == 0 else "error"
    return JobResult(
        job=asdict(job),
        cell_dir=str(cell_dir),
        command=command,
        status=status,
        returncode=completed.returncode,
        elapsed_seconds=round(elapsed, 3),
    )


def _print_dry_run(jobs: list[Job], args: argparse.Namespace) -> None:
    cells = sorted({job.cell_name for job in jobs})
    total_core_hours = sum(estimate_core_hours(job, args) for job in jobs)
    print("Intrinsic-evolution matrix — DRY RUN")
    print(f"  output_dir        : {args.output_dir}")
    print(f"  populations       : {args.populations}")
    print(f"  pressures         : {args.pressures}")
    print(f"  gene_flow         : {args.gene_flow}")
    print(f"  seeds             : {args.seeds}")
    print(f"  steps/warmup/snap : {args.num_steps} / {args.warmup_steps} / {args.snapshot_interval}")
    print(f"  jobs (workers)    : {args.jobs}")
    print(f"  disk_database     : {args.disk_database}")
    print(f"  cells             : {len(cells)}")
    print(f"  total runs        : {len(jobs)}")
    print(
        f"  est. core-hours   : ~{total_core_hours:.0f} "
        f"(upper bound @ {args.sec_per_agent_step:.3f} s/agent/step, cap-based)"
    )
    print()
    print("Cells:")
    for cell in cells:
        print(f"  {cell}")
    print()
    print("Example command (first job):")
    first = jobs[0]
    example = build_command(first, args, Path(args.output_dir) / first.cell_name)
    print("  " + " ".join(example))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fan out the intrinsic-evolution research matrix (pressure x gene-flow "
            "x population x seed) across CPU cores, one seed per subprocess."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--populations",
        nargs="+",
        default=["sim", "research"],
        choices=list(POPULATION_LEVELS),
        metavar="POP",
        help="Named population levels (environment profiles).",
    )
    parser.add_argument(
        "--pressures",
        nargs="+",
        default=list(PRESSURES),
        choices=list(PRESSURES),
        metavar="PRESSURE",
        help="Selection-pressure presets. 'none' is the drift-only null.",
    )
    parser.add_argument(
        "--gene-flow",
        nargs="+",
        default=list(GENE_FLOWS),
        choices=list(GENE_FLOWS),
        metavar="MODE",
        help="Inheritance modes: mutation-only and/or crossover.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        metavar="SEED",
        help="Replicate seeds (>=5 recommended for the evidence gate).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/intrinsic_matrix",
        help="Base directory for all cell/seed artifacts.",
    )
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=200)
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of concurrent subprocesses (default: CPU count).",
    )
    parser.add_argument(
        "--disk-database",
        action="store_true",
        default=True,
        help="Use disk-backed SQLite in each run (recommended for long horizons).",
    )
    parser.add_argument(
        "--memory-database",
        dest="disk_database",
        action="store_false",
        help="Opt into :memory: DB (development default) instead of disk DB.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (cell, seed) runs already completed to --num-steps.",
    )
    parser.add_argument(
        "--sec-per-agent-step",
        type=float,
        default=0.02,
        help="Planning constant for --dry-run cost estimate (s per alive agent per step).",
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
        help="Print the planned matrix and cost estimate, then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not _SEED_SWEEP_SCRIPT.is_file():
        print(f"error: seed-sweep runner not found at {_SEED_SWEEP_SCRIPT}", file=sys.stderr)
        return 2

    jobs = build_matrix(args.populations, args.pressures, args.gene_flow, args.seeds)
    if not jobs:
        print("error: empty matrix (check --populations/--pressures/--gene-flow/--seeds)", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_dry_run(jobs, args)
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.jobs))
    print(
        f"Launching {len(jobs)} runs across {workers} worker(s) "
        f"into {output_dir} ...",
        file=sys.stderr,
    )

    results: list[JobResult] = []
    n_ok = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {pool.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            if result.status == "ok":
                n_ok += 1
            else:
                n_fail += 1
            print(
                f"  [{n_ok + n_fail}/{len(jobs)}] {result.status:5s} "
                f"rc={result.returncode} {result.elapsed_seconds}s "
                f"{result.cell_dir} seed={result.job['seed']}",
                file=sys.stderr,
            )

    manifest = {
        "output_dir": str(output_dir),
        "populations": args.populations,
        "pressures": args.pressures,
        "gene_flow": args.gene_flow,
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "disk_database": args.disk_database,
        "workers": workers,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "results": [asdict(result) for result in results],
    }
    manifest_path = output_dir / "matrix_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nDone: {n_ok} ok, {n_fail} failed. Manifest: {manifest_path}", file=sys.stderr)
    if n_ok:
        cells = sorted({result.cell_dir for result in results if result.status == "ok"})
        print("\nAggregate each cell with:", file=sys.stderr)
        for cell in cells:
            print(
                f"  python scripts/analyze_stable_profile_seed_sweep.py --sweep-dir {cell}",
                file=sys.stderr,
            )

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
