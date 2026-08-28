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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread

from farm.runners.matrix_live_status import publish_live_status

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_SWEEP_SCRIPT = _REPO_ROOT / "scripts" / "run_stable_profile_seed_sweep.py"

# SIGKILL / SIGTERM — usually operator stop, Spot reset, or OOM — not a sim bug.
_RETRYABLE_RETURNCODES = frozenset({-9, -15})
_MAX_SIGNAL_RETRIES = 2


def _default_jobs() -> int:
    """Leave one core free so sshd/metadata stay responsive under load."""
    return max(1, (os.cpu_count() or 1) - 1)

# Thread-pool env vars pinned to 1 for every child: these are many tiny
# per-agent DQNs, so oversubscribed BLAS pools slow the whole matrix down.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Named population levels. Each maps to (environment, optional profile overlay,
# approximate max_population). Grid size and max_population come from the profile
# when set (see farm/config/profiles/). The max_population figure is only used
# to produce an upper-bound cost estimate in --dry-run.
POPULATION_LEVELS: dict[str, tuple[str, str | None, int]] = {
    "dev": ("development", None, 50),
    "sim": ("development", "simulation", 300),
    "research": ("development", "research", 500),
    "production": ("production", None, 1000),
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
    log_tail: str | None = None


def _log_tail(log_path: Path, *, max_bytes: int = 4096) -> str:
    """Last chunk of a seed log for failure diagnosis without SSH."""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            text = handle.read().decode("utf-8", errors="replace")
        return text.replace("\r", "\n").strip()[-1500:]
    except OSError:
        return ""


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
    environment, profile, _ = POPULATION_LEVELS[job.population]
    command: list[str] = [
        sys.executable,
        str(_SEED_SWEEP_SCRIPT),
        "--environment",
        environment,
    ]
    if profile is not None:
        command.extend(["--profile", profile])
    command.extend(
        [
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
    )
    if getattr(args, "checkpoint_interval", None) is not None:
        command.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
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
    _, _, approx_max_pop = POPULATION_LEVELS[job.population]
    total_steps = args.num_steps + args.warmup_steps
    seconds = args.sec_per_agent_step * approx_max_pop * total_steps
    return seconds / 3600.0


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in _THREAD_VARS:
        env.setdefault(var, "1")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _nice_worker() -> None:
    """Lower worker priority so sshd / guest-agent keep CPU under load."""
    try:
        os.nice(10)
    except OSError:
        pass


def _apply_nice_to_command(command: list[str]) -> list[str]:
    """Prefix *command* with ``nice -n 10`` on POSIX; return unchanged elsewhere.

    Using ``nice`` as an executable prefix avoids calling ``os.nice`` in a
    ``preexec_fn``, which is unsafe when the parent process runs threads
    (the child can deadlock between fork and exec).
    """
    if os.name == "posix":
        return ["nice", "-n", "10"] + command
    return command


def run_job(job: Job, args: argparse.Namespace) -> JobResult:
    """Execute one (cell, seed) job as a subprocess and capture its outcome."""
    cell_dir = Path(args.output_dir) / job.cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(job, args, cell_dir)

    log_path = cell_dir / f"seed_{job.seed}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            _apply_nice_to_command(command),
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
        log_tail=_log_tail(log_path) if status != "ok" else None,
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
        "--checkpoint-interval",
        type=int,
        default=50,
        help=(
            "Mid-run checkpoint cadence passed to each seed job "
            "(default: 50 so Spot/stop interruptions retain progress)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help=(
            "Number of concurrent subprocesses "
            "(default: CPU count minus 1, leaving headroom for SSH/metadata)."
        ),
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=60.0,
        help="Seconds between live-status heartbeats (file + GCE guest attributes).",
    )
    parser.add_argument(
        "--no-guest-attributes",
        action="store_true",
        help="Write matrix_live_status.json only; do not publish GCE guest attributes.",
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
    recent: list[dict] = []
    n_ok = 0
    n_fail = 0
    status_lock = Lock()
    emit_lock = Lock()
    stop_heartbeat = Event()
    use_guest_attrs = not args.no_guest_attributes

    def _emit_status(note: str = "") -> None:
        # Serialize snapshot-plus-publication with emit_lock so a slow heartbeat
        # cannot overwrite a newer job_complete or finished publication.
        with emit_lock:
            with status_lock:
                snapshot_ok = n_ok
                snapshot_fail = n_fail
                snapshot_recent = list(recent)
            try:
                publish_live_status(
                    output_dir,
                    total_jobs=len(jobs),
                    n_ok=snapshot_ok,
                    n_fail=snapshot_fail,
                    workers=workers,
                    recent=snapshot_recent,
                    note=note,
                    guest_attributes=use_guest_attrs,
                )
            except Exception as exc:  # noqa: BLE001 — status must never kill the matrix
                print(f"warning: live status publish failed: {exc}", file=sys.stderr)

    def _heartbeat_loop() -> None:
        interval = max(5.0, float(args.status_interval))
        while not stop_heartbeat.wait(interval):
            _emit_status(note="heartbeat")

    _emit_status(note="started")
    heartbeat = Thread(target=_heartbeat_loop, name="matrix-status-heartbeat", daemon=True)
    heartbeat.start()

    signal_attempts: dict[Job, int] = {job: 0 for job in jobs}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(run_job, job, args): job for job in jobs}
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    result = future.result()
                    rc = result.returncode
                    if (
                        rc in _RETRYABLE_RETURNCODES
                        and signal_attempts[job] < _MAX_SIGNAL_RETRIES
                    ):
                        signal_attempts[job] += 1
                        print(
                            f"  retry {signal_attempts[job]}/{_MAX_SIGNAL_RETRIES} "
                            f"after rc={rc} {result.elapsed_seconds}s "
                            f"{result.cell_dir} seed={result.job['seed']}",
                            file=sys.stderr,
                        )
                        pending[pool.submit(run_job, job, args)] = job
                        _emit_status(note="retry_after_signal")
                        continue

                    results.append(result)
                    with status_lock:
                        if result.status == "ok":
                            n_ok += 1
                        else:
                            n_fail += 1
                        entry = {
                            "status": result.status,
                            "returncode": result.returncode,
                            "elapsed_seconds": result.elapsed_seconds,
                            "population": result.job.get("population"),
                            "pressure": result.job.get("pressure"),
                            "gene_flow": result.job.get("gene_flow"),
                            "seed": result.job.get("seed"),
                            "cell_dir": result.cell_dir,
                        }
                        if result.log_tail:
                            entry["log_tail"] = result.log_tail[-400:]
                        if rc in _RETRYABLE_RETURNCODES:
                            entry["log_tail"] = (
                                f"killed by signal rc={rc} "
                                f"(operator stop / Spot reset / OOM); "
                                f"not a simulation error"
                            )
                        recent.append(entry)
                        done_count = n_ok + n_fail
                        result_status = result.status
                        result_rc = result.returncode
                        result_elapsed = result.elapsed_seconds
                        result_seed = result.job["seed"]
                        result_cell = result.cell_dir
                    print(
                        f"  [{done_count}/{len(jobs)}] {result_status:5s} "
                        f"rc={result_rc} {result_elapsed}s "
                        f"{result_cell} seed={result_seed}",
                        file=sys.stderr,
                    )
                    _emit_status(note="job_complete")
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)
        with status_lock:
            all_done = (n_ok + n_fail) >= len(jobs)
        _emit_status(note="finished" if all_done else "stopped")

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
