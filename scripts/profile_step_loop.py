#!/usr/bin/env python3
"""Stage 0 baseline profiler: 30x30 grid / 30 agents.

Runs ``run_simulation`` under cProfile for a small, RL-realistic scenario
that mirrors what feels slow in interactive use, and prints the top-N
functions by cumulative and total (self) time. Saves a binary
``.prof`` file consumable by snakeviz for interactive exploration.

Usage:
    python -m scripts.profile_step_loop --steps 100
    python -m scripts.profile_step_loop --steps 200 --no-train
    snakeviz simulations/profile_step_loop.prof
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from farm.config import SimulationConfig


def build_config(width: int, height: int, agents: int, train: bool) -> SimulationConfig:
    """Construct the small-grid scenario used for the baseline."""
    from farm.config import SimulationConfig

    config = SimulationConfig.from_centralized_config(environment="development")
    config.environment.width = width
    config.environment.height = height

    third = max(1, agents // 3)
    config.population.system_agents = third
    config.population.independent_agents = third
    config.population.control_agents = max(0, agents - 2 * third)
    config.population.max_population = max(agents * 2, 60)

    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = False
    config.database.enable_validation = False

    if not train:
        config.learning.training_frequency = 64

    return config


@contextmanager
def _patched_no_training(enabled: bool):
    """Temporarily patch Tianshou wrappers to skip training for this run only."""
    if not enabled:
        yield
        return

    from farm.core.decision.algorithms.tianshou import TianshouWrapper

    original_should_train = TianshouWrapper.should_train

    def _no_train(self) -> bool:
        return False

    TianshouWrapper.should_train = _no_train
    try:
        yield
    finally:
        TianshouWrapper.should_train = original_should_train


def _percentile(values: List[float], percentile: float) -> float:
    """Compute a percentile with linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    rank = (len(vals) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be a positive integer")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")
    if args.top <= 0:
        raise ValueError("--top must be a positive integer")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers")
    if args.agents <= 0:
        raise ValueError("--agents must be a positive integer")


def _resolve_run_output_path(base_out: str, run_idx: int, repeats: int) -> str:
    if repeats <= 1:
        return base_out
    root, ext = os.path.splitext(base_out)
    return f"{root}_run{run_idx + 1:02d}{ext}"


def _collect_runtime_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        import torch

        metadata["torch_version"] = torch.__version__
    except Exception:
        metadata["torch_version"] = None

    try:
        import tianshou

        metadata["tianshou_version"] = tianshou.__version__
    except Exception:
        metadata["tianshou_version"] = None

    try:
        metadata["git_sha"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        metadata["git_sha"] = None

    return metadata


def fmt_pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "  -  "
    return f"{(part / whole) * 100:5.1f}%"


def print_top(stats: pstats.Stats, sort_key: str, n: int, total_tt: float) -> None:
    print(f"\n--- Top {n} by {sort_key} ---")
    print(
        f"{'rank':>4}  {'ncalls':>10}  {'tottime':>10}  {'%tt':>6}  "
        f"{'cumtime':>10}  {'percall':>10}  function"
    )
    stats.sort_stats(sort_key)
    items = list(stats.stats.items())
    if sort_key == "cumulative":
        items.sort(key=lambda kv: kv[1][3], reverse=True)
    elif sort_key == "tottime":
        items.sort(key=lambda kv: kv[1][2], reverse=True)
    for rank, (func, (cc, nc, tt, ct, _callers)) in enumerate(items[:n], start=1):
        filename, lineno, name = func
        try:
            short = f"{os.path.relpath(filename)}:{lineno}({name})"
        except ValueError:
            short = f"{filename}:{lineno}({name})"
        percall = (tt / nc if sort_key == "tottime" else ct / nc) if nc else 0.0
        print(
            f"{rank:>4}  {nc:>10}  {tt:>10.4f}  {fmt_pct(tt, total_tt):>6}  "
            f"{ct:>10.4f}  {percall:>10.6f}  {short}"
        )


def _run_single_profile(
    *,
    args: argparse.Namespace,
    run_index: int,
    out_path: str,
    print_tables: bool,
) -> Dict[str, Any]:
    from farm.core.simulation import run_simulation

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    config = build_config(args.width, args.height, args.agents, train=not args.no_train)
    config.seed = args.seed

    print("=" * 72)
    print(f"Stage 0 baseline profile (run {run_index + 1}/{args.repeats})")
    print("=" * 72)
    print(f"  grid:     {args.width}x{args.height}")
    print(
        f"  agents:   {args.agents} (sys/ind/ctrl ~= "
        f"{config.population.system_agents}/"
        f"{config.population.independent_agents}/"
        f"{config.population.control_agents})"
    )
    print(f"  steps:    {args.steps} (warmup {args.warmup_steps})")
    print(
        f"  training: {'disabled' if args.no_train else 'enabled '} "
        f"(training_frequency={config.learning.training_frequency})"
    )
    print(f"  seed:     {args.seed}")
    print(f"  in-mem db: {config.database.use_in_memory_db}")
    print(f"  out:      {out_path}")
    print()

    total_steps = args.warmup_steps + args.steps
    profiler = cProfile.Profile()
    profile_wall_start: Optional[float] = None
    run_start = time.perf_counter()

    def on_step_end(_env: object, step_idx: int) -> None:
        nonlocal profile_wall_start
        if step_idx == args.warmup_steps - 1:
            profile_wall_start = time.perf_counter()
            profiler.enable()

    if args.warmup_steps == 0:
        profile_wall_start = time.perf_counter()
        profiler.enable()

    env = run_simulation(
        num_steps=total_steps,
        config=config,
        path=None,
        save_config=False,
        seed=args.seed,
        disable_console_logging=True,
        on_step_end=on_step_end if args.warmup_steps > 0 else None,
    )
    profiler.disable()
    run_end = time.perf_counter()
    if args.warmup_steps > 0 and profile_wall_start is not None:
        print(f"  warmup wall: {profile_wall_start - run_start:.2f}s")
    wall = run_end - profile_wall_start if profile_wall_start is not None else 0.0

    profiler.dump_stats(out_path)

    n_alive_end = sum(1 for a in getattr(env, "agent_objects", []) if getattr(a, "alive", False))
    steps_per_sec = args.steps / wall if wall > 0 else float("inf")
    print()
    print("=" * 72)
    print("Wall-clock summary")
    print("=" * 72)
    print(f"  wall:           {wall:.3f} s")
    print(f"  steps:          {args.steps}")
    print(f"  steps/sec:      {steps_per_sec:.2f}")
    print(f"  ms / step:      {wall * 1000.0 / max(1, args.steps):.2f}")
    print(f"  alive at end:   {n_alive_end}")

    stats = pstats.Stats(profiler)
    total_tt = sum(v[2] for v in stats.stats.values())

    if print_tables:
        print_top(stats, "cumulative", args.top, total_tt)
        print_top(stats, "tottime", args.top, total_tt)

    root, _ext = os.path.splitext(out_path)
    text_out = root + ".txt"
    with open(text_out, "w", encoding="utf-8") as fh:
        s = StringIO()
        st = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        st.print_stats(80)
        fh.write(s.getvalue())
        s2 = StringIO()
        st2 = pstats.Stats(profiler, stream=s2).sort_stats("tottime")
        st2.print_stats(80)
        fh.write("\n\n")
        fh.write(s2.getvalue())

    print()
    print(f"Saved binary profile: {out_path}")
    print(f"Saved text  profile: {text_out}")
    print(f"View interactively:  snakeviz {out_path}")
    return {
        "run_index": run_index + 1,
        "profile_path": out_path,
        "text_profile_path": text_out,
        "wall_seconds": wall,
        "steps_per_second": steps_per_sec,
        "ms_per_step": wall * 1000.0 / max(1, args.steps),
        "alive_at_end": n_alive_end,
    }


def main() -> int:
    if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234567890)
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Disable RL training updates (isolates inference cost from training)",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--out",
        type=str,
        default="simulations/profile_step_loop.prof",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Steps to run before enabling the profiler (same run; setup not in profile)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs to capture timing variance (profiles each run).",
    )
    args = parser.parse_args()
    _validate_args(args)

    runtime_metadata = _collect_runtime_metadata()
    print("=" * 72)
    print("Runtime metadata")
    print("=" * 72)
    for key, value in runtime_metadata.items():
        print(f"  {key}: {value}")
    print()

    run_results: List[Dict[str, Any]] = []
    with _patched_no_training(args.no_train):
        for run_idx in range(args.repeats):
            run_out = _resolve_run_output_path(args.out, run_idx, args.repeats)
            run_results.append(
                _run_single_profile(
                    args=args,
                    run_index=run_idx,
                    out_path=run_out,
                    print_tables=(run_idx == args.repeats - 1),
                )
            )

    walls = [r["wall_seconds"] for r in run_results]
    sps = [r["steps_per_second"] for r in run_results]
    summary = {
        "repeats": args.repeats,
        "wall_seconds": {
            "min": min(walls),
            "max": max(walls),
            "mean": statistics.fmean(walls),
            "median": statistics.median(walls),
            "p95": _percentile(walls, 0.95),
            "stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        },
        "steps_per_second": {
            "min": min(sps),
            "max": max(sps),
            "mean": statistics.fmean(sps),
            "median": statistics.median(sps),
            "p95": _percentile(sps, 0.95),
            "stdev": statistics.stdev(sps) if len(sps) > 1 else 0.0,
        },
        "runtime_metadata": runtime_metadata,
        "runs": run_results,
    }

    print()
    print("=" * 72)
    print("Repeat summary")
    print("=" * 72)
    print(f"  repeats:        {summary['repeats']}")
    print(f"  wall median:    {summary['wall_seconds']['median']:.3f} s")
    print(f"  wall p95:       {summary['wall_seconds']['p95']:.3f} s")
    print(f"  wall mean:      {summary['wall_seconds']['mean']:.3f} s")
    print(f"  wall stdev:     {summary['wall_seconds']['stdev']:.3f} s")
    print(f"  steps/sec med:  {summary['steps_per_second']['median']:.2f}")
    print(f"  steps/sec p95:  {summary['steps_per_second']['p95']:.2f}")

    root, _ext = os.path.splitext(args.out)
    summary_out = root + "_summary.json"
    with open(summary_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved repeat summary: {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
