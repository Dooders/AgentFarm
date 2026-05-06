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
import os
import pstats
import sys
import time
from io import StringIO
from typing import TYPE_CHECKING, Optional

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


def _disable_training_globally() -> None:
    """Force every Tianshou wrapper to skip ``train()`` for inference-only runs."""
    from farm.core.decision.algorithms.tianshou import TianshouWrapper

    def _no_train(self) -> bool:
        return False

    TianshouWrapper.should_train = _no_train


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
        percall = ct / nc if nc else 0.0
        print(
            f"{rank:>4}  {nc:>10}  {tt:>10.4f}  {fmt_pct(tt, total_tt):>6}  "
            f"{ct:>10.4f}  {percall:>10.6f}  {short}"
        )


def main() -> int:
    if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    from farm.core.simulation import run_simulation

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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.no_train:
        _disable_training_globally()

    config = build_config(args.width, args.height, args.agents, train=not args.no_train)
    config.seed = args.seed

    print("=" * 72)
    print("Stage 0 baseline profile")
    print("=" * 72)
    print(f"  grid:     {args.width}x{args.height}")
    print(f"  agents:   {args.agents} (sys/ind/ctrl ~= "
          f"{config.population.system_agents}/"
          f"{config.population.independent_agents}/"
          f"{config.population.control_agents})")
    print(f"  steps:    {args.steps} (warmup {args.warmup_steps})")
    print(f"  training: {'disabled' if args.no_train else 'enabled '}"
          f"(training_frequency={config.learning.training_frequency})")
    print(f"  seed:     {args.seed}")
    print(f"  in-mem db: {config.database.use_in_memory_db}")
    print()

    total_steps = args.warmup_steps + args.steps
    profiler = cProfile.Profile()
    profile_wall_start: Optional[float] = None
    run_start = time.time()

    def on_step_end(_env: object, step_idx: int) -> None:
        nonlocal profile_wall_start
        if step_idx == args.warmup_steps - 1:
            profile_wall_start = time.time()
            profiler.enable()

    if args.warmup_steps == 0:
        profile_wall_start = time.time()
        profiler.enable()

    env = run_simulation(
        num_steps=total_steps,
        config=config,
        path=None,
        save_config=False,
        disable_console_logging=True,
        on_step_end=on_step_end if args.warmup_steps > 0 else None,
    )
    profiler.disable()
    run_end = time.time()
    if args.warmup_steps > 0 and profile_wall_start is not None:
        print(f"  warmup wall: {profile_wall_start - run_start:.2f}s")
    wall = run_end - profile_wall_start if profile_wall_start is not None else 0.0

    profiler.dump_stats(args.out)

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

    print_top(stats, "cumulative", args.top, total_tt)
    print_top(stats, "tottime", args.top, total_tt)

    root, _ext = os.path.splitext(args.out)
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
    print(f"Saved binary profile: {args.out}")
    print(f"Saved text  profile: {text_out}")
    print(f"View interactively:  snakeviz {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
