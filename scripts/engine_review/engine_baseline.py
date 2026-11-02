#!/usr/bin/env python
"""Engine throughput and memory baseline harness.

Runs a fixed set of short simulations (population, perception radius, learning on/off, DB
backend, forced movement) and reports wall time, ms per agent-step, peak RSS and KD-tree
rebuilds per configuration. This is the harness that produced
``docs/research/reviews/2026-11-engine-review/scaling_results.jsonl``.

Each configuration runs in its own subprocess so that peak RSS is attributable and
``PYTHONHASHSEED=0`` is applied uniformly.

Usage (from the repository root):

    python scripts/engine_review/engine_baseline.py --suite smoke --out /tmp/baseline.json
    python scripts/engine_review/engine_baseline.py --suite review --out baseline.json
    python scripts/engine_review/engine_baseline.py --suite smoke --compare \\
        docs/research/reviews/2026-11-engine-review/baseline_952c6b14.json --threshold 0.10

``--compare`` exits non-zero when ms/agent-step or peak RSS regress by more than the threshold
on any configuration present in both files. Absolute numbers are hardware-specific: only
compare files produced on the same machine class.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from farm.config import SimulationConfig
from farm.core.agent.core import AgentCore
from farm.core.observations import ObservationConfig
from farm.core.services.implementations import EnvironmentValidationService
from farm.core.simulation import run_simulation
from farm.utils.logging import configure_logging

SUITES: dict[str, list[dict[str, Any]]] = {
    # Fast enough for CI (~1 minute on 4 vCPU at commit 952c6b14).
    "smoke": [
        {"label": "smoke_pop_15", "agents": 5, "max_population": 15, "steps": 100},
        {"label": "smoke_pop_15_nolearn", "agents": 5, "max_population": 15, "steps": 100, "learning": False},
        {"label": "smoke_pop_15_R3", "agents": 5, "max_population": 15, "steps": 100, "obs_R": 3},
    ],
    # The full set used in the 2026-11 review (~8 minutes on 4 vCPU).
    "review": [
        {"label": "pop_15", "agents": 5, "max_population": 15, "steps": 200},
        {"label": "pop_30", "agents": 10, "max_population": 30, "steps": 200},
        {"label": "pop_60", "agents": 20, "max_population": 60, "steps": 200},
        {"label": "pop_120", "agents": 40, "max_population": 120, "steps": 200},
        {"label": "pop_30_nolearn", "agents": 10, "max_population": 30, "steps": 200, "learning": False},
        {"label": "pop_120_nolearn", "agents": 40, "max_population": 120, "steps": 200, "learning": False},
        {"label": "R3", "agents": 10, "max_population": 30, "steps": 200, "obs_R": 3},
        {"label": "R10", "agents": 10, "max_population": 30, "steps": 200, "obs_R": 10},
        {"label": "R3_nolearn", "agents": 10, "max_population": 30, "steps": 200, "obs_R": 3, "learning": False},
        {"label": "R10_nolearn", "agents": 10, "max_population": 30, "steps": 200, "obs_R": 10, "learning": False},
        {"label": "pop_30_diskdb", "agents": 10, "max_population": 30, "steps": 200, "disk_db": True},
        {"label": "pop_30_fixmove_nolearn", "agents": 10, "max_population": 30, "steps": 200, "learning": False, "fix_move": True},
        {"label": "pop_120_fixmove_nolearn", "agents": 40, "max_population": 120, "steps": 200, "learning": False, "fix_move": True},
        {"label": "pop_300_fixmove_nolearn", "agents": 100, "max_population": 300, "steps": 100, "learning": False, "fix_move": True, "width": 100, "height": 100},
        {"label": "pop_300_nolearn", "agents": 100, "max_population": 300, "steps": 100, "learning": False, "width": 100, "height": 100},
    ],
}

COMPARED_METRICS = ("ms_per_agent_step", "peak_rss_mb")


# --------------------------------------------------------------------------- child process


def run_one(opts: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Run a single configuration in-process and return its metrics."""
    configure_logging(environment="development", log_dir=str(work_dir / "logs"), log_level="ERROR", disable_console=True)

    cfg = SimulationConfig.from_centralized_config(environment="development")
    n = int(opts.get("agents", 10))
    cfg.population.system_agents = n
    cfg.population.independent_agents = n
    cfg.population.control_agents = n
    cfg.population.max_population = int(opts.get("max_population", 3 * n))
    if "width" in opts:
        cfg.environment.width = int(opts["width"])
        cfg.environment.height = int(opts["height"])
    if "obs_R" in opts:
        cfg.observation = ObservationConfig(R=int(opts["obs_R"]), fov_radius=int(opts["obs_R"]))
    cfg.database.enable_validation = False
    if opts.get("disk_db", False):
        cfg.database.use_in_memory_db = False
        cfg.database.persist_db_on_completion = True
    else:
        cfg.database.use_in_memory_db = True
        cfg.database.persist_db_on_completion = False
    if not opts.get("learning", True):
        cfg.performance.max_learning_updates_per_step = -1
    cfg.seed = int(opts.get("seed", 42))

    step_times: list[float] = []
    populations: list[int] = []
    kd_rebuilds = {"n": 0}
    clock = {"last": 0.0}

    def ready(env) -> None:
        if opts.get("fix_move", False):
            _patch_movement(env)
        original_rebuild = env.spatial_index._rebuild_kdtrees

        def counting_rebuild(*args, **kwargs):
            kd_rebuilds["n"] += 1
            return original_rebuild(*args, **kwargs)

        env.spatial_index._rebuild_kdtrees = counting_rebuild
        clock["last"] = time.perf_counter()

    def step_end(env, step: int) -> None:
        now = time.perf_counter()
        step_times.append(now - clock["last"])
        clock["last"] = now
        populations.append(len(env.alive_agent_objects))

    started = time.perf_counter()
    env = run_simulation(
        num_steps=int(opts.get("steps", 200)),
        config=cfg,
        path=str(work_dir / "sims") if opts.get("disk_db") else None,
        save_config=False,
        seed=cfg.seed,
        disable_console_logging=True,
        on_environment_ready=ready,
        on_step_end=step_end,
    )
    wall = time.perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    agent_steps = max(sum(populations), 1)
    return {
        "label": opts.get("label", ""),
        "opts": opts,
        "wall_s": round(wall, 2),
        "steps": len(step_times),
        "ms_per_step_mean": round(1000 * sum(step_times) / max(len(step_times), 1), 1),
        "pop_start": populations[0] if populations else None,
        "pop_mean": round(sum(populations) / max(len(populations), 1), 1),
        "pop_end": populations[-1] if populations else None,
        "ms_per_agent_step": round(1000 * sum(step_times) / agent_steps, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "kdtree_rebuilds": kd_rebuilds["n"],
        "final_resources": round(env.cached_total_resources, 1),
    }


def _patch_movement(env) -> None:
    """Give agents the attributes ``move_action`` looks for so movement actually happens.

    Only used by the ``fix_move`` configurations to measure the spatial-index cost of a moving
    population while review finding M0-A is unresolved. Remove once movement works natively.
    """
    validation_service = EnvironmentValidationService(env)
    AgentCore.validation_service = property(lambda self: validation_service)

    def update_position(self, position):
        self.position = position

    AgentCore.update_position = update_position


# --------------------------------------------------------------------------- parent process


def run_suite(configs: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    results = []
    env = dict(os.environ, PYTHONHASHSEED="0")
    for cfg in configs:
        proc = subprocess.run(
            [sys.executable, __file__, "--child", json.dumps(cfg), "--work-dir", str(work_dir)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        result_lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
        if proc.returncode != 0 or not result_lines:
            sys.stderr.write(f"[{cfg.get('label')}] failed (rc={proc.returncode})\n{proc.stderr[-4000:]}\n")
            continue
        result = json.loads(result_lines[-1][len("RESULT ") :])
        results.append(result)
        print(
            f"{result['label']:<28} wall={result['wall_s']:>7.2f}s  "
            f"ms/agent-step={result['ms_per_agent_step']:>6.2f}  "
            f"peak_rss={result['peak_rss_mb']:>8.1f}MB  kd_rebuilds={result['kdtree_rebuilds']}",
            flush=True,
        )
    return results


def hardware_context() -> dict[str, Any]:
    context: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        # Concurrent load inflates every timing; record it so a bad comparison is explainable.
        "load_avg_1m_at_start": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "python_version": platform.python_version(),
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
    }
    return context


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def compare(current: dict[str, Any], baseline: dict[str, Any], threshold: float) -> int:
    base_by_label = {r["label"]: r for r in baseline["results"]}
    regressions = []
    print(f"\nComparing against baseline from {baseline['context'].get('git_revision', '?')[:12]} "
          f"({baseline['context'].get('platform', '?')})")
    for result in current["results"]:
        base = base_by_label.get(result["label"])
        if base is None:
            continue
        if base["opts"] != result["opts"]:
            print(f"{result['label']:<28} skipped: configuration differs from baseline")
            continue
        for metric in COMPARED_METRICS:
            before, after = base[metric], result[metric]
            change = (after - before) / before if before else 0.0
            flag = "REGRESSION" if change > threshold else ("improved" if change < -threshold else "ok")
            print(f"{result['label']:<28} {metric:<18} {before:>9.2f} -> {after:>9.2f}  ({change:+.1%}) {flag}")
            if change > threshold:
                regressions.append((result["label"], metric, change))
    if regressions:
        print(f"\n{len(regressions)} regression(s) above {threshold:.0%}")
        return 1
    print("\nno regressions above threshold")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--only", nargs="*", default=None, help="labels to run (default: all in suite)")
    parser.add_argument("--out", type=Path, default=None, help="write results JSON here")
    parser.add_argument("--compare", type=Path, default=None, help="baseline JSON to compare against")
    parser.add_argument("--threshold", type=float, default=0.10, help="relative regression threshold")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/engine_baseline"))
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.child is not None:
        os.environ.setdefault("PYTHONHASHSEED", "0")
        print("RESULT " + json.dumps(run_one(json.loads(args.child), args.work_dir)))
        return 0

    configs = SUITES[args.suite]
    if args.only:
        configs = [c for c in configs if c["label"] in set(args.only)]
    report = {"suite": args.suite, "context": hardware_context(), "results": run_suite(configs, args.work_dir)}

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        return compare(report, baseline, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
