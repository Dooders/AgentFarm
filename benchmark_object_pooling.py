import argparse
import gc
import os
import time
from contextlib import contextmanager

import psutil

from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation


@contextmanager
def measure_memory():
    process = psutil.Process(os.getpid())
    gc.collect()
    before = process.memory_info().rss
    t0 = time.perf_counter()
    yield lambda: (
        time.perf_counter() - t0,
        process.memory_info().rss - before,
        process.memory_info().rss,
    )


def run_once(steps: int, agents: int, use_pooling: bool, seed: int | None) -> dict:
    # Configure initial agent counts
    cfg = SimulationConfig(
        width=200,
        height=200,
        system_agents=agents,
        independent_agents=0,
        control_agents=0,
        use_in_memory_db=True,
        persist_db_on_completion=False,
        max_steps=steps,
        simulation_steps=steps,
    )

    # Toggle pooling via env var (benchmark switch)
    os.environ["FARM_DISABLE_POOLING"] = "0" if use_pooling else "1"

    with measure_memory() as finish:
        env = run_simulation(num_steps=steps, config=cfg, save_config=False, path=None, seed=seed)
        # Ensure cleanup ASAP to capture memory release
        env.cleanup()
    elapsed, mem_delta, peak = finish()

    return {
        "use_pooling": use_pooling,
        "elapsed_s": elapsed,
        "mem_delta_bytes": mem_delta,
        "peak_rss_bytes": peak,
        "agents": agents,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark object pooling impact")
    parser.add_argument("--agents", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Warm-up
    run_once(steps=5, agents=50, use_pooling=True, seed=args.seed)
    run_once(steps=5, agents=50, use_pooling=False, seed=args.seed)

    with_pool = run_once(args.steps, args.agents, True, args.seed)
    without_pool = run_once(args.steps, args.agents, False, args.seed)

    def fmt_mb(b):
        return f"{b/1024/1024:.2f} MB"

    print("Object Pooling Benchmark")
    print(f"Agents: {args.agents}, Steps: {args.steps}")
    print(
        f"With pooling   -> time: {with_pool['elapsed_s']:.2f}s, mem Δ: {fmt_mb(with_pool['mem_delta_bytes'])}, peak RSS: {fmt_mb(with_pool['peak_rss_bytes'])}"
    )
    print(
        f"Without pooling-> time: {without_pool['elapsed_s']:.2f}s, mem Δ: {fmt_mb(without_pool['mem_delta_bytes'])}, peak RSS: {fmt_mb(without_pool['peak_rss_bytes'])}"
    )

    # Improvement estimates
    time_impr = (without_pool["elapsed_s"] - with_pool["elapsed_s"]) / max(
        without_pool["elapsed_s"], 1e-6
    )
    mem_impr = (without_pool["mem_delta_bytes"] - with_pool["mem_delta_bytes"]) / max(
        without_pool["mem_delta_bytes"], 1e-6
    )
    print(
        f"Estimated improvements -> time: {time_impr*100:.1f}%, memory: {mem_impr*100:.1f}%"
    )


if __name__ == "__main__":
    main()

