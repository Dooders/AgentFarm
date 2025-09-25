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

    # Get pool statistics if available
    pool_stats = None
    if use_pooling and hasattr(env, "agent_pool") and env.agent_pool is not None:
        pool_stats = {
            "total_created": env.agent_pool.total_created,
            "total_reused": env.agent_pool.total_reused,
            "pool_size": env.agent_pool.size(),
            "reuse_rate": (env.agent_pool.total_reused / max(env.agent_pool.total_created, 1)) * 100,
        }

    return {

    result = {
        "use_pooling": use_pooling,
        "elapsed_s": elapsed,
        "mem_delta_bytes": mem_delta,
        "peak_rss_bytes": peak,
        "agents": agents,
        "steps": steps,
        "pool_stats": pool_stats,
    }
    
    # Capture pool statistics if pooling was used
    if use_pooling and hasattr(env, 'agent_pool') and env.agent_pool is not None:
        result["pool_stats"] = env.agent_pool.get_stats()
    
    return result


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
    
    # Show pool statistics if available
    if 'pool_stats' in with_pool:
        stats = with_pool['pool_stats']
        print(f"Pool stats: {stats['total_reused']} reused, {stats['total_created']} created, {stats['reuse_rate_percent']:.1f}% reuse rate")

    # Show pool statistics
    if with_pool["pool_stats"]:
        print("Pool Statistics:")
        print(f"  Total created: {with_pool['pool_stats']['total_created']}")
        print(f"  Total reused: {with_pool['pool_stats']['total_reused']}")
        print(f"  Reuse rate: {with_pool['pool_stats']['reuse_rate']:.1f}%")
        print(f"  Pool size: {with_pool['pool_stats']['pool_size']}")

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

