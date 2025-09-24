import argparse
import gc
import os
import random
import time
from contextlib import contextmanager

import psutil

from farm.core.pool import AgentPool


class FakeAgent:
    """A synthetic heavy object to emulate BaseAgent allocation cost.

    Allocates a sizable bytearray and nested lists to create measurable RSS.
    """

    def __init__(self, agent_id: str, payload_kb: int = 64):
        self.reset(agent_id=agent_id, payload_kb=payload_kb)

    def reset(self, *, agent_id: str, payload_kb: int = 64) -> None:
        self.agent_id = agent_id
        # Heavy payload simulating tensors/buffers
        self.buffer = bytearray(payload_kb * 1024)
        self.list_payload = [0] * (payload_kb * 16)

    def prepare_for_release(self) -> None:
        # Drop references to allow reuse
        self.list_payload.clear()


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


def run_once(num_objects: int, payload_kb: int, use_pooling: bool, max_pool: int | None) -> dict:
    if use_pooling:
        pool = AgentPool(FakeAgent, max_size=max_pool or None)
    else:
        pool = None

    created: list[FakeAgent] = []
    with measure_memory() as finish:
        # Acquire many objects
        for i in range(num_objects):
            agent_id = f"a{i}"
            if pool is None:
                obj = FakeAgent(agent_id=agent_id, payload_kb=payload_kb)
            else:
                obj = pool.acquire(agent_id=agent_id, payload_kb=payload_kb)
            created.append(obj)

        # Randomly release half
        random.shuffle(created)
        to_release = created[: num_objects // 2]
        for obj in to_release:
            if pool is None:
                continue
            pool.release(obj)

        # Re-acquire the same number again
        reacquired = []
        for i in range(len(to_release)):
            agent_id = f"b{i}"
            if pool is None:
                obj = FakeAgent(agent_id=agent_id, payload_kb=payload_kb)
            else:
                obj = pool.acquire(agent_id=agent_id, payload_kb=payload_kb)
            reacquired.append(obj)

        # Keep strong references until after measuring
        gc.collect()
    elapsed, mem_delta, peak = finish()

    stats = {
        "use_pooling": use_pooling,
        "elapsed_s": elapsed,
        "mem_delta_bytes": mem_delta,
        "peak_rss_bytes": peak,
        "num_objects": num_objects,
        "payload_kb": payload_kb,
    }
    if pool is not None:
        stats.update({
            "pool_size": pool.size(),
            "total_created": pool.total_created,
            "total_reused": pool.total_reused,
            "capacity": pool.capacity(),
        })
    return stats


def main():
    parser = argparse.ArgumentParser(description="Synthetic benchmark for AgentPool")
    parser.add_argument("--objects", type=int, default=2000)
    parser.add_argument("--payload-kb", type=int, default=64)
    parser.add_argument("--max-pool", type=int, default=1000)
    args = parser.parse_args()

    # Warm-up
    run_once(100, args.payload_kb, True, args.max_pool)
    run_once(100, args.payload_kb, False, None)

    with_pool = run_once(args.objects, args.payload_kb, True, args.max_pool)
    without_pool = run_once(args.objects, args.payload_kb, False, None)

    def fmt_mb(b):
        return f"{b/1024/1024:.2f} MB"

    print("Synthetic Object Pooling Benchmark (FakeAgent)")
    print(f"Objects: {args.objects}, Payload: {args.payload_kb}KB, MaxPool: {args.max_pool}")
    print(
        f"With pooling   -> time: {with_pool['elapsed_s']:.3f}s, mem Δ: {fmt_mb(with_pool['mem_delta_bytes'])}, peak RSS: {fmt_mb(with_pool['peak_rss_bytes'])}, reused: {with_pool.get('total_reused', 0)}, created: {with_pool.get('total_created', 0)}"
    )
    print(
        f"Without pooling-> time: {without_pool['elapsed_s']:.3f}s, mem Δ: {fmt_mb(without_pool['mem_delta_bytes'])}, peak RSS: {fmt_mb(without_pool['peak_rss_bytes'])}"
    )

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

