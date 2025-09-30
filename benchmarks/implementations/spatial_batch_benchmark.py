"""Micro-benchmark for SpatialIndex batch updates.

Runs a small sweep over number of entities and measures batch update processing time.
This is intended as an informational benchmark (non-gating in CI).
"""

import time
from unittest.mock import Mock

from farm.core.spatial import SpatialIndex


def run_benchmark(width: float = 1000.0, height: float = 1000.0) -> None:
    for num_entities in (100, 500, 1000, 2000):
        idx = SpatialIndex(width=width, height=height, enable_batch_updates=True, max_batch_size=100)
        entities = [Mock() for _ in range(num_entities)]
        for i, entity in enumerate(entities):
            idx.add_position_update(entity, (i * 1.0, i * 1.0), (i * 1.5, i * 1.5), "agent")
        t0 = time.time()
        idx.process_batch_updates(force=True)
        dt = time.time() - t0
        stats = idx.get_batch_update_stats()
        print(f"entities={num_entities:5d} time={dt:.6f}s avg_batch={stats['average_batch_size']:.1f}")


if __name__ == "__main__":
    run_benchmark()

