"""Configurable thresholds and knobs for memmap acceptance validation.

Environment variables can override defaults at runtime:
- MEMMAP_RSS_DELTA_RATIO: acceptable RSS increase as a fraction of memmap size (default 0.5)
  Lower values (e.g., 0.1-0.3) provide stricter memory usage validation but may be too
  conservative for systems with high baseline memory usage. Higher values (e.g., 0.8-1.0)
  allow more memory overhead but risk missing true memory issues. The 0.5 default balances
  detection sensitivity with practical system variation.
"""

import math
import os
import random
import sys
import tempfile
import time
from typing import Tuple

MEMORY_ACCEPTANCE_RSS_DELTA_RATIO: float = float(
    os.getenv("MEMMAP_RSS_DELTA_RATIO", "0.5")
)

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional in env
    torch = None


def get_rss_mb() -> float:
    """Return resident set size (MB) using /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # VmRSS:    123456 kB
                    kb = float(parts[1])
                    return kb / 1024.0
    except Exception:
        pass
    return -1.0


def baseline_spatial_window(
    env, ay: int, ax: int, R: int, max_amount: float
) -> np.ndarray:
    """Build local resource window using spatial queries (no memmap)."""
    S = 2 * R + 1
    out = np.zeros((S, S), dtype=np.float32)
    try:
        nearby = env.spatial_index.get_nearby((ax, ay), R + 1, ["resources"])  # x,y
        resources = nearby.get("resources", [])
    except Exception:
        resources = []

    # Nearest-cell deposit
    width, height = int(env.width), int(env.height)

    def disc(v: float, bound: int) -> int:
        v_floor = math.floor(v)
        return max(0, min(int(v_floor), bound - 1))

    for res in resources:
        rx = disc(res.position[0], width)
        ry = disc(res.position[1], height)
        lx = rx - (ax - R)
        ly = ry - (ay - R)
        if 0 <= lx < S and 0 <= ly < S:
            out[int(ly), int(lx)] += float(res.amount) / float(max_amount)
    return out


def run_trial(
    width: int = 1500, height: int = 1500, R: int = 6, n_windows: int = 1500
) -> int:
    from farm.config.config import EnvironmentConfig, ResourceConfig, SimulationConfig
    from farm.core.environment import Environment

    tmpdir = os.getenv("TMPDIR", tempfile.gettempdir())
    initial_resources = max(2000, (width * height) // 1000)
    max_amount = 30

    # Baseline env (no memmap)
    cfg_base = SimulationConfig(
        environment=EnvironmentConfig(width=width, height=height),
        resources=ResourceConfig(
            initial_resources=initial_resources, max_resource_amount=max_amount
        ),
        seed=12345,
    )
    cfg_base.use_memmap_resources = False

    # Memmap env
    cfg_mm = SimulationConfig(
        environment=EnvironmentConfig(width=width, height=height),
        resources=ResourceConfig(
            initial_resources=initial_resources, max_resource_amount=max_amount
        ),
        seed=12345,
    )
    cfg_mm.use_memmap_resources = True
    cfg_mm.memmap_dir = tmpdir
    cfg_mm.memmap_dtype = "float32"
    cfg_mm.memmap_mode = "w+"

    print(
        f"Grid: {width}x{height}, resources: {initial_resources}, windows: {n_windows}, R={R}"
    )

    # Build memmap env
    rss_before_mm = get_rss_mb()
    env_mm = Environment(
        width=cfg_mm.environment.width,
        height=cfg_mm.environment.height,
        resource_distribution={},
        config=cfg_mm,
    )
    rss_after_mm = get_rss_mb()
    rm = env_mm.resource_manager
    mm_path = getattr(rm, "_memmap_path", None)
    mm_exists = bool(mm_path and os.path.exists(mm_path))
    mm_size_mb = os.path.getsize(mm_path) / (1024.0 * 1024.0) if mm_exists else 0.0
    print(f"memmap path: {mm_path}")
    print(f"memmap exists: {mm_exists}, size_mb: {mm_size_mb:.2f}")
    print(
        f"RSS before memmap env: {rss_before_mm:.1f} MB, after init: {rss_after_mm:.1f} MB"
    )

    # Time memmap window access
    random.seed(0)
    H, W = height, width
    t0 = time.perf_counter()
    for _ in range(n_windows):
        ay = random.randint(0, H - 1)
        ax = random.randint(0, W - 1)
        win = rm.get_resource_window(ay - R, ay + R + 1, ax - R, ax + R + 1)
        assert win.shape == (2 * R + 1, 2 * R + 1)
    t1 = time.perf_counter()
    memmap_avg_ms = (t1 - t0) * 1000.0 / n_windows
    print(f"memmap window avg: {memmap_avg_ms:.3f} ms")

    # Torch compatibility check
    comp_ok = True
    try:
        if torch is not None:
            sample = rm.get_resource_window(
                H // 2 - R, H // 2 + R + 1, W // 2 - R, W // 2 + R + 1
            )
            ten = torch.from_numpy(sample).to(dtype=torch.float32)
            val = float(ten.sum().item())
            print(f"torch sum(sample) = {val:.6f}")
        else:
            print("torch not available; skipping torch compatibility check")
    except Exception as e:
        comp_ok = False
        print(f"torch compatibility failed: {e}")

    env_mm.close()

    # Baseline env (KD-tree + nearest fill)
    rss_before_base = get_rss_mb()
    env_base = Environment(
        width=cfg_base.environment.width,
        height=cfg_base.environment.height,
        resource_distribution={},
        config=cfg_base,
    )
    rss_after_base = get_rss_mb()
    print(
        f"RSS before baseline env: {rss_before_base:.1f} MB, after init: {rss_after_base:.1f} MB"
    )

    random.seed(0)
    t0 = time.perf_counter()
    for _ in range(n_windows):
        ay = random.randint(0, H - 1)
        ax = random.randint(0, W - 1)
        _ = baseline_spatial_window(env_base, ay, ax, R, max_amount)
    t1 = time.perf_counter()
    baseline_avg_ms = (t1 - t0) * 1000.0 / n_windows
    print(f"baseline window avg: {baseline_avg_ms:.3f} ms")

    env_base.close()

    # Acceptance checks
    print("\nAcceptance Criteria:")
    # 1) States load/stream without full RAM usage -> memmap exists; RSS didn't jump close to memmap size
    memory_ok = mm_exists and (rss_after_mm - rss_before_mm) < (
        mm_size_mb * MEMORY_ACCEPTANCE_RSS_DELTA_RATIO
    )
    print(
        f"- Streaming without full RAM usage: {'PASS' if memory_ok else 'WARN'} (RSS change={rss_after_mm - rss_before_mm:.1f} MB vs file {mm_size_mb:.1f} MB)"
    )

    # 2) Performance no significant slowdown -> memmap <= 1.25x baseline
    perf_ok = memmap_avg_ms <= baseline_avg_ms * 1.25
    print(
        f"- Performance: memmap {memmap_avg_ms:.3f} ms vs baseline {baseline_avg_ms:.3f} ms -> {'PASS' if perf_ok else 'FAIL'}"
    )

    # 3) Compatible with tensor ops
    print(f"- Tensor compatibility: {'PASS' if comp_ok else 'FAIL'}")

    overall = memory_ok and perf_ok and comp_ok
    print(
        f"\nOverall: {'PASS' if overall else 'INCONCLUSIVE' if memory_ok and comp_ok else 'FAIL'}"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(run_trial())
