from __future__ import annotations

"""
Instrumentation for sampling system metrics via psutil during an iteration.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


class _Sampler(threading.Thread):
    def __init__(self, interval_s: float, out_list: List[Dict[str, Any]], max_samples: int | None = None):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.out_list = out_list
        self._stop = threading.Event()
        self.max_samples = max_samples

    def run(self) -> None:
        if psutil is None:
            return
        process = psutil.Process()
        # Prime CPU percent
        try:
            process.cpu_percent(interval=None)
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                ts = time.monotonic()
                cpu = process.cpu_percent(interval=None)
                mem = process.memory_info().rss
                io = process.io_counters() if hasattr(process, "io_counters") else None
                self.out_list.append(
                    {
                        "ts": ts,
                        "cpu_percent": cpu,
                        "rss_bytes": mem,
                        "read_bytes": getattr(io, "read_bytes", None),
                        "write_bytes": getattr(io, "write_bytes", None),
                    }
                )
                # Enforce max samples
                if self.max_samples is not None and len(self.out_list) >= self.max_samples:
                    break
            except Exception:
                pass
            # sleep using monotonic reference
            target = time.monotonic() + self.interval_s
            while not self._stop.is_set():
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.05))

    def stop(self) -> None:
        self._stop.set()


@contextmanager
def psutil_sampling(run_dir: str, name: str, iteration_index: int, metrics: Dict[str, Any], interval_ms: int = 200, max_samples: int | None = 1000):
    """Context manager that samples psutil metrics and writes a JSONL artifact."""
    if psutil is None:
        yield
        return

    samples: List[Dict[str, Any]] = []
    sampler = _Sampler(interval_s=max(0.01, interval_ms / 1000.0), out_list=samples, max_samples=max_samples)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()
        sampler.join(timeout=1.5)
        prefix = f"{name}_iter{iteration_index:03d}"
        jsonl_path = os.path.join(run_dir, f"{prefix}_psutil.jsonl")
        try:
            with open(jsonl_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            metrics["psutil_artifact"] = jsonl_path
            # Aggregate stats
            if samples:
                rss_vals = [s.get("rss_bytes", 0) or 0 for s in samples]
                cpu_vals = [s.get("cpu_percent", 0.0) or 0.0 for s in samples]
                metrics["psutil_summary"] = {
                    "rss_bytes": {
                        "min": int(min(rss_vals)),
                        "max": int(max(rss_vals)),
                        "mean": float(sum(rss_vals) / len(rss_vals)),
                    },
                    "cpu_percent": {
                        "min": float(min(cpu_vals)),
                        "max": float(max(cpu_vals)),
                        "mean": float(sum(cpu_vals) / len(cpu_vals)),
                    },
                    "samples": len(samples),
                }
        except Exception:
            pass

