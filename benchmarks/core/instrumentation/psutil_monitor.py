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
    def __init__(self, interval_s: float, out_list: List[Dict[str, Any]]):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.out_list = out_list
        self._stop = threading.Event()

    def run(self) -> None:
        if psutil is None:
            return
        process = psutil.Process()
        while not self._stop.is_set():
            try:
                ts = time.time()
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
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop.set()


@contextmanager
def psutil_sampling(run_dir: str, name: str, iteration_index: int, metrics: Dict[str, Any], interval_ms: int = 200):
    """Context manager that samples psutil metrics and writes a JSONL artifact."""
    if psutil is None:
        yield
        return

    samples: List[Dict[str, Any]] = []
    sampler = _Sampler(interval_s=max(0.01, interval_ms / 1000.0), out_list=samples)
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
        except Exception:
            pass

