from __future__ import annotations

"""
Instrumentation for capturing cProfile data and a lightweight JSON summary.
"""

import cProfile
import json
import os
import pstats
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple


def _summarize_stats_from_file(prof_path: str, top_n: int = 30) -> Dict[str, Any]:
    stats = pstats.Stats(prof_path)
    # Build list of rows (file, line, func, cc, nc, tt, ct)
    rows: List[Tuple[str, int, str, int, int, float, float]] = []
    for (filename, line, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append((filename, line, funcname, cc, nc, tt, ct))

    def _fmt(row):
        file, line, func, cc, nc, tt, ct = row
        return {
            "function": func,
            "file": file,
            "line": line,
            "calls": nc,
            "primitive_calls": cc,
            "internal_time": tt,
            "cumulative_time": ct,
        }

    by_cumulative = sorted(rows, key=lambda r: r[6], reverse=True)[:top_n]
    by_internal = sorted(rows, key=lambda r: r[5], reverse=True)[:top_n]
    by_calls = sorted(rows, key=lambda r: r[4], reverse=True)[:top_n]

    return {
        "top_cumulative": [_fmt(r) for r in by_cumulative],
        "top_internal": [_fmt(r) for r in by_internal],
        "top_calls": [_fmt(r) for r in by_calls],
    }


@contextmanager
def cprofile_capture(run_dir: str, name: str, iteration_index: int, metrics: Dict[str, Any], top_n: int = 30):
    """Context manager to capture cProfile into artifacts and add summary paths to metrics."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        prefix = f"{name}_iter{iteration_index:03d}"
        prof_path = os.path.join(run_dir, f"{prefix}.prof")
        try:
            profiler.dump_stats(prof_path)
            metrics["cprofile_artifact"] = prof_path
            # Summarize using pstats on dumped file for consistency
            summary = _summarize_stats_from_file(prof_path, top_n=top_n)
            summary_path = os.path.join(run_dir, f"{prefix}_cprofile_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            metrics["cprofile_summary_path"] = summary_path
        except Exception:
            # Best-effort; ignore failures gracefully
            pass

