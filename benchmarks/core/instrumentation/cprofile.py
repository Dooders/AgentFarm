from __future__ import annotations

"""
Instrumentation for capturing cProfile data and a lightweight JSON summary.
"""

import cProfile
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple


def _summarize_stats(profile: cProfile.Profile, top_n: int = 30) -> Dict[str, Any]:
    # stats.stats: dict[(filename, line, funcname) -> (cc, nc, tt, ct, callers)]
    stats_items: List[Tuple[Tuple[str, int, str], Tuple[int, int, float, float, Dict]]]
    stats_items = list(profile.getstats())  # type: ignore[attr-defined]
    # For Python's cProfile.Profile, .getstats() may not include callers; fallback to .getstats only
    # However, getstats provides (func_std_string, cc, nc, tt, ct, callers) in older versions
    # To be robust, try profile.stats when available
    try:
        # type: ignore[attr-defined]
        stats_dict = profile.stats  # type: ignore
        stats_items = list(stats_dict.items())  # type: ignore
        def _row(item):
            (filename, line, funcname), (cc, nc, tt, ct, callers) = item  # type: ignore[misc]
            return filename, line, funcname, cc, nc, tt, ct
        rows = [_row(it) for it in stats_items]  # (file, line, func, cc, nc, tt, ct)
    except Exception:
        # Fallback path: best-effort empty summary
        rows = []

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
        except Exception:
            pass

        try:
            summary = _summarize_stats(profiler, top_n=top_n)
            summary_path = os.path.join(run_dir, f"{prefix}_cprofile_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            metrics["cprofile_summary_path"] = summary_path
        except Exception:
            # Summary is best-effort
            pass

