"""Per-run lineage summary helpers shared across A/B analyzer scripts.

Both ``scripts/compare_crossover_arms.py`` and
``scripts/compare_inheritance_arms.py`` need the same ``cluster_lineage.jsonl``
parsing logic. Previously they reached across the script boundary via a
name-mangled private import (``from scripts.compare_crossover_arms import
_lineage_metrics``); promoting the helpers here gives both call sites a
stable, documented surface and makes the parsing trivially unit-testable
without standing up matplotlib.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

CLUSTER_LINEAGE_FILENAME = "cluster_lineage.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts; silently skips malformed lines."""
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(values: List[float]) -> float:
    """Arithmetic mean of ``values``; returns NaN on empty input.

    Self-contained (no ``statistics`` import) so this module stays free of
    optional dependencies and matches the previous behaviour exactly.
    """
    if not values:
        return float("nan")
    return sum(values) / len(values)


def lineage_metrics(run_dir: Path) -> Dict[str, Any]:
    """Per-run lineage summary from ``cluster_lineage.jsonl``.

    Returns a dict with keys:

    - ``cluster_count_trace``: ``[(step, k), ...]`` where ``k`` is the
      number of distinct cluster IDs observed at that snapshot step.
    - ``mean_k``: mean cluster count across snapshot steps (``nan`` when
      no rows were recorded).
    - ``churn_rate``: mean per-step fraction of cluster IDs at step ``t``
      that do not survive to step ``t+1``. In ``[0, 1]``. ``nan`` when
      fewer than two clustering steps were recorded.

    Missing or unreadable lineage files yield the empty/NaN shape; callers
    can then uniformly skip seeds with incomplete data.
    """
    rows = _read_jsonl(run_dir / CLUSTER_LINEAGE_FILENAME)
    if not rows:
        return {
            "cluster_count_trace": [],
            "mean_k": float("nan"),
            "churn_rate": float("nan"),
        }

    steps_to_ids: Dict[int, set] = defaultdict(set)
    for row in rows:
        try:
            step = int(row["step"])
            cluster_id = row["cluster_id"]
        except (KeyError, TypeError, ValueError):
            continue
        steps_to_ids[step].add(cluster_id)

    sorted_steps = sorted(steps_to_ids.keys())
    trace = [(s, len(steps_to_ids[s])) for s in sorted_steps]
    mean_k = _mean([k for _, k in trace]) if trace else float("nan")

    if len(sorted_steps) < 2:
        churn = float("nan")
    else:
        churns: List[float] = []
        for s_prev, s_next in zip(sorted_steps, sorted_steps[1:]):
            prev = steps_to_ids[s_prev]
            nxt = steps_to_ids[s_next]
            if not prev:
                continue
            died = prev - nxt
            churns.append(len(died) / len(prev))
        churn = _mean(churns) if churns else float("nan")

    return {
        "cluster_count_trace": trace,
        "mean_k": mean_k,
        "churn_rate": churn if not math.isnan(churn) else float("nan"),
    }


__all__ = ["CLUSTER_LINEAGE_FILENAME", "lineage_metrics"]
