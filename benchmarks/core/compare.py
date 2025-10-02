from __future__ import annotations

"""
Compare two RunResult JSON files and emit a concise Markdown summary.
"""

import json
from typing import Any, Dict


def load_result(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _get_metric(res: Dict[str, Any], key: str, sub: str, default: float = 0.0) -> float:
    return float(((res.get("metrics") or {}).get(key) or {}).get(sub, default))


def compare_results(a_path: str, b_path: str) -> str:
    a = load_result(a_path)
    b = load_result(b_path)

    lines = []
    lines.append(f"# Compare: {a.get('name')} vs {b.get('name')}")
    lines.append("")
    lines.append(f"A: `{a_path}` (run_id={a.get('run_id')})")
    lines.append(f"B: `{b_path}` (run_id={b.get('run_id')})")
    lines.append("")
    lines.append("## Duration (seconds)")
    for sub in ("mean", "p50", "p95"):
        va = _get_metric(a, "duration_s", sub)
        vb = _get_metric(b, "duration_s", sub)
        delta = vb - va
        rel = ((vb / va) - 1.0) * 100.0 if va else 0.0
        lines.append(f"- {sub}: A={va:.6f}, B={vb:.6f} (Δ={delta:+.6f}, {rel:+.2f}%)")

    # Common throughput metric key (if present)
    def _tp(res: Dict[str, Any]) -> float:
        # Look in last iteration metrics
        iters = res.get("iteration_metrics") or []
        if iters:
            last = iters[-1].get("metrics") or {}
            for k in ("observes_per_sec", "operations_per_second", "iterations_per_second"):
                if k in last:
                    return float(last.get(k, 0.0))
        return 0.0

    ta = _tp(a)
    tb = _tp(b)
    if ta or tb:
        d = tb - ta
        r = ((tb / ta) - 1.0) * 100.0 if ta else 0.0
        lines.append("")
        lines.append("## Throughput")
        lines.append(f"- A={ta:.2f}, B={tb:.2f} (Δ={d:+.2f}, {r:+.2f}%)")

    return "\n".join(lines)

