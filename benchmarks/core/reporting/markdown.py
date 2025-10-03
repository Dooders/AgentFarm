from __future__ import annotations

"""
Markdown report generation for individual runs and sweeps.
"""

import json
import os
from typing import Any, Dict, List

from benchmarks.core.results import RunResult


def _format_duration_section(result: RunResult) -> str:
    m = result.metrics.get("duration_s", {}) if result.metrics else {}
    lines = ["### Duration"]
    if m:
        lines.append(f"- mean: {m.get('mean', 0):.6f}s")
        lines.append(f"- p50: {m.get('p50', 0):.6f}s")
        lines.append(f"- p95: {m.get('p95', 0):.6f}s")
    else:
        lines.append("- No duration metrics available")
    return "\n".join(lines)


def write_run_report(result: RunResult, run_dir: str) -> str:
    """Write a concise Markdown report for a single run into run_dir."""
    lines: List[str] = []
    lines.append(f"# {result.name} — {result.run_id}")
    lines.append("")
    lines.append("## Parameters")
    lines.append("```json")
    lines.append(json.dumps(result.parameters, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Iterations")
    lines.append(f"Warmup: {result.iterations.get('warmup', 0)}, Measured: {result.iterations.get('measured', 0)}")
    lines.append("")
    lines.append(_format_duration_section(result))
    lines.append("")
    if result.artifacts:
        lines.append("### Artifacts")
        for a in result.artifacts:
            rel = os.path.relpath(a.path, run_dir) if os.path.isabs(a.path) else a.path
            lines.append(f"- {a.name} ({a.type}): `{rel}`")
        lines.append("")
    if result.notes:
        lines.append("### Notes")
        lines.append(result.notes)
        lines.append("")

    path = os.path.join(run_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_sweep_report(results: List[RunResult], out_dir: str, title: str = "Sweep Summary") -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| run_id | parameters | mean(s) | p95(s) | output |")
    lines.append("|---|---|---:|---:|---|")
    for r in results:
        m = r.metrics.get("duration_s", {}) if r.metrics else {}
        mean = m.get("mean", 0.0)
        p95 = m.get("p95", 0.0)
        params_json = json.dumps(r.parameters, separators=(",", ":"))
        lines.append(f"| {r.run_id} | `{params_json}` | {mean:.6f} | {p95:.6f} | `{os.path.basename(r.name)}_{r.run_id}` |")
    path = os.path.join(out_dir, "SWEEP_SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

