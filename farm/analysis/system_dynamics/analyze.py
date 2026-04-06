"""
System dynamics analysis: foundation module runs plus cross-domain synthesis and reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from farm.analysis.common.context import AnalysisContext
from farm.analysis.population.module import population_module
from farm.analysis.resources.module import resources_module
from farm.analysis.temporal.module import temporal_module
from farm.analysis.system_dynamics.compute import json_safe, synthesize_system_dynamics
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def run_foundation_analyses(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Run population, resources, and temporal analysis groups into subfolders.

    Failures are logged and skipped so synthesis can still run on merged data.
    """
    del df  # Uses experiment_path from context
    exp_raw = ctx.metadata.get("experiment_path")
    if not exp_raw:
        ctx.logger.warning("No experiment_path in context; skipping foundation analyses")
        return

    experiment_path = Path(exp_raw)
    if not experiment_path.is_dir():
        ctx.logger.warning("experiment_path is not a directory: %s", experiment_path)
        return

    ref_root = ctx.output_path / "_system_dynamics_refs"
    ref_root.mkdir(parents=True, exist_ok=True)

    modules = [
        ("population", population_module, "analysis"),
        ("resources", resources_module, "analysis"),
        ("temporal", temporal_module, "analysis"),
    ]

    for name, module, group in modules:
        out_dir = ref_root / name
        try:
            module.run_analysis(
                experiment_path=experiment_path,
                output_path=out_dir,
                group=group,
            )
            ctx.logger.info("Foundation submodule %s completed -> %s", name, out_dir)
        except Exception as err:
            ctx.logger.warning("Foundation submodule %s failed: %s", name, err)


def analyze_system_dynamics_synthesis(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Cross-domain metrics (correlations, Granger, feedback heuristics)."""
    synthesis = synthesize_system_dynamics(df)
    ctx.metadata.setdefault("_system_dynamics", {})["synthesis"] = synthesis

    out = ctx.get_output_file("system_dynamics_synthesis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(synthesis), f, indent=2)

    ctx.logger.info("Wrote synthesis to %s", out)


def write_unified_system_dynamics_report(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Single JSON + HTML summary combining synthesis (and optional foundation paths)."""
    del df
    bundle = ctx.metadata.get("_system_dynamics", {})
    synthesis = bundle.get("synthesis")
    if synthesis is None:
        synthesis = {}
        ctx.logger.warning("No synthesis in context; report will be minimal")

    ref_root = ctx.output_path / "_system_dynamics_refs"
    foundation_meta = {
        "population_dir": str(ref_root / "population") if (ref_root / "population").exists() else None,
        "resources_dir": str(ref_root / "resources") if (ref_root / "resources").exists() else None,
        "temporal_dir": str(ref_root / "temporal") if (ref_root / "temporal").exists() else None,
    }

    report = {
        "module": "system_dynamics",
        "experiment_path": ctx.metadata.get("experiment_path"),
        "synthesis": synthesis,
        "foundation_outputs": foundation_meta,
    }

    json_path = ctx.get_output_file("system_dynamics_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    html_path = ctx.get_output_file("system_dynamics_report.html")
    _write_summary_html(html_path, report)

    ctx.logger.info("Unified report: %s, %s", json_path, html_path)


def _write_summary_html(path: Path, report: dict) -> None:
    """Minimal HTML overview for quick viewing."""
    syn = report.get("synthesis") or {}
    rp = syn.get("resource_population") or {}
    levels = rp.get("levels") or {}
    fd = rp.get("first_differences") or {}

    r_level = levels.get("pearson_r")
    r_diff = fd.get("pearson_r")
    feedback = (syn.get("feedback_loop_candidates") or {}).get("count", "—")

    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>System dynamics report</title></head><body>",
        "<h1>System dynamics</h1>",
        f"<p>Experiment: {report.get('experiment_path', '')}</p>",
        "<h2>Resource–population</h2>",
        "<ul>",
        f"<li>Levels Pearson r: {r_level}</li>",
        f"<li>First-difference Pearson r: {r_diff}</li>",
        "</ul>",
        "<h2>Feedback-loop candidates</h2>",
        f"<p>Heuristic periods flagged: {feedback}</p>",
        "<p>See <code>system_dynamics_report.json</code> for full detail.</p>",
        "</body></html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
