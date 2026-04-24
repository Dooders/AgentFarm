"""
Genetics Analysis Functions

High-level analysis functions that operate on the normalized DataFrames
produced by the genetics compute layer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from farm.analysis.common.context import AnalysisContext
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_genetics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for a population-genetics DataFrame.

    Accepts either a DataFrame produced by
    :func:`~farm.analysis.genetics.compute.build_agent_genetics_dataframe`
    (DB-backed, columns include ``generation`` and ``action_weights``) or one
    produced by
    :func:`~farm.analysis.genetics.compute.build_evolution_experiment_dataframe`
    (evolution-experiment-backed, columns include ``fitness`` and
    ``chromosome_values``).

    Parameters
    ----------
    df:
        Input DataFrame.  Empty DataFrames are handled gracefully.

    Returns
    -------
    dict
        Summary statistics appropriate for the detected source type.
    """
    if df.empty:
        return {"total_agents": 0}

    result: Dict[str, Any] = {"total_agents": len(df)}

    # --- DB-backed frame ---
    if "generation" in df.columns:
        result["generation_counts"] = df["generation"].value_counts().to_dict()
        result["max_generation"] = int(df["generation"].max())
        result["mean_generation"] = float(df["generation"].mean())

    if "parent_ids" in df.columns:
        result["pct_with_parents"] = float(
            (df["parent_ids"].apply(lambda p: len(p) > 0)).mean() * 100
        )

    if "action_weights" in df.columns:
        non_empty = df["action_weights"].apply(bool)
        result["pct_with_action_weights"] = float(non_empty.mean() * 100)

    # --- Evolution-experiment frame ---
    if "fitness" in df.columns:
        result["best_fitness"] = float(df["fitness"].max())
        result["mean_fitness"] = float(df["fitness"].mean())
        result["min_fitness"] = float(df["fitness"].min())

    if "chromosome_values" in df.columns and not df["chromosome_values"].empty:
        values_by_gene: Dict[str, list] = {}
        skipped_rows = 0
        skipped_values = 0
        for row in df["chromosome_values"]:
            if not isinstance(row, dict):
                skipped_rows += 1
                continue
            for gene, raw_value in row.items():
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    skipped_values += 1
                    continue
                if not math.isfinite(numeric_value):
                    skipped_values += 1
                    continue
                values_by_gene.setdefault(gene, []).append(numeric_value)

        if skipped_rows or skipped_values:
            logger.warning(
                "analyze_genetics: skipped malformed chromosome data rows=%d values=%d",
                skipped_rows,
                skipped_values,
            )

        gene_stats: Dict[str, Any] = {}
        for gene, values in sorted(values_by_gene.items()):
            series = pd.Series(values, dtype=float)
            if not series.empty:
                gene_stats[gene] = {
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
        result["gene_statistics"] = gene_stats

    return result


def generate_genetics_report(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    formats: Optional[str] = "markdown",
    **kwargs: Any,
) -> Optional[Path]:
    """Generate a genetics summary report in Markdown and/or HTML.

    Computes the summary statistics via :func:`analyze_genetics` and writes
    one or both of the following files under ``ctx.output_path``:

    * ``genetics_report.md``  (always written when *formats* includes
      ``"markdown"`` or ``"both"``)
    * ``genetics_report.html`` (written when *formats* includes ``"html"``
      or ``"both"``)

    A machine-readable ``genetics_summary.json`` is also saved alongside
    the report.

    Parameters
    ----------
    df:
        Input DataFrame (DB-backed or evolution-experiment-backed).
    ctx:
        Analysis context supplying output paths and a logger.
    formats:
        ``"markdown"`` (default), ``"html"``, or ``"both"``.

    Returns
    -------
    pathlib.Path or None
        Path to the primary Markdown report, or ``None`` on failure.
    """
    if formats not in {"markdown", "html", "both"}:
        logger.warning(
            "generate_genetics_report: unknown format %r; defaulting to 'markdown'", formats
        )
        formats = "markdown"

    try:
        stats = analyze_genetics(df)

        # ------------------------------------------------------------------
        # JSON summary (machine-readable)
        # ------------------------------------------------------------------
        json_path = ctx.get_output_file("genetics_summary.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2)
        except Exception as exc:
            logger.warning("generate_genetics_report: could not write JSON summary: %s", exc)

        # ------------------------------------------------------------------
        # Build Markdown content
        # ------------------------------------------------------------------
        lines: List[str] = [
            "# Genetics Analysis Report",
            "",
            "## Population Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total agents / candidates | {stats.get('total_agents', 0)} |",
        ]

        if "max_generation" in stats:
            lines.append(f"| Max generation | {stats['max_generation']} |")
        if "mean_generation" in stats:
            lines.append(f"| Mean generation | {stats['mean_generation']:.2f} |")
        if "pct_with_parents" in stats:
            lines.append(f"| % with parents | {stats['pct_with_parents']:.1f}% |")
        if "pct_with_action_weights" in stats:
            lines.append(f"| % with action weights | {stats['pct_with_action_weights']:.1f}% |")

        if "best_fitness" in stats:
            lines += [
                "",
                "## Fitness Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Best fitness | {stats['best_fitness']:.4f} |",
                f"| Mean fitness | {stats['mean_fitness']:.4f} |",
                f"| Min fitness  | {stats['min_fitness']:.4f} |",
            ]

        if "gene_statistics" in stats and stats["gene_statistics"]:
            lines += [
                "",
                "## Gene Statistics",
                "",
                "| Gene | Mean | Std | Min | Max |",
                "|------|------|-----|-----|-----|",
            ]
            for gene, gstats in sorted(stats["gene_statistics"].items()):
                lines.append(
                    f"| {gene} | {gstats['mean']:.4g} | {gstats['std']:.4g} "
                    f"| {gstats['min']:.4g} | {gstats['max']:.4g} |"
                )

        if "generation_counts" in stats:
            lines += [
                "",
                "## Generation Distribution",
                "",
                "| Generation | Count |",
                "|------------|-------|",
            ]
            for gen_key in sorted(
                (k for k in stats["generation_counts"].keys() if str(k).lstrip("-").isdigit()),
                key=lambda x: int(x),
            ):
                lines.append(f"| {gen_key} | {stats['generation_counts'][gen_key]} |")

        lines += [
            "",
            "---",
            "_Generated by the AgentFarm genetics analysis module._",
        ]

        md_content = "\n".join(lines)

        # ------------------------------------------------------------------
        # Write Markdown
        # ------------------------------------------------------------------
        md_path: Optional[Path] = None
        if formats in ("markdown", "both"):
            md_path = ctx.get_output_file("genetics_report.md")
            try:
                with open(md_path, "w", encoding="utf-8") as fh:
                    fh.write(md_content)
                logger.info("generate_genetics_report: wrote Markdown report to %s", md_path)
            except Exception as exc:
                logger.warning("generate_genetics_report: could not write Markdown: %s", exc)
                md_path = None

        # ------------------------------------------------------------------
        # Write HTML
        # ------------------------------------------------------------------
        if formats in ("html", "both"):
            html_path = ctx.get_output_file("genetics_report.html")
            try:
                import html as html_lib

                def _md_table_to_html(md_lines: List[str]) -> str:
                    """Minimal Markdown-table → HTML conversion."""
                    html_rows: List[str] = []
                    header_done = False
                    for line in md_lines:
                        line = line.strip()
                        if not line.startswith("|"):
                            continue
                        if set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
                            continue  # separator row
                        cells = [c.strip() for c in line.split("|")[1:-1]]
                        if not header_done:
                            html_rows.append(
                                "<thead><tr>"
                                + "".join(f"<th>{html_lib.escape(c)}</th>" for c in cells)
                                + "</tr></thead><tbody>"
                            )
                            header_done = True
                        else:
                            html_rows.append(
                                "<tr>"
                                + "".join(f"<td>{html_lib.escape(c)}</td>" for c in cells)
                                + "</tr>"
                            )
                    if html_rows:
                        html_rows.append("</tbody>")
                    return "<table border='1' cellpadding='6' cellspacing='0'>" + "".join(html_rows) + "</table>"

                # Convert markdown to basic HTML
                html_body_parts: List[str] = []
                i = 0
                md_lines_list = md_content.splitlines()
                while i < len(md_lines_list):
                    raw = md_lines_list[i]
                    stripped = raw.strip()
                    if stripped.startswith("# "):
                        html_body_parts.append(f"<h1>{html_lib.escape(stripped[2:])}</h1>")
                    elif stripped.startswith("## "):
                        html_body_parts.append(f"<h2>{html_lib.escape(stripped[3:])}</h2>")
                    elif stripped.startswith("|"):
                        # Collect table block
                        table_block: List[str] = []
                        while i < len(md_lines_list) and md_lines_list[i].strip().startswith("|"):
                            table_block.append(md_lines_list[i])
                            i += 1
                        html_body_parts.append(_md_table_to_html(table_block))
                        continue
                    elif stripped == "---":
                        html_body_parts.append("<hr/>")
                    elif stripped.startswith("_") and stripped.endswith("_"):
                        html_body_parts.append(f"<em>{html_lib.escape(stripped.strip('_'))}</em>")
                    elif stripped:
                        html_body_parts.append(f"<p>{html_lib.escape(stripped)}</p>")
                    i += 1

                html_content = (
                    "<!DOCTYPE html><html><head>"
                    "<meta charset='utf-8'/>"
                    "<title>Genetics Analysis Report</title>"
                    "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;}"
                    "table{border-collapse:collapse;}th,td{text-align:left;padding:6px;}"
                    "</style>"
                    "</head><body>"
                    + "\n".join(html_body_parts)
                    + "</body></html>"
                )
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(html_content)
                logger.info("generate_genetics_report: wrote HTML report to %s", html_path)
            except Exception as exc:
                logger.warning("generate_genetics_report: could not write HTML: %s", exc)

        return md_path or ctx.get_output_file("genetics_report.md")

    except Exception as exc:
        logger.warning("generate_genetics_report failed: %s", exc)
        return None
