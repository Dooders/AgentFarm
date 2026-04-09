#!/usr/bin/env python3
"""Aggregate existing comparison-matrix JSON reports into a Markdown/CSV summary.

This script reads the JSON report files produced by ``run_comparison_matrix.py``
(or by individual ``validate_recombination.py`` / ``validate_quantized.py``
calls) and merges them into a single summary table, making it easy to paste
results into documentation or GitHub issues.

Supported report types
----------------------
* **Recombination reports** — JSON files with a ``comparisons`` key and
  ``summary`` section (output of :class:`RecombinationEvaluator`).
* **Quantized-fidelity reports** — JSON files with a ``fidelity``, ``latency``,
  and ``size`` key (output of :class:`QuantizedValidator`).

The script auto-detects report type by key inspection.

How to run
----------
::

    # Summarise all JSON reports in a directory
    python scripts/summarise_comparison_matrix.py \\
        --report-dir reports/comparison_matrix

    # Summarise specific files
    python scripts/summarise_comparison_matrix.py \\
        --files \\
            reports/comparison_matrix/row_A_child_vs_parents.json \\
            reports/comparison_matrix/row_B_child_vs_students.json \\
            reports/comparison_matrix/row_C_child_vs_int8_students.json

    # Custom output path
    python scripts/summarise_comparison_matrix.py \\
        --report-dir reports/comparison_matrix \\
        --output-md  reports/summary.md \\
        --output-csv reports/summary.csv

Output
------
Writes ``comparison_matrix_summary.md`` and ``comparison_matrix_summary.csv``
to ``--report-dir`` (or the paths given by ``--output-md`` / ``--output-csv``).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

_RECOMBI_COLUMNS = [
    "label", "type",
    "child_vs_ref_a_agreement", "child_vs_ref_b_agreement", "oracle_agreement",
    "kl_ref_a", "kl_ref_b", "cosine_ref_a", "cosine_ref_b", "passed",
]

_QUANT_COLUMNS = [
    "label", "type",
    "action_agreement", "kl_divergence", "mean_cosine_similarity",
    "float_ms", "quant_ms", "size_ratio", "passed",
]


def _detect_type(report: Dict[str, Any]) -> str:
    """Return ``'recombination'`` or ``'quantized_fidelity'`` based on top-level keys."""
    if "comparisons" in report and "summary" in report:
        return "recombination"
    if "fidelity" in report and "latency" in report:
        return "quantized_fidelity"
    return "unknown"


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _extract_recombination(label: str, report: Dict[str, Any]) -> Dict[str, str]:
    summary = report.get("summary", {})
    comparisons = report.get("comparisons", {})
    cmp_a = comparisons.get("child_vs_parent_a", {})
    cmp_b = comparisons.get("child_vs_parent_b", {})
    return {
        "label": label,
        "type": "recombination",
        "child_vs_ref_a_agreement": _fmt(summary.get("child_agrees_with_parent_a")),
        "child_vs_ref_b_agreement": _fmt(summary.get("child_agrees_with_parent_b")),
        "oracle_agreement": _fmt(summary.get("oracle_agreement")),
        "kl_ref_a": _fmt(cmp_a.get("kl_divergence"), 6),
        "kl_ref_b": _fmt(cmp_b.get("kl_divergence"), 6),
        "cosine_ref_a": _fmt(cmp_a.get("mean_cosine_similarity")),
        "cosine_ref_b": _fmt(cmp_b.get("mean_cosine_similarity")),
        "passed": str(report.get("passed", "n/a")),
    }


def _extract_quant_fidelity(label: str, report: Dict[str, Any]) -> Dict[str, str]:
    fidelity = report.get("fidelity", {})
    latency = report.get("latency", {})
    size = report.get("size", {})
    size_ratio = size.get("size_ratio")
    kl = fidelity.get("kl_divergence_float_vs_quant")
    if kl is None:
        kl = fidelity.get("kl_divergence")
    return {
        "label": label,
        "type": "quantized_fidelity",
        "action_agreement": _fmt(fidelity.get("action_agreement")),
        "kl_divergence": _fmt(kl, 6) if kl is not None else "n/a",
        "mean_cosine_similarity": _fmt(fidelity.get("mean_cosine_similarity")),
        "float_ms": _fmt(latency.get("float_inference_ms")),
        "quant_ms": _fmt(latency.get("quantized_inference_ms")),
        "size_ratio": _fmt(size_ratio) if size_ratio is not None else "n/a",
        "passed": str(report.get("passed", "n/a")),
    }


def _load_reports(report_dir: str, extra_files: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(path, report_dict)`` pairs from directory glob and ``extra_files``."""
    paths: List[str] = list(extra_files)
    if report_dir:
        paths += sorted(glob.glob(os.path.join(report_dir, "*.json")))
    # Deduplicate, preserving order.
    seen = set()
    unique: List[str] = []
    for p in paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(p)
    result = []
    for path in unique:
        try:
            with open(path, encoding="utf-8") as fh:
                report = json.load(fh)
            result.append((path, report))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Warning: could not load {path}: {exc}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Markdown / CSV rendering
# ---------------------------------------------------------------------------


def _md_table(columns: List[str], rows: List[Dict[str, str]]) -> str:
    header = " | ".join(columns)
    sep = " | ".join(["---"] * len(columns))
    lines = [f"| {header} |", f"| {sep} |"]
    for r in rows:
        cells = " | ".join(r.get(c, "") for c in columns)
        lines.append(f"| {cells} |")
    return "\n".join(lines)


def _build_markdown(
    recombi_rows: List[Dict[str, str]],
    quant_rows: List[Dict[str, str]],
) -> str:
    lines = [
        "# Comparison Matrix Summary",
        "",
        "> Generated by `scripts/summarise_comparison_matrix.py`.",
        "",
    ]
    if recombi_rows:
        lines += [
            "## Recombination rows (child vs references)",
            "",
            "**Column guide:**",
            "- `child_vs_ref_a_agreement` / `child_vs_ref_b_agreement` — top-1 action agreement fraction.",
            "- `oracle_agreement` — fraction where child matches *at least one* reference.",
            "- `kl_ref_a` / `kl_ref_b` — KL divergence (lower = more similar).",
            "- `cosine_ref_a` / `cosine_ref_b` — mean cosine similarity on Q-logits (higher = more similar).",
            "",
            _md_table(_RECOMBI_COLUMNS, recombi_rows),
            "",
        ]
    if quant_rows:
        lines += [
            "## Quantized fidelity rows (float vs int8)",
            "",
            "**Column guide:**",
            "- `action_agreement` — fraction of states where float and int8 models agree (top-1).",
            "- `float_ms` / `quant_ms` — median single-sample inference latency in milliseconds.",
            "- `size_ratio` — on-disk size ratio (quantized / float); < 1 means smaller quantized model.",
            "",
            _md_table(_QUANT_COLUMNS, quant_rows),
            "",
        ]
    lines += [
        "## Methodology notes",
        "",
        "- All rows used the **same** state buffer (see `states` section in individual JSON reports).",
        "- Offline metrics only (action agreement, KL, MSE, MAE, cosine); no online rollout returns.",
        "- Recombination rows use `RecombinationEvaluator`; quantized-fidelity rows use `QuantizedValidator`.",
        "- For exact commands, see `docs/design/crossover_strategies.md` §10.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate comparison-matrix JSON reports into a Markdown/CSV summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--report-dir",
        default="",
        help="Directory containing *.json report files to summarise.",
    )
    p.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Additional JSON report file paths (supplemental or instead of --report-dir).",
    )
    p.add_argument(
        "--output-md",
        default="",
        help="Output Markdown path (default: <report-dir>/comparison_matrix_summary.md).",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path (default: <report-dir>/comparison_matrix_summary.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.report_dir and not args.files:
        print("Error: provide --report-dir or --files.", file=sys.stderr)
        sys.exit(1)

    loaded = _load_reports(args.report_dir, list(args.files or []))
    if not loaded:
        print("No JSON reports found.", file=sys.stderr)
        sys.exit(1)

    recombi_rows: List[Dict[str, str]] = []
    quant_rows: List[Dict[str, str]] = []
    unknown: List[str] = []

    for path, report in loaded:
        # Use matrix_row label if present (set by run_comparison_matrix.py),
        # otherwise fall back to the filename stem.
        label = report.get("matrix_row") or os.path.splitext(os.path.basename(path))[0]
        rtype = _detect_type(report)
        if rtype == "recombination":
            recombi_rows.append(_extract_recombination(label, report))
        elif rtype == "quantized_fidelity":
            quant_rows.append(_extract_quant_fidelity(label, report))
        else:
            unknown.append(path)
            print(f"  Warning: could not detect report type for {path}; skipping.", file=sys.stderr)

    if not recombi_rows and not quant_rows:
        print("No recognised reports found.", file=sys.stderr)
        sys.exit(1)

    md_text = _build_markdown(recombi_rows, quant_rows)

    # Determine output paths
    out_dir = args.report_dir or "."
    md_path = args.output_md or os.path.join(out_dir, "comparison_matrix_summary.md")
    csv_path = args.output_csv or os.path.join(out_dir, "comparison_matrix_summary.csv")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    print(f"Markdown summary: {md_path}")

    # CSV — all rows combined with a 'type' column to distinguish
    all_rows = recombi_rows + quant_rows
    all_keys: List[str] = []
    seen_keys = set()
    for r in all_rows:
        for k in r:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV summary      : {csv_path}")

    print(f"\nSummarised {len(recombi_rows)} recombination + {len(quant_rows)} quantized-fidelity report(s).")
    if unknown:
        print(f"Skipped {len(unknown)} unrecognised report(s).")


if __name__ == "__main__":
    main()
