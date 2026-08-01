#!/usr/bin/env python3
"""Combine intrinsic-goals summaries across selection-pressure levels.

Reads the ``intrinsic_goals_summary.json`` produced by each pressure level of
:mod:`scripts.run_intrinsic_goals_pressure_sweep` (or three standalone
``run_intrinsic_goals_experiment`` runs) and folds them into a single
cross-pressure comparison so the question in issue #892 is answerable at a
glance:

    Does the strength of selection determine whether maladaptive/random
    objectives get purged, and does it modulate the population-suppression
    effect?

Outputs (written to ``--out-dir``, default ``--sweep-dir``):

- ``combined_comparison.json`` — one entry per pressure with the paired
  ``unique - uniform`` deltas (mean, 95% CI, Cohen's dz) for the population and
  action-mix metrics, plus per-arm goal-diversity start/end sums.
- ``combined_comparison.md`` — the same data as human-readable tables.
- ``intrinsic_goals_pressure_sweep.png`` — a small figure of the paired
  population deltas, end goal-diversity, and gather-share delta vs pressure
  (skipped gracefully if matplotlib is unavailable).

Example::

    python scripts/analyze_intrinsic_goals_pressure_sweep.py --sweep-dir experiments

    # Explicit summary paths (any labels/order)
    python scripts/analyze_intrinsic_goals_pressure_sweep.py \
        --summary low=experiments/intrinsic_goals_sweep_low/intrinsic_goals_summary.json \
        --summary high=experiments/intrinsic_goals_sweep_high/intrinsic_goals_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.hyperparameter_chromosome import (  # noqa: E402
    INTRINSIC_REWARD_GENE_NAMES,
    default_hyperparameter_chromosome,
)

DEFAULT_PRESSURES: Tuple[str, ...] = ("low", "medium", "high")
SWEEP_DIR_TEMPLATE = "intrinsic_goals_sweep_{pressure}"
SUMMARY_FILENAME = "intrinsic_goals_summary.json"

# The three experiment arms, in display order.
ARMS: Tuple[str, ...] = ("uniform", "shared", "unique")

# Paired contrasts, each mapping the ``paired_deltas`` key to a human label.
# The decomposition: shared-uniform isolates the mean-shift off the tuned
# default, unique-shared isolates pure heterogeneity, unique-uniform is the
# total effect (the original headline).
CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("unique_minus_uniform", "unique - uniform (total)"),
    ("shared_minus_uniform", "shared - uniform (mean-shift off default)"),
    ("unique_minus_shared", "unique - shared (heterogeneity)"),
)

# Population/vital metrics reported as paired deltas.
POPULATION_METRICS: Tuple[str, ...] = (
    "mean_population",
    "final_population",
    "peak_population",
    "total_births",
    "total_deaths",
)

# The single action-share delta the issue calls out ("does the +gather shift
# weaken as selection purges the bad goals?"). Others remain in the JSON.
GATHER_METRIC = "action_share[gather]"


class SummaryError(ValueError):
    """Raised when a summary file is missing required aggregate fields."""


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise SummaryError(f"{path}: expected a JSON object at the top level.")
    if not summary.get("aggregate"):
        raise SummaryError(
            f"{path}: no 'aggregate' block. The pressure sweep needs "
            "multi-replicate runs (--num-replicates > 1) so paired statistics "
            "are available."
        )
    return summary


def _paired(
    summary: Dict[str, Any], contrast: str, metric: str
) -> Optional[Dict[str, Any]]:
    """Return the paired-delta record for *metric* under *contrast*, or ``None``."""
    paired = summary["aggregate"].get("paired_deltas", {})
    block = paired.get(contrast, {})
    record = block.get(metric)
    if record is None:
        return None
    ci = record.get("ci95", [None, None])
    return {
        "delta_mean": record.get("delta_mean"),
        "ci95": ci,
        "cohen_dz": record.get("cohen_dz"),
        "p_value": record.get("p_value"),
        "significant_p05": record.get("significant_p05"),
    }


def _diversity_sums(summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Mean (over replicates) of summed goal-diversity std per arm, start & end.

    Note: this raw summed metric mixes genes with very different scales and is
    dominated (~80%) by ``reward_death_penalty`` (range [0, 50]).  See
    :func:`_normalized_diversity` for the per-gene, span-normalized view.
    """
    replicates = summary.get("replicates", [])
    out: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        starts: List[float] = []
        ends: List[float] = []
        for rep in replicates:
            arm_summary = rep.get(arm, {})
            starts.append(sum(arm_summary.get("goal_diversity_start", {}).values()))
            ends.append(sum(arm_summary.get("goal_diversity_end", {}).values()))
        n = len(replicates) if replicates else 1
        out[arm] = {
            "start_sum": sum(starts) / n if starts else 0.0,
            "end_sum": sum(ends) / n if ends else 0.0,
        }
    return out


def _gene_spans() -> Dict[str, float]:
    """Span (max - min) of each intrinsic-goal gene, used to normalize std."""
    chromosome = default_hyperparameter_chromosome()
    spans: Dict[str, float] = {}
    for name in INTRINSIC_REWARD_GENE_NAMES:
        gene = chromosome.get_gene(name)
        if gene is None:
            continue
        span = gene.max_value - gene.min_value
        spans[name] = span if span > 0 else 1.0
    return spans


def _normalized_diversity(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Per-gene goal diversity normalized by each gene's span (std / span).

    Because a uniform draw on ``[a, b]`` has std ``(b - a) / sqrt(12)``, every
    gene starts near ``1/sqrt(12) ≈ 0.289`` on this normalized scale regardless
    of its range.  This makes purging in *any* gene visible, unlike the raw
    summed metric that ``reward_death_penalty`` dominates.

    Returns ``per_gene`` (unique-arm start/end per gene) and
    ``mean_across_genes`` (mean normalized std per arm, start & end).
    """
    replicates = summary.get("replicates", [])
    spans = _gene_spans()
    n = len(replicates) if replicates else 1

    def _mean_norm(arm: str, phase_key: str, gene: str) -> float:
        vals: List[float] = []
        for rep in replicates:
            per_gene = rep.get(arm, {}).get(phase_key, {})
            if gene in per_gene:
                vals.append(per_gene[gene] / spans[gene])
        return sum(vals) / len(vals) if vals else 0.0

    per_gene_unique: Dict[str, Dict[str, float]] = {
        gene: {
            "start": _mean_norm("unique", "goal_diversity_start", gene),
            "end": _mean_norm("unique", "goal_diversity_end", gene),
        }
        for gene in spans
    }

    mean_across_genes: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        starts = [_mean_norm(arm, "goal_diversity_start", g) for g in spans]
        ends = [_mean_norm(arm, "goal_diversity_end", g) for g in spans]
        mean_across_genes[arm] = {
            "start": sum(starts) / len(starts) if starts else 0.0,
            "end": sum(ends) / len(ends) if ends else 0.0,
        }

    return {
        "per_gene_unique": per_gene_unique,
        "mean_across_genes": mean_across_genes,
        "num_replicates": n,
    }


def _per_arm_mean(summary: Dict[str, Any], metric: str, arm: str) -> Optional[float]:
    per_arm = summary["aggregate"].get("per_arm", {}).get(arm, {})
    record = per_arm.get(metric)
    return record.get("mean") if record else None


def analyze_summary(pressure: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a single per-pressure summary into a cross-pressure entry."""
    aggregate = summary["aggregate"]
    diversity = _diversity_sums(summary)
    normalized = _normalized_diversity(summary)
    entry: Dict[str, Any] = {
        "pressure": pressure,
        "num_replicates": aggregate.get("num_replicates"),
        "seeds": aggregate.get("seeds"),
        "contrasts": {
            contrast: {
                "population_deltas": {
                    metric: _paired(summary, contrast, metric)
                    for metric in POPULATION_METRICS
                },
                "gather_share_delta": _paired(summary, contrast, GATHER_METRIC),
            }
            for contrast, _label in CONTRASTS
        },
        "goal_diversity": diversity,
        "normalized_diversity": normalized,
        # Per-arm end-diversity sum straight from the aggregate (mirrors
        # goal_diversity["*"]["end_sum"] but sourced from the pre-aggregated
        # goal_diversity_end_sum metric when present).
        "goal_diversity_end_sum_per_arm": {
            arm: _per_arm_mean(summary, "goal_diversity_end_sum", arm) for arm in ARMS
        },
    }
    return entry


def _resolve_summary_paths(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Return ordered (pressure_label, summary_path) pairs from CLI args."""
    if args.summary:
        pairs: List[Tuple[str, str]] = []
        for item in args.summary:
            if "=" not in item:
                raise SystemExit(
                    f"--summary must be LABEL=PATH; got {item!r}."
                )
            label, path = item.split("=", 1)
            pairs.append((label.strip(), path.strip()))
        return pairs
    return [
        (
            pressure,
            os.path.join(
                args.sweep_dir,
                SWEEP_DIR_TEMPLATE.format(pressure=pressure),
                SUMMARY_FILENAME,
            ),
        )
        for pressure in args.pressures
    ]


def collect_entries(
    pairs: List[Tuple[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Load and analyze each summary; return (entries, skipped-with-reason)."""
    entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for pressure, path in pairs:
        if not os.path.isfile(path):
            skipped.append({"pressure": pressure, "path": path, "reason": "missing"})
            continue
        try:
            summary = _load_summary(path)
        except (SummaryError, json.JSONDecodeError, OSError) as exc:
            skipped.append({"pressure": pressure, "path": path, "reason": str(exc)})
            continue
        entries.append(analyze_summary(pressure, summary))
    return entries, skipped


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _delta_cell(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return "n/a"
    lo, hi = record.get("ci95", [None, None])
    star = "*" if record.get("significant_p05") else ""
    return (
        f"{_fmt(record.get('delta_mean'))}{star} "
        f"[{_fmt(lo)}, {_fmt(hi)}] (dz={_fmt(record.get('cohen_dz'))})"
    )


def render_markdown(entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Intrinsic goals: selection-pressure sweep", ""]
    if not entries:
        lines.append("_No summaries found._")
        return "\n".join(lines) + "\n"

    sep = "|---" * (len(entries) + 1) + "|"
    pressure_header = "| Metric | " + " | ".join(e["pressure"] for e in entries) + " |"

    lines.append(
        "Paired deltas are `treatment - baseline` per seed "
        "(mean [95% CI] (Cohen's dz)); `*` marks p < 0.05. Contrasts decompose "
        "the effect: `shared - uniform` is the mean-shift off the tuned default, "
        "`unique - shared` is pure goal heterogeneity, and `unique - uniform` is "
        "the total."
    )
    lines.append("")

    for contrast, label in CONTRASTS:
        lines.append(f"## {label}")
        lines.append("")
        lines.append(pressure_header)
        lines.append(sep)
        for metric in POPULATION_METRICS:
            row = [metric]
            for e in entries:
                block = e["contrasts"].get(contrast, {})
                row.append(_delta_cell(block.get("population_deltas", {}).get(metric)))
            lines.append("| " + " | ".join(row) + " |")
        gather_row = ["gather-share Δ"]
        for e in entries:
            block = e["contrasts"].get(contrast, {})
            gather_row.append(_delta_cell(block.get("gather_share_delta")))
        lines.append("| " + " | ".join(gather_row) + " |")
        lines.append("")

    lines.append("## Goal diversity (raw summed population std) by arm")
    lines.append("")
    lines.append(
        "_Raw sum across genes; dominated (~80%) by `reward_death_penalty` "
        "(range [0, 50]). See the span-normalized tables below._"
    )
    lines.append("")
    div_header = "| Arm / phase | " + " | ".join(e["pressure"] for e in entries) + " |"
    lines.append(div_header)
    lines.append(sep)
    for arm in ARMS:
        for phase in ("start_sum", "end_sum"):
            row = [f"{arm} {phase.replace('_sum', '')}"]
            for e in entries:
                row.append(_fmt(e["goal_diversity"][arm][phase]))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Goal diversity (span-normalized, mean across genes) by arm")
    lines.append("")
    lines.append(
        "_Each gene's std divided by its range before averaging, so a fresh "
        "uniform draw sits near 1/sqrt(12) ≈ 0.29 for every gene._"
    )
    lines.append("")
    lines.append(div_header)
    lines.append(sep)
    for arm in ARMS:
        for phase in ("start", "end"):
            row = [f"{arm} {phase}"]
            for e in entries:
                row.append(
                    _fmt(e["normalized_diversity"]["mean_across_genes"][arm][phase])
                )
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-gene goal diversity (span-normalized, unique arm)")
    lines.append("")
    lines.append(
        "_Start is shared across pressures (random init). A gene whose end "
        "value falls well below ~0.29 is being purged._"
    )
    lines.append("")
    gene_header = (
        "| Gene | start | "
        + " | ".join(f"end ({e['pressure']})" for e in entries)
        + " |"
    )
    lines.append(gene_header)
    lines.append("|---" * (len(entries) + 2) + "|")
    for gene in INTRINSIC_REWARD_GENE_NAMES:
        first = entries[0]["normalized_diversity"]["per_gene_unique"].get(gene, {})
        row = [gene.replace("reward_", ""), _fmt(first.get("start"))]
        for e in entries:
            per_gene = e["normalized_diversity"]["per_gene_unique"].get(gene, {})
            row.append(_fmt(per_gene.get("end")))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _maybe_plot(entries: List[Dict[str, Any]], out_path: str) -> Optional[str]:
    if not entries:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Plot skipped: {exc}", file=sys.stderr)
        return None

    pressures = [e["pressure"] for e in entries]
    x = np.arange(len(pressures))
    contrast_colors = {
        "unique_minus_uniform": "#d62728",
        "shared_minus_uniform": "#ff7f0e",
        "unique_minus_shared": "#9467bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _mean_err(rec: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        if rec and rec.get("delta_mean") is not None:
            lo, hi = rec.get("ci95", [rec["delta_mean"], rec["delta_mean"]])
            err = (hi - lo) / 2.0 if lo is not None and hi is not None else 0.0
            return rec["delta_mean"], err
        return np.nan, 0.0

    # 1) Total (unique - uniform) population deltas vs pressure.
    ax = axes[0]
    plot_metrics = ("mean_population", "final_population", "peak_population")
    for metric in plot_metrics:
        means, errs = [], []
        for e in entries:
            block = e["contrasts"].get("unique_minus_uniform", {})
            m, err = _mean_err(block.get("population_deltas", {}).get(metric))
            means.append(m)
            errs.append(err)
        ax.errorbar(x, means, yerr=errs, marker="o", capsize=4, label=metric)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pressures)
    ax.set_title("Total population Δ (unique − uniform)\nvs selection pressure")
    ax.set_xlabel("selection pressure")
    ax.set_ylabel("Δ population")
    ax.legend(fontsize=8)

    # 2) Span-normalized mean diversity vs pressure (end), per arm, with the
    #    fresh-random-draw reference line at 1/sqrt(12).
    ax = axes[1]
    arm_colors = {"uniform": "#1f77b4", "shared": "#ff7f0e", "unique": "#d62728"}
    for arm in ARMS:
        end = [
            e["normalized_diversity"]["mean_across_genes"][arm]["end"] for e in entries
        ]
        ax.plot(x, end, marker="o", color=arm_colors[arm], label=f"end ({arm})")
    ax.axhline(
        1.0 / np.sqrt(12.0),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.0,
        label="fresh uniform draw",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(pressures)
    ax.set_ylim(bottom=0.0)
    ax.set_title("Goal diversity (span-normalized)\nmean across genes, end")
    ax.set_xlabel("selection pressure")
    ax.set_ylabel("mean std / gene span")
    ax.legend(fontsize=8)

    # 3) Gather-share delta vs pressure, decomposed by contrast.
    ax = axes[2]
    for contrast, _label in CONTRASTS:
        means, errs = [], []
        for e in entries:
            block = e["contrasts"].get(contrast, {})
            m, err = _mean_err(block.get("gather_share_delta"))
            means.append(m)
            errs.append(err)
        ax.errorbar(
            x,
            means,
            yerr=errs,
            marker="o",
            capsize=4,
            color=contrast_colors[contrast],
            label=contrast.replace("_", " "),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pressures)
    ax.set_title("Gather-share Δ by contrast\nvs selection pressure")
    ax.set_xlabel("selection pressure")
    ax.set_ylabel("Δ fraction of agents gathering")
    ax.legend(fontsize=8)

    fig.suptitle("Intrinsic goals: selection-pressure sweep", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine intrinsic-goals summaries across selection-pressure "
            "levels into a single cross-pressure comparison (table + figure)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="experiments",
        help=(
            "Base directory containing intrinsic_goals_sweep_<pressure>/ "
            "subdirectories (used when --summary is not given)."
        ),
    )
    parser.add_argument(
        "--pressures",
        nargs="+",
        default=list(DEFAULT_PRESSURES),
        metavar="PRESSURE",
        help="Pressure labels to look for under --sweep-dir.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Explicit LABEL=PATH to a summary JSON. Repeatable; overrides "
            "--sweep-dir/--pressures discovery."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write combined outputs (default: --sweep-dir).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = args.out_dir or args.sweep_dir
    os.makedirs(out_dir, exist_ok=True)

    pairs = _resolve_summary_paths(args)
    entries, skipped = collect_entries(pairs)

    for item in skipped:
        print(
            f"Skipped {item['pressure']} ({item['path']}): {item['reason']}",
            file=sys.stderr,
        )

    if not entries:
        print("No usable summaries found; nothing to combine.", file=sys.stderr)
        return 1

    combined = {
        "pressures": [e["pressure"] for e in entries],
        "skipped": skipped,
        "entries": entries,
    }
    json_path = os.path.join(out_dir, "combined_comparison.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, default=str)

    md_path = os.path.join(out_dir, "combined_comparison.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(entries))

    figure_path = _maybe_plot(
        entries, os.path.join(out_dir, "intrinsic_goals_pressure_sweep.png")
    )

    print(f"Combined JSON : {json_path}")
    print(f"Combined table: {md_path}")
    print(f"Figure        : {figure_path or 'skipped'}")
    print(f"Pressures      : {combined['pressures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
