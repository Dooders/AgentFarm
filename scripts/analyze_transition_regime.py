#!/usr/bin/env python3
"""Analyze transition-regime intrinsic-evolution sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.analysis.transition_regime import (  # noqa: E402
    MechanismEvidence,
    ModeAssignment,
    TransitionProbability,
    TransitionRegimeSummary,
    TransitionRunMetrics,
    extract_transition_run_metrics,
    summarize_transition_regime,
)


MODE_COLORS = {
    "low_speciation": "#B8580E",
    "high_speciation": "#2E7D32",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def _parse_range_bins(raw_bins: Optional[Sequence[str]]) -> Optional[List[Tuple[float, float]]]:
    if not raw_bins:
        return None
    bins: List[Tuple[float, float]] = []
    for raw in raw_bins:
        if ":" in raw:
            left, right = raw.split(":", 1)
        elif "," in raw:
            left, right = raw.split(",", 1)
        else:
            value = float(raw)
            bins.append((value, value))
            continue
        lo = float(left)
        hi = float(right)
        if hi < lo:
            raise ValueError(f"range bin upper bound must be ≥ lower bound; got {raw!r}")
        bins.append((lo, hi))
    return bins


def _discover_factor_records(sweep_dir: Path) -> List[Dict[str, Any]]:
    manifest = _read_json(sweep_dir / "transition_regime_manifest.json")
    runs = manifest.get("runs")
    if isinstance(runs, list) and runs:
        return [dict(run) for run in runs]

    records: List[Dict[str, Any]] = []
    for metadata_path in sorted(sweep_dir.rglob("transition_factor_metadata.json")):
        record = _read_json(metadata_path)
        if record:
            record.setdefault("run_dir", str(metadata_path.parent))
            records.append(record)
    return records


def _load_metrics(sweep_dir: Path, parameter_name: str) -> List[TransitionRunMetrics]:
    metrics: List[TransitionRunMetrics] = []
    for record in _discover_factor_records(sweep_dir):
        run_dir_value = record.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute():
            run_dir = sweep_dir / run_dir
        record.setdefault("parameter_name", parameter_name)
        metrics.append(extract_transition_run_metrics(run_dir, record))
    return metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_mode_assignments_csv(path: Path, assignments: Sequence[ModeAssignment]) -> None:
    rows = [
        {
            "run_dir": assignment.run_dir,
            "seed": assignment.seed,
            "parameter_value": assignment.parameter_value,
            "intervention": assignment.intervention,
            "mode": assignment.mode,
            "confidence": assignment.confidence,
            "classifier": assignment.classifier,
            "score": assignment.score,
        }
        for assignment in assignments
    ]
    _write_csv(
        path,
        rows,
        [
            "run_dir",
            "seed",
            "parameter_value",
            "intervention",
            "mode",
            "confidence",
            "classifier",
            "score",
        ],
    )


def _write_probability_csv(path: Path, probabilities: Sequence[TransitionProbability]) -> None:
    rows = [
        {
            "parameter_name": probability.parameter_name,
            "parameter_range": probability.parameter_range,
            "range_min": probability.range_min,
            "range_max": probability.range_max,
            "intervention": probability.intervention,
            "mode": probability.mode,
            "n": probability.n,
            "k": probability.k,
            "p": probability.p,
            "ci95_low": probability.ci95[0],
            "ci95_high": probability.ci95[1],
        }
        for probability in probabilities
    ]
    _write_csv(
        path,
        rows,
        [
            "parameter_name",
            "parameter_range",
            "range_min",
            "range_max",
            "intervention",
            "mode",
            "n",
            "k",
            "p",
            "ci95_low",
            "ci95_high",
        ],
    )


def _plot_modes(
    metrics: Sequence[TransitionRunMetrics],
    assignments: Sequence[ModeAssignment],
    output_dir: Path,
) -> Optional[Path]:
    if not metrics or not assignments:
        return None
    by_run = {assignment.run_dir: assignment for assignment in assignments}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    for metric in metrics:
        assignment = by_run.get(metric.run_dir)
        if assignment is None:
            continue
        if math.isnan(metric.final_speciation):
            continue
        color = MODE_COLORS.get(assignment.mode, "#6B7280")
        marker = "x" if metric.intervention != "baseline" else "o"
        ax.scatter(
            metric.parameter_value,
            metric.final_speciation,
            c=color,
            marker=marker,
            s=52,
            alpha=0.85,
            edgecolors="white" if marker == "o" else color,
            linewidths=0.6,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("parameter value")
    ax.set_ylabel("final speciation index")
    ax.set_title("Transition-regime mode assignments")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = output_dir / "mode_assignments_vs_parameter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_probabilities(
    probabilities: Sequence[TransitionProbability],
    output_dir: Path,
) -> Optional[Path]:
    usable = [
        probability
        for probability in probabilities
        if probability.n > 0
        and probability.range_min is not None
        and probability.range_max is not None
        and not math.isnan(probability.p)
    ]
    if not usable:
        return None
    xs = [
        (float(probability.range_min) + float(probability.range_max)) / 2.0
        for probability in usable
    ]
    ys = [probability.p for probability in usable]
    yerr_low = [probability.p - probability.ci95[0] for probability in usable]
    yerr_high = [probability.ci95[1] - probability.p for probability in usable]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(
        xs,
        ys,
        yerr=[yerr_low, yerr_high],
        fmt="o-",
        color="#2E7D32",
        ecolor="#9CA3AF",
        capsize=4,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(usable[0].parameter_name)
    ax.set_ylabel(f"P({usable[0].mode})")
    ax.set_title("Transition probability by parameter range")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = output_dir / "transition_probability_by_parameter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_mechanisms(
    mechanisms: Sequence[MechanismEvidence],
    output_dir: Path,
) -> Optional[Path]:
    if not mechanisms:
        return None
    names = [mechanism.mechanism for mechanism in mechanisms]
    effects = [mechanism.effect_size if mechanism.effect_size == mechanism.effect_size else 0.0 for mechanism in mechanisms]
    colors = ["#2E7D32" if mechanism.supported else "#9CA3AF" for mechanism in mechanisms]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(names, effects, color=colors, alpha=0.85)
    ax.axhline(0.0, color="#374151", lw=1.0)
    ax.set_ylabel("effect size")
    ax.set_title("Candidate mechanism evidence")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = output_dir / "mechanism_evidence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _format_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _format_markdown(summary: TransitionRegimeSummary, artifacts: Dict[str, Optional[str]]) -> str:
    lines: List[str] = [
        "# Transition-Regime Analysis",
        "",
        f"- **profile**: {summary.profile}",
        f"- **parameter**: `{summary.parameter_name}`",
        f"- **runs analyzed**: {len(summary.metrics)}",
        f"- **mode counts**: {summary.mode_counts}",
        "",
    ]

    lines += ["## Mode assignments", ""]
    lines += ["| intervention | parameter | seed | mode | confidence | final speciation |", "| --- | --- | --- | --- | --- | --- |"]
    metrics_by_run = {metric.run_dir: metric for metric in summary.metrics}
    for assignment in summary.mode_assignments:
        metric = metrics_by_run.get(assignment.run_dir)
        final_spec = metric.final_speciation if metric else float("nan")
        lines.append(
            f"| {assignment.intervention} | {_format_float(assignment.parameter_value)} "
            f"| {assignment.seed} | {assignment.mode} | {_format_float(assignment.confidence)} "
            f"| {_format_float(final_spec)} |"
        )
    lines.append("")

    lines += ["## Transition probabilities", ""]
    lines += ["| range | mode | n | k | p | Wilson 95% CI |", "| --- | --- | --- | --- | --- | --- |"]
    for probability in summary.probabilities:
        lines.append(
            f"| {probability.parameter_range} | {probability.mode} | {probability.n} | {probability.k} "
            f"| {_format_float(probability.p)} | "
            f"[{_format_float(probability.ci95[0])}, {_format_float(probability.ci95[1])}] |"
        )
    lines.append("")

    lines += ["## Mechanism evidence", ""]
    lines += ["| mechanism | supported | effect size | comparison | description |", "| --- | --- | --- | --- | --- |"]
    for mechanism in summary.mechanisms:
        lines.append(
            f"| {mechanism.mechanism} | {mechanism.supported} | {_format_float(mechanism.effect_size)} "
            f"| {mechanism.comparison} | {mechanism.description} |"
        )
    lines.append("")

    lines += ["## Exit criterion", ""]
    if summary.exit_paragraph:
        lines += [summary.exit_paragraph, ""]
    else:
        lines += ["No valid exit paragraph yet.", ""]
        for reason in summary.evidence_gate_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines += ["## Artifacts", ""]
    for name, path in artifacts.items():
        lines.append(f"- **{name}**: {path or 'not produced'}")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a transition-regime sweep produced by run_transition_regime_experiment.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--parameter-name", type=str, default="initial_agent_resource_level")
    parser.add_argument("--mode-features", nargs="+", default=None)
    parser.add_argument(
        "--range-bins",
        nargs="+",
        default=None,
        help="Parameter bins as lo:hi, lo,hi, or exact values. Defaults to exact observed values.",
    )
    parser.add_argument("--min-runs-per-range", type=int, default=6)
    parser.add_argument("--min-mode-count", type=int, default=2)
    parser.add_argument("--mechanism-effect-threshold", type=float, default=0.5)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "transition_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    range_bins = _parse_range_bins(args.range_bins)
    metrics = _load_metrics(sweep_dir, args.parameter_name)
    if not metrics:
        print(f"No transition-regime runs found under {sweep_dir}", file=sys.stderr)
        return 1

    summary_kwargs: Dict[str, Any] = {
        "parameter_name": args.parameter_name,
        "range_bins": range_bins,
        "min_runs_per_range": args.min_runs_per_range,
        "min_mode_count": args.min_mode_count,
        "mechanism_effect_threshold": args.mechanism_effect_threshold,
    }
    if args.mode_features:
        summary_kwargs["mode_features"] = args.mode_features
    summary = summarize_transition_regime(metrics, **summary_kwargs)

    summary_path = output_dir / "transition_regime_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8")

    mode_csv = output_dir / "mode_assignments.csv"
    _write_mode_assignments_csv(mode_csv, summary.mode_assignments)
    probability_csv = output_dir / "transition_probability_by_parameter.csv"
    _write_probability_csv(probability_csv, summary.probabilities)

    artifacts: Dict[str, Optional[str]] = {
        "mode_assignments_vs_parameter": None,
        "transition_probability_by_parameter": None,
        "mechanism_evidence": None,
        "mode_assignments_csv": str(mode_csv),
        "transition_probability_csv": str(probability_csv),
        "summary_json": str(summary_path),
        "exit_paragraph": None,
    }
    path = _plot_modes(summary.metrics, summary.mode_assignments, output_dir)
    artifacts["mode_assignments_vs_parameter"] = str(path) if path else None
    path = _plot_probabilities(summary.probabilities, output_dir)
    artifacts["transition_probability_by_parameter"] = str(path) if path else None
    path = _plot_mechanisms(summary.mechanisms, output_dir)
    artifacts["mechanism_evidence"] = str(path) if path else None

    exit_path = output_dir / "exit_paragraph.txt"
    if summary.exit_paragraph:
        exit_path.write_text(summary.exit_paragraph + "\n", encoding="utf-8")
        artifacts["exit_paragraph"] = str(exit_path)
    elif exit_path.exists():
        exit_path.unlink()

    markdown_path = output_dir / "transition_regime_summary.md"
    markdown_path.write_text(_format_markdown(summary, artifacts), encoding="utf-8")

    print(f"Transition-regime analysis written to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Mode assignments: {mode_csv}")
    print(f"Probabilities: {probability_csv}")
    if summary.exit_paragraph:
        print(f"Exit paragraph: {exit_path}")
    else:
        print("Exit paragraph: not generated")
        for reason in summary.evidence_gate_reasons:
            print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
