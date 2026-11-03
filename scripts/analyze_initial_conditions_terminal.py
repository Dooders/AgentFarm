#!/usr/bin/env python3
"""Aggregate the initial-conditions terminal-state sweep.

Reads each ``baseline/arm_result.json`` written by
``scripts/run_initial_conditions_terminal.py`` and writes:

- ``runs.csv`` — one row per (cell, seed)
- ``cell_summary.csv`` — per-cell mean and 95% Student-t interval
- ``slopes.csv`` — late-window mean on overshoot, pooled within each offspring level
- ``contrasts.csv`` — offspring 8 minus offspring 5, paired by seed, at each founder level
- ``summary.md`` — the tables plus the buffer contrast
- ``population_trajectories.png`` — one panel per cell
- ``analysis.json`` — the same numbers, for the devlog

The late window is the last quarter of the recorded population series (the
runner's own definition). Overshoot is peak population minus that mean.
Oscillation amplitude is max minus min over the first 200 steps.

A directional claim counts only when its 95% interval excludes zero and the
sign agrees in at least 5 of 6 seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_initial_conditions_terminal import (
    ARM_RESULT_RELATIVE,
    CELL_PLAN_FILE,
    FOUNDER_LEVELS,
    OFFSPRING_LEVELS,
)

OSCILLATION_WINDOW = 200
REGISTERED_SEEDS = 6
SIGN_AGREEMENT_REQUIRED = 5
ALPHA = 0.05

TRANSIENT_METRICS = ("peak_population", "peak_step", "overshoot", "oscillation_amplitude")
TERMINAL_METRICS = ("final_population", "late_window_mean")
MECHANISM_METRICS = ("starvation_deaths", "mean_lifespan_at_death", "max_generation", "total_births")
CONTRAST_METRICS = ("late_window_mean", "mean_lifespan_at_death", "max_generation")
CELL_METRICS = TRANSIENT_METRICS + TERMINAL_METRICS + MECHANISM_METRICS

BUFFER_NOTE = (
    "The resource-buffer comparison moved founder resources, node count, and regeneration together. "
    "On that axis the six-seed mean populations were 84 / 96 / 108 (conservative / balanced / buffered) "
    "and scaled with the ongoing food budget, while the single-seed peaks (137 / 120 / 171 on seed 42) "
    "did not line up with the endings. This sweep holds the food budget fixed, so a slope of late-window "
    "population on overshoot is the boom's own association, not the food supply's."
)


@dataclass(frozen=True)
class RunRow:
    """Metrics for one finished (cell, seed)."""

    founder: int
    offspring: int
    seed: int
    peak_population: float
    peak_step: int
    overshoot: float
    oscillation_amplitude: float
    final_population: float
    late_window_mean: float
    starvation_deaths: float
    mean_lifespan_at_death: float
    max_generation: float
    total_births: float
    n_steps: int


@dataclass(frozen=True)
class MeanInterval:
    """Mean and two-sided Student-t interval."""

    n: int
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class CellSummary:
    founder: int
    offspring: int
    metrics: dict[str, MeanInterval]


@dataclass(frozen=True)
class SlopeResult:
    offspring: int
    n: int
    slope: float
    low: float
    high: float
    covers_zero: bool
    seeds_agreeing: int
    n_seeds: int
    directional: str | None


@dataclass(frozen=True)
class ContrastResult:
    founder: int
    metric: str
    n: int
    mean: float
    low: float
    high: float
    excludes_zero: bool
    n_positive: int
    n_negative: int
    directional: str | None


def late_window(values: Sequence[float]) -> list[float]:
    """Last quarter of a series, or the last point when the series is shorter than 4."""
    count = len(values)
    if count == 0:
        return []
    return list(values[count - max(1, count // 4) :])


def series_metrics(steps: Sequence[int], population: Sequence[float]) -> dict[str, float]:
    """Peak, overshoot, and early oscillation from the recorded population series."""
    if not population or len(steps) != len(population):
        raise ValueError("population series is empty or misaligned with steps.")
    paired = [(int(step), float(value)) for step, value in zip(steps, population)]
    peak = max(value for _, value in paired)
    peak_step = next(step for step, value in paired if value == peak)
    late = late_window([value for _, value in paired])
    late_mean = sum(late) / len(late)
    early = [value for step, value in paired if step < OSCILLATION_WINDOW]
    if not early:
        early = [value for _, value in paired[:OSCILLATION_WINDOW]]
    return {
        "peak_population": peak,
        "peak_step": peak_step,
        "late_window_mean": late_mean,
        "overshoot": peak - late_mean,
        "oscillation_amplitude": max(early) - min(early),
    }


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def run_row_from_arm(plan: Mapping[str, Any], arm: Mapping[str, Any]) -> tuple[RunRow, list[int], list[float]]:
    """Build one metrics row and keep the population trace for the figure."""
    series = arm.get("series") or {}
    steps = [int(step) for step in series.get("steps") or []]
    population = [float(value) for value in series.get("population") or []]
    derived = series_metrics(steps, population)
    summary = arm.get("summary") or {}
    row = RunRow(
        founder=int(plan["founder_initial_resources"]),
        offspring=int(plan["offspring_initial_resources"]),
        seed=int(plan["seed"]),
        peak_population=derived["peak_population"],
        peak_step=int(derived["peak_step"]),
        overshoot=derived["overshoot"],
        oscillation_amplitude=derived["oscillation_amplitude"],
        final_population=_finite(summary.get("final_population"), population[-1]),
        late_window_mean=derived["late_window_mean"],
        starvation_deaths=_finite(summary.get("starvation_deaths")),
        mean_lifespan_at_death=_finite(summary.get("mean_lifespan_at_death")),
        max_generation=_finite(summary.get("max_generation")),
        total_births=_finite(summary.get("total_births")),
        n_steps=len(population),
    )
    return row, steps, population


def discover_runs(sweep_dir: Path) -> list[tuple[RunRow, list[int], list[float]]]:
    """Load every cell whose plan and baseline ``arm_result.json`` are both present."""
    loaded: list[tuple[RunRow, list[int], list[float]]] = []
    for plan_path in sorted(sweep_dir.glob(f"founder_*/offspring_*/seed_*/{CELL_PLAN_FILE}")):
        arm_path = plan_path.parent / ARM_RESULT_RELATIVE
        if not arm_path.is_file():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        loaded.append(run_row_from_arm(plan, arm))
    return loaded


def mean_interval(values: Sequence[float], alpha: float = ALPHA) -> MeanInterval:
    """Two-sided Student-t interval. Non-finite values are dropped."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    count = len(clean)
    if count == 0:
        return MeanInterval(0, float("nan"), float("nan"), float("nan"))
    mean = sum(clean) / count
    if count < 2:
        return MeanInterval(count, mean, mean, mean)
    variance = sum((value - mean) ** 2 for value in clean) / (count - 1)
    standard_error = math.sqrt(variance / count)
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=count - 1))
    return MeanInterval(count, mean, mean - critical * standard_error, mean + critical * standard_error)


def summarize_cells(rows: Sequence[RunRow]) -> list[CellSummary]:
    """Per-cell means. Order follows the registered founder and offspring levels."""
    founders = _ordered_levels((row.founder for row in rows), FOUNDER_LEVELS)
    offsprings = _ordered_levels((row.offspring for row in rows), OFFSPRING_LEVELS)
    summaries: list[CellSummary] = []
    for founder in founders:
        for offspring in offsprings:
            group = [row for row in rows if row.founder == founder and row.offspring == offspring]
            if not group:
                continue
            metrics = {name: mean_interval([getattr(row, name) for row in group]) for name in CELL_METRICS}
            summaries.append(CellSummary(founder=founder, offspring=offspring, metrics=metrics))
    return summaries


def _ordered_levels(present: Iterable[int], preferred: Sequence[int]) -> list[int]:
    seen = list(dict.fromkeys(int(value) for value in present))
    ordered = [level for level in preferred if level in seen]
    ordered.extend(level for level in sorted(seen) if level not in ordered)
    return ordered


def _covers_zero(low: float, high: float) -> bool:
    return math.isfinite(low) and math.isfinite(high) and low <= 0.0 <= high


def _directional(low: float, high: float, positive: int, negative: int, n_seeds: int) -> str | None:
    """The registered rule: interval excludes zero and 5 of 6 seeds share the sign."""
    if n_seeds != REGISTERED_SEEDS:
        return None
    if high < 0.0 and negative >= SIGN_AGREEMENT_REQUIRED:
        return "negative"
    if low > 0.0 and positive >= SIGN_AGREEMENT_REQUIRED:
        return "positive"
    return None


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Ordinary-least-squares slope and its standard error.

    A perfect fit has a zero standard error, so the confidence interval is the
    point. ``linregress`` reports a missing error when the response does not
    vary, which would turn a slope of zero into an empty interval.
    """
    count = len(x)
    if count < 2 or len(x) != len(y):
        return float("nan"), float("nan")
    x_mean = sum(x) / count
    y_mean = sum(y) / count
    ssx = sum((value - x_mean) ** 2 for value in x)
    if ssx == 0.0:
        return float("nan"), float("nan")
    slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / ssx
    if count < 3:
        return slope, float("nan")
    intercept = y_mean - slope * x_mean
    sse = sum((y_value - (intercept + slope * x_value)) ** 2 for x_value, y_value in zip(x, y))
    standard_error = math.sqrt((sse / (count - 2)) / ssx)
    return slope, standard_error


def slope_for_offspring(rows: Sequence[RunRow], offspring: int) -> SlopeResult:
    """Pooled OLS of late-window mean on overshoot at one offspring level.

    Each (founder, seed) row is one point. Sign agreement is the sign of the
    within-seed slope across founder levels, which is the seed-level reading
    of the same association.
    """
    group = [row for row in rows if row.offspring == offspring]
    x = [row.overshoot for row in group]
    y = [row.late_window_mean for row in group]
    seeds = sorted({row.seed for row in group})
    seed_slopes: list[float] = []
    for seed in seeds:
        points = [row for row in group if row.seed == seed]
        seed_x = [row.overshoot for row in points]
        seed_y = [row.late_window_mean for row in points]
        if len(points) < 2 or len(set(seed_x)) < 2:
            seed_slopes.append(float("nan"))
            continue
        seed_slope, _seed_error = _ols_slope(seed_x, seed_y)
        seed_slopes.append(seed_slope)

    if len(group) < 3 or len(set(x)) < 2:
        nan = float("nan")
        return SlopeResult(offspring, len(group), nan, nan, nan, True, 0, len(seeds), None)

    slope, standard_error = _ols_slope(x, y)
    degrees = len(group) - 2
    critical = float(student_t.ppf(1.0 - ALPHA / 2.0, df=degrees))
    half_width = critical * standard_error
    low = slope - half_width
    high = slope + half_width
    finite_slopes = [value for value in seed_slopes if math.isfinite(value)]
    positive = sum(value > 0.0 for value in finite_slopes)
    negative = sum(value < 0.0 for value in finite_slopes)
    if slope > 0.0:
        agreeing = positive
    elif slope < 0.0:
        agreeing = negative
    else:
        agreeing = sum(value == 0.0 for value in finite_slopes)
    # Seeds whose within-founder slope is undefined do not agree.
    return SlopeResult(
        offspring=offspring,
        n=len(group),
        slope=slope,
        low=low,
        high=high,
        covers_zero=_covers_zero(low, high),
        seeds_agreeing=agreeing,
        n_seeds=len(seeds),
        directional=_directional(low, high, positive, negative, len(seeds)),
    )


def contrasts_for_metric(rows: Sequence[RunRow], metric: str) -> list[ContrastResult]:
    """Paired offspring-8 minus offspring-5 at each founder level."""
    founders = _ordered_levels((row.founder for row in rows), FOUNDER_LEVELS)
    results: list[ContrastResult] = []
    for founder in founders:
        by_seed: dict[int, dict[int, float]] = {}
        for row in rows:
            if row.founder != founder:
                continue
            by_seed.setdefault(row.seed, {})[row.offspring] = float(getattr(row, metric))
        differences = []
        for seed in sorted(by_seed):
            pair = by_seed[seed]
            if 5 in pair and 8 in pair and math.isfinite(pair[5]) and math.isfinite(pair[8]):
                differences.append(pair[8] - pair[5])
        interval = mean_interval(differences)
        positive = sum(value > 0.0 for value in differences)
        negative = sum(value < 0.0 for value in differences)
        results.append(
            ContrastResult(
                founder=founder,
                metric=metric,
                n=interval.n,
                mean=interval.mean,
                low=interval.low,
                high=interval.high,
                excludes_zero=not _covers_zero(interval.low, interval.high),
                n_positive=positive,
                n_negative=negative,
                directional=_directional(interval.low, interval.high, positive, negative, interval.n),
            )
        )
    return results


def _fmt(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _interval(stats: MeanInterval, digits: int = 1) -> str:
    return f"{_fmt(stats.mean, digits)} [{_fmt(stats.low, digits)}, {_fmt(stats.high, digits)}]"


def render_summary(
    rows: Sequence[RunRow],
    cells: Sequence[CellSummary],
    slopes: Sequence[SlopeResult],
    contrasts: Sequence[ContrastResult],
) -> str:
    """Markdown tables a reader can check without opening the CSV files."""
    lines = [
        "# Initial conditions and the terminal population",
        "",
        (
            "Baseline arm only. Food budget pinned at 60 resource nodes, medium selection pressure, "
            "population cap 300, 3,000 steps, `policy_temperature` 0.1, platform genes frozen. "
            f"{len(rows)} finished runs."
        ),
        "",
        (
            "The late-window mean is the primary endpoint. A directional claim counts only when its "
            "95% interval excludes zero and the sign agrees in at least 5 of 6 seeds."
        ),
        "",
        "## Per-cell means",
        "",
        (
            "| Founder | Offspring | Peak | Peak step | Overshoot | Early amplitude | "
            "Final | Late-window mean | Starvation deaths | Lifespan | Max generation | Births |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        metrics = cell.metrics
        lines.append(
            "| {founder} | {offspring} | {peak} | {step} | {overshoot} | {amplitude} | "
            "{final} | {late} | {starvation} | {life} | {generation} | {births} |".format(
                founder=cell.founder,
                offspring=cell.offspring,
                peak=_interval(metrics["peak_population"], 1),
                step=_interval(metrics["peak_step"], 0),
                overshoot=_interval(metrics["overshoot"], 1),
                amplitude=_interval(metrics["oscillation_amplitude"], 1),
                final=_interval(metrics["final_population"], 1),
                late=_interval(metrics["late_window_mean"], 1),
                starvation=_interval(metrics["starvation_deaths"], 0),
                life=_interval(metrics["mean_lifespan_at_death"], 1),
                generation=_interval(metrics["max_generation"], 2),
                births=_interval(metrics["total_births"], 0),
            )
        )

    lines.extend(
        [
            "",
            "## H0 — slope of late-window mean on overshoot",
            "",
            (
                "Pooled across founder levels within each offspring level. "
                "Ordinary least squares treats each (founder, seed) row as a point. "
                "Seed agreement is the sign of that seed's own slope across founder levels."
            ),
            "",
            "| Offspring | n | Slope | 95% CI | Covers zero | Seeds agreeing | Directional claim |",
            "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for slope in slopes:
        claim = slope.directional or "none"
        lines.append(
            f"| {slope.offspring} | {slope.n} | {_fmt(slope.slope, 3)} | "
            f"[{_fmt(slope.low, 3)}, {_fmt(slope.high, 3)}] | "
            f"{'yes' if slope.covers_zero else 'no'} | {slope.seeds_agreeing}/{slope.n_seeds} | {claim} |"
        )

    lines.extend(
        [
            "",
            "## H1 — offspring 8 minus offspring 5",
            "",
            "Paired by seed at each founder level.",
            "",
            "| Founder | Metric | Mean difference | 95% CI | Excludes zero | Sign counts (+/−) | Directional claim |",
            "| ---: | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for contrast in contrasts:
        claim = contrast.directional or "none"
        lines.append(
            f"| {contrast.founder} | `{contrast.metric}` | {_fmt(contrast.mean, 2)} | "
            f"[{_fmt(contrast.low, 2)}, {_fmt(contrast.high, 2)}] | "
            f"{'yes' if contrast.excludes_zero else 'no'} | "
            f"{contrast.n_positive}/{contrast.n_negative} | {claim} |"
        )

    lines.extend(["", "## Buffer contrast", "", BUFFER_NOTE, ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _mean_curve(traces: Sequence[tuple[Sequence[int], Sequence[float]]]) -> tuple[list[int], list[float]]:
    """Mean population at steps present in every trace of the cell."""
    if not traces:
        return [], []
    buckets: dict[int, list[float]] = {}
    for steps, population in traces:
        for step, value in zip(steps, population):
            buckets.setdefault(int(step), []).append(float(value))
    common = sorted(step for step, values in buckets.items() if len(values) == len(traces))
    return common, [sum(buckets[step]) / len(traces) for step in common]


def plot_trajectories(
    traces: Sequence[tuple[int, int, int, Sequence[int], Sequence[float]]],
    path: Path,
) -> None:
    """One panel per cell. Seeds are faint; the cell mean is the bold line.

    ``traces`` entries are ``(founder, offspring, seed, steps, population)``.
    """
    founders = _ordered_levels((founder for founder, *_ in traces), FOUNDER_LEVELS)
    offsprings = _ordered_levels((offspring for _, offspring, *_ in traces), OFFSPRING_LEVELS)
    if not founders or not offsprings:
        return
    figure, axes = plt.subplots(
        len(offsprings),
        len(founders),
        figsize=(3.2 * len(founders), 2.8 * len(offsprings)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, offspring in enumerate(offsprings):
        for column_index, founder in enumerate(founders):
            axis = axes[row_index][column_index]
            cell_traces = [
                (steps, population)
                for trace_founder, trace_offspring, _seed, steps, population in traces
                if trace_founder == founder and trace_offspring == offspring
            ]
            for steps, population in cell_traces:
                axis.plot(steps, population, color="0.45", alpha=0.35, linewidth=0.8)
            mean_steps, mean_population = _mean_curve(cell_traces)
            if mean_steps:
                axis.plot(mean_steps, mean_population, color="#1a365d", linewidth=2.0)
            axis.set_title(f"founder {founder}, offspring {offspring}", fontsize=9)
            if row_index == len(offsprings) - 1:
                axis.set_xlabel("step")
            if column_index == 0:
                axis.set_ylabel("population")
    figure.suptitle("Baseline population at a fixed food budget", fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _jsonable_interval(stats: MeanInterval) -> dict[str, Any]:
    return asdict(stats)


def write_outputs(
    sweep_dir: Path,
    loaded: Sequence[tuple[RunRow, list[int], list[float]]],
    devlog_figure: Path | None = None,
) -> dict[str, Any]:
    """Write the aggregate tables, the trajectory figure, and the markdown summary."""
    rows = [row for row, _steps, _population in loaded]
    traces = [
        (row.founder, row.offspring, row.seed, steps, population) for row, steps, population in loaded
    ]
    cells = summarize_cells(rows)
    offsprings = _ordered_levels((row.offspring for row in rows), OFFSPRING_LEVELS)
    slopes = [slope_for_offspring(rows, offspring) for offspring in offsprings]
    contrast_rows = [contrast for metric in CONTRAST_METRICS for contrast in contrasts_for_metric(rows, metric)]

    aggregate = sweep_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    _write_csv(aggregate / "runs.csv", [asdict(row) for row in rows], list(RunRow.__dataclass_fields__))

    cell_csv_rows = []
    for cell in cells:
        flat: dict[str, Any] = {"founder": cell.founder, "offspring": cell.offspring}
        for name, stats in cell.metrics.items():
            flat[f"{name}_n"] = stats.n
            flat[f"{name}_mean"] = stats.mean
            flat[f"{name}_low"] = stats.low
            flat[f"{name}_high"] = stats.high
        cell_csv_rows.append(flat)
    if cell_csv_rows:
        _write_csv(aggregate / "cell_summary.csv", cell_csv_rows, list(cell_csv_rows[0].keys()))

    _write_csv(aggregate / "slopes.csv", [asdict(slope) for slope in slopes], list(SlopeResult.__dataclass_fields__))
    _write_csv(
        aggregate / "contrasts.csv",
        [asdict(contrast) for contrast in contrast_rows],
        list(ContrastResult.__dataclass_fields__),
    )

    with (aggregate / "population_series.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["founder", "offspring", "seed", "step", "population"])
        for row, steps, population in loaded:
            for step, value in zip(steps, population):
                writer.writerow([row.founder, row.offspring, row.seed, int(step), value])

    figure_path = aggregate / "population_trajectories.png"
    plot_trajectories(traces, figure_path)
    if devlog_figure is not None:
        devlog_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, devlog_figure)

    summary = render_summary(rows, cells, slopes, contrast_rows)
    (aggregate / "summary.md").write_text(summary, encoding="utf-8")

    payload = {
        "n_runs": len(rows),
        "cells": [
            {
                "founder": cell.founder,
                "offspring": cell.offspring,
                "metrics": {name: _jsonable_interval(stats) for name, stats in cell.metrics.items()},
            }
            for cell in cells
        ],
        "slopes": [asdict(slope) for slope in slopes],
        "contrasts": [asdict(contrast) for contrast in contrast_rows],
    }
    (aggregate / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, default=Path("experiments/initial_conditions_terminal"))
    parser.add_argument(
        "--devlog-figure",
        type=Path,
        default=None,
        help="Also copy the trajectory figure to this path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    loaded = discover_runs(args.sweep_dir)
    if not loaded:
        raise SystemExit(f"No finished cells under {args.sweep_dir}")
    payload = write_outputs(args.sweep_dir, loaded, devlog_figure=args.devlog_figure)
    print(json.dumps({"n_runs": payload["n_runs"], "aggregate": str(args.sweep_dir / "aggregate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
