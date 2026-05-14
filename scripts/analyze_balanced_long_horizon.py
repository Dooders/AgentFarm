#!/usr/bin/env python3
"""Aggregate long-horizon balanced-profile runs.

Reads ``stable_balanced/seed_*`` directories produced by
``scripts/run_balanced_long_horizon_experiment.py`` (same layout as the general
stable-profile sweep). Emphasises:

- Speciation slope over the **full** logged horizon vs a **late window**
  (default: last 1000 steps, matching the original short-run length).
- **Final GMM cluster count** (``n_clusters`` from the last snapshot row).

Outputs (under ``<sweep-dir>/aggregate`` by default)::

    long_horizon_summary.json
    long_horizon_summary.md
    speciation_trajectories.png
    final_n_clusters_bar.png

Usage
-----
::

    python scripts/analyze_balanced_long_horizon.py \\
        --sweep-dir experiments/balanced_long_horizon

    # Compare CI widths to the short balanced cohort in another directory
    python scripts/analyze_balanced_long_horizon.py \\
        --sweep-dir experiments/balanced_long_horizon \\
        --compare-sweep-dir experiments/stable_profile_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    PROFILE_COLORS,
    _classify_speciation_direction,
    _discover_runs,
    _extract_run_metrics,
    _fmt,
    _mean_trajectory,
    _read_jsonl,
    _speciation_slope,
    _summarise,
)

PROFILE = "balanced"


def _late_window_bounds(max_step: float, window_steps: int) -> Tuple[float, float]:
    """Inclusive step range for the late window (length ``window_steps``)."""
    if math.isnan(max_step) or window_steps < 1:
        return float("nan"), float("nan")
    hi = float(max_step)
    lo = hi - float(window_steps) + 1.0
    return lo, hi


def _speciation_slope_window(
    trajectory: List[Dict[str, Any]],
    step_low: float,
    step_high: float,
) -> float:
    """Slope of linear fit on speciation_index vs step (per 100 steps), windowed."""
    if math.isnan(step_low) or math.isnan(step_high) or step_high < step_low:
        return float("nan")
    pairs = [
        (float(row["step"]), float(row["speciation_index"]))
        for row in trajectory
        if "speciation_index" in row
        and row.get("speciation_index") is not None
        and step_low <= float(row["step"]) <= step_high
    ]
    if len(pairs) < 2:
        return float("nan")
    steps = np.array([p[0] for p in pairs])
    vals = np.array([p[1] for p in pairs])
    try:
        slope = float(np.polyfit(steps, vals, 1)[0]) * 100.0
    except (np.linalg.LinAlgError, ValueError):
        slope = float("nan")
    return slope


def _final_n_clusters_from_trajectory(trajectory: List[Dict[str, Any]]) -> Optional[int]:
    for row in reversed(trajectory):
        sq = row.get("speciation_quality")
        if isinstance(sq, dict) and sq.get("n_clusters") is not None:
            try:
                return int(sq["n_clusters"])
            except (TypeError, ValueError):
                continue
    return None


def _final_n_clusters_from_lineage(run_dir: Path) -> Optional[int]:
    path = run_dir / "cluster_lineage.jsonl"
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    if not rows:
        return None
    max_step = None
    for row in rows:
        if "step" not in row:
            continue
        try:
            s = int(row["step"])
        except (TypeError, ValueError):
            continue
        max_step = s if max_step is None else max(max_step, s)
    if max_step is None:
        return None
    return sum(1 for row in rows if row.get("step") == max_step)


def _final_n_clusters(run_dir: Path, trajectory: List[Dict[str, Any]]) -> float:
    nc = _final_n_clusters_from_trajectory(trajectory)
    if nc is not None:
        return float(nc)
    nc2 = _final_n_clusters_from_lineage(run_dir)
    if nc2 is not None:
        return float(nc2)
    return float("nan")


def _max_logged_step(trajectory: List[Dict[str, Any]]) -> float:
    steps = [float(r["step"]) for r in trajectory if "step" in r and r["step"] is not None]
    return max(steps) if steps else float("nan")


def _extract_long_horizon_metrics(
    run_dir: Path,
    *,
    window_steps: int,
) -> Optional[Dict[str, Any]]:
    """Per-run metrics including late-window slope and final cluster count."""
    trajectory = _read_jsonl(run_dir / "intrinsic_gene_trajectory.jsonl")
    if not trajectory:
        return None

    base = _extract_run_metrics(run_dir)
    if base is None:
        return None

    max_step = _max_logged_step(trajectory)
    lo, hi = _late_window_bounds(max_step, window_steps)
    slope_full = _speciation_slope(trajectory)
    slope_late = _speciation_slope_window(trajectory, lo, hi)

    n_clusters_final = _final_n_clusters(run_dir, trajectory)

    return {
        **base,
        "max_logged_step": max_step,
        "late_window_step_low": lo,
        "late_window_step_high": hi,
        "late_window_steps_configured": window_steps,
        "speciation_slope_full": slope_full,
        "speciation_slope_late": slope_late,
        "speciation_direction_full": _classify_speciation_direction(slope_full),
        "speciation_direction_late": _classify_speciation_direction(slope_late),
        "n_clusters_final": n_clusters_final,
        "trajectory": trajectory,
    }


def _scalar_values(runs: Sequence[Dict[str, Any]], key: str) -> List[float]:
    return [float(r[key]) for r in runs if key in r and not math.isnan(float(r[key]))]


def _cluster_frequency_table(values: Sequence[float]) -> Dict[str, Any]:
    ints = [int(v) for v in values if not math.isnan(v)]
    if not ints:
        return {"counts": {}, "mode": None, "n": 0}
    ctr = Counter(ints)
    mode_val, mode_count = ctr.most_common(1)[0]
    return {
        "counts": {str(k): v for k, v in sorted(ctr.items())},
        "mode": mode_val,
        "mode_count": mode_count,
        "n": len(ints),
    }


def _plot_speciation_trajectories(runs: List[Dict[str, Any]], output: Path) -> Optional[Path]:
    if not runs:
        return None
    color = PROFILE_COLORS.get(PROFILE, "#333333")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    traces: List[Tuple[np.ndarray, np.ndarray]] = []

    for run in runs:
        traj = run.get("trajectory", [])
        spec = [
            (float(r["step"]), float(r["speciation_index"]))
            for r in traj
            if "speciation_index" in r and r["speciation_index"] is not None
        ]
        if not spec:
            continue
        steps = np.array([s[0] for s in spec])
        vals = np.array([s[1] for s in spec])
        ax.plot(steps, vals, color=color, alpha=0.4, lw=1.2)
        traces.append((steps, vals))

    if len(traces) > 1:
        mean = _mean_trajectory(traces)
        if mean is not None:
            grid, mean_trace = mean
            ax.plot(grid, mean_trace, color=color, lw=2.5, label="mean")

    ax.set_title(f"{PROFILE} — speciation index (long horizon)", fontsize=12, color=color, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("speciation index")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output / "speciation_trajectories.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_n_clusters_bar(runs: List[Dict[str, Any]], output: Path) -> Optional[Path]:
    if not runs:
        return None
    seeds = [int(r["seed"]) for r in runs]
    ks = [float(r["n_clusters_final"]) for r in runs]
    if all(math.isnan(k) for k in ks):
        return None

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    x = np.arange(len(seeds))
    bars = ax.bar(x, [k if not math.isnan(k) else 0.0 for k in ks], color=PROFILE_COLORS.get(PROFILE, "#666"))
    patch_list = list(bars.patches) if hasattr(bars, "patches") else list(bars)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("final n_clusters (GMM k)")
    ax.set_title("Final cluster count by seed", fontsize=12)
    for rect, k in zip(patch_list, ks):
        if not math.isnan(k):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                str(int(k)),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output / "final_n_clusters_bar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _ci_width(agg: Dict[str, Any]) -> float:
    lo, hi = agg["ci95"]
    if any(math.isnan(float(x)) for x in (lo, hi)):
        return float("nan")
    return float(hi) - float(lo)


def _build_markdown(
    *,
    window_steps: int,
    primary_label: str,
    compare_label: Optional[str],
    per_seed: List[Dict[str, Any]],
    summaries: Dict[str, Dict[str, Any]],
    cluster_info: Dict[str, Any],
    compare_summaries: Optional[Dict[str, Dict[str, Any]]],
    artifacts: Dict[str, Optional[str]],
) -> str:
    lines: List[str] = [
        "# Balanced long-horizon — aggregated results",
        "",
        f"**Primary sweep:** `{primary_label}`",
        "",
        f"- Late window: **last {window_steps} logged steps** (inclusive end step).",
        "  If ``num_steps`` ≤ window length, late-window slope equals the full-run slope.",
        "",
    ]
    if compare_label:
        lines.append(f"- **Comparison sweep:** `{compare_label}` (balanced seeds only).")
        lines.append("")

    lines += ["## Per-seed metrics", "", "| seed | spec_final | slope_full | slope_late | dir_full | dir_late | n_clusters | pop_final |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in sorted(per_seed, key=lambda r: r["seed"]):
        lines.append(
            f"| {row['seed']} "
            f"| {_fmt(row['speciation_final'])} "
            f"| {_fmt(row['speciation_slope_full'], 4)} "
            f"| {_fmt(row['speciation_slope_late'], 4)} "
            f"| {row['speciation_direction_full']} "
            f"| {row['speciation_direction_late']} "
            f"| {_fmt(row['n_clusters_final'], 0)} "
            f"| {_fmt(row['population_final'], 1)} |"
        )
    lines.append("")

    lines += ["## Summaries (primary)", ""]
    sf = summaries["speciation_final"]
    slf = summaries["speciation_slope_full"]
    sll = summaries["speciation_slope_late"]
    lines += [
        "| Metric | mean | 95% CI | CI width |",
        "| --- | --- | --- | --- |",
        (
            f"| Final speciation index | {_fmt(sf['mean'])} "
            f"| {_fmt(sf['ci95'])} | {_fmt(_ci_width(sf), 4)} |"
        ),
        (
            f"| Slope full (/100 steps) | {_fmt(slf['mean'], 4)} "
            f"| {_fmt(slf['ci95'], 4)} | {_fmt(_ci_width(slf), 4)} |"
        ),
        (
            f"| Slope late (/100 steps) | {_fmt(sll['mean'], 4)} "
            f"| {_fmt(sll['ci95'], 4)} | {_fmt(_ci_width(sll), 4)} |"
        ),
        "",
    ]

    lines.append("### Final cluster counts (discrete)")
    lines.append("")
    lines.append(f"- Frequencies: `{cluster_info.get('counts', {})}`")
    lines.append(f"- Mode: **{cluster_info.get('mode')}** ({cluster_info.get('mode_count')} / {cluster_info.get('n')} seeds)")
    lines.append("")

    lines.append("## Interpretation hints (wiki falsifiable targets)")
    lines.append("")
    lines.append(
        "- **Narrowing dispersion:** late-window and full-run CIs on final speciation tighten vs a "
        "1000-step balanced cohort → consistent with one attractor and slow convergence."
    )
    lines.append(
        "- **Persistent / widening dispersion:** CIs stay wide or grow → consistent with a "
        "multi-modal regime or unresolved mode-switching at the longer horizon."
    )
    lines.append(
        "- **Cluster counts:** a single dominant ``n_clusters`` across seeds vs a spread indicates "
        "whether runs agree on structural clustering at the final snapshot."
    )
    lines.append("")

    if compare_summaries:
        lines.append("## Comparison — CI widths (balanced)")
        lines.append("")
        psf = summaries["speciation_final"]
        csf = compare_summaries["speciation_final"]
        plf = summaries["speciation_slope_late"]
        clf = compare_summaries["speciation_slope_late"]
        lines += [
            "| Metric | primary CI width | compare CI width |",
            "| --- | --- | --- |",
            (
                f"| Final speciation | {_fmt(_ci_width(psf), 4)} "
                f"| {_fmt(_ci_width(csf), 4)} |"
            ),
            (
                f"| Late-window slope | {_fmt(_ci_width(plf), 4)} "
                f"| {_fmt(_ci_width(clf), 4)} |"
            ),
            "",
            "*Compare sweep uses the same late-window definition; if its horizon is shorter than "
            "the window, late slope equals the full-run slope for that directory.*",
            "",
        ]

    lines.append("## Artifacts")
    lines.append("")
    for name, path in artifacts.items():
        lines.append(f"- **{name}:** {path or 'not produced'}")
    lines.append("")
    return "\n".join(lines)


def _load_balanced_runs(sweep_dir: Path, window_steps: int) -> List[Dict[str, Any]]:
    run_map = _discover_runs(sweep_dir, [PROFILE])
    if PROFILE not in run_map:
        return []
    runs: List[Dict[str, Any]] = []
    for seed, run_dir in sorted(run_map[PROFILE], key=lambda t: t[0]):
        metrics = _extract_long_horizon_metrics(run_dir, window_steps=window_steps)
        if metrics is None:
            print(f"  Skipping {run_dir} — trajectory missing or empty.")
            continue
        metrics["seed"] = seed
        runs.append(metrics)
        print(
            f"  [balanced] seed={seed}: spec_final={_fmt(metrics['speciation_final'])}, "
            f"slope_late={_fmt(metrics['speciation_slope_late'], 4)}, "
            f"n_k={_fmt(metrics['n_clusters_final'], 0)}"
        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate balanced long-horizon intrinsic-evolution runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Directory produced by run_balanced_long_horizon_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write aggregates (default: <sweep-dir>/aggregate).",
    )
    parser.add_argument(
        "--late-window-steps",
        type=int,
        default=1000,
        help="Number of logged steps in the late window (inclusive of final step).",
    )
    parser.add_argument(
        "--compare-sweep-dir",
        type=str,
        default=None,
        help=(
            "Optional second sweep directory; only ``stable_balanced`` seeds are loaded "
            "for CI-width comparison (e.g. short 1000-step cohort)."
        ),
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {sweep_dir} …")
    runs = _load_balanced_runs(sweep_dir, window_steps=args.late_window_steps)
    if not runs:
        print("No balanced runs found.", file=sys.stderr)
        return 1
    print(f"Loaded {len(runs)} run(s).")

    summaries = {
        "speciation_final": _summarise(_scalar_values(runs, "speciation_final")),
        "speciation_slope_full": _summarise(_scalar_values(runs, "speciation_slope_full")),
        "speciation_slope_late": _summarise(_scalar_values(runs, "speciation_slope_late")),
        "population_final": _summarise(_scalar_values(runs, "population_final")),
    }
    cluster_info = _cluster_frequency_table([r["n_clusters_final"] for r in runs])

    compare_summaries: Optional[Dict[str, Dict[str, Any]]] = None
    compare_runs: Optional[List[Dict[str, Any]]] = None
    if args.compare_sweep_dir:
        cmp_dir = Path(args.compare_sweep_dir)
        if cmp_dir.is_dir():
            print(f"Loading comparison runs from {cmp_dir} …")
            compare_runs = _load_balanced_runs(cmp_dir, window_steps=args.late_window_steps)
            if compare_runs:
                compare_summaries = {
                    "speciation_final": _summarise(_scalar_values(compare_runs, "speciation_final")),
                    "speciation_slope_late": _summarise(_scalar_values(compare_runs, "speciation_slope_late")),
                }
        else:
            print(f"Warning: compare sweep dir not found: {cmp_dir}", file=sys.stderr)

    artifacts: Dict[str, Optional[str]] = {}
    p = _plot_speciation_trajectories(runs, output_dir)
    artifacts["speciation_trajectories"] = str(p) if p else None
    p = _plot_n_clusters_bar(runs, output_dir)
    artifacts["final_n_clusters_bar"] = str(p) if p else None

    per_seed_public = [
        {
            "seed": r["seed"],
            "speciation_final": r["speciation_final"],
            "speciation_slope_full": r["speciation_slope_full"],
            "speciation_slope_late": r["speciation_slope_late"],
            "speciation_direction_full": r["speciation_direction_full"],
            "speciation_direction_late": r["speciation_direction_late"],
            "n_clusters_final": r["n_clusters_final"],
            "population_final": r["population_final"],
            "max_logged_step": r["max_logged_step"],
            "late_window_step_low": r["late_window_step_low"],
            "late_window_step_high": r["late_window_step_high"],
        }
        for r in runs
    ]

    summary_payload: Dict[str, Any] = {
        "profile": PROFILE,
        "late_window_steps": args.late_window_steps,
        "per_seed": per_seed_public,
        "summaries": summaries,
        "n_clusters_distribution": cluster_info,
        "artifacts": artifacts,
    }
    if compare_summaries:
        summary_payload["compare_sweep_dir"] = str(Path(args.compare_sweep_dir).resolve())
        summary_payload["compare_summaries"] = compare_summaries

    summary_path = output_dir / "long_horizon_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, default=str)

    md = _build_markdown(
        window_steps=args.late_window_steps,
        primary_label=str(sweep_dir.resolve()),
        compare_label=str(Path(args.compare_sweep_dir).resolve()) if args.compare_sweep_dir else None,
        per_seed=runs,
        summaries=summaries,
        cluster_info=cluster_info,
        compare_summaries=compare_summaries,
        artifacts=artifacts,
    )
    md_path = output_dir / "long_horizon_summary.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nAggregation complete. Outputs in: {output_dir}")
    print(f"  JSON: {summary_path}")
    print(f"  MD:   {md_path}")
    for name, path in artifacts.items():
        if path:
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
