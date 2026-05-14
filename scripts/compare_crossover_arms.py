#!/usr/bin/env python3
"""Compare crossover-rerun arms against the no-crossover baseline sweep.

Reads per-seed run dirs from a baseline sweep and one or more crossover-arm
sweeps (both produced by ``run_stable_profile_seed_sweep.py``) and emits a
paired-seed comparison answering: "Does gene flow collapse the rising
speciation pattern?"

Outputs
-------
- ``crossover_rerun_summary.json`` — machine-readable aggregates and per-arm
  paired-delta statistics
- ``crossover_rerun_summary.md`` — verdict-per-profile markdown report
- ``speciation_trajectories_with_arms.png`` — speciation traces per profile,
  baseline + each arm
- ``paired_delta_heatmap.png`` — mean paired delta for key metrics per
  (profile, arm)
- ``lineage_cluster_count.png`` — mean cluster-count trajectory per profile,
  one curve per arm

Usage
-----
::

    python scripts/compare_crossover_arms.py \\
        --baseline-dir experiments/stable_profile_sweep \\
        --treatment-dir experiments/crossover_rerun/uniform \\
        --treatment-dir experiments/crossover_rerun/blend \\
        --arm-labels uniform blend \\
        --output-dir experiments/crossover_rerun/aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
    CONVERGENT_GENES,
    PROFILE_COLORS,
    PROFILE_ORDER,
    SIGN_AGREEMENT_THRESHOLD,
    _classify_speciation_direction,
    _discover_runs,
    _extract_run_metrics,
    _mean,
    _mean_trajectory,
    _sign_agreement,
    _t_ci,
    _variance,
)

# ── Lineage-file parsing ──────────────────────────────────────────────────────

CLUSTER_LINEAGE_FILENAME = "cluster_lineage.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _lineage_metrics(run_dir: Path) -> Dict[str, Any]:
    """Per-run lineage summary from ``cluster_lineage.jsonl``.

    Returns
    -------
    dict with keys:
      - ``cluster_count_trace``: list of (step, k) tuples (k = #clusters at step)
      - ``mean_k``: mean cluster count across clustering steps (NaN if absent)
      - ``churn_rate``: mean per-step fraction of cluster IDs at step t that
        do NOT survive to step t+1.  In [0, 1].  NaN if fewer than two
        clustering steps were recorded.
    """
    rows = _read_jsonl(run_dir / CLUSTER_LINEAGE_FILENAME)
    if not rows:
        return {
            "cluster_count_trace": [],
            "mean_k": float("nan"),
            "churn_rate": float("nan"),
        }

    steps_to_ids: Dict[int, set] = defaultdict(set)
    for r in rows:
        try:
            step = int(r["step"])
            cid = r["cluster_id"]
        except (KeyError, TypeError, ValueError):
            continue
        steps_to_ids[step].add(cid)

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
        "churn_rate": churn,
    }


# ── Paired-delta stats ────────────────────────────────────────────────────────


def _paired_delta_summary(deltas: Sequence[float]) -> Dict[str, Any]:
    """Mean / variance / 95% CI / sign-agreement for paired deltas (NaNs filtered)."""
    clean = [d for d in deltas if not (isinstance(d, float) and math.isnan(d))]
    if not clean:
        return {
            "mean_delta": float("nan"),
            "variance": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "sign_agreement": float("nan"),
            "n": 0,
        }
    lo, hi = _t_ci(clean)
    return {
        "mean_delta": _mean(clean),
        "variance": _variance(clean),
        "ci95": [lo, hi],
        "sign_agreement": _sign_agreement(clean),
        "n": len(clean),
    }


def _ci_excludes_zero(ci: Sequence[float]) -> bool:
    """True iff a two-sided CI is finite and does not straddle zero."""
    if len(ci) != 2:
        return False
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def _classify_collapse_verdict(
    spec_final_delta: Dict[str, Any],
    spec_slope_delta: Dict[str, Any],
) -> str:
    """One-line verdict for "does gene flow collapse the rising speciation pattern?".

    "Collapse" means the speciation trajectory under the treatment is
    *lower* and/or *less rising* than the baseline (i.e. negative paired
    deltas with both robustness criteria satisfied).
    """
    robust = (
        spec_final_delta.get("sign_agreement", 0.0) >= SIGN_AGREEMENT_THRESHOLD
        and _ci_excludes_zero(spec_final_delta.get("ci95", []))
    ) or (
        spec_slope_delta.get("sign_agreement", 0.0) >= SIGN_AGREEMENT_THRESHOLD
        and _ci_excludes_zero(spec_slope_delta.get("ci95", []))
    )
    if not robust:
        return "no robust effect"
    sf_mean = spec_final_delta.get("mean_delta", 0.0)
    sl_mean = spec_slope_delta.get("mean_delta", 0.0)
    if sf_mean < 0 or sl_mean < 0:
        return "robustly collapses"
    return "robustly amplifies"


# ── Per-run data extraction ───────────────────────────────────────────────────


def _extract_full_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Combine baseline-style metrics with lineage metrics for one run."""
    metrics = _extract_run_metrics(run_dir)
    if metrics is None:
        return None
    metrics.update({"lineage": _lineage_metrics(run_dir)})
    return metrics


def _load_arm(sweep_dir: Path, profiles: Sequence[str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load per-(profile, seed) run metrics for one sweep directory.

    Returns a nested dict ``{profile: {seed: metrics}}``.
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    run_map = _discover_runs(sweep_dir, list(profiles))
    for profile, seed_dirs in run_map.items():
        out[profile] = {}
        for seed, run_dir in seed_dirs:
            metrics = _extract_full_run(run_dir)
            if metrics is None:
                continue
            out[profile][seed] = metrics
    return out


# ── Paired aggregation ────────────────────────────────────────────────────────


METRIC_KEYS: List[str] = [
    "speciation_final",
    "speciation_slope",
    "speciation_mean",
    "population_mean",
]
LINEAGE_KEYS: List[str] = ["mean_k", "churn_rate"]


def _paired_metric_deltas(
    baseline_runs: Dict[int, Dict[str, Any]],
    treatment_runs: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute paired-seed deltas for every metric key in ``METRIC_KEYS``,
    plus every gene appearing in both baseline and treatment, plus lineage
    metrics.  Each delta is ``treatment - baseline``.
    """
    paired_seeds = sorted(set(baseline_runs.keys()) & set(treatment_runs.keys()))
    out: Dict[str, Dict[str, Any]] = {}

    for key in METRIC_KEYS:
        deltas = []
        for seed in paired_seeds:
            b = baseline_runs[seed].get(key)
            t = treatment_runs[seed].get(key)
            if b is None or t is None:
                continue
            if math.isnan(b) or math.isnan(t):
                continue
            deltas.append(t - b)
        out[key] = _paired_delta_summary(deltas)

    for lk in LINEAGE_KEYS:
        deltas = []
        for seed in paired_seeds:
            b = baseline_runs[seed].get("lineage", {}).get(lk)
            t = treatment_runs[seed].get("lineage", {}).get(lk)
            if b is None or t is None:
                continue
            if math.isnan(b) or math.isnan(t):
                continue
            deltas.append(t - b)
        out[f"lineage.{lk}"] = _paired_delta_summary(deltas)

    gene_names: set[str] = set()
    for seed in paired_seeds:
        gene_names.update(baseline_runs[seed].get("gene_pct_shift", {}).keys())
        gene_names.update(treatment_runs[seed].get("gene_pct_shift", {}).keys())
    for gene in sorted(gene_names):
        deltas = []
        for seed in paired_seeds:
            b = baseline_runs[seed].get("gene_pct_shift", {}).get(gene)
            t = treatment_runs[seed].get("gene_pct_shift", {}).get(gene)
            if b is None or t is None:
                continue
            if math.isnan(b) or math.isnan(t):
                continue
            deltas.append(t - b)
        if deltas:
            out[f"gene.{gene}"] = _paired_delta_summary(deltas)

    out["_paired_seeds"] = paired_seeds  # type: ignore[assignment]
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────


def _arm_color(label: str, idx: int) -> str:
    palette = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return palette[idx % len(palette)]


def _plot_speciation_with_arms(
    baseline: Dict[str, Dict[int, Dict[str, Any]]],
    arms: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    output: Path,
) -> Optional[Path]:
    profiles = [p for p in PROFILE_ORDER if p in baseline and baseline[p]]
    if not profiles:
        return None

    arm_labels = list(arms.keys())
    fig, axes = plt.subplots(
        1, len(profiles), figsize=(5.2 * len(profiles), 4.2), sharey=True
    )
    if len(profiles) == 1:
        axes = [axes]

    def _trace_pairs(runs: Dict[int, Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        traces: List[Tuple[np.ndarray, np.ndarray]] = []
        for run in runs.values():
            traj = run.get("trajectory", [])
            spec = [
                (float(r["step"]), float(r["speciation_index"]))
                for r in traj
                if r.get("speciation_index") is not None
            ]
            if not spec:
                continue
            traces.append(
                (np.array([s[0] for s in spec]), np.array([s[1] for s in spec]))
            )
        return traces

    for ax, profile in zip(axes, profiles):
        # baseline
        base_color = PROFILE_COLORS.get(profile, "#333333")
        base_traces = _trace_pairs(baseline.get(profile, {}))
        for s, v in base_traces:
            ax.plot(s, v, color=base_color, alpha=0.18, lw=1.0)
        if len(base_traces) > 1:
            m = _mean_trajectory(base_traces)
            if m is not None:
                grid, vals = m
                ax.plot(grid, vals, color=base_color, lw=2.4, label="baseline")

        for idx, label in enumerate(arm_labels):
            color = _arm_color(label, idx)
            arm_traces = _trace_pairs(arms[label].get(profile, {}))
            for s, v in arm_traces:
                ax.plot(s, v, color=color, alpha=0.18, lw=1.0)
            if len(arm_traces) > 1:
                m = _mean_trajectory(arm_traces)
                if m is not None:
                    grid, vals = m
                    ax.plot(grid, vals, color=color, lw=2.4, label=label)

        ax.set_title(profile, fontsize=12, fontweight="bold", color=base_color)
        ax.set_xlabel("step")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("speciation index")
    fig.suptitle(
        "Speciation Index — baseline vs crossover arms (paired seeds)", fontsize=13
    )
    fig.tight_layout()
    out = output / "speciation_trajectories_with_arms.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_paired_delta_heatmap(
    deltas: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    output: Path,
) -> Optional[Path]:
    """Heatmap rows = (profile, arm), cols = metric → mean delta."""
    # Build (profile, arm) row keys
    rows: List[Tuple[str, str]] = []
    for profile in PROFILE_ORDER:
        if profile not in deltas:
            continue
        for arm in deltas[profile]:
            rows.append((profile, arm))
    if not rows:
        return None

    headline_keys = [
        "speciation_final",
        "speciation_slope",
        "speciation_mean",
        "population_mean",
        "lineage.mean_k",
        "lineage.churn_rate",
    ]
    gene_keys = [
        f"gene.{g}" for g in CONVERGENT_GENES + ["learning_rate", "ensemble_size"]
    ]
    all_keys = headline_keys + gene_keys

    # Restrict to keys present in at least one cell
    present_keys: List[str] = []
    for key in all_keys:
        for profile, arm in rows:
            d = deltas[profile][arm].get(key, {})
            if not math.isnan(d.get("mean_delta", float("nan"))):
                present_keys.append(key)
                break

    if not present_keys:
        return None

    matrix = np.full((len(rows), len(present_keys)), float("nan"))
    robust_mask = np.zeros_like(matrix, dtype=bool)
    for i, (profile, arm) in enumerate(rows):
        for j, key in enumerate(present_keys):
            d = deltas[profile][arm].get(key, {})
            mean = d.get("mean_delta", float("nan"))
            matrix[i, j] = mean
            if (
                d.get("sign_agreement", 0.0) >= SIGN_AGREEMENT_THRESHOLD
                and _ci_excludes_zero(d.get("ci95", []))
            ):
                robust_mask[i, j] = True

    vmax = float(np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 1.0
    fig, ax = plt.subplots(
        figsize=(0.65 * len(present_keys) + 2.5, 0.55 * len(rows) + 1.5)
    )
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    row_labels = [f"{p} / {a}" for p, a in rows]
    col_labels = [k.replace("gene.", "").replace("lineage.", "L:") for k in present_keys]
    ax.set_xticks(range(len(present_keys)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Paired-seed mean delta (treatment − baseline). * = robust", fontsize=12)

    for i in range(len(rows)):
        for j in range(len(present_keys)):
            v = matrix[i, j]
            if math.isnan(v):
                continue
            star = "*" if robust_mask[i, j] else ""
            ax.text(
                j, i, f"{v:+.2f}{star}",
                ha="center", va="center", fontsize=7,
                color="black" if abs(v) < 0.5 * vmax else "white",
            )

    plt.colorbar(im, ax=ax, label="mean delta", shrink=0.8)
    fig.tight_layout()
    out = output / "paired_delta_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_lineage_cluster_count(
    baseline: Dict[str, Dict[int, Dict[str, Any]]],
    arms: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    output: Path,
) -> Optional[Path]:
    profiles = [p for p in PROFILE_ORDER if p in baseline]
    if not profiles:
        return None

    arm_labels = list(arms.keys())
    fig, axes = plt.subplots(
        1, len(profiles), figsize=(5.0 * len(profiles), 3.8), sharey=True
    )
    if len(profiles) == 1:
        axes = [axes]

    def _traces(runs: Dict[int, Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for r in runs.values():
            trace = r.get("lineage", {}).get("cluster_count_trace", [])
            if not trace:
                continue
            steps = np.array([t[0] for t in trace], dtype=float)
            ks = np.array([t[1] for t in trace], dtype=float)
            out.append((steps, ks))
        return out

    rendered_any = False
    for ax, profile in zip(axes, profiles):
        base_color = PROFILE_COLORS.get(profile, "#333333")
        bt = _traces(baseline.get(profile, {}))
        if len(bt) > 1:
            m = _mean_trajectory(bt)
            if m is not None:
                grid, vals = m
                ax.plot(grid, vals, color=base_color, lw=2.2, label="baseline")
                rendered_any = True

        for idx, label in enumerate(arm_labels):
            ts = _traces(arms[label].get(profile, {}))
            if len(ts) > 1:
                m = _mean_trajectory(ts)
                if m is not None:
                    grid, vals = m
                    ax.plot(grid, vals, color=_arm_color(label, idx), lw=2.2, label=label)
                    rendered_any = True

        ax.set_title(profile, fontsize=12, color=base_color, fontweight="bold")
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")

    if not rendered_any:
        plt.close(fig)
        return None

    axes[0].set_ylabel("# clusters (mean across seeds)")
    fig.suptitle("Cluster count per step — baseline vs crossover arms", fontsize=13)
    fig.tight_layout()
    out = output / "lineage_cluster_count.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Markdown ─────────────────────────────────────────────────────────────────


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{decimals}f}"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{_fmt(v[0], decimals)}, {_fmt(v[1], decimals)}]"
    return str(v)


def _build_markdown(
    deltas: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    verdicts: Dict[str, Dict[str, str]],
    arm_labels: Sequence[str],
    artifacts: Dict[str, Optional[str]],
) -> str:
    """Build a verdict-headlined markdown report."""
    lines: List[str] = []
    profiles = [p for p in PROFILE_ORDER if p in deltas]

    lines += [
        "# Crossover rerun — paired-seed comparison",
        "",
        "Compares crossover-enabled arms against the no-crossover baseline "
        "across the three stable resource profiles, paired by seed.",
        "",
        "## Verdict: does gene flow collapse the rising speciation pattern?",
        "",
    ]
    lines += ["| Profile | " + " | ".join(arm_labels) + " |",
              "| --- |" + " --- |" * len(arm_labels)]
    for profile in profiles:
        cells = [verdicts.get(profile, {}).get(arm, "n/a") for arm in arm_labels]
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "*Verdict thresholds: paired-delta 95% CI excludes zero AND "
        f"within-profile sign agreement ≥ {SIGN_AGREEMENT_THRESHOLD:.0%}.*"
    )
    lines.append("")

    # Per-(profile, arm) headline-metric tables
    headline_keys = [
        ("speciation_final", "speciation final"),
        ("speciation_slope", "speciation slope/100"),
        ("speciation_mean", "speciation mean"),
        ("population_mean", "population mean"),
        ("lineage.mean_k", "cluster count (mean k)"),
        ("lineage.churn_rate", "cluster churn rate"),
    ]
    lines += ["## Paired deltas (treatment − baseline)", ""]
    for profile in profiles:
        lines += [f"### {profile}", ""]
        lines += [
            "| Arm | Metric | Mean delta | 95% CI | Sign agreement | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for arm in arm_labels:
            arm_d = deltas[profile].get(arm, {})
            for key, label in headline_keys:
                d = arm_d.get(key, {})
                if not d or (
                    isinstance(d.get("mean_delta"), float)
                    and math.isnan(d["mean_delta"])
                ):
                    continue
                lines.append(
                    f"| {arm} | {label} "
                    f"| {_fmt(d.get('mean_delta'), 4)} "
                    f"| {_fmt(d.get('ci95'), 4)} "
                    f"| {_fmt(d.get('sign_agreement'), 2)} "
                    f"| {d.get('n', 0)} |"
                )
        lines.append("")

    # Gene-shift deltas table
    gene_list = CONVERGENT_GENES + ["learning_rate", "ensemble_size"]
    lines += ["## Gene-shift deltas (treatment − baseline; mean % shift)", ""]
    header = "| Profile | Arm | " + " | ".join(gene_list) + " |"
    lines.append(header)
    lines.append("| --- | --- |" + " --- |" * len(gene_list))
    for profile in profiles:
        for arm in arm_labels:
            arm_d = deltas[profile].get(arm, {})
            cells = []
            for g in gene_list:
                d = arm_d.get(f"gene.{g}", {})
                mean = d.get("mean_delta", float("nan"))
                cells.append(_fmt(mean, 1) if not math.isnan(mean) else "n/a")
            lines.append(f"| {profile} | {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    # Artifacts
    lines += ["## Artifacts", ""]
    for name, path in artifacts.items():
        if path:
            lines.append(f"- [{name}]({path})")
        else:
            lines.append(f"- {name}: not produced")
    lines.append("")

    return "\n".join(lines)


# ── Driver ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired-seed comparison of crossover arms vs no-crossover baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument(
        "--treatment-dir",
        type=str,
        action="append",
        required=True,
        help="One per crossover arm; repeat for each arm.",
    )
    parser.add_argument(
        "--arm-labels",
        nargs="+",
        required=True,
        help="Human-readable label per --treatment-dir, in the same order.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILE_ORDER),
        choices=list(PROFILE_ORDER),
        metavar="PROFILE",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if len(args.treatment_dir) != len(args.arm_labels):
        print(
            "Number of --treatment-dir entries must match --arm-labels.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.is_dir():
        print(f"Baseline directory not found: {baseline_dir}", file=sys.stderr)
        return 1

    print(f"Loading baseline from {baseline_dir} ...")
    baseline = _load_arm(baseline_dir, args.profiles)

    arms: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for label, tdir in zip(args.arm_labels, args.treatment_dir):
        tpath = Path(tdir)
        if not tpath.is_dir():
            print(f"Treatment dir not found, skipping: {tpath}", file=sys.stderr)
            continue
        print(f"Loading arm '{label}' from {tpath} ...")
        arms[label] = _load_arm(tpath, args.profiles)

    # deltas[profile][arm] → {metric_key: paired_delta_summary}
    deltas: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    verdicts: Dict[str, Dict[str, str]] = {}
    for profile in args.profiles:
        if profile not in baseline or not baseline[profile]:
            continue
        deltas[profile] = {}
        verdicts[profile] = {}
        for arm_label, arm_runs in arms.items():
            paired = _paired_metric_deltas(
                baseline.get(profile, {}), arm_runs.get(profile, {})
            )
            deltas[profile][arm_label] = paired
            verdicts[profile][arm_label] = _classify_collapse_verdict(
                paired.get("speciation_final", {}),
                paired.get("speciation_slope", {}),
            )

    artifacts: Dict[str, Optional[str]] = {}
    p = _plot_speciation_with_arms(baseline, arms, output_dir)
    artifacts["speciation_trajectories_with_arms"] = str(p) if p else None
    p = _plot_paired_delta_heatmap(deltas, output_dir)
    artifacts["paired_delta_heatmap"] = str(p) if p else None
    p = _plot_lineage_cluster_count(baseline, arms, output_dir)
    artifacts["lineage_cluster_count"] = str(p) if p else None

    summary = {
        "baseline_dir": str(baseline_dir),
        "treatment_dirs": [str(Path(t)) for t in args.treatment_dir],
        "arm_labels": list(args.arm_labels),
        "profiles": list(args.profiles),
        "deltas": deltas,
        "verdicts": verdicts,
        "artifacts": artifacts,
    }
    summary_path = output_dir / "crossover_rerun_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    md = _build_markdown(deltas, verdicts, list(args.arm_labels), artifacts)
    md_path = output_dir / "crossover_rerun_summary.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nComparison complete. Outputs in: {output_dir}")
    print(f"  summary JSON : {summary_path}")
    print(f"  summary MD   : {md_path}")
    for name, path in artifacts.items():
        if path:
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
