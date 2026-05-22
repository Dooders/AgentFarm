#!/usr/bin/env python3
"""Compare Baldwinian baseline against inheritance-mode treatment arm(s).

Outputs
-------
- ``inheritance_ab_summary.json`` — machine-readable paired comparisons
- ``inheritance_ab_summary.md`` — verdict-focused markdown report
- ``paired_delta_heatmap.png`` — mean paired deltas by profile/arm
- ``speciation_trajectories_with_arms.png`` — speciation traces by profile/arm
- ``startup_transient_comparison.png`` — startup-transient paired deltas
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    PROFILE_ORDER,
    SIGN_AGREEMENT_THRESHOLD,
    _discover_runs,
    _extract_run_metrics,
    _mean,
    _t_ci,
    _variance,
)
from scripts.compare_crossover_arms import _lineage_metrics  # noqa: E402

METRIC_KEYS: List[str] = [
    "speciation_final",
    "speciation_slope",
    "speciation_mean",
    "population_mean",
    "population_final",
]
LINEAGE_KEYS: List[str] = ["mean_k", "churn_rate"]
STABILITY_KEYS: List[str] = [
    "startup_transient.peak_birth_rate",
    "startup_transient.peak_death_rate",
    "startup_transient.oscillation_amplitude",
]
INFO_KEYS: List[str] = ["lamarckian_warmstart_rate"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Baldwinian baseline against one or more inheritance-mode "
            "treatment sweeps, paired by seed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=True,
        help="Sweep dir used as baseline (typically the baldwinian arm).",
    )
    parser.add_argument(
        "--treatment-dir",
        action="append",
        required=True,
        help="Treatment sweep dir(s) to compare against the baseline.",
    )
    parser.add_argument(
        "--arm-labels",
        nargs="+",
        default=None,
        help="Optional labels for treatment arms (same order as --treatment-dir).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=PROFILE_ORDER,
        choices=PROFILE_ORDER,
        metavar="PROFILE",
        help="Profiles to include in the comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/inheritance_ab/aggregate",
        help="Output directory for summary + plots.",
    )
    return parser


def _paired_delta_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if not math.isnan(v)]
    if not clean:
        return {
            "mean_delta": float("nan"),
            "variance": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "sign_agreement": float("nan"),
            "n": 0,
        }

    positive = sum(1 for v in clean if v > 0.0)
    negative = sum(1 for v in clean if v < 0.0)
    sign_agreement = max(positive, negative) / len(clean)
    lo, hi = _t_ci(clean)
    return {
        "mean_delta": _mean(clean),
        "variance": _variance(clean),
        "ci95": [lo, hi],
        "sign_agreement": sign_agreement,
        "n": len(clean),
    }


def _ci_excludes_zero(ci: Sequence[float]) -> bool:
    if len(ci) != 2:
        return False
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return False
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def _extract_metadata_metrics(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "intrinsic_evolution_metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    startup = meta.get("startup_transient_metrics", {})
    inheritance = meta.get("policy_inheritance_metrics", {})
    applied = int(inheritance.get("lamarckian_warmstart_applied", 0))
    skipped = int(inheritance.get("lamarckian_warmstart_skipped", 0))
    denom = applied + skipped
    warmstart_rate = float(applied / denom) if denom > 0 else float("nan")
    return {
        "startup_transient": {
            "peak_birth_rate": float(startup.get("peak_birth_rate", float("nan"))),
            "peak_death_rate": float(startup.get("peak_death_rate", float("nan"))),
            "oscillation_amplitude": float(startup.get("oscillation_amplitude", float("nan"))),
        },
        "lamarckian_warmstart_rate": warmstart_rate,
    }


def _extract_full_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    metrics = _extract_run_metrics(run_dir)
    if metrics is None:
        return None
    metrics.update({"lineage": _lineage_metrics(run_dir)})
    metrics.update(_extract_metadata_metrics(run_dir))
    return metrics


def _load_arm(
    sweep_dir: Path, profiles: Sequence[str]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
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


def _paired_metric_deltas(
    baseline_runs: Dict[int, Dict[str, Any]],
    treatment_runs: Dict[int, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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

    for sk in STABILITY_KEYS:
        metric_key = sk.split(".", 1)[1]
        deltas = []
        for seed in paired_seeds:
            b = baseline_runs[seed].get("startup_transient", {}).get(metric_key)
            t = treatment_runs[seed].get("startup_transient", {}).get(metric_key)
            if b is None or t is None:
                continue
            if math.isnan(b) or math.isnan(t):
                continue
            deltas.append(t - b)
        out[sk] = _paired_delta_summary(deltas)

    for ik in INFO_KEYS:
        deltas = []
        for seed in paired_seeds:
            b = baseline_runs[seed].get(ik)
            t = treatment_runs[seed].get(ik)
            if b is None or t is None:
                continue
            if math.isnan(b) or math.isnan(t):
                continue
            deltas.append(t - b)
        out[ik] = _paired_delta_summary(deltas)

    out["_paired_seeds"] = paired_seeds  # type: ignore[assignment]
    return out


def _is_robust(summary: Dict[str, Any]) -> bool:
    return (
        summary.get("sign_agreement", 0.0) >= SIGN_AGREEMENT_THRESHOLD
        and _ci_excludes_zero(summary.get("ci95", []))
    )


def _classify_regime_verdict(profile_deltas: Dict[str, Dict[str, Any]]) -> str:
    performance = profile_deltas.get("population_mean", {})
    stability = profile_deltas.get("startup_transient.oscillation_amplitude", {})
    speciation = profile_deltas.get("speciation_slope", {})

    has_perf_win = _is_robust(performance) and performance.get("mean_delta", 0.0) > 0.0
    has_stability_loss = _is_robust(stability) and stability.get("mean_delta", 0.0) > 0.0
    has_collapse = _is_robust(speciation) and speciation.get("mean_delta", 0.0) < 0.0

    if has_perf_win and not has_stability_loss:
        return "net recommend lamarckian"
    if has_stability_loss and not has_perf_win:
        return "net recommend baldwinian"
    if has_perf_win and has_stability_loss:
        return "performance win + stability loss"
    if has_collapse:
        return "speciation collapse risk"
    return "no robust effect"


def _extract_speciation_trace(run: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    trajectory = run.get("trajectory", [])
    steps: List[int] = []
    values: List[float] = []
    for row in trajectory:
        value = row.get("speciation_index")
        step = row.get("step")
        if value is None or step is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        steps.append(int(step))
        values.append(value)
    return steps, values


def _plot_speciation_trajectories(
    baseline: Dict[str, Dict[int, Dict[str, Any]]],
    treatments: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    profiles: Sequence[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(profiles), 1, figsize=(11, 3.6 * len(profiles)), sharex=False)
    if len(profiles) == 1:
        axes = [axes]

    for ax, profile in zip(axes, profiles):
        arms = [("baldwinian", baseline.get(profile, {}))]
        arms.extend((label, runs.get(profile, {})) for label, runs in treatments.items())
        for label, runs in arms:
            all_steps = sorted(
                {
                    step
                    for run in runs.values()
                    for step in _extract_speciation_trace(run)[0]
                }
            )
            if not all_steps:
                continue
            step_to_vals: Dict[int, List[float]] = {s: [] for s in all_steps}
            for run in runs.values():
                steps, values = _extract_speciation_trace(run)
                for s, v in zip(steps, values):
                    step_to_vals[s].append(v)
            means = [float(np.mean(step_to_vals[s])) if step_to_vals[s] else float("nan") for s in all_steps]
            ax.plot(all_steps, means, label=label)

        ax.set_title(f"{profile} - speciation index trajectory")
        ax.set_xlabel("step")
        ax.set_ylabel("speciation_index")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_paired_delta_heatmap(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    profiles: Sequence[str],
    arm_labels: Sequence[str],
    output_path: Path,
) -> None:
    columns = [
        "population_mean",
        "speciation_slope",
        "startup_transient.peak_death_rate",
        "startup_transient.oscillation_amplitude",
        "lineage.churn_rate",
    ]
    row_labels: List[str] = []
    matrix: List[List[float]] = []
    for profile in profiles:
        for arm in arm_labels:
            row_labels.append(f"{profile}:{arm}")
            row = []
            arm_d = deltas.get(profile, {}).get(arm, {})
            for col in columns:
                value = arm_d.get(col, {}).get("mean_delta", float("nan"))
                row.append(float(value))
            matrix.append(row)
    if not matrix:
        return

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 * len(columns) + 4, 0.45 * len(row_labels) + 3))
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            text = "nan" if math.isnan(v) else f"{v:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean paired delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_startup_transient(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    profiles: Sequence[str],
    arm_labels: Sequence[str],
    output_path: Path,
) -> None:
    metrics = [
        "startup_transient.peak_birth_rate",
        "startup_transient.peak_death_rate",
        "startup_transient.oscillation_amplitude",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        width = 0.8 / max(1, len(arm_labels))
        x = np.arange(len(profiles), dtype=float)
        for idx, arm in enumerate(arm_labels):
            values = [
                float(
                    deltas.get(profile, {})
                    .get(arm, {})
                    .get(metric, {})
                    .get("mean_delta", float("nan"))
                )
                for profile in profiles
            ]
            offset = (idx - (len(arm_labels) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=arm)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=20, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _fmt(value: Any, places: int = 3) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return f"{value:.{places}f}"
    if isinstance(value, list) and len(value) == 2:
        return f"[{_fmt(value[0], places)}, {_fmt(value[1], places)}]"
    return str(value)


def _build_markdown(
    deltas: Dict[str, Dict[str, Dict[str, Any]]],
    verdicts: Dict[str, Dict[str, str]],
    arm_labels: Sequence[str],
) -> str:
    lines: List[str] = [
        "# Inheritance-mode A/B comparison",
        "",
        (
            "Compares inheritance-mode treatment arms against a Baldwinian baseline "
            "using paired-by-seed deltas per profile."
        ),
        "",
        "## Regime verdicts",
        "",
        "| Profile | " + " | ".join(arm_labels) + " |",
        "| --- |" + " --- |" * len(arm_labels),
    ]
    for profile in PROFILE_ORDER:
        if profile not in deltas:
            continue
        cells = [verdicts.get(profile, {}).get(arm, "n/a") for arm in arm_labels]
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")

    lines += [
        "",
        (
            "*Robustness gate: paired 95% CI excludes zero and sign agreement "
            f"is at least {SIGN_AGREEMENT_THRESHOLD:.0%}.*"
        ),
        "",
    ]

    headline_keys = [
        ("population_mean", "population mean"),
        ("population_final", "population final"),
        ("speciation_slope", "speciation slope/100"),
        ("startup_transient.peak_death_rate", "startup peak death rate"),
        ("startup_transient.oscillation_amplitude", "startup oscillation amplitude"),
        ("lamarckian_warmstart_rate", "warmstart rate delta"),
    ]
    lines += ["## Paired deltas (treatment - baseline)", ""]
    for profile in PROFILE_ORDER:
        if profile not in deltas:
            continue
        lines += [f"### {profile}", ""]
        lines += [
            "| Arm | Metric | Mean delta | 95% CI | Sign agreement | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for arm in arm_labels:
            arm_data = deltas.get(profile, {}).get(arm, {})
            for key, label in headline_keys:
                summary = arm_data.get(key, {})
                if not summary:
                    continue
                lines.append(
                    f"| {arm} | {label} | {_fmt(summary.get('mean_delta'))} "
                    f"| {_fmt(summary.get('ci95'))} | {_fmt(summary.get('sign_agreement'), 2)} "
                    f"| {summary.get('n', 0)} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    baseline_dir = Path(args.baseline_dir)
    treatment_dirs = [Path(p) for p in args.treatment_dir]
    if args.arm_labels is None:
        arm_labels = [p.name for p in treatment_dirs]
    else:
        arm_labels = list(args.arm_labels)
    if len(arm_labels) != len(treatment_dirs):
        raise ValueError("--arm-labels must match --treatment-dir count.")

    profiles = [p for p in PROFILE_ORDER if p in args.profiles]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_arm(baseline_dir, profiles)
    treatments = {label: _load_arm(path, profiles) for label, path in zip(arm_labels, treatment_dirs)}

    deltas: Dict[str, Dict[str, Dict[str, Any]]] = {}
    verdicts: Dict[str, Dict[str, str]] = {}
    for profile in profiles:
        deltas[profile] = {}
        verdicts[profile] = {}
        baseline_runs = baseline.get(profile, {})
        for label in arm_labels:
            treatment_runs = treatments[label].get(profile, {})
            profile_deltas = _paired_metric_deltas(baseline_runs, treatment_runs)
            deltas[profile][label] = profile_deltas
            verdicts[profile][label] = _classify_regime_verdict(profile_deltas)

    summary = {
        "comparison_type": "inheritance_mode_ab",
        "baseline_dir": str(baseline_dir),
        "treatments": {label: str(path) for label, path in zip(arm_labels, treatment_dirs)},
        "profiles": profiles,
        "deltas": deltas,
        "verdicts": verdicts,
    }

    json_path = output_dir / "inheritance_ab_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = output_dir / "inheritance_ab_summary.md"
    md_path.write_text(_build_markdown(deltas, verdicts, arm_labels), encoding="utf-8")

    _plot_speciation_trajectories(
        baseline=baseline,
        treatments=treatments,
        profiles=profiles,
        output_path=output_dir / "speciation_trajectories_with_arms.png",
    )
    _plot_paired_delta_heatmap(
        deltas=deltas,
        profiles=profiles,
        arm_labels=arm_labels,
        output_path=output_dir / "paired_delta_heatmap.png",
    )
    _plot_startup_transient(
        deltas=deltas,
        profiles=profiles,
        arm_labels=arm_labels,
        output_path=output_dir / "startup_transient_comparison.png",
    )

    print(f"Wrote summary JSON: {json_path}")
    print(f"Wrote summary markdown: {md_path}")
    print(f"Wrote plots under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
