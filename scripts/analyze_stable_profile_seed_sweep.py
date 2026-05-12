#!/usr/bin/env python3
"""Aggregate seed-sweep results across stable resource profiles.

Reads artifacts produced by ``scripts/run_stable_profile_seed_sweep.py`` and
emits:

- ``seed_sweep_summary.json`` — machine-readable aggregate stats per profile
  (mean, variance, 95% CI for speciation index and gene shifts)
- ``seed_sweep_summary.md`` — human-readable comparison note with robustness
  assessment
- ``speciation_trajectories.png`` — all-seed speciation-index traces per
  profile, with per-profile mean ribbon
- ``learning_rate_shift_boxplot.png`` — per-profile learning_rate % shift
  distribution (box-and-whisker)
- ``gene_shift_heatmap.png`` — per-profile mean % shift for key genes, with
  direction-agreement annotation

Usage
-----
::

    python scripts/analyze_stable_profile_seed_sweep.py \\
        --sweep-dir experiments/stable_profile_sweep

    # Restrict to specific profiles
    python scripts/analyze_stable_profile_seed_sweep.py \\
        --sweep-dir experiments/stable_profile_sweep \\
        --profiles conservative buffered

    # Point at a custom output directory
    python scripts/analyze_stable_profile_seed_sweep.py \\
        --sweep-dir experiments/stable_profile_sweep \\
        --output-dir experiments/stable_profile_sweep/aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── Constants ─────────────────────────────────────────────────────────────────

PROFILE_ORDER: List[str] = ["conservative", "balanced", "buffered"]

PROFILE_COLORS: Dict[str, str] = {
    "conservative": "#B8580E",
    "balanced": "#6B7280",
    "buffered": "#2E7D32",
}

# Genes where the original single-seed comparison found a consistent direction
# across all three buffer levels.
CONVERGENT_GENES: List[str] = [
    "attack_weight",
    "share_weight",
    "attack_mult_desperate",
    "move_mult_no_resources",
    "memory_size",
    "dqn_hidden_size",
    "epsilon_start",
    "per_alpha",
    "target_update_freq",
]

# Genes where the direction differed by resource level in the original run.
DIRECTION_FLIP_GENES: List[str] = [
    "learning_rate",
    "ensemble_size",
    "reproduce_mult_wealthy",
    "reproduce_mult_poor",
    "gamma",
]

# Speciation trajectory slope threshold (per 100 steps) for classifying
# direction: |slope| < SLOPE_EPSILON → "stable".
SLOPE_EPSILON = 1e-4

# 95% two-sided t-distribution multiplier for small N; exact values for
# n = 2..10 used by _t_ci below.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
    5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
}


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ── Statistical helpers ───────────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _variance(xs: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return float(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _t_ci(xs: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Return (lower, upper) two-sided (1-alpha) t confidence interval."""
    n = len(xs)
    if n < 2:
        m = _mean(xs)
        return m, m
    m = _mean(xs)
    se = math.sqrt(_variance(xs) / n)
    t = _T95.get(n - 1, 2.0)  # fall back to ~2 for n ≥ 11
    return m - t * se, m + t * se


def _pct_shift(initial: float, final: float) -> float:
    if math.isnan(initial) or math.isnan(final) or initial == 0.0:
        return float("nan")
    return 100.0 * (final - initial) / abs(initial)


def _speciation_slope(trajectory: List[Dict[str, Any]]) -> float:
    """Slope of a linear fit on speciation_index vs step (per 100 steps)."""
    pairs = [
        (float(row["step"]), float(row["speciation_index"]))
        for row in trajectory
        if "speciation_index" in row and row.get("speciation_index") is not None
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


def _classify_speciation_direction(slope: float) -> str:
    if math.isnan(slope):
        return "unknown"
    if slope > SLOPE_EPSILON:
        return "diverging"
    if slope < -SLOPE_EPSILON:
        return "merging"
    return "stable"


# ── Per-run data extraction ───────────────────────────────────────────────────

def _extract_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract key metrics from a single run directory.

    Returns None if the run dir does not contain the minimum required files.
    """
    trajectory = _read_jsonl(run_dir / "intrinsic_gene_trajectory.jsonl")
    snapshots = _read_jsonl(run_dir / "intrinsic_gene_snapshots.jsonl")

    if not trajectory:
        return None

    # ── Speciation metrics ───────────────────────────────────────────────────
    spec_values = [
        float(r["speciation_index"])
        for r in trajectory
        if "speciation_index" in r and r["speciation_index"] is not None
    ]
    speciation_final = spec_values[-1] if spec_values else float("nan")
    speciation_mean = _mean(spec_values) if spec_values else float("nan")
    speciation_slope = _speciation_slope(trajectory)

    # ── Per-gene shifts ───────────────────────────────────────────────────────
    gene_initial: Dict[str, float] = {}
    gene_final: Dict[str, float] = {}

    non_empty_snaps = [s for s in snapshots if s.get("agents")]
    if non_empty_snaps:
        first = non_empty_snaps[0]
        last = non_empty_snaps[-1]
        gene_names = list(first["agents"][0].get("chromosome", {}).keys()) if first.get("agents") else []
        for g in gene_names:
            vals = [
                float(a["chromosome"][g])
                for a in first.get("agents", [])
                if g in a.get("chromosome", {})
            ]
            if vals:
                gene_initial[g] = _mean(vals)
            vals = [
                float(a["chromosome"][g])
                for a in last.get("agents", [])
                if g in a.get("chromosome", {})
            ]
            if vals:
                gene_final[g] = _mean(vals)

    gene_pct_shift: Dict[str, float] = {
        g: _pct_shift(gene_initial.get(g, float("nan")), gene_final.get(g, float("nan")))
        for g in gene_final
    }

    # ── Population metrics ────────────────────────────────────────────────────
    n_alive = [r.get("n_alive", 0) for r in trajectory]
    pop_mean = _mean([float(x) for x in n_alive])
    pop_final = float(n_alive[-1]) if n_alive else float("nan")

    return {
        "speciation_final": speciation_final,
        "speciation_mean": speciation_mean,
        "speciation_slope": speciation_slope,
        "speciation_direction": _classify_speciation_direction(speciation_slope),
        "gene_pct_shift": gene_pct_shift,
        "gene_initial": gene_initial,
        "gene_final": gene_final,
        "population_mean": pop_mean,
        "population_final": pop_final,
        "trajectory": trajectory,
        "run_dir": str(run_dir),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate_profile(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed metrics for a single profile."""
    n = len(runs)

    def _agg_scalar(key: str) -> Dict[str, Any]:
        vals = [r[key] for r in runs if not math.isnan(r.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "variance": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
        lo, hi = _t_ci(vals)
        return {
            "mean": _mean(vals),
            "variance": _variance(vals),
            "ci95": [lo, hi],
            "n": len(vals),
        }

    agg: Dict[str, Any] = {
        "n_seeds": n,
        "speciation_final": _agg_scalar("speciation_final"),
        "speciation_mean": _agg_scalar("speciation_mean"),
        "speciation_slope": _agg_scalar("speciation_slope"),
        "speciation_directions": [r["speciation_direction"] for r in runs],
        "population_mean": _agg_scalar("population_mean"),
        "population_final": _agg_scalar("population_final"),
    }

    # Direction agreement: what fraction of seeds agree on the modal direction?
    directions = [r["speciation_direction"] for r in runs]
    dir_counts = Counter(directions)
    modal_dir, modal_count = dir_counts.most_common(1)[0]
    agg["speciation_direction_modal"] = modal_dir
    agg["speciation_direction_agreement"] = modal_count / n if n else float("nan")

    # Gene-level aggregation
    gene_names = set()
    for r in runs:
        gene_names.update(r.get("gene_pct_shift", {}).keys())

    per_gene: Dict[str, Dict[str, Any]] = {}
    for gene in sorted(gene_names):
        vals = [
            r["gene_pct_shift"][gene]
            for r in runs
            if gene in r.get("gene_pct_shift", {})
            and not math.isnan(r["gene_pct_shift"][gene])
        ]
        if not vals:
            continue
        lo, hi = _t_ci(vals)
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
        sign_agreement = abs(sum(signs)) / len(signs) if signs else 0.0
        per_gene[gene] = {
            "mean_pct_shift": _mean(vals),
            "variance": _variance(vals),
            "ci95": [lo, hi],
            "n": len(vals),
            "sign_agreement": sign_agreement,
            "all_positive": all(v > 0 for v in vals),
            "all_negative": all(v < 0 for v in vals),
        }
    agg["per_gene"] = per_gene

    return agg


# ── Robustness assessment ─────────────────────────────────────────────────────

def _assess_robustness(
    profile_aggs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify genes and speciation direction as robust or seed-sensitive."""
    profiles = [p for p in PROFILE_ORDER if p in profile_aggs]

    # Speciation-direction robustness per profile: is the modal direction
    # seen in >= 75% of seeds?
    spec_direction_robust: Dict[str, bool] = {}
    for p in profiles:
        agg = profile_aggs[p]
        spec_direction_robust[p] = agg.get("speciation_direction_agreement", 0.0) >= 0.75

    # Gene direction-flip between profiles: is learning_rate robustly
    # positive in buffered and negative in conservative?
    learning_rate_flip_robust = False
    if "buffered" in profile_aggs and "conservative" in profile_aggs:
        buf_lr = profile_aggs["buffered"]["per_gene"].get("learning_rate", {})
        con_lr = profile_aggs["conservative"]["per_gene"].get("learning_rate", {})
        if buf_lr.get("all_positive") and con_lr.get("all_negative"):
            learning_rate_flip_robust = True
        elif (
            buf_lr.get("mean_pct_shift", 0) > 0
            and con_lr.get("mean_pct_shift", 0) < 0
            and buf_lr.get("sign_agreement", 0) >= 0.75
            and con_lr.get("sign_agreement", 0) >= 0.75
        ):
            learning_rate_flip_robust = True

    # Per-gene: robust convergence across all profiles (same sign in every
    # profile's seeds with >= 75% within-profile sign agreement)
    gene_names = set()
    for agg in profile_aggs.values():
        gene_names.update(agg.get("per_gene", {}).keys())

    convergent_robust: List[str] = []
    direction_flip_robust: List[str] = []
    seed_sensitive: List[str] = []

    for gene in sorted(gene_names):
        per_profile_means = []
        per_profile_sign_agreement = []
        for p in profiles:
            g = profile_aggs[p].get("per_gene", {}).get(gene)
            if g:
                per_profile_means.append(g["mean_pct_shift"])
                per_profile_sign_agreement.append(g["sign_agreement"])
            else:
                per_profile_means.append(float("nan"))
                per_profile_sign_agreement.append(0.0)

        valid_means = [m for m in per_profile_means if not math.isnan(m)]
        valid_agreement = [a for a in per_profile_sign_agreement]

        if not valid_means:
            continue

        within_profile_robust = all(a >= 0.75 for a in valid_agreement if not math.isnan(a))
        signs = [1 if m > 0 else -1 if m < 0 else 0 for m in valid_means]
        all_same_sign = len(set(signs)) == 1 and 0 not in signs

        if not within_profile_robust:
            seed_sensitive.append(gene)
        elif all_same_sign:
            convergent_robust.append(gene)
        else:
            # Different signs across profiles but each is stable within seeds
            direction_flip_robust.append(gene)

    return {
        "speciation_direction_robust": spec_direction_robust,
        "learning_rate_flip_robust": learning_rate_flip_robust,
        "convergent_robust_genes": convergent_robust,
        "direction_flip_robust_genes": direction_flip_robust,
        "seed_sensitive_genes": seed_sensitive,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_speciation_trajectories(
    profile_runs: Dict[str, List[Dict[str, Any]]],
    output: Path,
) -> Optional[Path]:
    profiles = [p for p in PROFILE_ORDER if p in profile_runs and profile_runs[p]]
    if not profiles:
        return None

    fig, axes = plt.subplots(
        1, len(profiles), figsize=(5.0 * len(profiles), 4.0), sharey=True
    )
    if len(profiles) == 1:
        axes = [axes]

    for ax, profile in zip(axes, profiles):
        color = PROFILE_COLORS.get(profile, "#333333")
        runs = profile_runs[profile]
        all_steps: Optional[np.ndarray] = None
        all_traces: List[np.ndarray] = []

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
            ax.plot(steps, vals, color=color, alpha=0.35, lw=1.2)
            if all_steps is None or len(steps) > len(all_steps):
                all_steps = steps
            all_traces.append((steps, vals))

        # Mean ribbon across seeds (interpolated to a common step grid)
        if all_steps is not None and len(all_traces) > 1:
            common_steps = all_steps
            interp_traces = []
            for steps, vals in all_traces:
                interp_vals = np.interp(common_steps, steps, vals)
                interp_traces.append(interp_vals)
            stack = np.stack(interp_traces, axis=0)
            mean_trace = stack.mean(axis=0)
            ax.plot(common_steps, mean_trace, color=color, lw=2.5, label="mean")

        ax.set_title(profile, fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("step")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("speciation index")
    fig.suptitle("Speciation Index — All Seeds per Profile", fontsize=13)
    fig.tight_layout()
    out = output / "speciation_trajectories.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_learning_rate_boxplot(
    profile_runs: Dict[str, List[Dict[str, Any]]],
    output: Path,
) -> Optional[Path]:
    profiles = [p for p in PROFILE_ORDER if p in profile_runs and profile_runs[p]]
    if not profiles:
        return None

    data: List[List[float]] = []
    labels: List[str] = []
    colors: List[str] = []
    for profile in profiles:
        vals = [
            r["gene_pct_shift"].get("learning_rate", float("nan"))
            for r in profile_runs[profile]
            if "gene_pct_shift" in r
        ]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            data.append(vals)
            labels.append(profile)
            colors.append(PROFILE_COLORS.get(profile, "#333333"))

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="#aaaaaa", lw=1.2, ls="--")
    ax.set_ylabel("learning_rate % shift (initial → final)")
    ax.set_title("learning_rate shift by profile", fontsize=12)
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output / "learning_rate_shift_boxplot.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_gene_shift_heatmap(
    profile_aggs: Dict[str, Dict[str, Any]],
    robustness: Dict[str, Any],
    output: Path,
) -> Optional[Path]:
    profiles = [p for p in PROFILE_ORDER if p in profile_aggs]
    if not profiles:
        return None

    genes = CONVERGENT_GENES + DIRECTION_FLIP_GENES
    available = set()
    for agg in profile_aggs.values():
        available.update(agg.get("per_gene", {}).keys())
    genes = [g for g in genes if g in available]
    if not genes:
        return None

    matrix = np.full((len(genes), len(profiles)), float("nan"))
    for j, profile in enumerate(profiles):
        for i, gene in enumerate(genes):
            g = profile_aggs[profile].get("per_gene", {}).get(gene)
            if g:
                matrix[i, j] = g["mean_pct_shift"]

    vmax = float(np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 1.0
    fig, ax = plt.subplots(figsize=(3.5 * len(profiles), 0.55 * len(genes) + 1.5))
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles, fontsize=11)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.set_title("Mean % gene shift by profile (seed sweep)", fontsize=12)

    # Annotate cells with value
    for i in range(len(genes)):
        for j in range(len(profiles)):
            v = matrix[i, j]
            if not math.isnan(v):
                ax.text(
                    j, i, f"{v:+.1f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(v) < 0.5 * vmax else "white",
                )

    # Add separator line between convergent and flip-gene groups
    n_conv = sum(1 for g in genes if g in CONVERGENT_GENES)
    if 0 < n_conv < len(genes):
        ax.axhline(n_conv - 0.5, color="#555555", lw=1.5, ls="--")

    plt.colorbar(im, ax=ax, label="mean % shift", shrink=0.8)
    fig.tight_layout()
    out = output / "gene_shift_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{decimals}f}"
    if isinstance(v, (list, tuple)):
        return f"[{_fmt(v[0], decimals)}, {_fmt(v[1], decimals)}]"
    return str(v)


def _build_markdown(
    profile_aggs: Dict[str, Dict[str, Any]],
    robustness: Dict[str, Any],
    n_seeds: Dict[str, int],
    artifacts: Dict[str, Optional[str]],
) -> str:
    profiles = [p for p in PROFILE_ORDER if p in profile_aggs]
    lines: List[str] = []

    lines += [
        "# Stable Profile Seed Sweep — Aggregated Results",
        "",
        "Aggregated over a multi-seed sweep per profile "
        f"({', '.join(str(n_seeds.get(p, '?')) for p in profiles)} seeds for "
        f"{', '.join(profiles)} respectively).",
        "",
    ]

    # ── Speciation summary ────────────────────────────────────────────────────
    lines += ["## Speciation index", ""]
    lines += [
        (
            "| Profile | Seeds | Mean final | 95% CI | Mean slope/100 steps | "
            + "Modal direction | Direction agreement |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile in profiles:
        agg = profile_aggs[profile]
        sf = agg["speciation_final"]
        sl = agg["speciation_slope"]
        lines.append(
            f"| {profile} | {agg['n_seeds']} "
            f"| {_fmt(sf['mean'])} "
            f"| {_fmt(sf['ci95'])} "
            f"| {_fmt(sl['mean'], 4)} "
            f"| {agg['speciation_direction_modal']} "
            f"| {_fmt(agg['speciation_direction_agreement'], 2)} |"
        )
    lines.append("")

    spec_robust_profiles = [
        p for p in profiles if robustness["speciation_direction_robust"].get(p, False)
    ]
    spec_sensitive_profiles = [
        p for p in profiles if not robustness["speciation_direction_robust"].get(p, False)
    ]
    if spec_robust_profiles:
        lines.append(
            f"Speciation-trajectory direction is **robust** (≥75% seed agreement) "
            f"for: **{', '.join(spec_robust_profiles)}**."
        )
    if spec_sensitive_profiles:
        lines.append(
            f"Direction is **seed-sensitive** (< 75% agreement) "
            f"for: {', '.join(spec_sensitive_profiles)}."
        )
    lines.append("")

    # ── Learning-rate shift ───────────────────────────────────────────────────
    lines += ["## learning_rate shift", ""]
    lines += [
        "| Profile | Mean % shift | 95% CI | Sign agreement | Assessment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for profile in profiles:
        lr = profile_aggs[profile].get("per_gene", {}).get("learning_rate")
        if lr is None:
            lines.append(f"| {profile} | n/a | n/a | n/a | n/a |")
            continue
        sa = lr.get("sign_agreement", float("nan"))
        if lr.get("all_positive"):
            assessment = "consistently ↑"
        elif lr.get("all_negative"):
            assessment = "consistently ↓"
        elif sa >= 0.75:
            sign_str = "↑" if lr["mean_pct_shift"] > 0 else "↓"
            assessment = f"predominantly {sign_str}"
        else:
            assessment = "mixed"
        lines.append(
            f"| {profile} "
            f"| {_fmt(lr['mean_pct_shift'], 1)} "
            f"| {_fmt(lr['ci95'])} "
            f"| {_fmt(sa, 2)} "
            f"| {assessment} |"
        )
    lines.append("")

    if robustness["learning_rate_flip_robust"]:
        lines.append(
            "**learning_rate direction flip is robust across seeds:** "
            "buffered consistently selects faster learners; "
            "conservative consistently selects slower learners."
        )
    else:
        lines.append(
            "learning_rate direction flip **not fully robust across seeds**: "
            "check per-profile sign agreement above."
        )
    lines.append("")

    # ── Gene robustness summary ───────────────────────────────────────────────
    lines += ["## Gene-shift robustness", ""]

    conv = robustness["convergent_robust_genes"]
    flip = robustness["direction_flip_robust_genes"]
    sensitive = robustness["seed_sensitive_genes"]

    if conv:
        lines.append(
            f"**Robustly convergent** (same direction in every profile's seeds, "
            f"≥75% within-profile sign agreement): "
            f"{', '.join(f'`{g}`' for g in conv)}."
        )
    if flip:
        lines.append(
            f"**Robustly direction-flipping** (stable within each profile's seeds, "
            f"but sign changes across profiles): "
            f"{', '.join(f'`{g}`' for g in flip)}."
        )
    if sensitive:
        lines.append(
            f"**Seed-sensitive** (< 75% within-profile sign agreement in at least "
            f"one profile): {', '.join(f'`{g}`' for g in sensitive)}."
        )
    lines.append("")

    # ── Per-gene table ────────────────────────────────────────────────────────
    all_genes = CONVERGENT_GENES + DIRECTION_FLIP_GENES
    available_genes = set()
    for agg in profile_aggs.values():
        available_genes.update(agg.get("per_gene", {}).keys())
    table_genes = [g for g in all_genes if g in available_genes]

    if table_genes:
        lines += ["## Per-gene shift table (mean % shift across seeds)", ""]
        header = "| Gene | Group | " + " | ".join(
            f"{p} mean ({n_seeds.get(p,'?')} seeds)" for p in profiles
        ) + " |"
        lines.append(header)
        lines.append("| --- | --- |" + " --- |" * len(profiles))

        for gene in table_genes:
            group = "convergent" if gene in CONVERGENT_GENES else "dir-flip"
            cells = []
            for profile in profiles:
                g = profile_aggs[profile].get("per_gene", {}).get(gene)
                if g:
                    cells.append(f"{_fmt(g['mean_pct_shift'], 1)}")
                else:
                    cells.append("n/a")
            lines.append(f"| `{gene}` | {group} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── Artifacts ────────────────────────────────────────────────────────────
    lines += ["## Artifacts", ""]
    for name, path in artifacts.items():
        if path:
            lines.append(f"- [{name}]({path})")
        else:
            lines.append(f"- {name}: not produced")
    lines.append("")

    return "\n".join(lines)


# ── Discovery ─────────────────────────────────────────────────────────────────

def _discover_runs(
    sweep_dir: Path, profiles: List[str]
) -> Dict[str, List[Tuple[int, Path]]]:
    """Discover per-seed run dirs under ``stable_{profile}/seed_{seed}``."""
    found: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

    for profile in profiles:
        profile_dir = sweep_dir / f"stable_{profile}"
        if not profile_dir.is_dir():
            continue
        for seed_dir in profile_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            if not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            found[profile].append((seed, seed_dir))
        found[profile].sort(key=lambda t: t[0])

    return dict(found)


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate seed-sweep results across stable resource profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Base directory produced by run_stable_profile_seed_sweep.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write aggregated artifacts (defaults to <sweep_dir>/aggregate).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["conservative", "balanced", "buffered"],
        choices=["conservative", "balanced", "buffered"],
        metavar="PROFILE",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_map = _discover_runs(sweep_dir, args.profiles)

    if not run_map:
        print(
            f"No per-seed run directories found under {sweep_dir}. "
            "Run run_stable_profile_seed_sweep.py first.",
            file=sys.stderr,
        )
        return 1

    # ── Load per-seed metrics ─────────────────────────────────────────────────
    profile_runs: Dict[str, List[Dict[str, Any]]] = {}
    for profile, seed_dirs in sorted(run_map.items()):
        runs: List[Dict[str, Any]] = []
        for seed, run_dir in seed_dirs:
            metrics = _extract_run_metrics(run_dir)
            if metrics is None:
                print(f"  Skipping {run_dir} — trajectory not found or empty.")
                continue
            metrics["seed"] = seed
            runs.append(metrics)
            print(f"  [{profile}] seed={seed}: "
                  f"spec_dir={metrics['speciation_direction']}, "
                  f"lr_shift={_fmt(metrics['gene_pct_shift'].get('learning_rate', float('nan')), 1)}%")
        profile_runs[profile] = runs
        print(f"Loaded {len(runs)} runs for profile '{profile}'.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    profile_aggs: Dict[str, Dict[str, Any]] = {}
    n_seeds: Dict[str, int] = {}
    for profile, runs in profile_runs.items():
        if not runs:
            continue
        profile_aggs[profile] = _aggregate_profile(runs)
        n_seeds[profile] = len(runs)

    robustness = _assess_robustness(profile_aggs)

    # ── Plots ─────────────────────────────────────────────────────────────────
    artifacts: Dict[str, Optional[str]] = {}

    p = _plot_speciation_trajectories(profile_runs, output_dir)
    artifacts["speciation_trajectories"] = str(p) if p else None

    p = _plot_learning_rate_boxplot(profile_runs, output_dir)
    artifacts["learning_rate_shift_boxplot"] = str(p) if p else None

    p = _plot_gene_shift_heatmap(profile_aggs, robustness, output_dir)
    artifacts["gene_shift_heatmap"] = str(p) if p else None

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "profiles_analyzed": [p for p in PROFILE_ORDER if p in profile_aggs],
        "aggregates": profile_aggs,
        "robustness": robustness,
        "artifacts": artifacts,
    }
    summary_path = output_dir / "seed_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md = _build_markdown(profile_aggs, robustness, n_seeds, artifacts)
    md_path = output_dir / "seed_sweep_summary.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nAggregation complete. Outputs in: {output_dir}")
    for name, path in artifacts.items():
        if path:
            print(f"  {name}: {path}")
    print(f"  summary JSON : {summary_path}")
    print(f"  summary MD   : {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
