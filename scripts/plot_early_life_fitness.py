#!/usr/bin/env python3
"""Render figures for the early-life offspring fitness devlog post.

Reads ``early_life_summary.json`` produced by
``scripts/analyze_early_life_fitness.py`` and writes three figures:

- ``early_life_reward_delta.png`` — paired mean deltas (lamarckian - baldwinian)
  for net early RL reward and positive-action fraction, by profile x horizon.
- ``early_life_survival.png`` — survival-to-age curves per arm.
- ``early_life_reward_vs_age.png`` — mean cumulative RL reward vs age per arm
  (does any early signal decay?).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.analyze_stable_profile_seed_sweep import PROFILE_COLORS, PROFILE_ORDER  # noqa: E402

ARM_COLORS = {"baseline": "#6B7280", "treatment": "#B8580E"}


def _profiles(summary: Dict[str, Any]) -> List[str]:
    return [p for p in PROFILE_ORDER if p in summary.get("paired", {})]


def _f(x: Any) -> float:
    """Coerce a JSON scalar (possibly ``null``) to float, mapping None -> NaN."""
    return float("nan") if x is None else float(x)


def _curve_mean(curves: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Average per-seed {age: value} curves onto a shared integer age grid."""
    per_age: Dict[int, List[float]] = {}
    for seed_curve in curves.values():
        for age_str, val in seed_curve.items():
            per_age.setdefault(int(age_str), []).append(float(val))
    ages = np.array(sorted(per_age))
    means = np.array([float(np.mean(per_age[a])) for a in ages])
    return ages, means


def plot_reward_delta(summary: Dict[str, Any], out: Path) -> Path:
    profiles = _profiles(summary)
    ages = [int(a) for a in summary["ages"]]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    for ax, metric, title in (
        (axes[0], "rl_reward_at_age",
         "Net RL reward delta (lamarckian - baldwinian)"),
        (axes[1], "decision_success_rate", "Positive-action-fraction delta"),
    ):
        width = 0.8 / len(ages)
        x = np.arange(len(profiles))
        for j, age in enumerate(ages):
            means, los, his = [], [], []
            for profile in profiles:
                v = summary["paired"][profile]["ages"][str(age)]["verdicts"][metric]
                mean_delta = _f(v["mean_delta"])
                ci_lo, ci_hi = _f(v["ci95"][0]), _f(v["ci95"][1])
                means.append(mean_delta)
                los.append(mean_delta - ci_lo)
                his.append(ci_hi - mean_delta)
            offset = (j - (len(ages) - 1) / 2) * width
            ax.bar(
                x + offset, means, width * 0.92,
                yerr=[los, his], capsize=3,
                label=f"N={age}", alpha=0.85,
            )
        ax.axhline(0, color="#333333", lw=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="horizon", fontsize=9)

    fig.suptitle("Early-life paired deltas by profile and horizon", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_survival(summary: Dict[str, Any], out: Path) -> Path:
    profiles = _profiles(summary)
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.0 * len(profiles), 4.0),
                             sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        for arm in ("baseline", "treatment"):
            curves = summary["survival_curves"][arm].get(profile, {})
            if not curves:
                continue
            ages, means = _curve_mean(curves)
            label = summary[f"{arm}_arm"]
            ax.plot(ages, means, color=ARM_COLORS[arm], lw=2.0, label=label)
        ax.set_title(profile, fontsize=12, color=PROFILE_COLORS.get(profile, "#333"),
                     fontweight="bold")
        ax.set_xlabel("age (steps lived)")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("fraction of offspring surviving")
    fig.suptitle("Offspring survival vs age (uncensored)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reward_vs_age(summary: Dict[str, Any], out: Path) -> Path:
    profiles = _profiles(summary)
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.0 * len(profiles), 4.0),
                             sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        for arm in ("baseline", "treatment"):
            curves = summary["rl_reward_curves"][arm].get(profile, {})
            if not curves:
                continue
            ages, means = _curve_mean(curves)
            label = summary[f"{arm}_arm"]
            ax.plot(ages, means, color=ARM_COLORS[arm], lw=2.0, label=label)
        ax.set_title(profile, fontsize=12, color=PROFILE_COLORS.get(profile, "#333"),
                     fontweight="bold")
        ax.set_xlabel("age (steps lived)")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("mean cumulative RL reward")
    fig.suptitle("Mean cumulative RL reward vs age", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot early-life fitness figures.")
    parser.add_argument("--summary", type=str, required=True,
                        help="Path to early_life_summary.json.")
    parser.add_argument("--output-dir", type=str, default="docs/devlog/figures")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_reward_delta(summary, out_dir / "early_life_reward_delta.png")
    p2 = plot_survival(summary, out_dir / "early_life_survival.png")
    p3 = plot_reward_vs_age(summary, out_dir / "early_life_reward_vs_age.png")
    for p in (p1, p2, p3):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
