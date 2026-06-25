#!/usr/bin/env python3
"""Render figures for the transferable-signal precondition gate devlog (#904).

Reads ``signal_budget_summary.json`` produced by
``scripts/measure_transferable_signal.py`` and writes three figures:

- ``transferable_signal_budget.png`` — per-profile mean budget on the gate
  decision metric (survival-decoupled early-age net reward) with 95% CI and a
  zero reference line. The gate verdict: every CI sits above zero.
- ``transferable_signal_policy_deltas.png`` — distribution of all per-policy
  decision-metric deltas by profile.
- ``transferable_signal_reward_vs_age.png`` — mean reward delta vs agent age,
  showing the decision-quality edge compounding over the episode (survival
  delta is identically zero under the non-degenerate baseline).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    PROFILE_COLORS,
    PROFILE_ORDER,
)


def _profiles(summary: Dict[str, Any]) -> List[str]:
    per_profile = summary["aggregate"]["per_profile"]
    return [p for p in PROFILE_ORDER if p in per_profile]


def _cells_by_profile(summary: Dict[str, Any], profile: str) -> List[Dict[str, Any]]:
    return [c for c in summary["cells"] if c["profile"] == profile]


def _verdict_for(block: Dict[str, Any], gate_metric: str) -> Dict[str, Any]:
    """Gate-metric verdict, tolerating both the new and legacy schema."""
    verdicts = block.get("verdicts")
    if verdicts and gate_metric in verdicts:
        return verdicts[gate_metric]
    return block["reward_verdict"]


def _series_for(block: Dict[str, Any], gate_metric: str) -> List[float]:
    per_seed = block.get("budget_per_seed")
    if per_seed and gate_metric in per_seed:
        return per_seed[gate_metric]
    return block["reward_budget_per_seed"]


def plot_budget(summary: Dict[str, Any], out: Path) -> Path:
    """Per-profile mean budget (gate decision metric) with 95% CI + zero line."""
    profiles = _profiles(summary)
    per_profile = summary["aggregate"]["per_profile"]
    gate_metric = summary["aggregate"].get("gate_metric", "reward_budget")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axhline(0.0, color="#9CA3AF", lw=1.0, ls="--", zorder=1)

    his = [float(_verdict_for(per_profile[p], gate_metric)["ci95"][1]) for p in profiles]
    los = [float(_verdict_for(per_profile[p], gate_metric)["ci95"][0]) for p in profiles]
    finite_his = [v for v in his if math.isfinite(v)]
    finite_los = [v for v in los if math.isfinite(v)]
    y_hi = max(finite_his) if finite_his else 1.0
    y_lo = min(0.0, min(finite_los)) if finite_los else 0.0
    # Enforce a minimum span so ax.set_ylim receives distinct limits even when
    # CI endpoints are identical (e.g. a single seed where CI is undefined).
    _MIN_SPAN = 1e-6
    span = max(y_hi - y_lo, _MIN_SPAN)
    ax.set_ylim(y_lo - span * 0.12, y_hi + span * 0.32)

    for i, profile in enumerate(profiles):
        verdict = _verdict_for(per_profile[profile], gate_metric)
        mean = float(verdict["mean_delta"])
        lo, hi = (float(verdict["ci95"][0]), float(verdict["ci95"][1]))
        color = PROFILE_COLORS.get(profile, "#333333")
        # Per-seed points (jittered) beside the summary marker so they don't
        # sit underneath the error bar.
        seeds = _series_for(per_profile[profile], gate_metric)
        jitter = (np.random.default_rng(i).random(len(seeds)) - 0.5) * 0.10 + 0.16
        ax.scatter(
            np.full(len(seeds), i) + jitter,
            seeds,
            color=color,
            alpha=0.3,
            s=26,
            zorder=2,
        )
        ax.errorbar(
            i,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=6,
            markersize=9,
            zorder=3,
        )
        ax.annotate(
            f"{mean:.1f}\n[{lo:.1f}, {hi:.1f}]\nsign {verdict['sign_agreement']:.2f}",
            (i, hi),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
        )

    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles)
    ax.set_xlim(-0.5, len(profiles) - 0.5)
    ax.set_ylabel(f"Decision-metric budget: {gate_metric}\n(end-of-life - init)")
    gate = summary["aggregate"].get("gate", "?")
    ax.set_title(
        f"Transferable decision-signal per profile (gate: {gate})", pad=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _delta_key(gate_metric: str) -> str:
    """Map a budget key (``*_budget``) to its per-policy delta key (``*_delta``)."""
    base = gate_metric[:-len("_budget")] if gate_metric.endswith("_budget") else gate_metric
    return f"{base}_delta"


def plot_policy_deltas(summary: Dict[str, Any], out: Path) -> Path:
    """Distribution of every per-policy decision-metric delta, by profile."""
    profiles = _profiles(summary)
    gate_metric = summary["aggregate"].get("gate_metric", "reward_budget")
    delta_key = _delta_key(gate_metric)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axhline(0.0, color="#9CA3AF", lw=1.0, ls="--", zorder=1)

    rng = np.random.default_rng(0)
    n_total = 0
    for i, profile in enumerate(profiles):
        deltas = [
            p[delta_key]
            for c in _cells_by_profile(summary, profile)
            for p in c["per_policy"]
            if delta_key in p
        ]
        n_total += len(deltas)
        color = PROFILE_COLORS.get(profile, "#333333")
        jitter = (rng.random(len(deltas)) - 0.5) * 0.28
        ax.scatter(
            np.full(len(deltas), i) + jitter,
            deltas,
            color=color,
            alpha=0.45,
            s=26,
            zorder=2,
        )
        if deltas:
            ax.scatter(
                i,
                float(np.mean(deltas)),
                marker="_",
                color=color,
                s=900,
                linewidths=2.5,
                zorder=3,
            )

    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles)
    ax.set_ylabel(f"Per-policy {delta_key} (end-of-life - init)")
    ax.set_title(
        f"Per-policy decision-signal distribution "
        f"({n_total} policies; dash = profile mean)"
    )
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _reward_age_keys(summary: Dict[str, Any]) -> List[int]:
    ages = summary.get("config", {}).get("reward_ages") or [10, 25, 50]
    return [int(a) for a in ages]


def plot_reward_vs_age(summary: Dict[str, Any], out: Path) -> Path:
    """Mean reward delta vs agent age - the decision-quality edge compounds.

    Replaces the old reward-vs-survival scatter: under the non-degenerate
    baseline survival delta is identically zero, so the signal lives entirely
    in how much *better* the policy forages, which grows with age.
    """
    profiles = _profiles(summary)
    ages = _reward_age_keys(summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axhline(0.0, color="#9CA3AF", lw=0.8, ls="--", zorder=1)

    for profile in profiles:
        per_policy = [
            p for c in _cells_by_profile(summary, profile) for p in c["per_policy"]
        ]
        means: List[float] = []
        sems: List[float] = []
        for a in ages:
            vals = np.array(
                [p[f"reward_age_{a}_delta"] for p in per_policy
                 if f"reward_age_{a}_delta" in p],
                dtype=float,
            )
            means.append(float(vals.mean()) if vals.size else np.nan)
            sems.append(
                float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            )
        ax.errorbar(
            ages,
            means,
            yerr=sems,
            marker="o",
            capsize=4,
            color=PROFILE_COLORS.get(profile, "#333333"),
            label=profile,
            zorder=2,
        )

    ax.set_xlabel("Agent age (steps since birth) at reward snapshot")
    ax.set_ylabel("Net reward delta (end-of-life - init)")
    ax.set_title("Decision-quality edge compounds with age (survival delta = 0)")
    ax.set_xticks(ages)
    ax.legend(title="profile", frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=str,
        default="experiments/transferable_signal/signal_budget_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="docs/research/devlog/figures",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        plot_budget(summary, out_dir / "transferable_signal_budget.png"),
        plot_policy_deltas(summary, out_dir / "transferable_signal_policy_deltas.png"),
        plot_reward_vs_age(
            summary, out_dir / "transferable_signal_reward_vs_age.png"
        ),
    ]
    for path in figures:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
