#!/usr/bin/env python3
"""Render figures for the spatial-cooperation pre-flight devlog (2026-11-01).

Reads the per-arm ``arm_result.json`` records written by
``scripts/run_intrinsic_cooperation_experiment.py`` and writes two figures:

- ``intrinsic_cooperation_pilot_dynamics.png`` — the full-length pilot (seeds
  1–2, 3,000 steps): population, unit coverage, the policies' own probability
  of ``bond``, and the expected exit payment, per arm. Lines are the seed mean
  of a centred rolling mean; bands span the two seeds.
- ``intrinsic_cooperation_lineage_depth.png`` — maximum generation reached,
  births per 1,000 steps, and the survival curve of lifespan at death, across
  the registered regime at 3,000 and 9,000 steps and the reproduction sweep.

The raw records are git-ignored; regenerate them with the commands in the
devlog's Reproduce section.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

ARM_STYLE = {
    "baseline": ("Baseline", "#7f7f7f"),
    "placebo": ("Placebo", "#c8862a"),
    "cooperation": ("Cooperation", "#1f4e9c"),
    "cooperation_no_motive": ("Cooperation, no motive reward", "#6fa8dc"),
}


@dataclass(frozen=True)
class Regime:
    label: str
    root: str
    color: str


REGIMES = (
    Regime("Registered\n3,000 steps", "intrinsic_cooperation_pilot", "#333333"),
    Regime("Registered\n9,000 steps", "intrinsic_cooperation_pilot_9000", "#999999"),
    Regime("A: richer\nnewborns", "repro_sweep/A_offspring8", "#d95f02"),
    Regime("B: lower\nreserve", "repro_sweep/B_reserve5", "#7570b3"),
    Regime("C: both", "repro_sweep/C_both", "#1b9e77"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def _load_arm(experiments: Path, root: str, rep: int, arm: str) -> Optional[dict]:
    path = experiments / root / f"rep_{rep:02d}" / arm / "arm_result.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _smooth(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _seed_band(records: List[dict], key: str, window: int):
    curves = np.vstack([_smooth(r["series"][key], window) for r in records])
    steps = np.asarray(records[0]["series"]["steps"])
    return steps, curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)


def plot_dynamics(experiments: Path, out: Path, window: int) -> None:
    root = REGIMES[0].root
    records: Dict[str, List[dict]] = {}
    for arm in ARM_STYLE:
        loaded = [_load_arm(experiments, root, rep, arm) for rep in (0, 1)]
        records[arm] = [r for r in loaded if r is not None]

    panels = (
        ("population", "(a) Population", "alive agents", tuple(ARM_STYLE)),
        ("coverage", "(b) Share of agents in a unit", "coverage", ("cooperation", "cooperation_no_motive")),
        (
            "policy_bond_prob",
            "(c) Policies' own probability of bond",
            "probability (before priors and mask)",
            ("placebo", "cooperation", "cooperation_no_motive"),
        ),
        (
            "expected_exit_payment",
            "(d) Expected exit payment (recognition × exit cost)",
            "population mean",
            ("placebo", "cooperation", "cooperation_no_motive"),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, (key, title, ylabel, arms) in zip(axes.flat, panels):
        for arm in arms:
            if not records[arm]:
                continue
            label, color = ARM_STYLE[arm]
            steps, mean, low, high = _seed_band(records[arm], key, window)
            ax.fill_between(steps, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(steps, mean, color=color, linewidth=1.6, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0, 1].set_ylim(bottom=0)
    axes[1, 0].axhline(1 / 10, color="black", linestyle=":", linewidth=1)
    axes[1, 0].text(0.01, 0.97, "dotted: uniform over 10 actions", transform=axes[1, 0].transAxes, va="top",
                    fontsize=8, color="#444444")
    for ax in axes[1]:
        ax.set_xlabel("step")
    handles = [plt.Line2D([], [], color=c, linewidth=2) for _, c in ARM_STYLE.values()]
    fig.legend(handles, [lbl for lbl, _ in ARM_STYLE.values()], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Full-length pilot, seeds 1–2: line = seed mean of a {window}-step rolling mean, band = range over seeds",
        x=0.01, ha="left", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _regime_records(experiments: Path, regime: Regime, arm: str) -> List[dict]:
    loaded = [_load_arm(experiments, regime.root, rep, arm) for rep in (0, 1)]
    return [r for r in loaded if r is not None]


def plot_lineage_depth(experiments: Path, out: Path) -> None:
    fig, (ax_gen, ax_births, ax_life) = plt.subplots(1, 3, figsize=(13.5, 4.4),
                                                     gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    x = np.arange(len(REGIMES))
    arm_offsets = {"baseline": -0.18, "cooperation": 0.18}
    seed_markers = ("o", "s")
    for arm, offset in arm_offsets.items():
        _, color = ARM_STYLE[arm]
        for i, regime in enumerate(REGIMES):
            recs = _regime_records(experiments, regime, arm)
            for k, rec in enumerate(recs):
                xpos = i + offset + (k - 0.5) * 0.14
                summary = rec["summary"]
                steps = len(rec["series"]["steps"])
                style = {"color": color, "marker": seed_markers[k], "s": 38, "zorder": 3}
                ax_gen.scatter(xpos, summary["max_generation"], **style)
                ax_births.scatter(xpos, 1000 * summary["total_births"] / steps, **style)
    legend_handles = [
        plt.Line2D([], [], color=ARM_STYLE[arm][1], marker="o", linestyle="", label=ARM_STYLE[arm][0])
        for arm in arm_offsets
    ] + [
        plt.Line2D([], [], color="black", marker=m, linestyle="", markerfacecolor="none", label=f"seed {k + 1}")
        for k, m in enumerate(seed_markers)
    ]

    ax_gen.set_title("(a) Deepest generation reached")
    ax_gen.set_ylabel("max generation (founders = 0)")
    ax_gen.set_ylim(0, 6)
    ax_gen.set_yticks(range(0, 7))
    ax_gen.axhline(3, color="black", linestyle=":", linewidth=1)
    ax_gen.annotate("gate: ≥ 3 (cooperation arm)", (len(REGIMES) - 0.5, 3), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8)
    ax_gen.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=8)
    ax_births.set_title("(b) Births per 1,000 steps")
    ax_births.set_ylabel("births / 1,000 steps")
    ax_births.set_ylim(bottom=0)
    for ax in (ax_gen, ax_births):
        ax.set_xticks(x)
        ax.set_xticklabels([r.label for r in REGIMES], fontsize=8)
        ax.grid(axis="x", visible=False)

    for regime in REGIMES:
        lifespans = np.concatenate(
            [np.asarray(r["lifespans"], dtype=float) for r in _regime_records(experiments, regime, "baseline")]
        )
        if lifespans.size == 0:
            continue
        ordered = np.sort(lifespans)
        survival = 1.0 - np.arange(len(ordered)) / len(ordered)
        ax_life.step(ordered, survival, where="post", color=regime.color, linewidth=1.5,
                     label=regime.label.replace("\n", " "))
    ax_life.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax_life.annotate("median", (32, 0.5), xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax_life.set_xscale("log")
    ax_life.set_yscale("log")
    ax_life.set_xlim(30, 6000)
    ax_life.set_title("(c) Lifespan at death, baseline arm")
    ax_life.set_xlabel("steps lived (log)")
    ax_life.set_ylabel("share of deaths at least this old (log)")
    ax_life.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-dir", default=str(_repo_root / "experiments"))
    parser.add_argument("--output-dir", default=str(_repo_root / "docs" / "research" / "devlog" / "figures"))
    parser.add_argument("--window", type=int, default=50, help="Rolling-mean window in steps.")
    args = parser.parse_args(argv)

    experiments = Path(args.experiments_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    plot_dynamics(experiments, out_dir / "intrinsic_cooperation_pilot_dynamics.png", args.window)
    plot_lineage_depth(experiments, out_dir / "intrinsic_cooperation_lineage_depth.png")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
