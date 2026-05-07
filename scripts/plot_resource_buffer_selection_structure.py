#!/usr/bin/env python3
"""Plot selection-sign flips and cluster-separation trajectories by resource buffer.

This script renders a two-panel figure:
  A) Signed gene-effect changes (percent) by resource regime, split into
     "sign-flipping" and "convergent" genes with row shading.
  B) Speciation-index trajectories over time, annotated by trajectory shape
     and endpoint value.

It can run from built-in defaults (matching the stable profile comparison docs)
or from CSV inputs.

CSV schema:
- selection CSV columns:  gene,regime,effect_pct
- speciation CSV columns: regime,step,speciation_index
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REGIME_ORDER: Sequence[str] = ("conservative", "balanced", "buffered")

REGIME_COLORS: Dict[str, str] = {
    "conservative": "#B8580E",
    "balanced": "#6B7280",
    "buffered": "#2E7D32",
}

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "conservative": "8 res / 32 nodes / 0.14 regen",
    "balanced": "10 res / 34 nodes / 0.15 regen",
    "buffered": "12 res / 36 nodes / 0.16 regen",
}

TRAJECTORY_LABELS: Dict[str, str] = {
    "conservative": "merging",
    "balanced": "V-shape",
    "buffered": "diverging",
}

PAGE_BG = "#F8FAFC"
FLIP_ROW_BG = "#FEF3C7"
TEXT_DARK = "#1F2933"
TEXT_MUTED = "#52606D"


@dataclass(frozen=True)
class SelectionPoint:
    gene: str
    regime: str
    effect_pct: float


@dataclass(frozen=True)
class SpeciationPoint:
    regime: str
    step: int
    speciation_index: float


DEFAULT_SELECTION: Sequence[SelectionPoint] = (
    SelectionPoint("attack_weight", "conservative", -20.6),
    SelectionPoint("attack_weight", "balanced", -25.0),
    SelectionPoint("attack_weight", "buffered", -9.3),
    SelectionPoint("share_weight", "conservative", -5.8),
    SelectionPoint("share_weight", "balanced", -27.7),
    SelectionPoint("share_weight", "buffered", -25.6),
    SelectionPoint("attack_mult_desperate", "conservative", -3.7),
    SelectionPoint("attack_mult_desperate", "balanced", -13.6),
    SelectionPoint("attack_mult_desperate", "buffered", -14.7),
    SelectionPoint("move_mult_no_resources", "conservative", -20.6),
    SelectionPoint("move_mult_no_resources", "balanced", -22.2),
    SelectionPoint("move_mult_no_resources", "buffered", -15.4),
    SelectionPoint("memory_size", "conservative", -27.8),
    SelectionPoint("memory_size", "balanced", -8.5),
    SelectionPoint("memory_size", "buffered", -24.4),
    SelectionPoint("dqn_hidden_size", "conservative", -13.9),
    SelectionPoint("dqn_hidden_size", "balanced", -21.9),
    SelectionPoint("dqn_hidden_size", "buffered", -12.3),
    SelectionPoint("epsilon_start", "conservative", -5.1),
    SelectionPoint("epsilon_start", "balanced", -6.2),
    SelectionPoint("epsilon_start", "buffered", -8.1),
    SelectionPoint("per_alpha", "conservative", 12.3),
    SelectionPoint("per_alpha", "balanced", 11.1),
    SelectionPoint("per_alpha", "buffered", 3.6),
    SelectionPoint("target_update_freq", "conservative", 21.9),
    SelectionPoint("target_update_freq", "balanced", 33.5),
    SelectionPoint("target_update_freq", "buffered", 5.1),
    SelectionPoint("learning_rate", "conservative", -8.3),
    SelectionPoint("learning_rate", "balanced", -6.0),
    SelectionPoint("learning_rate", "buffered", 23.1),
    SelectionPoint("ensemble_size", "conservative", -25.9),
    SelectionPoint("ensemble_size", "balanced", -8.8),
    SelectionPoint("ensemble_size", "buffered", 2.6),
    SelectionPoint("reproduce_mult_wealthy", "conservative", -4.6),
    SelectionPoint("reproduce_mult_wealthy", "balanced", 0.4),
    SelectionPoint("reproduce_mult_wealthy", "buffered", 8.4),
    SelectionPoint("reproduce_mult_poor", "conservative", 21.2),
    SelectionPoint("reproduce_mult_poor", "balanced", 0.8),
    SelectionPoint("reproduce_mult_poor", "buffered", -5.2),
    SelectionPoint("gamma", "conservative", -3.0),
    SelectionPoint("gamma", "balanced", 0.3),
    SelectionPoint("gamma", "buffered", -2.7),
)

DEFAULT_SPECIATION: Sequence[SpeciationPoint] = (
    SpeciationPoint("conservative", 50, 0.732),
    SpeciationPoint("conservative", 500, 0.713),
    SpeciationPoint("conservative", 1000, 0.684),
    SpeciationPoint("balanced", 50, 0.708),
    SpeciationPoint("balanced", 500, 0.654),
    SpeciationPoint("balanced", 1000, 0.711),
    SpeciationPoint("buffered", 50, 0.653),
    SpeciationPoint("buffered", 500, 0.729),
    SpeciationPoint("buffered", 1000, 0.753),
)

FLIP_THRESHOLD_PCT = 1.5  # values within this band of zero count as "near zero"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot resource-buffer effects on selection and cluster structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: gene,regime,effect_pct",
    )
    parser.add_argument(
        "--speciation-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: regime,step,speciation_index",
    )
    parser.add_argument(
        "--top-n-genes",
        type=int,
        default=12,
        help="Show top N genes (sign-flipping first by flip magnitude, then convergent by mean absolute effect size).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/devlog/figures/resource_buffer_selection_structure.png"),
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Saved image DPI.")
    return parser.parse_args()


def _read_selection_csv(path: Path) -> List[SelectionPoint]:
    rows: List[SelectionPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gene", "regime", "effect_pct"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
            raise ValueError(f"Selection CSV missing required columns: {missing}")
        for row in reader:
            rows.append(
                SelectionPoint(
                    gene=row["gene"].strip(),
                    regime=row["regime"].strip(),
                    effect_pct=float(row["effect_pct"]),
                )
            )
    return rows


def _read_speciation_csv(path: Path) -> List[SpeciationPoint]:
    rows: List[SpeciationPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"regime", "step", "speciation_index"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
            raise ValueError(f"Speciation CSV missing required columns: {missing}")
        for row in reader:
            rows.append(
                SpeciationPoint(
                    regime=row["regime"].strip(),
                    step=int(row["step"]),
                    speciation_index=float(row["speciation_index"]),
                )
            )
    return rows


def _group_by_gene(points: Iterable[SelectionPoint]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for point in points:
        grouped.setdefault(point.gene, {})[point.regime] = point.effect_pct
    return grouped


def _validate_regimes(regimes: Iterable[str], context: str) -> None:
    unknown = sorted(set(regimes) - set(REGIME_ORDER))
    if unknown:
        unknown_text = ", ".join(unknown)
        expected_text = ", ".join(REGIME_ORDER)
        raise ValueError(f"Unknown regime in {context}: {unknown_text}. Expected only: {expected_text}")


def _is_sign_flipping(values: Dict[str, float]) -> bool:
    """A gene flips sign if it has at least one meaningfully positive and one negative regime value."""

    has_pos = any(v > FLIP_THRESHOLD_PCT for v in values.values())
    has_neg = any(v < -FLIP_THRESHOLD_PCT for v in values.values())
    return has_pos and has_neg


def _classify_and_order_genes(
    selection_points: Sequence[SelectionPoint],
    top_n: int,
) -> Tuple[List[str], List[str]]:
    """Return (flipping_genes, convergent_genes) ordered for display.

    Top-N is applied to the union; flipping genes are kept first.
    """

    grouped = _group_by_gene(selection_points)
    flipping: List[Tuple[str, float]] = []
    convergent: List[Tuple[str, float]] = []
    for gene, values in grouped.items():
        magnitude = float(np.mean([abs(v) for v in values.values()])) if values else 0.0
        if _is_sign_flipping(values):
            flip_amount = max(values.values()) - min(values.values())
            flipping.append((gene, flip_amount))
        else:
            convergent.append((gene, magnitude))

    flipping.sort(key=lambda item: item[1], reverse=True)
    convergent.sort(key=lambda item: item[1], reverse=True)

    flip_names = [name for name, _ in flipping][:top_n]
    conv_names = [name for name, _ in convergent]

    remaining = max(0, top_n - len(flip_names))
    return flip_names, conv_names[:remaining]


def _plot_selection_panel(
    ax: plt.Axes,
    selection_points: Sequence[SelectionPoint],
    flipping_genes: Sequence[str],
    convergent_genes: Sequence[str],
) -> None:
    grouped = _group_by_gene(selection_points)
    ordered_genes = list(flipping_genes) + list(convergent_genes)
    n = len(ordered_genes)
    y_positions = np.arange(n)
    flip_set = set(flipping_genes)
    offsets = {"conservative": -0.24, "balanced": 0.0, "buffered": 0.24}

    all_effects = [point.effect_pct for point in selection_points]
    x_max = max(abs(min(all_effects, default=0.0)), abs(max(all_effects, default=0.0))) + 5.0

    ax.set_xlim(-x_max, x_max)

    for idx, gene in enumerate(ordered_genes):
        if gene in flip_set:
            ax.add_patch(
                Rectangle(
                    (-x_max, idx - 0.5),
                    2 * x_max,
                    1.0,
                    facecolor=FLIP_ROW_BG,
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0,
                )
            )

    if flipping_genes and convergent_genes:
        boundary_y = len(flipping_genes) - 0.5
        ax.axhline(boundary_y, color="#9CA3AF", lw=0.8, ls="-", alpha=0.7, zorder=1)

    ax.axvline(0, color="#374151", lw=1.0, ls="--", alpha=0.85, zorder=1)

    for regime in REGIME_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        for i, gene in enumerate(ordered_genes):
            regime_map = grouped.get(gene, {})
            if regime not in regime_map:
                continue
            effect = regime_map[regime]
            y = y_positions[i] + offsets[regime]
            ax.plot([0, effect], [y, y], color=REGIME_COLORS[regime], lw=2.2, alpha=0.9, zorder=2)
            xs.append(effect)
            ys.append(y)
        ax.scatter(xs, ys, s=34, color=REGIME_COLORS[regime], edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_yticks(y_positions)
    tick_labels = []
    for gene in ordered_genes:
        if gene in flip_set:
            tick_labels.append(f"{gene}")
        else:
            tick_labels.append(gene)
    ax.set_yticklabels(tick_labels, fontsize=9.5)
    for tick_label, gene in zip(ax.get_yticklabels(), ordered_genes):
        if gene in flip_set:
            tick_label.set_fontweight("bold")
            tick_label.set_color(TEXT_DARK)
        else:
            tick_label.set_color(TEXT_MUTED)

    ax.invert_yaxis()
    ax.set_xlabel("Selection effect: percent change in mean from step 0 \u2192 step 1000", color=TEXT_DARK)
    ax.set_title("A) Signed selection effects by gene", loc="left", fontsize=12.5, fontweight="bold", color=TEXT_DARK)
    ax.grid(axis="x", alpha=0.25, ls=":", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors=TEXT_MUTED)

    arrow_y = -0.85
    label_y = -1.55
    ax.annotate(
        "",
        xy=(-x_max * 0.55, arrow_y),
        xytext=(-x_max * 0.05, arrow_y),
        arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.2),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(x_max * 0.55, arrow_y),
        xytext=(x_max * 0.05, arrow_y),
        arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.2),
        annotation_clip=False,
    )
    ax.text(
        -x_max * 0.30,
        label_y,
        "decreased",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )
    ax.text(
        x_max * 0.30,
        label_y,
        "increased",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )

    if flipping_genes:
        ax.text(
            x_max * 0.97,
            (len(flipping_genes) - 1) / 2.0,
            "sign-flipping",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            color="#92400E",
            fontweight="bold",
            alpha=0.85,
        )
    if convergent_genes:
        conv_center = len(flipping_genes) + (len(convergent_genes) - 1) / 2.0
        ax.text(
            x_max * 0.97,
            conv_center,
            "convergent",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            color=TEXT_MUTED,
            fontweight="bold",
            alpha=0.85,
        )

    ax.set_ylim(n - 0.5, -2.1)


def _plot_speciation_panel(ax: plt.Axes, points: Sequence[SpeciationPoint]) -> None:
    by_regime: Dict[str, List[SpeciationPoint]] = {}
    for point in points:
        by_regime.setdefault(point.regime, []).append(point)

    all_y = [row.speciation_index for row in points]
    all_x = [row.step for row in points]
    if not all_y or not all_x:
        return
    y_min = min(all_y) - 0.012
    y_max = max(all_y) + 0.022
    x_min = min(all_x)
    x_max = max(all_x)
    x_pad = (x_max - x_min) * 0.18

    for regime in REGIME_ORDER:
        regime_points = sorted(by_regime.get(regime, []), key=lambda row: row.step)
        if not regime_points:
            continue
        xs = [row.step for row in regime_points]
        ys = [row.speciation_index for row in regime_points]
        color = REGIME_COLORS[regime]
        ax.plot(xs, ys, marker="o", ms=6, lw=2.4, color=color, alpha=0.95, zorder=3)
        ax.scatter(xs, ys, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=4)

        end_step = xs[-1]
        end_value = ys[-1]
        shape_label = TRAJECTORY_LABELS.get(regime, "")
        annotation = f"{regime}\n{end_value:.3f} ({shape_label})"
        ax.annotate(
            annotation,
            xy=(end_step, end_value),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            va="center",
            ha="left",
        )

    ax.set_xlim(x_min - x_pad * 0.2, x_max + x_pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Simulation step", color=TEXT_DARK)
    ax.set_ylabel("Speciation index (silhouette of GMM clusters)", color=TEXT_DARK)
    ax.set_title("B) Cluster separation over time", loc="left", fontsize=12.5, fontweight="bold", color=TEXT_DARK)
    ax.grid(alpha=0.25, ls=":", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors=TEXT_MUTED)


def _build_legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for regime in REGIME_ORDER:
        label = f"{regime}  ({REGIME_DESCRIPTIONS[regime]})"
        handles.append(
            Line2D(
                [0],
                [0],
                color=REGIME_COLORS[regime],
                marker="o",
                lw=2.4,
                ms=7,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=label,
            )
        )
    return handles


def _load_selection_points(path: Path | None) -> List[SelectionPoint]:
    return _read_selection_csv(path) if path else list(DEFAULT_SELECTION)


def _load_speciation_points(path: Path | None) -> List[SpeciationPoint]:
    return _read_speciation_csv(path) if path else list(DEFAULT_SPECIATION)


def _render_figure(
    output_path: Path,
    selection_points: Sequence[SelectionPoint],
    speciation_points: Sequence[SpeciationPoint],
    top_n_genes: int,
    dpi: int,
) -> None:
    _validate_regimes((row.regime for row in selection_points), context="selection data")
    _validate_regimes((row.regime for row in speciation_points), context="speciation data")

    flipping_genes, convergent_genes = _classify_and_order_genes(selection_points, top_n=top_n_genes)
    n_genes = len(flipping_genes) + len(convergent_genes)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(15.5, max(7.0, 4.2 + 0.42 * n_genes)),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    fig.patch.set_facecolor(PAGE_BG)
    ax_left.set_facecolor(PAGE_BG)
    ax_right.set_facecolor(PAGE_BG)

    _plot_selection_panel(ax_left, selection_points, flipping_genes, convergent_genes)
    _plot_speciation_panel(ax_right, speciation_points)

    fig.suptitle(
        "Resource Buffer Shapes Selection Direction and Population Structure",
        fontsize=15,
        fontweight="bold",
        color=TEXT_DARK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        "Three intrinsic-evolution runs differing only in initial resources, node count, and regen rate "
        "(seed=42, 1000 logged steps, low selection pressure, no crossover)",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_MUTED,
        style="italic",
    )

    fig.legend(
        handles=_build_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9.5,
        handletextpad=0.6,
        columnspacing=2.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.93))
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    selection_points = _load_selection_points(args.selection_csv)
    speciation_points = _load_speciation_points(args.speciation_csv)
    _render_figure(
        output_path=args.output,
        selection_points=selection_points,
        speciation_points=speciation_points,
        top_n_genes=args.top_n_genes,
        dpi=args.dpi,
    )
    print(f"Wrote figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
