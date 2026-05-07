#!/usr/bin/env python3
"""Render a two-track chromosome anatomy diagram for hyperparameter evolution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke


@dataclass(frozen=True)
class GeneGroup:
    """A semantic group of related genes."""

    name: str
    genes: Sequence[str]
    color: str


@dataclass(frozen=True)
class ChromosomeTrack:
    """A logical chromosome track in the combined gene vector."""

    name: str
    subtitle: str
    groups: Sequence[GeneGroup]


CHROMOSOME_TRACKS: Sequence[ChromosomeTrack] = (
    ChromosomeTrack(
        name="Chromosome A",
        subtitle="Learning / RL hyperparameters",
        groups=(
            GeneGroup(
                name="Core DQN",
                genes=(
                    "learning_rate",
                    "gamma",
                    "tau",
                    "batch_size",
                    "target_update_freq",
                    "dqn_hidden_size",
                    "rl_train_freq",
                    "memory_size",
                ),
                color="#4C78A8",
            ),
            GeneGroup(
                name="Exploration schedule",
                genes=("epsilon_start", "epsilon_min", "epsilon_decay"),
                color="#5BA3A3",
            ),
            GeneGroup(
                name="Prioritized replay",
                genes=("per_alpha", "per_beta_start", "per_beta_end"),
                color="#54A24B",
            ),
            GeneGroup(
                name="Ensembling",
                genes=("ensemble_size",),
                color="#E89E36",
            ),
        ),
    ),
    ChromosomeTrack(
        name="Chromosome B",
        subtitle="Action-policy priors",
        groups=(
            GeneGroup(
                name="Base action weights",
                genes=(
                    "move_weight",
                    "gather_weight",
                    "share_weight",
                    "attack_weight",
                    "reproduce_weight",
                ),
                color="#D6604D",
            ),
            GeneGroup(
                name="State-conditional multipliers",
                genes=(
                    "move_mult_no_resources",
                    "gather_mult_low_resources",
                    "share_mult_wealthy",
                    "share_mult_poor",
                    "attack_mult_desperate",
                    "attack_mult_stable",
                    "reproduce_mult_wealthy",
                    "reproduce_mult_poor",
                ),
                color="#A26FA0",
            ),
            GeneGroup(
                name="Policy thresholds",
                genes=(
                    "attack_starvation_threshold",
                    "attack_defense_threshold",
                    "reproduce_resource_threshold",
                ),
                color="#8C6A55",
            ),
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot chromosome A/B anatomy as a two-track gene map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chromosome_map.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved image DPI.",
    )
    return parser.parse_args()


def _group_spans(track: ChromosomeTrack) -> List[Dict[str, int]]:
    spans: List[Dict[str, int]] = []
    cursor = 0
    for group in track.groups:
        start = cursor
        end = start + len(group.genes)
        spans.append({"start": start, "end": end})
        cursor = end
    return spans


def _contrast_text_color(hex_color: str) -> str:
    """Pick black or white text for readability against ``hex_color``."""

    rgb = [int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]

    def _to_linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _to_linear(rgb[0]) + 0.7152 * _to_linear(rgb[1]) + 0.0722 * _to_linear(rgb[2])
    return "#FFFFFF" if luminance < 0.5 else "#1F2933"


def _bracket_path(x: float, y_top: float, y_bottom: float, depth: float) -> MplPath:
    """Build a square bracket path: |‾  spanning the group on the right side."""

    vertices = [
        (x, y_top),
        (x + depth, y_top),
        (x + depth, y_bottom),
        (x, y_bottom),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
    return MplPath(vertices, codes)


def _draw_track(
    ax: plt.Axes,
    track: ChromosomeTrack,
    x_base: float,
    col_width: float,
    max_gene_count: int,
) -> None:
    spans = _group_spans(track)
    all_genes = [gene for group in track.groups for gene in group.genes]
    total_genes = len(all_genes)
    y_offset = (max_gene_count - total_genes) / 2.0
    track_letter = track.name.split()[-1]

    gene_color_map: Dict[int, str] = {}
    for group, span in zip(track.groups, spans):
        for offset in range(span["end"] - span["start"]):
            gene_color_map[span["start"] + offset] = group.color

    for group, span in zip(track.groups, spans):
        height = span["end"] - span["start"]
        y_bottom = y_offset + (total_genes - span["end"])
        y_top = y_bottom + height
        y_center = y_bottom + (height / 2.0)

        shadow = FancyBboxPatch(
            (x_base + 0.04, y_bottom - 0.05),
            col_width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#1F2933",
            edgecolor="none",
            alpha=0.18,
            zorder=1,
        )
        ax.add_patch(shadow)

        group_patch = FancyBboxPatch(
            (x_base, y_bottom),
            col_width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=group.color,
            edgecolor="white",
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )
        ax.add_patch(group_patch)

        bracket_x = x_base + col_width + 0.08
        bracket = PathPatch(
            _bracket_path(bracket_x, y_top - 0.05, y_bottom + 0.05, depth=0.18),
            edgecolor=group.color,
            facecolor="none",
            linewidth=1.6,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(bracket)

        gene_word = "gene" if height == 1 else "genes"
        ax.text(
            bracket_x + 0.30,
            y_center + 0.12,
            group.name,
            ha="left",
            va="center",
            fontsize=10.5,
            color=group.color,
            fontweight="bold",
        )
        ax.text(
            bracket_x + 0.30,
            y_center - 0.28,
            f"{height} {gene_word}",
            ha="left",
            va="center",
            fontsize=8.6,
            color="#52606D",
        )

    for idx, gene in enumerate(all_genes):
        y_bottom = y_offset + (total_genes - idx - 1)
        y_center = y_bottom + 0.5
        gene_color = gene_color_map[idx]
        text_color = _contrast_text_color(gene_color)

        ax.plot(
            [x_base + 0.04, x_base + col_width - 0.04],
            [y_bottom, y_bottom],
            color="white",
            linewidth=0.8,
            alpha=0.55,
            zorder=3,
        )

        index_label = f"{track_letter}{idx + 1:02d}"
        ax.text(
            x_base + 0.10,
            y_center,
            index_label,
            ha="left",
            va="center",
            fontsize=7.8,
            color=text_color,
            alpha=0.75,
            family="monospace",
            zorder=4,
        )

        ax.text(
            x_base + 0.45,
            y_center,
            gene,
            ha="left",
            va="center",
            fontsize=9.6,
            color=text_color,
            zorder=4,
        )

    header_y = max_gene_count + 1.15
    ax.text(
        x_base + (col_width / 2),
        header_y + 0.55,
        track.name,
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#1F2933",
    )
    ax.text(
        x_base + (col_width / 2),
        header_y + 0.05,
        track.subtitle,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#52606D",
    )
    ax.text(
        x_base + (col_width / 2),
        header_y - 0.45,
        f"{total_genes} genes",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#52606D",
        fontstyle="italic",
    )


def _track_extent(track: ChromosomeTrack) -> int:
    return sum(len(group.genes) for group in track.groups)


def _plot_dimensions() -> Tuple[float, float, float, float, float]:
    """Return (col_width, left_x, right_x, x_lim, gene_count)."""

    gene_count = max(_track_extent(track) for track in CHROMOSOME_TRACKS)
    col_width = 2.45
    left_x = 0.55
    right_x = 5.65
    x_lim = right_x + col_width + 2.95
    return col_width, left_x, right_x, x_lim, gene_count


def _render_chromosome_map(output: Path, dpi: int) -> None:
    col_width, left_x, right_x, x_lim, gene_count = _plot_dimensions()
    total_genes = sum(_track_extent(track) for track in CHROMOSOME_TRACKS)

    fig, ax = plt.subplots(figsize=(14.5, 11.2))
    fig.patch.set_facecolor("#F5F6F8")
    ax.set_facecolor("#F5F6F8")

    _draw_track(ax, CHROMOSOME_TRACKS[0], left_x, col_width, gene_count)
    _draw_track(ax, CHROMOSOME_TRACKS[1], right_x, col_width, gene_count)

    center_x = x_lim / 2.0

    title_y = gene_count + 3.20
    subtitle_y = gene_count + 2.55

    ax.text(
        center_x,
        title_y,
        "Hyperparameter Chromosome Anatomy",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#1F2933",
        path_effects=[withStroke(linewidth=3, foreground="#F5F6F8")],
    )
    a_count = _track_extent(CHROMOSOME_TRACKS[0])
    b_count = _track_extent(CHROMOSOME_TRACKS[1])
    ax.text(
        center_x,
        subtitle_y,
        (
            f"Per-agent genome: {total_genes} genes "
            f"({CHROMOSOME_TRACKS[0].name}: {a_count}  +  {CHROMOSOME_TRACKS[1].name}: {b_count}) "
            "— combined into a single vector for crossover and mutation"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#52606D",
        fontstyle="italic",
    )

    footer_y = -0.65
    ax.text(
        center_x,
        footer_y,
        (
            f"Index labels (A01–A{a_count:02d}, B01–B{b_count:02d}) reflect each gene's position in the combined vector.  "
            "Inheritance is Baldwinian: only this chromosome is passed on; learned policies are not."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#7A8794",
    )

    ax.set_xlim(0, x_lim)
    ax.set_ylim(-1.2, gene_count + 3.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    _render_chromosome_map(args.output, args.dpi)
    print(f"Wrote chromosome map figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
