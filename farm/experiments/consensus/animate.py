"""Social-media-ready MP4 animation of the consensus experiment dynamic.

Renders one real trial (same voters, same candidates) under every paradigm in
sequence: voters and candidates appear in preference space, the election
splits supporters from non-supporters, the winner's budget allocation grows,
and the welfare readout lands. A final panel compares the paradigms.

Output targets X/Twitter: 1280x720 H.264, ~30 seconds, large text, no audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from farm.experiments.consensus.allocation import allocate
from farm.experiments.consensus.paradigms import PARADIGMS, run_election
from farm.experiments.consensus.population import (
    PROJECTS,
    Candidates,
    Population,
    generate_candidates,
    generate_population,
)

PARADIGM_COLORS = {
    "party": "#c0392b",
    "individual": "#2980b9",
    "score": "#27ae60",
    "latent_match": "#8e44ad",
}

TAGLINES = {
    "party": "Two parties nominate. Voters pick a side.",
    "individual": "No parties. Vote for your nearest candidate.",
    "score": "Score every candidate 0-10. Best average wins.",
    "latent_match": "Elect whoever best matches the average voter.",
}

PROJECT_LABELS = ("Core\nservices", "Coalition\nclub", "Outgroup\nrepair", "Prestige", "Buffer")

LOSER_COLOR = "#b8b8b8"
CANDIDATE_COLOR = "#222222"
EVERYONE_ELSE_COLOR = "#e67e22"

# Segment lengths in seconds (per paradigm), then the final comparison hold.
INTRO_S, REVEAL_S, GROW_S, HOLD_S, FINAL_S = 1.2, 1.8, 1.6, 1.9, 5.0

X_DIM = PROJECTS.index("coalition_club")
Y_DIM = PROJECTS.index("outgroup_repair")


@dataclass(frozen=True)
class ParadigmOutcome:
    """Everything one animation segment needs about a paradigm's result."""

    name: str
    winner: int
    supporters: np.ndarray
    alloc: np.ndarray
    supporter_welfare: float
    loser_welfare: float
    gap: float
    loser_share: float


def _outcomes(population: Population, candidates: Candidates) -> list[ParadigmOutcome]:
    results = []
    for name in PARADIGMS:
        election = run_election(name, population, candidates)
        lam = float(candidates.lam[election.winner])
        alloc = allocate(population.benefits, election.supporters, candidates.platforms[election.winner], lam)
        utilities = population.benefits @ alloc
        supporters = election.supporters
        supporter_welfare = float(utilities[supporters].mean())
        loser_welfare = float(utilities[~supporters].mean())
        results.append(
            ParadigmOutcome(
                name=name,
                winner=election.winner,
                supporters=supporters,
                alloc=alloc,
                supporter_welfare=supporter_welfare,
                loser_welfare=loser_welfare,
                gap=supporter_welfare - loser_welfare,
                loser_share=float((~supporters).mean()),
            )
        )
    return results


def _draw_scatter(ax, population: Population, candidates: Candidates, outcome: ParadigmOutcome, revealed: bool):
    ax.clear()
    x, y = population.prefs[:, X_DIM], population.prefs[:, Y_DIM]
    if revealed:
        color = np.where(outcome.supporters, PARADIGM_COLORS[outcome.name], LOSER_COLOR)
        ax.scatter(x, y, s=22, c=color, alpha=0.75, linewidths=0)
    else:
        ax.scatter(x, y, s=22, c="#8a8a8a", alpha=0.6, linewidths=0)
    cx, cy = candidates.platforms[:, X_DIM], candidates.platforms[:, Y_DIM]
    ax.scatter(cx, cy, s=260, marker="*", c=CANDIDATE_COLOR, zorder=4)
    if revealed:
        ax.scatter(
            [cx[outcome.winner]],
            [cy[outcome.winner]],
            s=760,
            marker="*",
            c=PARADIGM_COLORS[outcome.name],
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
        )
    ax.set_xlabel("preference: coalition club →", fontsize=13)
    ax.set_ylabel("preference: outgroup repair →", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def _draw_bars(ax, outcome: ParadigmOutcome, growth: float):
    ax.clear()
    values = outcome.alloc * growth
    ax.bar(range(len(PROJECTS)), values, color=PARADIGM_COLORS[outcome.name], alpha=0.85)
    ax.set_xticks(range(len(PROJECTS)))
    ax.set_xticklabels(PROJECT_LABELS, fontsize=11)
    # Headroom above the tallest bar so value labels never clip at the top.
    ax.set_ylim(0, max(0.45, float(outcome.alloc.max()) + 0.07))
    ax.set_ylabel("share of budget", fontsize=13)
    ax.set_title("Winner's budget allocation", fontsize=15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if growth >= 1.0:
        for i, value in enumerate(outcome.alloc):
            ax.text(i, value + 0.012, f"{value:.2f}", ha="center", fontsize=11)


def _paradigm_segment(fig, ax_scatter, ax_bars, caption, metrics, population, candidates, outcome, fps, writer):
    fig.suptitle(
        f"{outcome.name.replace('_', ' ')}  —  {TAGLINES[outcome.name]}",
        fontsize=21,
        fontweight="bold",
        color=PARADIGM_COLORS[outcome.name],
    )

    def frames(seconds: float) -> int:
        return max(round(seconds * fps), 1)

    _draw_scatter(ax_scatter, population, candidates, outcome, revealed=False)
    _draw_bars(ax_bars, outcome, growth=0.0)
    caption.set_text(f"{len(population)} voters (dots), {len(candidates)} candidates (stars) — the same for every rule")
    for _ in range(frames(INTRO_S)):
        writer.grab_frame()

    _draw_scatter(ax_scatter, population, candidates, outcome, revealed=True)
    caption.set_text(f"Election held: colored dots supported the winner ({1 - outcome.loser_share:.0%} of voters)")
    for _ in range(frames(REVEAL_S)):
        writer.grab_frame()

    n_grow = frames(GROW_S)
    caption.set_text("The winner allocates a fixed budget across 5 public projects")
    for k in range(1, n_grow + 1):
        _draw_bars(ax_bars, outcome, growth=k / n_grow)
        writer.grab_frame()

    metrics.set_text(
        f"supporters {outcome.supporter_welfare:.3f}   vs   others {outcome.loser_welfare:.3f}\n"
        f"gap {outcome.gap:.3f}   ·   non-supporters {outcome.loser_share:.0%}"
    )
    metrics.set_visible(True)
    caption.set_text("Mean utility for supporters vs everyone else")
    for _ in range(frames(HOLD_S)):
        writer.grab_frame()


def _final_segment(fig, outcomes: list[ParadigmOutcome], fps: int, writer) -> None:
    fig.clear()
    fig.suptitle("Same voters. Same candidates. Different rules.", fontsize=23, fontweight="bold")
    ax = fig.add_axes((0.18, 0.16, 0.75, 0.6))
    names = [o.name.replace("_", " ") for o in outcomes]
    y = np.arange(len(outcomes))
    ax.barh(y + 0.2, [o.supporter_welfare for o in outcomes], height=0.36, color="#666666", label="supporters")
    ax.barh(
        y - 0.2,
        [o.loser_welfare for o in outcomes],
        height=0.36,
        color=EVERYONE_ELSE_COLOR,
        label="everyone else",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=15)
    for label, outcome in zip(ax.get_yticklabels(), outcomes):
        label.set_color(PARADIGM_COLORS[outcome.name])
        label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("mean utility (this trial)", fontsize=13)
    # Reverse legend entries so their order matches the visual stacking of the
    # bars, and park the legend above the axes so it never covers a bar.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=13, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.5,
        0.045,
        "Party selection rewards its bloc; individual-centered rules treat non-supporters better.\n"
        "One representative trial — 250-trial averages in the repo's REPORT.md",
        ha="center",
        fontsize=13,
        color="#444444",
    )
    for _ in range(max(round(FINAL_S * fps), 1)):
        writer.grab_frame()


def render_animation(
    out_path: Path,
    voters: int = 400,
    n_candidates: int = 8,
    population_type: str = "two_cluster",
    seed: int = 0,
    trial: int = 0,
    fps: int = 30,
) -> Path:
    """Render the paradigm-comparison MP4 for one reproducible trial."""
    from farm.experiments.consensus.experiment import ExperimentConfig, _trial_rng

    config = ExperimentConfig(voters=voters, candidates=n_candidates, population=population_type, seed=seed)
    rng = _trial_rng(config, trial)
    population = generate_population(rng, voters, population_type)
    candidates = generate_candidates(rng, n_candidates, population)
    outcomes = _outcomes(population, candidates)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=fps, codec="h264", bitrate=3200)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out_path), dpi=100):
        for outcome in outcomes:
            fig.clear()
            ax_scatter = fig.add_axes((0.06, 0.14, 0.42, 0.62))
            ax_bars = fig.add_axes((0.56, 0.14, 0.4, 0.62))
            caption = fig.text(0.5, 0.035, "", ha="center", fontsize=15, color="#333333")
            # Metrics live in figure space above the bar axes so they never collide with bars.
            metrics = fig.text(
                0.76,
                0.875,
                "",
                ha="center",
                va="center",
                fontsize=13.5,
                visible=False,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#999999"},
            )
            _paradigm_segment(
                fig, ax_scatter, ax_bars, caption, metrics, population, candidates, outcome, fps, writer
            )
        _final_segment(fig, outcomes, fps, writer)

    plt.close(fig)
    return out_path
