"""Produced overview animation of the consensus experiment (manim).

Unlike ``animate.py`` (which renders raw simulation frames of one trial with
matplotlib), this module builds a presentation-style explainer for a general
audience: what the experiment asks, how the world is set up, the four
selection paradigms, and the headline results. All result numbers are loaded
from a run's ``summary.csv`` so the video can never drift from the data.

Requires the optional ``manim`` dependency (``pip install manim``) plus ffmpeg.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Star,
    Text,
    VGroup,
    Write,
    tempconfig,
)
from numpy.random import default_rng

PARADIGM_COLORS = {
    "party": "#e74c3c",
    "individual": "#3498db",
    "score": "#2ecc71",
    "latent_match": "#af7ac5",
}
ACCENT = "#f1c40f"
MUTED = "#aab2bd"

PARADIGM_ONE_LINERS = {
    "party": "Two parties nominate.\nVoters pick a side.",
    "individual": "No parties.\nVote your nearest candidate.",
    "score": "Rate every candidate 0\u201310.\nBest average wins.",
    "latent_match": "Elect the closest match\nto the average voter.",
}

PROJECT_NAMES = ("Core services", "Coalition club", "Outgroup repair", "Prestige", "Buffer")


@dataclass(frozen=True)
class OverviewNumbers:
    """Real aggregates the video displays, keyed by paradigm."""

    loser_welfare: dict
    lambda_winner: dict
    lambda_correlated: dict


def load_numbers(default_results: Path, correlated_results: Path) -> OverviewNumbers:
    """Read the displayed aggregates from two runs' summary.csv files."""

    def by_paradigm(path: Path, column: str) -> dict:
        summary = pd.read_csv(path / "summary.csv")
        return dict(zip(summary["paradigm"], summary[column]))

    return OverviewNumbers(
        loser_welfare=by_paradigm(default_results, "loser_welfare_mean"),
        lambda_winner=by_paradigm(default_results, "lambda_winner_mean"),
        lambda_correlated=by_paradigm(correlated_results, "lambda_winner_mean"),
    )


def _format_range(values) -> str:
    values = list(values)
    lo, hi = f"{min(values):.2f}", f"{max(values):.2f}"
    return lo if lo == hi else f"{lo}\u2013{hi}"


class ConsensusOverview(Scene):
    """~80 second audience explainer of the experiment and its results."""

    def __init__(self, numbers: OverviewNumbers, **kwargs):
        super().__init__(**kwargs)
        self.numbers = numbers

    def construct(self):
        self._title_card()
        self._world()
        self._rule()
        self._paradigms()
        self._results()
        self._takeaway()

    def _clear(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    def _title_card(self):
        title = Text("After the election, who gets helped?", font_size=46, weight=BOLD)
        sub = Text(
            "A simulation of four ways to pick a leader \u2014 and what each does to the losers",
            font_size=26,
            color=MUTED,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(title), run_time=2)
        self.play(FadeIn(sub))
        self.wait(2.2)
        self._clear()

    def _world(self):
        rng = default_rng(7)
        bloc_a = VGroup(
            *[
                Dot([x, y, 0], radius=0.055, color="#e67e22")
                for x, y in rng.normal([-3.6, 1.0], 0.75, size=(70, 2))
            ]
        )
        bloc_b = VGroup(
            *[
                Dot([x, y, 0], radius=0.055, color="#5dade2")
                for x, y in rng.normal([-1.4, -1.2], 0.75, size=(70, 2))
            ]
        )
        caption = Text("400 voters, split into rival blocs", font_size=30).to_edge(RIGHT, buff=0.8).shift(UP * 1.6)
        self.play(FadeIn(bloc_a, lag_ratio=0.01), FadeIn(bloc_b, lag_ratio=0.01), FadeIn(caption), run_time=2)
        self.wait(1.2)

        stars = VGroup(
            *[
                Star(outer_radius=0.16, color=ACCENT, fill_opacity=1).move_to([x, y, 0])
                for x, y in rng.normal([-2.5, 0.0], 1.3, size=(8, 2))
            ]
        )
        caption2 = Text(
            "8 candidates \u2014 each with a platform\nand a hidden loyalty trait \u03bb",
            font_size=30,
            t2c={"\u03bb": ACCENT},
        ).next_to(caption, DOWN, buff=0.7)
        self.play(FadeIn(stars, lag_ratio=0.1), FadeIn(caption2), run_time=1.6)
        self.wait(2)

        boxes = VGroup()
        for name in PROJECT_NAMES:
            box = RoundedRectangle(corner_radius=0.12, width=2.4, height=0.8, color=MUTED)
            # Re-create long labels at a smaller font size (scaling collapses word gaps).
            label = Text(name, font_size=20)
            if label.width > box.width - 0.3:
                label = Text(name, font_size=15)
            label.move_to(box)
            boxes.add(VGroup(box, label))
        boxes.arrange(RIGHT, buff=0.25).to_edge(DOWN, buff=0.6)
        caption3 = Text("The winner splits one fixed budget across 5 projects", font_size=28).next_to(
            boxes, UP, buff=0.5
        )
        self.play(FadeIn(caption3), LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in boxes], lag_ratio=0.15))
        self.wait(2.4)
        self._clear()

    def _rule(self):
        header = Text("The winner's spending rule", font_size=38, weight=BOLD).to_edge(UP, buff=0.9)
        formula = Text(
            "allocation  =  \u03bb \u00b7 help my supporters  +  (1\u2212\u03bb) \u00b7 help everyone",
            font_size=32,
            t2c={"\u03bb": ACCENT},
        )
        legend = Text(
            "\u03bb = 1  \u2192  serve only your own voters        \u03bb = 0  \u2192  serve the whole town",
            font_size=26,
            color=MUTED,
            t2c={"\u03bb": ACCENT},
        ).next_to(formula, DOWN, buff=0.8)
        warning = Text("Voters never see \u03bb on the ballot.", font_size=26, t2c={"\u03bb": ACCENT}).next_to(
            legend, DOWN, buff=0.8
        )
        self.play(FadeIn(header))
        self.play(Write(formula), run_time=2.2)
        self.play(FadeIn(legend))
        self.wait(1.4)
        self.play(FadeIn(warning))
        self.wait(2.2)
        self._clear()

    def _paradigms(self):
        header = Text("Four ways to pick the winner", font_size=38, weight=BOLD).to_edge(UP, buff=0.7)
        cards = VGroup()
        for name, color in PARADIGM_COLORS.items():
            title = Text(name.replace("_", " "), font_size=28, weight=BOLD, color=color)
            body = Text(PARADIGM_ONE_LINERS[name], font_size=22, color=MUTED, line_spacing=0.9)
            box = RoundedRectangle(corner_radius=0.15, width=5.6, height=2.2, color=color)
            title.move_to(box.get_top() + DOWN * 0.5)
            body.move_to(box.get_center() + DOWN * 0.35)
            cards.add(VGroup(box, title, body))
        cards.arrange_in_grid(rows=2, cols=2, buff=0.45).next_to(header, DOWN, buff=0.55)
        self.play(FadeIn(header))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.3), run_time=3)
        self.wait(3)
        self._clear()

    def _results(self):
        header = Text(
            "Result: how well do non-supporters do?", font_size=36, weight=BOLD
        ).to_edge(UP, buff=0.7)
        self.play(FadeIn(header))

        rows = VGroup()
        max_value = max(self.numbers.loser_welfare.values())
        bar_left, max_bar_width, top_y, row_step = -3.6, 5.8, 1.9, 1.05
        for i, (name, color) in enumerate(PARADIGM_COLORS.items()):
            value = self.numbers.loser_welfare[name]
            y = top_y - i * row_step
            bar = Rectangle(
                width=max_bar_width * value / max_value,
                height=0.5,
                color=color,
                fill_color=color,
                fill_opacity=0.85,
                stroke_width=0,
            ).move_to([bar_left, y, 0], aligned_edge=LEFT)
            label = Text(name.replace("_", " "), font_size=26, color=color, weight=BOLD)
            label.next_to(bar.get_left(), LEFT, buff=0.4)
            number = Text(f"{value:.3f}", font_size=24).next_to(bar, RIGHT, buff=0.3)
            rows.add(VGroup(label, bar, number))

        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.15))
        self.play(
            LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(r[2]) for r in rows], lag_ratio=0.2),
            run_time=2.4,
        )
        footnote = Text(
            "mean utility of voters who did NOT back the winner (250 seeded trials)",
            font_size=22,
            color=MUTED,
        ).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(footnote))
        self.wait(2.8)
        self.play(FadeOut(footnote))

        lam_line = Text(
            f"Winners' loyalty: \u03bb \u2248 {_format_range(self.numbers.lambda_winner.values())} "
            "under every rule \u2014 no rule picked kinder people.",
            font_size=24,
            t2c={"\u03bb": ACCENT},
        ).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(lam_line))
        self.wait(3)

        correlated = (self.numbers.lambda_correlated[name] for name in ("score", "latent_match"))
        twist = Text(
            f"Twist: if platforms reveal loyalty, consensus rules elect \u03bb \u2248 {_format_range(correlated)} winners.",
            font_size=23,
            t2c={"\u03bb": ACCENT},
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(twist))
        self.wait(3.2)
        self._clear()

    def _takeaway(self):
        line1 = Text("Selection rules didn't pick kinder winners \u2014", font_size=36, weight=BOLD)
        line2 = Text("they changed who the winner owes.", font_size=36, weight=BOLD, color=ACCENT)
        line2.next_to(line1, DOWN, buff=0.4)
        credit = Text(
            "AgentFarm \u00b7 consensus experiment \u00b7 seeded & reproducible",
            font_size=22,
            color=MUTED,
        ).to_edge(DOWN, buff=0.8)
        self.play(Write(line1), run_time=1.6)
        self.play(Write(line2), run_time=1.6)
        self.play(FadeIn(credit))
        self.wait(3)


def render_overview(
    out_path: Path,
    default_results: Path,
    correlated_results: Path,
    fps: int = 30,
    quality: str = "high",
) -> Path:
    """Render the produced overview MP4 from real run outputs."""
    numbers = load_numbers(default_results, correlated_results)
    width, height = (1280, 720) if quality == "high" else (854, 480)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as media_dir, tempconfig(
        {
            "pixel_width": width,
            "pixel_height": height,
            "frame_rate": fps,
            "media_dir": media_dir,
            "output_file": "consensus_overview",
            "disable_caching": True,
            "progress_bar": "none",
            "verbosity": "WARNING",
        }
    ):
        scene = ConsensusOverview(numbers)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)
        out_path.write_bytes(produced.read_bytes())
    return out_path
