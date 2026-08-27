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
    AnimationGroup,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Star,
    SurroundingRectangle,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
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
BACKGROUND = "#12161d"

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
        self._question()
        self._world()
        self._rule()
        self._paradigms()
        self._results()
        self._takeaway()

    def _clear(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    def _kicker(self, label: str) -> Text:
        """Small uppercase chapter marker pinned to the top-left corner."""
        return Text(label.upper(), font_size=20, color=ACCENT, weight=BOLD).to_corner(UP + LEFT, buff=0.45)

    def _title_card(self):
        title = Text("After the election, who gets helped?", font_size=46, weight=BOLD)
        underline = Line(title.get_corner(DOWN + LEFT), title.get_corner(DOWN + RIGHT), color=ACCENT, stroke_width=5)
        underline.shift(DOWN * 0.25)
        sub = Text(
            "A simulation of four ways to pick a leader \u2014 and what each does to the losers",
            font_size=26,
            color=MUTED,
        ).next_to(underline, DOWN, buff=0.45)
        # FadeIn instead of Write throughout: Write's partially-stroked Pango
        # glyphs read as rendering glitches after social-media compression.
        self.play(FadeIn(title, shift=UP * 0.25), run_time=1)
        self.wait(0.4)
        self.play(Create(underline), run_time=0.6)
        self.play(FadeIn(sub), run_time=1)
        self.wait(2.2)
        self._clear()

    def _question(self):
        line1 = Text("Every election creates winners \u2014 and losers.", font_size=34, weight=BOLD)
        line2 = Text(
            "Does how we pick the winner change how the losers are treated?",
            font_size=30,
            t2c={"how we pick": ACCENT},
        ).next_to(line1, DOWN, buff=0.55)
        line3 = Text("We built a simulation to find out.", font_size=24, color=MUTED).next_to(line2, DOWN, buff=0.7)
        VGroup(line1, line2, line3).move_to([0, 0.2, 0])
        self.play(FadeIn(self._kicker("The question")), FadeIn(line1), run_time=1)
        self.wait(1.2)
        self.play(FadeIn(line2), run_time=1)
        self.wait(1.6)
        self.play(FadeIn(line3), run_time=0.8)
        self.wait(1.2)
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
        self.play(
            FadeIn(self._kicker("The setup")),
            FadeIn(bloc_a, lag_ratio=0.01),
            FadeIn(bloc_b, lag_ratio=0.01),
            FadeIn(caption),
            run_time=2,
        )
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
        # Dim the setup scatter so the caption stays readable over the blue cluster.
        self.play(
            bloc_a.animate.set_opacity(0.15),
            bloc_b.animate.set_opacity(0.15),
            stars.animate.set_opacity(0.2),
            FadeIn(caption3),
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in boxes], lag_ratio=0.15),
        )
        self.wait(2.4)
        self._clear()

    def _rule(self):
        header = Text("The winner's spending rule", font_size=38, weight=BOLD).to_edge(UP, buff=0.9)
        formula = Text(
            "allocation  =  \u03bb \u00b7 help my supporters  +  (1\u2212\u03bb) \u00b7 help everyone",
            font_size=32,
            t2c={"\u03bb": ACCENT},
        )
        formula_box = SurroundingRectangle(formula, corner_radius=0.15, buff=0.35, color=ACCENT, stroke_width=2)
        legend = Text(
            "\u03bb = 1  \u2192  serve only your own voters        \u03bb = 0  \u2192  serve the whole town",
            font_size=26,
            color=MUTED,
            t2c={"\u03bb": ACCENT},
        ).next_to(formula_box, DOWN, buff=0.7)
        warning = Text("Voters never see \u03bb on the ballot.", font_size=26, t2c={"\u03bb": ACCENT}).next_to(
            legend, DOWN, buff=0.7
        )
        self.play(FadeIn(self._kicker("The rule")), FadeIn(header))
        self.play(FadeIn(formula, shift=UP * 0.2), run_time=1)
        self.wait(1.2)
        self.play(Create(formula_box), run_time=0.6)
        self.play(FadeIn(legend), run_time=0.8)
        self.wait(1.0)
        self.play(FadeIn(warning))
        self.wait(2.2)
        self._clear()

    def _paradigms(self):
        header = Text("Four ways to pick the winner", font_size=38, weight=BOLD).to_edge(UP, buff=0.7)
        cards = VGroup()
        for name, color in PARADIGM_COLORS.items():
            title = Text(name.replace("_", " "), font_size=28, weight=BOLD, color=color)
            body = Text(PARADIGM_ONE_LINERS[name], font_size=22, color=MUTED, line_spacing=0.9)
            box = RoundedRectangle(
                corner_radius=0.15, width=5.6, height=2.2, color=color, fill_color=color, fill_opacity=0.08
            )
            title.move_to(box.get_top() + DOWN * 0.5)
            body.move_to(box.get_center() + DOWN * 0.35)
            cards.add(VGroup(box, title, body))
        cards.arrange_in_grid(rows=2, cols=2, buff=0.45).next_to(header, DOWN, buff=0.55)
        self.play(FadeIn(self._kicker("The contenders")), FadeIn(header))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.3), run_time=3)
        self.wait(3)
        self._clear()

    def _results(self):
        header = Text(
            "Result: how well do non-supporters do?", font_size=36, weight=BOLD
        ).to_edge(UP, buff=0.7)
        self.play(FadeIn(self._kicker("The result")), FadeIn(header))

        rows = VGroup()
        trackers = []
        values = self.numbers.loser_welfare
        max_value = max(values.values())
        bar_left, max_bar_width, top_y, row_step = -3.6, 5.8, 1.9, 1.05
        for i, (name, color) in enumerate(PARADIGM_COLORS.items()):
            value = values[name]
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
            # Pango-based count-up (DecimalNumber would require LaTeX); the number
            # rides the tip of its bar while both animate.
            tracker = ValueTracker(0.0)
            number = always_redraw(
                lambda b=bar, t=tracker: Text(f"{t.get_value():.3f}", font_size=24).next_to(b, RIGHT, buff=0.3)
            )
            trackers.append(tracker)
            rows.add(VGroup(label, bar, number))

        baseline = Line(
            [bar_left, top_y + 0.45, 0],
            [bar_left, top_y - (len(rows) - 1) * row_step - 0.45, 0],
            color=MUTED,
            stroke_width=2,
        )
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.15), Create(baseline))
        self.add(*[r[2] for r in rows])
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(GrowFromEdge(r[1], LEFT), t.animate.set_value(values[name]))
                    for r, t, name in zip(rows, trackers, PARADIGM_COLORS)
                ],
                lag_ratio=0.2,
            ),
            run_time=2.4,
        )
        for r in rows:
            r[2].clear_updaters()

        # Data-driven headline stat: how much better electoral losers do under the
        # best consensus rule than under party selection.
        pct_gain = (values["latent_match"] - values["party"]) / values["party"]
        chip_text = Text(f"+{pct_gain:.0%} vs party", font_size=20, weight=BOLD, color=BACKGROUND)
        chip_box = RoundedRectangle(
            corner_radius=0.12,
            width=chip_text.width + 0.45,
            height=chip_text.height + 0.32,
            color=ACCENT,
            fill_color=ACCENT,
            fill_opacity=1,
            stroke_width=0,
        )
        chip = VGroup(chip_box, chip_text.move_to(chip_box)).next_to(rows[-1][2], RIGHT, buff=0.35)
        self.play(FadeIn(chip, shift=LEFT * 0.25), run_time=0.8)

        footnote = Text(
            "average benefit to voters who did NOT back the winner (250 simulated elections per rule)",
            font_size=22,
            color=MUTED,
        ).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(footnote))
        self.wait(2.0)
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
            f"Twist: if platforms reveal loyalty, score & latent match elect \u03bb \u2248 {_format_range(correlated)} winners.",
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
        self.play(FadeIn(line1, shift=UP * 0.2), run_time=0.9)
        self.wait(0.7)
        self.play(FadeIn(line2, shift=UP * 0.2), run_time=0.9)
        self.wait(0.7)
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
            "background_color": BACKGROUND,
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
