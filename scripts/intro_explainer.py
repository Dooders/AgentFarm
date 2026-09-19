#!/usr/bin/env python3
"""Manim explainer: agent, environment, actions, and a grid example.

Render a shareable MP4 (from the repo root):

    manim -qm -o intro-explainer.mp4 scripts/intro_explainer.py IntroExplainer

Or:

    python scripts/render_intro_explainer.py
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    Write,
)

from farm.core.intro_storyboard import (
    ACTIONS,
    AGENT_KINDS,
    AGENT_TRAITS,
    AGENTS,
    BALANCED,
    BG,
    CARD,
    COOPERATIVE,
    FOOD,
    FOOD_START,
    GRID_EDGE,
    GRID_SIZE,
    INK,
    LOOP_STEPS,
    MUTED,
    SELF_INTERESTED,
    TURNS,
    kind_color,
)

CELL = 0.52


def ink_text(body: str, size: int = 36, color: str = INK) -> Text:
    """Plain-language title or caption. Avoids LaTeX."""
    return Text(body, font_size=size, color=color)


def heading(body: str) -> Text:
    """Section heading pinned to the top of the frame."""
    text = ink_text(body, 40)
    text.to_edge(UP, buff=0.38)
    return text


def caption(body: str) -> Text:
    """Muted line under the heading."""
    text = ink_text(body, 26, MUTED)
    text.next_to(ORIGIN, UP, buff=3.05)
    text.to_edge(UP, buff=1.05)
    return text


def pill(label: str, accent: str = GRID_EDGE) -> VGroup:
    """Small rounded label."""
    text = ink_text(label, 22)
    box = RoundedRectangle(
        width=max(text.width + 0.42, 1.4),
        height=0.52,
        corner_radius=0.16,
    )
    box.set_fill(CARD, 1).set_stroke(accent, 1.6)
    text.move_to(box.get_center())
    return VGroup(box, text)


def card(label: str, width: float = 2.55, height: float = 1.05) -> VGroup:
    """Larger step card for the look-decide-act loop."""
    text = ink_text(label, 26)
    box = RoundedRectangle(width=width, height=height, corner_radius=0.16)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.5)
    text.move_to(box.get_center())
    return VGroup(box, text)


def cell_point(grid_x: int, grid_y: int, origin=ORIGIN) -> object:
    """Manim point for a grid cell, y increasing upward."""
    shift_x = (grid_x - (GRID_SIZE - 1) / 2) * CELL
    shift_y = (grid_y - (GRID_SIZE - 1) / 2) * CELL
    return origin + [shift_x, shift_y, 0]


def make_grid(origin=ORIGIN) -> VGroup:
    """Empty teaching grid."""
    cells = VGroup()
    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            square = Square(CELL)
            square.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.2)
            square.move_to(cell_point(grid_x, grid_y, origin))
            cells.add(square)
    return cells


def food_patch(grid_x: int, grid_y: int, origin=ORIGIN, scale: float = 1.0) -> RoundedRectangle:
    """Green rounded square sitting inside a cell."""
    patch = RoundedRectangle(width=CELL * 0.72, height=CELL * 0.72, corner_radius=0.08)
    patch.set_fill(FOOD, 0.85).set_stroke(width=0)
    patch.scale(scale)
    patch.move_to(cell_point(grid_x, grid_y, origin))
    return patch


def agent_dot(color: str, radius: float = 0.16) -> Circle:
    """One agent."""
    dot = Circle(radius=radius)
    dot.set_fill(color, 1).set_stroke(INK, 1.4)
    return dot


class IntroExplainer(Scene):
    """Shareable walkthrough of the research intro."""

    def construct(self) -> None:
        self.camera.background_color = BG
        self._hook()
        self._what_is_an_agent()
        self._what_is_the_environment()
        self._what_do_agents_do()
        self._grid_example()
        self._close()

    def _fade_all(self, run_time: float = 0.45) -> None:
        lingering = [mob for mob in list(self.mobjects) if mob is not None]
        if lingering:
            self.play(*[FadeOut(mob) for mob in lingering], run_time=run_time)

    def _hook(self) -> None:
        title = ink_text("A limited world.", 52)
        line = ink_text("Many agents share the food.", 32, MUTED)
        line.next_to(title, DOWN, buff=0.35)
        question = ink_text("What mix of helpfulness", 34)
        question2 = ink_text("and self-interest actually works?", 34)
        question2.next_to(question, DOWN, buff=0.18)
        q_group = VGroup(question, question2)
        box = RoundedRectangle(width=q_group.width + 0.8, height=q_group.height + 0.7, corner_radius=0.18)
        box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.5)
        q_group.move_to(box.get_center())
        card_group = VGroup(box, q_group)
        card_group.next_to(line, DOWN, buff=0.7)

        self.play(Write(title), run_time=1.1)
        self.play(FadeIn(line, shift=DOWN * 0.15), run_time=0.7)
        self.wait(0.6)
        self.play(FadeIn(card_group, shift=UP * 0.1), run_time=0.8)
        self.wait(2.2)
        self._fade_all()

    def _what_is_an_agent(self) -> None:
        title = heading("What is an agent?")
        sub = caption("One actor. It lives, looks, and chooses.")
        self.play(FadeIn(title), FadeIn(sub), run_time=0.6)

        dot = agent_dot("#6b7280", 0.28)
        dot.shift(DOWN * 0.15)
        self.play(GrowFromCenter(dot), run_time=0.5)

        labels = [
            (AGENT_TRAITS[0], UP * 1.45),
            (AGENT_TRAITS[1], RIGHT * 3.1),
            (AGENT_TRAITS[2], DOWN * 1.45),
            (AGENT_TRAITS[3], LEFT * 3.25),
        ]
        trait_mobs = VGroup()
        for body, direction in labels:
            label = ink_text(body, 26, MUTED)
            label.next_to(dot, direction, buff=0.35)
            trait_mobs.add(label)
        self.play(LaggedStart(*[FadeIn(mob, shift=DOWN * 0.1) for mob in trait_mobs], lag_ratio=0.18))
        self.wait(1.6)

        self.play(FadeOut(trait_mobs), FadeOut(sub), run_time=0.4)
        kinds_title = caption("Three tendencies — not personalities.")
        self.play(FadeIn(kinds_title), run_time=0.4)

        kind_group = VGroup()
        for item in AGENT_KINDS:
            circle = agent_dot(item["color"], 0.24)
            label = ink_text(item["label"], 26)
            hint = ink_text(item["hint"], 20, MUTED)
            label.next_to(circle, DOWN, buff=0.22)
            hint.next_to(label, DOWN, buff=0.12)
            kind_group.add(VGroup(circle, label, hint))
        kind_group.arrange(RIGHT, buff=1.35)
        kind_group.next_to(kinds_title, DOWN, buff=0.85)

        self.play(dot.animate.move_to(kind_group[0][0].get_center()).set_fill(COOPERATIVE, 1), run_time=0.6)
        self.remove(dot)
        self.add(kind_group[0][0])
        self.play(
            FadeIn(kind_group[0][1]),
            FadeIn(kind_group[0][2]),
            FadeIn(kind_group[1], shift=UP * 0.1),
            FadeIn(kind_group[2], shift=UP * 0.1),
            run_time=0.8,
        )
        self.wait(2.0)
        self._fade_all()

    def _what_is_the_environment(self) -> None:
        title = heading("What is the environment?")
        sub = caption("The world they share — a grid of places.")
        self.play(FadeIn(title), FadeIn(sub), run_time=0.55)

        origin = DOWN * 0.25
        grid = make_grid(origin)
        self.play(Create(grid, lag_ratio=0.01), run_time=1.6)

        food = VGroup(*[food_patch(x, y, origin) for x, y in FOOD_START])
        note = ink_text("Some squares hold food. The rules do not change.", 24, MUTED)
        note.to_edge(DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(patch, scale=0.7) for patch in food], lag_ratio=0.12), FadeIn(note))
        self.wait(2.2)
        self._fade_all()

    def _what_do_agents_do(self) -> None:
        title = heading("What do agents do?")
        sub = caption("Every turn is the same three steps.")
        self.play(FadeIn(title), FadeIn(sub), run_time=0.55)

        cards = VGroup(*[card(step, width=2.7 if step != "The world changes" else 3.35) for step in LOOP_STEPS])
        cards.arrange(RIGHT, buff=0.28)
        cards.shift(UP * 0.35)
        arrows = VGroup()
        for left, right in zip(cards, cards[1:]):
            arrow = ink_text("→", 36, MUTED)
            arrow.move_to((left.get_right() + right.get_left()) / 2)
            arrows.add(arrow)

        self.play(LaggedStart(*[FadeIn(mob, shift=RIGHT * 0.15) for mob in cards], lag_ratio=0.2), run_time=1.3)
        self.play(FadeIn(arrows), run_time=0.4)
        self.wait(1.3)

        action_pills = VGroup(*[pill(name) for name in ACTIONS])
        action_pills.arrange_in_grid(rows=2, cols=4, buff=0.22)
        # 7 pills: keep the last centered on the second row
        action_pills.arrange(RIGHT, buff=0.16)
        action_pills.next_to(cards, DOWN, buff=0.85)
        if action_pills.width > 12.6:
            action_pills.scale_to_fit_width(12.6)
        self.play(LaggedStart(*[FadeIn(mob, shift=DOWN * 0.1) for mob in action_pills], lag_ratio=0.08))
        self.wait(2.0)
        self._fade_all()

    def _grid_example(self) -> None:
        title = heading("An example on a grid")
        sub = caption("A handful of agents. A few patches of food.")
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        origin = DOWN * 0.15
        grid = make_grid(origin)
        self.play(FadeIn(grid), run_time=0.5)

        food = {cell: food_patch(*cell, origin) for cell in FOOD_START}
        self.play(LaggedStart(*[FadeIn(patch, scale=0.8) for patch in food.values()], lag_ratio=0.08), run_time=0.7)

        dots = {}
        intro_anims = []
        for agent_id, spec in AGENTS.items():
            dot = agent_dot(kind_color(spec["kind"]))
            dot.move_to(cell_point(*spec["start"], origin))
            dots[agent_id] = dot
            intro_anims.append(GrowFromCenter(dot))
        self.play(LaggedStart(*intro_anims, lag_ratio=0.08), run_time=0.8)

        legend = VGroup(
            _legend_item(COOPERATIVE, "Cooperative"),
            _legend_item(SELF_INTERESTED, "Self-interested"),
            _legend_item(BALANCED, "Balanced"),
            _legend_item(FOOD, "Food", square=True),
        )
        legend.arrange(RIGHT, buff=0.45)
        legend.to_edge(DOWN, buff=0.32)
        self.play(FadeIn(legend), run_time=0.4)

        beat = ink_text("They walk. They eat. Neighbors form.", 24, MUTED)
        beat.next_to(title, DOWN, buff=0.18)
        self.play(FadeOut(sub), FadeIn(beat), run_time=0.4)

        for turn in TURNS:
            animations = []
            for agent_id, cell in turn["moves"].items():
                animations.append(dots[agent_id].animate.move_to(cell_point(*cell, origin)))
            if animations:
                self.play(*animations, run_time=0.55)
            eaten = []
            for cell in turn["eat"]:
                patch = food.get(cell)
                if patch is not None:
                    eaten.append(FadeOut(patch, scale=0.4))
                    food.pop(cell, None)
            if eaten:
                self.play(*eaten, run_time=0.4)
            else:
                self.wait(0.12)

        closer = ink_text("Nobody told the blue agents to cluster.", 26, MUTED)
        closer.next_to(title, DOWN, buff=0.18)
        self.play(FadeOut(beat), FadeIn(closer), run_time=0.5)
        self.wait(2.0)
        self._fade_all()

    def _close(self) -> None:
        title = ink_text("The research measures what emerges.", 38)
        q1 = ink_text("Does a mix survive better", 32, MUTED)
        q2 = ink_text("than only helpers — or only competitors?", 32, MUTED)
        q1.next_to(title, DOWN, buff=0.45)
        q2.next_to(q1, DOWN, buff=0.18)
        group = VGroup(title, q1, q2)
        group.move_to(ORIGIN)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(q1), FadeIn(q2), run_time=0.7)
        self.wait(2.6)
        self._fade_all(run_time=0.6)
        self.wait(0.2)


def _legend_item(color: str, label: str, square: bool = False) -> VGroup:
    if square:
        mark = Square(0.22).set_fill(color, 0.9).set_stroke(width=0)
    else:
        mark = Circle(radius=0.11).set_fill(color, 1).set_stroke(INK, 1.0)
    text = ink_text(label, 20, MUTED)
    text.next_to(mark, RIGHT, buff=0.12)
    return VGroup(mark, text)
