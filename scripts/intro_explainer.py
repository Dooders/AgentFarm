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
    Line,
    MoveAlongPath,
    RoundedRectangle,
    Scene,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
    smooth,
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
    WORLD_CHANGES,
    kind_color,
)

CELL = 0.58
SAFE_WIDTH = 12.4


def ink_text(body: str, size: int = 36, color: str = INK) -> Text:
    """Plain-language title or caption. Avoids LaTeX."""
    return Text(body, font_size=size, color=color)


def heading(body: str) -> Text:
    """Section heading pinned to the top of the frame."""
    text = ink_text(body, 38)
    text.to_edge(UP, buff=0.42)
    return text


def caption(body: str) -> Text:
    """Muted line sitting on a fixed baseline under the heading."""
    text = ink_text(body, 24, MUTED)
    text.to_edge(UP, buff=1.08)
    return text


def pill(label: str) -> VGroup:
    """Action chip with room to breathe."""
    text = ink_text(label, 22)
    box = RoundedRectangle(
        width=text.width + 0.50,
        height=0.56,
        corner_radius=0.18,
    )
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.4)
    text.move_to(box.get_center())
    return VGroup(box, text)


def step_card(label: str) -> VGroup:
    """Equal-sized card for Look / Decide / Act."""
    text = ink_text(label, 28)
    box = RoundedRectangle(width=2.9, height=1.12, corner_radius=0.18)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.5)
    text.move_to(box.get_center())
    return VGroup(box, text)


def trait_chip(label: str) -> VGroup:
    """Small card used around the single-agent diagram."""
    text = ink_text(label, 22, MUTED)
    box = RoundedRectangle(width=text.width + 0.40, height=0.50, corner_radius=0.14)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.2)
    text.move_to(box.get_center())
    return VGroup(box, text)


def cell_point(grid_x: int, grid_y: int, origin=ORIGIN):
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
            square.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.1)
            square.move_to(cell_point(grid_x, grid_y, origin))
            cells.add(square)
    return cells


def food_patch(grid_x: int, grid_y: int, origin=ORIGIN) -> RoundedRectangle:
    """Green rounded square sitting inside a cell."""
    patch = RoundedRectangle(width=CELL * 0.68, height=CELL * 0.68, corner_radius=0.08)
    patch.set_fill(FOOD, 0.88).set_stroke(width=0)
    patch.move_to(cell_point(grid_x, grid_y, origin))
    return patch


def agent_dot(color: str, radius: float = 0.17) -> Circle:
    """One agent."""
    dot = Circle(radius=radius)
    dot.set_fill(color, 1).set_stroke(INK, 1.3)
    return dot


def arrow_mark() -> Text:
    return ink_text("→", 34, MUTED)


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

    def _fade_all(self, run_time: float = 0.40) -> None:
        lingering = [mob for mob in list(self.mobjects) if mob is not None]
        if lingering:
            self.play(*[FadeOut(mob) for mob in lingering], run_time=run_time, rate_func=smooth)

    def _open_section(self, title: str, subtitle: str) -> tuple[Text, Text]:
        head = heading(title)
        sub = caption(subtitle)
        self.play(
            FadeIn(head, shift=UP * 0.06),
            FadeIn(sub, shift=UP * 0.04),
            run_time=0.55,
            rate_func=smooth,
        )
        return head, sub

    def _hook(self) -> None:
        title = ink_text("A limited world.", 50)
        line = ink_text("Many agents share the food.", 28, MUTED)
        line.next_to(title, DOWN, buff=0.32)
        question = ink_text("What mix of helpfulness", 32)
        question2 = ink_text("and self-interest actually works?", 32)
        question2.next_to(question, DOWN, buff=0.16)
        q_group = VGroup(question, question2)
        box = RoundedRectangle(
            width=q_group.width + 0.95,
            height=q_group.height + 0.78,
            corner_radius=0.20,
        )
        box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.4)
        q_group.move_to(box.get_center())
        card_group = VGroup(box, q_group)
        card_group.next_to(line, DOWN, buff=0.72)

        self.play(Write(title), run_time=1.05)
        self.play(FadeIn(line, shift=DOWN * 0.12), run_time=0.65, rate_func=smooth)
        self.wait(0.55)
        self.play(FadeIn(card_group, shift=UP * 0.08), run_time=0.75, rate_func=smooth)
        self.wait(2.3)
        self._fade_all()

    def _what_is_an_agent(self) -> None:
        _title, sub = self._open_section("What is an agent?", "One actor. It lives, looks, and chooses.")

        dot = agent_dot("#6b7280", 0.30)
        dot.shift(DOWN * 0.10)
        self.play(GrowFromCenter(dot), run_time=0.45, rate_func=smooth)

        chips = [trait_chip(label) for label in AGENT_TRAITS]
        chips[0].next_to(dot, UP, buff=0.42)
        chips[1].next_to(dot, RIGHT, buff=0.48)
        chips[2].next_to(dot, DOWN, buff=0.42)
        chips[3].next_to(dot, LEFT, buff=0.48)
        trait_mobs = VGroup(*chips)
        self.play(
            LaggedStart(*[FadeIn(mob, shift=DOWN * 0.08) for mob in trait_mobs], lag_ratio=0.16),
            run_time=1.15,
        )
        self.wait(1.7)

        self.play(FadeOut(trait_mobs), FadeOut(sub), run_time=0.35, rate_func=smooth)
        kinds_title = caption("Three tendencies — not personalities.")
        self.play(FadeIn(kinds_title), run_time=0.35)

        kind_group = VGroup()
        for item in AGENT_KINDS:
            circle = agent_dot(item["color"], 0.24)
            label = ink_text(item["label"], 26)
            hint = ink_text(item["hint"], 20, MUTED)
            label.next_to(circle, DOWN, buff=0.24)
            hint.next_to(label, DOWN, buff=0.10)
            kind_group.add(VGroup(circle, label, hint))
        kind_group.arrange(RIGHT, buff=1.5)
        kind_group.next_to(kinds_title, DOWN, buff=0.90)

        self.play(
            dot.animate.move_to(kind_group[0][0].get_center()).set_fill(COOPERATIVE, 1),
            run_time=0.65,
            rate_func=smooth,
        )
        self.remove(dot)
        self.add(kind_group[0][0])
        self.play(
            FadeIn(kind_group[0][1]),
            FadeIn(kind_group[0][2]),
            FadeIn(kind_group[1], shift=UP * 0.08),
            FadeIn(kind_group[2], shift=UP * 0.08),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(2.1)
        self._fade_all()

    def _what_is_the_environment(self) -> None:
        self._open_section("What is the environment?", "The world they share — a grid of places.")

        origin = DOWN * 0.22
        grid = make_grid(origin)
        self.play(Create(grid, lag_ratio=0.012), run_time=1.5, rate_func=smooth)

        food = VGroup(*[food_patch(x, y, origin) for x, y in FOOD_START])
        note = ink_text("Some squares hold food. The rules do not change.", 24, MUTED)
        note.to_edge(DOWN, buff=0.42)
        self.play(
            LaggedStart(*[FadeIn(patch, scale=0.75) for patch in food], lag_ratio=0.14),
            FadeIn(note),
            run_time=0.95,
        )
        self.wait(2.3)
        self._fade_all()

    def _what_do_agents_do(self) -> None:
        self._open_section("What do agents do?", "Every turn is the same three steps.")

        cards = VGroup(*[step_card(step) for step in LOOP_STEPS])
        cards.arrange(RIGHT, buff=1.05)
        cards.shift(UP * 0.42)
        arrows = VGroup()
        for left, right in zip(cards, cards[1:]):
            mark = arrow_mark()
            mark.move_to((left.get_right() + right.get_left()) / 2)
            arrows.add(mark)

        self.play(
            LaggedStart(*[FadeIn(mob, shift=RIGHT * 0.12) for mob in cards], lag_ratio=0.18),
            run_time=1.15,
        )
        self.play(FadeIn(arrows), run_time=0.35)
        for card in cards:
            box = card[0]
            self.play(box.animate.set_stroke(INK, 2.0), run_time=0.22, rate_func=smooth)
            self.play(box.animate.set_stroke(GRID_EDGE, 1.5), run_time=0.18, rate_func=smooth)

        world = ink_text(WORLD_CHANGES, 26, INK)
        world.next_to(cards, DOWN, buff=0.42)
        self.play(FadeIn(world, shift=DOWN * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(0.85)

        top = VGroup(*[pill(name) for name in ACTIONS[:4]])
        bottom = VGroup(*[pill(name) for name in ACTIONS[4:]])
        top.arrange(RIGHT, buff=0.28)
        bottom.arrange(RIGHT, buff=0.28)
        action_pills = VGroup(top, bottom).arrange(DOWN, buff=0.24)
        action_pills.next_to(world, DOWN, buff=0.55)
        if action_pills.width > SAFE_WIDTH:
            action_pills.scale_to_fit_width(SAFE_WIDTH)
        self.play(
            LaggedStart(*[FadeIn(mob, shift=DOWN * 0.08) for mob in (*top, *bottom)], lag_ratio=0.07),
            run_time=0.95,
        )
        self.wait(2.1)
        self._fade_all()

    def _grid_example(self) -> None:
        _title, sub = self._open_section("An example on a grid", "A handful of agents. A few patches of food.")

        origin = DOWN * 0.08
        grid = make_grid(origin)
        self.play(FadeIn(grid), run_time=0.45, rate_func=smooth)

        food = {cell: food_patch(*cell, origin) for cell in FOOD_START}
        self.play(
            LaggedStart(*[FadeIn(patch, scale=0.8) for patch in food.values()], lag_ratio=0.08),
            run_time=0.7,
        )

        dots = {}
        intro_anims = []
        for agent_id, spec in AGENTS.items():
            dot = agent_dot(kind_color(spec["kind"]))
            dot.move_to(cell_point(*spec["start"], origin))
            dots[agent_id] = dot
            intro_anims.append(GrowFromCenter(dot))
        self.play(LaggedStart(*intro_anims, lag_ratio=0.07), run_time=0.75)

        legend = VGroup(
            _legend_item(COOPERATIVE, "Cooperative"),
            _legend_item(SELF_INTERESTED, "Self-interested"),
            _legend_item(BALANCED, "Balanced"),
            _legend_item(FOOD, "Food", square=True),
        )
        legend.arrange(RIGHT, buff=0.55)
        legend.to_edge(DOWN, buff=0.30)
        self.play(FadeIn(legend), run_time=0.35, rate_func=smooth)

        beat = caption("They walk. They eat. Neighbors form.")
        self.play(FadeOut(sub), FadeIn(beat), run_time=0.4, rate_func=smooth)

        for turn in TURNS:
            animations = []
            for agent_id, cell in turn["moves"].items():
                dest = cell_point(*cell, origin)
                start = dots[agent_id].get_center()
                if abs(start[0] - dest[0]) + abs(start[1] - dest[1]) < 1e-6:
                    continue
                path = Line(start, dest)
                path.set_opacity(0)
                animations.append(MoveAlongPath(dots[agent_id], path))
            if animations:
                self.play(*animations, run_time=0.72, rate_func=smooth)
            eaten = []
            for cell in turn["eat"]:
                patch = food.get(cell)
                if patch is not None:
                    eaten.append(patch.animate.scale(0.2).set_opacity(0))
                    food.pop(cell, None)
            if eaten:
                self.play(*eaten, run_time=0.4, rate_func=smooth)
            else:
                self.wait(0.06)

        cluster = VGroup(dots["blue_a"], dots["blue_b"], dots["red_a"], dots["red_b"])
        closer = caption("Nobody told the blue agents to cluster.")
        halo = SurroundingRectangle(cluster, color=INK, buff=0.20, stroke_width=2.0)
        self.play(FadeOut(beat), FadeIn(closer), run_time=0.4, rate_func=smooth)
        self.play(FadeIn(halo), run_time=0.45, rate_func=smooth)
        self.wait(2.1)
        self._fade_all()

    def _close(self) -> None:
        title = ink_text("The research measures what emerges.", 36)
        q1 = ink_text("Does a mix survive better", 28, MUTED)
        q2 = ink_text("than only helpers — or only competitors?", 28, MUTED)
        q1.next_to(title, DOWN, buff=0.42)
        q2.next_to(q1, DOWN, buff=0.16)
        group = VGroup(title, q1, q2)
        group.move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.06), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(q1), FadeIn(q2), run_time=0.65, rate_func=smooth)
        self.wait(2.8)
        self._fade_all(run_time=0.55)
        self.wait(0.15)


def _legend_item(color: str, label: str, square: bool = False) -> VGroup:
    if square:
        mark = Square(0.20).set_fill(color, 0.9).set_stroke(width=0)
    else:
        mark = Circle(radius=0.10).set_fill(color, 1).set_stroke(INK, 1.0)
    text = ink_text(label, 20, MUTED)
    text.next_to(mark, RIGHT, buff=0.14)
    return VGroup(mark, text)
