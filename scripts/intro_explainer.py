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
    smooth,
)

from farm.core.intro_storyboard import (
    ACTIONS,
    AGENT_KINDS,
    AGENT_TRAITS,
    AGENTS,
    BALANCED,
    BG,
    CAPTION_AGENT,
    CAPTION_CLUSTER,
    CAPTION_DO,
    CAPTION_ENV,
    CAPTION_GRID,
    CAPTION_KINDS,
    CAPTION_WALK,
    CARD,
    CLOSE_QUESTION,
    CLOSE_TITLE,
    COOPERATIVE,
    FOOD,
    FOOD_START,
    GRID_EDGE,
    GRID_SIZE,
    HOLD_LINE,
    HOLD_READ,
    HOOK_LINE,
    HOOK_QUESTION,
    HOOK_TITLE,
    INK,
    LOOP_STEPS,
    MUTED,
    NOTE_ENV,
    SECTION_AGENT,
    SECTION_DO,
    SECTION_ENV,
    SECTION_GRID,
    SELF_INTERESTED,
    TURNS,
    TYPE_FONT,
    WALK_TIME,
    WORLD_CHANGES,
    as_lines,
    hold_for,
    kind_color,
    type_role,
)

CELL = 0.58
SAFE_WIDTH = 12.4


def ink_text(body: str, role: str = "body", color: str = INK) -> Text:
    """Single Inter line. Avoids LaTeX."""
    spec = type_role(role)
    return Text(
        body,
        font=TYPE_FONT,
        font_size=float(spec["size"]),
        color=color,
        weight=str(spec["weight"]),
    )


def stack_lines(
    copy: str | tuple[str, ...],
    role: str,
    color: str = INK,
    *,
    buff: float = 0.10,
) -> VGroup:
    """Left-aligned stack of Inter lines, then centered as a group by the caller."""
    parts = [ink_text(line, role, color) for line in as_lines(copy)]
    if len(parts) == 1:
        return VGroup(parts[0])
    return VGroup(*parts).arrange(DOWN, buff=buff, aligned_edge=LEFT)


def section_banner(title: str, lines: tuple[str, ...]) -> tuple[VGroup, VGroup]:
    """Heading + short rule + caption, left-aligned, parked at the top center."""
    head = ink_text(title, "heading")
    cap = stack_lines(lines, "caption", MUTED, buff=0.08)
    rule = Line(ORIGIN, RIGHT * 0.70)
    rule.set_stroke("#9ca3af", 1.6)
    rule.next_to(head, DOWN, buff=0.14)
    rule.align_to(head, LEFT)
    cap.next_to(rule, DOWN, buff=0.20)
    cap.align_to(head, LEFT)
    banner = VGroup(head, rule, cap)
    banner.to_edge(UP, buff=0.36)
    banner.set_x(0)
    return banner, cap


def place_caption(copy: tuple[str, ...], banner: VGroup) -> VGroup:
    """Replacement caption that keeps the section lockup's left edge."""
    cap = stack_lines(copy, "caption", MUTED, buff=0.08)
    cap.next_to(banner[1], DOWN, buff=0.20)
    cap.align_to(banner[0], LEFT)
    return cap


def pill(label: str) -> VGroup:
    """Action chip — Medium Inter, room to breathe."""
    text = ink_text(label, "chip")
    box = RoundedRectangle(
        width=text.width + 0.58,
        height=0.56,
        corner_radius=0.16,
    )
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.25)
    text.move_to(box.get_center())
    return VGroup(box, text)


def step_card(label: str) -> VGroup:
    """Equal-sized card for Look / Decide / Act."""
    text = ink_text(label, "step")
    box = RoundedRectangle(width=2.85, height=1.18, corner_radius=0.16)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.4)
    text.move_to(box.get_center())
    return VGroup(box, text)


def trait_chip(label: str) -> VGroup:
    """Small card used around the single-agent diagram."""
    text = ink_text(label, "chip", MUTED)
    box = RoundedRectangle(width=text.width + 0.46, height=0.48, corner_radius=0.14)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.15)
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
    return ink_text("→", "label", MUTED)


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

    def _open_section(self, title: str, lines: tuple[str, ...]) -> tuple[VGroup, VGroup]:
        banner, cap = section_banner(title, lines)
        self.play(FadeIn(banner, shift=UP * 0.04), run_time=0.55, rate_func=smooth)
        self.wait(HOLD_LINE)
        return banner, cap

    def _hook(self) -> None:
        title = ink_text(HOOK_TITLE, "display")
        line = ink_text(HOOK_LINE, "lead", MUTED)
        lede = VGroup(title, line).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        rule = Line(ORIGIN, RIGHT * 0.70)
        rule.set_stroke("#9ca3af", 1.6)
        rule.next_to(lede, DOWN, buff=0.32)
        rule.align_to(lede, LEFT)
        question = stack_lines(HOOK_QUESTION, "lead", buff=0.12)
        box = RoundedRectangle(
            width=question.width + 1.05,
            height=question.height + 0.82,
            corner_radius=0.18,
        )
        box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.25)
        question.move_to(box.get_center())
        card_group = VGroup(box, question)
        card_group.next_to(rule, DOWN, buff=0.40)
        card_group.align_to(lede, LEFT)
        lockup = VGroup(lede, rule, card_group)
        lockup.move_to(ORIGIN)

        self.play(FadeIn(title, shift=UP * 0.06), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(line, shift=DOWN * 0.08), run_time=0.5, rate_func=smooth)
        self.wait(hold_for(HOOK_TITLE, HOOK_LINE))
        self.play(
            FadeIn(rule),
            FadeIn(card_group, shift=UP * 0.06),
            run_time=0.7,
            rate_func=smooth,
        )
        self.wait(hold_for(HOOK_QUESTION))
        self._fade_all()

    def _what_is_an_agent(self) -> None:
        banner, sub = self._open_section(SECTION_AGENT, CAPTION_AGENT)

        dot = agent_dot("#6b7280", 0.30)
        dot.shift(DOWN * 0.10)
        self.play(GrowFromCenter(dot), run_time=0.45, rate_func=smooth)

        chips = [trait_chip(label) for label in AGENT_TRAITS]
        chips[0].next_to(dot, UP, buff=0.44)
        chips[1].next_to(dot, RIGHT, buff=0.50)
        chips[2].next_to(dot, DOWN, buff=0.44)
        chips[3].next_to(dot, LEFT, buff=0.50)
        trait_mobs = VGroup(*chips)
        self.play(
            LaggedStart(*[FadeIn(mob, shift=DOWN * 0.08) for mob in trait_mobs], lag_ratio=0.16),
            run_time=1.15,
        )
        self.wait(hold_for(*AGENT_TRAITS, minimum=HOLD_READ))

        self.play(FadeOut(trait_mobs), FadeOut(sub), run_time=0.35, rate_func=smooth)
        kinds_title = place_caption(CAPTION_KINDS, banner)
        self.play(FadeIn(kinds_title), run_time=0.35)
        self.wait(HOLD_LINE)

        kind_group = VGroup()
        for item in AGENT_KINDS:
            circle = agent_dot(item["color"], 0.24)
            label = ink_text(item["label"], "label")
            hint = ink_text(item["hint"], "meta", MUTED)
            col = VGroup(circle, label, hint).arrange(DOWN, buff=0.18)
            kind_group.add(col)
        kind_group.arrange(RIGHT, buff=1.15)
        kind_group.next_to(banner, DOWN, buff=0.70)

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
        self.wait(hold_for(CAPTION_KINDS, minimum=HOLD_READ))
        self._fade_all()

    def _what_is_the_environment(self) -> None:
        self._open_section(SECTION_ENV, CAPTION_ENV)

        origin = DOWN * 0.22
        grid = make_grid(origin)
        self.play(Create(grid, lag_ratio=0.012), run_time=1.5, rate_func=smooth)

        food = VGroup(*[food_patch(x, y, origin) for x, y in FOOD_START])
        note = stack_lines(NOTE_ENV, "caption", MUTED, buff=0.08)
        note.to_edge(DOWN, buff=0.36)
        self.play(
            LaggedStart(*[FadeIn(patch, scale=0.75) for patch in food], lag_ratio=0.14),
            FadeIn(note),
            run_time=0.95,
        )
        self.wait(hold_for(NOTE_ENV))
        self._fade_all()

    def _what_do_agents_do(self) -> None:
        self._open_section(SECTION_DO, CAPTION_DO)

        cards = VGroup(*[step_card(step) for step in LOOP_STEPS])
        cards.arrange(RIGHT, buff=1.05)
        cards.shift(UP * 0.38)
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
            self.play(box.animate.set_stroke(INK, 2.0), run_time=0.32, rate_func=smooth)
            self.play(box.animate.set_stroke(GRID_EDGE, 1.4), run_time=0.26, rate_func=smooth)

        world = stack_lines(WORLD_CHANGES, "lead", buff=0.10)
        world.next_to(cards, DOWN, buff=0.42)
        self.play(FadeIn(world, shift=DOWN * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(hold_for(WORLD_CHANGES))

        top = VGroup(*[pill(name) for name in ACTIONS[:4]])
        bottom = VGroup(*[pill(name) for name in ACTIONS[4:]])
        top.arrange(RIGHT, buff=0.30)
        bottom.arrange(RIGHT, buff=0.30)
        action_pills = VGroup(top, bottom).arrange(DOWN, buff=0.26)
        action_pills.next_to(world, DOWN, buff=0.52)
        if action_pills.width > SAFE_WIDTH:
            action_pills.scale_to_fit_width(SAFE_WIDTH)
        self.play(
            LaggedStart(*[FadeIn(mob, shift=DOWN * 0.08) for mob in (*top, *bottom)], lag_ratio=0.07),
            run_time=0.95,
        )
        self.wait(HOLD_READ)
        self._fade_all()

    def _grid_example(self) -> None:
        banner, sub = self._open_section(SECTION_GRID, CAPTION_GRID)

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
        legend.arrange(RIGHT, buff=0.58)
        legend.to_edge(DOWN, buff=0.30)
        self.play(FadeIn(legend), run_time=0.35, rate_func=smooth)

        beat = place_caption(CAPTION_WALK, banner)
        self.play(FadeOut(sub), FadeIn(beat), run_time=0.4, rate_func=smooth)
        self.wait(hold_for(CAPTION_WALK))

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
                self.play(*animations, run_time=WALK_TIME, rate_func=smooth)
            eaten = []
            for cell in turn["eat"]:
                patch = food.get(cell)
                if patch is not None:
                    eaten.append(patch.animate.scale(0.2).set_opacity(0))
                    food.pop(cell, None)
            if eaten:
                self.play(*eaten, run_time=0.5, rate_func=smooth)
            else:
                self.wait(0.12)

        cluster = VGroup(dots["blue_a"], dots["blue_b"], dots["red_a"], dots["red_b"])
        closer = place_caption(CAPTION_CLUSTER, banner)
        halo = SurroundingRectangle(cluster, color=INK, buff=0.20, stroke_width=2.0)
        self.play(FadeOut(beat), FadeIn(closer), run_time=0.4, rate_func=smooth)
        self.play(FadeIn(halo), run_time=0.45, rate_func=smooth)
        self.wait(hold_for(CAPTION_CLUSTER))
        self._fade_all()

    def _close(self) -> None:
        title = stack_lines(CLOSE_TITLE, "heading", buff=0.10)
        question = stack_lines(CLOSE_QUESTION, "caption", MUTED, buff=0.10)
        group = VGroup(title, question).arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        group.move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.05), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(question, shift=DOWN * 0.04), run_time=0.6, rate_func=smooth)
        self.wait(hold_for(CLOSE_TITLE, CLOSE_QUESTION))
        self._fade_all(run_time=0.55)
        self.wait(0.15)


def _legend_item(color: str, label: str, square: bool = False) -> VGroup:
    if square:
        mark = Square(0.18).set_fill(color, 0.9).set_stroke(width=0)
    else:
        mark = Circle(radius=0.09).set_fill(color, 1).set_stroke(INK, 1.0)
    text = ink_text(label, "meta", MUTED)
    text.next_to(mark, RIGHT, buff=0.14)
    return VGroup(mark, text)
