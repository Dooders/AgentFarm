#!/usr/bin/env python3
"""Manim explainer: agent, environment, actions, and a grid example.

Render a shareable MP4 (from the repo root):

    manim -qm -o intro-explainer.mp4 scripts/intro_explainer.py IntroExplainer

Or:

    python scripts/render_intro_explainer.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    Mobject,
    MoveAlongPath,
    Paragraph,
    RoundedRectangle,
    Scene,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
    smooth,
)

from farm.core.intro_layout import (
    GAP_HEADING,
    MARGIN_BOTTOM,
    banner_top,
    fit_cell,
    frame_bottom,
    safe_width,
    stage_bounds,
    stage_center,
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

# Columns for the three agent kinds, so the row does not drift with label widths.
KIND_X = (-3.60, 0.0, 3.60)


def ink_text(body: str, role: str = "body", color: str = INK) -> Text:
    """One Inter line at a named type role. Avoids LaTeX."""
    spec = type_role(role)
    return Text(
        body,
        font=TYPE_FONT,
        font_size=float(spec["size"]),
        color=color,
        weight=str(spec["weight"]),
        disable_ligatures=True,
    )


def copy_block(copy: str | tuple[str, ...], role: str = "body", color: str = INK) -> Mobject:
    """Centered copy at a named type role.

    Multi-line copy comes from a single Pango layout, so the baselines are evenly
    spaced no matter which lines carry ascenders or descenders. Each line is then
    centered on the block, which centers on the frame axis.
    """
    lines = as_lines(copy)
    if len(lines) == 1:
        return ink_text(lines[0], role, color)
    spec = type_role(role)
    block = Paragraph(
        *lines,
        font=TYPE_FONT,
        font_size=float(spec["size"]),
        color=color,
        weight=str(spec["weight"]),
        line_spacing=float(spec["leading"]),
        alignment="center",
        disable_ligatures=True,
    )
    block.set_x(0)
    return block


@dataclass
class Banner:
    """Section heading with a caption that can be swapped in place."""

    group: VGroup
    heading: Text
    caption: Mobject
    caption_top: float

    @property
    def height(self) -> float:
        return float(self.group.height)

    def swap(self, copy: tuple[str, ...]) -> Mobject:
        """Build a replacement caption on the same top edge and center line."""
        caption = copy_block(copy, "caption", MUTED)
        caption.set_x(0)
        caption.shift(UP * (self.caption_top - caption.get_top()[1]))
        return caption


def section_banner(title: str, copy: tuple[str, ...]) -> Banner:
    """Centered heading and caption, parked in the top margin."""
    heading = ink_text(title, "heading")
    heading.set_x(0)
    heading.shift(UP * (banner_top() - heading.get_top()[1]))
    caption = copy_block(copy, "caption", MUTED)
    caption.set_x(0)
    caption_top = heading.get_bottom()[1] - GAP_HEADING
    caption.shift(UP * (caption_top - caption.get_top()[1]))
    return Banner(VGroup(heading, caption), heading, caption, caption_top)


@dataclass
class GridStage:
    """Square teaching grid sized to the room left between banner and footer."""

    cell: float
    center_y: float

    @classmethod
    def between(cls, top: float, bottom: float, size: int = GRID_SIZE) -> GridStage:
        return cls(fit_cell(top, bottom, size), stage_center(top, bottom))

    @property
    def span(self) -> float:
        return self.cell * GRID_SIZE

    def point(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Manim point for a cell, y increasing upward."""
        offset = (GRID_SIZE - 1) / 2
        return np.array(
            [
                (grid_x - offset) * self.cell,
                (grid_y - offset) * self.cell + self.center_y,
                0.0,
            ]
        )

    def board(self) -> VGroup:
        cells = VGroup()
        for grid_y in range(GRID_SIZE):
            for grid_x in range(GRID_SIZE):
                square = Square(self.cell)
                square.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.1)
                square.move_to(self.point(grid_x, grid_y))
                cells.add(square)
        return cells

    def patch(self, grid_x: int, grid_y: int) -> RoundedRectangle:
        side = self.cell * 0.68
        patch = RoundedRectangle(width=side, height=side, corner_radius=0.08)
        patch.set_fill(FOOD, 0.88).set_stroke(width=0)
        patch.move_to(self.point(grid_x, grid_y))
        return patch

    def dot_radius(self) -> float:
        return self.cell * 0.30


def card_box(content: Mobject, pad_x: float, pad_y: float, min_width: float = 0.0) -> VGroup:
    """Rounded card sized around centered content."""
    box = RoundedRectangle(
        width=max(content.width + pad_x, min_width),
        height=content.height + pad_y,
        corner_radius=0.16,
    )
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.25)
    box.set_x(0)
    content.move_to(box.get_center())
    return VGroup(box, content)


def pill(label: str) -> VGroup:
    """Action chip — Medium Inter, room to breathe."""
    text = ink_text(label, "chip")
    box = RoundedRectangle(width=text.width + 0.62, height=0.58, corner_radius=0.16)
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
    box = RoundedRectangle(width=text.width + 0.50, height=0.50, corner_radius=0.14)
    box.set_fill(CARD, 1).set_stroke(GRID_EDGE, 1.15)
    text.move_to(box.get_center())
    return VGroup(box, text)


def agent_dot(color: str, radius: float = 0.17) -> Circle:
    """One agent."""
    dot = Circle(radius=radius)
    dot.set_fill(color, 1).set_stroke(INK, 1.3)
    return dot


def kind_columns() -> VGroup:
    """Dot, name, and hint for each agent kind, on fixed columns."""
    columns = VGroup()
    for item, x_pos in zip(AGENT_KINDS, KIND_X):
        dot = agent_dot(item["color"], 0.24)
        label = ink_text(item["label"], "label")
        hint = copy_block(item["hint"], "meta", MUTED)
        column = VGroup(dot, label, hint).arrange(DOWN, buff=0.22)
        column.set_x(x_pos)
        columns.add(column)
    return columns


def loop_row() -> tuple[VGroup, VGroup]:
    """Look → Decide → Act cards with arrows between them."""
    cards = VGroup(*[step_card(step) for step in LOOP_STEPS])
    cards.arrange(RIGHT, buff=1.05)
    cards.set_x(0)
    arrows = VGroup()
    for left, right in zip(cards, cards[1:]):
        mark = ink_text("→", "label", MUTED)
        mark.move_to((left.get_right() + right.get_left()) / 2)
        arrows.add(mark)
    return cards, arrows


def action_rows() -> VGroup:
    """Every action, in two centered rows that fit the safe width."""
    top = VGroup(*[pill(name) for name in ACTIONS[:4]]).arrange(RIGHT, buff=0.30)
    bottom = VGroup(*[pill(name) for name in ACTIONS[4:]]).arrange(RIGHT, buff=0.30)
    rows = VGroup(top, bottom).arrange(DOWN, buff=0.26)
    if rows.width > safe_width():
        rows.scale_to_fit_width(safe_width())
    rows.set_x(0)
    return rows


def legend_row() -> VGroup:
    """Colour key for the grid example."""
    row = VGroup(
        _legend_item(COOPERATIVE, "Cooperative"),
        _legend_item(SELF_INTERESTED, "Self-interested"),
        _legend_item(BALANCED, "Balanced"),
        _legend_item(FOOD, "Food", square=True),
    )
    row.arrange(RIGHT, buff=0.58)
    row.set_x(0)
    return row


def _legend_item(color: str, label: str, square: bool = False) -> VGroup:
    if square:
        mark = Square(0.18).set_fill(color, 0.9).set_stroke(width=0)
    else:
        mark = Circle(radius=0.09).set_fill(color, 1).set_stroke(INK, 1.0)
    text = ink_text(label, "meta", MUTED)
    text.next_to(mark, RIGHT, buff=0.14)
    return VGroup(mark, text)


def seat_on_bottom(block: Mobject) -> Mobject:
    """Place a block on the bottom margin, centered."""
    block.set_x(0)
    block.shift(UP * (frame_bottom() + MARGIN_BOTTOM - block.get_bottom()[1]))
    return block


def seat_in_stage(block: Mobject, top: float, bottom: float) -> Mobject:
    """Center a block in a stage span."""
    block.set_x(0)
    block.shift(UP * (stage_center(top, bottom) - block.get_center()[1]))
    return block


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

    def _open_section(self, title: str, copy: tuple[str, ...]) -> Banner:
        banner = section_banner(title, copy)
        self.play(FadeIn(banner.group, shift=UP * 0.04), run_time=0.55, rate_func=smooth)
        self.wait(HOLD_LINE)
        return banner

    def _hook(self) -> None:
        title = copy_block(HOOK_TITLE, "display")
        line = copy_block(HOOK_LINE, "lead", MUTED)
        question = copy_block(HOOK_QUESTION, "lead")
        card = card_box(question, pad_x=1.30, pad_y=0.92, min_width=8.0)

        line.next_to(title, DOWN, buff=0.26)
        card.next_to(line, DOWN, buff=0.70)
        lockup = VGroup(title, line, card)
        lockup.move_to(ORIGIN)
        for part in (title, line, card):
            part.set_x(0)

        self.play(FadeIn(title, shift=UP * 0.06), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(line, shift=DOWN * 0.08), run_time=0.5, rate_func=smooth)
        self.wait(hold_for(HOOK_TITLE, HOOK_LINE))
        self.play(FadeIn(card, shift=UP * 0.06), run_time=0.7, rate_func=smooth)
        self.wait(hold_for(HOOK_QUESTION))
        self._fade_all()

    def _what_is_an_agent(self) -> None:
        banner = self._open_section(SECTION_AGENT, CAPTION_AGENT)
        top, bottom = stage_bounds(banner.height)

        dot = agent_dot("#6b7280", 0.30)
        seat_in_stage(dot, top, bottom)
        self.play(GrowFromCenter(dot), run_time=0.45, rate_func=smooth)

        chips = [trait_chip(label) for label in AGENT_TRAITS]
        chips[0].next_to(dot, UP, buff=0.46)
        chips[1].next_to(dot, RIGHT, buff=0.52)
        chips[2].next_to(dot, DOWN, buff=0.46)
        chips[3].next_to(dot, LEFT, buff=0.52)
        traits = VGroup(*chips)
        self.play(
            LaggedStart(*[FadeIn(mob, shift=DOWN * 0.08) for mob in traits], lag_ratio=0.16),
            run_time=1.15,
        )
        self.wait(hold_for(*AGENT_TRAITS, minimum=HOLD_READ))

        kinds_caption = banner.swap(CAPTION_KINDS)
        self.play(FadeOut(traits), FadeOut(banner.caption), run_time=0.35, rate_func=smooth)
        self.play(FadeIn(kinds_caption), run_time=0.35)
        self.wait(HOLD_LINE)

        columns = kind_columns()
        seat_in_stage(columns, top, bottom)
        self.play(
            dot.animate.move_to(columns[0][0].get_center()).set_fill(COOPERATIVE, 1),
            run_time=0.65,
            rate_func=smooth,
        )
        self.remove(dot)
        self.add(columns[0][0])
        self.play(
            FadeIn(columns[0][1]),
            FadeIn(columns[0][2]),
            FadeIn(columns[1], shift=UP * 0.08),
            FadeIn(columns[2], shift=UP * 0.08),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(hold_for(CAPTION_KINDS, minimum=HOLD_READ))
        self._fade_all()

    def _what_is_the_environment(self) -> None:
        banner = self._open_section(SECTION_ENV, CAPTION_ENV)
        note = seat_on_bottom(copy_block(NOTE_ENV, "caption", MUTED))
        top, bottom = stage_bounds(banner.height, note.height)
        stage = GridStage.between(top, bottom)

        board = stage.board()
        self.play(Create(board, lag_ratio=0.012), run_time=1.5, rate_func=smooth)

        food = VGroup(*[stage.patch(x, y) for x, y in FOOD_START])
        self.play(
            LaggedStart(*[FadeIn(patch, scale=0.75) for patch in food], lag_ratio=0.14),
            FadeIn(note),
            run_time=0.95,
        )
        self.wait(hold_for(NOTE_ENV))
        self._fade_all()

    def _what_do_agents_do(self) -> None:
        banner = self._open_section(SECTION_DO, CAPTION_DO)
        top, bottom = stage_bounds(banner.height)

        cards, arrows = loop_row()
        world = copy_block(WORLD_CHANGES, "lead")
        rows = action_rows()
        world.next_to(cards, DOWN, buff=0.52)
        rows.next_to(world, DOWN, buff=0.56)
        column = VGroup(cards, world, rows)
        seat_in_stage(column, top, bottom)
        for part in (cards, world, rows):
            part.set_x(0)
        for left, right, mark in zip(cards, cards[1:], arrows):
            mark.move_to((left.get_right() + right.get_left()) / 2)

        self.play(
            LaggedStart(*[FadeIn(mob, shift=RIGHT * 0.12) for mob in cards], lag_ratio=0.18),
            run_time=1.15,
        )
        self.play(FadeIn(arrows), run_time=0.35)
        for card in cards:
            box = card[0]
            self.play(box.animate.set_stroke(INK, 2.0), run_time=0.32, rate_func=smooth)
            self.play(box.animate.set_stroke(GRID_EDGE, 1.4), run_time=0.26, rate_func=smooth)

        self.play(FadeIn(world, shift=DOWN * 0.08), run_time=0.4, rate_func=smooth)
        self.wait(hold_for(WORLD_CHANGES))
        self.play(
            LaggedStart(
                *[FadeIn(mob, shift=DOWN * 0.08) for row in rows for mob in row],
                lag_ratio=0.07,
            ),
            run_time=0.95,
        )
        self.wait(HOLD_READ)
        self._fade_all()

    def _grid_example(self) -> None:
        banner = self._open_section(SECTION_GRID, CAPTION_GRID)
        legend = seat_on_bottom(legend_row())
        top, bottom = stage_bounds(banner.height, legend.height)
        stage = GridStage.between(top, bottom)

        board = stage.board()
        self.play(FadeIn(board), run_time=0.45, rate_func=smooth)

        food = {cell: stage.patch(*cell) for cell in FOOD_START}
        self.play(
            LaggedStart(*[FadeIn(patch, scale=0.8) for patch in food.values()], lag_ratio=0.08),
            run_time=0.7,
        )

        dots = {}
        arrivals = []
        for agent_id, spec in AGENTS.items():
            dot = agent_dot(kind_color(spec["kind"]), stage.dot_radius())
            dot.move_to(stage.point(*spec["start"]))
            dots[agent_id] = dot
            arrivals.append(GrowFromCenter(dot))
        self.play(LaggedStart(*arrivals, lag_ratio=0.07), run_time=0.75)
        self.play(FadeIn(legend), run_time=0.35, rate_func=smooth)

        beat = banner.swap(CAPTION_WALK)
        self.play(FadeOut(banner.caption), FadeIn(beat), run_time=0.4, rate_func=smooth)
        self.wait(hold_for(CAPTION_WALK))

        for turn in TURNS:
            walks = []
            for agent_id, cell in turn["moves"].items():
                dest = stage.point(*cell)
                start = dots[agent_id].get_center()
                if abs(start[0] - dest[0]) + abs(start[1] - dest[1]) < 1e-6:
                    continue
                path = Line(start, dest)
                path.set_opacity(0)
                walks.append(MoveAlongPath(dots[agent_id], path))
            if walks:
                self.play(*walks, run_time=WALK_TIME, rate_func=smooth)
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
        closer = banner.swap(CAPTION_CLUSTER)
        halo = SurroundingRectangle(cluster, color=INK, buff=0.20, stroke_width=2.0)
        self.play(FadeOut(beat), FadeIn(closer), run_time=0.4, rate_func=smooth)
        self.play(FadeIn(halo), run_time=0.45, rate_func=smooth)
        self.wait(hold_for(CAPTION_CLUSTER))
        self._fade_all()

    def _close(self) -> None:
        title = copy_block(CLOSE_TITLE, "heading")
        question = copy_block(CLOSE_QUESTION, "caption", MUTED)
        question.next_to(title, DOWN, buff=0.52)
        lockup = VGroup(title, question)
        lockup.move_to(ORIGIN)
        title.set_x(0)
        question.set_x(0)
        self.play(FadeIn(title, shift=UP * 0.05), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(question, shift=DOWN * 0.04), run_time=0.6, rate_func=smooth)
        self.wait(hold_for(CLOSE_TITLE, CLOSE_QUESTION))
        self._fade_all(run_time=0.55)
        self.wait(0.15)


SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("agent", SECTION_AGENT, CAPTION_AGENT),
    ("environment", SECTION_ENV, CAPTION_ENV),
    ("actions", SECTION_DO, CAPTION_DO),
    ("grid", SECTION_GRID, CAPTION_GRID),
)


def section_stage(name: str) -> tuple[float, float]:
    """Top and bottom Y of a section's stage, as the scene computes it."""
    for key, title, caption in SECTIONS:
        if key != name:
            continue
        banner = section_banner(title, caption)
        if key == "environment":
            footer = float(copy_block(NOTE_ENV, "caption", MUTED).height)
        elif key == "grid":
            footer = float(legend_row().height)
        else:
            footer = 0.0
        return stage_bounds(banner.height, footer)
    raise KeyError(name)


__all__ = [
    "SECTIONS",
    "Banner",
    "GridStage",
    "IntroExplainer",
    "action_rows",
    "copy_block",
    "ink_text",
    "kind_columns",
    "legend_row",
    "loop_row",
    "seat_in_stage",
    "seat_on_bottom",
    "section_banner",
    "section_stage",
]
