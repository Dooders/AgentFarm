"""Layout tests for the intro explainer.

The geometry tests run everywhere. The typesetting tests need Manim, which is an
optional extra (``pip install '.[intro]'``), so they skip when it is absent.
"""

from __future__ import annotations

import pytest

from farm.core.intro_layout import (
    FRAME_HEIGHT,
    GAP_BANNER,
    GAP_FOOTER,
    MARGIN_BOTTOM,
    MARGIN_TOP,
    STAGE_BIAS,
    banner_top,
    fit_cell,
    fits_frame,
    frame_bottom,
    frame_top,
    safe_width,
    seat_y,
    stage_bounds,
    stage_center,
)
from farm.core.intro_storyboard import GRID_SIZE

pytestmark = pytest.mark.unit


def test_frame_and_margins():
    assert frame_top() == FRAME_HEIGHT / 2
    assert frame_bottom() == -FRAME_HEIGHT / 2
    assert banner_top() == frame_top() - MARGIN_TOP
    assert 0 < safe_width() < 14.3


def test_stage_bounds_leave_room_for_banner_and_footer():
    top, bottom = stage_bounds(1.5)
    assert top == pytest.approx(banner_top() - 1.5 - GAP_BANNER)
    assert bottom == pytest.approx(frame_bottom() + MARGIN_BOTTOM + GAP_FOOTER)

    top_with_footer, bottom_with_footer = stage_bounds(1.5, 0.7)
    assert top_with_footer == top
    assert bottom_with_footer == pytest.approx(bottom + 0.7)
    assert bottom_with_footer > bottom


def test_stage_bounds_rejects_impossible_layouts():
    with pytest.raises(ValueError):
        stage_bounds(6.0, 2.0)


def test_fit_cell_keeps_the_grid_inside_the_stage():
    top, bottom = stage_bounds(1.5, 0.7)
    cell = fit_cell(top, bottom, GRID_SIZE)
    assert cell * GRID_SIZE <= top - bottom + 1e-9
    assert cell <= 0.56
    # A tall stage is capped, a short one shrinks the cell.
    assert fit_cell(3.0, -3.0, GRID_SIZE) == 0.56
    assert fit_cell(1.0, -1.0, GRID_SIZE) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        fit_cell(1.0, 2.0, GRID_SIZE)
    with pytest.raises(ValueError):
        fit_cell(1.0, -1.0, 0)


def test_seat_y_lifts_short_blocks_above_the_middle():
    top, bottom = 2.0, -2.0
    middle = stage_center(top, bottom)
    assert STAGE_BIAS < 0.5
    seated = seat_y(top, bottom, 1.0)
    assert seated > middle
    above = top - (seated + 0.5)
    below = (seated - 0.5) - bottom
    assert above < below
    assert above + below == pytest.approx(top - bottom - 1.0)
    # A plain centre is still available, and oversized blocks just centre.
    assert seat_y(top, bottom, 1.0, bias=0.5) == pytest.approx(middle)
    assert seat_y(top, bottom, 9.0) == pytest.approx(middle)


def test_stage_center_and_frame_fit():
    top, bottom = stage_bounds(1.4, 0.6)
    assert stage_center(top, bottom) == pytest.approx((top + bottom) / 2)
    assert fits_frame(top, bottom)
    assert not fits_frame(frame_top(), bottom)
    assert not fits_frame(top, frame_bottom())


manim = pytest.importorskip("manim", reason="Manim is an optional extra")
explainer = pytest.importorskip("scripts.intro_explainer", reason="Manim is an optional extra")

CENTER_TOLERANCE = 0.02


def _lines(block) -> list:
    """The individual lines of a copy block (a single line has no sub-lines)."""
    if isinstance(block, manim.Paragraph):
        return list(block)
    return [block]


@pytest.mark.parametrize(
    "copy",
    [
        ("One actor in the world.", "Nobody tells it what to do."),
        ("A single line."),
    ],
)
def test_copy_blocks_are_centered(copy):
    block = explainer.copy_block(copy, "caption")
    assert abs(block.get_center()[0]) < CENTER_TOLERANCE
    for line in _lines(block):
        assert abs(line.get_center()[0]) < CENTER_TOLERANCE


def test_multi_line_copy_has_even_baselines():
    """A Pango block keeps one pitch even when only some lines have descenders."""
    block = explainer.copy_block(
        ("Hnnn", "Hggg", "Hnnn"),
        "caption",
    )
    bottoms = [float(line.get_bottom()[1]) for line in block]
    no_descender_pitch = bottoms[0] - bottoms[2]
    single = explainer.copy_block(("Hnnn", "Hnnn"), "caption")
    expected = float(single[0].get_bottom()[1] - single[1].get_bottom()[1])
    assert no_descender_pitch == pytest.approx(2 * expected, abs=1e-6)


def test_section_banner_sits_in_the_top_margin_and_centers():
    banner = explainer.section_banner("What is an agent?", ("One line.", "Two lines."), 2)
    assert banner.eyebrow.get_top()[1] == pytest.approx(banner_top())
    assert abs(banner.eyebrow.get_center()[0]) < CENTER_TOLERANCE
    assert banner.heading.get_top()[1] < banner.eyebrow.get_bottom()[1]
    assert abs(banner.heading.get_center()[0]) < CENTER_TOLERANCE
    assert abs(banner.caption.get_center()[0]) < CENTER_TOLERANCE
    assert banner.caption.get_top()[1] < banner.heading.get_bottom()[1]
    swapped = banner.swap(("Different words here.", "On two lines."))
    assert swapped.get_top()[1] == pytest.approx(banner.caption.get_top()[1], abs=1e-6)
    assert abs(swapped.get_center()[0]) < CENTER_TOLERANCE


@pytest.mark.parametrize("name", ["agent", "environment", "actions", "grid"])
def test_section_stages_stay_inside_the_frame(name):
    top, bottom = explainer.section_stage(name)
    assert fits_frame(top, bottom)
    assert top > bottom


def test_grid_stage_fits_between_banner_and_footer():
    for name in ("environment", "grid"):
        top, bottom = explainer.section_stage(name)
        stage = explainer.GridStage.between(top, bottom)
        assert stage.span <= top - bottom + 1e-9
        board = stage.board()
        assert board.get_top()[1] <= top + 1e-6
        assert board.get_bottom()[1] >= bottom - 1e-6
        assert abs(board.get_center()[0]) < CENTER_TOLERANCE


def test_footer_blocks_clear_the_grid():
    _, bottom = explainer.section_stage("environment")
    note = explainer.seat_on_bottom(
        explainer.copy_block(("Some of the squares hold food.", "One more line."), "caption")
    )
    assert note.get_top()[1] < bottom
    assert note.get_bottom()[1] >= frame_bottom() + MARGIN_BOTTOM - 1e-6
    legend = explainer.seat_on_bottom(explainer.legend_row())
    grid_top, grid_bottom = explainer.section_stage("grid")
    assert legend.get_top()[1] < grid_bottom
    assert grid_top < banner_top()


def test_stage_blocks_fit_their_sections():
    for name, block in (
        ("actions", explainer.action_rows()),
        ("agent", explainer.kind_columns()),
    ):
        top, bottom = explainer.section_stage(name)
        explainer.seat_in_stage(block, top, bottom)
        assert block.get_top()[1] <= top + 1e-6
        assert block.get_bottom()[1] >= bottom - 1e-6
        assert block.width <= safe_width()


def test_the_whole_actions_column_fits_its_stage():
    """Cards, the turn sentence, and both chip rows are the tightest stack."""
    cards, _ = explainer.loop_row()
    world = explainer.copy_block(("After everyone acts, the map updates", "and the next turn begins."), "lead")
    rows = explainer.action_rows()
    world.next_to(cards, manim.DOWN, buff=0.52)
    rows.next_to(world, manim.DOWN, buff=0.56)
    column = manim.VGroup(cards, world, rows)
    top, bottom = explainer.section_stage("actions")
    assert column.height <= top - bottom
    explainer.seat_in_stage(column, top, bottom)
    assert column.get_top()[1] <= top + 1e-6
    assert column.get_bottom()[1] >= bottom - 1e-6
    assert world.get_top()[1] < cards.get_bottom()[1]
    assert rows.get_top()[1] < world.get_bottom()[1]


def test_board_is_one_panel_with_evenly_spaced_hairlines():
    top, bottom = explainer.section_stage("grid")
    stage = explainer.GridStage.between(top, bottom)
    panel, lines = stage.board()
    assert panel.width == pytest.approx(stage.span)
    assert panel.height == pytest.approx(stage.span)
    # Seven interior lines each way, none doubled up on a shared cell edge.
    assert len(lines) == 2 * (7)
    verticals = sorted(float(line.get_center()[0]) for line in lines if line.height > line.width)
    gaps = [b - a for a, b in zip(verticals, verticals[1:])]
    assert gaps == pytest.approx([stage.cell] * len(gaps), abs=1e-6)


def test_action_loop_row_is_centered_with_arrows_between_cards():
    cards, arrows = explainer.loop_row()
    assert abs(cards.get_center()[0]) < CENTER_TOLERANCE
    assert len(arrows) == len(cards) - 1
    for left, right, mark in zip(cards, cards[1:], arrows):
        assert left.get_right()[0] < mark.get_center()[0] < right.get_left()[0]
    for card in cards:
        box, text = card
        assert text.width > 0.4
        assert box.width > text.width


def test_trait_chips_are_equal_width_so_the_ring_is_symmetric():
    chips = explainer.trait_chips()
    widths = [float(chip.width) for chip in chips]
    assert widths == pytest.approx([widths[0]] * len(widths), abs=1e-6)
    for chip in chips:
        box, text = chip
        assert abs(box.get_center()[0] - text.get_center()[0]) < CENTER_TOLERANCE


def test_kind_columns_use_fixed_equal_columns():
    columns = explainer.kind_columns()
    centers = [float(column.get_center()[0]) for column in columns]
    assert centers == pytest.approx(list(explainer.KIND_X), abs=CENTER_TOLERANCE)
    gaps = [centers[1] - centers[0], centers[2] - centers[1]]
    assert gaps[0] == pytest.approx(gaps[1], abs=1e-6)
    for column in columns:
        for part in column:
            assert abs(part.get_center()[0] - column.get_center()[0]) < CENTER_TOLERANCE
