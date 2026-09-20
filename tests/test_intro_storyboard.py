"""Unit tests for the intro explainer storyboard."""

from __future__ import annotations

import pytest

from farm.core.intro_storyboard import (
    ACTIONS,
    AGENT_KINDS,
    AGENT_TRAITS,
    AGENTS,
    CAPTION_AGENT,
    CAPTION_CLUSTER,
    CAPTION_DO,
    CAPTION_ENV,
    CAPTION_GRID,
    CAPTION_WALK,
    CLOSE_QUESTION,
    CLOSE_TITLE,
    CREDIT_LINK,
    CREDIT_NAME,
    FOOD_START,
    GRID_SIZE,
    HOLD_LINE,
    HOLD_LONG,
    HOLD_READ,
    HOOK_LINE,
    HOOK_QUESTION,
    HOOK_TITLE,
    LOOP_STEPS,
    NOTE_ENV,
    SECTION_COUNT,
    TURNS,
    TYPE_FONT,
    TYPE_MIN_SIZE,
    TYPE_SCALE,
    WORLD_CHANGES,
    hold_for,
    in_bounds,
    kind_color,
    line_span,
    step_label,
    type_role,
    validate_storyboard,
)

pytestmark = pytest.mark.unit


def test_storyboard_is_internally_consistent():
    validate_storyboard()


def test_copy_covers_the_four_intro_beats():
    assert len(AGENT_TRAITS) == 4
    assert {item["key"] for item in AGENT_KINDS} == {"cooperative", "self_interested", "balanced"}
    assert tuple(LOOP_STEPS) == ("Look", "Decide", "Act")
    assert "map updates" in " ".join(WORLD_CHANGES).lower()
    assert "Eat" in ACTIONS
    assert "Walk" in ACTIONS
    assert "helpfulness" in " ".join(HOOK_QUESTION).lower()
    assert "self-interest" in " ".join(HOOK_QUESTION).lower()
    assert "actor" in " ".join(CAPTION_AGENT).lower()
    assert "share" in HOOK_LINE.lower() or "share" in HOOK_TITLE.lower()
    assert CLOSE_TITLE[-1].endswith(".")
    assert HOLD_READ >= 3.5
    assert HOLD_LONG > HOLD_READ
    assert hold_for(HOOK_QUESTION) >= 4.0
    assert hold_for("one two three") >= HOLD_LINE


def test_step_labels_cover_the_four_questions():
    assert SECTION_COUNT == 4
    assert step_label(1) == "1 of 4"
    assert step_label(SECTION_COUNT) == "4 of 4"
    for bad in (0, SECTION_COUNT + 1):
        with pytest.raises(ValueError):
            step_label(bad)


def test_credit_points_at_the_docs_site():
    assert CREDIT_NAME == "AgentFarm"
    assert CREDIT_LINK == "dooders.github.io/AgentFarm"
    assert " " not in CREDIT_LINK


def test_stacked_copy_avoids_dangling_punctuation():
    """A line should not end on an em dash or a bare conjunction-free fragment."""
    for lines in (CAPTION_GRID, CAPTION_ENV, NOTE_ENV, CAPTION_WALK, WORLD_CHANGES):
        for line in lines[:-1]:
            assert not line.rstrip().endswith("—"), line


def test_kind_color_and_bounds():
    assert kind_color("cooperative").startswith("#")
    assert in_bounds((0, 0))
    assert in_bounds((GRID_SIZE - 1, GRID_SIZE - 1))
    assert not in_bounds((-1, 0))
    assert not in_bounds((GRID_SIZE, 0))
    with pytest.raises(KeyError):
        kind_color("unknown")


def test_type_system_matches_docs_inter():
    assert TYPE_FONT == "Inter"
    assert set(TYPE_SCALE) == {
        "display",
        "heading",
        "lead",
        "body",
        "caption",
        "label",
        "step",
        "chip",
        "meta",
    }
    display = type_role("display")
    heading = type_role("heading")
    body = type_role("body")
    caption = type_role("caption")
    assert display["weight"] == "SEMIBOLD"
    assert heading["weight"] == "SEMIBOLD"
    assert type_role("label")["weight"] == "MEDIUM"
    assert caption["size"] < body["size"] < heading["size"] < display["size"]
    # Reading copy gets looser leading than display type, as in the docs CSS.
    assert caption["leading"] > heading["leading"] > display["leading"]
    for role, spec in TYPE_SCALE.items():
        assert spec["size"] >= TYPE_MIN_SIZE, role
        assert spec["leading"] > 0, role
    with pytest.raises(KeyError):
        type_role("unknown")


def test_stacked_copy_is_evenly_wrapped():
    pairs = (
        HOOK_QUESTION,
        CAPTION_ENV,
        NOTE_ENV,
        CAPTION_DO,
        CAPTION_WALK,
        CAPTION_CLUSTER,
        WORLD_CHANGES,
        CLOSE_TITLE,
        CLOSE_QUESTION,
    )
    for lines in pairs:
        assert line_span(lines) <= 12, lines
    for item in AGENT_KINDS:
        assert isinstance(item["hint"], tuple)
        assert line_span(item["hint"]) <= 4


def test_example_uses_every_agent_and_some_food():
    moved = set()
    eaten = set()
    for turn in TURNS:
        moved.update(turn["moves"])
        eaten.update(turn["eat"])
    assert moved == set(AGENTS)
    assert eaten
    assert eaten.issubset(set(FOOD_START))
