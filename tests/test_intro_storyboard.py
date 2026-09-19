"""Unit tests for the intro explainer storyboard."""

from __future__ import annotations

import pytest

from farm.core.intro_storyboard import (
    ACTIONS,
    AGENT_KINDS,
    AGENT_TRAITS,
    AGENTS,
    CAPTION_AGENT,
    CLOSE_TITLE,
    FOOD_START,
    GRID_SIZE,
    HOLD_LONG,
    HOLD_READ,
    HOOK_QUESTION,
    HOOK_TITLE,
    LOOP_STEPS,
    TURNS,
    TYPE_FONT,
    TYPE_SCALE,
    WORLD_CHANGES,
    in_bounds,
    kind_color,
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
    assert WORLD_CHANGES.startswith("Then")
    assert "Eat" in ACTIONS
    assert "Walk" in ACTIONS
    assert "share" in HOOK_QUESTION.lower() or "helping" in HOOK_QUESTION.lower()
    assert "individual" in CAPTION_AGENT.lower()
    assert CLOSE_TITLE.endswith(".")
    assert HOLD_READ > 2.5
    assert HOLD_LONG > HOLD_READ


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
    with pytest.raises(KeyError):
        type_role("unknown")


def test_example_uses_every_agent_and_some_food():
    moved = set()
    eaten = set()
    for turn in TURNS:
        moved.update(turn["moves"])
        eaten.update(turn["eat"])
    assert moved == set(AGENTS)
    assert eaten
    assert eaten.issubset(set(FOOD_START))
