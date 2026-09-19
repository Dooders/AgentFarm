"""Unit tests for the intro explainer storyboard."""

from __future__ import annotations

import pytest

from farm.core.intro_storyboard import (
    ACTIONS,
    AGENT_KINDS,
    AGENT_TRAITS,
    AGENTS,
    FOOD_START,
    GRID_SIZE,
    LOOP_STEPS,
    TURNS,
    WORLD_CHANGES,
    in_bounds,
    kind_color,
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


def test_kind_color_and_bounds():
    assert kind_color("cooperative").startswith("#")
    assert in_bounds((0, 0))
    assert in_bounds((GRID_SIZE - 1, GRID_SIZE - 1))
    assert not in_bounds((-1, 0))
    assert not in_bounds((GRID_SIZE, 0))
    with pytest.raises(KeyError):
        kind_color("unknown")


def test_example_uses_every_agent_and_some_food():
    moved = set()
    eaten = set()
    for turn in TURNS:
        moved.update(turn["moves"])
        eaten.update(turn["eat"])
    assert moved == set(AGENTS)
    assert eaten
    assert eaten.issubset(set(FOOD_START))
