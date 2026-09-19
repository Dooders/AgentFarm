"""Copy and motion plan for the shareable intro explainer.

Kept free of Manim so unit tests can check the story without a render.
"""

from __future__ import annotations

from typing import Any

INK = "#111827"
MUTED = "#374151"
BG = "#f7f7f8"
CARD = "#ffffff"
GRID_EDGE = "#d4d4d8"
FOOD = "#16a34a"

# Inter, same family as the docs site. Sizes are Manim font_size points.
# Word spacing is applied in the scene (Manim space glyphs are unreliable).
TYPE_FONT = "Inter"
TYPE_SCALE: dict[str, dict[str, int | str]] = {
    "display": {"size": 46, "weight": "SEMIBOLD"},
    "heading": {"size": 32, "weight": "SEMIBOLD"},
    "lead": {"size": 24, "weight": "MEDIUM"},
    "body": {"size": 25, "weight": "NORMAL"},
    "caption": {"size": 20, "weight": "NORMAL"},
    "label": {"size": 22, "weight": "MEDIUM"},
    "step": {"size": 26, "weight": "SEMIBOLD"},
    "chip": {"size": 20, "weight": "MEDIUM"},
    "meta": {"size": 16, "weight": "NORMAL"},
}


def type_role(name: str) -> dict[str, int | str]:
    """Return a copy of a named type-scale role."""
    try:
        return dict(TYPE_SCALE[name])
    except KeyError as exc:
        raise KeyError(name) from exc

COOPERATIVE = "#2563eb"
SELF_INTERESTED = "#dc2626"
BALANCED = "#d97706"

GRID_SIZE = 8

AGENT_TRAITS = (
    "Lives somewhere",
    "Carries food",
    "Looks around",
    "Chooses for itself",
)

AGENT_KINDS = (
    {"key": "cooperative", "label": "Cooperative", "color": COOPERATIVE, "hint": "Tends to share"},
    {"key": "self_interested", "label": "Self-interested", "color": SELF_INTERESTED, "hint": "Tends to keep food"},
    {"key": "balanced", "label": "Balanced", "color": BALANCED, "hint": "A middle path"},
)

HOOK_TITLE = "Food is limited."
HOOK_LINE = "Everyone in this world has to share it."
HOOK_QUESTION = "What mix of helping others\nand looking out for yourself actually works?"

SECTION_AGENT = "What is an agent?"
CAPTION_AGENT = "One individual — it lives here, looks around, and chooses."
CAPTION_KINDS = "They lean different ways. These are tendencies, not personalities."

SECTION_ENV = "What is the environment?"
CAPTION_ENV = "The map they all share. Here it is a grid of squares."
NOTE_ENV = "Some squares hold food. The rules are the same for everyone."

SECTION_DO = "What do agents do?"
CAPTION_DO = "Each turn, every living agent does the same three things."

SECTION_GRID = "An example on a grid"
CAPTION_GRID = "A few agents, a few patches of food. Nothing here is scripted."
CAPTION_WALK = "They walk toward food. Some eat. Neighbors appear."
CAPTION_CLUSTER = "Nobody told the blue agents to gather. It just happened."

CLOSE_TITLE = "The research measures what shows up."
CLOSE_QUESTION = "Does a mix last longer than a world\nof only helpers — or only competitors?"

# On-screen reading: ~150 wpm plus a rest so the last words are not cut off.
SECONDS_PER_WORD = 0.40
HOLD_REST = 1.8
HOLD_LINE = 2.2
HOLD_READ = 4.4
HOLD_LONG = 5.8
WALK_TIME = 1.05


def word_count(*texts: str) -> int:
    """Count whitespace-separated tokens across one or more copy strings."""
    total = 0
    for text in texts:
        total += sum(1 for token in text.replace("\n", " ").split() if token)
    return total


def hold_for(*texts: str, minimum: float = HOLD_LINE) -> float:
    """Seconds to leave copy up for a first-time viewer."""
    return max(minimum, word_count(*texts) * SECONDS_PER_WORD + HOLD_REST)

ACTIONS = (
    "Walk",
    "Eat",
    "Share",
    "Fight",
    "Defend",
    "Have offspring",
    "Wait",
)

LOOP_STEPS = (
    "Look",
    "Decide",
    "Act",
)

WORLD_CHANGES = "Then the map updates, and the next turn starts."

# Starting cells (x, y) with origin at bottom-left, y up.
AGENTS: dict[str, dict[str, Any]] = {
    "blue_a": {"kind": "cooperative", "start": (1, 6)},
    "blue_b": {"kind": "cooperative", "start": (2, 7)},
    "red_a": {"kind": "self_interested", "start": (7, 2)},
    "red_b": {"kind": "self_interested", "start": (6, 7)},
    "orange_a": {"kind": "balanced", "start": (3, 2)},
}

FOOD_START = (
    (5, 5),
    (5, 6),
    (1, 2),
    (6, 1),
)

# Each turn: optional new cell per agent, optional food cells that shrink.
# Every move is one orthogonal step so walks read as slides, not teleports.
TURNS: tuple[dict[str, Any], ...] = (
    {"moves": {"blue_a": (2, 6), "blue_b": (3, 7), "red_a": (7, 3), "orange_a": (3, 3)}, "eat": ()},
    {"moves": {"blue_a": (3, 6), "blue_b": (4, 7), "red_a": (7, 4), "red_b": (6, 6)}, "eat": ()},
    {"moves": {"blue_a": (4, 6), "blue_b": (5, 7), "red_a": (6, 4), "orange_a": (4, 3)}, "eat": ()},
    {"moves": {"blue_a": (4, 5), "blue_b": (5, 6), "red_a": (6, 5)}, "eat": ()},
    {"moves": {"blue_a": (5, 5), "orange_a": (4, 4)}, "eat": ((5, 5),)},
    {"moves": {"red_a": (6, 5), "red_b": (6, 6)}, "eat": ((5, 6),)},
)


def kind_color(kind: str) -> str:
    """Return the hex color for a named agent kind."""
    for item in AGENT_KINDS:
        if item["key"] == kind:
            return item["color"]
    raise KeyError(kind)


def in_bounds(cell: tuple[int, int], size: int = GRID_SIZE) -> bool:
    """True if ``cell`` is on the teaching grid."""
    x, y = cell
    return 0 <= x < size and 0 <= y < size


def validate_storyboard() -> None:
    """Raise if the scripted example leaves the grid or names unknown agents."""
    known = set(AGENTS)
    positions = {}
    for agent_id, spec in AGENTS.items():
        if spec["kind"] not in {item["key"] for item in AGENT_KINDS}:
            raise ValueError(f"Unknown kind for {agent_id}: {spec['kind']}")
        if not in_bounds(spec["start"]):
            raise ValueError(f"Start out of bounds for {agent_id}: {spec['start']}")
        positions[agent_id] = spec["start"]
    for cell in FOOD_START:
        if not in_bounds(cell):
            raise ValueError(f"Food out of bounds: {cell}")
    for index, turn in enumerate(TURNS):
        for agent_id, cell in turn["moves"].items():
            if agent_id not in known:
                raise ValueError(f"Turn {index} names unknown agent {agent_id}")
            if not in_bounds(cell):
                raise ValueError(f"Turn {index} moves {agent_id} out of bounds: {cell}")
            origin = positions[agent_id]
            step = abs(cell[0] - origin[0]) + abs(cell[1] - origin[1])
            if step > 1:
                raise ValueError(f"Turn {index} teleports {agent_id} from {origin} to {cell}")
            positions[agent_id] = cell
        for cell in turn["eat"]:
            if not in_bounds(cell):
                raise ValueError(f"Turn {index} eats out of bounds: {cell}")
