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
TYPE_FONT = "Inter"
# En space is a hair wider than Inter's default word space, so 720p still reads.
TYPE_SPACE = "\u2002"
TYPE_SCALE: dict[str, dict[str, int | str]] = {
    "display": {"size": 46, "weight": "SEMIBOLD"},
    "heading": {"size": 34, "weight": "SEMIBOLD"},
    "lead": {"size": 24, "weight": "MEDIUM"},
    "body": {"size": 25, "weight": "NORMAL"},
    "caption": {"size": 22, "weight": "NORMAL"},
    "label": {"size": 22, "weight": "MEDIUM"},
    "step": {"size": 26, "weight": "SEMIBOLD"},
    "chip": {"size": 20, "weight": "MEDIUM"},
    "meta": {"size": 18, "weight": "NORMAL"},
}
# Vertical gaps between stacked lines, in Manim units.
TYPE_STACK_BUFF: dict[str, float] = {
    "display": 0.22,
    "heading": 0.16,
    "lead": 0.16,
    "body": 0.16,
    "caption": 0.18,
    "label": 0.12,
    "step": 0.12,
    "chip": 0.10,
    "meta": 0.08,
}


def type_role(name: str) -> dict[str, int | str]:
    """Return a copy of a named type-scale role."""
    try:
        return dict(TYPE_SCALE[name])
    except KeyError as exc:
        raise KeyError(name) from exc


def stack_buff(name: str) -> float:
    """Return the vertical gap for a stacked type role."""
    try:
        return TYPE_STACK_BUFF[name]
    except KeyError as exc:
        raise KeyError(name) from exc


def open_words(body: str) -> str:
    """Replace ordinary spaces with the type system's word space."""
    return body.replace(" ", TYPE_SPACE)


def line_span(lines: tuple[str, ...]) -> int:
    """Character-count gap between the longest and shortest stacked line."""
    lengths = [len(line) for line in lines]
    return max(lengths) - min(lengths)


COOPERATIVE = "#2563eb"
SELF_INTERESTED = "#dc2626"
BALANCED = "#d97706"

GRID_SIZE = 8

AGENT_TRAITS = (
    "Lives on the map",
    "Carries some food",
    "Sees what's nearby",
    "Chooses its next move",
)

AGENT_KINDS = (
    {
        "key": "cooperative",
        "label": "Cooperative",
        "color": COOPERATIVE,
        "hint": ("Shares more,", "fights less"),
    },
    {
        "key": "self_interested",
        "label": "Self-interested",
        "color": SELF_INTERESTED,
        "hint": ("Keeps food,", "competes more"),
    },
    {
        "key": "balanced",
        "label": "Balanced",
        "color": BALANCED,
        "hint": ("A middle path,", "for comparison"),
    },
)

HOOK_TITLE = "A limited world."
HOOK_LINE = "Many individuals have to share the food."
HOOK_QUESTION = ("What mix of helpfulness and", "self-interest actually works?")

SECTION_AGENT = "What is an agent?"
CAPTION_AGENT = ("One actor in the world.", "Nobody tells it what to do.")
CAPTION_KINDS = ("They lean in different directions.", "These are tendencies, not personalities.")

SECTION_ENV = "What is the environment?"
CAPTION_ENV = ("The world they share is a grid,", "like a board-game board.")
NOTE_ENV = ("Some of the squares hold food.", "One meal is food someone else cannot eat.")

SECTION_DO = "What do agents do?"
CAPTION_DO = ("On every turn, each living agent", "does the same three things.")

SECTION_GRID = "An example on a grid"
CAPTION_GRID = ("A short run on a small map —", "a postcard, not the experiment.")
CAPTION_WALK = ("They walk toward the food.", "A patch shrinks when someone eats.")
CAPTION_CLUSTER = ("The blue agents gathered on their own.", "Nobody programmed them to do that.")

CLOSE_TITLE = ("The research measures", "the patterns that appear.")
CLOSE_QUESTION = ("Does a mix last longer than a world", "of only helpers — or only competitors?")

# On-screen reading: ~180 wpm plus a rest so the last words are not cut off.
SECONDS_PER_WORD = 0.33
HOLD_REST = 1.5
HOLD_LINE = 1.8
HOLD_READ = 3.8
HOLD_LONG = 5.2
WALK_TIME = 1.0


def as_lines(*texts: str | tuple[str, ...]) -> tuple[str, ...]:
    """Flatten copy strings or line-tuples into a single tuple of lines."""
    lines: list[str] = []
    for text in texts:
        if isinstance(text, tuple):
            lines.extend(text)
        else:
            lines.extend(part for part in text.split("\n") if part)
    return tuple(lines)


def word_count(*texts: str | tuple[str, ...]) -> int:
    """Count whitespace-separated tokens across one or more copy strings."""
    total = 0
    for line in as_lines(*texts):
        total += sum(1 for token in line.split() if token)
    return total


def hold_for(*texts: str | tuple[str, ...], minimum: float = HOLD_LINE) -> float:
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

WORLD_CHANGES = ("After everyone has acted, the map", "updates and the next turn begins.")

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
