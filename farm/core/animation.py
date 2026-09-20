"""Shared helpers for turning a simulation database into pictures.

``scripts/animate_simulation.py`` already rendered experiment runs as a scatter
plot. This module keeps that data query and adds a discrete **grid** renderer
aimed at teaching: one square per cell, food as green patches, agents as
colored dots.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle
from PIL import Image

from farm.utils.logging import get_logger

logger = get_logger(__name__)

PathLike = str | Path

# Everyday labels for the three population types used in the intro.
AGENT_DISPLAY: dict[str, dict[str, str]] = {
    "SystemAgent": {"label": "Cooperative", "color": "#2563eb"},
    "IndependentAgent": {"label": "Self-interested", "color": "#dc2626"},
    "ControlAgent": {"label": "Balanced", "color": "#d97706"},
}
AGENT_TYPE_ALIASES = {
    "systemagent": "SystemAgent",
    "system": "SystemAgent",
    "independentagent": "IndependentAgent",
    "independent": "IndependentAgent",
    "controlagent": "ControlAgent",
    "control": "ControlAgent",
}
UNKNOWN_AGENT_COLOR = "#6b7280"
MIN_VISIBLE_RESOURCE = 0.75
RESOURCE_COLOR = "#16a34a"
GRID_EDGE = "#d4d4d8"
FIG_FACE = "#f7f7f8"
AX_FACE = "#ffffff"
TITLE_COLOR = "#111827"
MUTED_COLOR = "#4b5563"


def world_size_from_config(config: Mapping[str, Any], default: tuple[int, int] = (100, 100)) -> tuple[int, int]:
    """Read ``(width, height)`` from a nested or flattened config dict."""
    environment = config.get("environment")
    if isinstance(environment, Mapping):
        width = environment.get("width", config.get("width", default[0]))
        height = environment.get("height", config.get("height", default[1]))
    else:
        width = config.get("width", default[0])
        height = config.get("height", default[1])
    return int(width), int(height)


def to_cell(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Map a continuous position onto a grid cell, clamped to the world."""
    cell_x = int(np.floor(float(x)))
    cell_y = int(np.floor(float(y)))
    cell_x = min(max(cell_x, 0), max(width - 1, 0))
    cell_y = min(max(cell_y, 0), max(height - 1, 0))
    return cell_x, cell_y


def normalize_agent_type(raw: Any) -> str:
    """Map database or class names onto the canonical ``*Agent`` keys."""
    key = str(raw).strip().replace("_", "").replace("-", "").lower()
    return AGENT_TYPE_ALIASES.get(key, str(raw))


def agent_style(raw: Any) -> dict[str, str]:
    """Return ``label`` and ``color`` for an agent type, with a gray fallback."""
    canonical = normalize_agent_type(raw)
    return AGENT_DISPLAY.get(canonical, {"label": str(raw), "color": UNKNOWN_AGENT_COLOR})


def cell_offsets(count: int) -> list[tuple[float, float]]:
    """Spread several occupants of the same cell so they do not cover each other."""
    if count <= 1:
        return [(0.5, 0.5)]
    radius = 0.22
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return [(0.5 + radius * float(np.cos(angle)), 0.5 + radius * float(np.sin(angle))) for angle in angles]


def get_state_at_step(conn: sqlite3.Connection, step_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return agent and resource rows for one simulation step."""
    agents = pd.read_sql_query(
        """
        SELECT a.agent_id, ag.agent_type, a.position_x, a.position_y, a.resource_level as resources
        FROM agent_states a
        JOIN agents ag ON a.agent_id = ag.agent_id
        WHERE a.step_number = ?
        """,
        conn,
        params=(step_number,),
    )
    resources = pd.read_sql_query(
        """
        SELECT resource_id, position_x, position_y, amount
        FROM resource_states
        WHERE step_number = ?
        """,
        conn,
        params=(step_number,),
    )
    return agents, resources


def list_step_numbers(conn: sqlite3.Connection) -> list[int]:
    """Sorted step numbers that have at least one agent or resource row."""
    agent_steps = pd.read_sql_query(
        "SELECT DISTINCT step_number FROM agent_states ORDER BY step_number",
        conn,
    )
    if agent_steps.empty:
        resource_steps = pd.read_sql_query(
            "SELECT DISTINCT step_number FROM resource_states ORDER BY step_number",
            conn,
        )
        if resource_steps.empty:
            return []
        return [int(step) for step in resource_steps["step_number"].tolist()]
    return [int(step) for step in agent_steps["step_number"].tolist()]


def create_grid_frame(
    agents: pd.DataFrame,
    resources: pd.DataFrame,
    step_number: int,
    width: int,
    height: int,
    *,
    title: str | None = None,
    subtitle: str = "Each square is a place. Green is food. Colored dots are agents.",
) -> plt.Figure:
    """Draw one teaching-oriented grid frame.

    Args:
        agents: Columns ``agent_id``, ``agent_type``, ``position_x``, ``position_y``, ``resources``.
        resources: Columns ``position_x``, ``position_y``, ``amount``.
        step_number: Step shown in the title.
        width: Grid width in cells.
        height: Grid height in cells.
        title: Optional override for the figure title.
        subtitle: Plain-language caption under the title.
    """
    fig, ax = plt.subplots(figsize=(7.0, 8.1), dpi=100)
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(AX_FACE)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_EDGE)

    for cell_x in range(width):
        for cell_y in range(height):
            ax.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    1,
                    1,
                    facecolor=AX_FACE,
                    edgecolor=GRID_EDGE,
                    linewidth=0.6,
                )
            )

    if resources is not None and not resources.empty:
        amounts: dict[tuple[int, int], float] = {}
        for _, row in resources.iterrows():
            cell = to_cell(row["position_x"], row["position_y"], width, height)
            amounts[cell] = amounts.get(cell, 0.0) + float(row["amount"])
        max_amount = max(amounts.values()) if amounts else 1.0
        for (cell_x, cell_y), amount in amounts.items():
            if amount < MIN_VISIBLE_RESOURCE:
                continue
            alpha = 0.30 + 0.55 * min(amount / max(max_amount, 1.0), 1.0)
            ax.add_patch(
                FancyBboxPatch(
                    (cell_x + 0.12, cell_y + 0.12),
                    0.76,
                    0.76,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=RESOURCE_COLOR,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1,
                )
            )

    if agents is not None and not agents.empty:
        grouped: dict[tuple[int, int], list[pd.Series]] = {}
        for _, row in agents.iterrows():
            cell = to_cell(row["position_x"], row["position_y"], width, height)
            grouped.setdefault(cell, []).append(row)

        for (cell_x, cell_y), occupants in grouped.items():
            offsets = cell_offsets(len(occupants))
            for row, (offset_x, offset_y) in zip(occupants, offsets):
                style = agent_style(row["agent_type"])
                energy = max(float(row.get("resources", 0.0) or 0.0), 0.0)
                radius = 0.20 + 0.10 * min(energy / 20.0, 1.0)
                ax.add_patch(
                    Circle(
                        (cell_x + offset_x, cell_y + offset_y),
                        radius,
                        facecolor=style["color"],
                        edgecolor="#111827",
                        linewidth=0.9,
                        zorder=3,
                    )
                )

    if title is None:
        title = f"A small grid world — turn {step_number}"
    fig.suptitle(title, fontsize=14, fontweight=600, color=TITLE_COLOR, y=0.98)
    ax.set_title(subtitle, fontsize=10, color=MUTED_COLOR, pad=8)

    legend_handles = [
        Patch(facecolor=RESOURCE_COLOR, edgecolor="none", alpha=0.7, label="Food"),
    ]
    seen_types = set()
    if agents is not None and not agents.empty:
        for agent_type in agents["agent_type"].unique():
            seen_types.add(normalize_agent_type(agent_type))
    for agent_type, style in AGENT_DISPLAY.items():
        if agent_type in seen_types or not seen_types:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=9,
                    markerfacecolor=style["color"],
                    markeredgecolor="#111827",
                    markeredgewidth=0.8,
                    label=style["label"],
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(len(legend_handles), 4),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    return fig


def write_grid_gif(
    db_path: PathLike,
    output_path: PathLike,
    width: int,
    height: int,
    *,
    fps: int = 8,
    skip_frames: int = 1,
    max_steps: int | None = None,
    title_prefix: str = "A small grid world",
) -> Path:
    """Render a GIF of every (or every nth) step in ``db_path``."""
    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_frames < 1:
        raise ValueError("skip_frames must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")

    conn = sqlite3.connect(str(db_path))
    try:
        steps = list_step_numbers(conn)
        if not steps:
            raise ValueError(f"No simulation steps found in {db_path}")
        if max_steps is not None:
            steps = [step for step in steps if step <= max_steps]
        steps = steps[::skip_frames]

        frame_paths: list[Path] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for step in steps:
                agents, resources = get_state_at_step(conn, step)
                fig = create_grid_frame(
                    agents,
                    resources,
                    step,
                    width,
                    height,
                    title=f"{title_prefix} — turn {step}",
                )
                frame_path = temp_root / f"frame_{step:04d}.png"
                fig.savefig(frame_path, dpi=100, facecolor=fig.get_facecolor())
                plt.close(fig)
                frame_paths.append(frame_path)

            _pngs_to_gif(frame_paths, output_path, fps=fps)
    finally:
        conn.close()

    logger.info("grid_gif_written", path=str(output_path), frames=len(steps), fps=fps)
    return output_path


def _pngs_to_gif(frame_paths: Sequence[Path], output_path: Path, fps: int) -> None:
    """Assemble PNG frames into a looping GIF with Pillow."""
    if not frame_paths:
        raise ValueError("No frames to write")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    duration_ms = round(1000.0 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    for image in images:
        image.close()


def write_grid_gif_from_states(
    states: Iterable[tuple[int, pd.DataFrame, pd.DataFrame]],
    output_path: PathLike,
    width: int,
    height: int,
    *,
    fps: int = 8,
    title_prefix: str = "A small grid world",
) -> Path:
    """Render a GIF from in-memory ``(step, agents, resources)`` tuples.

    Used by tests and by callers that already have step data loaded.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for step, agents, resources in states:
            fig = create_grid_frame(
                agents,
                resources,
                step,
                width,
                height,
                title=f"{title_prefix} — turn {step}",
            )
            frame_path = temp_root / f"frame_{int(step):04d}.png"
            fig.savefig(frame_path, dpi=100, facecolor=fig.get_facecolor())
            plt.close(fig)
            frame_paths.append(frame_path)
        _pngs_to_gif(frame_paths, output_path, fps=fps)
    return output_path
