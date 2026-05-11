#!/usr/bin/env python3
"""make_foraging_gif.py

Generate a GIF of agents foraging on the environment grid from a simulation
database. The output emphasizes the foraging dynamic: resource patches deplete
and regenerate over time while agents move across the grid and gather from
nearby nodes.

Two input modes are supported:

1. Direct database path (preferred for ad-hoc renders)::

    python scripts/make_foraging_gif.py \
        --db-path docs/sample/simulation.db \
        --output docs/devlog/figures/foraging-grid.gif

2. Experiment iteration folder (compatible with `animate_simulation.py`)::

    python scripts/make_foraging_gif.py \
        --experiment-path results/<experiment_dir> \
        --iteration 0 \
        --output docs/devlog/figures/foraging-grid.gif

The GIF is assembled with PIL (no ffmpeg/moviepy dependency), so the script
works in any environment with `matplotlib` and `pillow` installed.
"""

import argparse
import io
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


# Reasonable defaults for the foraging visualization. Designed to keep the
# resulting GIF small enough to embed in a markdown devlog while still showing
# meaningful dynamics.
DEFAULT_FPS = 10
DEFAULT_STEP_STRIDE = 5
DEFAULT_FIG_INCHES = 6.0
DEFAULT_DPI = 110

AGENT_TYPE_COLORS = {
    "BaseAgent": "#1f77b4",
    "SystemAgent": "#1f77b4",
    "IndependentAgent": "#d62728",
    "ControlAgent": "#ff7f0e",
}
DEFAULT_AGENT_COLOR = "#1f77b4"


@dataclass(frozen=True)
class WorldBounds:
    """World extents used for axis bounds when rendering frames."""

    width: float
    height: float


def _resolve_db_and_config(args: argparse.Namespace) -> tuple[Path, Optional[dict]]:
    """Return the simulation db path and an optional iteration `config.json` dict.

    The iteration config is only consulted as a fallback for world bounds; we
    prefer the per-simulation `simulation_config` table when present.
    """
    if args.db_path is not None:
        db_path = Path(args.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"--db-path does not exist: {db_path}")
        config_path = Path(args.config_path) if args.config_path else None
        if config_path is not None and not config_path.exists():
            raise FileNotFoundError(f"--config-path does not exist: {config_path}")
        config = json.loads(config_path.read_text()) if config_path else None
        return db_path, config

    if args.experiment_path is None or args.iteration is None:
        raise ValueError(
            "Provide --db-path, or both --experiment-path and --iteration."
        )

    folder = Path(args.experiment_path) / f"iteration_{args.iteration}"
    db_path = folder / "simulation.db"
    config_path = folder / "config.json"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing simulation.db in {folder}")
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return db_path, config


def _read_world_bounds_from_db(conn: sqlite3.Connection) -> Optional[WorldBounds]:
    """Pull `environment.width` / `environment.height` from `simulation_config`.

    The config blob stores the SimulationConfig as flattened dotted keys; the
    relevant entries are `environment.width` and `environment.height`.
    """
    try:
        row = conn.execute(
            "SELECT config_data FROM simulation_config LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    if row is None:
        return None
    try:
        cfg = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    width = cfg.get("environment.width") or cfg.get("width")
    height = cfg.get("environment.height") or cfg.get("height")
    if width is None or height is None:
        return None
    return WorldBounds(width=float(width), height=float(height))


def _read_world_bounds_from_config(config: Optional[dict]) -> Optional[WorldBounds]:
    if not config:
        return None
    width = config.get("width") or config.get("environment", {}).get("width")
    height = config.get("height") or config.get("environment", {}).get("height")
    if width is None or height is None:
        return None
    return WorldBounds(width=float(width), height=float(height))


def _infer_world_bounds_from_data(conn: sqlite3.Connection) -> WorldBounds:
    """Fallback: infer extents from the union of agent and resource positions."""
    row = conn.execute(
        """
        SELECT MAX(x), MAX(y) FROM (
            SELECT MAX(position_x) AS x, MAX(position_y) AS y FROM agent_states
            UNION ALL
            SELECT MAX(position_x), MAX(position_y) FROM resource_states
        )
        """
    ).fetchone()
    width = float(row[0]) if row and row[0] is not None else 100.0
    height = float(row[1]) if row and row[1] is not None else 100.0
    return WorldBounds(width=width, height=height)


def _resolve_world_bounds(
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    iteration_config: Optional[dict],
) -> WorldBounds:
    if args.world_width is not None and args.world_height is not None:
        return WorldBounds(width=float(args.world_width), height=float(args.world_height))
    bounds = _read_world_bounds_from_db(conn)
    if bounds is not None:
        return bounds
    bounds = _read_world_bounds_from_config(iteration_config)
    if bounds is not None:
        return bounds
    return _infer_world_bounds_from_data(conn)


def _select_steps(
    conn: sqlite3.Connection,
    *,
    max_steps: Optional[int],
    step_stride: int,
) -> list[int]:
    row = conn.execute("SELECT MAX(step_number) FROM agent_states").fetchone()
    last_step = int(row[0]) if row and row[0] is not None else -1
    if last_step < 0:
        return []
    if max_steps is not None:
        last_step = min(last_step, max(0, max_steps - 1))
    return list(range(0, last_step + 1, max(1, step_stride)))


def _load_agent_type_map(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT agent_id, agent_type FROM agents").fetchall()
    return {agent_id: agent_type for agent_id, agent_type in rows}


def _load_step_state(
    conn: sqlite3.Connection,
    step_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agents = pd.read_sql_query(
        """
        SELECT agent_id, position_x, position_y, resource_level
        FROM agent_states
        WHERE step_number = ?
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


def _load_timeline_series(
    conn: sqlite3.Connection, max_step: int
) -> pd.DataFrame:
    """Load per-step totals for the sparkline timeline panel."""
    timeline = pd.read_sql_query(
        """
        SELECT step_number, total_agents, total_resources
        FROM simulation_steps
        WHERE step_number <= ?
        ORDER BY step_number
        """,
        conn,
        params=(max_step,),
    )
    if timeline.empty:
        return pd.DataFrame(
            columns=["step_number", "total_agents", "total_resources"]
        )
    return timeline


def _load_death_markers_by_step(
    conn: sqlite3.Connection,
) -> dict[int, list[tuple[float, float]]]:
    """Return map of death step -> positions where agents die."""
    rows = conn.execute(
        """
        WITH death_steps AS (
            SELECT agent_id, CAST(death_time AS INTEGER) AS death_step
            FROM agents
            WHERE death_time IS NOT NULL
        ),
        last_positions AS (
            SELECT
                ds.agent_id,
                ds.death_step,
                ast.position_x,
                ast.position_y,
                ROW_NUMBER() OVER (
                    PARTITION BY ds.agent_id
                    ORDER BY ast.step_number DESC
                ) AS rn
            FROM death_steps ds
            JOIN agent_states ast
                ON ast.agent_id = ds.agent_id
               AND ast.step_number <= ds.death_step
        )
        SELECT death_step, position_x, position_y
        FROM last_positions
        WHERE rn = 1
        """
    ).fetchall()
    by_step: dict[int, list[tuple[float, float]]] = {}
    for death_step, pos_x, pos_y in rows:
        by_step.setdefault(int(death_step), []).append((float(pos_x), float(pos_y)))
    return by_step


def _load_birth_markers_by_step(
    conn: sqlite3.Connection,
) -> dict[int, list[tuple[float, float]]]:
    """Return map of birth step -> positions where agents are born."""
    rows = conn.execute(
        """
        SELECT
            CAST(a.birth_time AS INTEGER) AS birth_step,
            ast.position_x,
            ast.position_y
        FROM agents a
        JOIN agent_states ast
            ON ast.agent_id = a.agent_id
           AND ast.step_number = CAST(a.birth_time AS INTEGER)
        """
    ).fetchall()
    by_step: dict[int, list[tuple[float, float]]] = {}
    for birth_step, pos_x, pos_y in rows:
        by_step.setdefault(int(birth_step), []).append((float(pos_x), float(pos_y)))
    return by_step


def _project_deaths_onto_render_steps(
    render_steps: list[int],
    death_markers_by_step: dict[int, list[tuple[float, float]]],
) -> dict[int, list[tuple[float, float]]]:
    """Map each death to the first rendered frame at/after death time."""
    if not render_steps:
        return {}
    projected: dict[int, list[tuple[float, float]]] = {}
    for death_step in sorted(death_markers_by_step):
        target_step = next((s for s in render_steps if s >= death_step), None)
        if target_step is None:
            continue
        projected.setdefault(target_step, []).extend(death_markers_by_step[death_step])
    return projected


def _project_markers_onto_render_steps(
    render_steps: list[int],
    markers_by_step: dict[int, list[tuple[float, float]]],
) -> dict[int, list[tuple[float, float]]]:
    """Map event markers to the first rendered frame at/after event time."""
    if not render_steps:
        return {}
    projected: dict[int, list[tuple[float, float]]] = {}
    for marker_step in sorted(markers_by_step):
        target_step = next((s for s in render_steps if s >= marker_step), None)
        if target_step is None:
            continue
        projected.setdefault(target_step, []).extend(markers_by_step[marker_step])
    return projected


def _render_frame(
    *,
    agents: pd.DataFrame,
    resources: pd.DataFrame,
    agent_types: dict[str, str],
    bounds: WorldBounds,
    step_number: int,
    birth_markers: list[tuple[float, float]],
    death_markers: list[tuple[float, float]],
    timeline: pd.DataFrame,
    fig_inches: float,
    dpi: int,
) -> Image.Image:
    """Render one frame of the foraging grid as a PIL image."""
    fig: Figure = plt.figure(figsize=(fig_inches, fig_inches), dpi=dpi)
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    # Append directly under the map so width matches the square grid exactly.
    ax_timeline = divider.append_axes("bottom", size="14%", pad=0.06)

    # Light grid background to convey the discrete cell structure of the world.
    ax.set_xlim(0, bounds.width)
    ax.set_ylim(0, bounds.height)
    ax.set_aspect("equal")
    grid_step = max(1, int(round(max(bounds.width, bounds.height) / 25)))
    ax.set_xticks(range(0, int(bounds.width) + 1, grid_step))
    ax.set_yticks(range(0, int(bounds.height) + 1, grid_step))
    ax.grid(True, which="major", color="#dddddd", linewidth=0.5, zorder=0)
    ax.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")

    if not resources.empty:
        amounts = resources["amount"].clip(lower=0)
        max_amount = float(amounts.max()) if len(amounts) else 1.0
        sizes = 20 + 80 * (amounts / max(max_amount, 1.0))
        ax.scatter(
            resources["position_x"],
            resources["position_y"],
            s=sizes,
            c="#2ca02c",
            alpha=0.55,
            edgecolors="none",
            zorder=1,
            label="Resources",
        )

    if not agents.empty:
        colors = [
            AGENT_TYPE_COLORS.get(agent_types.get(aid, ""), DEFAULT_AGENT_COLOR)
            for aid in agents["agent_id"]
        ]
        agent_amounts = agents["resource_level"].clip(lower=0)
        sizes = 25 + 30 * (agent_amounts / max(float(agent_amounts.max() or 1.0), 1.0))
        ax.scatter(
            agents["position_x"],
            agents["position_y"],
            s=sizes,
            c=colors,
            edgecolors="black",
            linewidths=0.4,
            zorder=2,
        )

    if birth_markers:
        marker_x = [m[0] for m in birth_markers]
        marker_y = [m[1] for m in birth_markers]
        ax.scatter(
            marker_x,
            marker_y,
            marker="+",
            s=110,
            c="#ffd92f",
            linewidths=2.2,
            zorder=3,
        )

    if death_markers:
        marker_x = [m[0] for m in death_markers]
        marker_y = [m[1] for m in death_markers]
        ax.scatter(
            marker_x,
            marker_y,
            marker="x",
            s=90,
            c="#d62728",
            linewidths=1.8,
            zorder=3,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DEFAULT_AGENT_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
            label="Agents",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="none",
            alpha=0.55,
            markersize=7,
            label="Resources",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            color="#ffd92f",
            markersize=8,
            markeredgewidth=2.2,
            linestyle="none",
            label="Agent birth",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#d62728",
            markersize=7,
            markeredgewidth=1.8,
            linestyle="none",
            label="Agent death",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=8,
        framealpha=0.9,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.4,
    )

    ax.set_title(f"Foraging — step {step_number:>4d}", fontsize=10)
    if not timeline.empty:
        step_values = timeline["step_number"]
        agent_values = timeline["total_agents"]
        resource_values = timeline["total_resources"]
        agents_min = float(agent_values.min())
        agents_max = float(agent_values.max())
        normalized_agents = (agent_values - agents_min) / max(
            agents_max - agents_min, 1.0
        )

        # Log compression dampens the large early resource swing.
        resource_log = np.log1p(resource_values)
        resources_min = float(resource_log.min())
        resources_max = float(resource_log.max())
        normalized_resources = (resource_log - resources_min) / max(
            resources_max - resources_min, 1e-9
        )
        max_timeline_step = int(step_values.max())

        # Smooth both series for a cleaner sparkline panel.
        smooth_window = max(5, min(31, len(step_values) // 30))
        smoothed_agents = pd.Series(normalized_agents).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        smoothed_resources = pd.Series(normalized_resources).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()

        # Draw full series lightly as context.
        ax_timeline.plot(
            step_values,
            smoothed_agents,
            color=DEFAULT_AGENT_COLOR,
            linewidth=1.3,
            alpha=0.28,
            zorder=1,
        )
        ax_timeline.plot(
            step_values,
            smoothed_resources,
            color="#2ca02c",
            linewidth=1.3,
            alpha=0.28,
            zorder=1,
        )

        # Fade the future region to the right of current step.
        ax_timeline.axvspan(
            step_number,
            max_timeline_step,
            color="white",
            alpha=0.28,
            zorder=2,
        )

        # Re-draw historical segment as clear lines.
        history = timeline[timeline["step_number"] <= step_number]
        history_agents = pd.Series(
            (history["total_agents"] - agents_min) / max(agents_max - agents_min, 1.0)
        ).rolling(window=smooth_window, center=True, min_periods=1).mean()
        history_resources = pd.Series(
            (np.log1p(history["total_resources"]) - resources_min)
            / max(resources_max - resources_min, 1e-9)
        ).rolling(window=smooth_window, center=True, min_periods=1).mean()
        ax_timeline.plot(
            history["step_number"],
            history_agents,
            color=DEFAULT_AGENT_COLOR,
            linewidth=2.0,
            alpha=1.0,
            zorder=3,
        )
        ax_timeline.plot(
            history["step_number"],
            history_resources,
            color="#2ca02c",
            linewidth=2.0,
            alpha=1.0,
            zorder=3,
        )
        ax_timeline.axvline(
            step_number, color="#444444", linewidth=1.3, alpha=0.9, zorder=4
        )

        ax_timeline.set_xlim(0, max_timeline_step)
        ax_timeline.set_ylim(0, 1.05)
        ax_timeline.margins(x=0.0)

    # Sparkline-like minimal styling.
    ax_timeline.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax_timeline.spines.values():
        spine.set_color("#d2d2d2")
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.set_facecolor("#f8f9fb")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.06)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def _save_gif(frames: Iterable[Image.Image], output: Path, fps: int) -> None:
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No frames were produced; nothing to write.")
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    frames_list[0].save(
        output,
        save_all=True,
        append_images=frames_list[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def make_foraging_gif(args: argparse.Namespace) -> Path:
    db_path, iteration_config = _resolve_db_and_config(args)
    output = Path(args.output)

    with closing(sqlite3.connect(str(db_path))) as conn:
        bounds = _resolve_world_bounds(args, conn, iteration_config)
        steps = _select_steps(
            conn,
            max_steps=args.max_steps,
            step_stride=args.step_stride,
        )
        if not steps:
            raise RuntimeError(f"No step data found in {db_path}")
        agent_types = _load_agent_type_map(conn)
        birth_markers_by_step = _load_birth_markers_by_step(conn)
        death_markers_by_step = _load_death_markers_by_step(conn)
        projected_birth_markers = _project_markers_onto_render_steps(
            steps, birth_markers_by_step
        )
        projected_death_markers = _project_markers_onto_render_steps(
            steps, death_markers_by_step
        )
        timeline = _load_timeline_series(conn, max_step=steps[-1])

        print(
            f"Rendering {len(steps)} frames from {db_path} "
            f"(world {bounds.width:g}x{bounds.height:g}) -> {output}"
        )
        frames: list[Image.Image] = []
        for idx, step in enumerate(steps):
            agents, resources = _load_step_state(conn, step)
            frame = _render_frame(
                agents=agents,
                resources=resources,
                agent_types=agent_types,
                bounds=bounds,
                step_number=step,
                birth_markers=projected_birth_markers.get(step, []),
                death_markers=projected_death_markers.get(step, []),
                timeline=timeline,
                fig_inches=args.fig_inches,
                dpi=args.dpi,
            )
            frames.append(frame)
            if (idx + 1) % 25 == 0 or idx == len(steps) - 1:
                print(f"  rendered {idx + 1}/{len(steps)} frames")

    _save_gif(frames, output, fps=args.fps)
    print(f"Wrote GIF: {output} ({output.stat().st_size / 1024:.1f} KiB)")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a foraging-focused GIF from a simulation database.",
    )
    src = parser.add_argument_group("input source (choose one)")
    src.add_argument("--db-path", type=str, help="Path to a simulation.db file.")
    src.add_argument("--config-path", type=str, help="Optional config.json for world bounds.")
    src.add_argument("--experiment-path", type=str, help="Experiment directory containing iteration_<N>/.")
    src.add_argument("--iteration", type=int, help="Iteration number to render.")

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output",
        type=str,
        default="docs/devlog/figures/foraging-grid.gif",
        help="Path to write the resulting GIF (default: %(default)s).",
    )

    enc = parser.add_argument_group("encoding")
    enc.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second (default: %(default)s).")
    enc.add_argument("--step-stride", type=int, default=DEFAULT_STEP_STRIDE,
                     help="Render every Nth simulation step (default: %(default)s).")
    enc.add_argument("--max-steps", type=int, default=None,
                     help="Cap the number of simulation steps rendered.")
    enc.add_argument("--fig-inches", type=float, default=DEFAULT_FIG_INCHES,
                     help="Square figure size in inches (default: %(default)s).")
    enc.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                     help="Frame DPI (default: %(default)s).")

    bnd = parser.add_argument_group("world bounds (optional override)")
    bnd.add_argument("--world-width", type=float, default=None)
    bnd.add_argument("--world-height", type=float, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        make_foraging_gif(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
