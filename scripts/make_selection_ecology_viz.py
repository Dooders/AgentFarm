#!/usr/bin/env python3
"""Render a "selection through ecology" figure from a simulation database.

The script generates a three-panel figure suitable for the devlog section:

1) Ecological pressure over time: total resources and active population.
2) Per-agent resource budget decomposition:
   gathered_in - metabolic_cost - reproduction_cost.
3) Survival/reproduction phase plot:
   net resource per step vs reproduction events, split by fate.

Example:
    python scripts/make_selection_ecology_viz.py \
        --db-path docs/sample/simulation.db \
        --output docs/devlog/figures/selection-through-ecology.png
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


TEXT_DARK = "#1F2933"
TEXT_MUTED = "#52606D"
GRID_COLOR = "#D8DEE9"
BG_COLOR = "#F8FAFC"

COLOR_GATHER = "#2CA02C"
COLOR_METABOLISM = "#4C6EF5"
COLOR_REPRODUCTION = "#E67E22"
COLOR_SURVIVED = "#1F77B4"
COLOR_STARVED = "#D62728"
COLOR_OTHER_LINEAGE = "#9AA5B1"


@dataclass(frozen=True)
class EcologyConfig:
    base_consumption_rate: float
    starvation_threshold: float
    offspring_cost: float


def _smooth(series: pd.Series, n_points: int) -> pd.Series:
    window = max(5, min(41, int(max(5, n_points // 30))))
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _style_axis(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the selection-through-ecology figure from simulation telemetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to simulation SQLite database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/devlog/figures/selection-through-ecology.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--top-agents",
        type=int,
        default=22,
        help="Number of agents shown in the budget-decomposition panel.",
    )
    parser.add_argument(
        "--min-lifetime-steps",
        type=int,
        default=10,
        help="Minimum lifetime steps required for inclusion in budget/phase panels.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Saved image DPI.")
    return parser.parse_args()


def _load_simulation_config(conn: sqlite3.Connection) -> EcologyConfig:
    row = conn.execute("SELECT config_data FROM simulation_config LIMIT 1").fetchone()
    cfg = {}
    if row and row[0]:
        try:
            cfg = json.loads(row[0])
        except json.JSONDecodeError:
            cfg = {}

    def _pick(*keys: str, default: float) -> float:
        for key in keys:
            if key in cfg and cfg[key] is not None:
                return float(cfg[key])
        return float(default)

    return EcologyConfig(
        base_consumption_rate=_pick(
            "agent_behavior.base_consumption_rate",
            "base_consumption_rate",
            default=0.0,
        ),
        starvation_threshold=_pick(
            "agent_behavior.starvation_threshold",
            "starvation_threshold",
            default=np.nan,
        ),
        offspring_cost=_pick(
            "agent_behavior.offspring_cost",
            "offspring_cost",
            default=np.nan,
        ),
    )


def _load_step_series(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            step_number,
            total_resources,
            total_agents,
            births,
            deaths
        FROM simulation_steps
        ORDER BY step_number
        """,
        conn,
    )


def _load_agent_table(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            agent_id,
            birth_time,
            death_time,
            genome_id,
            generation
        FROM agents
        """,
        conn,
    )


def _load_starved_agents(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT agent_id
        FROM health_incidents
        WHERE cause = 'starvation'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _safe_json(details: object) -> dict:
    if isinstance(details, dict):
        return details
    if isinstance(details, str) and details:
        try:
            parsed = json.loads(details)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _load_action_budget(conn: sqlite3.Connection) -> pd.DataFrame:
    actions = pd.read_sql_query(
        """
        SELECT
            agent_id,
            action_type,
            resources_before,
            resources_after,
            details
        FROM agent_actions
        WHERE action_type IN ('gather', 'reproduce')
        """,
        conn,
    )
    if actions.empty:
        return pd.DataFrame(
            columns=[
                "agent_id",
                "gathered_total",
                "reproduction_cost_total",
                "reproduction_events",
            ]
        )

    gathered_by_agent: dict[str, float] = {}
    reproduction_cost_by_agent: dict[str, float] = {}
    reproduction_events_by_agent: dict[str, int] = {}

    for row in actions.itertuples(index=False):
        details = _safe_json(row.details)
        aid = str(row.agent_id)

        before = row.resources_before
        after = row.resources_after
        delta = np.nan
        if before is not None and after is not None:
            delta = float(after) - float(before)

        if row.action_type == "gather":
            gathered = 0.0
            if np.isfinite(delta) and delta > 0:
                gathered = float(delta)
            else:
                gathered = float(details.get("amount_gathered", 0.0) or 0.0)
            if gathered > 0:
                gathered_by_agent[aid] = gathered_by_agent.get(aid, 0.0) + gathered
            continue

        if row.action_type == "reproduce":
            paid_cost = 0.0
            if np.isfinite(delta) and delta < 0:
                paid_cost = float(-delta)
            else:
                # Fallback when before/after fields are unavailable.
                before_d = details.get("resources_before")
                after_d = details.get("resources_after")
                if before_d is not None and after_d is not None:
                    paid_cost = max(0.0, float(before_d) - float(after_d))
                else:
                    paid_cost = 0.0

            if paid_cost > 0:
                reproduction_cost_by_agent[aid] = (
                    reproduction_cost_by_agent.get(aid, 0.0) + paid_cost
                )
                reproduction_events_by_agent[aid] = (
                    reproduction_events_by_agent.get(aid, 0) + 1
                )

    all_ids = set(gathered_by_agent) | set(reproduction_cost_by_agent) | set(reproduction_events_by_agent)
    rows = []
    for aid in all_ids:
        rows.append(
            {
                "agent_id": aid,
                "gathered_total": gathered_by_agent.get(aid, 0.0),
                "reproduction_cost_total": reproduction_cost_by_agent.get(aid, 0.0),
                "reproduction_events": reproduction_events_by_agent.get(aid, 0),
            }
        )
    return pd.DataFrame(rows)


def _build_agent_budget_frame(
    agents: pd.DataFrame,
    action_budget: pd.DataFrame,
    starved_agents: set[str],
    *,
    final_step: int,
    base_consumption_rate: float,
) -> pd.DataFrame:
    frame = agents.copy()
    frame["death_time"] = pd.to_numeric(frame["death_time"], errors="coerce")
    frame["birth_time"] = pd.to_numeric(frame["birth_time"], errors="coerce").fillna(0)
    frame["end_time"] = frame["death_time"].fillna(final_step).clip(upper=final_step)
    frame["alive_steps"] = (frame["end_time"] - frame["birth_time"] + 1).clip(lower=0).astype(float)
    frame["metabolic_cost_total"] = frame["alive_steps"] * float(base_consumption_rate)
    frame["survived_to_end"] = frame["death_time"].isna() | (frame["death_time"] >= final_step)
    frame["starved"] = frame["agent_id"].astype(str).isin(starved_agents)

    frame = frame.merge(action_budget, on="agent_id", how="left")
    for col in ("gathered_total", "reproduction_cost_total", "reproduction_events"):
        frame[col] = frame[col].fillna(0.0)

    frame["reproduction_events"] = frame["reproduction_events"].astype(int)
    frame["net_resource"] = (
        frame["gathered_total"] - frame["metabolic_cost_total"] - frame["reproduction_cost_total"]
    )
    frame["net_per_step"] = frame["net_resource"] / frame["alive_steps"].replace(0, np.nan)
    frame["genome_id"] = frame["genome_id"].fillna("unknown")
    frame["generation"] = pd.to_numeric(frame["generation"], errors="coerce").fillna(-1).astype(int)
    return frame


def _plot_ecology_timeseries(ax: plt.Axes, steps: pd.DataFrame) -> None:
    if steps.empty:
        ax.text(0.5, 0.5, "No step telemetry available", ha="center", va="center", color=TEXT_MUTED)
        ax.set_axis_off()
        return

    smoothed_resources = _smooth(steps["total_resources"], n_points=len(steps))
    smoothed_agents = _smooth(steps["total_agents"], n_points=len(steps))

    ax.plot(
        steps["step_number"],
        steps["total_resources"],
        color=COLOR_GATHER,
        linewidth=1.0,
        alpha=0.25,
    )
    ax.plot(
        steps["step_number"],
        smoothed_resources,
        color=COLOR_GATHER,
        linewidth=2.5,
        label="Total resources (smoothed)",
    )
    ax.fill_between(
        steps["step_number"],
        smoothed_resources,
        color=COLOR_GATHER,
        alpha=0.08,
    )
    ax.set_ylabel("Resources in environment", color=COLOR_GATHER)
    ax.tick_params(axis="y", colors=COLOR_GATHER)
    ax.grid(True, axis="both", linestyle=":", color=GRID_COLOR, alpha=0.9)
    ax.set_xlabel("Simulation step")
    ax.set_title("A) Ecological pressure: resources and population", loc="left", fontsize=12, fontweight="bold")

    ax_right = ax.twinx()
    ax_right.plot(
        steps["step_number"],
        steps["total_agents"],
        color=COLOR_SURVIVED,
        linewidth=0.9,
        alpha=0.25,
    )
    ax_right.plot(
        steps["step_number"],
        smoothed_agents,
        color=COLOR_SURVIVED,
        linewidth=2.3,
        alpha=0.95,
        label="Active population (smoothed)",
    )
    ax_right.set_ylabel("Active agents", color=COLOR_SURVIVED)
    ax_right.tick_params(axis="y", colors=COLOR_SURVIVED)

    handles = []
    for axes in (ax, ax_right):
        h, _ = axes.get_legend_handles_labels()
        handles.extend(h)
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5)

    ax.annotate(
        f"end: {int(steps['total_agents'].iloc[-1])} agents",
        xy=(steps["step_number"].iloc[-1], smoothed_agents.iloc[-1]),
        xytext=(-90, 18),
        textcoords="offset points",
        color=COLOR_SURVIVED,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=COLOR_SURVIVED, lw=0.8, alpha=0.8),
    )
    _style_axis(ax)


def _plot_budget_breakdown(ax: plt.Axes, budget: pd.DataFrame, top_agents: int) -> None:
    if budget.empty:
        ax.text(0.5, 0.5, "No agent budget data available", ha="center", va="center", color=TEXT_MUTED)
        ax.set_axis_off()
        return

    ranked = budget.sort_values("net_resource", ascending=False)
    if len(ranked) > top_agents:
        keep_head = top_agents // 2
        keep_tail = top_agents - keep_head
        ranked = pd.concat([ranked.head(keep_head), ranked.tail(keep_tail)], axis=0)
    ranked = ranked.sort_values("net_resource", ascending=True).copy()

    y_pos = np.arange(len(ranked))
    gathered = ranked["gathered_total"].to_numpy()
    metabolic = ranked["metabolic_cost_total"].to_numpy()
    reproduction = ranked["reproduction_cost_total"].to_numpy()
    net = ranked["net_resource"].to_numpy()

    ax.barh(y_pos, gathered, color=COLOR_GATHER, alpha=0.82, label="Gathered in")
    ax.barh(y_pos, -metabolic, color=COLOR_METABOLISM, alpha=0.82, label="Metabolic out")
    ax.barh(y_pos, -reproduction, left=-metabolic, color=COLOR_REPRODUCTION, alpha=0.82, label="Reproduction out")
    net_colors = np.where(ranked["starved"].to_numpy(), COLOR_STARVED, COLOR_SURVIVED)
    ax.scatter(net, y_pos, color=net_colors, s=20, zorder=4, label="Net")

    labels = []
    for aid, gen, starved in zip(ranked["agent_id"], ranked["generation"], ranked["starved"]):
        star_mark = " *" if bool(starved) else ""
        labels.append(f"{aid[-6:]} (g{gen}){star_mark}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.8, color=TEXT_MUTED)
    ax.set_xlabel("Resource units (positive=inflow, negative=cost)")
    ax.set_title("B) Per-agent resource budget (selection bookkeeping)", loc="left", fontsize=12, fontweight="bold")
    ax.axvline(0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.9)
    ax.grid(True, axis="x", linestyle=":", color=GRID_COLOR, alpha=0.9)
    legend_handles = [
        Line2D([0], [0], color=COLOR_GATHER, lw=6, alpha=0.82, label="Gathered in"),
        Line2D([0], [0], color=COLOR_METABOLISM, lw=6, alpha=0.82, label="Metabolic out"),
        Line2D([0], [0], color=COLOR_REPRODUCTION, lw=6, alpha=0.82, label="Reproduction out"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_SURVIVED, label="Net (survived)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STARVED, label="Net (starved)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8, ncol=2)
    ax.text(
        0.003,
        1.01,
        "* starved",
        transform=ax.transAxes,
        fontsize=8,
        color=COLOR_STARVED,
        ha="left",
        va="bottom",
    )
    _style_axis(ax)


def _plot_phase(ax: plt.Axes, budget: pd.DataFrame, top_lineages: int = 6) -> None:
    if budget.empty:
        ax.text(0.5, 0.5, "No phase-plot data available", ha="center", va="center", color=TEXT_MUTED)
        ax.set_axis_off()
        return

    lineage_counts = budget["genome_id"].value_counts()
    featured = set(lineage_counts.head(top_lineages).index.tolist())
    budget = budget.copy()
    budget["lineage_label"] = budget["genome_id"].apply(lambda gid: gid if gid in featured else "other")

    unique_labels = list(dict.fromkeys(budget["lineage_label"]))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    lineage_colors = {}
    for i, label in enumerate(unique_labels):
        lineage_colors[label] = COLOR_OTHER_LINEAGE if label == "other" else cmap(i)

    status_split = [
        ("survived_to_end", "Survived to end", "o", 46, False),
        ("starved", "Starved", "X", 58, True),
    ]
    rendered_any = False

    for status_col, status_label, marker, size, edge in status_split:
        subset = budget[budget[status_col]]
        if subset.empty:
            continue
        for lineage_label, chunk in subset.groupby("lineage_label", sort=False):
            rendered_any = True
            ax.scatter(
                chunk["net_per_step"],
                chunk["reproduction_events"],
                s=size,
                marker=marker,
                alpha=0.83,
                c=[lineage_colors[lineage_label]],
                edgecolors="black" if edge else "white",
                linewidths=0.45,
            )

    if not rendered_any:
        ax.text(0.5, 0.5, "No phase-plot data available", ha="center", va="center", color=TEXT_MUTED)
        ax.set_axis_off()
        return

    x_min = float(np.nanmin(budget["net_per_step"]))
    x_max = float(np.nanmax(budget["net_per_step"]))
    x_pad = max(0.1, (x_max - x_min) * 0.07)
    left_limit = x_min - x_pad
    right_limit = x_max + x_pad
    ax.axvspan(left_limit, 0, color=COLOR_STARVED, alpha=0.05, zorder=0)
    ax.axvspan(0, right_limit, color=COLOR_SURVIVED, alpha=0.05, zorder=0)
    ax.axvline(0, color="#333333", linewidth=1.15, linestyle="--", alpha=0.95)
    ax.set_xlim(left_limit, right_limit)
    ax.set_xlabel("Net resource per step")
    ax.set_ylabel("Reproduction events (lifetime)")
    ax.set_title("C) Ecological selection boundary (x=0)", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, axis="both", linestyle=":", color=GRID_COLOR, alpha=0.9)

    fate_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=7, label="Survived"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#666666", markeredgecolor="black", markersize=7, label="Starved"),
    ]
    lineage_handles = []
    lineage_order = [label for label in unique_labels if label != "other"]
    lineage_order = lineage_order[:top_lineages]
    lineage_counts_map = budget["lineage_label"].value_counts().to_dict()
    lineage_alias = {
        label: f"L{i + 1} (n={int(lineage_counts_map.get(label, 0))})"
        for i, label in enumerate(lineage_order)
    }
    for label in lineage_order:
        lineage_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=lineage_colors[label],
                markeredgecolor="white",
                markersize=6.5,
                label=lineage_alias[label],
            )
        )
    if "other" in lineage_colors:
        lineage_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLOR_OTHER_LINEAGE,
                markeredgecolor="white",
                markersize=6.5,
                label=f"Other (n={int(lineage_counts_map.get('other', 0))})",
            )
        )

    legend_fate = ax.legend(
        handles=fate_handles,
        loc="upper left",
        frameon=False,
        fontsize=8.0,
        title="Marker = fate",
        title_fontsize=8.0,
    )
    ax.add_artist(legend_fate)
    ax.legend(
        handles=lineage_handles,
        loc="upper right",
        frameon=False,
        fontsize=7.8,
        title="Color = lineage",
        title_fontsize=8.0,
    )
    if lineage_alias:
        mapping_text = ", ".join([f"{alias}={lineage_id[:8]}" for lineage_id, alias in lineage_alias.items()])
        ax.text(
            0.01,
            -0.20,
            f"Lineage key: {mapping_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=TEXT_MUTED,
        )
    _style_axis(ax)


def _render_figure(
    *,
    steps: pd.DataFrame,
    budget: pd.DataFrame,
    eco_cfg: EcologyConfig,
    output_path: Path,
    top_agents: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.8, 13.8),
        gridspec_kw={"height_ratios": [1.05, 1.22, 1.08]},
    )
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    _plot_ecology_timeseries(axes[0], steps)
    _plot_budget_breakdown(axes[1], budget, top_agents=top_agents)
    _plot_phase(axes[2], budget)

    subtitle = (
        f"Base consumption={eco_cfg.base_consumption_rate:.3f} | "
        f"Offspring cost={eco_cfg.offspring_cost:.3f} | "
        f"Starvation threshold={eco_cfg.starvation_threshold:.0f}"
    )
    fig.suptitle(
        "Selection Through Ecology, Not an Explicit Fitness Function",
        fontsize=16,
        fontweight="bold",
        color=TEXT_DARK,
        y=0.988,
    )
    fig.text(0.5, 0.963, subtitle, ha="center", va="center", fontsize=10, color=TEXT_MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.948))
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_selection_ecology_figure(
    db_path: Path,
    output_path: Path,
    *,
    top_agents: int,
    min_lifetime_steps: int,
    dpi: int,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DB path does not exist: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        eco_cfg = _load_simulation_config(conn)
        steps = _load_step_series(conn)
        agents = _load_agent_table(conn)
        starved_agents = _load_starved_agents(conn)
        action_budget = _load_action_budget(conn)

    if steps.empty:
        raise RuntimeError("No rows found in simulation_steps; cannot render figure.")
    if agents.empty:
        raise RuntimeError("No rows found in agents; cannot render figure.")

    final_step = int(steps["step_number"].max())
    budget = _build_agent_budget_frame(
        agents,
        action_budget,
        starved_agents,
        final_step=final_step,
        base_consumption_rate=eco_cfg.base_consumption_rate,
    )
    budget = budget[budget["alive_steps"] >= float(min_lifetime_steps)].copy()
    budget = budget.replace([np.inf, -np.inf], np.nan).dropna(subset=["net_per_step"])

    _render_figure(
        steps=steps,
        budget=budget,
        eco_cfg=eco_cfg,
        output_path=output_path,
        top_agents=top_agents,
        dpi=dpi,
    )


def main() -> int:
    args = _parse_args()
    make_selection_ecology_figure(
        db_path=args.db_path,
        output_path=args.output,
        top_agents=args.top_agents,
        min_lifetime_steps=args.min_lifetime_steps,
        dpi=args.dpi,
    )
    print(f"Wrote figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
