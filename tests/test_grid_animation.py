"""Unit tests for the teaching-oriented grid animation helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from farm.core.animation import (
    AGENT_DISPLAY,
    agent_style,
    cell_offsets,
    create_grid_frame,
    get_state_at_step,
    list_step_numbers,
    normalize_agent_type,
    to_cell,
    world_size_from_config,
    write_grid_gif,
    write_grid_gif_from_states,
)

pytestmark = pytest.mark.unit


def test_world_size_reads_nested_and_flat_config():
    assert world_size_from_config({"environment": {"width": 16, "height": 12}}) == (16, 12)
    assert world_size_from_config({"width": 8, "height": 9}) == (8, 9)
    assert world_size_from_config({}) == (100, 100)


def test_to_cell_floors_and_clamps():
    assert to_cell(3.9, 0.1, 8, 8) == (3, 0)
    assert to_cell(-2.0, 99.0, 8, 8) == (0, 7)


def test_normalize_agent_type_aliases():
    assert normalize_agent_type("system") == "SystemAgent"
    assert normalize_agent_type("IndependentAgent") == "IndependentAgent"
    assert normalize_agent_type("control") == "ControlAgent"
    assert agent_style("system")["label"] == AGENT_DISPLAY["SystemAgent"]["label"]
    assert agent_style("system")["color"] == AGENT_DISPLAY["SystemAgent"]["color"]


def test_create_grid_frame_accepts_database_type_names():
    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "agent_type": "system",
                "position_x": 1.2,
                "position_y": 2.8,
                "resources": 5.0,
            }
        ]
    )
    resources = pd.DataFrame(
        [{"resource_id": "r1", "position_x": 4.0, "position_y": 4.0, "amount": 8.0}]
    )
    fig = create_grid_frame(agents, resources, step_number=1, width=8, height=8)
    try:
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert "Food" in labels
        assert AGENT_DISPLAY["SystemAgent"]["label"] in labels
    finally:
        plt.close(fig)


def test_cell_offsets_spread_occupants():
    assert cell_offsets(1) == [(0.5, 0.5)]
    offsets = cell_offsets(3)
    assert len(offsets) == 3
    assert len({offsets[0], offsets[1], offsets[2]}) == 3


def test_create_grid_frame_draws_known_agent_types():
    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "agent_type": "SystemAgent",
                "position_x": 1.2,
                "position_y": 2.8,
                "resources": 5.0,
            },
            {
                "agent_id": "a2",
                "agent_type": "IndependentAgent",
                "position_x": 1.4,
                "position_y": 2.1,
                "resources": 12.0,
            },
        ]
    )
    resources = pd.DataFrame(
        [{"resource_id": "r1", "position_x": 4.0, "position_y": 4.0, "amount": 8.0}]
    )

    fig = create_grid_frame(agents, resources, step_number=7, width=8, height=8)
    try:
        assert fig.axes
        legend = fig.axes[0].get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert "Food" in labels
        assert AGENT_DISPLAY["SystemAgent"]["label"] in labels
        assert AGENT_DISPLAY["IndependentAgent"]["label"] in labels
    finally:
        plt.close(fig)


def test_write_grid_gif_from_states(tmp_path: Path):
    agents = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "agent_type": "ControlAgent",
                "position_x": 0.5,
                "position_y": 0.5,
                "resources": 3.0,
            }
        ]
    )
    resources = pd.DataFrame(
        [{"resource_id": "r1", "position_x": 1.0, "position_y": 1.0, "amount": 4.0}]
    )
    output = tmp_path / "tiny.gif"
    write_grid_gif_from_states(
        [(0, agents, resources), (1, agents, resources)],
        output,
        width=4,
        height=4,
        fps=4,
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def _seed_tiny_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE agents (agent_id TEXT, agent_type TEXT);
        CREATE TABLE agent_states (
            agent_id TEXT,
            position_x REAL,
            position_y REAL,
            resource_level REAL,
            step_number INTEGER
        );
        CREATE TABLE resource_states (
            resource_id TEXT,
            position_x REAL,
            position_y REAL,
            amount REAL,
            step_number INTEGER
        );
        INSERT INTO agents VALUES ('a1', 'SystemAgent');
        INSERT INTO agent_states VALUES ('a1', 2.2, 1.1, 4.0, 0);
        INSERT INTO agent_states VALUES ('a1', 3.2, 1.4, 5.0, 1);
        INSERT INTO resource_states VALUES ('r1', 2.0, 2.0, 6.0, 0);
        INSERT INTO resource_states VALUES ('r1', 2.0, 2.0, 3.0, 1);
        """
    )
    conn.commit()
    conn.close()


def test_sqlite_helpers_and_write_grid_gif(tmp_path: Path):
    db_path = tmp_path / "sim.db"
    _seed_tiny_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        assert list_step_numbers(conn) == [0, 1]
        agents, resources = get_state_at_step(conn, 1)
        assert list(agents["agent_id"]) == ["a1"]
        assert float(resources.iloc[0]["amount"]) == 3.0
    finally:
        conn.close()

    output = tmp_path / "from-db.gif"
    write_grid_gif(db_path, output, width=6, height=6, fps=5)
    assert output.is_file()
    assert output.stat().st_size > 0
