"""Short run_simulation smoke checks against persisted SQLite state."""

import json
import sqlite3

import pytest

from farm.config import SimulationConfig
from farm.config.config import (
    DatabaseConfig,
    EnvironmentConfig,
    PopulationConfig,
    ResourceConfig,
)
from farm.core.simulation import run_simulation
from farm.utils.identity import Identity


@pytest.mark.unit
@pytest.mark.db
def test_run_simulation_persisted_db_invariants(tmp_path):
    """Run a few steps into tmp_path and assert DB rows stay internally consistent."""
    simulation_id = str(Identity().simulation_id(prefix="inv"))
    num_steps = 5
    seed = 12345

    config = SimulationConfig(
        environment=EnvironmentConfig(width=50, height=50),
        resources=ResourceConfig(initial_resources=50),
        population=PopulationConfig(
            system_agents=2,
            independent_agents=2,
            control_agents=0,
        ),
        database=DatabaseConfig(use_in_memory_db=False),
    )
    config.seed = seed

    out_dir = tmp_path / "sim_out"
    out_dir.mkdir()
    path_str = str(out_dir)

    env = run_simulation(
        num_steps=num_steps,
        config=config,
        path=path_str,
        simulation_id=simulation_id,
        seed=seed,
    )

    db_file = out_dir / f"simulation_{simulation_id}.db"
    assert db_file.is_file(), f"expected database at {db_file}"

    con = sqlite3.connect(str(db_file))
    try:
        status, end_time, _rs = con.execute(
            "SELECT status, end_time, results_summary FROM simulations WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()
        assert status == "completed"
        assert end_time is not None

        cnt, lo, hi = con.execute(
            "SELECT COUNT(*), MIN(step_number), MAX(step_number) FROM simulation_steps WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()
        # One metrics row per environment.time at update(): 0..num_steps after loop + final update.
        assert cnt == num_steps + 1
        assert lo == 0
        assert hi == num_steps

        last_pop = con.execute(
            "SELECT total_agents FROM simulation_steps WHERE simulation_id = ? ORDER BY step_number DESC LIMIT 1",
            (simulation_id,),
        ).fetchone()[0]
        agent_rows = con.execute(
            "SELECT COUNT(*) FROM agents WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()[0]
        assert last_pop == agent_rows

        first_pop = con.execute(
            "SELECT total_agents FROM simulation_steps WHERE simulation_id = ? ORDER BY step_number ASC LIMIT 1",
            (simulation_id,),
        ).fetchone()[0]
        births, deaths = con.execute(
            "SELECT COALESCE(SUM(births), 0), COALESCE(SUM(deaths), 0) FROM simulation_steps WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()
        assert first_pop + births - deaths == last_pop

        cfg_raw = con.execute(
            "SELECT config_data FROM simulation_config WHERE simulation_id = ? ORDER BY timestamp DESC LIMIT 1",
            (simulation_id,),
        ).fetchone()[0]
        cfg = json.loads(cfg_raw)
        assert cfg.get("seed") == seed
    finally:
        con.close()

    env.cleanup()
