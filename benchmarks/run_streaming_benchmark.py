import os
import tempfile
import time
import sqlite3
from contextlib import closing
from typing import Tuple

import pandas as pd

from farm.analysis.data.loaders import SQLiteLoader


def create_temp_db(num_steps: int = 200000) -> Tuple[str, str]:
    """Create a temporary SQLite DB with a minimal schema and populate steps.

    Returns (db_path, simulation_id)
    """
    tmp_dir = tempfile.mkdtemp(prefix="agentfarm_bench_")
    db_path = os.path.join(tmp_dir, "sim.db")
    sim_id = "sim_bench"
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        # simulations
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                parameters TEXT,
                results_summary TEXT,
                simulation_db_path TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT OR REPLACE INTO simulations (simulation_id, experiment_id)
            VALUES (?, ?)
            """,
            (sim_id, "bench"),
        )
        # simulation_steps with JSON strings for counts compatible with loader expectations
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_steps (
                step_number INTEGER,
                simulation_id TEXT,
                agent_counts TEXT,
                resource_counts TEXT,
                timestamp TEXT,
                PRIMARY KEY (step_number, simulation_id)
            )
            """
        )
        # Populate rows
        batch = []
        for i in range(num_steps):
            agent_counts = '{"system": %d, "independent": %d, "control": %d}' % (
                (i % 10) + 1,
                (i % 7) + 1,
                (i % 5) + 1,
            )
            resource_counts = '{"food": %d}' % ((i % 11) + 1)
            batch.append((i, sim_id, agent_counts, resource_counts, ""))
            if len(batch) >= 1000:
                cur.executemany(
                    "INSERT OR REPLACE INTO simulation_steps(step_number, simulation_id, agent_counts, resource_counts, timestamp) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                batch = []
        if batch:
            cur.executemany(
                "INSERT OR REPLACE INTO simulation_steps(step_number, simulation_id, agent_counts, resource_counts, timestamp) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
        conn.commit()
    return db_path, sim_id


def measure_full_load(db_path: str, simulation_id: str) -> float:
    query = (
        "SELECT step_number, simulation_id, agent_counts, resource_counts, timestamp "
        "FROM simulation_steps WHERE simulation_id = ?"
    )
    t0 = time.perf_counter()
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(simulation_id,))
    _ = len(df)
    t1 = time.perf_counter()
    return t1 - t0


def measure_streaming(db_path: str, simulation_id: str, chunk_size: int = 10000) -> float:
    loader = SQLiteLoader(db_path=db_path)
    query = (
        "SELECT step_number, simulation_id, agent_counts, resource_counts, timestamp "
        "FROM simulation_steps WHERE simulation_id = :sim_id"
    )
    t0 = time.perf_counter()
    total = 0
    for chunk in loader.execute_query_iter(query, params={"sim_id": simulation_id}, chunk_size=chunk_size):
        total += len(chunk)
    t1 = time.perf_counter()
    assert total > 0
    return t1 - t0


def main():
    db_path, sim_id = create_temp_db()
    full_time = measure_full_load(db_path, sim_id)
    stream_time = measure_streaming(db_path, sim_id)
    print({
        "num_steps": 200000,
        "full_load_seconds": round(full_time, 3),
        "stream_seconds": round(stream_time, 3),
        "stream_vs_full_pct": round((stream_time / full_time - 1) * 100, 2) if full_time > 0 else None,
    })


if __name__ == "__main__":
    main()

