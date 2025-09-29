import os
import tempfile
import time
import sqlite3
import json as pyjson
import resource
from contextlib import closing
from typing import Tuple, List, Dict, Any

import pandas as pd

from farm.analysis.data.loaders import SQLiteLoader
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy import Column, Integer, String, Text


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
        # Index to accelerate WHERE simulation_id = ?
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sim_steps_sim_id ON simulation_steps(simulation_id)")

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


def current_peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_full_load(db_path: str, simulation_id: str) -> Dict[str, Any]:
    query = (
        "SELECT step_number, simulation_id, agent_counts, resource_counts, timestamp "
        "FROM simulation_steps WHERE simulation_id = ?"
    )
    before_mem = current_peak_rss_mb()
    t0 = time.perf_counter()
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(simulation_id,))
    _ = len(df)
    t1 = time.perf_counter()
    after_mem = current_peak_rss_mb()
    return {"seconds": t1 - t0, "peak_rss_mb": max(before_mem, after_mem)}


def measure_streaming(db_path: str, simulation_id: str, chunk_size: int = 10000, select_step_only: bool = False) -> Dict[str, Any]:
    loader = SQLiteLoader(db_path=db_path)
    if select_step_only:
        query = "SELECT step_number FROM simulation_steps WHERE simulation_id = :sim_id"
    else:
        query = (
            "SELECT step_number, simulation_id, agent_counts, resource_counts, timestamp "
            "FROM simulation_steps WHERE simulation_id = :sim_id"
        )
    before_mem = current_peak_rss_mb()
    t0 = time.perf_counter()
    total = 0
    for chunk in loader.execute_query_iter(query, params={"sim_id": simulation_id}, chunk_size=chunk_size):
        total += len(chunk)
    t1 = time.perf_counter()
    assert total > 0
    after_mem = current_peak_rss_mb()
    return {"seconds": t1 - t0, "peak_rss_mb": max(before_mem, after_mem)}


def measure_orm_streaming(db_path: str, simulation_id: str, chunk_size: int = 10000, select_step_only: bool = False) -> Dict[str, Any]:
    """Benchmark ORM-style streaming using a minimal mapped class for this benchmark schema."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base = declarative_base()

    class BenchStep(Base):  # type: ignore
        __tablename__ = "simulation_steps"
        # Define composite PK to satisfy SQLAlchemy mapper
        step_number = Column(Integer, primary_key=True)
        simulation_id = Column(String(64), primary_key=True)
        agent_counts = Column(Text)
        resource_counts = Column(Text)
        timestamp = Column(Text)

    total = 0
    before_mem = current_peak_rss_mb()
    t0 = time.perf_counter()
    with Session(engine) as session:
        q = session.query(BenchStep).filter(BenchStep.simulation_id == simulation_id)
        for row in q.yield_per(chunk_size):
            if select_step_only:
                _ = row.step_number
            else:
                _ = row.agent_counts
            total += 1
    t1 = time.perf_counter()
    assert total > 0
    after_mem = current_peak_rss_mb()
    return {"seconds": t1 - t0, "peak_rss_mb": max(before_mem, after_mem)}


def main():
    db_path, sim_id = create_temp_db()
    # Full-load baseline
    full = measure_full_load(db_path, sim_id)

    # Streaming across chunk sizes and projections
    chunk_sizes: List[int] = [2000, 10000, 50000, 100000]
    raw_all: Dict[int, Dict[str, Any]] = {}
    raw_step_only: Dict[int, Dict[str, Any]] = {}
    orm_all: Dict[int, Dict[str, Any]] = {}

    for cs in chunk_sizes:
        raw_all[cs] = measure_streaming(db_path, sim_id, chunk_size=cs, select_step_only=False)
        raw_step_only[cs] = measure_streaming(db_path, sim_id, chunk_size=cs, select_step_only=True)
        orm_all[cs] = measure_orm_streaming(db_path, sim_id, chunk_size=cs, select_step_only=False)

    def best_of(results: Dict[int, Dict[str, Any]]):
        best_cs = min(results, key=lambda k: results[k]["seconds"])  # type: ignore
        return best_cs, results[best_cs]

    best_raw_all_cs, best_raw_all = best_of(raw_all)
    best_raw_step_cs, best_raw_step = best_of(raw_step_only)
    best_orm_all_cs, best_orm_all = best_of(orm_all)

    output = {
        "num_steps": 200000,
        "full_load": {"seconds": round(full["seconds"], 3), "peak_rss_mb": round(full["peak_rss_mb"], 1)},
        "best_raw_stream_all": {"chunk_size": best_raw_all_cs, "seconds": round(best_raw_all["seconds"], 3), "peak_rss_mb": round(best_raw_all["peak_rss_mb"], 1)},
        "best_raw_stream_step_only": {"chunk_size": best_raw_step_cs, "seconds": round(best_raw_step["seconds"], 3), "peak_rss_mb": round(best_raw_step["peak_rss_mb"], 1)},
        "best_orm_stream_all": {"chunk_size": best_orm_all_cs, "seconds": round(best_orm_all["seconds"], 3), "peak_rss_mb": round(best_orm_all["peak_rss_mb"], 1)},
    }
    # Percent deltas vs full load
    for k in ["best_raw_stream_all", "best_raw_stream_step_only", "best_orm_stream_all"]:
        sec = output[k]["seconds"]
        output[k]["vs_full_pct"] = round((sec / output["full_load"]["seconds"] - 1) * 100, 2)

    print(pyjson.dumps(output))


if __name__ == "__main__":
    main()

