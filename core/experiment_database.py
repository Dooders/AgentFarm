import sqlite3
import threading
import json
import pandas as pd
from datetime import datetime


class ExperimentDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._thread_local = threading.local()
        self._setup_connection()
        self._create_tables()

    def _setup_connection(self):
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._thread_local.conn.execute("PRAGMA foreign_keys = ON")
            self._thread_local.cursor = self._thread_local.conn.cursor()

    @property
    def conn(self):
        self._setup_connection()
        return self._thread_local.conn

    @property
    def cursor(self):
        self._setup_connection()
        return self._thread_local.cursor

    def _create_tables(self):
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS Simulations (
                simulation_id INTEGER PRIMARY KEY,
                start_time INTEGER,
                end_time INTEGER,
                status TEXT,
                parameters TEXT,
                results_summary TEXT,
                simulation_db_path TEXT
            );
            """
        )
        self.conn.commit()

    def add_simulation(self, parameters: dict, simulation_db_path: str) -> int:
        start_time = int(datetime.utcnow().timestamp())
        self.cursor.execute(
            """
            INSERT INTO Simulations (start_time, status, parameters, simulation_db_path)
            VALUES (?, ?, ?, ?)
            """,
            (start_time, "pending", json.dumps(parameters), simulation_db_path),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def update_simulation_status(self, simulation_id: int, status: str, results_summary: dict = None):
        end_time = int(datetime.utcnow().timestamp()) if status in ["completed", "failed"] else None
        self.cursor.execute(
            """
            UPDATE Simulations
            SET status = ?, end_time = ?, results_summary = ?
            WHERE simulation_id = ?
            """,
            (status, end_time, json.dumps(results_summary) if results_summary else None, simulation_id),
        )
        self.conn.commit()

    def get_simulation(self, simulation_id: int) -> dict:
        self.cursor.execute(
            """
            SELECT * FROM Simulations WHERE simulation_id = ?
            """,
            (simulation_id,),
        )
        row = self.cursor.fetchone()
        if row:
            return {
                "simulation_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "status": row[3],
                "parameters": json.loads(row[4]),
                "results_summary": json.loads(row[5]) if row[5] else None,
                "simulation_db_path": row[6],
            }
        return None

    def list_simulations(self, status: str = None) -> list:
        query = "SELECT * FROM Simulations"
        params = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [
            {
                "simulation_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "status": row[3],
                "parameters": json.loads(row[4]),
                "results_summary": json.loads(row[5]) if row[5] else None,
                "simulation_db_path": row[6],
            }
            for row in rows
        ]

    def delete_simulation(self, simulation_id: int):
        self.cursor.execute(
            """
            DELETE FROM Simulations WHERE simulation_id = ?
            """,
            (simulation_id,),
        )
        self.conn.commit()

    def export_experiment_data(self, filepath: str):
        self.cursor.execute("SELECT * FROM Simulations")
        rows = self.cursor.fetchall()
        data = [
            {
                "simulation_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "status": row[3],
                "parameters": json.loads(row[4]),
                "results_summary": json.loads(row[5]) if row[5] else None,
                "simulation_db_path": row[6],
            }
            for row in rows
        ]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def get_aggregate_results(self) -> dict:
        self.cursor.execute("SELECT results_summary FROM Simulations WHERE status = 'completed'")
        rows = self.cursor.fetchall()
        aggregate_results = {}
        for row in rows:
            if row[0]:
                results_summary = json.loads(row[0])
                for key, value in results_summary.items():
                    if isinstance(value, (int, float)):
                        aggregate_results[key] = aggregate_results.get(key, 0) + value
        num_simulations = len(rows)
        for key in aggregate_results:
            aggregate_results[key] /= num_simulations
        return aggregate_results
