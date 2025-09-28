"""
SQLite Pragma Profile Benchmark implementation.

This benchmark tests the performance of different SQLite pragma profiles
in the AgentFarm database module under various workloads.
"""

import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, Optional

from benchmarks.base.benchmark import Benchmark
from farm.core.config_hydra_bridge import HydraHydraSimulationConfig
from farm.core.simulation import run_simulation
from farm.database import InMemorySimulationDatabase, SimulationDatabase
from farm.database.pragma_docs import PRAGMA_PROFILES, get_pragma_profile
from farm.utils.identity import Identity

# Shared Identity instance for efficiency
_shared_identity = Identity()


class PragmaProfileBenchmark(Benchmark):
    """
    Benchmark for comparing different SQLite pragma profiles.

    This benchmark tests the performance of different SQLite pragma profiles
    under various workloads:
    1. Write-heavy workload (many inserts)
    2. Read-heavy workload (many queries)
    3. Mixed workload (balanced reads and writes)

    It compares the performance of the following profiles:
    - balanced: Good balance of performance and data safety
    - performance: Maximum performance, reduced data safety
    - safety: Maximum data safety, reduced performance
    - memory: Optimized for low memory usage
    """

    def __init__(
        self,
        num_records: int = 100000,
        db_size_mb: int = 100,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pragma profile benchmark.

        Parameters
        ----------
        num_records : int
            Number of records to insert for write tests
        db_size_mb : int
            Target database size in MB for read tests
        parameters : Dict[str, Any], optional
            Additional parameters for the benchmark
        """
        super().__init__(
            name="pragma_profile",
            description="Benchmark comparing different SQLite pragma profiles",
            parameters=parameters or {},
        )

        # Set benchmark-specific parameters
        self.parameters.update(
            {
                "num_records": num_records,
                "db_size_mb": db_size_mb,
            }
        )

        # Initialize benchmark-specific attributes
        self.temp_dir = None
        self.profiles = ["balanced", "performance", "safety", "memory"]
        self.workloads = ["write_heavy", "read_heavy", "mixed"]

        # Track open database connections and files for proper cleanup
        self.open_dbs = []
        self.open_connections = []

        # Generate a unique run ID for this benchmark run
        self.run_id = _shared_identity.run_id(8)

    def setup(self) -> None:
        """
        Set up the benchmark environment.
        """
        # Create temporary directory for benchmark results
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")

        # Clear any previous state
        self.open_dbs = []
        self.open_connections = []

    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.

        Returns
        -------
        Dict[str, Any]
            Raw results from the benchmark run
        """
        results = {}

        # Test each profile with each workload
        for profile in self.profiles:
            profile_results = {}

            for workload in self.workloads:
                print(f"Testing {profile} profile with {workload} workload...")

                # Run the appropriate test for this workload
                if workload == "write_heavy":
                    duration = self._test_write_performance(profile)
                elif workload == "read_heavy":
                    duration = self._test_read_performance(profile)
                else:  # mixed
                    duration = self._test_mixed_performance(profile)

                profile_results[workload] = duration
                print(
                    f"  {profile} profile, {workload} workload: {duration:.2f} seconds"
                )

            results[profile] = profile_results

        # Calculate relative performance (normalized to balanced profile)
        normalized_results = {}

        for workload in self.workloads:
            normalized_results[workload] = {}
            baseline = results["balanced"][workload]

            for profile in self.profiles:
                # Calculate speedup relative to balanced profile
                speedup = (
                    baseline / results[profile][workload]
                    if results[profile][workload] > 0
                    else float("inf")
                )
                normalized_results[workload][profile] = speedup

                # Print speedup (>1 means faster, <1 means slower)
                print(
                    f"  {profile} profile is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than balanced for {workload} workload"
                )

        # Close all open database connections before returning
        self._close_all_connections()

        # Combine raw and normalized results
        return {"raw_results": results, "normalized_results": normalized_results}

    def _test_write_performance(self, profile: str) -> float:
        """
        Test write performance with the specified profile.

        Parameters
        ----------
        profile : str
            Pragma profile to test

        Returns
        -------
        float
            Duration in seconds
        """
        # Create a database file for this test with a unique name
        unique_suffix = f"{self.run_id}_{int(time.time())}"
        if self.temp_dir is None:
            raise RuntimeError("temp_dir not initialized")
        db_path = os.path.join(
            self.temp_dir, f"{profile}_write_test_{unique_suffix}.db"
        )

        # Create a config with the specified profile
        config = HydraSimulationConfig()
        config.db_pragma_profile = profile

        # Create the database
        db = SimulationDatabase(db_path, config)
        self.open_dbs.append(db)

        # Get the number of records to insert
        num_records = self.parameters["num_records"]

        # Start timing
        start_time = time.time()

        # Perform write-heavy operations
        conn = db.engine.raw_connection()
        self.open_connections.append(conn)
        cursor = conn.cursor()

        # Create a test table with a unique name
        table_name = f"write_test_{unique_suffix}"
        cursor.execute(
            f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, data TEXT, value REAL, timestamp INTEGER)"
        )

        # Insert records in batches
        batch_size = 1000
        for i in range(0, num_records, batch_size):
            # Create a batch of records
            records = []
            for j in range(batch_size):
                if i + j < num_records:
                    record_id = i + j
                    data = f"data_{record_id}"
                    value = record_id / 1000.0
                    timestamp = int(time.time())
                    records.append((record_id, data, value, timestamp))

            # Insert the batch
            cursor.executemany(
                f"INSERT INTO {table_name} (id, data, value, timestamp) VALUES (?, ?, ?, ?)",
                records,
            )

            # Commit every batch
            conn.commit()

        # End timing
        end_time = time.time()

        # Clean up
        cursor.close()
        conn.close()
        self.open_connections.remove(conn)

        db.close()
        self.open_dbs.remove(db)

        return end_time - start_time

    def _test_read_performance(self, profile: str) -> float:
        """
        Test read performance with the specified profile.

        Parameters
        ----------
        profile : str
            Pragma profile to test

        Returns
        -------
        float
            Duration in seconds
        """
        # Create a database file for this test with a unique name
        unique_suffix = f"{self.run_id}_{int(time.time())}"
        if self.temp_dir is None:
            raise RuntimeError("temp_dir not initialized")
        db_path = os.path.join(self.temp_dir, f"{profile}_read_test_{unique_suffix}.db")

        # Create a config with the specified profile
        config = HydraSimulationConfig()
        config.db_pragma_profile = profile

        # Create and populate the database
        table_name = f"read_test_{unique_suffix}"
        self._prepare_read_test_database(db_path, profile, table_name)

        # Open the database with the specified profile
        db = SimulationDatabase(db_path, config)
        self.open_dbs.append(db)

        # Start timing
        start_time = time.time()

        # Perform read-heavy operations
        conn = db.engine.raw_connection()
        self.open_connections.append(conn)
        cursor = conn.cursor()

        # Perform various read operations

        # 1. Full table scan
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = cursor.fetchone()
        total_count = result[0] if result else 0

        # 2. Filtered queries
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE value > 50.0")
        result = cursor.fetchone()
        filtered_count = result[0] if result else 0

        # 3. Aggregation
        cursor.execute(f"SELECT AVG(value), MIN(value), MAX(value) FROM {table_name}")
        result = cursor.fetchone()
        avg, min_val, max_val = result if result else (0, 0, 0)

        # 4. Joins (if applicable)
        cursor.execute(
            f"""
            SELECT t1.id, t1.data, t2.value 
            FROM {table_name} t1
            JOIN {table_name} t2 ON t1.id = t2.id
            LIMIT 1000
        """
        )
        join_results = cursor.fetchall()

        # 5. Range queries
        cursor.execute(f"SELECT * FROM {table_name} WHERE id BETWEEN 1000 AND 2000")
        range_results = cursor.fetchall()

        # End timing
        end_time = time.time()

        # Clean up
        cursor.close()
        conn.close()
        self.open_connections.remove(conn)

        db.close()
        self.open_dbs.remove(db)

        return end_time - start_time

    def _prepare_read_test_database(
        self, db_path: str, profile: str, table_name: str
    ) -> None:
        """
        Prepare a database for read testing.

        Parameters
        ----------
        db_path : str
            Path to the database file
        profile : str
            Pragma profile to use
        table_name : str
            Name of the table to create
        """
        # Create a config with the specified profile
        config = HydraSimulationConfig()
        config.db_pragma_profile = profile

        # Create the database
        db = SimulationDatabase(db_path, config)
        self.open_dbs.append(db)

        # Get the target database size
        db_size_mb = self.parameters["db_size_mb"]

        # Create a connection
        conn = db.engine.raw_connection()
        self.open_connections.append(conn)
        cursor = conn.cursor()

        # Create a test table
        cursor.execute(
            f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, data TEXT, value REAL, timestamp INTEGER)"
        )

        # Insert records until we reach the target size
        batch_size = 1000
        record_id = 0
        current_size_mb = 0

        while current_size_mb < db_size_mb:
            # Create a batch of records
            records = []
            for j in range(batch_size):
                data = (
                    f"data_{record_id}" * 100
                )  # Make the data larger to reach target size faster
                value = record_id / 1000.0
                timestamp = int(time.time())
                records.append((record_id, data, value, timestamp))
                record_id += 1

            # Insert the batch
            cursor.executemany(
                f"INSERT INTO {table_name} (id, data, value, timestamp) VALUES (?, ?, ?, ?)",
                records,
            )

            # Commit every batch
            conn.commit()

            # Check current database size
            cursor.execute("PRAGMA page_count")
            result = cursor.fetchone()
            page_count = result[0] if result else 0
            cursor.execute("PRAGMA page_size")
            result = cursor.fetchone()
            page_size = result[0] if result else 0
            current_size_mb = (page_count * page_size) / (1024 * 1024)

            if record_id % 10000 == 0:
                print(
                    f"  Prepared {record_id} records, current size: {current_size_mb:.2f} MB"
                )

        # Create an index to support read operations
        cursor.execute(f"CREATE INDEX idx_{table_name}_value ON {table_name} (value)")
        conn.commit()

        # Clean up
        cursor.close()
        conn.close()
        self.open_connections.remove(conn)

        db.close()
        self.open_dbs.remove(db)

    def _test_mixed_performance(self, profile: str) -> float:
        """
        Test mixed read/write performance with the specified profile.

        Parameters
        ----------
        profile : str
            Pragma profile to test

        Returns
        -------
        float
            Duration in seconds
        """
        # Create a database file for this test with a unique name
        unique_suffix = f"{self.run_id}_{int(time.time())}"
        if self.temp_dir is None:
            raise RuntimeError("temp_dir not initialized")
        db_path = os.path.join(
            self.temp_dir, f"{profile}_mixed_test_{unique_suffix}.db"
        )

        # Create a config with the specified profile
        config = HydraSimulationConfig()
        config.db_pragma_profile = profile

        # Create the database
        db = SimulationDatabase(db_path, config)
        self.open_dbs.append(db)

        # Get the number of operations to perform
        num_operations = (
            self.parameters["num_records"] // 10
        )  # Fewer operations for mixed workload

        # Start timing
        start_time = time.time()

        # Perform mixed read/write operations
        conn = db.engine.raw_connection()
        self.open_connections.append(conn)
        cursor = conn.cursor()

        # Create test tables with a unique name
        table_name = f"mixed_test_{unique_suffix}"
        cursor.execute(
            f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, data TEXT, value REAL, timestamp INTEGER)"
        )
        cursor.execute(f"CREATE INDEX idx_{table_name}_value ON {table_name} (value)")

        # Perform interleaved read and write operations
        for i in range(num_operations):
            # Insert a record
            cursor.execute(
                f"INSERT INTO {table_name} (id, data, value, timestamp) VALUES (?, ?, ?, ?)",
                (i, f"data_{i}", i / 1000.0, int(time.time())),
            )

            # Every 10 inserts, perform some reads
            if i % 10 == 0:
                # Random point query
                query_id = i // 2 if i > 0 else 0
                cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (query_id,))
                result = cursor.fetchone()

                # Range query
                start_id = max(0, i - 100)
                end_id = i
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE id BETWEEN ? AND ?",
                    (start_id, end_id),
                )
                result = cursor.fetchone()
                count = result[0] if result else 0

                # Aggregation
                cursor.execute(
                    f"SELECT AVG(value) FROM {table_name} WHERE id <= ?", (i,)
                )
                result = cursor.fetchone()
                avg_value = result[0] if result else 0

            # Commit every 100 operations
            if i % 100 == 0:
                conn.commit()

        # Final commit
        conn.commit()

        # End timing
        end_time = time.time()

        # Clean up
        cursor.close()
        conn.close()
        self.open_connections.remove(conn)

        db.close()
        self.open_dbs.remove(db)

        return end_time - start_time

    def _close_all_connections(self):
        """Close all open database connections."""
        # Close all open connections
        for conn in list(self.open_connections):
            try:
                conn.close()
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                if conn in self.open_connections:
                    self.open_connections.remove(conn)

        # Close all open databases
        for db in list(self.open_dbs):
            try:
                db.close()
            except Exception as e:
                print(f"Error closing database: {e}")
            finally:
                if db in self.open_dbs:
                    self.open_dbs.remove(db)

    def cleanup(self) -> None:
        """
        Clean up after the benchmark.
        """
        # First, close all open database connections
        self._close_all_connections()

        # Give the system a moment to release file handles
        time.sleep(1)

        # Now try to remove the temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                print(
                    f"Warning: Could not remove temporary directory {self.temp_dir}: {e}"
                )
                print(
                    "This is not critical, but you may want to clean it up manually later."
                )
