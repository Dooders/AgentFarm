"""
#! might not be used right now
Data Loading Utilities

This module provides concrete implementations of DataLoader and DatabaseLoader
classes for loading data from various sources, with a focus on:
1. Database loaders for various AgentFarm database tables
2. File loaders for CSV, JSON, and other file formats
3. Helper functions for common data loading patterns

These loaders implement the interfaces defined in the base module and provide
standardized access to data for analysis components.

Examples:
    # Load data from a SQLite database
    loader = SQLiteLoader(db_path='path/to/simulation.db')
    agents_df = loader.load_data(table='agents')
    resources_df = loader.load_data(table='resources')

    # Load data for a specific simulation
    sim_loader = SimulationLoader(
        db_path='path/to/simulation.db',
        simulation_id=42  # Or simulation_name='experiment_1'
    )
    steps_df = sim_loader.load_data(table='steps')

    # Get time series data for population analysis
    time_series = sim_loader.load_time_series()

    # Load data from CSV and JSON files
    csv_loader = CSVLoader(file_path='path/to/data.csv')
    csv_data = csv_loader.load_data(index_col=0)  # Pass pandas read_csv parameters

    json_loader = JSONLoader(file_path='path/to/data.json')
    json_data = json_loader.load_data()

    # Load and analyze multiple experiments
    experiment_loader = ExperimentLoader(db_paths=[
        'path/to/experiment1.db',
        'path/to/experiment2.db',
        'path/to/experiment3.db'
    ])
    # Get combined data from all simulations
    all_sims = experiment_loader.load_data(table='simulations')

    # Access metadata about a database
    metadata = loader.get_metadata()
    print(f"Available tables: {metadata['tables']}")
    print(f"Record counts: {metadata['record_counts']}")
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from farm.analysis.base import DatabaseLoader, DataLoader
from farm.database.models import (
    AgentModel,
    ReproductionEventModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
)


class SQLiteLoader(DatabaseLoader):
    """SQLite database loader for AgentFarm simulation databases."""

    def __init__(self, db_path: str):
        """Initialize the SQLite database loader.

        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__(db_path=db_path)
        self._engine = None
        self._session_factory = None

    def connect(self):
        """Establish a connection to the SQLite database."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        self._engine = create_engine(f"sqlite:///{self.db_path}")
        self._session_factory = sessionmaker(bind=self._engine)
        self._connection = self._engine.connect()

    def disconnect(self):
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get_session(self) -> Session:
        """Get a database session.

        Returns:
            Session: SQLAlchemy session
        """
        if self._session_factory is None:
            self.connect()
        if self._session_factory is None:
            raise RuntimeError("Failed to create session factory")
        return self._session_factory()

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            pd.DataFrame: Query results
        """
        if self._connection is None:
            self.connect()

        if params is None:
            params = {}

        if self._connection is None:
            raise RuntimeError("Failed to establish database connection")

        result = self._connection.execute(text(query), params)
        columns = list(result.keys())
        data = result.fetchall()

        return pd.DataFrame(data, columns=columns)

    def execute_query_iter(
        self, query: str, params: Optional[Dict[str, Any]] = None, chunk_size: int = 10000
    ) -> "Iterator[pd.DataFrame]":
        """Execute a SQL query and stream results in DataFrame chunks.

        Args:
            query: SQL query string
            params: Query parameters
            chunk_size: Number of rows per yielded chunk

        Yields:
            DataFrame chunks of up to chunk_size rows
        """
        if self._connection is None:
            self.connect()

        if params is None:
            params = {}

        if self._connection is None:
            raise RuntimeError("Failed to establish database connection")

        # Use server-side cursor-like behavior via fetchmany loop
        result = self._connection.execute(text(query), params)
        columns = list(result.keys())
        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)

    def load_data(self, *args, **kwargs) -> pd.DataFrame:
        """Load data from the database.

        This is a generic method that forwards to more specific loading methods
        based on the 'table' parameter.

        Args:
            table: The table to load data from
                ('agents', 'resources', 'steps', 'reproductions', 'simulations')
            **kwargs: Additional parameters for specific loaders

        Returns:
            pd.DataFrame: Loaded data
        """
        if "table" not in kwargs:
            raise ValueError("The 'table' parameter is required")

        table = kwargs.pop("table")

        loaders = {
            "agents": self.load_agents,
            "resources": self.load_resources,
            "steps": self.load_steps,
            "reproductions": self.load_reproduction_events,
            "simulations": self.load_simulations,
        }

        if table not in loaders:
            raise ValueError(
                f"Unknown table: {table}. Available tables: {list(loaders.keys())}"
            )

        return loaders[table](**kwargs)

    def load_agents(self, simulation_id: Optional[int] = None) -> pd.DataFrame:
        """Load agent data from the database.

        Args:
            simulation_id: Filter by simulation ID

        Returns:
            pd.DataFrame: Agent data
        """
        session = self.get_session()
        query = session.query(AgentModel)

        if simulation_id is not None:
            query = query.filter(AgentModel.simulation_id == simulation_id)

        agents = query.all()
        session.close()

        # Convert to DataFrame
        data = []
        for agent in agents:
            agent_dict = {
                "id": agent.id,
                "simulation_id": agent.simulation_id,
                "agent_type": agent.agent_type,
                "generation": agent.generation,
                "parent_id": agent.parent_id,
                "birth_step": agent.birth_step,
                "death_step": agent.death_step,
                "genome": agent.genome,
                "position_x": agent.position_x,
                "position_y": agent.position_y,
                "initial_health": agent.initial_health,
                "initial_energy": agent.initial_energy,
            }
            data.append(agent_dict)

        return pd.DataFrame(data)

    def iter_agents(
        self, simulation_id: Optional[int] = None, chunk_size: int = 10000
    ) -> "Iterator[pd.DataFrame]":
        session = self.get_session()
        try:
            query = session.query(AgentModel)
            if simulation_id is not None:
                query = query.filter(AgentModel.simulation_id == simulation_id)
            buffer: list = []
            for agent in query.yield_per(chunk_size):
                buffer.append(
                    {
                        "id": agent.id,
                        "simulation_id": agent.simulation_id,
                        "agent_type": agent.agent_type,
                        "generation": agent.generation,
                        "parent_id": agent.parent_id,
                        "birth_step": agent.birth_step,
                        "death_step": agent.death_step,
                        "genome": agent.genome,
                        "position_x": agent.position_x,
                        "position_y": agent.position_y,
                        "initial_health": agent.initial_health,
                        "initial_energy": agent.initial_energy,
                    }
                )
                if len(buffer) >= chunk_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            session.close()

    def load_resources(self, simulation_id: Optional[int] = None) -> pd.DataFrame:
        """Load resource data from the database.

        Args:
            simulation_id: Filter by simulation ID

        Returns:
            pd.DataFrame: Resource data
        """
        session = self.get_session()
        query = session.query(ResourceModel)

        if simulation_id is not None:
            query = query.filter(ResourceModel.simulation_id == simulation_id)

        resources = query.all()
        session.close()

        # Convert to DataFrame
        data = []
        for resource in resources:
            resource_dict = {
                "id": resource.id,
                "simulation_id": resource.simulation_id,
                "resource_type": resource.resource_type,
                "position_x": resource.position_x,
                "position_y": resource.position_y,
                "creation_step": resource.creation_step,
                "depletion_step": resource.depletion_step,
                "initial_value": resource.initial_value,
            }
            data.append(resource_dict)

        return pd.DataFrame(data)

    def iter_resources(
        self, simulation_id: Optional[int] = None, chunk_size: int = 10000
    ) -> "Iterator[pd.DataFrame]":
        session = self.get_session()
        try:
            query = session.query(ResourceModel)
            if simulation_id is not None:
                query = query.filter(ResourceModel.simulation_id == simulation_id)
            buffer: list = []
            for resource in query.yield_per(chunk_size):
                buffer.append(
                    {
                        "id": resource.id,
                        "simulation_id": resource.simulation_id,
                        "resource_type": resource.resource_type,
                        "position_x": resource.position_x,
                        "position_y": resource.position_y,
                        "creation_step": resource.creation_step,
                        "depletion_step": resource.depletion_step,
                        "initial_value": resource.initial_value,
                    }
                )
                if len(buffer) >= chunk_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            session.close()

    def load_steps(self, simulation_id: Optional[int] = None) -> pd.DataFrame:
        """Load simulation step data from the database.

        Args:
            simulation_id: Filter by simulation ID

        Returns:
            pd.DataFrame: Step data
        """
        session = self.get_session()
        query = session.query(SimulationStepModel)

        if simulation_id is not None:
            query = query.filter(SimulationStepModel.simulation_id == simulation_id)

        steps = query.all()
        session.close()

        # Convert to DataFrame
        data = []
        for step in steps:
            step_dict = {
                "id": step.id,
                "simulation_id": step.simulation_id,
                "step_number": step.step_number,
                "agent_counts": json.loads(step.agent_counts),
                "resource_counts": json.loads(step.resource_counts),
                "timestamp": step.timestamp,
            }
            data.append(step_dict)

        return pd.DataFrame(data)

    def iter_steps(
        self, simulation_id: Optional[int] = None, chunk_size: int = 10000
    ) -> "Iterator[pd.DataFrame]":
        session = self.get_session()
        try:
            query = session.query(SimulationStepModel)
            if simulation_id is not None:
                query = query.filter(SimulationStepModel.simulation_id == simulation_id)
            buffer: list = []
            for step in query.yield_per(chunk_size):
                buffer.append(
                    {
                        "id": step.id,
                        "simulation_id": step.simulation_id,
                        "step_number": step.step_number,
                        "agent_counts": json.loads(step.agent_counts),
                        "resource_counts": json.loads(step.resource_counts),
                        "timestamp": step.timestamp,
                    }
                )
                if len(buffer) >= chunk_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            session.close()

    def load_reproduction_events(
        self, simulation_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Load reproduction event data from the database.

        Args:
            simulation_id: Filter by simulation ID

        Returns:
            pd.DataFrame: Reproduction event data
        """
        session = self.get_session()
        query = session.query(ReproductionEventModel)

        if simulation_id is not None:
            query = query.filter(ReproductionEventModel.simulation_id == simulation_id)

        events = query.all()
        session.close()

        # Convert to DataFrame
        data = []
        for event in events:
            event_dict = {
                "id": event.id,
                "simulation_id": event.simulation_id,
                "step_number": event.step_number,
                "parent_id": event.parent_id,
                "child_id": event.child_id,
                "parent_health": event.parent_health,
                "parent_energy": event.parent_energy,
                "parent_age": event.parent_age,
                "child_genome": event.child_genome,
                "mutation_rate": event.mutation_rate,
            }
            data.append(event_dict)

        return pd.DataFrame(data)

    def iter_reproduction_events(
        self, simulation_id: Optional[int] = None, chunk_size: int = 10000
    ) -> "Iterator[pd.DataFrame]":
        session = self.get_session()
        try:
            query = session.query(ReproductionEventModel)
            if simulation_id is not None:
                query = query.filter(ReproductionEventModel.simulation_id == simulation_id)
            buffer: list = []
            for event in query.yield_per(chunk_size):
                buffer.append(
                    {
                        "id": event.id,
                        "simulation_id": event.simulation_id,
                        "step_number": event.step_number,
                        "parent_id": event.parent_id,
                        "child_id": event.child_id,
                        "parent_health": event.parent_health,
                        "parent_energy": event.parent_energy,
                        "parent_age": event.parent_age,
                        "child_genome": event.child_genome,
                        "mutation_rate": event.mutation_rate,
                    }
                )
                if len(buffer) >= chunk_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            session.close()

    def load_simulations(self) -> pd.DataFrame:
        """Load simulation metadata from the database.

        Returns:
            pd.DataFrame: Simulation metadata
        """
        session = self.get_session()
        simulations = session.query(Simulation).all()
        session.close()

        # Convert to DataFrame
        data = []
        for sim in simulations:
            sim_dict = {
                "simulation_id": sim.simulation_id,
                "experiment_id": sim.experiment_id,
                "start_time": sim.start_time,
                "end_time": sim.end_time,
                "status": sim.status,
                "parameters": sim.parameters,
                "results_summary": sim.results_summary,
                "simulation_db_path": sim.simulation_db_path,
            }
            data.append(sim_dict)

        return pd.DataFrame(data)

    def iter_simulations(self, chunk_size: int = 10000) -> "Iterator[pd.DataFrame]":
        session = self.get_session()
        try:
            buffer: list = []
            for sim in session.query(Simulation).yield_per(chunk_size):
                buffer.append(
                    {
                        "simulation_id": sim.simulation_id,
                        "experiment_id": sim.experiment_id,
                        "start_time": sim.start_time,
                        "end_time": sim.end_time,
                        "status": sim.status,
                        "parameters": sim.parameters,
                        "results_summary": sim.results_summary,
                        "simulation_db_path": sim.simulation_db_path,
                    }
                )
                if len(buffer) >= chunk_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            session.close()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the database.

        Returns:
            Dict[str, Any]: Database metadata
        """
        if self._connection is None:
            self.connect()

        # Get general database info
        tables = self.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        table_list = tables["name"].tolist()

        # Get counts
        counts = {}
        for table in table_list:
            count = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = count["count"].iloc[0]

        # Get simulation IDs
        if "simulations" in table_list:
            sims = self.execute_query(
                "SELECT simulation_id, experiment_id FROM simulations"
            )
            sim_list = sims.to_dict("records")
        else:
            sim_list = []

        return {
            "database_path": self.db_path,
            "tables": table_list,
            "record_counts": counts,
            "simulations": sim_list,
        }


class SimulationLoader(SQLiteLoader):
    """Specialized loader for a specific simulation."""

    def __init__(
        self,
        db_path: str,
        simulation_id: Optional[int] = None,
        simulation_name: Optional[str] = None,
    ):
        """Initialize the simulation loader.

        Args:
            db_path: Path to the database file
            simulation_id: ID of the simulation to load
            simulation_name: Name of the simulation to load (alternative to simulation_id)
        """
        super().__init__(db_path=db_path)
        self.simulation_id = simulation_id
        self.simulation_name = simulation_name

        # If only name is provided, look up the ID
        if simulation_id is None and simulation_name is not None:
            self.connect()
            query = f"SELECT simulation_id FROM simulations WHERE experiment_id = :name"
            result = self.execute_query(query, {"name": simulation_name})
            if not result.empty:
                self.simulation_id = result["simulation_id"].iloc[0]
            self.disconnect()

    def load_data(self, table: str = "steps", **kwargs) -> pd.DataFrame:
        """Load data for the specified simulation.

        Args:
            table: The table to load data from
            **kwargs: Additional parameters for the loader

        Returns:
            pd.DataFrame: Loaded data
        """
        if self.simulation_id is None:
            raise ValueError("No simulation ID or name specified, or name not found")

        kwargs["simulation_id"] = self.simulation_id
        return super().load_data(table=table, **kwargs)

    def iter_data(self, table: str = "steps", chunk_size: int = 10000, **kwargs):
        if self.simulation_id is None:
            raise ValueError("No simulation ID or name specified, or name not found")

        kwargs["simulation_id"] = self.simulation_id
        table_iters = {
            "agents": self.iter_agents,
            "resources": self.iter_resources,
            "steps": self.iter_steps,
            "reproductions": self.iter_reproduction_events,
            "simulations": self.iter_simulations,
        }
        if table not in table_iters:
            raise ValueError(
                f"Unknown table: {table}. Available: {list(table_iters.keys())}"
            )
        yield from table_iters[table](chunk_size=chunk_size, **kwargs)

    def load_time_series(self) -> pd.DataFrame:
        """Load time series data for the simulation.

        Returns:
            pd.DataFrame: Time series data
        """
        steps_df = self.load_data(table="steps")

        # Extract agent counts by type
        agent_counts = []
        for _, row in steps_df.iterrows():
            step = row["step_number"]
            counts = row["agent_counts"]
            for agent_type, count in counts.items():
                agent_counts.append(
                    {"step": step, "agent_type": agent_type, "count": count}
                )

        return pd.DataFrame(agent_counts)

    def iter_time_series(self, chunk_size: int = 10000) -> "Iterator[pd.DataFrame]":
        """Stream time series rows derived from steps in chunks."""
        buffer: list = []
        for steps_df in self.iter_data(table="steps", chunk_size=chunk_size):
            for _, row in steps_df.iterrows():
                step = row["step_number"]
                counts = row["agent_counts"]
                for agent_type, count in counts.items():
                    buffer.append({"step": step, "agent_type": agent_type, "count": count})
                    if len(buffer) >= chunk_size:
                        yield pd.DataFrame(buffer)
                        buffer = []
        if buffer:
            yield pd.DataFrame(buffer)

    def get_simulation_config(self) -> Dict[str, Any]:
        """Get the configuration for the simulation.

        Returns:
            Dict[str, Any]: Simulation configuration
        """
        if self.simulation_id is None:
            raise ValueError("No simulation ID or name specified, or name not found")

        self.connect()
        query = "SELECT parameters FROM simulations WHERE simulation_id = :sim_id"
        result = self.execute_query(query, {"sim_id": self.simulation_id})
        self.disconnect()

        if result.empty:
            return {}

        config_str = result["parameters"].iloc[0]
        return config_str if config_str else {}


class CSVLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, file_path: str):
        """Initialize the CSV loader.

        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path

    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            **kwargs: Additional parameters for pd.read_csv

        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        return pd.read_csv(self.file_path, **kwargs)

    def iter_data(self, chunksize: int = 100000, **kwargs):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        # Pandas returns an iterator of DataFrames when chunksize is provided
        reader = pd.read_csv(self.file_path, chunksize=chunksize, **kwargs)
        for chunk in reader:
            yield chunk

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the CSV file.

        Returns:
            Dict[str, Any]: File metadata
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # Get file info
        file_path = Path(self.file_path)
        file_size = file_path.stat().st_size
        file_modified = file_path.stat().st_mtime

        # Peek at the data
        df = pd.read_csv(self.file_path, nrows=5)

        return {
            "file_path": str(file_path),
            "file_size": file_size,
            "file_modified": file_modified,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview_rows": len(df),
        }


class JSONLoader(DataLoader):
    """Data loader for JSON files."""

    def __init__(self, file_path: str):
        """Initialize the JSON loader.

        Args:
            file_path: Path to the JSON file
        """
        self.file_path = file_path

    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from a JSON file.

        Args:
            **kwargs: Additional parameters for pd.read_json

        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        return pd.read_json(self.file_path, **kwargs)

    def iter_data(self, chunksize: int = 100000, lines: Optional[bool] = None, **kwargs):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        # If file is JSON Lines or caller requests lines mode, use pandas chunking
        if lines is True or (lines is None and self.file_path.lower().endswith((".jsonl", ".ndjson"))):
            reader = pd.read_json(self.file_path, lines=True, chunksize=chunksize, **kwargs)
            for chunk in reader:
                yield chunk
            return

        # Fallback: load once when not line-delimited (chunking not supported without extra deps)
        yield self.load_data(**kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the JSON file.

        Returns:
            Dict[str, Any]: File metadata
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        # Get file info
        file_path = Path(self.file_path)
        file_size = file_path.stat().st_size
        file_modified = file_path.stat().st_mtime

        # Read the JSON structure
        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Determine if it's a list of records or a single object
        structure_type = "list" if isinstance(data, list) else "object"

        return {
            "file_path": str(file_path),
            "file_size": file_size,
            "file_modified": file_modified,
            "structure_type": structure_type,
            "record_count": len(data) if structure_type == "list" else 1,
        }


class ExperimentLoader(DataLoader):
    """Loader for multiple simulation runs in an experiment."""

    def __init__(self, db_paths: List[str]):
        """Initialize the experiment loader.

        Args:
            db_paths: List of paths to simulation database files
        """
        self.db_paths = db_paths
        self._loaders = [SQLiteLoader(db_path) for db_path in db_paths]

    def load_data(self, table: str = "simulations", **kwargs) -> pd.DataFrame:
        """Load data from all simulation databases.

        Args:
            table: The table to load data from
            **kwargs: Additional parameters for specific loaders

        Returns:
            pd.DataFrame: Combined data from all simulations
        """
        all_data = []

        for loader in self._loaders:
            try:
                data = loader.load_data(table=table, **kwargs)
                # Add database path as a reference
                data["db_path"] = loader.db_path
                all_data.append(data)
            except Exception as e:
                print(f"Error loading data from {loader.db_path}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def iter_data(self, table: str = "simulations", chunk_size: int = 10000, **kwargs):
        for loader in self._loaders:
            try:
                # Dispatch to iterator on each SQLiteLoader
                if isinstance(loader, SQLiteLoader):
                    sim_loader = SimulationLoader(db_path=loader.db_path)
                    iterator = sim_loader.iter_data(table=table, chunk_size=chunk_size, **kwargs)
                else:
                    iterator = loader.iter_data(**kwargs)
                for chunk in iterator:
                    chunk = chunk.copy()
                    chunk["db_path"] = loader.db_path  # type: ignore[attr-defined]
                    yield chunk
            except Exception as e:
                print(f"Error streaming data from {getattr(loader, 'db_path', 'unknown')}: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about all the simulation databases.

        Returns:
            Dict[str, Any]: Combined metadata
        """
        metadata = []

        for loader in self._loaders:
            try:
                meta = loader.get_metadata()
                metadata.append(meta)
            except Exception as e:
                print(f"Error getting metadata from {loader.db_path}: {e}")

        return {
            "databases": metadata,
            "total_databases": len(metadata),
        }
