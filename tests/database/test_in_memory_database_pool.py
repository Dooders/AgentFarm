"""The in-memory simulation database must keep one SQLite connection for its whole life."""

import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from farm.database.database import InMemorySimulationDatabase

pytestmark = pytest.mark.unit


def _tables(connection) -> set[str]:
    return {row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}


def test_in_memory_database_shares_one_connection():
    db = InMemorySimulationDatabase(simulation_id="pool_test")
    try:
        assert isinstance(db.engine.pool, StaticPool)
        # A second simultaneous checkout from a QueuePool would open a new,
        # empty ":memory:" database; with one shared connection it sees the schema.
        with db.engine.connect() as first, db.engine.connect() as second:
            assert "agent_states" in _tables(first)
            assert _tables(second) == _tables(first)
    finally:
        db.close()


def test_in_memory_database_never_recycles_its_connection():
    db = InMemorySimulationDatabase(simulation_id="recycle_test")
    try:
        # Replacing the connection after pool_recycle (an hour by default)
        # swapped in an empty database mid-run; -1 disables recycling.
        assert db.engine.pool._recycle == -1
    finally:
        db.close()
