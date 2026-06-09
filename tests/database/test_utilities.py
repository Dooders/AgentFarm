"""Tests for farm/database/utilities.py covering all helper functions."""

import os
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from farm.database.models import Base, Simulation
from farm.database.utilities import (
    as_dict,
    create_database_schema,
    create_prepared_statements,
    execute_query,
    execute_with_retry,
    format_agent_state,
    format_position,
    parse_position,
    safe_json_loads,
    setup_db,
    validate_export_format,
)


class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"key": "value", "number": 42}') == {"key": "value", "number": 42}

    def test_invalid_json_returns_none(self):
        assert safe_json_loads("invalid json {") is None

    def test_empty_input_returns_none(self):
        assert safe_json_loads(None) is None
        assert safe_json_loads("") is None

    def test_non_string_input_returns_none(self):
        # json.loads raises TypeError for non-str/bytes, hitting the generic handler.
        assert safe_json_loads(object()) is None


class TestAsDict:
    def test_converts_model_columns(self):
        sim = Simulation(simulation_id="sim_1", parameters={"a": 1}, simulation_db_path="/tmp/x.db")
        result = as_dict(sim)
        assert result["simulation_id"] == "sim_1"
        assert result["parameters"] == {"a": 1}
        assert set(result) == {col.name for col in Simulation.__table__.columns}


class TestPositionFormatting:
    def test_format_position(self):
        assert format_position((10.5, 20.7)) == "10.5, 20.7"

    def test_parse_position(self):
        assert parse_position("10.5, 20.7") == (10.5, 20.7)

    def test_round_trip(self):
        assert parse_position(format_position((1.25, -3.5))) == (1.25, -3.5)

    def test_parse_invalid_string_returns_origin(self):
        assert parse_position("not-a-position") == (0.0, 0.0)

    def test_parse_none_returns_origin(self):
        assert parse_position(None) == (0.0, 0.0)


class TestCreateDatabaseSchema:
    def test_creates_tables_on_sqlite(self):
        engine = create_engine("sqlite:///:memory:")
        create_database_schema(engine, Base)
        from sqlalchemy import inspect

        assert "simulations" in inspect(engine).get_table_names()

    def test_propagates_sqlalchemy_error(self):
        base = Mock()
        base.metadata.create_all.side_effect = SQLAlchemyError("schema failure")
        with pytest.raises(SQLAlchemyError):
            create_database_schema(Mock(), base)


class TestValidateExportFormat:
    @pytest.mark.parametrize("fmt", ["csv", "excel", "json", "parquet", "CSV", "Json"])
    def test_supported_formats(self, fmt):
        assert validate_export_format(fmt)

    @pytest.mark.parametrize("fmt", ["xml", "yaml", ""])
    def test_unsupported_formats(self, fmt):
        assert not validate_export_format(fmt)


class TestFormatAgentState:
    def test_full_state(self):
        state = {
            "position": (3.0, 4.0),
            "current_health": 0.8,
            "starting_health": 1.0,
            "resource_level": 5.0,
            "is_defending": True,
            "total_reward": 2.5,
            "starvation_counter": 1,
        }
        result = format_agent_state("agent_1", 10, state, simulation_id="sim_1")
        assert result["agent_id"] == "agent_1"
        assert result["step_number"] == 10
        assert result["age"] == 10
        assert result["position_x"] == 3.0
        assert result["position_y"] == 4.0
        assert result["is_defending"] is True
        assert result["simulation_id"] == "sim_1"

    def test_defaults_and_no_simulation_id(self):
        result = format_agent_state("agent_1", 0, {})
        assert result["position_x"] == 0.0
        assert result["current_health"] == 0.0
        assert result["is_defending"] is False
        assert "simulation_id" not in result


class TestExecuteWithRetry:
    def test_success_commits_and_returns(self):
        session = Mock()
        result = execute_with_retry(session, lambda: "ok")
        assert result == "ok"
        session.commit.assert_called_once()
        session.rollback.assert_not_called()

    def test_retries_then_succeeds(self):
        session = Mock()
        operation = Mock(side_effect=[SQLAlchemyError("transient"), "ok"])
        result = execute_with_retry(session, operation)
        assert result == "ok"
        assert operation.call_count == 2
        session.rollback.assert_called_once()

    def test_raises_after_max_retries(self):
        session = Mock()
        operation = Mock(side_effect=SQLAlchemyError("persistent"))
        with pytest.raises(SQLAlchemyError):
            execute_with_retry(session, operation, max_retries=3)
        assert operation.call_count == 3
        assert session.rollback.call_count == 3


class TestExecuteQueryDecorator:
    def test_wraps_method_in_transaction(self):
        class Repo:
            def __init__(self):
                self.db = Mock()
                self.db._execute_in_transaction = lambda query: query("session-token")

            @execute_query
            def fetch(self, session, value):
                return (session, value)

        assert Repo().fetch(42) == ("session-token", 42)


class TestCreatePreparedStatements:
    def test_prepares_expected_statements(self):
        connection = Mock()
        connection.prepare.side_effect = lambda sql: f"prepared:{sql.split()[2]}"
        session = Mock()
        session.connection.return_value.connection = connection

        statements = create_prepared_statements(session)

        assert set(statements) == {"insert_agent_state", "insert_resource"}
        assert connection.prepare.call_count == 2


class TestSetupDb:
    def test_none_path_returns_none(self):
        assert setup_db(None, "sim_1") is None

    def test_memory_path_creates_in_memory_database(self):
        from farm.database.database import InMemorySimulationDatabase

        db = setup_db(":memory:", "sim_mem", parameters={"seed": 1})
        try:
            assert isinstance(db, InMemorySimulationDatabase)
        finally:
            db.close()

    def test_file_path_creates_database_with_simulation_record(self, tmp_path):
        from farm.database.database import SimulationDatabase

        db_path = str(tmp_path / "sim.db")
        db = setup_db(db_path, "sim_file")
        try:
            assert isinstance(db, SimulationDatabase)
            assert os.path.exists(db_path)
        finally:
            db.close()

    def test_existing_file_is_removed(self, tmp_path):
        db_path = tmp_path / "stale.db"
        db_path.write_text("stale contents")
        db = setup_db(str(db_path), "sim_fresh")
        try:
            # The stale file was replaced by a fresh SQLite database.
            assert db_path.read_bytes().startswith(b"SQLite format 3")
        finally:
            db.close()

    def test_unremovable_file_falls_back_to_unique_name(self, tmp_path):
        db_path = tmp_path / "locked.db"
        db_path.write_text("locked")
        with patch("os.remove", side_effect=OSError("locked")):
            db = setup_db(str(db_path), "sim_locked")
        try:
            assert str(tmp_path / "locked_") in db.db_path
            assert db.db_path != str(db_path)
        finally:
            db.close()
