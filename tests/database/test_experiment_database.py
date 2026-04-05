"""Tests for farm/database/experiment_database.py.

Uses SQLite :memory: / temp files to test ExperimentDatabase,
ExperimentDataLogger, and SimulationContext without a real DB file.
"""

import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock

import pytest

from farm.database.data_logging import DataLoggingConfig
from farm.database.experiment_database import (
    ExperimentDataLogger,
    ExperimentDatabase,
    SimulationContext,
)


# ---------------------------------------------------------------------------
# ExperimentDataLogger
# ---------------------------------------------------------------------------


class TestExperimentDataLoggerInit(unittest.TestCase):

    def _make_db(self):
        db = Mock()
        db._execute_in_transaction = lambda f: f(Mock())
        return db

    def test_requires_simulation_id(self):
        db = self._make_db()
        with self.assertRaises(ValueError):
            ExperimentDataLogger(db, simulation_id=None)

    def test_init_sets_simulation_id(self):
        db = self._make_db()
        logger = ExperimentDataLogger(db, simulation_id="sim_exp_001")
        self.assertEqual(logger.simulation_id, "sim_exp_001")

    def test_inherits_from_data_logger(self):
        from farm.database.data_logging import DataLogger
        db = self._make_db()
        logger = ExperimentDataLogger(db, simulation_id="sim_001")
        self.assertIsInstance(logger, DataLogger)

    def test_custom_config(self):
        db = self._make_db()
        config = DataLoggingConfig(buffer_size=50, commit_interval=5)
        logger = ExperimentDataLogger(db, simulation_id="sim_001", config=config)
        self.assertEqual(logger._buffer_size, 50)


class TestExperimentDataLoggerLogAgent(unittest.TestCase):

    def setUp(self):
        self.mock_session = Mock()
        self.db = Mock()

        def execute_in_transaction(func):
            return func(self.mock_session)

        self.db._execute_in_transaction = execute_in_transaction
        self.logger = ExperimentDataLogger(self.db, simulation_id="exp_sim_001")

    def test_log_agent_delegates_to_parent(self):
        self.mock_session.bulk_insert_mappings = Mock()
        self.logger.log_agent(
            agent_id="agent_1",
            birth_time=0,
            agent_type="SystemAgent",
            position=(5.0, 10.0),
            initial_resources=100.0,
            starting_health=50.0,
        )
        self.mock_session.bulk_insert_mappings.assert_called_once()


# ---------------------------------------------------------------------------
# ExperimentDatabase
# ---------------------------------------------------------------------------


class TestExperimentDatabase(unittest.TestCase):

    def test_init_creates_database_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/exp_test.db"
            db = ExperimentDatabase(db_path=db_path, experiment_id="exp_001")
            self.assertEqual(db.experiment_id, "exp_001")
            db.close()

    def test_get_simulation_ids_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/exp_test2.db"
            db = ExperimentDatabase(db_path=db_path, experiment_id="exp_002")
            sim_ids = db.get_simulation_ids()
            self.assertIsInstance(sim_ids, list)
            db.close()

    def test_create_simulation_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/exp_test3.db"
            db = ExperimentDatabase(db_path=db_path, experiment_id="exp_003")
            ctx = db.create_simulation_context(
                simulation_id="sim_001", parameters={"width": 10}
            )
            self.assertIsNotNone(ctx)
            db.close()

    def test_update_simulation_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/exp_test4.db"
            db = ExperimentDatabase(db_path=db_path, experiment_id="exp_004")
            # Create a simulation record first
            db.create_simulation_context(simulation_id="sim_upd", parameters={})
            # Update the status
            db.update_simulation_status(simulation_id="sim_upd", status="completed")
            sim_ids = db.get_simulation_ids()
            self.assertIn("sim_upd", sim_ids)
            db.close()

    def test_update_experiment_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/exp_test5.db"
            db = ExperimentDatabase(db_path=db_path, experiment_id="exp_005")
            # Should not raise
            db.update_experiment_status(status="completed")
            db.close()


# ---------------------------------------------------------------------------
# SimulationContext
# ---------------------------------------------------------------------------


class TestSimulationContext(unittest.TestCase):

    def _make_experiment_db(self, tmpdir):
        db_path = f"{tmpdir}/ctx_test.db"
        return ExperimentDatabase(db_path=db_path, experiment_id="exp_ctx")

    def test_simulation_context_has_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_experiment_db(tmpdir)
            ctx = db.create_simulation_context(
                simulation_id="sim_ctx_001", parameters={}
            )
            self.assertIsNotNone(ctx)
            # Context should expose a logger
            self.assertIsNotNone(ctx.logger)
            db.close()

    def test_simulation_context_has_simulation_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_experiment_db(tmpdir)
            ctx = db.create_simulation_context(
                simulation_id="sim_ctx_002", parameters={}
            )
            self.assertEqual(ctx.simulation_id, "sim_ctx_002")
            db.close()

    def test_simulation_context_log_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_experiment_db(tmpdir)
            ctx = db.create_simulation_context(
                simulation_id="sim_ctx_003", parameters={}
            )
            # log_step should work without error using empty states
            ctx.log_step(
                step_number=1,
                agent_states=[],
                resource_states=[],
                metrics={"total_agents": 0},
            )
            db.close()

    def test_simulation_context_log_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_experiment_db(tmpdir)
            ctx = db.create_simulation_context(
                simulation_id="sim_ctx_004", parameters={}
            )
            ctx.log_agent(
                agent_id="agent_01",
                birth_time=0,
                agent_type="SystemAgent",
                position=(3.0, 4.0),
                initial_resources=100.0,
                starting_health=50.0,
            )
            db.close()

    def test_simulation_context_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_experiment_db(tmpdir)
            ctx = db.create_simulation_context(
                simulation_id="sim_ctx_005", parameters={}
            )
            ctx.logger.flush_all_buffers()
            db.close()


if __name__ == "__main__":
    unittest.main()
