"""Tests for database data logging module.

Covers DataLoggingConfig, DataLogger, and ShardedDataLogger classes using
mocked database sessions to avoid heavy integration dependencies.
"""

import unittest
from unittest.mock import Mock

import pytest

from farm.database.data_logging import DataLogger, DataLoggingConfig, ShardedDataLogger


class TestDataLoggingConfig(unittest.TestCase):
    """Tests for DataLoggingConfig dataclass."""

    def test_default_values(self):
        config = DataLoggingConfig()
        self.assertEqual(config.buffer_size, 1000)
        self.assertEqual(config.commit_interval, 30)

    def test_custom_values(self):
        config = DataLoggingConfig(buffer_size=500, commit_interval=60)
        self.assertEqual(config.buffer_size, 500)
        self.assertEqual(config.commit_interval, 60)


class TestDataLoggerInit(unittest.TestCase):
    """Tests for DataLogger initialization."""

    def _make_db(self):
        """Create a minimal mock database that satisfies the DataLogger API."""
        db = Mock()
        db._execute_in_transaction = lambda f: f(Mock())
        return db

    def test_init_requires_simulation_id(self):
        db = self._make_db()
        with self.assertRaises(ValueError):
            DataLogger(db, simulation_id=None)

    def test_init_with_simulation_id(self):
        db = self._make_db()
        logger = DataLogger(db, simulation_id="sim_001")
        self.assertEqual(logger.simulation_id, "sim_001")

    def test_init_with_custom_config(self):
        db = self._make_db()
        config = DataLoggingConfig(buffer_size=200, commit_interval=10)
        logger = DataLogger(db, simulation_id="sim_001", config=config)
        self.assertEqual(logger._buffer_size, 200)
        self.assertEqual(logger._commit_interval, 10)

    def test_init_empty_buffers(self):
        db = self._make_db()
        logger = DataLogger(db, simulation_id="sim_001")
        self.assertEqual(logger._action_buffer, [])
        self.assertEqual(logger._health_incident_buffer, [])


class TestDataLoggerLogAgent(unittest.TestCase):
    """Tests for DataLogger.log_agent method.

    log_agent delegates to log_agents_batch which calls
    session.bulk_insert_mappings, not session.add.
    """

    def setUp(self):
        self.mock_session = Mock()
        self.db = Mock()

        def execute_in_transaction(func):
            return func(self.mock_session)

        self.db._execute_in_transaction = execute_in_transaction
        self.logger = DataLogger(self.db, simulation_id="sim_001")

    def test_log_agent_calls_bulk_insert(self):
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

    def test_log_agent_with_optional_params(self):
        self.mock_session.bulk_insert_mappings = Mock()

        self.logger.log_agent(
            agent_id="agent_2",
            birth_time=5,
            agent_type="IndependentAgent",
            position=(3.0, 7.0),
            initial_resources=50.0,
            starting_health=100.0,
            genome_id="genome_abc",
            generation=2,
            action_weights={"move": 0.5, "eat": 0.5},
        )
        self.mock_session.bulk_insert_mappings.assert_called_once()


class TestDataLoggerBuffering(unittest.TestCase):
    """Tests for DataLogger buffering and flush behaviour."""

    def setUp(self):
        self.mock_session = Mock()
        self.db = Mock()

        def execute_in_transaction(func):
            return func(self.mock_session)

        self.db._execute_in_transaction = execute_in_transaction
        self.logger = DataLogger(
            self.db,
            simulation_id="sim_001",
            config=DataLoggingConfig(buffer_size=5, commit_interval=9999),
        )

    def test_action_buffer_accumulates(self):
        for i in range(3):
            self.logger.log_agent_action(
                step_number=i,
                agent_id="agent_1",
                action_type="move",
            )
        self.assertEqual(len(self.logger._action_buffer), 3)

    def test_flush_all_buffers_empties_buffers(self):
        self.logger.log_agent_action(
            step_number=0, agent_id="a1", action_type="move"
        )
        self.logger.log_health_incident(
            step_number=0,
            agent_id="a1",
            health_before=100.0,
            health_after=90.0,
            cause="attack",
        )

        self.mock_session.bulk_insert_mappings = Mock()
        self.mock_session.commit = Mock()
        self.logger.flush_all_buffers()

        self.assertEqual(len(self.logger._action_buffer), 0)
        self.assertEqual(len(self.logger._health_incident_buffer), 0)

    def test_needs_flush_false_when_empty(self):
        """needs_flush should be False when all buffers are empty."""
        self.assertFalse(self.logger.needs_flush)

    def test_needs_flush_true_after_action_buffered(self):
        """needs_flush should be True once an action is buffered."""
        self.logger.log_agent_action(
            step_number=0, agent_id="a1", action_type="move"
        )
        self.assertTrue(self.logger.needs_flush)

    def test_needs_flush_false_after_flush(self):
        """needs_flush should return False after all buffers are flushed."""
        self.logger.log_agent_action(
            step_number=0, agent_id="a1", action_type="move"
        )
        self.mock_session.bulk_insert_mappings = Mock()
        self.mock_session.commit = Mock()
        self.logger.flush_all_buffers()
        self.assertFalse(self.logger.needs_flush)

    def test_flush_all_buffers_noop_when_empty(self):
        """flush_all_buffers should not touch the DB when all buffers are empty."""
        self.db._execute_in_transaction = Mock()
        self.logger.flush_all_buffers()
        self.db._execute_in_transaction.assert_not_called()

    def test_flush_if_needed_triggers_on_interval(self):
        """flush_if_needed should flush when commit interval has elapsed."""
        self.logger.log_agent_action(
            step_number=0, agent_id="a1", action_type="move"
        )
        # Wind the clock back so the interval appears elapsed.
        self.logger._last_commit_time -= self.logger._commit_interval + 1
        self.mock_session.bulk_insert_mappings = Mock()
        self.mock_session.commit = Mock()
        self.logger.flush_if_needed()
        self.assertFalse(self.logger.needs_flush)

    def test_flush_if_needed_skips_before_interval(self):
        """flush_if_needed should not flush before commit interval has elapsed."""
        self.logger.log_agent_action(
            step_number=0, agent_id="a1", action_type="move"
        )
        # Leave _last_commit_time at 'now' so interval has NOT elapsed.
        self.logger._last_commit_time = float("inf")  # far future
        self.logger.flush_if_needed()
        # Buffer must still have the pending item.
        self.assertTrue(self.logger.needs_flush)

    def test_auto_flush_on_buffer_full(self):
        """When buffer reaches buffer_size the actions should be flushed."""
        self.mock_session.bulk_insert_mappings = Mock()
        self.mock_session.commit = Mock()

        # Fill exactly to buffer_size (5)
        for i in range(5):
            self.logger.log_agent_action(
                step_number=i, agent_id="agent_1", action_type="move"
            )

        # After 5 inserts (== buffer_size) the buffer should have been flushed
        self.assertEqual(len(self.logger._action_buffer), 0)

    def test_negative_step_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.logger.log_agent_action(
                step_number=-1, agent_id="a1", action_type="move"
            )


class TestDataLoggerLogStep(unittest.TestCase):
    """Tests for DataLogger.log_step method.

    agent_states and resource_states must be tuples/lists (indexed), not dicts.
    """

    def setUp(self):
        self.mock_session = Mock()
        self.db = Mock()

        def execute_in_transaction(func):
            return func(self.mock_session)

        self.db._execute_in_transaction = execute_in_transaction
        self.logger = DataLogger(self.db, simulation_id="sim_001")

    def _make_agent_state_tuple(self):
        # (agent_id, pos_x, pos_y, resource, health, start_health,
        #  starvation, is_defending, total_reward, age)
        return ("a1", 1.0, 2.0, 50.0, 100.0, 100.0, 0, False, 0.0, 1)

    def test_log_step_empty_states(self):
        """log_step with empty states should not raise."""
        self.mock_session.add = Mock()
        self.logger.log_step(
            step_number=1,
            agent_states=[],
            resource_states=[],
            metrics={"total_agents": 0},
        )
        self.mock_session.add.assert_called_once()

    def test_log_step_with_agent_states(self):
        self.mock_session.add = Mock()
        self.mock_session.bulk_insert_mappings = Mock()
        state = self._make_agent_state_tuple()
        self.logger.log_step(
            step_number=1,
            agent_states=[state],
            resource_states=[],
            metrics={"total_agents": 1},
        )
        self.mock_session.add.assert_called_once()
        self.mock_session.bulk_insert_mappings.assert_called()

    def test_log_step_with_resource_states(self):
        self.mock_session.add = Mock()
        self.mock_session.bulk_insert_mappings = Mock()
        # resource state: (resource_id, amount, pos_x, pos_y)
        resource_state = ("res_1", 100.0, 3.0, 4.0)
        self.logger.log_step(
            step_number=2,
            agent_states=[],
            resource_states=[resource_state],
            metrics={"total_agents": 0},
        )
        self.mock_session.bulk_insert_mappings.assert_called()

    def test_log_resource(self):
        self.mock_session.bulk_insert_mappings = Mock()
        self.logger.log_resource(
            resource_id="res_1", initial_amount=100.0, position=(5.0, 5.0)
        )
        self.mock_session.bulk_insert_mappings.assert_called_once()


class TestDataLoggerLogAgentsBatch(unittest.TestCase):
    """Tests for DataLogger.log_agents_batch method."""

    def setUp(self):
        self.mock_session = Mock()
        self.db = Mock()

        def execute_in_transaction(func):
            return func(self.mock_session)

        self.db._execute_in_transaction = execute_in_transaction
        self.logger = DataLogger(self.db, simulation_id="sim_001")

    def _make_agent_dict(self, idx: int) -> dict:
        return {
            "simulation_id": "sim_001",
            "agent_id": f"agent_{idx}",
            "birth_time": 0,
            "agent_type": "SystemAgent",
            "position": (float(idx), float(idx)),
            "initial_resources": 100.0,
            "starting_health": 50.0,
        }

    def test_log_agents_batch(self):
        agent_data_list = [self._make_agent_dict(i) for i in range(3)]
        self.mock_session.bulk_insert_mappings = Mock()
        self.logger.log_agents_batch(agent_data_list)
        self.mock_session.bulk_insert_mappings.assert_called_once()

    def test_log_agents_batch_empty_list_still_calls_bulk_insert(self):
        """Empty batch still calls bulk_insert_mappings (with empty list)."""
        self.mock_session.bulk_insert_mappings = Mock()
        self.logger.log_agents_batch([])
        self.mock_session.bulk_insert_mappings.assert_called_once()

    def test_log_agents_batch_missing_field_raises(self):
        bad_data = [{"agent_id": "a1"}]  # missing required fields
        self.mock_session.bulk_insert_mappings = Mock()
        with self.assertRaises((ValueError, KeyError)):
            self.logger.log_agents_batch(bad_data)


class TestShardedDataLogger(unittest.TestCase):
    """Tests for ShardedDataLogger routing behaviour."""

    def _make_shard_logger(self):
        shard_logger = Mock()
        shard_logger.log_agent_states = Mock()
        shard_logger.log_resources = Mock()
        shard_logger.log_metrics = Mock()
        shard_logger.log_agent_action = Mock()
        shard_logger.flush_all_buffers = Mock()
        db_shard = Mock()
        db_shard.logger = shard_logger
        return db_shard

    def _make_sharded_db(self):
        sharded_db = Mock()
        shard = self._make_shard_logger()
        # ShardedDataLogger uses keys: "agents", "resources", "metrics", "actions"
        sharded_db.shards = {
            0: {
                "agents": shard,
                "resources": shard,
                "metrics": shard,
                "actions": shard,
            }
        }
        sharded_db._get_shard_for_step = Mock(return_value=0)
        return sharded_db, shard

    def test_log_agent_states_routes_to_shard(self):
        sharded_db, shard = self._make_sharded_db()
        logger = ShardedDataLogger(sharded_db, simulation_id="sim_001")
        logger.log_agent_states(step_number=1, agent_states=[])
        sharded_db._get_shard_for_step.assert_called_with(1)
        shard.logger.log_agent_states.assert_called_once()

    def test_log_resources_routes_to_shard(self):
        sharded_db, shard = self._make_sharded_db()
        logger = ShardedDataLogger(sharded_db, simulation_id="sim_001")
        logger.log_resources(step_number=2, resource_states=[])
        sharded_db._get_shard_for_step.assert_called_with(2)
        shard.logger.log_resources.assert_called_once()

    def test_log_metrics_routes_to_shard(self):
        sharded_db, shard = self._make_sharded_db()
        logger = ShardedDataLogger(sharded_db, simulation_id="sim_001")
        logger.log_metrics(step_number=3, metrics={"agents": 10})
        shard.logger.log_metrics.assert_called_once()

    def test_log_action_routes_to_shard(self):
        sharded_db, shard = self._make_sharded_db()
        logger = ShardedDataLogger(sharded_db, simulation_id="sim_001")
        action_data = {
            "step_number": 1,
            "agent_id": "a1",
            "action_type": "move",
        }
        logger.log_action(step_number=1, action_data=action_data)
        shard.logger.log_agent_action.assert_called_once()

    def test_flush_all_buffers_calls_all_shards(self):
        sharded_db, shard = self._make_sharded_db()
        sharded_db.shards = {
            0: {
                "agents": shard,
                "resources": shard,
                "metrics": shard,
                "actions": shard,
            },
            1: {
                "agents": shard,
                "resources": shard,
                "metrics": shard,
                "actions": shard,
            },
        }
        logger = ShardedDataLogger(sharded_db, simulation_id="sim_001")
        logger.flush_all_buffers()
        # With 2 shards × 4 shard types = 8 calls
        self.assertGreaterEqual(shard.logger.flush_all_buffers.call_count, 4)


if __name__ == "__main__":
    unittest.main()
