"""Tests for farm/loggers/attack_logger.py."""
import unittest
from unittest.mock import MagicMock

from farm.loggers.attack_logger import AttackLogger


def _make_mock_agent(agent_id="agent_1", current_health=80.0, starting_health=100.0):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.current_health = current_health
    agent.starting_health = starting_health
    return agent


class TestAttackLoggerNoDb(unittest.TestCase):
    """AttackLogger with no database should be a no-op."""

    def setUp(self):
        self.logger = AttackLogger(db=None)

    def test_log_defense_no_db_returns_none(self):
        agent = _make_mock_agent()
        result = self.logger.log_defense(step_number=1, agent=agent)
        self.assertIsNone(result)

    def test_log_attack_attempt_no_db_returns_none(self):
        agent = _make_mock_agent()
        result = self.logger.log_attack_attempt(
            step_number=1,
            agent=agent,
            action_target_id="target_1",
            target_position=(10.0, 20.0),
            success=True,
        )
        self.assertIsNone(result)


class TestAttackLoggerWithDb(unittest.TestCase):
    """AttackLogger with a mocked database should call the logger methods."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.logger = MagicMock()
        self.logger = AttackLogger(db=self.mock_db)

    def test_log_defense_calls_db(self):
        agent = _make_mock_agent(agent_id="defender_1", current_health=60.0, starting_health=100.0)
        self.logger.log_defense(step_number=5, agent=agent)

        self.mock_db.logger.log_agent_action.assert_called_once()
        call_kwargs = self.mock_db.logger.log_agent_action.call_args[1]
        self.assertEqual(call_kwargs["step_number"], 5)
        self.assertEqual(call_kwargs["agent_id"], "defender_1")
        self.assertEqual(call_kwargs["action_type"], "defend")
        self.assertEqual(call_kwargs["reward"], 0)
        self.assertIn("is_defending", call_kwargs["details"])
        self.assertTrue(call_kwargs["details"]["is_defending"])
        self.assertAlmostEqual(call_kwargs["details"]["health_ratio"], 0.6)

    def test_log_attack_attempt_success(self):
        agent = _make_mock_agent(agent_id="attacker_1", current_health=90.0, starting_health=100.0)
        self.logger.log_attack_attempt(
            step_number=10,
            agent=agent,
            action_target_id="target_2",
            target_position=(3.0, 4.0),
            success=True,
            targets_found=1,
            damage_dealt=25.0,
        )

        self.mock_db.logger.log_agent_action.assert_called_once()
        call_kwargs = self.mock_db.logger.log_agent_action.call_args[1]
        self.assertEqual(call_kwargs["step_number"], 10)
        self.assertEqual(call_kwargs["agent_id"], "attacker_1")
        self.assertEqual(call_kwargs["action_type"], "attack")
        self.assertEqual(call_kwargs["action_target_id"], "target_2")
        self.assertEqual(call_kwargs["reward"], 0)
        details = call_kwargs["details"]
        self.assertTrue(details["success"])
        self.assertEqual(details["target_position"], (3.0, 4.0))
        self.assertEqual(details["targets_found"], 1)
        self.assertEqual(details["damage_dealt"], 25.0)
        self.assertAlmostEqual(details["health_ratio"], 0.9)
        self.assertNotIn("reason", details)

    def test_log_attack_attempt_failure_with_reason(self):
        agent = _make_mock_agent()
        self.logger.log_attack_attempt(
            step_number=3,
            agent=agent,
            action_target_id=None,
            target_position=(0.0, 0.0),
            success=False,
            reason="no_targets",
        )

        call_kwargs = self.mock_db.logger.log_agent_action.call_args[1]
        details = call_kwargs["details"]
        self.assertFalse(details["success"])
        self.assertEqual(details["reason"], "no_targets")

    def test_log_attack_attempt_default_values(self):
        agent = _make_mock_agent()
        self.logger.log_attack_attempt(
            step_number=1,
            agent=agent,
            action_target_id=None,
            target_position=(1.0, 2.0),
            success=False,
        )
        details = self.mock_db.logger.log_agent_action.call_args[1]["details"]
        self.assertEqual(details["targets_found"], 0)
        self.assertEqual(details["damage_dealt"], 0.0)

    def test_log_attack_without_reason_not_in_details(self):
        agent = _make_mock_agent()
        self.logger.log_attack_attempt(
            step_number=1,
            agent=agent,
            action_target_id="t1",
            target_position=(1.0, 1.0),
            success=True,
            reason=None,
        )
        details = self.mock_db.logger.log_agent_action.call_args[1]["details"]
        self.assertNotIn("reason", details)


class TestAttackLoggerInit(unittest.TestCase):
    def test_init_with_none_db(self):
        logger = AttackLogger()
        self.assertIsNone(logger.db)

    def test_init_with_db(self):
        mock_db = MagicMock()
        logger = AttackLogger(db=mock_db)
        self.assertIs(logger.db, mock_db)
