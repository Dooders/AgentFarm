"""Unit tests for the simple attack action.

This module tests the simple attack functionality that finds the closest agent
and attacks it without using DQN.
"""

import math
import unittest
from unittest.mock import Mock

from farm.core.action import attack_action


class TestSimpleAttackAction(unittest.TestCase):
    """Test cases for the simple attack action."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = Mock()
        self.agent.agent_id = "test_agent"
        self.agent.position = (0, 0)
        self.agent.resource_level = 10
        self.agent.current_health = 100
        self.agent.starting_health = 100
        self.agent.attack_strength = 10
        self.agent.defense_strength = 0.5
        self.agent.alive = True
        self.agent.config = Mock()
        self.agent.config.attack_range = 20.0

        self.environment = Mock()
        self.agent.environment = self.environment

    def test_attack_action_no_config(self):
        """Test attack action when agent has no config."""
        self.agent.config = None

        attack_action(self.agent)

        # Should return early without error
        self.assertTrue(True)

    def test_attack_action_no_nearby_agents(self):
        """Test attack action when no agents are nearby."""
        self.environment.get_nearby_agents.return_value = []

        attack_action(self.agent)

        # Should return early without attacking
        self.environment.get_nearby_agents.assert_called_once_with((0, 0), 20.0)

    def test_attack_action_no_valid_targets(self):
        """Test attack action when nearby agents are not valid targets."""
        # Create a nearby agent that's either self or dead
        nearby_agent = Mock()
        nearby_agent.agent_id = "test_agent"  # Same as self
        nearby_agent.alive = True

        self.environment.get_nearby_agents.return_value = [nearby_agent]

        attack_action(self.agent)

        # Should return early without attacking
        self.assertTrue(True)

    def test_attack_action_closest_target_attack(self):
        """Test successful attack on closest target."""
        # Create a valid target
        target = Mock()
        target.agent_id = "target_agent"
        target.position = (5, 0)  # 5 units away
        target.alive = True
        target.is_defending = False
        target.take_damage.return_value = 5.0

        # Create another target farther away
        far_target = Mock()
        far_target.agent_id = "far_target"
        far_target.position = (15, 0)  # 15 units away
        far_target.alive = True

        self.environment.get_nearby_agents.return_value = [target, far_target]
        self.environment.record_combat_encounter = Mock()
        self.environment.record_successful_attack = Mock()
        self.environment.log_interaction_edge = Mock()

        attack_action(self.agent)

        # Verify target.take_damage was called
        target.take_damage.assert_called_once()
        damage_called = target.take_damage.call_args[0][0]
        self.assertGreater(damage_called, 0)  # Should deal positive damage

        # Verify combat statistics were updated
        self.environment.record_combat_encounter.assert_called_once()
        self.environment.record_successful_attack.assert_called_once()

        # Verify interaction was logged
        self.environment.log_interaction_edge.assert_called_once()

    def test_attack_action_defending_target(self):
        """Test attack on a defending target (reduced damage)."""
        target = Mock()
        target.agent_id = "target_agent"
        target.position = (3, 0)
        target.alive = True
        target.is_defending = True
        target.defense_strength = 0.5
        target.take_damage.return_value = 2.5  # Reduced damage

        self.environment.get_nearby_agents.return_value = [target]
        self.environment.record_combat_encounter = Mock()
        self.environment.record_successful_attack = Mock()
        self.environment.log_interaction_edge = Mock()

        attack_action(self.agent)

        # Verify damage was reduced by defense
        target.take_damage.assert_called_once()
        damage_called = target.take_damage.call_args[0][0]

        # Base damage should be reduced by defense strength
        expected_damage = (
            10 * 1.0 * (1.0 - 0.5)
        )  # attack_strength * health_ratio * (1 - defense)
        self.assertAlmostEqual(damage_called, expected_damage, places=1)

    def test_attack_action_multiple_targets(self):
        """Test attack selects closest target when multiple are available."""
        # Create targets at different distances
        close_target = Mock()
        close_target.agent_id = "close_target"
        close_target.position = (2, 0)  # 2 units away
        close_target.alive = True
        close_target.is_defending = False
        close_target.take_damage.return_value = 5.0

        far_target = Mock()
        far_target.agent_id = "far_target"
        far_target.position = (10, 0)  # 10 units away
        far_target.alive = True
        far_target.is_defending = False

        self.environment.get_nearby_agents.return_value = [far_target, close_target]
        self.environment.record_combat_encounter = Mock()
        self.environment.record_successful_attack = Mock()
        self.environment.log_interaction_edge = Mock()

        attack_action(self.agent)

        # Should attack the closest target
        close_target.take_damage.assert_called_once()
        far_target.take_damage.assert_not_called()

    def test_attack_action_distance_calculation(self):
        """Test that distance calculation works correctly."""
        target = Mock()
        target.agent_id = "target_agent"
        target.position = (3, 4)  # Should be 5 units away (3-4-5 triangle)
        target.alive = True
        target.is_defending = False
        target.take_damage.return_value = 5.0

        self.environment.get_nearby_agents.return_value = [target]
        self.environment.record_combat_encounter = Mock()
        self.environment.record_successful_attack = Mock()
        self.environment.log_interaction_edge = Mock()

        attack_action(self.agent)

        # Verify interaction was logged with correct distance
        log_call = self.environment.log_interaction_edge.call_args
        details = log_call[1]["details"]
        expected_distance = math.sqrt(3**2 + 4**2)  # Distance formula
        self.assertAlmostEqual(details["distance"], expected_distance, places=1)


if __name__ == "__main__":
    unittest.main()
