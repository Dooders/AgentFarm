#!/usr/bin/env python3
"""Comprehensive test suite for object pooling functionality"""

import os
import sys
import unittest
sys.path.insert(0, '/workspace')

# Enable pooling
os.environ["FARM_DISABLE_POOLING"] = "0"

# Mock agent class for testing
class MockAgent:
    def __init__(self, agent_id, position, resource_level):
        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.starvation_counter = 0
        self.starvation_threshold = 3
        self.episode_rewards = [1, 2, 3]  # Test data
        self.losses = [0.1, 0.2, 0.3]    # Test data
        self.total_reward = 10.5
        self.current_health = 80
        self.starting_health = 100
        self.is_defending = True
        self.defense_timer = 2
        self.orientation = 45.0
        self._cached_selection_state = "test_cache"
        self._cached_selection_time = 123
        self.previous_state = "test_state"
        self.previous_state_tensor = "test_tensor"
        self.previous_action = "test_action"
        self._previous_action_index = 5
        self._previous_enabled_actions = ["action1", "action2"]

    def reset(self, agent_id=None, position=None, resource_level=None, **kwargs):
        if agent_id is not None:
            self.agent_id = agent_id
        if position is not None:
            self.position = position
        if resource_level is not None:
            self.resource_level = resource_level
        self.alive = True
        self.starvation_counter = 0

    def prepare_for_release(self):
        """Prepare this agent to be returned to an object pool."""
        # Clear cached selection state
        if hasattr(self, "_cached_selection_state"):
            self._cached_selection_state = None
            self._cached_selection_time = -1
        # Reset per-episode trackers
        self.previous_state = None
        if hasattr(self, "previous_state_tensor"):
            self.previous_state_tensor = None
        self.previous_action = None
        self._previous_action_index = 0
        self._previous_enabled_actions = None

        # Clear episode-specific data
        self.episode_rewards.clear()
        self.losses.clear()

        # Reset agent state to minimize memory retention
        self.starvation_counter = 0
        self.total_reward = 0.0
        self.current_health = self.starting_health  # Reset to full health
        self.is_defending = False
        self.defense_timer = 0
        self.orientation = 0.0

    def check_starvation(self):
        """Check if agent should die from starvation"""
        if self.resource_level <= 0:
            self.starvation_counter += 1
            if self.starvation_counter >= self.starvation_threshold:
                self.alive = False
                return True
        else:
            self.starvation_counter = 0
        return False

    def act(self):
        """Simulate agent action - consume resources"""
        if not self.alive:
            return

        # Consume resources each turn
        self.resource_level -= 1

        # Check if agent should die from starvation
        if self.check_starvation():
            print(f"  Agent {self.agent_id} died from starvation")


class TestAgentPool(unittest.TestCase):
    def setUp(self):
        from farm.core.pool import AgentPool
        self.pool = AgentPool(MockAgent, max_size=10)

    def test_pool_creation(self):
        """Test that pool is created correctly"""
        self.assertEqual(self.pool.total_created, 0)
        self.assertEqual(self.pool.total_reused, 0)
        self.assertEqual(self.pool.size(), 0)

    def test_agent_acquire(self):
        """Test acquiring agents from pool"""
        agent = self.pool.acquire(
            agent_id="test_1",
            position=(1, 1),
            resource_level=5
        )

        self.assertEqual(self.pool.total_created, 1)
        self.assertEqual(self.pool.total_reused, 0)
        self.assertEqual(agent.agent_id, "test_1")
        self.assertEqual(agent.position, (1, 1))
        self.assertEqual(agent.resource_level, 5)

    def test_agent_release(self):
        """Test releasing agents back to pool"""
        agent = self.pool.acquire(
            agent_id="test_1",
            position=(1, 1),
            resource_level=5
        )

        # Verify agent was created
        self.assertEqual(self.pool.total_created, 1)
        self.assertEqual(self.pool.size(), 0)

        # Release agent
        self.pool.release(agent)

        # Verify agent was released
        self.assertEqual(self.pool.total_created, 1)
        self.assertEqual(self.pool.size(), 1)

    def test_agent_reuse(self):
        """Test that released agents are reused"""
        # Acquire first agent
        agent1 = self.pool.acquire(
            agent_id="test_1",
            position=(1, 1),
            resource_level=5
        )

        # Release it
        self.pool.release(agent1)

        # Acquire second agent (should reuse first)
        agent2 = self.pool.acquire(
            agent_id="test_2",
            position=(2, 2),
            resource_level=10
        )

        # Verify reuse occurred
        self.assertEqual(self.pool.total_created, 1)
        self.assertEqual(self.pool.total_reused, 1)
        self.assertEqual(self.pool.size(), 0)  # Pool should be empty after reuse
        self.assertIs(agent1, agent2)  # Should be the same object

    def test_prepare_for_release(self):
        """Test that prepare_for_release properly cleans up agent state"""
        agent = self.pool.acquire(
            agent_id="test_1",
            position=(1, 1),
            resource_level=5
        )

        # Set some test data
        agent.episode_rewards = [1, 2, 3, 4, 5]
        agent.losses = [0.1, 0.2, 0.3]
        agent.total_reward = 123.45
        agent.current_health = 50
        agent.is_defending = True
        agent.defense_timer = 5
        agent.orientation = 90.0
        agent._cached_selection_state = "test_cache"
        agent.previous_state = "test_state"

        # Prepare for release
        agent.prepare_for_release()

        # Verify cleanup
        self.assertEqual(agent.episode_rewards, [])
        self.assertEqual(agent.losses, [])
        self.assertEqual(agent.total_reward, 0.0)
        self.assertEqual(agent.current_health, agent.starting_health)
        self.assertFalse(agent.is_defending)
        self.assertEqual(agent.defense_timer, 0)
        self.assertEqual(agent.orientation, 0.0)
        self.assertIsNone(agent._cached_selection_state)
        self.assertIsNone(agent.previous_state)

    def test_pool_max_size(self):
        """Test that pool respects maximum size"""
        # Create pool with max size 3
        from farm.core.pool import AgentPool
        small_pool = AgentPool(MockAgent, max_size=3)

        # Create 5 agents
        agents = []
        for i in range(5):
            agent = small_pool.acquire(
                agent_id=f"test_{i}",
                position=(i, i),
                resource_level=5
            )
            agents.append(agent)

        # Release all 5 agents
        for agent in agents:
            small_pool.release(agent)

        # Pool should only keep 3 agents (max size)
        self.assertEqual(small_pool.size(), 3)
        self.assertEqual(small_pool.total_created, 5)

    def test_reset_method(self):
        """Test that reset method works correctly"""
        agent = self.pool.acquire(
            agent_id="test_1",
            position=(1, 1),
            resource_level=5
        )

        # Modify agent
        agent.agent_id = "modified"
        agent.position = (10, 10)
        agent.resource_level = 100

        # Reset agent
        agent.reset(
            agent_id="reset_test",
            position=(20, 20),
            resource_level=25
        )

        # Verify reset worked
        self.assertEqual(agent.agent_id, "reset_test")
        self.assertEqual(agent.position, (20, 20))
        self.assertEqual(agent.resource_level, 25)


class TestAgentLifecycle(unittest.TestCase):
    def setUp(self):
        from farm.core.pool import AgentPool
        self.pool = AgentPool(MockAgent, max_size=10)
        self.agents = []
        self.dead_agents = []

    def test_starvation_death(self):
        """Test that agents die from starvation and get released"""
        # Create agent with low resources
        agent = self.pool.acquire(
            agent_id="starve_test",
            position=(1, 1),
            resource_level=2
        )

        # Simulate starvation
        for i in range(5):
            agent.act()

        # Agent should die
        self.assertFalse(agent.alive)

        # Release to pool
        self.pool.release(agent)
        self.assertEqual(self.pool.size(), 1)

    def test_reuse_after_death(self):
        """Test full lifecycle: create -> death -> release -> reuse"""
        # Create agent
        agent1 = self.pool.acquire(
            agent_id="lifecycle_test",
            position=(1, 1),
            resource_level=2
        )

        # Kill agent through starvation
        for i in range(5):
            agent1.act()

        self.assertFalse(agent1.alive)

        # Release to pool
        self.pool.release(agent1)
        self.assertEqual(self.pool.size(), 1)

        # Create new agent (should reuse old one)
        agent2 = self.pool.acquire(
            agent_id="reused_test",
            position=(2, 2),
            resource_level=5
        )

        # Verify reuse
        self.assertEqual(self.pool.total_created, 1)
        self.assertEqual(self.pool.total_reused, 1)
        self.assertIs(agent1, agent2)  # Same object

        # Verify reset worked
        self.assertEqual(agent2.agent_id, "reused_test")
        self.assertEqual(agent2.position, (2, 2))
        self.assertEqual(agent2.resource_level, 5)
        self.assertTrue(agent2.alive)


if __name__ == "__main__":
    print("Running comprehensive pooling tests...")
    unittest.main(verbosity=2)