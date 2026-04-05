"""Tests for farm/database/memory.py (Redis-backed agent memory).

The module defines AgentMemory and AgentMemoryManager classes that use Redis.
These tests use fakeredis to avoid a real Redis dependency.

Marked with ``integration`` so they are excluded from the fast default
pytest run (-m "not slow and not integration").  If fakeredis is not
installed the tests are skipped gracefully.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

# Skip the entire module if fakeredis is not installed
fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")

from farm.database.memory import AgentMemory, AgentMemoryManager, RedisMemoryConfig


def _make_fake_client():
    """Return a fakeredis server client with decode_responses=True."""
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def _make_state(position_x=5.0, position_y=10.0, health=100.0):
    """Return a state-like object with to_dict() for farm/database/memory serialization."""
    state_dict = {
        "position_x": position_x,
        "position_y": position_y,
        "current_health": health,
    }
    state = Mock()
    state.to_dict = Mock(return_value=state_dict)
    return state


class TestRedisMemoryConfigViaMemoryModule(unittest.TestCase):
    """Verify config defaults are accessible via farm.database.memory."""

    def test_default_config(self):
        config = RedisMemoryConfig()
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 6379)
        self.assertEqual(config.memory_limit, 1000)

    def test_connection_params_property(self):
        config = RedisMemoryConfig(host="redis-host", port=6380)
        params = config.connection_params
        self.assertEqual(params["host"], "redis-host")
        self.assertEqual(params["port"], 6380)


class TestAgentMemoryWithFakeRedis(unittest.TestCase):
    """Functional tests for AgentMemory using fakeredis."""

    def setUp(self):
        self.fake_client = _make_fake_client()
        self.memory = AgentMemory(
            agent_id="agent_001",
            redis_client=self.fake_client,
        )

    def test_remember_and_retrieve_state(self):
        state = _make_state()
        ok = self.memory.remember_state(step=1, state=state)
        self.assertTrue(ok)

        retrieved = self.memory.retrieve_state(1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["step"], 1)

    def test_retrieve_nonexistent_state_returns_none(self):
        result = self.memory.retrieve_state(9999)
        self.assertIsNone(result)

    def test_retrieve_recent_states(self):
        state = _make_state()
        for step in range(5):
            self.memory.remember_state(step=step, state=state)

        recent = self.memory.retrieve_recent_states(count=3)
        self.assertEqual(len(recent), 3)

    def test_retrieve_states_by_timeframe(self):
        state = _make_state()
        for step in range(10):
            self.memory.remember_state(step=step, state=state)

        states = self.memory.retrieve_states_by_timeframe(3, 7)
        self.assertEqual(len(states), 5)

    def test_remember_state_with_action_and_reward(self):
        state = _make_state()
        ok = self.memory.remember_state(
            step=5,
            state=state,
            action="move",
            reward=2.5,
        )
        self.assertTrue(ok)
        retrieved = self.memory.retrieve_state(5)
        self.assertEqual(retrieved["action"], "move")
        self.assertAlmostEqual(retrieved["reward"], 2.5)

    def test_remember_state_with_metadata(self):
        state = _make_state()
        self.memory.remember_state(
            step=3,
            state=state,
            metadata={"zone": "combat"},
        )
        result = self.memory.retrieve_state(3)
        self.assertEqual(result["metadata"]["zone"], "combat")

    def test_search_by_metadata(self):
        state = _make_state()
        self.memory.remember_state(step=1, state=state, metadata={"zone": "safe"})
        self.memory.remember_state(step=2, state=state, metadata={"zone": "danger"})

        results = self.memory.search_by_metadata("zone", "safe")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["zone"], "safe")

    def test_search_by_state_value(self):
        for i in range(3):
            state = _make_state(health=float(50 + i * 10))
            self.memory.remember_state(step=i, state=state)

        results = self.memory.search_by_state_value("current_health", 60.0)
        self.assertGreaterEqual(len(results), 1)

    def test_search_by_position(self):
        state = _make_state(position_x=5.0, position_y=5.0)
        self.memory.remember_state(step=1, state=state)
        far_state = _make_state(position_x=100.0, position_y=100.0)
        self.memory.remember_state(step=2, state=far_state)

        results = self.memory.search_by_position((5.0, 5.0), radius=1.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["state"]["position_x"], 5.0)

    def test_retrieve_action_history(self):
        state = _make_state()
        for i, action in enumerate(["move", "eat", "attack", "move"]):
            self.memory.remember_state(step=i, state=state, action=action)

        history = self.memory.retrieve_action_history(limit=4)
        self.assertIn("move", history)
        self.assertIn("eat", history)

    def test_get_action_frequency(self):
        state = _make_state()
        for i, action in enumerate(["move", "move", "eat"]):
            self.memory.remember_state(step=i, state=state, action=action)

        freq = self.memory.get_action_frequency()
        self.assertEqual(freq.get("move", 0), 2)
        self.assertEqual(freq.get("eat", 0), 1)

    def test_clear_memory(self):
        state = _make_state()
        self.memory.remember_state(step=1, state=state)
        ok = self.memory.clear_memory()
        self.assertTrue(ok)
        self.assertIsNone(self.memory.retrieve_state(1))

    def test_cleanup_old_memories_when_limit_exceeded(self):
        """Memory cleanup should keep count at or below memory_limit."""
        config = RedisMemoryConfig(memory_limit=3, cleanup_interval=1)
        fake_client = _make_fake_client()
        mem = AgentMemory(agent_id="a1", config=config, redis_client=fake_client)
        state = _make_state()

        # Store 5 states – cleanup should trim to 3
        for i in range(5):
            mem.remember_state(step=i, state=state)

        timeline_key = f"{mem._agent_key_prefix}:timeline"
        count = fake_client.zcard(timeline_key)
        self.assertLessEqual(count, 3)


@pytest.mark.integration
class TestAgentMemoryManagerWithFakeRedis(unittest.TestCase):
    """Tests for AgentMemoryManager singleton pattern (integration marker)."""

    def tearDown(self):
        # Reset singleton state so tests don't bleed into each other
        AgentMemoryManager._instance = None

    def test_get_or_create_memory(self):
        fake_client = _make_fake_client()
        with patch("farm.database.memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake_client
            manager = AgentMemoryManager()
            mem = manager.get_memory("agent_001")
            self.assertIsNotNone(mem)
            self.assertEqual(mem.agent_id, "agent_001")

    def test_same_agent_id_returns_same_instance(self):
        fake_client = _make_fake_client()
        with patch("farm.database.memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake_client
            manager = AgentMemoryManager()
            m1 = manager.get_memory("agent_A")
            m2 = manager.get_memory("agent_A")
            self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
