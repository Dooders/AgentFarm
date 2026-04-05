"""Tests for farm/memory/redis_memory.py.

Uses fakeredis so no real Redis connection is required.
Tests both AgentMemory and AgentMemoryManager with the
fakeredis backend.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")

from farm.memory.redis_memory import (
    AgentMemory,
    AgentMemoryManager,
    RedisMemoryConfig,
)


def _make_fake_client():
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def _make_state(position_x=1.0, position_y=2.0, health=100.0):
    """Return a state-like object with as_dict() for redis_memory serialization."""
    state_dict = {
        "position_x": position_x,
        "position_y": position_y,
        "current_health": health,
    }
    state = Mock()
    state.as_dict = Mock(return_value=state_dict)
    return state


# ---------------------------------------------------------------------------
# RedisMemoryConfig
# ---------------------------------------------------------------------------


class TestRedisMemoryConfig(unittest.TestCase):

    def test_default_values(self):
        cfg = RedisMemoryConfig()
        self.assertEqual(cfg.host, "localhost")
        self.assertEqual(cfg.port, 6379)
        self.assertEqual(cfg.db, 0)
        self.assertEqual(cfg.memory_limit, 1000)
        self.assertEqual(cfg.ttl, 3600)

    def test_connection_params_includes_decode_responses(self):
        cfg = RedisMemoryConfig()
        params = cfg.connection_params
        self.assertTrue(params.get("decode_responses"))

    def test_custom_host_port(self):
        cfg = RedisMemoryConfig(host="cache-server", port=6380)
        self.assertEqual(cfg.connection_params["host"], "cache-server")
        self.assertEqual(cfg.connection_params["port"], 6380)


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


class TestAgentMemoryInit(unittest.TestCase):

    def test_init_with_fake_redis_succeeds(self):
        fake = _make_fake_client()
        mem = AgentMemory(agent_id="a1", redis_client=fake)
        self.assertEqual(mem.agent_id, "a1")

    def test_init_failed_connection_raises(self):
        bad_client = Mock()
        bad_client.ping = Mock(side_effect=Exception("connection refused"))
        with self.assertRaises(Exception):
            AgentMemory(agent_id="a1", redis_client=bad_client)


class TestAgentMemoryRememberAndRetrieve(unittest.TestCase):

    def setUp(self):
        self.fake = _make_fake_client()
        self.mem = AgentMemory(agent_id="agent_test", redis_client=self.fake)

    def test_remember_state_returns_true(self):
        ok = self.mem.remember_state(step=1, state=_make_state())
        self.assertTrue(ok)

    def test_retrieve_state_returns_stored_data(self):
        state = _make_state(health=75.0)
        self.mem.remember_state(step=5, state=state)
        result = self.mem.retrieve_state(5)
        self.assertIsNotNone(result)
        self.assertEqual(result["step"], 5)
        self.assertAlmostEqual(result["state"]["current_health"], 75.0)

    def test_retrieve_missing_step_returns_none(self):
        self.assertIsNone(self.mem.retrieve_state(999))

    def test_remember_with_action_and_reward(self):
        self.mem.remember_state(step=10, state=_make_state(), action="eat", reward=3.0)
        result = self.mem.retrieve_state(10)
        self.assertEqual(result["action"], "eat")
        self.assertAlmostEqual(result["reward"], 3.0)

    def test_remember_with_metadata(self):
        self.mem.remember_state(
            step=7, state=_make_state(), metadata={"zone": "forest"}
        )
        result = self.mem.retrieve_state(7)
        self.assertEqual(result["metadata"]["zone"], "forest")

    def test_remember_state_failure_returns_false(self):
        """If Redis is broken the method should return False instead of raising."""
        bad_client = Mock()
        bad_client.ping = Mock(return_value=True)
        bad_client.zadd = Mock(side_effect=Exception("redis error"))
        mem = AgentMemory(agent_id="a_broken", redis_client=bad_client)
        ok = mem.remember_state(step=1, state=_make_state())
        self.assertFalse(ok)


class TestAgentMemoryRetrieveMethods(unittest.TestCase):

    def setUp(self):
        self.fake = _make_fake_client()
        self.mem = AgentMemory(agent_id="agent_r", redis_client=self.fake)
        state = _make_state()
        for step in range(10):
            self.mem.remember_state(step=step, state=state, action=f"action_{step % 3}")

    def test_retrieve_recent_states(self):
        recent = self.mem.retrieve_recent_states(count=3)
        self.assertEqual(len(recent), 3)

    def test_retrieve_recent_states_more_than_stored(self):
        recent = self.mem.retrieve_recent_states(count=100)
        self.assertEqual(len(recent), 10)

    def test_retrieve_states_by_timeframe(self):
        states = self.mem.retrieve_states_by_timeframe(2, 5)
        self.assertEqual(len(states), 4)

    def test_retrieve_states_empty_range(self):
        states = self.mem.retrieve_states_by_timeframe(100, 200)
        self.assertEqual(states, [])

    def test_retrieve_action_history(self):
        history = self.mem.retrieve_action_history(limit=5)
        self.assertEqual(len(history), 5)

    def test_get_action_frequency(self):
        freq = self.mem.get_action_frequency()
        total = sum(freq.values())
        self.assertEqual(total, 10)


class TestAgentMemorySearch(unittest.TestCase):

    def setUp(self):
        self.fake = _make_fake_client()
        self.mem = AgentMemory(agent_id="agent_s", redis_client=self.fake)

    def test_search_by_metadata_match(self):
        state = _make_state()
        self.mem.remember_state(step=1, state=state, metadata={"zone": "alpha"})
        self.mem.remember_state(step=2, state=state, metadata={"zone": "beta"})
        results = self.mem.search_by_metadata("zone", "alpha")
        self.assertEqual(len(results), 1)

    def test_search_by_metadata_no_match(self):
        state = _make_state()
        self.mem.remember_state(step=1, state=state, metadata={"zone": "alpha"})
        results = self.mem.search_by_metadata("zone", "gamma")
        self.assertEqual(results, [])

    def test_search_by_state_value(self):
        self.mem.remember_state(step=1, state=_make_state(health=100.0))
        self.mem.remember_state(step=2, state=_make_state(health=50.0))
        results = self.mem.search_by_state_value("current_health", 100.0)
        self.assertEqual(len(results), 1)

    def test_search_by_position(self):
        self.mem.remember_state(step=1, state=_make_state(position_x=5.0, position_y=5.0))
        self.mem.remember_state(step=2, state=_make_state(position_x=100.0, position_y=100.0))
        results = self.mem.search_by_position((5.0, 5.0), radius=1.0)
        self.assertEqual(len(results), 1)

    def test_search_by_position_no_match(self):
        self.mem.remember_state(step=1, state=_make_state(position_x=5.0, position_y=5.0))
        results = self.mem.search_by_position((50.0, 50.0), radius=0.1)
        self.assertEqual(results, [])


class TestAgentMemoryClearAndCleanup(unittest.TestCase):

    def setUp(self):
        self.fake = _make_fake_client()
        self.mem = AgentMemory(agent_id="agent_cc", redis_client=self.fake)

    def test_clear_memory(self):
        self.mem.remember_state(step=1, state=_make_state())
        ok = self.mem.clear_memory()
        self.assertTrue(ok)
        self.assertIsNone(self.mem.retrieve_state(1))

    def test_cleanup_trims_to_memory_limit(self):
        config = RedisMemoryConfig(memory_limit=3, cleanup_interval=1)
        fake = _make_fake_client()
        mem = AgentMemory(agent_id="trim_agent", config=config, redis_client=fake)
        state = _make_state()
        for i in range(6):
            mem.remember_state(step=i, state=state)
        timeline_key = f"{mem._agent_key_prefix}:timeline"
        count = fake.zcard(timeline_key)
        self.assertLessEqual(count, 3)


# ---------------------------------------------------------------------------
# AgentMemoryManager
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgentMemoryManager(unittest.TestCase):

    def tearDown(self):
        AgentMemoryManager._instance = None

    def test_singleton_get_instance(self):
        fake = _make_fake_client()
        with patch("farm.memory.redis_memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake
            mgr1 = AgentMemoryManager.get_instance()
            mgr2 = AgentMemoryManager.get_instance()
            self.assertIs(mgr1, mgr2)

    def test_get_memory_creates_agent_memory(self):
        fake = _make_fake_client()
        with patch("farm.memory.redis_memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake
            mgr = AgentMemoryManager.get_instance()
            mem = mgr.get_memory("agent_A")
            self.assertIsNotNone(mem)
            self.assertEqual(mem.agent_id, "agent_A")

    def test_get_memory_caches_instance(self):
        fake = _make_fake_client()
        with patch("farm.memory.redis_memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake
            mgr = AgentMemoryManager.get_instance()
            m1 = mgr.get_memory("agent_B")
            m2 = mgr.get_memory("agent_B")
            self.assertIs(m1, m2)

    def test_clear_all_memories(self):
        fake = _make_fake_client()
        with patch("farm.memory.redis_memory.redis") as mock_redis:
            mock_redis.Redis.return_value = fake
            mgr = AgentMemoryManager.get_instance()
            mem = mgr.get_memory("agent_C")
            state = _make_state()
            mem.remember_state(step=1, state=state)
            ok = mgr.clear_all_memories()
            self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
