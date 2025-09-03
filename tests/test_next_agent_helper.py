import unittest

from farm.core.environment import select_next_agent


class TestSelectNextAgentHelper(unittest.TestCase):
    def test_round_robin_basic(self):
        agents = ["a1", "a2", "a3"]
        terminations = {}
        truncations = {}

        # From a1 -> a2
        nxt, wrapped = select_next_agent("a1", agents, terminations, truncations)
        self.assertEqual(nxt, "a2")
        self.assertFalse(wrapped)

        # From a2 -> a3
        nxt, wrapped = select_next_agent("a2", agents, terminations, truncations)
        self.assertEqual(nxt, "a3")
        self.assertFalse(wrapped)

        # From a3 -> a1 (wrap)
        nxt, wrapped = select_next_agent("a3", agents, terminations, truncations)
        self.assertEqual(nxt, "a1")
        self.assertTrue(wrapped)

    def test_skips_inactive_agents(self):
        agents = ["a1", "a2", "a3", "a4"]
        terminations = {"a2": True}
        truncations = {"a3": True}

        # From a1, a2 is terminated and a3 is truncated, so next is a4
        nxt, wrapped = select_next_agent("a1", agents, terminations, truncations)
        self.assertEqual(nxt, "a4")
        self.assertFalse(wrapped)

        # From a4, wrap to a1 skipping inactive
        nxt, wrapped = select_next_agent("a4", agents, terminations, truncations)
        self.assertEqual(nxt, "a1")
        self.assertTrue(wrapped)

    def test_handles_no_agents_and_none_current(self):
        nxt, wrapped = select_next_agent(None, [], {}, {})
        self.assertIsNone(nxt)
        self.assertFalse(wrapped)

        # When current is None, select first available
        agents = ["a1", "a2"]
        nxt, wrapped = select_next_agent(None, agents, {}, {})
        self.assertEqual(nxt, "a1")
        self.assertFalse(wrapped)

    def test_all_inactive_returns_none(self):
        agents = ["a1", "a2"]
        terminations = {"a1": True}
        truncations = {"a2": True}
        nxt, wrapped = select_next_agent("a1", agents, terminations, truncations)
        self.assertIsNone(nxt)
        self.assertFalse(wrapped)


if __name__ == "__main__":
    unittest.main()

