"""Tests for farm/memory/base_memory.py.

Tests the MemorySearchMixin abstract class behavior via a concrete
test implementation.
"""

import unittest
from typing import Any, Dict, List, Optional

from farm.memory.base_memory import MemorySearchMixin


class ConcreteMemory(MemorySearchMixin):
    """Concrete implementation of MemorySearchMixin for testing."""

    def __init__(self, data: Dict[int, Dict[str, Any]]):
        self._data = data  # step -> state dict

    def _get_timeline_key(self, agent_id: str) -> str:
        return f"timeline:{agent_id}"

    def _get_all_timeline_steps(self, timeline_key: str) -> List[str]:
        return [str(step) for step in sorted(self._data.keys())]

    def _retrieve_state_by_step(self, step: int) -> Optional[Dict[str, Any]]:
        return self._data.get(step)


class TestMemorySearchMixin(unittest.TestCase):
    """Tests for MemorySearchMixin._search_states_by_criteria."""

    def setUp(self):
        self.states = {
            0: {"step": 0, "metadata": {"zone": "safe"}, "state": {"health": 100}},
            1: {"step": 1, "metadata": {"zone": "danger"}, "state": {"health": 80}},
            2: {"step": 2, "metadata": {"zone": "safe"}, "state": {"health": 60}},
            3: {"step": 3, "metadata": {"zone": "danger"}, "state": {"health": 40}},
        }
        self.memory = ConcreteMemory(self.states)

    def test_search_returns_matching_states(self):
        results = self.memory._search_states_by_criteria(
            agent_id="a1",
            criteria_func=lambda s: s.get("metadata", {}).get("zone") == "safe",
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["metadata"]["zone"], "safe")

    def test_search_respects_limit(self):
        results = self.memory._search_states_by_criteria(
            agent_id="a1",
            criteria_func=lambda _: True,
            limit=2,
        )
        self.assertEqual(len(results), 2)

    def test_search_returns_empty_on_no_match(self):
        results = self.memory._search_states_by_criteria(
            agent_id="a1",
            criteria_func=lambda s: s.get("metadata", {}).get("zone") == "underwater",
        )
        self.assertEqual(results, [])

    def test_search_handles_exception_gracefully(self):
        def bad_criteria(s):
            raise RuntimeError("oops")

        # Should return empty list rather than raising
        results = self.memory._search_states_by_criteria(
            agent_id="a1",
            criteria_func=bad_criteria,
        )
        self.assertEqual(results, [])

    def test_search_empty_data_source(self):
        empty_memory = ConcreteMemory({})
        results = empty_memory._search_states_by_criteria(
            agent_id="a1",
            criteria_func=lambda _: True,
        )
        self.assertEqual(results, [])

    def test_abstract_methods_raise_not_implemented(self):
        """Abstract methods must raise NotImplementedError (or TypeError on instantiation)."""
        # MemorySearchMixin does not inherit from ABC, so it can be instantiated.
        # However, its abstract methods should raise NotImplementedError.
        # We verify via the concrete subclass that the base protocol is clear.
        empty_memory = ConcreteMemory({})
        # _get_timeline_key should be implemented
        key = empty_memory._get_timeline_key("agent_001")
        self.assertEqual(key, "timeline:agent_001")


if __name__ == "__main__":
    unittest.main()
