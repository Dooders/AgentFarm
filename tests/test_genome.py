"""Tests for farm/core/genome.py (Genome utility class)."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from farm.core.genome import Genome, ACTION_FUNCTIONS


def _make_mock_genome():
    """Return a sample genome dict for testing."""
    return {
        "action_set": [
            ("attack", 0.2),
            ("gather", 0.3),
            ("share", 0.1),
            ("move", 0.2),
            ("reproduce", 0.1),
            ("defend", 0.05),
            ("pass", 0.05),
        ],
        "module_states": {"decision_module": {"weights": [0.1, 0.9]}},
        "agent_type": "AgentCore",
        "resource_level": 50.0,
        "current_health": 100.0,
    }


class TestActionFunctions(unittest.TestCase):
    def test_all_actions_registered(self):
        for name in ["attack", "gather", "share", "move", "reproduce", "defend", "pass"]:
            self.assertIn(name, ACTION_FUNCTIONS)
            self.assertTrue(callable(ACTION_FUNCTIONS[name]))


class TestGenomeSave(unittest.TestCase):
    def test_save_no_path_returns_json_string(self):
        genome = _make_mock_genome()
        result = Genome.save(genome)
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertEqual(parsed["agent_type"], "AgentCore")

    def test_save_with_path_writes_file(self):
        genome = _make_mock_genome()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            Genome.save(genome, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["agent_type"], "AgentCore")
        finally:
            os.unlink(path)

    def test_save_with_path_returns_none(self):
        genome = _make_mock_genome()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = Genome.save(genome, path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)


class TestGenomeLoad(unittest.TestCase):
    def test_load_from_dict(self):
        genome = _make_mock_genome()
        loaded = Genome.load(genome)
        self.assertIs(loaded, genome)

    def test_load_from_json_string(self):
        genome = _make_mock_genome()
        json_str = json.dumps(genome)
        loaded = Genome.load(json_str)
        self.assertEqual(loaded["agent_type"], genome["agent_type"])

    def test_load_from_file(self):
        genome = _make_mock_genome()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(genome, f)
            path = f.name
        try:
            loaded = Genome.load(path)
            self.assertEqual(loaded["agent_type"], genome["agent_type"])
            self.assertEqual(loaded["resource_level"], genome["resource_level"])
        finally:
            os.unlink(path)

    def test_load_from_json_array_string(self):
        data = [{"key": "value"}]
        json_str = json.dumps(data)
        loaded = Genome.load(json_str)
        self.assertEqual(loaded, data)


class TestGenomeMutate(unittest.TestCase):
    def test_mutate_preserves_keys(self):
        genome = _make_mock_genome()
        mutated = Genome.mutate(genome, mutation_rate=1.0)
        self.assertIn("action_set", mutated)
        self.assertIn("module_states", mutated)
        self.assertIn("agent_type", mutated)

    def test_mutate_weights_normalized(self):
        genome = _make_mock_genome()
        mutated = Genome.mutate(genome, mutation_rate=1.0)
        total = sum(w for _, w in mutated["action_set"])
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_mutate_does_not_modify_original(self):
        genome = _make_mock_genome()
        original_weights = [w for _, w in genome["action_set"]]
        Genome.mutate(genome, mutation_rate=1.0)
        for i, (_, w) in enumerate(genome["action_set"]):
            self.assertEqual(w, original_weights[i])

    def test_mutate_zero_rate_unchanged_weights(self):
        genome = _make_mock_genome()
        mutated = Genome.mutate(genome, mutation_rate=0.0)
        original_weights = [w for _, w in genome["action_set"]]
        mutated_weights = [w for _, w in mutated["action_set"]]
        # Weights still sum to 1 (they get renormalized even if unchanged)
        self.assertAlmostEqual(sum(mutated_weights), 1.0, places=10)
        for ow, mw in zip(original_weights, mutated_weights):
            self.assertAlmostEqual(ow, mw, places=10)

    def test_mutate_preserves_action_names(self):
        genome = _make_mock_genome()
        mutated = Genome.mutate(genome, mutation_rate=1.0)
        original_names = [n for n, _ in genome["action_set"]]
        mutated_names = [n for n, _ in mutated["action_set"]]
        self.assertEqual(original_names, mutated_names)


class TestGenomeCrossover(unittest.TestCase):
    def test_crossover_returns_dict(self):
        g1 = _make_mock_genome()
        g2 = _make_mock_genome()
        # Adjust g2 weights
        g2["action_set"] = [(n, 1.0 / len(g2["action_set"])) for n, _ in g2["action_set"]]
        child = Genome.crossover(g1, g2)
        self.assertIsInstance(child, dict)

    def test_crossover_has_action_set(self):
        g1 = _make_mock_genome()
        g2 = _make_mock_genome()
        child = Genome.crossover(g1, g2)
        self.assertIn("action_set", child)

    def test_crossover_with_mutation(self):
        g1 = _make_mock_genome()
        g2 = _make_mock_genome()
        child = Genome.crossover(g1, g2, mutation_rate=0.5)
        total = sum(w for _, w in child["action_set"])
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_crossover_preserves_other_fields(self):
        g1 = _make_mock_genome()
        g2 = _make_mock_genome()
        g2["agent_type"] = "OtherAgent"
        child = Genome.crossover(g1, g2)
        # Child inherits from g1
        self.assertEqual(child["agent_type"], "AgentCore")

    def test_crossover_does_not_modify_parents(self):
        g1 = _make_mock_genome()
        g2 = _make_mock_genome()
        original_g1 = json.dumps(g1)
        Genome.crossover(g1, g2)
        self.assertEqual(json.dumps(g1), original_g1)


class TestGenomeClone(unittest.TestCase):
    def test_clone_is_equal(self):
        genome = _make_mock_genome()
        clone = Genome.clone(genome)
        self.assertEqual(clone["agent_type"], genome["agent_type"])
        self.assertEqual(clone["resource_level"], genome["resource_level"])

    def test_clone_is_independent(self):
        genome = _make_mock_genome()
        clone = Genome.clone(genome)
        clone["agent_type"] = "Modified"
        self.assertEqual(genome["agent_type"], "AgentCore")

    def test_clone_deep_copies_nested(self):
        genome = _make_mock_genome()
        clone = Genome.clone(genome)
        clone["module_states"]["decision_module"]["weights"][0] = 999
        self.assertEqual(genome["module_states"]["decision_module"]["weights"][0], 0.1)


class TestGenomeFromAgent(unittest.TestCase):
    def _make_mock_agent(self):
        agent = MagicMock()
        agent.agent_type = "AgentCore"
        agent.resource_level = 75.0
        agent.current_health = 90.0

        # Create mock actions
        action1 = MagicMock()
        action1.name = "gather"
        action1.weight = 0.6
        action2 = MagicMock()
        action2.name = "move"
        action2.weight = 0.4
        agent.actions = [action1, action2]

        return agent

    def test_from_agent_returns_dict(self):
        agent = MagicMock()
        agent.agent_type = "AgentCore"
        agent.resource_level = 75.0
        agent.current_health = 90.0

        action1 = MagicMock()
        action1.name = "gather"
        action1.weight = 0.6
        agent.actions = [action1]

        # No modules with _module suffix on a fresh mock
        genome = Genome.from_agent(agent)
        self.assertIn("action_set", genome)
        self.assertIn("module_states", genome)
        self.assertEqual(genome["agent_type"], "AgentCore")
        self.assertEqual(genome["resource_level"], 75.0)
        self.assertEqual(genome["current_health"], 90.0)

    def test_from_agent_action_set_format(self):
        agent = MagicMock()
        agent.agent_type = "Test"
        agent.resource_level = 10.0
        agent.current_health = 50.0

        a1 = MagicMock()
        a1.name = "move"
        a1.weight = 1.0
        agent.actions = [a1]

        genome = Genome.from_agent(agent)
        self.assertEqual(genome["action_set"], [("move", 1.0)])


class TestGenomeToAgent(unittest.TestCase):
    def test_to_agent_without_factory_raises(self):
        genome = _make_mock_genome()
        with self.assertRaises(RuntimeError):
            Genome.to_agent(genome, "agent_1", (0, 0), MagicMock())

    def test_to_agent_with_factory(self):
        genome = _make_mock_genome()

        # Mock environment
        env = MagicMock()
        env.spatial_service = MagicMock()

        # Mock agent returned by factory
        mock_agent = MagicMock()
        mock_agent.get_component.return_value = None  # No combat component

        factory = MagicMock(return_value=mock_agent)

        result = Genome.to_agent(
            genome, "agent_1", (5, 5), env, agent_factory=factory
        )
        self.assertIs(result, mock_agent)
        factory.assert_called_once()

    def test_to_agent_loads_module_states(self):
        genome = _make_mock_genome()
        env = MagicMock()

        mock_module = MagicMock()
        mock_agent = MagicMock()
        mock_agent.get_component.return_value = None
        # Make hasattr work for decision_module
        mock_agent.decision_module = mock_module

        factory = MagicMock(return_value=mock_agent)
        Genome.to_agent(genome, "agent_1", (0, 0), env, agent_factory=factory)

        mock_module.load_state_dict.assert_called_once_with({"weights": [0.1, 0.9]})

    def test_to_agent_with_combat_component(self):
        genome = _make_mock_genome()
        env = MagicMock()

        combat_comp = MagicMock()
        combat_comp.config.starting_health = 100.0

        mock_agent = MagicMock()
        mock_agent.get_component.return_value = combat_comp

        factory = MagicMock(return_value=mock_agent)
        Genome.to_agent(genome, "agent_1", (0, 0), env, agent_factory=factory)

        # Health should be capped and set
        expected_health = min(
            genome["current_health"], combat_comp.config.starting_health
        )
        mock_agent.state.update_health.assert_called_once_with(expected_health)
