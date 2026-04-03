"""Tests for farm/core/collector.py (DataCollector class)."""
import unittest
from unittest.mock import MagicMock

import numpy as np

from farm.core.collector import DataCollector


def _make_mock_agent(agent_id, alive=True, resource_level=10.0, total_reward=5.0, position=(0, 0)):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.alive = alive
    agent.resource_level = resource_level
    agent.total_reward = total_reward
    agent.position = position
    return agent


def _make_mock_resource(amount=5.0):
    resource = MagicMock()
    resource.amount = amount
    return resource


def _make_mock_environment(agents=None, resources=None, width=10, height=10, time=0):
    env = MagicMock()
    env.agent_objects = agents or []
    env.agents = agents or []
    env.resources = resources or []
    env.width = width
    env.height = height
    env.time = time
    env.get_initial_agent_count.return_value = len(agents) if agents else 1
    return env


class TestDataCollectorInit(unittest.TestCase):
    def test_init_defaults(self):
        config = MagicMock()
        collector = DataCollector(config)
        self.assertEqual(collector.data, [])
        self.assertEqual(collector.births_this_cycle, 0)
        self.assertEqual(collector.deaths_this_cycle, 0)
        self.assertEqual(collector.competitive_interactions, 0)
        self.assertEqual(collector.agent_resource_history, {})


class TestDataCollectorCollect(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.collector = DataCollector(self.config)

    def test_collect_adds_data_point(self):
        agent = _make_mock_agent("a1", resource_level=15.0)
        resource = _make_mock_resource(10.0)
        env = _make_mock_environment(agents=[agent], resources=[resource])

        self.collector.collect(env, step=1)
        self.assertEqual(len(self.collector.data), 1)

    def test_collect_step_number(self):
        env = _make_mock_environment()
        self.collector.collect(env, step=42)
        self.assertEqual(self.collector.data[0]["step"], 42)

    def test_collect_total_agent_count(self):
        agents = [
            _make_mock_agent("a1", alive=True),
            _make_mock_agent("a2", alive=False),
        ]
        env = _make_mock_environment(agents=agents)
        self.collector.collect(env, step=1)
        # Only alive agents counted
        self.assertEqual(self.collector.data[0]["total_agent_count"], 1)

    def test_collect_total_resources(self):
        resources = [_make_mock_resource(3.0), _make_mock_resource(7.0)]
        env = _make_mock_environment(resources=resources)
        self.collector.collect(env, step=1)
        self.assertAlmostEqual(self.collector.data[0]["total_resources"], 10.0)

    def test_collect_average_resource_per_agent(self):
        agents = [
            _make_mock_agent("a1", resource_level=10.0),
            _make_mock_agent("a2", resource_level=20.0),
        ]
        env = _make_mock_environment(agents=agents)
        self.collector.collect(env, step=1)
        self.assertAlmostEqual(self.collector.data[0]["average_resource_per_agent"], 15.0)

    def test_collect_no_agents_average_zero(self):
        env = _make_mock_environment(agents=[])
        self.collector.collect(env, step=1)
        self.assertEqual(self.collector.data[0]["average_resource_per_agent"], 0)

    def test_collect_resets_cycle_counters(self):
        self.collector.births_this_cycle = 3
        self.collector.deaths_this_cycle = 2
        self.collector.competitive_interactions = 5
        env = _make_mock_environment()
        self.collector.collect(env, step=1)
        self.assertEqual(self.collector.births_this_cycle, 0)
        self.assertEqual(self.collector.deaths_this_cycle, 0)
        self.assertEqual(self.collector.competitive_interactions, 0)

    def test_collect_births_deaths_recorded(self):
        self.collector.births_this_cycle = 2
        self.collector.deaths_this_cycle = 1
        env = _make_mock_environment()
        self.collector.collect(env, step=1)
        self.assertEqual(self.collector.data[0]["births"], 2)
        self.assertEqual(self.collector.data[0]["deaths"], 1)

    def test_collect_competitive_interactions_recorded(self):
        self.collector.competitive_interactions = 4
        env = _make_mock_environment()
        self.collector.collect(env, step=1)
        self.assertEqual(self.collector.data[0]["competitive_interactions"], 4)

    def test_collect_tracks_agent_resource_history(self):
        agent = _make_mock_agent("a1", resource_level=25.0)
        env = _make_mock_environment(agents=[agent])
        self.collector.collect(env, step=1)
        self.assertIn("a1", self.collector.agent_resource_history)
        self.assertEqual(self.collector.agent_resource_history["a1"], [25.0])

    def test_collect_multiple_steps_appends_history(self):
        agent = _make_mock_agent("a1", resource_level=10.0)
        env = _make_mock_environment(agents=[agent])
        self.collector.collect(env, step=1)
        agent.resource_level = 20.0
        self.collector.collect(env, step=2)
        self.assertEqual(self.collector.agent_resource_history["a1"], [10.0, 20.0])


class TestCalculateAverageLifespan(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_no_agents_returns_zero(self):
        env = _make_mock_environment(agents=[], time=10)
        result = self.collector._calculate_average_lifespan(env)
        self.assertEqual(result, 0)

    def test_with_agents(self):
        agent1 = _make_mock_agent("a1")
        agent1.birth_time = 0
        agent2 = _make_mock_agent("a2")
        agent2.birth_time = 5
        env = _make_mock_environment(agents=[agent1, agent2], time=10)
        result = self.collector._calculate_average_lifespan(env)
        # agent1 lived 10, agent2 lived 5, average = 7.5
        self.assertAlmostEqual(result, 7.5)


class TestCalculateResourceEfficiency(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_no_agents_returns_zero(self):
        result = self.collector._calculate_resource_efficiency([])
        self.assertEqual(result, 0)

    def test_with_agents(self):
        agents = [
            _make_mock_agent("a1", resource_level=10.0, total_reward=20.0),
            _make_mock_agent("a2", resource_level=5.0, total_reward=10.0),
        ]
        result = self.collector._calculate_resource_efficiency(agents)
        # a1: 20/10 = 2.0, a2: 10/5 = 2.0, avg = 2.0
        self.assertAlmostEqual(result, 2.0)


class TestCalculateResourceDensity(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_density_calculation(self):
        resources = [_make_mock_resource(20.0), _make_mock_resource(30.0)]
        env = _make_mock_environment(resources=resources, width=5, height=4)
        result = self.collector._calculate_resource_density(env)
        # total_resources=50, total_area=20
        self.assertAlmostEqual(result, 2.5)


class TestCalculatePopulationStability(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_less_than_10_steps_returns_one(self):
        self.collector.data = [{"total_agent_count": i} for i in range(5)]
        result = self.collector._calculate_population_stability()
        self.assertEqual(result, 1.0)

    def test_stable_population_high_stability(self):
        # All same count → std=0, stability=1
        self.collector.data = [{"total_agent_count": 10} for _ in range(15)]
        result = self.collector._calculate_population_stability()
        self.assertAlmostEqual(result, 1.0)

    def test_variable_population_lower_stability(self):
        counts = [10, 20, 5, 15, 8, 12, 18, 3, 25, 7, 14]
        self.collector.data = [{"total_agent_count": c} for c in counts]
        result = self.collector._calculate_population_stability()
        self.assertLess(result, 1.0)


class TestCalculateAvgResourceAccumulation(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_empty_history_returns_zero(self):
        result = self.collector._calculate_avg_resource_accumulation()
        self.assertEqual(result, 0)

    def test_single_entry_history_returns_zero(self):
        self.collector.agent_resource_history = {"a1": [10.0]}
        result = self.collector._calculate_avg_resource_accumulation()
        self.assertEqual(result, 0)

    def test_growing_resources(self):
        self.collector.agent_resource_history = {"a1": [0, 5, 10, 15, 20]}
        result = self.collector._calculate_avg_resource_accumulation()
        # rate = (20 - 0) / 5 = 4.0
        self.assertAlmostEqual(result, 4.0)


class TestCalculateResourceInequality(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(MagicMock())

    def test_no_agents_returns_zero(self):
        result = self.collector._calculate_resource_inequality([])
        self.assertEqual(result, 0)

    def test_all_zero_resources_returns_zero(self):
        agents = [_make_mock_agent("a1", resource_level=0), _make_mock_agent("a2", resource_level=0)]
        result = self.collector._calculate_resource_inequality(agents)
        self.assertEqual(result, 0)

    def test_equal_resources_low_inequality(self):
        agents = [_make_mock_agent(f"a{i}", resource_level=10.0) for i in range(5)]
        result = self.collector._calculate_resource_inequality(agents)
        self.assertAlmostEqual(result, 0.0)

    def test_unequal_resources_nonzero_inequality(self):
        agents = [
            _make_mock_agent("a1", resource_level=1.0),
            _make_mock_agent("a2", resource_level=99.0),
        ]
        result = self.collector._calculate_resource_inequality(agents)
        self.assertGreater(result, 0)


class TestCalculateAverageResources(unittest.TestCase):
    def test_no_alive_agents_returns_zero(self):
        collector = DataCollector(MagicMock())
        dead_agent = _make_mock_agent("a1", alive=False)
        env = _make_mock_environment(agents=[dead_agent])
        result = collector.calculate_average_resources(env)
        self.assertEqual(result, 0)

    def test_with_alive_agents(self):
        collector = DataCollector(MagicMock())
        agents = [
            _make_mock_agent("a1", resource_level=10.0),
            _make_mock_agent("a2", resource_level=30.0),
        ]
        env = _make_mock_environment(agents=agents)
        result = collector.calculate_average_resources(env)
        self.assertAlmostEqual(result, 20.0)


class TestToDataframe(unittest.TestCase):
    def test_empty_returns_empty_dataframe(self):
        collector = DataCollector(MagicMock())
        df = collector.to_dataframe()
        self.assertTrue(df.empty)

    def test_with_data_returns_dataframe(self):
        collector = DataCollector(MagicMock())
        agent = _make_mock_agent("a1")
        env = _make_mock_environment(agents=[agent])
        collector.collect(env, step=1)
        df = collector.to_dataframe()
        self.assertFalse(df.empty)
        self.assertIn("step", df.columns)
        self.assertIn("total_agent_count", df.columns)
