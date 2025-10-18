import unittest

import numpy as np

from farm.config import SimulationConfig
from farm.core.agent import AgentCore
from farm.core.environment import Environment


class TestCoreIntegration(unittest.TestCase):
    def setUp(self):
        # Lean config for faster tests
        from farm.config.config import (
            EnvironmentConfig,
            PopulationConfig,
            ResourceConfig,
        )

        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=30, height=30),
            population=PopulationConfig(
                system_agents=0, independent_agents=0, control_agents=0
            ),
            resources=ResourceConfig(initial_resources=5),
            max_steps=50,
            seed=1234,
        )

        # Start environment with no initial agents; we'll add them explicitly
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={"amount": 5},
            config=self.config,
            db_path=":memory:",
        )

        # Add two agents
        a1 = AgentCore(
            agent_id=self.env.get_next_agent_id(),
            position=(5, 5),
            resource_level=5,
            environment=self.env,
            spatial_service=self.env.spatial_service,
        )
        a2 = AgentCore(
            agent_id=self.env.get_next_agent_id(),
            position=(7, 7),
            resource_level=5,
            environment=self.env,
            spatial_service=self.env.spatial_service,
        )
        self.env.add_agent(a1)
        self.env.add_agent(a2)

    def tearDown(self):
        self.env.cleanup()

    def test_end_to_end_step_cycle_and_metrics(self):
        # Reset and perform several steps cycling agents
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)

        steps = 0
        while steps < 10 and self.env.agents:
            action = int(self.env.action_space().sample())
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            if terminated or truncated:
                break
            steps += 1

        # Metrics should be a dict with expected keys
        metrics = self.env._calculate_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_agents", metrics)
        self.assertTrue(any("resource" in k for k in metrics.keys()))

    def test_dynamic_action_space_update(self):
        # Initially use default mapping length
        initial_n = self.env.action_space().n

        # Reduce enabled actions to just move and pass
        self.env.update_action_space(["move", "pass"])
        self.assertEqual(self.env.action_space().n, 2)

        # Step with new space
        obs, info = self.env.reset()
        action = int(self.env.action_space().sample())
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(obs, np.ndarray)

        # Restore full mapping to ensure it can expand again
        self.env.update_action_space(None)
        self.assertGreaterEqual(self.env.action_space().n, initial_n)

    def test_reproduction_and_removal_paths(self):
        # Force an agent to reproduce by calling its method and registering offspring
        parent = self.env.agent_objects[0]
        # Ensure services are injected (Environment.add_agent does so)
        success = parent.reproduce()
        # Reproduction may be probabilistic; we allow either outcome but environment should stay consistent
        self.assertIn(parent.agent_id, self.env._agent_objects)

        # Remove an agent and ensure environment structures update
        target = self.env.agent_objects[0]
        self.env.remove_agent(target)
        self.assertNotIn(target.agent_id, self.env._agent_objects)
        self.assertNotIn(target.agent_id, self.env.agents)

    def test_database_logging_paths(self):
        # Ensure DB exists for in-memory path
        self.assertIsNotNone(self.env.db)

        # Log an interaction edge (should not raise)
        self.env.log_interaction_edge(
            source_type="agent",
            source_id=self.env.agents[0],
            target_type="resource",
            target_id="r1",
            interaction_type="gather",
            details={"amount": 1.0},
        )

        # Process an action and ensure DB logging path is exercised
        obs, info = self.env.reset()
        action = int(self.env.action_space().sample())
        _ = self.env.step(action)


if __name__ == "__main__":
    unittest.main()
