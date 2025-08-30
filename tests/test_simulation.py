import os
import shutil
import unittest

import numpy as np

from farm.core.agent import BaseAgent
from farm.core.config import SimulationConfig
from farm.core.environment import Environment
from farm.core.resources import Resource
from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel


class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "test_data"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.config = SimulationConfig.from_yaml(
            os.path.join("tests", "test_config.yaml")
        )
        self.db_path = os.path.join(self.test_dir, "test_simulation.db")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_environment_initialization(self):
        """Test that environment is initialized correctly."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.initial_resources,
            },
            db_path=self.db_path,
        )

        self.assertEqual(len(env.resources), self.config.initial_resources)
        self.assertEqual(len(env.agents), 0)
        self.assertEqual(env.width, self.config.width)
        self.assertEqual(env.height, self.config.height)

    def test_agent_creation(self):
        """Test that agents are created with correct attributes."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.initial_resources,
            },
            db_path=self.db_path,
        )

        system_agent = BaseAgent(
            "0", (25, 25), self.config.initial_resource_level, env.spatial_service
        )
        independent_agent = BaseAgent(
            "1", (25, 25), self.config.initial_resource_level, env.spatial_service
        )

        self.assertTrue(system_agent.alive)
        self.assertTrue(independent_agent.alive)
        self.assertEqual(
            system_agent.resource_level, self.config.initial_resource_level
        )
        self.assertEqual(
            independent_agent.resource_level, self.config.initial_resource_level
        )

    def test_resource_consumption(self):
        """Test that resources are consumed correctly."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": 1},
            db_path=self.db_path,
        )

        resource = Resource(0, (25, 25), amount=10)
        env.resources = [resource]

        agent = BaseAgent(
            "0", (25, 25), self.config.initial_resource_level, env.spatial_service
        )
        env.add_agent(agent)

        agent.act()

        self.assertLess(resource.amount, 10)
        self.assertGreater(agent.resource_level, self.config.initial_resource_level)

    def test_agent_death(self):
        """Test that agents die when resources are depleted."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={"type": "random", "amount": 0},
            db_path=self.db_path,
        )

        agent = BaseAgent("0", (25, 25), 1, env.spatial_service)
        env.add_agent(agent)

        for _ in range(20):
            if agent.alive:
                agent.act()

        self.assertFalse(agent.alive)

    def test_agent_reproduction(self):
        """Test that agents reproduce when conditions are met."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.initial_resources,
            },
            db_path=self.db_path,
        )

        agent = BaseAgent("0", (25, 25), 20, env.spatial_service)
        env.add_agent(agent)

        initial_agent_count = len(env.agents)
        agent.reproduce()

        self.assertGreater(len(env.agents), initial_agent_count)

    def test_database_logging(self):
        """Test that simulation state is correctly logged to database."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.initial_resources,
            },
            db_path=self.db_path,
        )

        agent = BaseAgent(
            "0", (25, 25), self.config.initial_resource_level, env.spatial_service
        )
        env.add_agent(agent)

        for _ in range(5):
            env.update()

        db = SimulationDatabase(self.db_path)
        data = db.query.gui_repository.get_simulation_data(step_number=1)

        self.assertIsNotNone(data["agent_states"])
        self.assertIsNotNone(data["resource_states"])
        self.assertIsNotNone(data["metrics"])

        db.close()

    def test_batch_agent_addition(self):
        """Test that multiple agents can be added efficiently in batch."""
        env = Environment(
            width=self.config.width,
            height=self.config.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.initial_resources,
            },
            db_path=self.db_path,
        )

        agents = [
            BaseAgent(
                str(i),
                (25, 25),
                self.config.initial_resource_level,
                env.spatial_service,
            )
            for i in range(10)
        ]

        for agent in agents:
            env.add_agent(agent)

        self.assertEqual(len(env.agents), 10)

        db = SimulationDatabase(self.db_path)
        session = db.Session()
        count = session.query(AgentModel).count()
        self.assertEqual(count, 10)
        session.close()
        db.close()


if __name__ == "__main__":
    unittest.main()
