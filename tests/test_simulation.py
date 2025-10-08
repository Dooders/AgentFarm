import os
import shutil
import unittest

import numpy as np

from farm.config import (
    AgentBehaviorConfig,
    EnvironmentConfig,
    ResourceConfig,
    SimulationConfig,
)
from farm.core.agent import AgentFactory
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
        self.config = SimulationConfig.from_centralized_config(environment="testing")
        self.db_path = os.path.join(self.test_dir, "test_simulation.db")

    def tearDown(self):
        """Clean up test environment."""
        # Clean up any environment instances that might have database connections
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up environment: {e}")

        # Clean up any database instances
        if hasattr(self, "db") and self.db is not None:
            try:
                self.db.close()
            except Exception as e:
                print(f"Warning: Error closing database: {e}")

        # Force garbage collection to release file handles
        import gc

        gc.collect()

        # Wait a moment for connections to be fully closed
        import time

        time.sleep(0.1)

        # Now try to remove the test directory
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError as e:
                print(f"Warning: Could not remove test directory: {e}")
                # Try to remove individual files
                try:
                    for root, dirs, files in os.walk(self.test_dir, topdown=False):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except PermissionError:
                                pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except PermissionError:
                                pass
                    os.rmdir(self.test_dir)
                except Exception:
                    pass

    def test_environment_initialization(self):
        """Test that environment is initialized correctly."""
        env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.resources.initial_resources,
            },
            db_path=self.db_path,
        )

        self.assertEqual(len(env.resources), self.config.resources.initial_resources)
        self.assertEqual(len(env.agents), 0)
        self.assertEqual(env.width, self.config.environment.width)
        self.assertEqual(env.height, self.config.environment.height)

    def test_agent_creation(self):
        """Test that agents are created with correct attributes."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.resources.initial_resources,
            },
            db_path=self.db_path,
        )

        # Create agents with proper environment reference using AgentFactory
        # Services will be injected by environment when agents are added
        factory = AgentFactory(
            spatial_service=self.env.spatial_service,
        )
        
        system_agent = factory.create_default_agent(
            agent_id="test_agent_0",
            position=(25, 25),
            initial_resources=self.config.agent_behavior.initial_resource_level,
        )
        independent_agent = factory.create_default_agent(
            agent_id="test_agent_1",
            position=(25, 25),
            initial_resources=self.config.agent_behavior.initial_resource_level,
        )

        self.assertTrue(system_agent.alive)
        self.assertTrue(independent_agent.alive)
        
        # Access resources through component
        system_resource = system_agent.get_component("resource")
        independent_resource = independent_agent.get_component("resource")
        self.assertEqual(
            system_resource.level,
            self.config.agent_behavior.initial_resource_level,
        )
        self.assertEqual(
            independent_resource.level,
            self.config.agent_behavior.initial_resource_level,
        )

    def test_resource_consumption(self):
        """Test that resources are consumed correctly."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={"type": "random", "amount": 1},
            db_path=self.db_path,
        )

        resource = Resource(0, (25, 25), amount=10)
        self.env.resources = [resource]

        factory = AgentFactory(
            spatial_service=self.env.spatial_service,
        )
        agent = factory.create_default_agent(
            agent_id="test_agent_2",
            position=(25, 25),
            initial_resources=self.config.agent_behavior.initial_resource_level,
        )
        self.env.add_agent(agent)

        # Test direct resource consumption through the environment
        initial_amount = resource.amount
        resource_comp = agent.get_component("resource")
        initial_agent_resources = resource_comp.level

        # Consume resources directly
        consumed = self.env.consume_resource(resource, 2.0)
        resource_comp.add(consumed)

        # Check that resources were consumed and agent received them
        self.assertEqual(consumed, 2.0)
        self.assertLess(resource.amount, initial_amount)
        self.assertGreater(resource_comp.level, initial_agent_resources)

    def test_agent_death(self):
        """Test that agents die when resources are depleted."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={"type": "random", "amount": 0},
            db_path=self.db_path,
        )

        # Create a test-specific config object to avoid mutating shared state
        test_config = SimulationConfig(
            environment=EnvironmentConfig(
                width=self.config.environment.width,
                height=self.config.environment.height,
            ),
            resources=ResourceConfig(
                initial_resources=self.config.resources.initial_resources,
            ),
            agent_behavior=AgentBehaviorConfig(
                initial_resource_level=self.config.agent_behavior.initial_resource_level,
                base_consumption_rate=1.0,  # Force higher consumption rate for testing
            ),
        )

        # Create agent with very low starvation threshold for testing
        class TestAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.starvation_threshold = 3  # Override default threshold of 100

        agent = TestAgent(
            "test_agent_3",
            (25, 25),
            1,
            self.env.spatial_service,
            environment=self.env,
            config=test_config,
        )
        self.env.add_agent(agent)

        # Run until agent dies or we hit a reasonable limit
        for _ in range(10):
            if agent.alive:
                agent.act()
            else:
                break

        self.assertFalse(agent.alive)

    def test_agent_reproduction(self):
        """Test that agents reproduce when conditions are met."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.resources.initial_resources,
            },
            db_path=self.db_path,
        )

        factory = AgentFactory(
            spatial_service=self.env.spatial_service,
        )
        agent = factory.create_default_agent(
            agent_id="test_agent_4",
            position=(25, 25),
            initial_resources=20,
        )
        self.env.add_agent(agent)

        initial_agent_count = len(self.env.agents)
        reproduction_comp = agent.get_component("reproduction")
        reproduction_comp.reproduce()

        self.assertGreater(len(self.env.agents), initial_agent_count)

    def test_database_logging(self):
        """Test that simulation state is correctly logged to database."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.resources.initial_resources,
            },
            db_path=self.db_path,
        )

        factory = AgentFactory(
            spatial_service=self.env.spatial_service,
        )
        agent = factory.create_default_agent(
            agent_id="test_agent_5",
            position=(25, 25),
            initial_resources=self.config.agent_behavior.initial_resource_level,
        )
        self.env.add_agent(agent)

        for _ in range(5):
            self.env.update()

        self.db = SimulationDatabase(self.db_path, simulation_id="test_simulation")
        data = self.db.query.gui_repository.get_simulation_data(step_number=1)

        self.assertIsNotNone(data["agent_states"])
        self.assertIsNotNone(data["resource_states"])
        self.assertIsNotNone(data["metrics"])

    def test_batch_agent_addition(self):
        """Test that multiple agents can be added efficiently in batch."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={
                "type": "random",
                "amount": self.config.resources.initial_resources,
            },
            db_path=self.db_path,
        )

        factory = AgentFactory(
            spatial_service=self.env.spatial_service,
        )
        
        agents = [
            factory.create_default_agent(
                agent_id=f"test_agent_{i}",
                position=(25, 25),
                initial_resources=self.config.agent_behavior.initial_resource_level,
            )
            for i in range(10)
        ]

        for agent in agents:
            self.env.add_agent(agent)

        self.assertEqual(len(self.env.agents), 10)

        self.db = SimulationDatabase(self.db_path, simulation_id="test_simulation")
        session = self.db.Session()
        count = session.query(AgentModel).count()
        self.assertEqual(count, 10)
        session.close()


if __name__ == "__main__":
    unittest.main()
