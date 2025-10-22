import os
import shutil
import unittest
from unittest.mock import Mock

import numpy as np

from farm.config import (
    AgentBehaviorConfig,
    EnvironmentConfig,
    ResourceConfig,
    SimulationConfig,
)
from farm.core.agent import (
    AgentCore,
    AgentFactory,
    AgentServices,
    AgentComponentConfig,
    DefaultAgentBehavior,
)
from farm.core.agent.config.component_configs import (
    MovementConfig,
    ResourceConfig as ComponentResourceConfig,
    CombatConfig,
    PerceptionConfig,
    ReproductionConfig,
)
from farm.core.agent.components import (
    MovementComponent,
    ResourceComponent,
    CombatComponent,
    PerceptionComponent,
    ReproductionComponent,
)
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
    
    def create_agent(self, agent_id: str, position: tuple, environment: Environment, resource_level: float = 100.0) -> AgentCore:
        """Helper function to create an agent using the new AgentCore architecture."""
        # Create mock services
        from farm.core.services.implementations import EnvironmentTimeService
        time_service = EnvironmentTimeService(environment)
        
        services = AgentServices(
            spatial_service=environment.spatial_service,
            time_service=time_service,
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(),
            lifecycle_service=Mock(),
        )
        
        # Create default behavior
        behavior = DefaultAgentBehavior()
        
        # Create default components with proper configs
        components = [
            MovementComponent(services, MovementConfig()),
            ResourceComponent(services, ComponentResourceConfig()),
            CombatComponent(services, CombatConfig()),
            PerceptionComponent(services, PerceptionConfig()),
            ReproductionComponent(services, ReproductionConfig()),
        ]
        
        # Create config
        config = AgentComponentConfig()
        
        # Create the agent
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            services=services,
            behavior=behavior,
            components=components,
            config=config,
            environment=environment,
            initial_resources=resource_level,
        )
        
        # Attach components to core
        for component in components:
            component.attach(agent)
        
        # Set initial resource level after attachment
        resource_component = agent.get_component("resource")
        if resource_component:
            resource_component.level = resource_level
        
        return agent

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

        # Create agents using the new architecture
        system_agent = self.create_agent(
            "test_agent_0",
            (25, 25),
            self.env,
            self.config.agent_behavior.initial_resource_level,
        )
        independent_agent = self.create_agent(
            "test_agent_1",
            (25, 25),
            self.env,
            self.config.agent_behavior.initial_resource_level,
        )

        self.assertTrue(system_agent.alive)
        self.assertTrue(independent_agent.alive)
        # Check that agents were created with correct IDs and positions
        self.assertEqual(system_agent.agent_id, "test_agent_0")
        self.assertEqual(independent_agent.agent_id, "test_agent_1")
        self.assertEqual(system_agent.state.position, (25, 25))
        self.assertEqual(independent_agent.state.position, (25, 25))

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

        agent = self.create_agent(
            "test_agent_2",
            (25, 25),
            self.env,
            self.config.agent_behavior.initial_resource_level,
        )
        self.env.add_agent(agent)

        # Test direct resource consumption through the environment
        initial_amount = resource.amount

        # Consume resources directly
        consumed = self.env.consume_resource(resource, 2.0)

        # Check that resources were consumed
        self.assertEqual(consumed, 2.0)
        self.assertLess(resource.amount, initial_amount)
        # For now, just verify the resource was consumed
        self.assertTrue(consumed > 0)

    def test_agent_death(self):
        """Test that agents die when resources are depleted."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={"type": "random", "amount": 0},  # No resources
            db_path=self.db_path,
        )

        # Create agent with very low resources
        agent = self.create_agent(
            "test_agent_death",
            (25, 25),
            self.env,
            resource_level=1.0,  # Very low resources
        )
        self.env.add_agent(agent)

        # Get resource component and configure for quick starvation
        resource_component = agent.get_component("resource")
        # Update only the needed fields in the config for testing
        from dataclasses import replace
        resource_component.config = replace(
            resource_component.config,
            base_consumption_rate=1.0,  # Consume 1 resource per step
            starvation_threshold=1,     # Die after 1 step of starvation
        )

        # Verify agent is alive initially
        self.assertTrue(agent.alive)
        self.assertEqual(resource_component.level, 1.0)

        # Run one step - agent should consume resources and die
        agent.step()  # Call agent step to trigger resource consumption
        self.env.update()  # Update environment state

        # Agent should be dead due to starvation
        self.assertFalse(agent.alive)
        self.assertEqual(resource_component.level, 0.0)  # Resources consumed
        self.assertTrue(resource_component.is_starving)

    def test_agent_reproduction(self):
        """Test that agents reproduce when conditions are met."""
        self.env = Environment(
            width=self.config.environment.width,
            height=self.config.environment.height,
            resource_distribution={"type": "random", "amount": 0},  # No resources
            db_path=self.db_path,
        )

        # Create agent with sufficient resources for reproduction
        agent = self.create_agent(
            "test_agent_reproduction",
            (25, 25),
            self.env,
            resource_level=15.0,  # Sufficient resources for one reproduction
        )
        self.env.add_agent(agent)

        # Get reproduction component and configure for testing
        reproduction_component = agent.get_component("reproduction")
        # Create new config with modified values for testing
        from farm.core.agent.config.component_configs import ReproductionConfig
        reproduction_component.config = ReproductionConfig(
            offspring_cost=10.0,  # Cost to reproduce
            offspring_initial_resources=5.0,  # Initial resources for offspring
        )

        # Get resource component
        resource_component = agent.get_component("resource")

        # Verify agent can reproduce
        self.assertTrue(reproduction_component.can_reproduce())
        self.assertEqual(resource_component.level, 15.0)

        # Test reproduction
        offspring = reproduction_component.reproduce()

        # Verify reproduction succeeded (template method returns None)
        self.assertIsNone(offspring)  # Template method returns None
        self.assertEqual(resource_component.level, 5.0)  # Resources reduced by cost (15.0 - 10.0)
        self.assertEqual(reproduction_component.total_offspring, 1)  # Offspring count increased

        # Verify agent can no longer reproduce (insufficient resources)
        self.assertFalse(reproduction_component.can_reproduce())

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

        agent = self.create_agent(
            "test_agent_5",
            (25, 25),
            self.env,
            self.config.agent_behavior.initial_resource_level,
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

        agents = [
            self.create_agent(
                f"test_agent_{i}",
                (25, 25),
                self.env,
                self.config.agent_behavior.initial_resource_level,
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
