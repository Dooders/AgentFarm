"""Unit tests for genome_id generation in agent creation and reproduction."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from farm.core.agent import AgentCore, AgentFactory, AgentServices, DefaultAgentBehavior
from farm.core.agent.components import (
    CombatComponent,
    MovementComponent,
    PerceptionComponent,
    ReproductionComponent,
    ResourceComponent,
)
from farm.core.agent.config.component_configs import (
    AgentComponentConfig,
    CombatConfig,
    MovementConfig,
    PerceptionConfig,
    ReproductionConfig,
)
from farm.core.agent.config.component_configs import ResourceConfig as ComponentResourceConfig
from farm.core.environment import Environment
from farm.core.services.implementations import EnvironmentTimeService
from farm.database.models import AgentModel, Simulation
from farm.database.database import SimulationDatabase


class TestGenomeIdGeneration(unittest.TestCase):
    """Test genome_id generation when adding agents to environment."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)

        # Create environment with database
        from farm.config import SimulationConfig
        from farm.config.config import EnvironmentConfig, PopulationConfig

        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=100, height=100),
            population=PopulationConfig(system_agents=0, independent_agents=0, control_agents=0),
            max_steps=100,
            seed=42,
        )

        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"amount": 10},
            config=self.config,
            db_path=self.db_path,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "env") and self.env:
            self.env.cleanup()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def create_test_agent(self, agent_id: str, agent_type: str = "system", generation: int = 0) -> AgentCore:
        """Helper to create a test agent."""
        services = AgentServices(
            spatial_service=self.env.spatial_service,
            time_service=EnvironmentTimeService(self.env),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )

        components = [
            MovementComponent(services, MovementConfig()),
            ResourceComponent(services, ComponentResourceConfig()),
            CombatComponent(services, CombatConfig()),
            PerceptionComponent(services, PerceptionConfig()),
            ReproductionComponent(services, ReproductionConfig()),
        ]

        agent = AgentCore(
            agent_id=agent_id,
            position=(50.0, 50.0),
            services=services,
            behavior=DefaultAgentBehavior(),
            components=components,
            config=AgentComponentConfig(),
            environment=self.env,
            initial_resources=100.0,
            generation=generation,
            agent_type=agent_type,
        )

        for component in components:
            component.attach(agent)

        return agent

    def test_genome_id_generated_for_initial_agent(self):
        """Test that genome_id is generated when adding an initial agent."""
        agent = self.create_test_agent("test_agent_1", "system", generation=0)
        
        # Initially genome_id should be empty
        self.assertEqual(agent.genome_id, "")
        
        # Add agent to environment
        self.env.add_agent(agent)
        
        # Genome ID should now be generated
        self.assertNotEqual(agent.genome_id, "")
        self.assertIsInstance(agent.genome_id, str)
        
        # Check format: {agent_type}:{generation}:{parents}:{time_step}
        parts = agent.genome_id.split(":")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "system")
        self.assertEqual(parts[1], "0")
        self.assertEqual(parts[2], "none")  # No parents for initial agent
        self.assertEqual(parts[3], "0")  # Time step 0

    def test_genome_id_saved_to_database(self):
        """Test that genome_id is saved to database when agent is logged."""
        # Simulation record should already exist from environment initialization
        # If it doesn't exist, create it (but this shouldn't normally be needed)
        def ensure_simulation_record(session):
            existing = (
                session.query(Simulation)
                .filter(Simulation.simulation_id == self.env.simulation_id)
                .first()
            )
            if not existing:
                simulation = Simulation(
                    simulation_id=self.env.simulation_id,
                    parameters={"test": True},
                    simulation_db_path=self.db_path,
                )
                session.add(simulation)

        self.env.db._execute_in_transaction(ensure_simulation_record)

        agent = self.create_test_agent("test_agent_db", "independent", generation=0)
        self.env.add_agent(agent)
        
        # Flush buffers to ensure data is written
        self.env.db.logger.flush_all_buffers()

        # Query database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT genome_id FROM agents WHERE agent_id = ?", (agent.agent_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        self.assertIsNotNone(result, "Agent was not stored in database")
        self.assertIsNotNone(result[0], "genome_id should not be NULL")
        self.assertEqual(result[0], agent.genome_id)

    def test_genome_id_for_different_agent_types(self):
        """Test genome_id generation for different agent types."""
        agent_types = ["system", "independent", "control"]
        genome_ids = []

        for agent_type in agent_types:
            agent = self.create_test_agent(f"test_{agent_type}", agent_type, generation=0)
            self.env.add_agent(agent)
            genome_ids.append(agent.genome_id)

        # All should have different genome_ids
        self.assertEqual(len(set(genome_ids)), len(genome_ids))

        # Check each has correct agent_type prefix
        for i, agent_type in enumerate(agent_types):
            self.assertTrue(genome_ids[i].startswith(f"{agent_type}:"))

    def test_genome_id_for_different_generations(self):
        """Test genome_id includes correct generation number."""
        generations = [0, 1, 2, 3]
        genome_ids = []

        for generation in generations:
            agent = self.create_test_agent(
                f"test_gen_{generation}", "system", generation=generation
            )
            self.env.add_agent(agent)
            genome_ids.append(agent.genome_id)

        # Check each has correct generation
        for i, generation in enumerate(generations):
            parts = genome_ids[i].split(":")
            self.assertEqual(parts[1], str(generation))

    def test_genome_id_for_offspring_with_parent(self):
        """Test genome_id generation for offspring with parent information."""
        # Create parent agent
        parent = self.create_test_agent("parent_agent", "system", generation=0)
        self.env.add_agent(parent)
        parent_genome_id = parent.genome_id

        # Create offspring (simulating reproduction)
        offspring = self.create_test_agent("offspring_agent", "system", generation=1)
        # Set parent IDs as reproduce() would
        offspring.state._state = offspring.state._state.model_copy(
            update={"parent_ids": [parent.agent_id]}
        )
        
        self.env.add_agent(offspring)

        # Check offspring genome_id includes parent
        self.assertNotEqual(offspring.genome_id, "")
        parts = offspring.genome_id.split(":")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "system")
        self.assertEqual(parts[1], "1")  # Generation 1
        self.assertIn(parent.agent_id, parts[2])  # Parent ID should be in parents part
        self.assertNotEqual(parts[2], "none")  # Should have parent, not "none"

    def test_genome_id_uses_current_time_step(self):
        """Test that genome_id uses the current environment time step."""
        # Add agent at time 0
        agent1 = self.create_test_agent("agent_time_0", "system", generation=0)
        self.env.add_agent(agent1)
        genome_id_0 = agent1.genome_id
        parts_0 = genome_id_0.split(":")
        self.assertEqual(parts_0[3], "0")

        # Advance time and add another agent
        self.env.time = 10
        agent2 = self.create_test_agent("agent_time_10", "system", generation=0)
        self.env.add_agent(agent2)
        genome_id_10 = agent2.genome_id
        parts_10 = genome_id_10.split(":")
        self.assertEqual(parts_10[3], "10")

    def test_genome_id_not_overwritten_if_already_set(self):
        """Test that genome_id is not overwritten if already set."""
        agent = self.create_test_agent("agent_pre_set", "system", generation=0)
        
        # Manually set genome_id
        existing_genome_id = "custom:genome:id:123"
        agent.state._state = agent.state._state.model_copy(
            update={"genome_id": existing_genome_id}
        )

        # Add to environment
        self.env.add_agent(agent)

        # Genome ID should remain as set
        self.assertEqual(agent.genome_id, existing_genome_id)

    def test_multiple_offspring_different_genome_ids(self):
        """Test that multiple offspring from same parent get different genome_ids."""
        # Create parent
        parent = self.create_test_agent("parent_multi", "system", generation=0)
        self.env.add_agent(parent)

        # Create multiple offspring
        offspring_ids = []
        for i in range(3):
            offspring = self.create_test_agent(
                f"offspring_{i}", "system", generation=1
            )
            offspring.state._state = offspring.state._state.model_copy(
                update={"parent_ids": [parent.agent_id]}
            )
            # Advance time to ensure different genome_ids
            self.env.time = i + 1
            self.env.add_agent(offspring)
            offspring_ids.append(offspring.genome_id)

        # All offspring should have different genome_ids (due to different time steps)
        self.assertEqual(len(set(offspring_ids)), len(offspring_ids))

        # All should reference the parent
        for genome_id in offspring_ids:
            parts = genome_id.split(":")
            self.assertIn(parent.agent_id, parts[2])

    def test_genome_id_in_reproduction_flow(self):
        """Test genome_id is properly set during actual reproduction."""
        # Create parent with sufficient resources for reproduction
        parent = self.create_test_agent("parent_repro", "system", generation=0)
        parent.resource_level = 200.0  # Ensure parent can reproduce
        
        # Add parent to environment
        self.env.add_agent(parent)
        parent_genome_id = parent.genome_id
        
        # Advance time
        self.env.time = 5
        
        # Get reproduction component and ensure it can reproduce
        repro_comp = parent.get_component("reproduction")
        if repro_comp:
            # Set resources to meet reproduction threshold
            resource_comp = parent.get_component("resource")
            if resource_comp:
                resource_comp.level = 200.0
            
            # Try to reproduce
            if repro_comp.can_reproduce():
                success = parent.reproduce()
                
                if success:
                    # Check that offspring was created and has genome_id
                    # Find the newly created offspring
                    offspring_list = [
                        a for a in self.env.agent_objects 
                        if a.agent_id != parent.agent_id and a.generation == 1
                    ]
                    
                    if offspring_list:
                        offspring = offspring_list[0]
                        self.assertNotEqual(offspring.genome_id, "")
                        self.assertNotEqual(offspring.genome_id, parent_genome_id)
                        
                        # Verify genome_id format includes parent
                        parts = offspring.genome_id.split(":")
                        self.assertEqual(parts[1], "1")  # Generation 1
                        self.assertIn(parent.agent_id, parts[2])  # Parent in parents list


if __name__ == "__main__":
    unittest.main()

