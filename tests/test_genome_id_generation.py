"""Unit tests for genome_id generation in agent creation and reproduction."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from farm.core.agent import AgentCore, AgentServices, DefaultAgentBehavior
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
from farm.database.models import Simulation


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
        
        # Check format: new format is ::1 for initial agents (first of kind, counter starts at 1)
        self.assertEqual(agent.genome_id, "::1")

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

        # All should have the same genome_id format (::) for initial agents with no parents
        # They will be differentiated by counters if multiple are created
        # First should be ::1, second ::2, third ::3 (counters start at 1)
        self.assertEqual(genome_ids[0], "::1")
        self.assertEqual(genome_ids[1], "::2")
        self.assertEqual(genome_ids[2], "::3")

    def test_genome_id_for_different_generations(self):
        """Test genome_id generation (generation no longer part of format)."""
        generations = [0, 1, 2, 3]
        genome_ids = []

        for generation in generations:
            agent = self.create_test_agent(
                f"test_gen_{generation}", "system", generation=generation
            )
            self.env.add_agent(agent)
            genome_ids.append(agent.genome_id)

        # All initial agents with no parents should get :: format with counters
        # First gets ::1, subsequent get ::2, ::3, ::4 (counters start at 1)
        for i in range(len(generations)):
            self.assertEqual(genome_ids[i], f"::{i+1}")

    def test_genome_id_for_offspring_with_parent(self):
        """Test genome_id generation for offspring with parent information."""
        # Create parent agent
        parent = self.create_test_agent("parent_agent", "system", generation=0)
        self.env.add_agent(parent)

        # Create offspring (simulating reproduction)
        offspring = self.create_test_agent("offspring_agent", "system", generation=1)
        # Set parent IDs as reproduce() would
        offspring.state._state = offspring.state._state.model_copy(
            update={"parent_ids": [parent.agent_id]}
        )
        
        self.env.add_agent(offspring)

        # Check offspring genome_id format: should be parent_id:1 for cloning (first offspring, counter starts at 1)
        self.assertNotEqual(offspring.genome_id, "")
        self.assertEqual(offspring.genome_id, f"{parent.agent_id}:1")

    def test_genome_id_no_longer_uses_time_step(self):
        """Test that genome_id no longer includes time step in new format."""
        # Add agent at time 0
        agent1 = self.create_test_agent("agent_time_0", "system", generation=0)
        self.env.add_agent(agent1)
        genome_id_0 = agent1.genome_id
        # Should be ::1 for initial agent with no parents (first of kind, counter starts at 1)
        self.assertEqual(genome_id_0, "::1")

        # Advance time and add another agent
        self.env.time = 10
        agent2 = self.create_test_agent("agent_time_10", "system", generation=0)
        self.env.add_agent(agent2)
        genome_id_10 = agent2.genome_id
        # Should be ::2 (counter for second initial agent, counters start at 1)
        self.assertEqual(genome_id_10, "::2")

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
        """Test that multiple offspring from same parent get different genome_ids with counters."""
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
            self.env.add_agent(offspring)
            offspring_ids.append(offspring.genome_id)

        # All offspring should have different genome_ids with counters
        # First: parent_id:1, second: parent_id:2, third: parent_id:3 (counters start at 1)
        self.assertEqual(len(set(offspring_ids)), len(offspring_ids))
        self.assertEqual(offspring_ids[0], f"{parent.agent_id}:1")
        self.assertEqual(offspring_ids[1], f"{parent.agent_id}:2")
        self.assertEqual(offspring_ids[2], f"{parent.agent_id}:3")

        # All should reference the parent
        for genome_id in offspring_ids:
            self.assertTrue(genome_id.startswith(f"{parent.agent_id}:"))

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
                        # Should be parent_id:1 for cloning (first offspring, counter starts at 1)
                        self.assertTrue(offspring.genome_id.startswith(f"{parent.agent_id}:"))
                        # Should have counter starting from 1
                        self.assertIn(":1", offspring.genome_id)

    def test_genome_id_cache_initialized_on_startup(self):
        """Test that the genome ID cache is initialized on environment startup."""
        self.assertIsInstance(self.env._genome_id_cache, set)
        self.assertTrue(self.env._genome_id_cache_loaded)

    def test_genome_id_cache_populated_after_add_agent(self):
        """Test that adding an agent populates the genome ID cache."""
        agent = self.create_test_agent("cache_agent_1", "system", generation=0)
        self.assertEqual(len(self.env._genome_id_cache), 0)  # Empty before adding

        self.env.add_agent(agent)

        # Cache must contain the newly assigned genome_id
        self.assertIn(agent.genome_id, self.env._genome_id_cache)

    def test_bulk_add_agents_no_db_query_per_agent(self):
        """Test that bulk agent creation uses the cache and avoids per-agent DB queries."""
        n_agents = 10

        db_query_calls = []

        original_execute = self.env.db._execute_in_transaction

        def tracking_execute(fn, *args, **kwargs):
            db_query_calls.append(fn)
            return original_execute(fn, *args, **kwargs)

        with patch.object(self.env.db, "_execute_in_transaction", side_effect=tracking_execute):
            for i in range(n_agents):
                agent = self.create_test_agent(f"bulk_agent_{i}", "system", generation=0)
                self.env.add_agent(agent)

        # The only DB operations should be the log_agents_batch writes, not existence queries.
        # Each add_agent may trigger at most one log_agents_batch call.
        # No call should come from check_genome_id_exists (which would be n_agents * O(counter) calls).
        # We count only how many calls touch AgentModel for filtering (the query path).
        query_call_count = len(db_query_calls)

        # With the cache, the genome existence checker should never reach the DB
        # for agents added in the same session.  The only DB interactions should be
        # the batch-write calls (one per add_agent via log_agents_batch).
        self.assertLessEqual(
            query_call_count,
            n_agents,
            f"Expected at most {n_agents} DB calls (one batch-write per agent), "
            f"got {query_call_count}. Cache may not be preventing existence queries.",
        )

        # All agents should have a genome_id in the cache
        for i in range(n_agents):
            agent_id = f"bulk_agent_{i}"
            agent_obj = self.env._agent_objects.get(agent_id)
            self.assertIsNotNone(agent_obj)
            self.assertIn(agent_obj.genome_id, self.env._genome_id_cache)


if __name__ == "__main__":
    unittest.main()

