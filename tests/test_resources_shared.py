import unittest
import sys
import os
import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from farm.core.environment import Environment
from farm.agents.base_agent import BaseAgent
from farm.agents.system_agent import SystemAgent
from farm.core.config import SimulationConfig
from farm.actions.share import share_action


class TestResourcesShared(unittest.TestCase):
    """Test case for verifying the resources_shared metric calculation."""

    def setUp(self):
        """Set up a test environment with agents."""
        # Create a simple configuration
        self.config = SimulationConfig()
        self.config.max_resource_amount = 100
        self.config.initial_resources = 10
        self.config.resource_regen_rate = 0.1
        self.config.resource_regen_amount = 5
        self.config.max_population = 100
        
        # Create an environment with in-memory database (db_path=None)
        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"amount": 10, "type": "uniform"},
            db_path=None,
            max_resource=100,
            config=self.config
        )
        
        # Add two agents close to each other
        self.agent1 = SystemAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=20,
            environment=self.env,
            generation=0
        )
        
        self.agent2 = SystemAgent(
            agent_id="agent2",
            position=(15, 15),  # Close enough to agent1 for sharing
            resource_level=5,
            environment=self.env,
            generation=0
        )
        
        self.env.add_agent(self.agent1)
        self.env.add_agent(self.agent2)
        
        # Force the share module to always share resources
        # This is a test-specific patch to ensure sharing happens
        original_get_share_decision = self.agent1.share_module.get_share_decision
        
        def mock_get_share_decision(agent, state):
            # Always return SHARE_MEDIUM (2), the second agent, and 5 resources
            return 2, self.agent2, 5
            
        self.agent1.share_module.get_share_decision = mock_get_share_decision

    def test_resources_shared_tracking(self):
        """Test that resources_shared is properly tracked when agents share resources."""
        # Verify initial state
        self.assertEqual(self.env.resources_shared, 0)
        self.assertEqual(self.env.resources_shared_this_step, 0)
        
        # Initial resource levels
        agent1_initial_resources = self.agent1.resource_level
        agent2_initial_resources = self.agent2.resource_level
        
        # Execute share action
        share_action(self.agent1)
        
        # Verify resources were transferred
        self.assertEqual(self.agent1.resource_level, agent1_initial_resources - 5)
        self.assertEqual(self.agent2.resource_level, agent2_initial_resources + 5)
        
        # Verify resources_shared counters were updated
        self.assertEqual(self.env.resources_shared, 5)
        self.assertEqual(self.env.resources_shared_this_step, 5)
        
        # Execute another share action
        share_action(self.agent1)
        
        # Verify resources were transferred again
        self.assertEqual(self.agent1.resource_level, agent1_initial_resources - 10)
        self.assertEqual(self.agent2.resource_level, agent2_initial_resources + 10)
        
        # Verify resources_shared counters were updated cumulatively
        self.assertEqual(self.env.resources_shared, 10)
        self.assertEqual(self.env.resources_shared_this_step, 10)
        
        # Update environment to simulate a step
        self.env.update()
        
        # Verify resources_shared_this_step was reset
        self.assertEqual(self.env.resources_shared, 10)  # Total should remain
        self.assertEqual(self.env.resources_shared_this_step, 0)  # This step should reset
        
        # Verify metrics contain the resources_shared values
        metrics = self.env._calculate_metrics()
        self.assertEqual(metrics["resources_shared"], 10)
        self.assertEqual(metrics["resources_shared_this_step"], 0)

    def test_resources_shared_in_metrics(self):
        """Test that resources_shared appears in the metrics dictionary."""
        # Execute share action
        share_action(self.agent1)
        
        # Calculate metrics
        metrics = self.env._calculate_metrics()
        
        # Verify resources_shared is in metrics
        self.assertIn("resources_shared", metrics)
        self.assertIn("resources_shared_this_step", metrics)
        self.assertEqual(metrics["resources_shared"], 5)
        self.assertEqual(metrics["resources_shared_this_step"], 5)

    def tearDown(self):
        """Clean up resources."""
        self.env.cleanup()


if __name__ == "__main__":
    unittest.main() 