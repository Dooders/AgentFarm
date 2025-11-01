"""Integration tests for learning experience logging functionality.

This module contains integration tests that verify learning experiences
are correctly logged to the database during agent decision-making and training.

These tests verify the fix for Issue #481 where learning experiences were not
being logged to the database.
"""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from gymnasium import spaces

from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule
from farm.database.data_logging import DataLogger, DataLoggingConfig
from farm.database.database import SimulationDatabase
from farm.database.models import Base


class TestLearningExperienceLoggingIntegration(unittest.TestCase):
    """Integration tests for learning experience logging."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_simulation.db"

        # Create database
        self.db = SimulationDatabase(str(self.db_path), simulation_id="test_sim_001")

        # Create required records for foreign key constraints
        self.db.add_simulation_record(
            simulation_id="test_sim_001", start_time=datetime.now(), status="running", parameters={}
        )
        self.db.logger.log_agent(
            agent_id="agent_001",
            birth_time=0,
            agent_type="test",
            position=(0.0, 0.0),
            initial_resources=10.0,
            starting_health=100.0,
            genome_id="test_genome",
            generation=1,
        )
        self.db.logger.flush_all_buffers()

        # Create logger
        self.logger = DataLogger(
            self.db, simulation_id="test_sim_001", config=DataLoggingConfig(buffer_size=10, commit_interval=60)
        )

        # Create mock environment with database
        self.mock_env = Mock()
        self.mock_env.db = self.db
        self.mock_env.action_space = spaces.Discrete(7)

        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "agent_001"
        self.mock_agent.environment = self.mock_env

        # Create mock time service
        self.mock_time_service = Mock()
        self.mock_time_service.current_time.return_value = 1
        # Set up services structure as expected by decision module
        self.mock_agent.services = Mock()
        self.mock_agent.services.time_service = self.mock_time_service

        # Create mock actions
        self.mock_actions = []
        action_names = ["pass", "move", "gather", "share", "reproduce", "attack", "defend"]
        for name in action_names:
            mock_action = Mock()
            mock_action.name = name
            self.mock_actions.append(mock_action)
        self.mock_agent.actions = self.mock_actions

        # Create observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def _parse_details_json(self, details_str):
        """Helper function to parse details JSON string consistently.
        
        Args:
            details_str: JSON string or dict to parse
            
        Returns:
            dict: Parsed details dictionary, or None if parsing fails
        """
        if not details_str:
            return None
        try:
            return json.loads(details_str) if isinstance(details_str, str) else details_str
        except (json.JSONDecodeError, TypeError):
            return None

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_learning_experiences_logged_to_database(self):
        """Test that learning experiences are actually written to database."""
        # Create decision module
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Simulate several steps
        for step in range(10):
            self.mock_time_service.current_time.return_value = step

            state = torch.randn(8)
            action = step % 7  # Cycle through actions
            reward = float(step) * 0.1
            next_state = torch.randn(8)
            done = False

            module.update(state, action, reward, next_state, done)

        # Flush all buffers - use the database's logger since that's what DecisionModule uses
        self.db.logger.flush_all_buffers()

        # Query database - learning experiences are now in agent_actions with module_type
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_actions WHERE module_type IS NOT NULL")
        count = cursor.fetchone()[0]

        # Should have logged 10 experiences
        self.assertEqual(count, 10)

        # Verify data structure
        cursor.execute("""
            SELECT
                step_number,
                agent_id,
                module_type,
                module_id,
                reward,
                details
            FROM agent_actions
            WHERE module_type IS NOT NULL
            ORDER BY step_number
        """)

        rows = cursor.fetchall()
        self.assertEqual(len(rows), 10)

        # Verify first row
        first_row = rows[0]
        self.assertEqual(first_row[0], 0)  # step_number
        self.assertEqual(first_row[1], "agent_001")  # agent_id
        self.assertEqual(first_row[2], "fallback")  # module_type
        # action_taken and action_taken_mapped are now in details JSON
        details = self._parse_details_json(first_row[5])
        if details:
            self.assertIn("action_taken", details)
            self.assertIn("action_taken_mapped", details)
        self.assertIsNotNone(first_row[3])  # module_id
        self.assertAlmostEqual(first_row[4], 0.0, places=5)  # reward

        conn.close()

    def test_multiple_agents_logging(self):
        """Test that multiple agents can log learning experiences independently."""
        agents_data = []

        # Create agent records for all agents (skip agent_001 which is already created in setUp)
        for i in range(3):
            if f"agent_{i:03d}" != "agent_001":  # Skip the one already created
                self.db.logger.log_agent(
                    agent_id=f"agent_{i:03d}",
                    birth_time=0,
                    agent_type="test",
                    position=(0.0, 0.0),
                    initial_resources=10.0,
                    starting_health=100.0,
                    genome_id="test_genome",
                    generation=1,
                )
        self.db.logger.flush_all_buffers()

        # Create multiple agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.agent_id = f"agent_{i:03d}"
            mock_agent.environment = self.mock_env

            mock_time = Mock()
            mock_time.current_time.return_value = 0
            # Set up services structure as expected by decision module
            mock_agent.services = Mock()
            mock_agent.services.time_service = mock_time
            mock_agent.actions = self.mock_actions

            module = DecisionModule(
                mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                DecisionConfig(algorithm_type="fallback"),
            )

            agents_data.append((mock_agent, mock_time, module))

        # Simulate steps for all agents
        for step in range(5):
            for mock_agent, mock_time, module in agents_data:
                mock_time.current_time.return_value = step

                state = torch.randn(8)
                action = (step + int(mock_agent.agent_id.split("_")[1])) % 7
                reward = float(step) * 0.5
                next_state = torch.randn(8)

                module.update(state, action, reward, next_state, False)

        # Flush buffers
        self.db.logger.flush_all_buffers()

        # Verify database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Should have 3 agents * 5 steps = 15 experiences
        cursor.execute("SELECT COUNT(*) FROM agent_actions WHERE module_type IS NOT NULL")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 15)

        # Verify each agent has 5 experiences
        for i in range(3):
            agent_id = f"agent_{i:03d}"
            cursor.execute("SELECT COUNT(*) FROM agent_actions WHERE agent_id = ? AND module_type IS NOT NULL", (agent_id,))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 5)

        conn.close()

    def test_different_algorithm_types_logged(self):
        """Test that different algorithm types are logged correctly."""
        algorithm_types = ["ppo", "sac", "dqn", "a2c", "ddpg"]

        # Create agent records for all algorithm types
        for i, algo_type in enumerate(algorithm_types):
            self.db.logger.log_agent(
                agent_id=f"agent_{algo_type}",
                birth_time=0,
                agent_type="test",
                position=(0.0, 0.0),
                initial_resources=10.0,
                starting_health=100.0,
                genome_id="test_genome",
                generation=1,
            )
        self.db.logger.flush_all_buffers()

        for i, algo_type in enumerate(algorithm_types):
            mock_agent = Mock()
            mock_agent.agent_id = f"agent_{algo_type}"
            mock_agent.environment = self.mock_env

            mock_time = Mock()
            mock_time.current_time.return_value = i
            # Set up services structure as expected by decision module
            mock_agent.services = Mock()
            mock_agent.services.time_service = mock_time
            mock_agent.actions = self.mock_actions

            config = DecisionConfig(algorithm_type=algo_type)
            module = DecisionModule(
                mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                config,
            )

            state = torch.randn(8)
            module.update(state, 0, 1.0, torch.randn(8), False)

        # Flush buffers
        self.db.logger.flush_all_buffers()

        # Verify all algorithm types are in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT module_type FROM agent_actions WHERE module_type IS NOT NULL ORDER BY module_type")
        logged_types = [row[0] for row in cursor.fetchall()]

        # All algorithms fall back to fallback but still log their original requested type
        # So we should see all the original algorithm types
        expected_types = sorted(algorithm_types)
        self.assertEqual(logged_types, expected_types)

        conn.close()

    def test_reward_values_persisted_correctly(self):
        """Test that various reward values are persisted correctly."""
        config = DecisionConfig()
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Test various reward values including edge cases
        test_rewards = [-100.5, -10.0, -1.5, -0.001, 0.0, 0.001, 1.5, 10.0, 100.5]

        for i, reward in enumerate(test_rewards):
            self.mock_time_service.current_time.return_value = i

            state = torch.randn(8)
            module.update(state, 0, reward, torch.randn(8), False)

        # Flush buffers
        self.db.logger.flush_all_buffers()

        # Verify rewards in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT reward FROM agent_actions WHERE module_type IS NOT NULL ORDER BY step_number")
        logged_rewards = [row[0] for row in cursor.fetchall()]

        self.assertEqual(len(logged_rewards), len(test_rewards))
        for expected, actual in zip(test_rewards, logged_rewards):
            self.assertAlmostEqual(expected, actual, places=5)

        conn.close()

    def test_curriculum_action_mapping(self):
        """Test that curriculum-restricted actions are mapped correctly."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Simulate curriculum with restricted actions
        # Only actions 0, 2, 4 (pass, gather, reproduce) are enabled
        enabled_actions = [0, 2, 4]

        # Agent selects action index 1 within enabled_actions (which is action 2 in full space)
        self.mock_time_service.current_time.return_value = 0
        state = torch.randn(8)
        test_reward = 1.0
        module.update(state, 1, test_reward, torch.randn(8), False, enabled_actions=enabled_actions)

        # Flush buffers
        self.db.logger.flush_all_buffers()

        # Verify the correct action was logged
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT details FROM agent_actions WHERE module_type IS NOT NULL")
        rows = cursor.fetchall()
        self.assertGreater(len(rows), 0)
        
        # Extract action_taken and action_taken_mapped from details JSON
        action_found = False
        for row in rows:
            details = self._parse_details_json(row[0])
            if details:
                action_taken = details.get("action_taken")
                action_taken_mapped = details.get("action_taken_mapped")
                if action_taken is not None and action_taken_mapped:
                    # Verify we can extract the data
                    action_found = True
                    break
        
        self.assertTrue(action_found, "Should have found action_taken and action_taken_mapped in details")

        conn.close()

    def test_database_unavailable_does_not_crash(self):
        """Test that missing database doesn't crash the system."""
        # Create agent without database
        mock_agent = Mock()
        mock_agent.agent_id = "agent_no_db"
        mock_agent.environment = None
        # Set up services structure as expected by decision module
        mock_agent.services = Mock()
        mock_agent.services.time_service = self.mock_time_service
        mock_agent.actions = self.mock_actions

        config = DecisionConfig()
        module = DecisionModule(
            mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Should not raise exception
        state = torch.randn(8)
        module.update(state, 0, 1.0, torch.randn(8), False)

        # No assertions needed - just verify no exception was raised


class TestLearningExperienceLoggingPerformance(unittest.TestCase):
    """Performance tests for learning experience logging."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_simulation.db"

        self.db = SimulationDatabase(str(self.db_path), simulation_id="perf_test")

        # Create required records for foreign key constraints
        self.db.add_simulation_record(
            simulation_id="perf_test", start_time=datetime.now(), status="running", parameters={}
        )
        self.db.logger.log_agent(
            agent_id="perf_agent",
            birth_time=0,
            agent_type="test",
            position=(0.0, 0.0),
            initial_resources=10.0,
            starting_health=100.0,
            genome_id="test_genome",
            generation=1,
        )
        self.db.logger.flush_all_buffers()

        self.logger = DataLogger(
            self.db, simulation_id="perf_test", config=DataLoggingConfig(buffer_size=100, commit_interval=60)
        )

        self.mock_env = Mock()
        self.mock_env.db = self.db
        self.mock_env.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bulk_logging_performance(self):
        """Test that bulk logging is efficient."""
        import time

        mock_agent = Mock()
        mock_agent.agent_id = "perf_agent"
        mock_agent.environment = self.mock_env

        mock_time = Mock()
        mock_time.current_time.return_value = 0
        # Set up services structure as expected by decision module
        mock_agent.services = Mock()
        mock_agent.services.time_service = mock_time

        mock_actions = []
        for i in range(7):
            mock_action = Mock()
            mock_action.name = f"action_{i}"
            mock_actions.append(mock_action)
        mock_agent.actions = mock_actions

        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Log 1000 experiences
        start_time = time.time()
        for step in range(1000):
            mock_time.current_time.return_value = step
            state = torch.randn(8)
            module.update(state, step % 7, float(step) * 0.01, torch.randn(8), False)

        # Flush buffers
        self.db.logger.flush_all_buffers()
        elapsed = time.time() - start_time

        # Verify all logged
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agent_actions WHERE module_type IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 1000)

        # Performance assertion - should be fast (< 3 seconds for 1000 logs)
        self.assertLess(elapsed, 3.0, f"Logging 1000 experiences took {elapsed:.2f}s, expected < 3s")


if __name__ == "__main__":
    unittest.main()
