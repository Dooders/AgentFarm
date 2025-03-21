import unittest
import time
import random
import os
import json
import threading
from typing import List, Dict, Tuple
from database.database import SimulationDatabase, AgentState

class TestDatabasePerformance(unittest.TestCase):
    """Performance test suite for SimulationDatabase."""
    
    def setUp(self):
        """Set up test database and test data."""
        self.test_db_path = f"test_performance_{time.time()}.db"  # Unique DB file for each test
        self.db = SimulationDatabase(self.test_db_path)
        self.num_agents = 1000
        self.num_steps = 100
        self.used_agent_ids = set()  # Track used agent IDs
        self._lock = threading.Lock()  # Lock for thread-safe ID generation
        
    def tearDown(self):
        """Clean up test database."""
        try:
            self.db.close()
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
        except Exception as e:
            print(f"Warning: Cleanup failed - {e}")
            
    def _generate_unique_agent_id(self) -> int:
        """Generate a unique agent ID thread-safely."""
        with self._lock:
            while True:
                agent_id = random.randint(1, 100000)
                if agent_id not in self.used_agent_ids:
                    self.used_agent_ids.add(agent_id)
                    return agent_id
            
    def _generate_random_agent_data(self) -> Dict:
        """Generate random agent data for testing."""
        return {
            "agent_id": self._generate_unique_agent_id(),
            "birth_time": random.randint(0, 100),
            "agent_type": random.choice(["SystemAgent", "IndependentAgent", "ControlAgent"]),
            "position": (random.uniform(0, 100), random.uniform(0, 100)),
            "initial_resources": random.uniform(0, 100),
            "starting_health": random.uniform(50, 100),
            "starvation_threshold": random.randint(10, 50),
            "genome_id": f"genome_{random.randint(1, 1000)}",
            "generation": random.randint(0, 10)
        }
        
    def _generate_random_state_data(self) -> Dict:
        """Generate random state data for testing."""
        return {
            "current_health": random.uniform(0, 100),
            "resource_level": random.uniform(0, 100),
            "position": (random.uniform(0, 100), random.uniform(0, 100)),
            "is_defending": random.choice([True, False]),
            "total_reward": random.uniform(-100, 100),
            "starvation_threshold": random.randint(10, 50)
        }

    def test_batch_agent_creation_performance(self):
        """Test performance of creating multiple agents in batch."""
        start_time = time.time()
        
        # Generate agent data
        agent_data_list = [self._generate_random_agent_data() for _ in range(self.num_agents)]
        
        # Use batch insert
        self.db.log_agents_batch(agent_data_list)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance (adjust threshold as needed)
        self.assertLess(execution_time, 5.0, 
                       f"Batch agent creation took too long: {execution_time:.2f} seconds")
        
    def test_state_update_performance(self):
        """Test performance of updating agent states."""
        # Create test agent
        agent_data = self._generate_random_agent_data()
        self.db.log_agent(
            agent_data["agent_id"],
            agent_data["birth_time"],
            agent_data["agent_type"],
            agent_data["position"],
            agent_data["initial_resources"],
            agent_data["starting_health"],
            agent_data["starvation_threshold"]
        )
        
        start_time = time.time()
        
        # Generate all state updates at once
        state_updates = []
        for step in range(self.num_steps):
            state_data = self._generate_random_state_data()
            state_updates.append({
                'agent_id': agent_data["agent_id"],
                'step_number': step,
                'state_data': state_data
            })
            
        # Perform batch update
        def _batch_update(session):
            for update in state_updates:
                state_data = update['state_data']
                agent_state = AgentState(
                    step_number=update['step_number'],
                    agent_id=update['agent_id'],
                    current_health=state_data["current_health"],
                    starting_health=100.0,  # Default for test
                    resource_level=state_data["resource_level"],
                    position_x=state_data["position"][0],
                    position_y=state_data["position"][1],
                    is_defending=state_data["is_defending"],
                    total_reward=state_data["total_reward"],
                    starvation_threshold=state_data["starvation_threshold"],
                    age=update['step_number']
                )
                session.add(agent_state)
            session.commit()
            
        self.db._execute_in_transaction(_batch_update)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance
        self.assertLess(execution_time, 2.0,
                       f"State updates took too long: {execution_time:.2f} seconds")
        
    def test_query_performance(self):
        """Test performance of various query operations."""
        # Create test agent
        agent_data = self._generate_random_agent_data()
        self.db.log_agent(
            agent_data["agent_id"],
            agent_data["birth_time"],
            agent_data["agent_type"],
            agent_data["position"],
            agent_data["initial_resources"],
            agent_data["starting_health"],
            agent_data["starvation_threshold"]
        )
        
        # Add some state data
        for step in range(self.num_steps):
            state_data = self._generate_random_state_data()
            self.db.update_agent_state(agent_data["agent_id"], step, state_data)
        
        # Test various query operations
        start_time = time.time()
        
        # Test get_agent_data
        agent_info = self.db.get_agent_data(agent_data["agent_id"])
        self.assertIsNotNone(agent_info)
        
        # Test get_agent_types
        agent_types = self.db.get_agent_types()
        self.assertIsInstance(agent_types, list)
        
        # Test get_agent_behaviors
        behaviors = self.db.get_agent_behaviors()
        self.assertIsInstance(behaviors, list)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable query performance
        self.assertLess(execution_time, 1.0,
                       f"Query operations took too long: {execution_time:.2f} seconds")
        
    def test_concurrent_access_performance(self):
        """Test database performance under concurrent access."""
        def worker(worker_id: int):
            """Worker function for concurrent testing."""
            for _ in range(10):
                agent_data = self._generate_random_agent_data()
                self.db.log_agent(
                    agent_data["agent_id"],
                    agent_data["birth_time"],
                    agent_data["agent_type"],
                    agent_data["position"],
                    agent_data["initial_resources"],
                    agent_data["starting_health"],
                    agent_data["starvation_threshold"]
                )
                
        start_time = time.time()
        
        # Create and start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable concurrent performance
        self.assertLess(execution_time, 3.0,
                       f"Concurrent operations took too long: {execution_time:.2f} seconds")
        
    def test_buffer_performance(self):
        """Test performance of buffer operations."""
        # First create some agents to satisfy foreign key constraints
        agent_ids = []
        for _ in range(10):
            agent_data = self._generate_random_agent_data()
            self.db.log_agent(
                agent_data["agent_id"],
                agent_data["birth_time"],
                agent_data["agent_type"],
                agent_data["position"],
                agent_data["initial_resources"],
                agent_data["starting_health"],
                agent_data["starvation_threshold"]
            )
            agent_ids.append(agent_data["agent_id"])
            
        start_time = time.time()
        
        # Add multiple actions to buffer
        for _ in range(self.num_agents):
            self.db.log_agent_action(
                step_number=random.randint(0, 100),
                agent_id=random.choice(agent_ids),  # Use existing agent IDs
                action_type=random.choice(["move", "attack", "defend", "share"]),
                action_target_id=random.choice(agent_ids),  # Use existing agent IDs
                resources_before=random.uniform(0, 100),
                resources_after=random.uniform(0, 100),
                reward=random.uniform(-10, 10)
            )
            
        # Flush buffers
        self.db.flush_all_buffers()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable buffer performance
        self.assertLess(execution_time, 2.0,
                       f"Buffer operations took too long: {execution_time:.2f} seconds")
        
    def test_export_performance(self):
        """Test performance of data export operations."""
        # Create test data
        for _ in range(100):
            agent_data = self._generate_random_agent_data()
            self.db.log_agent(
                agent_data["agent_id"],
                agent_data["birth_time"],
                agent_data["agent_type"],
                agent_data["position"],
                agent_data["initial_resources"],
                agent_data["starting_health"],
                agent_data["starvation_threshold"]
            )
            
        start_time = time.time()
        
        # Test different export formats
        formats = ["csv", "json"]  # Removed 'excel' as it requires additional dependencies
        for format_type in formats:
            self.db.export_data(
                f"test_export.{format_type}",
                format=format_type,
                data_types=["metrics", "agents", "resources", "actions"]
            )
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable export performance
        self.assertLess(execution_time, 5.0,
                       f"Export operations took too long: {execution_time:.2f} seconds")
        
        # Clean up export files
        for format_type in formats:
            if os.path.exists(f"test_export.{format_type}"):
                os.remove(f"test_export.{format_type}")

    def test_configuration_performance(self):
        """Test performance of configuration operations."""
        start_time = time.time()
        
        # Generate test configuration
        test_config = {
            "simulation_params": {
                "world_size": [1000, 1000],  # Changed to list for JSON compatibility
                "initial_agents": 100,
                "resource_spawn_rate": 0.1,
                "max_steps": 1000
            },
            "agent_params": {
                "vision_range": 50,
                "max_speed": 5,
                "metabolism_rate": 0.1,
                "reproduction_threshold": 100
            },
            "environment_params": {
                "temperature_range": [-10, 40],  # Changed to list for JSON compatibility
                "weather_patterns": ["sunny", "rainy", "stormy"],
                "terrain_types": ["grass", "water", "mountain"]
            }
        }
        
        # Test saving and retrieving configuration multiple times
        for _ in range(100):
            self.db.save_configuration(test_config)
            retrieved_config = self.db.get_configuration()
            self.assertEqual(retrieved_config["simulation_params"]["world_size"], [1000, 1000])
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable configuration performance
        self.assertLess(execution_time, 2.0,
                       f"Configuration operations took too long: {execution_time:.2f} seconds")

    def test_large_scale_performance(self):
        """Test database performance with large-scale data operations."""
        start_time = time.time()
        
        # Create a large number of agents
        num_large_agents = 5000
        agent_data_list = [self._generate_random_agent_data() for _ in range(num_large_agents)]
        self.db.log_agents_batch(agent_data_list)
        
        # Create multiple states for each agent
        num_states_per_agent = 10
        for agent_data in agent_data_list[:100]:  # Test with first 100 agents
            for step in range(num_states_per_agent):
                state_data = self._generate_random_state_data()
                self.db.update_agent_state(agent_data["agent_id"], step, state_data)
        
        # Perform various queries
        agent_types = self.db.get_agent_types()
        self.assertGreater(len(agent_types), 0)
        
        behaviors = self.db.get_agent_behaviors()
        self.assertIsInstance(behaviors, list)
        
        # Get population statistics instead of advanced statistics
        stats = self.db.get_population_statistics()
        self.assertIsInstance(stats, dict)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance for large-scale operations
        self.assertLess(execution_time, 30.0,
                       f"Large-scale operations took too long: {execution_time:.2f} seconds")

    def test_transaction_rollback_performance(self):
        """Test performance of transaction rollback under error conditions."""
        start_time = time.time()
        
        # Create some initial agents
        agent_data_list = [self._generate_random_agent_data() for _ in range(10)]
        self.db.log_agents_batch(agent_data_list)
        
        # Attempt operations that should trigger rollbacks
        for _ in range(100):
            try:
                # Try to create agent with duplicate ID (should fail and rollback)
                self.db.log_agent(
                    agent_data_list[0]["agent_id"],  # Duplicate ID
                    0, "TestAgent", (0, 0), 100, 100, 10
                )
            except Exception:
                pass  # Expected to fail
                
            try:
                # Try to update non-existent agent (should fail and rollback)
                self.db.update_agent_state(999999, 0, self._generate_random_state_data())
            except Exception:
                pass  # Expected to fail
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable rollback performance
        self.assertLess(execution_time, 2.0,
                       f"Transaction rollbacks took too long: {execution_time:.2f} seconds")

    def test_concurrent_read_write_performance(self):
        """Test performance under concurrent read and write operations."""
        def reader_worker():
            """Worker function for read operations."""
            for _ in range(50):
                self.db.get_agent_types()
                self.db.get_agent_behaviors()
                self.db.get_population_statistics()  # Changed from advanced_statistics
                
        def writer_worker():
            """Worker function for write operations."""
            for _ in range(20):
                agent_data = self._generate_random_agent_data()
                self.db.log_agent(
                    agent_data["agent_id"],
                    agent_data["birth_time"],
                    agent_data["agent_type"],
                    agent_data["position"],
                    agent_data["initial_resources"],
                    agent_data["starting_health"],
                    agent_data["starvation_threshold"]
                )
                
        start_time = time.time()
        
        # Create and start reader and writer threads
        threads = []
        num_readers = 3
        num_writers = 2
        
        for _ in range(num_readers):
            thread = threading.Thread(target=reader_worker)
            threads.append(thread)
            thread.start()
            
        for _ in range(num_writers):
            thread = threading.Thread(target=writer_worker)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable concurrent read/write performance
        self.assertLess(execution_time, 5.0,
                       f"Concurrent read/write operations took too long: {execution_time:.2f} seconds")

if __name__ == "__main__":
    unittest.main() 