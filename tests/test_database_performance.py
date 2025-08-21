import unittest
import time
import random
import os
import json
import threading
from typing import List, Dict, Tuple
from farm.database.database import SimulationDatabase
from farm.database.data_logging import DataLogger
from farm.database.models import AgentStateModel

class TestDatabasePerformance(unittest.TestCase):
    """Performance test suite for SimulationDatabase."""
    
    def setUp(self):
        """Set up test database and test data."""
        self.test_db_path = f"test_performance_{time.time()}.db"  # Unique DB file for each test
        self.db = SimulationDatabase(self.test_db_path)
        self.logger = DataLogger(self.db, simulation_id="test_simulation")
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
        self.logger.log_agents_batch(agent_data_list)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance (adjust threshold as needed)
        self.assertLess(execution_time, 5.0, 
                       f"Batch agent creation took too long: {execution_time:.2f} seconds")
        
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

if __name__ == "__main__":
    unittest.main() 