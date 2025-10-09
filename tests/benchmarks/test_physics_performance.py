"""Performance benchmarks for physics engines.

This module contains benchmarks to ensure that the physics abstraction layer
doesn't introduce significant performance regressions.
"""

import time
import unittest
from typing import List

import numpy as np
import pytest

from farm.config.config import EnvironmentConfig, PopulationConfig, ResourceConfig, SimulationConfig
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.physics import create_physics_engine
from farm.core.resources import Resource


class TestPhysicsPerformance(unittest.TestCase):
    """Performance benchmarks for physics engines."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=200, height=200),
            population=PopulationConfig(system_agents=100, independent_agents=50, control_agents=25),
            resources=ResourceConfig(initial_resources=50),
            max_steps=100,
            seed=42,
        )
        
        self.physics = create_physics_engine(self.config, seed=42)
        self.env = Environment(
            physics_engine=self.physics,
            resource_distribution={"type": "random", "amount": 50},
            config=self.config,
            seed=42
        )

    def benchmark_position_validation(self):
        """Benchmark position validation performance."""
        # Generate test positions
        positions = []
        for _ in range(10000):
            x = np.random.uniform(-10, 210)  # Some outside bounds
            y = np.random.uniform(-10, 210)
            positions.append((x, y))
        
        # Benchmark position validation
        start_time = time.time()
        
        valid_count = 0
        for position in positions:
            if self.env.is_valid_position(position):
                valid_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance requirements: should validate 10K positions in < 0.1 seconds
        self.assertLess(duration, 0.1, f"Position validation took {duration:.3f}s, expected < 0.1s")
        
        # Should find reasonable number of valid positions
        self.assertGreater(valid_count, 8000)  # Most positions should be valid
        self.assertLess(valid_count, 10000)    # Some should be invalid
        
        print(f"Position validation: {len(positions)} positions in {duration:.3f}s "
              f"({len(positions)/duration:.0f} positions/sec)")

    def benchmark_nearby_queries(self):
        """Benchmark nearby entity queries."""
        # Add agents at random positions
        agents = []
        for i in range(1000):
            agent = BaseAgent(
                agent_id=f"benchmark_agent_{i}",
                position=(np.random.uniform(0, 200), np.random.uniform(0, 200)),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Generate test queries
        queries = []
        for _ in range(1000):
            position = (np.random.uniform(0, 200), np.random.uniform(0, 200))
            radius = np.random.uniform(5, 50)
            queries.append((position, radius))
        
        # Benchmark nearby agent queries
        start_time = time.time()
        
        total_found = 0
        for position, radius in queries:
            nearby = self.env.get_nearby_agents(position, radius)
            total_found += len(nearby)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance requirements: should handle 1K queries in < 0.5 seconds
        self.assertLess(duration, 0.5, f"Nearby queries took {duration:.3f}s, expected < 0.5s")
        
        # Should find reasonable number of agents
        self.assertGreater(total_found, 0)
        
        print(f"Nearby queries: {len(queries)} queries in {duration:.3f}s "
              f"({len(queries)/duration:.0f} queries/sec, {total_found} total results)")

    def benchmark_spatial_index_updates(self):
        """Benchmark spatial index updates."""
        # Add agents
        agents = []
        for i in range(1000):
            agent = BaseAgent(
                agent_id=f"update_agent_{i}",
                position=(np.random.uniform(0, 200), np.random.uniform(0, 200)),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Benchmark position updates
        start_time = time.time()
        
        for i in range(1000):
            # Move agents to new positions
            for agent in agents:
                new_position = (
                    np.random.uniform(0, 200),
                    np.random.uniform(0, 200)
                )
                agent.position = new_position
            
            # Mark positions as dirty and process updates
            self.env.mark_positions_dirty()
            self.env.process_batch_spatial_updates(force=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance requirements: should handle 1K updates in < 2.0 seconds
        self.assertLess(duration, 2.0, f"Spatial updates took {duration:.3f}s, expected < 2.0s")
        
        print(f"Spatial updates: {1000} update cycles in {duration:.3f}s "
              f"({1000/duration:.0f} cycles/sec)")

    def benchmark_observation_generation(self):
        """Benchmark observation generation."""
        # Add agents
        agents = []
        for i in range(100):
            agent = BaseAgent(
                agent_id=f"obs_agent_{i}",
                position=(np.random.uniform(0, 200), np.random.uniform(0, 200)),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Benchmark observation generation
        start_time = time.time()
        
        total_observations = 0
        for _ in range(100):  # 100 observation cycles
            for agent in agents:
                observation = self.env.observe(agent.agent_id)
                total_observations += 1
                self.assertIsNotNone(observation)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance requirements: should generate 10K observations in < 1.0 seconds
        self.assertLess(duration, 1.0, f"Observation generation took {duration:.3f}s, expected < 1.0s")
        
        print(f"Observation generation: {total_observations} observations in {duration:.3f}s "
              f"({total_observations/duration:.0f} observations/sec)")

    def benchmark_full_simulation_step(self):
        """Benchmark complete simulation step performance."""
        # Add agents
        agents = []
        for i in range(1000):
            agent = BaseAgent(
                agent_id=f"sim_agent_{i}",
                position=(np.random.uniform(0, 200), np.random.uniform(0, 200)),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Benchmark simulation steps
        start_time = time.time()
        
        for step in range(10):  # 10 simulation steps
            self.env.step()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance requirements: should complete 10 steps in < 5.0 seconds
        self.assertLess(duration, 5.0, f"Simulation steps took {duration:.3f}s, expected < 5.0s")
        
        print(f"Simulation steps: 10 steps in {duration:.3f}s "
              f"({10/duration:.1f} steps/sec)")

    def test_performance_regression(self):
        """Test that performance hasn't regressed significantly."""
        # Run all benchmarks
        self.benchmark_position_validation()
        self.benchmark_nearby_queries()
        self.benchmark_spatial_index_updates()
        self.benchmark_observation_generation()
        self.benchmark_full_simulation_step()


class TestPhysicsMemoryUsage(unittest.TestCase):
    """Test memory usage of physics engines."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=100, height=100),
            population=PopulationConfig(system_agents=50, independent_agents=25, control_agents=10),
            resources=ResourceConfig(initial_resources=25),
            max_steps=50,
            seed=42,
        )

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and run simulation
        physics = create_physics_engine(self.config, seed=42)
        env = Environment(
            physics_engine=physics,
            resource_distribution={"type": "random", "amount": 25},
            config=self.config,
            seed=42
        )
        
        # Run simulation for many steps
        for step in range(100):
            env.step()
            
            # Check memory usage every 20 steps
            if step % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (< 100MB)
                self.assertLess(memory_increase, 100, 
                              f"Memory increased by {memory_increase:.1f}MB at step {step}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"Memory usage: {total_increase:.1f}MB increase over 100 steps")
        
        # Total memory increase should be reasonable
        self.assertLess(total_increase, 200, 
                       f"Total memory increase {total_increase:.1f}MB is too high")

    def test_large_environment_performance(self):
        """Test performance with large environment."""
        # Create large environment
        large_config = SimulationConfig(
            environment=EnvironmentConfig(width=500, height=500),
            population=PopulationConfig(system_agents=200, independent_agents=100, control_agents=50),
            resources=ResourceConfig(initial_resources=100),
            max_steps=20,
            seed=42,
        )
        
        physics = create_physics_engine(large_config, seed=42)
        env = Environment(
            physics_engine=physics,
            resource_distribution={"type": "random", "amount": 100},
            config=large_config,
            seed=42
        )
        
        # Benchmark large environment operations
        start_time = time.time()
        
        # Run simulation
        for step in range(10):
            env.step()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle large environment reasonably well
        self.assertLess(duration, 10.0, f"Large environment took {duration:.3f}s, expected < 10.0s")
        
        print(f"Large environment: 10 steps in {duration:.3f}s "
              f"({10/duration:.1f} steps/sec)")


if __name__ == "__main__":
    unittest.main()
