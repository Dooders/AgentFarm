"""
Performance benchmarks for the component-based agent system.

These benchmarks measure the performance of agent creation, execution,
component access, and state serialization in the current system.
"""

import time
from unittest.mock import Mock
from farm.core.agent import AgentFactory, AgentConfig


def create_mock_services():
    """Create mock services for testing."""
    spatial_service = Mock()
    spatial_service.get_nearby = Mock(return_value={"agents": [], "resources": []})
    spatial_service.get_nearest = Mock(return_value={})
    spatial_service.mark_positions_dirty = Mock()
    
    time_service = Mock()
    time_service.current_time = Mock(side_effect=range(10000))
    
    lifecycle_service = Mock()
    lifecycle_service.get_next_agent_id = Mock(side_effect=lambda: f"agent_{time.time()}")
    lifecycle_service.add_agent = Mock()
    lifecycle_service.remove_agent = Mock()
    
    return spatial_service, time_service, lifecycle_service


class TestAgentCreationPerformance:
    """Benchmark agent creation."""
    
    def test_create_single_agent(self):
        """Benchmark creating a single agent."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        # Warm up
        for _ in range(10):
            factory.create_default_agent(
                agent_id=f"warmup",
                position=(0, 0),
                initial_resources=100
            )
        
        # Benchmark
        start = time.perf_counter()
        iterations = 1000
        
        for i in range(iterations):
            agent = factory.create_default_agent(
                agent_id=f"agent_{i}",
                position=(0, 0),
                initial_resources=100
            )
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000  # ms
        
        print(f"\nAgent Creation Performance:")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time:.4f}ms per agent")
        print(f"  Rate: {iterations/duration:.0f} agents/second")
        
        # Should be fast (< 1ms per agent)
        assert avg_time < 1.0, f"Agent creation too slow: {avg_time:.4f}ms"
    
    def test_create_agents_with_different_configs(self):
        """Benchmark creating agents with custom configs."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        configs = [
            AgentConfig(),
            AgentConfig(),
            AgentConfig(),
        ]
        
        start = time.perf_counter()
        iterations = 500
        
        for i in range(iterations):
            config = configs[i % len(configs)]
            agent = factory.create_default_agent(
                agent_id=f"agent_{i}",
                position=(i % 100, i % 100),
                initial_resources=100,
                config=config
            )
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000
        
        print(f"\nCustom Config Creation Performance:")
        print(f"  Average: {avg_time:.4f}ms per agent")
        
        assert avg_time < 1.5


class TestAgentExecutionPerformance:
    """Benchmark agent execution (act method)."""
    
    def test_single_agent_turns(self):
        """Benchmark single agent executing turns."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        agent = factory.create_default_agent(
            agent_id="benchmark",
            position=(50, 50),
            initial_resources=10000  # Enough to not die
        )
        
        # Warm up
        for _ in range(100):
            agent.act()
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        
        for i in range(iterations):
            time_service.current_time.return_value = i
            agent.act()
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000000  # microseconds
        
        print(f"\nSingle Agent Turn Performance:")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time:.2f}μs per turn")
        print(f"  Rate: {iterations/duration:.0f} turns/second")
        
        # Should be fast (< 100μs per turn)
        assert avg_time < 100, f"Agent turn too slow: {avg_time:.2f}μs"
    
    def test_multi_agent_turns(self):
        """Benchmark multiple agents executing turns."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        # Create population
        num_agents = 100
        agents = []
        for i in range(num_agents):
            agent = factory.create_default_agent(
                agent_id=f"agent_{i}",
                position=(i % 100, i // 100),
                initial_resources=10000
            )
            agents.append(agent)
        
        # Benchmark
        start = time.perf_counter()
        turns = 100
        
        for turn in range(turns):
            time_service.current_time.return_value = turn
            for agent in agents:
                if agent.alive:
                    agent.act()
        
        duration = time.perf_counter() - start
        total_turns = num_agents * turns
        avg_time = (duration / total_turns) * 1000000  # microseconds
        
        print(f"\nMulti-Agent Performance:")
        print(f"  Agents: {num_agents}")
        print(f"  Turns each: {turns}")
        print(f"  Total turns: {total_turns}")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Average: {avg_time:.2f}μs per turn")
        print(f"  Throughput: {total_turns/duration:.0f} turns/second")
        
        assert avg_time < 150


class TestComponentAccessPerformance:
    """Benchmark component access patterns."""
    
    def test_get_component_performance(self):
        """Benchmark getting components."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        agent = factory.create_default_agent(
            agent_id="test",
            position=(0, 0),
            initial_resources=100
        )
        
        # Benchmark
        start = time.perf_counter()
        iterations = 100000
        
        for _ in range(iterations):
            movement = agent.get_component("movement")
            resource = agent.get_component("resource")
            combat = agent.get_component("combat")
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000000  # microseconds
        
        print(f"\nComponent Access Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Total time: {duration:.3f}s")
        print(f"  Average: {avg_time:.3f}μs per 3 accesses")
        
        # Should be very fast (< 1μs per access)
        assert avg_time < 3.0
    
    def test_component_method_call_performance(self):
        """Benchmark calling component methods."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        agent = factory.create_default_agent(
            agent_id="test",
            position=(0, 0),
            initial_resources=100
        )
        
        movement = agent.get_component("movement")
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        
        for i in range(iterations):
            movement.move_by(0.1, 0.1)
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000000  # microseconds
        
        print(f"\nComponent Method Call Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time:.2f}μs per call")
        
        assert avg_time < 10.0


class TestStateSerializationPerformance:
    """Benchmark state serialization."""
    
    def test_save_state_performance(self):
        """Benchmark saving agent state."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        agent = factory.create_default_agent(
            agent_id="test",
            position=(50, 50),
            initial_resources=100
        )
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            state = agent.get_state_dict()
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000000  # microseconds
        
        print(f"\nState Save Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time:.2f}μs per save")
        
        assert avg_time < 50.0
    
    def test_load_state_performance(self):
        """Benchmark loading agent state."""
        spatial_service, time_service, lifecycle_service = create_mock_services()
        
        factory = AgentFactory(
            spatial_service=spatial_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )
        
        agent = factory.create_default_agent(
            agent_id="test",
            position=(50, 50),
            initial_resources=100
        )
        
        # Get state to load
        state = agent.get_state_dict()
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        
        for _ in range(iterations):
            agent.load_state_dict(state)
        
        duration = time.perf_counter() - start
        avg_time = (duration / iterations) * 1000000  # microseconds
        
        print(f"\nState Load Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Average: {avg_time:.2f}μs per load")
        
        assert avg_time < 50.0


def run_all_benchmarks():
    """Run all performance benchmarks and generate report."""
    print("\n" + "="*60)
    print("AGENT PERFORMANCE BENCHMARKS")
    print("="*60)
    
    test_classes = [
        TestAgentCreationPerformance,
        TestAgentExecutionPerformance,
        TestComponentAccessPerformance,
        TestStateSerializationPerformance,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 60)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                except AssertionError as e:
                    print(f"  ⚠️  {method_name}: {e}")
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()