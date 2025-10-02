#!/usr/bin/env python3
"""Line profile observation generation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.core.environment import Environment

# Import line_profiler
try:
    from line_profiler import profile
except ImportError:
    # Fallback decorator if line_profiler not available
    def profile(func):
        return func

# Patch the method with profiling decorator
original_get_observation = Environment._get_observation

@profile
def profiled_get_observation(self, agent_id):
    return original_get_observation(self, agent_id)

Environment._get_observation = profiled_get_observation

def main():
    """Run simulation with profiled observation generation."""
    config = SimulationConfig.from_centralized_config(environment="development")
    
    # Create environment
    env = Environment(
        width=100,
        height=100,
        resource_distribution={"type": "random", "amount": 100},
        config=config,
    )
    
    # Create test agents
    from farm.core.agent import BaseAgent
    for i in range(20):
        agent = BaseAgent(
            agent_id=env.get_next_agent_id(),
            position=(50.0, 50.0),
            resource_level=10,
            spatial_service=env.spatial_service,
            environment=env,
        )
        env.add_agent(agent)
    
    env.spatial_index.update()
    
    # Generate observations to profile
    print("Generating observations for profiling...")
    for _ in range(5):
        for agent_id in env.agents:
            _ = env._get_observation(agent_id)
    
    env.cleanup()
    print("Profiling complete!")

if __name__ == "__main__":
    main()
