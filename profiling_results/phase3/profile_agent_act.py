#!/usr/bin/env python3
"""Line profile agent action execution"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.core.environment import Environment
from farm.core.agent import BaseAgent

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch agent.act with profiling
original_act = BaseAgent.act

@profile
def profiled_act(self):
    return original_act(self)

BaseAgent.act = profiled_act

def main():
    """Run simulation with profiled agent actions."""
    config = SimulationConfig.from_centralized_config(environment="development")
    
    env = Environment(
        width=100,
        height=100,
        resource_distribution={"type": "random", "amount": 100},
        config=config,
    )
    
    # Create agents
    for i in range(10):
        agent = BaseAgent(
            agent_id=env.get_next_agent_id(),
            position=(50.0 + i, 50.0),
            resource_level=50,
            spatial_service=env.spatial_service,
            environment=env,
        )
        env.add_agent(agent)
    
    env.spatial_index.update()
    
    # Run agent actions
    print("Executing agent actions for profiling...")
    for _ in range(10):
        for agent in env.agent_objects:
            if agent.alive:
                agent.act()
        env.update()
    
    env.cleanup()
    print("Profiling complete!")

if __name__ == "__main__":
    main()
