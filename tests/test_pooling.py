#!/usr/bin/env python3
"""Simple test to understand pooling behavior"""

import os
import sys
sys.path.insert(0, '/workspace')

from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.core.pool import pooling_enabled

# Enable pooling
os.environ["FARM_DISABLE_POOLING"] = "0"

def test_pooling():
    print("Testing object pooling...")

    # Configure initial agent counts
    cfg = SimulationConfig(
        width=50,
        height=50,
        system_agents=100,
        independent_agents=0,
        control_agents=0,
        use_in_memory_db=True,
        persist_db_on_completion=False,
        max_steps=10,
        simulation_steps=10,
    )

    env = run_simulation(num_steps=10, config=cfg, save_config=False, path=None, seed=1234)

    # Check pool statistics
    if hasattr(env, "agent_pool") and env.agent_pool is not None:
        pool = env.agent_pool
        print("Pool Statistics:")
        print(f"  Total created: {pool.total_created}")
        print(f"  Total reused: {pool.total_reused}")
        print(f"  Reuse rate: {(pool.total_reused / max(pool.total_created, 1)) * 100:.1f}%")
        print(f"  Pool size: {pool.size()}")

        # Check if agents are being released
        print(f"  Active agents in environment: {len(env.agents)}")
        print(f"  Total agent objects: {len(env.agent_objects)}")

        # Count agents that should be dead (starved)
        dead_agents = [agent for agent in env.agent_objects if not agent.alive]
        print(f"  Dead agents: {len(dead_agents)}")
    else:
        print("No agent pool found!")

    env.cleanup()

if __name__ == "__main__":
    test_pooling()