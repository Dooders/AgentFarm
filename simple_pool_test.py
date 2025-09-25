#!/usr/bin/env python3
"""Simple test to understand pooling behavior without dependencies"""

import os
import sys
sys.path.insert(0, '/workspace')

# Enable pooling
os.environ["FARM_DISABLE_POOLING"] = "0"

# Simple mock agent class for testing
class MockAgent:
    def __init__(self, agent_id, position, resource_level):
        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True

    def reset(self, agent_id=None, position=None, resource_level=None, **kwargs):
        if agent_id is not None:
            self.agent_id = agent_id
        if position is not None:
            self.position = position
        if resource_level is not None:
            self.resource_level = resource_level
        self.alive = True

    def prepare_for_release(self):
        self.alive = False

def test_pool():
    print("Testing AgentPool directly...")

    from farm.core.pool import AgentPool

    # Create pool
    pool = AgentPool(MockAgent, max_size=50)

    print("Initial pool state:")
    print(f"  Total created: {pool.total_created}")
    print(f"  Total reused: {pool.total_reused}")
    print(f"  Pool size: {pool.size()}")

    # Acquire some agents
    agents = []
    for i in range(20):
        agent = pool.acquire(
            agent_id=f"agent_{i}",
            position=(i, i),
            resource_level=10
        )
        agents.append(agent)

    print("\nAfter acquiring 20 agents:")
    print(f"  Total created: {pool.total_created}")
    print(f"  Total reused: {pool.total_reused}")
    print(f"  Pool size: {pool.size()}")
    print(f"  Reuse rate: {(pool.total_reused / max(pool.total_created, 1)) * 100:.1f}%")

    # Release some agents back to pool
    for i in range(5):
        pool.release(agents[i])

    print("\nAfter releasing 5 agents:")
    print(f"  Total created: {pool.total_created}")
    print(f"  Total reused: {pool.total_reused}")
    print(f"  Pool size: {pool.size()}")
    print(f"  Reuse rate: {(pool.total_reused / max(pool.total_created, 1)) * 100:.1f}%")

    # Acquire more agents (should reuse from pool)
    for i in range(5):
        agent = pool.acquire(
            agent_id=f"new_agent_{i}",
            position=(i, i),
            resource_level=5
        )
        agents.append(agent)

    print("\nAfter acquiring 5 more agents (should reuse):")
    print(f"  Total created: {pool.total_created}")
    print(f"  Total reused: {pool.total_reused}")
    print(f"  Pool size: {pool.size()}")
    print(f"  Reuse rate: {(pool.total_reused / max(pool.total_created, 1)) * 100:.1f}%")

if __name__ == "__main__":
    test_pool()