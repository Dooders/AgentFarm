#!/usr/bin/env python3
"""Test agent reuse after death and pool release"""

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
        self.starvation_counter = 0
        self.starvation_threshold = 3  # Die after 3 turns of no resources

    def reset(self, agent_id=None, position=None, resource_level=None, **kwargs):
        if agent_id is not None:
            self.agent_id = agent_id
        if position is not None:
            self.position = position
        if resource_level is not None:
            self.resource_level = resource_level
        self.alive = True
        self.starvation_counter = 0

    def prepare_for_release(self):
        self.alive = False
        self.starvation_counter = 0

    def check_starvation(self):
        """Check if agent should die from starvation"""
        if self.resource_level <= 0:
            self.starvation_counter += 1
            if self.starvation_counter >= self.starvation_threshold:
                self.alive = False
                return True
        else:
            self.starvation_counter = 0
        return False

    def act(self):
        """Simulate agent action - consume resources"""
        if not self.alive:
            return

        # Consume resources each turn
        self.resource_level -= 1

        # Check if agent should die from starvation
        if self.check_starvation():
            print(f"  Agent {self.agent_id} died from starvation")

# Mock environment for testing
class MockEnvironment:
    def __init__(self):
        from farm.core.pool import AgentPool
        self.agent_pool = AgentPool(MockAgent, max_size=50)
        self.agents = []
        self.agent_objects = {}

    def add_agent(self, agent):
        self.agents.append(agent.agent_id)
        self.agent_objects[agent.agent_id] = agent

    def remove_agent(self, agent):
        """Remove agent and release to pool"""
        if agent.agent_id in self.agent_objects:
            del self.agent_objects[agent.agent_id]
        if agent.agent_id in self.agents:
            self.agents.remove(agent.agent_id)

        # Release to pool for reuse
        try:
            if self.agent_pool is not None:
                self.agent_pool.release(agent)
                print(f"  Released agent {agent.agent_id} to pool")
        except Exception as e:
            print(f"  Failed to release agent {agent.agent_id}: {e}")

    def get_next_agent_id(self):
        return f"agent_{len(self.agents)}"

def test_reuse():
    print("Testing agent reuse after death and pool release...")

    env = MockEnvironment()

    # Create initial agents with low resources
    print("Creating 3 initial agents with resource_level=1:")
    for i in range(3):
        agent = env.agent_pool.acquire(
            agent_id=f"agent_{i}",
            position=(i, i),
            resource_level=1  # Will die quickly
        )
        env.add_agent(agent)

    print(f"Initial pool state: created={env.agent_pool.total_created}, reused={env.agent_pool.total_reused}, size={env.agent_pool.size()}")

    # Simulate steps until agents die
    step = 0
    while step < 10:
        print(f"\nStep {step}:")
        print(f"  Active agents: {len(env.agents)}")
        print(f"  Pool: created={env.agent_pool.total_created}, reused={env.agent_pool.total_reused}, size={env.agent_pool.size()}")

        # Let all agents act (consume resources)
        agents_to_remove = []
        for agent_id in env.agents:
            agent = env.agent_objects[agent_id]
            agent.act()
            if not agent.alive:
                agents_to_remove.append(agent)

        # Remove dead agents (this should trigger pool release)
        for agent in agents_to_remove:
            print(f"  Agent {agent.agent_id} died, removing...")
            env.remove_agent(agent)

        # Create new agents from the pool (should reuse released agents)
        new_agents = 0
        while env.agent_pool.size() > 0 and new_agents < len(agents_to_remove):
            agent = env.agent_pool.acquire(
                agent_id=env.get_next_agent_id(),
                position=(new_agents, new_agents),
                resource_level=1
            )
            env.add_agent(agent)
            new_agents += 1
            print(f"  Created new agent {agent.agent_id} from pool")

        step += 1

        # Stop if no more agents are dying and we have a stable population
        if len(agents_to_remove) == 0 and step > 3:
            break

    print("\nFinal results:")
    print(f"  Total agents created: {env.agent_pool.total_created}")
    print(f"  Total agents reused: {env.agent_pool.total_reused}")
    print(f"  Reuse rate: {(env.agent_pool.total_reused / max(env.agent_pool.total_created, 1)) * 100:.1f}%")
    print(f"  Final pool size: {env.agent_pool.size()}")
    print(f"  Active agents remaining: {len(env.agents)}")

if __name__ == "__main__":
    test_reuse()