#!/usr/bin/env python3
"""Test realistic simulation scenario with agent death and reuse"""

import os
import sys
sys.path.insert(0, '/workspace')

# Enable pooling
os.environ["FARM_DISABLE_POOLING"] = "0"

# Mock agent class for testing
class MockAgent:
    def __init__(self, agent_id, position, resource_level):
        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.starvation_counter = 0
        self.starvation_threshold = 100  # Realistic threshold like in config
        self.base_consumption_rate = 0.15  # Realistic consumption rate
        self.current_health = 100
        self.starting_health = 100
        self.episode_rewards = []
        self.losses = []
        self.total_reward = 0.0
        self.is_defending = False
        self.defense_timer = 0
        self.orientation = 0.0

    def reset(self, agent_id=None, position=None, resource_level=None, **kwargs):
        if agent_id is not None:
            self.agent_id = agent_id
        if position is not None:
            self.position = position
        if resource_level is not None:
            self.resource_level = resource_level
        self.alive = True
        self.starvation_counter = 0
        self.current_health = self.starting_health
        self.total_reward = 0.0
        self.is_defending = False
        self.defense_timer = 0
        self.orientation = 0.0

    def prepare_for_release(self):
        """Prepare this agent to be returned to an object pool."""
        # Clear cached selection state
        if hasattr(self, "_cached_selection_state"):
            self._cached_selection_state = None
            self._cached_selection_time = -1
        # Reset per-episode trackers
        self.previous_state = None
        if hasattr(self, "previous_state_tensor"):
            self.previous_state_tensor = None
        self.previous_action = None
        self._previous_action_index = 0
        self._previous_enabled_actions = None

        # Clear episode-specific data
        self.episode_rewards.clear()
        self.losses.clear()

        # Reset agent state to minimize memory retention
        self.starvation_counter = 0
        self.total_reward = 0.0
        self.current_health = self.starting_health  # Reset to full health
        self.is_defending = False
        self.defense_timer = 0
        self.orientation = 0.0

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

        # Consume resources each turn (realistic rate)
        self.resource_level -= self.base_consumption_rate

        # Check if agent should die from starvation
        if self.check_starvation():
            print(f"  Agent {self.agent_id} died from starvation after {self.starvation_counter} steps")

# Mock environment for testing
class MockEnvironment:
    def __init__(self, max_agents=100):
        from farm.core.pool import AgentPool
        self.agent_pool = AgentPool(MockAgent, max_size=max_agents)
        self.agents = []
        self.agent_objects = {}
        self.max_agents = max_agents
        self.step = 0

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
        except Exception as e:
            print(f"  Failed to release agent {agent.agent_id}: {e}")

    def get_next_agent_id(self):
        return f"agent_{len(self.agents)}"

    def run_step(self):
        """Run one simulation step"""
        self.step += 1
        agents_to_remove = []

        # Let all agents act
        for agent_id in list(self.agents):  # Use list() to avoid modification during iteration
            agent = self.agent_objects[agent_id]
            agent.act()
            if not agent.alive:
                agents_to_remove.append(agent)

        # Remove dead agents
        for agent in agents_to_remove:
            self.remove_agent(agent)

        # Create new agents to maintain population (simulate reproduction/replacement)
        while len(self.agents) < self.max_agents and self.agent_pool.size() > 0:
            agent = self.agent_pool.acquire(
                agent_id=self.get_next_agent_id(),
                position=(self.step % 50, self.step % 50),  # Move around
                resource_level=1.0  # Start with minimal resources
            )
            self.add_agent(agent)

        return len(agents_to_remove), len(self.agents)

def test_realistic_simulation():
    print("Testing realistic simulation scenario...")
    print("This simulates a long-running simulation where agents die and get reused.")

    # Create environment with limited population
    env = MockEnvironment(max_agents=50)

    # Create initial population
    print("Creating initial population of 50 agents...")
    for i in range(50):
        agent = env.agent_pool.acquire(
            agent_id=f"initial_{i}",
            position=(i % 20, i // 20),
            resource_level=2.0  # Start with some resources
        )
        env.add_agent(agent)

    print(f"Initial state: {len(env.agents)} agents, pool created={env.agent_pool.total_created}, reused={env.agent_pool.total_reused}")

    # Run simulation for enough steps to see agent death and reuse
    max_steps = 500
    death_events = 0

    for step in range(max_steps):
        deaths, current_agents = env.run_step()

        if deaths > 0:
            death_events += deaths
            if step % 50 == 0 or step < 10:  # Show first few steps and every 50 steps
                print(f"Step {step}: {deaths} deaths, {current_agents} alive, pool: created={env.agent_pool.total_created}, reused={env.agent_pool.total_reused}, size={env.agent_pool.size()}")

        # Stop if population stabilizes (no more deaths)
        if deaths == 0 and step > 100:
            break

    print("\nFinal Results:")
    print(f"  Simulation ran for {env.step} steps")
    print(f"  Total agent deaths: {death_events}")
    print(f"  Final population: {len(env.agents)} agents")
    print(f"  Total agents created: {env.agent_pool.total_created}")
    print(f"  Total agents reused: {env.agent_pool.total_reused}")
    print(f"  Reuse rate: {(env.agent_pool.total_reused / max(env.agent_pool.total_created, 1)) * 100:.1f}%")
    print(f"  Final pool size: {env.agent_pool.size()}")

    # Calculate effectiveness
    if death_events > 0:
        expected_reuse_rate = min(100.0, (death_events / env.agent_pool.total_created) * 100)
        actual_reuse_rate = (env.agent_pool.total_reused / env.agent_pool.total_created) * 100

        print("\nEffectiveness:")
        print(f"  Expected reuse rate (based on deaths): {expected_reuse_rate:.1f}%")
        print(f"  Actual reuse rate: {actual_reuse_rate:.1f}%")

        if actual_reuse_rate > 50:
            print("  ✅ Pooling is working effectively!")
        elif actual_reuse_rate > 10:
            print("  ⚠️  Pooling is working but could be more effective")
        else:
            print("  ❌ Pooling needs improvement")

    return env.agent_pool.total_reused / max(env.agent_pool.total_created, 1)

if __name__ == "__main__":
    reuse_rate = test_realistic_simulation()
    if reuse_rate > 0.5:
        print(f"\n🎉 SUCCESS: Pooling achieved {reuse_rate*100:.1f}% reuse rate!")
    else:
        print(f"\n⚠️  WARNING: Pooling only achieved {reuse_rate*100:.1f}% reuse rate")