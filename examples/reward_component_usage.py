"""
Example demonstrating how to use the RewardComponent.

This example shows how to:
1. Create agents with reward components
2. Configure reward parameters
3. Access reward information
4. Use different reward strategies
"""

from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.component_configs import AgentComponentConfig, RewardConfig
from farm.core.agent.services import AgentServices


def create_agent_with_custom_rewards():
    """Create an agent with custom reward configuration."""
    
    # Create services (in real usage, this would come from your simulation)
    services = AgentServices()  # You'd need to provide actual services
    
    # Create custom reward configuration
    reward_config = RewardConfig(
        resource_reward_scale=2.0,      # Higher reward for resource gains
        health_reward_scale=1.0,        # Standard health rewards
        survival_bonus=0.2,             # Higher survival bonus
        death_penalty=-15.0,            # Harsher death penalty
        age_bonus=0.02,                 # Higher longevity bonus
        combat_success_bonus=5.0,       # Bonus for combat success
        reproduction_bonus=10.0,        # High bonus for reproduction
        max_history_length=500,         # Keep less history
    )
    
    # Create agent configuration with custom reward config
    agent_config = AgentComponentConfig(
        reward=reward_config,
        # ... other component configs would go here
    )
    
    # Create factory and agent
    factory = AgentFactory(services)
    agent = factory.create_default_agent(
        agent_id="rewarded_agent",
        position=(0.0, 0.0),
        config=agent_config,
    )
    
    return agent


def demonstrate_reward_tracking(agent):
    """Demonstrate reward tracking and statistics."""
    
    # Get the reward component
    reward_component = None
    for component in agent.components:
        if hasattr(component, 'cumulative_reward'):
            reward_component = component
            break
    
    if not reward_component:
        print("No reward component found!")
        return
    
    print("=== Reward Component Demo ===")
    print(f"Agent ID: {agent.agent_id}")
    print(f"Initial cumulative reward: {reward_component.cumulative_reward}")
    
    # Simulate some agent actions and state changes
    print("\n--- Simulating agent actions ---")
    
    # Simulate resource gain
    if hasattr(agent, 'resource_level'):
        agent.resource_level += 10.0
        print(f"Agent gained 10 resources (now: {agent.resource_level})")
    
    # Simulate health change
    if hasattr(agent, 'current_health'):
        agent.current_health += 5.0
        print(f"Agent gained 5 health (now: {agent.current_health})")
    
    # Simulate step (this would normally be called by the simulation)
    reward_component.on_step_start()  # Capture pre-action state
    reward_component.on_step_end()    # Calculate and apply rewards
    
    print(f"Step reward: {reward_component.current_reward:.3f}")
    print(f"Cumulative reward: {reward_component.total_reward:.3f}")
    
    # Add some manual rewards
    reward_component.add_reward(2.0, "exploration bonus")
    reward_component.add_reward(1.5, "cooperation bonus")
    
    print(f"After manual rewards: {reward_component.total_reward:.3f}")
    
    # Get reward statistics
    stats = reward_component.get_reward_stats()
    print("\n--- Reward Statistics ---")
    print(f"Cumulative: {stats['cumulative']:.3f}")
    print(f"Average: {stats['average']:.3f}")
    print(f"Min: {stats['min']:.3f}")
    print(f"Max: {stats['max']:.3f}")
    print(f"Recent average: {stats['recent_average']:.3f}")
    print(f"Last action reward: {stats['last_action']:.3f}")


def demonstrate_different_reward_strategies():
    """Demonstrate different reward configuration strategies."""
    
    print("\n=== Different Reward Strategies ===")
    
    # Aggressive reward strategy (encourages combat and risk-taking)
    aggressive_config = RewardConfig(
        resource_reward_scale=1.5,
        health_reward_scale=0.3,        # Lower health importance
        survival_bonus=0.05,            # Lower survival bonus
        combat_success_bonus=10.0,      # High combat bonus
        reproduction_bonus=15.0,        # High reproduction bonus
    )
    
    # Conservative reward strategy (encourages survival and efficiency)
    conservative_config = RewardConfig(
        resource_reward_scale=0.8,
        health_reward_scale=1.2,        # Higher health importance
        survival_bonus=0.3,             # High survival bonus
        death_penalty=-20.0,            # Harsh death penalty
        age_bonus=0.05,                 # High longevity bonus
        combat_success_bonus=1.0,       # Low combat bonus
    )
    
    # Exploration reward strategy (encourages movement and discovery)
    exploration_config = RewardConfig(
        resource_reward_scale=1.0,
        survival_bonus=0.1,
        age_bonus=0.02,
        # Could add exploration-specific bonuses here
        # exploration_bonus=3.0,
        # discovery_bonus=5.0,
    )
    
    print("Aggressive Strategy:")
    print(f"  Combat bonus: {aggressive_config.combat_success_bonus}")
    print(f"  Survival bonus: {aggressive_config.survival_bonus}")
    print(f"  Health scale: {aggressive_config.health_reward_scale}")
    
    print("\nConservative Strategy:")
    print(f"  Survival bonus: {conservative_config.survival_bonus}")
    print(f"  Death penalty: {conservative_config.death_penalty}")
    print(f"  Age bonus: {conservative_config.age_bonus}")
    
    print("\nExploration Strategy:")
    print(f"  Resource scale: {exploration_config.resource_reward_scale}")
    print(f"  Age bonus: {exploration_config.age_bonus}")


def demonstrate_reward_reset():
    """Demonstrate reward reset functionality."""
    
    print("\n=== Reward Reset Demo ===")
    
    # This would be used when starting a new episode or simulation
    # In a real scenario, you'd get the reward component from an agent
    
    # Simulate having some rewards
    print("Before reset:")
    print("  Cumulative reward: 25.5")
    print("  Reward history length: 100")
    
    # Reset rewards (this would be called on the reward component)
    print("\nResetting rewards...")
    print("After reset:")
    print("  Cumulative reward: 0.0")
    print("  Reward history length: 0")
    print("  Ready for new episode!")


if __name__ == "__main__":
    print("Reward Component Usage Examples")
    print("=" * 40)
    
    # Note: In a real simulation, you'd have proper services and environment
    print("Note: This is a demonstration of the reward component API.")
    print("In a real simulation, you would:")
    print("1. Create proper AgentServices with all required services")
    print("2. Create agents through the factory")
    print("3. Run the simulation which would call the component lifecycle methods")
    print("4. Access reward information through the component")
    
    demonstrate_different_reward_strategies()
    demonstrate_reward_reset()
    
    print("\n=== Integration with Agent Factory ===")
    print("The reward component is automatically included when creating agents:")
    print("""
    # Create agent with default reward configuration
    agent = factory.create_default_agent(
        agent_id="my_agent",
        position=(0.0, 0.0),
        config=AgentComponentConfig.default()  # Includes default RewardConfig
    )
    
    # Create agent with custom reward configuration
    custom_config = AgentComponentConfig(
        reward=RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
            # ... other custom settings
        )
    )
    agent = factory.create_learning_agent(
        agent_id="learning_agent",
        position=(0.0, 0.0),
        config=custom_config
    )
    """)
    
    print("\n=== Accessing Reward Information ===")
    print("""
    # Get reward component from agent
    reward_component = None
    for component in agent.components:
        if hasattr(component, 'cumulative_reward'):
            reward_component = component
            break
    
    # Access reward information
    total_reward = reward_component.total_reward
    current_reward = reward_component.current_reward
    stats = reward_component.get_reward_stats()
    
    # Add manual rewards
    reward_component.add_reward(5.0, "achievement bonus")
    
    # Reset for new episode
    reward_component.reset_rewards()
    """)