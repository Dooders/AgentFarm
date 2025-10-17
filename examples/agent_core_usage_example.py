#!/usr/bin/env python3
"""
Example: Using the New Component-Based AgentCore System

This script demonstrates how to use the new AgentCore architecture to create
agents with different configurations, run simulations, and access components.

Key features demonstrated:
- Creating agents with AgentFactory
- Using different agent types (default, learning, minimal, aggressive, etc.)
- Accessing and configuring agent components
- Running agent steps in a simulation loop
- Customizing agent behavior
"""

from unittest.mock import Mock

from farm.core.agent import (
    AgentCore,
    AgentFactory,
    AgentServices,
    AgentComponentConfig,
    DefaultAgentBehavior,
    LearningAgentBehavior,
)


def create_mock_services() -> AgentServices:
    """Create mock services for demonstration purposes.
    
    In a real application, these would be provided by the Environment.
    """
    return AgentServices(
        spatial_service=Mock(),
        time_service=Mock(current_time=Mock(return_value=0)),
        metrics_service=Mock(),
        logging_service=Mock(),
        validation_service=Mock(is_valid_position=Mock(return_value=True)),
        lifecycle_service=Mock(),
    )


def example_1_basic_agent_creation():
    """Example 1: Creating a basic agent with the factory."""
    print("\n" + "="*60)
    print("Example 1: Basic Agent Creation")
    print("="*60)
    
    # Create services
    services = create_mock_services()
    
    # Create factory
    factory = AgentFactory(services)
    
    # Create a default agent
    agent = factory.create_default_agent(
        agent_id="agent_001",
        position=(50.0, 50.0),
        initial_resources=100.0,
    )
    
    print(f"Created agent: {agent.agent_id}")
    print(f"  Position: {agent.position}")
    print(f"  Resources: {agent.resource_level}")
    print(f"  Health: {agent.current_health}")
    print(f"  Alive: {agent.alive}")


def example_2_different_agent_types():
    """Example 2: Creating different types of agents."""
    print("\n" + "="*60)
    print("Example 2: Different Agent Types")
    print("="*60)
    
    services = create_mock_services()
    factory = AgentFactory(services)
    
    # Create different agent types
    types = {
        "Default": factory.create_default_agent("default_001", (0.0, 0.0)),
        "Learning": factory.create_learning_agent("learning_001", (10.0, 0.0)),
        "Minimal": factory.create_minimal_agent("minimal_001", (20.0, 0.0)),
        "Aggressive": factory.create_aggressive_agent("aggressive_001", (30.0, 0.0)),
        "Defensive": factory.create_defensive_agent("defensive_001", (40.0, 0.0)),
        "Efficient": factory.create_efficient_agent("efficient_001", (50.0, 0.0)),
    }
    
    for type_name, agent in types.items():
        combat = agent.get_component("combat")
        resource = agent.get_component("resource")
        
        print(f"\n{type_name}:")
        print(f"  Behavior: {agent.behavior.__class__.__name__}")
        print(f"  Health: {combat.health if combat else 'N/A'}")
        print(f"  Attack: {combat.attack_strength if combat else 'N/A':.1f}")
        print(f"  Consumption Rate: {resource.config.base_consumption_rate}")


def example_3_accessing_components():
    """Example 3: Accessing and using agent components."""
    print("\n" + "="*60)
    print("Example 3: Accessing Components")
    print("="*60)
    
    services = create_mock_services()
    factory = AgentFactory(services)
    agent = factory.create_default_agent("agent_002", (0.0, 0.0), initial_resources=100.0)
    
    # Access movement component
    movement = agent.get_component("movement")
    if movement:
        print(f"\nMovement Component:")
        print(f"  Position: {movement.position}")
        print(f"  Max Movement: {movement.config.max_movement}")
        print(f"  Perception Radius: {movement.config.perception_radius}")
        
        # Move agent
        movement.set_position((10.0, 20.0))
        print(f"  New Position: {movement.position}")
    
    # Access resource component
    resource = agent.get_component("resource")
    if resource:
        print(f"\nResource Component:")
        print(f"  Current Level: {resource.level}")
        print(f"  Consumption Rate: {resource.config.base_consumption_rate}")
        print(f"  Starvation Threshold: {resource.config.starvation_threshold}")
        
        # Consume resources
        resource.on_step_start()
        print(f"  After consumption: {resource.level}")
    
    # Access combat component
    combat = agent.get_component("combat")
    if combat:
        print(f"\nCombat Component:")
        print(f"  Health: {combat.health}")
        print(f"  Is Defending: {combat.is_defending}")
        print(f"  Attack Strength: {combat.attack_strength:.1f}")
        
        # Start defense
        combat.start_defense()
        print(f"  After starting defense: {combat.is_defending}")


def example_4_custom_configuration():
    """Example 4: Creating agents with custom configurations."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    from farm.core.agent.config import (
        MovementConfig,
        ResourceConfig,
        CombatConfig,
    )
    
    services = create_mock_services()
    factory = AgentFactory(services)
    
    # Create custom configuration
    custom_config = AgentComponentConfig(
        movement=MovementConfig(
            max_movement=15.0,  # Very fast movement
            perception_radius=10,  # Very good perception
        ),
        resource=ResourceConfig(
            base_consumption_rate=0.5,  # Very efficient
            starvation_threshold=200,  # Can survive longer without food
        ),
        combat=CombatConfig(
            starting_health=200.0,  # Very healthy
            base_attack_strength=20.0,  # Strong attacks
        ),
    )
    
    # Create agent with custom config
    custom_agent = factory.create_learning_agent(
        agent_id="custom_001",
        position=(0.0, 0.0),
        config=custom_config,
    )
    
    print(f"Created custom agent: {custom_agent.agent_id}")
    
    movement = custom_agent.get_component("movement")
    resource = custom_agent.get_component("resource")
    combat = custom_agent.get_component("combat")
    
    print(f"\nMovement Config:")
    print(f"  Max Movement: {movement.config.max_movement}")
    print(f"  Perception Radius: {movement.config.perception_radius}")
    
    print(f"\nResource Config:")
    print(f"  Consumption Rate: {resource.config.base_consumption_rate}")
    print(f"  Starvation Threshold: {resource.config.starvation_threshold}")
    
    print(f"\nCombat Config:")
    print(f"  Starting Health: {combat.config.starting_health}")
    print(f"  Base Attack: {combat.config.base_attack_strength}")


def example_5_agent_lifecycle():
    """Example 5: Running agent through a simulation lifecycle."""
    print("\n" + "="*60)
    print("Example 5: Agent Lifecycle Simulation")
    print("="*60)
    
    services = create_mock_services()
    factory = AgentFactory(services)
    
    # Create agent
    agent = factory.create_default_agent(
        agent_id="sim_001",
        position=(50.0, 50.0),
        initial_resources=100.0,
    )
    
    print(f"Initial state:")
    print(f"  Position: {agent.position}")
    print(f"  Resources: {agent.resource_level:.1f}")
    print(f"  Health: {agent.current_health:.1f}")
    print(f"  Alive: {agent.alive}")
    
    # Simulate 5 steps
    print(f"\nSimulating 5 steps...")
    for step in range(5):
        agent.step()  # Can also use agent.act() for backward compatibility
        print(f"  Step {step + 1}: Resources={agent.resource_level:.1f}, Health={agent.current_health:.1f}")
    
    # Check final state
    print(f"\nFinal state:")
    print(f"  Position: {agent.position}")
    print(f"  Resources: {agent.resource_level:.1f}")
    print(f"  Health: {agent.current_health:.1f}")
    print(f"  Alive: {agent.alive}")


def example_6_damage_and_defense():
    """Example 6: Combat with damage and defense mechanics."""
    print("\n" + "="*60)
    print("Example 6: Combat - Damage and Defense")
    print("="*60)
    
    services = create_mock_services()
    factory = AgentFactory(services)
    agent = factory.create_default_agent("combat_001", (0.0, 0.0))
    
    combat = agent.get_component("combat")
    
    print(f"Initial health: {combat.health:.1f}")
    
    # Apply damage without defense
    print(f"\nApplying 20 damage (not defending):")
    actual_damage = combat.take_damage(20.0)
    print(f"  Actual damage: {actual_damage:.1f}")
    print(f"  Health after: {combat.health:.1f}")
    
    # Apply damage with defense
    print(f"\nStarting defense and applying 20 damage:")
    combat.start_defense()
    print(f"  Defending: {combat.is_defending}")
    actual_damage = combat.take_damage(20.0)
    print(f"  Actual damage: {actual_damage:.1f} (50% reduction)")
    print(f"  Health after: {combat.health:.1f}")


def example_7_agent_properties():
    """Example 7: Using agent properties that delegate to components."""
    print("\n" + "="*60)
    print("Example 7: Agent Properties")
    print("="*60)
    
    services = create_mock_services()
    factory = AgentFactory(services)
    agent = factory.create_default_agent(
        agent_id="props_001",
        position=(100.0, 200.0),
        initial_resources=150.0,
    )
    
    print("Using agent properties (which delegate to components):")
    print(f"  agent.position: {agent.position}")
    print(f"  agent.resource_level: {agent.resource_level}")
    print(f"  agent.current_health: {agent.current_health}")
    print(f"  agent.is_defending: {agent.is_defending}")
    print(f"  agent.defense_timer: {agent.defense_timer}")
    
    # Modify properties
    print(f"\nModifying properties:")
    agent.position = (150.0, 250.0)
    agent.resource_level = 50.0
    
    print(f"  agent.position: {agent.position}")
    print(f"  agent.resource_level: {agent.resource_level}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AgentCore Component-Based System - Usage Examples")
    print("="*60)
    
    example_1_basic_agent_creation()
    example_2_different_agent_types()
    example_3_accessing_components()
    example_4_custom_configuration()
    example_5_agent_lifecycle()
    example_6_damage_and_defense()
    example_7_agent_properties()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
