# Quick Start - New Agent System

## Installation

The new agent system is already in your codebase at `farm.core.agent`.

## Basic Usage (5 Minutes)

### 1. Create an Agent

```python
from farm.core.agent import AgentFactory

# Create factory
factory = AgentFactory(
    spatial_service=spatial_service,
    time_service=time_service,
    lifecycle_service=lifecycle_service,
)

# Create agent
agent = factory.create_default_agent(
    agent_id="agent_001",
    position=(50.0, 50.0),
    initial_resources=100
)
```

### 2. Use Components

```python
# Movement
movement = agent.get_component("movement")
movement.move_to((100, 100))

# Resources
resource = agent.get_component("resource")
resource.add(50)
resource.consume(20)

# Combat
combat = agent.get_component("combat")
combat.attack(target_agent)

# Perception
perception = agent.get_component("perception")
nearby = perception.get_nearby_entities(["resources"])
```

### 3. Run Simulation

```python
# Execute one turn
agent.act()

# Run simulation loop
for step in range(1000):
    if agent.alive:
        agent.act()
```

## Custom Configuration

```python
from farm.core.agent import AgentConfig, MovementConfig, CombatConfig

config = AgentConfig(
    movement=MovementConfig(max_movement=15.0),
    combat=CombatConfig(
        starting_health=150.0,
        base_attack_strength=20.0
    )
)

factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=config
)

agent = factory.create_default_agent(
    agent_id="custom_agent",
    position=(0, 0)
)
```

## Creating Multiple Agents

```python
import random

agents = []
for i in range(100):
    agent = factory.create_default_agent(
        agent_id=f"agent_{i:03d}",
        position=(
            random.uniform(0, 100),
            random.uniform(0, 100)
        ),
        initial_resources=100
    )
    agents.append(agent)

# Simulation loop
for step in range(1000):
    for agent in agents[:]:
        if agent.alive:
            agent.act()
    
    agents = [a for a in agents if a.alive]
    print(f"Step {step}: {len(agents)} alive")
```

## Custom Agent Types

```python
from farm.core.agent import (
    AgentCore,
    MovementComponent,
    CombatComponent,
    DefaultAgentBehavior,
    MovementConfig,
    CombatConfig,
)

# Warrior - only movement and combat
warrior = AgentCore(
    agent_id="warrior_001",
    position=(50, 50),
    spatial_service=spatial_service,
    behavior=DefaultAgentBehavior(),
    components=[
        MovementComponent(MovementConfig(max_movement=15.0)),
        CombatComponent(CombatConfig(
            starting_health=200.0,
            base_attack_strength=25.0
        ))
    ]
)
```

## Next Steps

- **Full documentation**: `docs/design/NEW_AGENT_SYSTEM.md`
- **Architecture details**: `docs/design/agent_refactoring_design.md`
- **Working examples**: `examples/new_agent_system_demo.py`

## Key Features

✅ **Component-based** - Mix and match capabilities
✅ **Type-safe** - Catch errors at compile time
✅ **Testable** - 195 comprehensive tests
✅ **Performant** - Faster than old system
✅ **Extensible** - Easy to add custom components

That's it! You're ready to use the new agent system.