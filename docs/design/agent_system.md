# New Agent System - Complete Guide

## Overview

The agent module is a modern, component-based architecture following SOLID principles. This guide shows you everything you can do with the new system.

**Note**: This system replaces the old `BaseAgent`. Use `AgentCore` and components directly.

---

## Quick Start (5 minutes)

### Install & Import

```python
from farm.core.agent import AgentFactory, AgentConfig
```

### Create Your First Agent

```python
# Create factory
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig(),  # Optional: custom default config
    time_service=time_service,      # Optional: for time tracking
    lifecycle_service=lifecycle_service  # Optional: for agent lifecycle
)

# Create agent
agent = factory.create_default_agent(
    agent_id="agent_001",
    position=(50.0, 50.0),
    initial_resources=100
)

# Use components
movement = agent.get_component("movement")
movement.move_to((100.0, 100.0))

resource = agent.get_component("resource")
print(f"Resources: {resource.level}")

# Execute turn
agent.act()
```

Done! You're using the new system! ✅

---

## Core Concepts

### 1. Components (Composition Pattern)

Components are pluggable capabilities that can be mixed and matched:

```python
# Available components
from farm.core.agent.components import (
    MovementComponent,      # Movement & navigation
    ResourceComponent,      # Resource tracking
    CombatComponent,        # Combat mechanics
    PerceptionComponent,    # Environment observation
    ReproductionComponent,  # Offspring creation
)

# Add to agent
agent.add_component(MovementComponent(config.movement))
agent.add_component(ResourceComponent(initial_resources=100, config=config.resource))

# Access component
movement = agent.get_component("movement")
if movement:
    movement.move_to((100, 100))
```

### 2. Behaviors (Strategy Pattern)

Behaviors determine how agents make decisions:

```python
from farm.core.agent.behaviors import (
    DefaultAgentBehavior,   # Random actions
    LearningAgentBehavior,  # RL-based decisions
)

# Create agent with specific behavior
agent = AgentCore(
    agent_id="agent_001",
    position=(0, 0),
    spatial_service=spatial_service,
    behavior=DefaultAgentBehavior(),  # or LearningAgentBehavior()
    components=[...]
)
```

### 3. Configuration (Value Objects)

Type-safe, immutable configuration:

```python
from farm.core.agent.config import AgentConfig, MovementConfig, CombatConfig

# Default configuration
config = AgentConfig()

# Custom configuration
config = AgentConfig(
    movement=MovementConfig(max_movement=15.0),
    combat=CombatConfig(starting_health=150.0, base_attack_strength=20.0)
)

# Access values (type-safe!)
max_movement = config.movement.max_movement  # IDE autocomplete works!
starting_health = config.combat.starting_health
```

### 4. Factory (Builder Pattern)

Clean agent construction:

```python
from farm.core.agent import AgentFactory

factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig(),  # Optional: custom default config
    time_service=time_service,      # Optional: for time tracking
    lifecycle_service=lifecycle_service,  # Optional: for agent lifecycle
    agent_parameters=None  # Optional: agent type-specific parameters
)

# Create different agent types
default_agent = factory.create_default_agent(...)
learning_agent = factory.create_learning_agent(...)
minimal_agent = factory.create_minimal_agent(...)
```

---

## Common Tasks

### Task 1: Create Agent Population

```python
from farm.core.agent import AgentFactory, AgentConfig

factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig()  # Optional
)

# Create 100 agents
agents = []
for i in range(100):
    agent = factory.create_default_agent(
        agent_id=f"agent_{i:03d}",
        position=(random.uniform(0, 100), random.uniform(0, 100)),
        initial_resources=100
    )
    agents.append(agent)

print(f"Created {len(agents)} agents")
```

### Task 2: Move Agents

```python
# Get movement component
movement = agent.get_component("movement")

# Move to absolute position
movement.move_to((100.0, 100.0))

# Move by relative offset
movement.move_by(10.0, -5.0)

# Random movement
movement.random_move()  # Random distance up to max_movement
movement.random_move(5.0)  # Random move exactly 5 units

# Move toward target with stop distance
movement.move_toward_entity(target.position, stop_distance=5.0)

# Check reachability
if movement.can_reach(target.position):
    movement.move_to(target.position)

# Get distance
distance = movement.distance_to(target.position)
```

### Task 3: Manage Resources

```python
# Get resource component
resource = agent.get_component("resource")

# Add resources (e.g., from gathering)
resource.add(50)

# Consume resources (e.g., for action cost)
if resource.consume(20):
    print("Successfully consumed 20 resources")

# Check resources
if resource.has_resources(100):
    print("Agent has at least 100 resources")

# Check starvation
if resource.is_starving:
    print(f"Agent starving for {resource.starvation_steps} steps!")
```

### Task 4: Combat

```python
# Get combat component
combat = agent.get_component("combat")

# Attack another agent
result = combat.attack(target_agent)
if result['success']:
    print(f"Dealt {result['damage_dealt']} damage!")
    print(f"Target health: {result['target_health']}")
    if result['target_killed']:
        print("Target eliminated!")
else:
    print(f"Attack failed: {result.get('error', 'Unknown error')}")

# Take damage
damage_taken = combat.take_damage(25.0)
print(f"Took {damage_taken} damage")

# Start defending
combat.start_defense(duration=3)

# Check health
if combat.health_ratio < 0.3:
    print("Critical health!")

# Heal
healed = combat.heal(50.0)
print(f"Restored {healed} health")
```

### Task 5: Perception

```python
# Get perception component
perception = agent.get_component("perception")

# Find nearby entities
nearby = perception.get_nearby_entities(["resources", "agents"])
resources = nearby["resources"]
agents = nearby["agents"]

# Find nearby entities with custom radius
nearby_custom = perception.get_nearby_entities(["resources"], radius=15.0)

# Find nearest
nearest = perception.get_nearest_entity(["resources"])
if nearest["resources"]:
    target_resource = nearest["resources"]

# Multi-channel observation (recommended for learning agents)
observation = perception.get_observation()
if observation:
    tensor = observation.tensor()  # Shape: (num_channels, 2R+1, 2R+1)
    print(f"Multi-channel observation shape: {tensor.shape}")
    
    # Update observation with world state
    world_layers = {
        "RESOURCES": resource_tensor,
        "OBSTACLES": obstacle_tensor
    }
    perception.update_observation(world_layers)

# Legacy perception grid (deprecated)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    grid = perception.create_perception_grid()
    print(f"Grid shape: {grid.shape}")

# Check visibility
if perception.can_see(target.position):
    print("Target is visible!")

# Count nearby
count = perception.count_nearby("resources")  # Uses default radius
count_custom = perception.count_nearby("resources", radius=10)
print(f"Found {count} resources within default radius")
print(f"Found {count_custom} resources within 10 units")
```

### Task 6: Reproduction

```python
# Get reproduction component
reproduction = agent.get_component("reproduction")

# Check if can reproduce
if reproduction.can_reproduce():
    result = reproduction.reproduce()
    
    if result['success']:
        print(f"Created offspring {result['offspring_id']}")
        print(f"Cost: {result['cost']} resources")
    else:
        print(f"Reproduction failed: {result.get('error', 'Unknown error')}")

# Get reproduction info
info = reproduction.get_reproduction_info()
print(f"Need {info['required_resources']} to reproduce")
print(f"Have {info['current_resources']} resources")
print(f"Reproduced {info['reproduction_count']} times")
```

### Task 7: Run Simulation

```python
from farm.core.agent import AgentFactory

# Setup
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig(),  # Optional: custom default config
    time_service=time_service,      # Optional: for time tracking
    lifecycle_service=lifecycle_service,  # Optional: for agent lifecycle
    agent_parameters=None  # Optional: agent type-specific parameters
)

# Create population
agents = [
    factory.create_default_agent(
        agent_id=f"agent_{i}",
        position=(random.uniform(0, 100), random.uniform(0, 100)),
        initial_resources=100
    )
    for i in range(100)
]

# Run simulation
for step in range(1000):
    # Update time
    time_service.set_time(step)
    
    # Each agent acts
    for agent in agents[:]:  # Copy list (agents may die/reproduce)
        if agent.alive:
            agent.act()
    
    # Update agent list
    agents = [a for a in agents if a.alive]
    
    if step % 100 == 0:
        print(f"Step {step}: {len(agents)} agents alive")
```

---

## Advanced Usage

### Custom Agent Types

Create specialized agents with specific components:

```python
from farm.core.agent import (
    AgentCore, MovementComponent, CombatComponent,
    DefaultAgentBehavior, MovementConfig, CombatConfig
)

# Warrior agent - only movement and combat
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

# Scout agent - fast movement, high perception
scout = AgentCore(
    agent_id="scout_001",
    position=(25, 75),
    spatial_service=spatial_service,
    behavior=DefaultAgentBehavior(),
    components=[
        MovementComponent(MovementConfig(max_movement=20.0)),
        PerceptionComponent(spatial_service, PerceptionConfig(perception_radius=15))
    ]
)
```

### Custom Components

Create your own components:

```python
from farm.core.agent.components.base import IAgentComponent

class StealthComponent(IAgentComponent):
    """Custom stealth component."""
    
    def __init__(self, detection_radius: float = 5.0):
        self._detection_radius = detection_radius
        self._is_hidden = False
    
    @property
    def name(self) -> str:
        return "stealth"
    
    def activate_stealth(self) -> None:
        """Enter stealth mode."""
        self._is_hidden = True
    
    def deactivate_stealth(self) -> None:
        """Exit stealth mode."""
        self._is_hidden = False
    
    @property
    def is_hidden(self) -> bool:
        return self._is_hidden

# Use custom component
agent.add_component(StealthComponent(detection_radius=3.0))

stealth = agent.get_component("stealth")
stealth.activate_stealth()
if stealth.is_hidden:
    print("Agent is hidden!")
```

### Custom Behaviors

Create your own decision-making strategies:

```python
from farm.core.agent.behaviors.base_behavior import IAgentBehavior

class SwarmBehavior(IAgentBehavior):
    """Behavior that coordinates with nearby agents."""
    
    def execute_turn(self, agent):
        """Execute swarm logic."""
        perception = agent.get_component("perception")
        movement = agent.get_component("movement")
        
        if not (perception and movement):
            return
        
        # Get nearby agents
        nearby = perception.get_nearby_entities(["agents"])
        nearby_agents = nearby.get("agents", [])
        
        if nearby_agents:
            # Move toward center of mass
            avg_x = sum(a.position[0] for a in nearby_agents) / len(nearby_agents)
            avg_y = sum(a.position[1] for a in nearby_agents) / len(nearby_agents)
            movement.move_toward_entity((avg_x, avg_y), stop_distance=2.0)
        else:
            # Explore randomly
            movement.random_move()

# Use custom behavior
agent = AgentCore(
    agent_id="swarm_001",
    position=(50, 50),
    spatial_service=spatial_service,
    behavior=SwarmBehavior(),
    components=[...]
)
```

---

## Replacing Old BaseAgent Code

If you have code using the old `BaseAgent`, replace it with the new system:

```python
# Old way (don't use)
from farm.core.agent import BaseAgent
agent = BaseAgent(agent_id="001", position=(0,0), resource_level=100, ...)

# New way (use this)
from farm.core.agent import AgentFactory
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig()  # Optional
)
agent = factory.create_default_agent(agent_id="001", position=(0,0), initial_resources=100)
```

The new system is cleaner, more testable, and follows SOLID principles.

---

## Best Practices

### 1. Use Factory

✅ **Do**: Use AgentFactory for construction
```python
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig()  # Optional
)
agent = factory.create_default_agent(...)
```

❌ **Don't**: Manually create AgentCore
```python
agent = AgentCore(...)  # Too verbose, error-prone
```

### 2. Check Component Availability

✅ **Do**: Check before using
```python
movement = agent.get_component("movement")
if movement:
    movement.move_to((100, 100))
```

❌ **Don't**: Assume component exists
```python
movement = agent.get_component("movement")
movement.move_to((100, 100))  # Might be None!
```

### 3. Use Configuration Objects

✅ **Do**: Use type-safe config
```python
config = AgentConfig(
    movement=MovementConfig(max_movement=15.0)
)
max_movement = config.movement.max_movement
```

❌ **Don't**: Use old verbose pattern
```python
max_movement = get_nested_then_flat(...)  # Old way
```

### 4. Compose, Don't Inherit

✅ **Do**: Add components
```python
agent.add_component(MyComponent())
```

❌ **Don't**: Subclass AgentCore
```python
class MyAgent(AgentCore):  # Avoid inheritance
    pass
```

---

## API Reference

### AgentCore

```python
class AgentCore:
    # Properties
    agent_id: str
    alive: bool
    position: tuple[float, float]
    position_3d: tuple[float, float, float]
    state_manager: StateManager
    
    # Methods
    def add_component(component: IAgentComponent) -> None
    def get_component(name: str) -> Optional[IAgentComponent]
    def has_component(name: str) -> bool
    def remove_component(name: str) -> Optional[IAgentComponent]
    def act() -> None
    def terminate() -> None
    def get_state_dict() -> dict
    def load_state_dict(state: dict) -> None
```

### AgentFactory

```python
class AgentFactory:
    def __init__(
        self,
        spatial_service: ISpatialQueryService,  # Required
        default_config: Optional[AgentConfig] = None,  # Optional
        time_service: Optional[ITimeService] = None,  # Optional
        lifecycle_service: Optional[IAgentLifecycleService] = None,  # Optional
        agent_parameters: Optional[Dict[str, Dict[str, float]]] = None  # Optional
    )
    
    def create_default_agent(
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 100,
        config: Optional[AgentConfig] = None,
        agent_type: Optional[str] = None  # For future agent type specialization
    ) -> AgentCore
    
    def create_learning_agent(
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 100,
        config: Optional[AgentConfig] = None,
        decision_module: Optional[DecisionModule] = None
    ) -> AgentCore
    
    def create_minimal_agent(
        agent_id: str,
        position: tuple[float, float],
        components: Optional[List[IAgentComponent]] = None
    ) -> AgentCore
```

### Components

```python
# MovementComponent
movement = agent.get_component("movement")
movement.move_to(position: tuple) -> bool  # Returns True if move was successful
movement.move_by(delta_x: float, delta_y: float) -> bool  # Returns True if move was successful
movement.random_move(distance: Optional[float] = None) -> bool  # Returns True if move was successful
movement.move_toward_entity(target: tuple, stop_distance: float = 0.0) -> bool  # Returns True if move was successful
movement.can_reach(target: tuple) -> bool  # Returns True if target is reachable
movement.distance_to(target: tuple) -> float  # Returns distance to target

# ResourceComponent
resource = agent.get_component("resource")
# Constructor: ResourceComponent(initial_resources: int, config: ResourceConfig)
resource.level: int
resource.add(amount: int) -> None
resource.consume(amount: int) -> bool
resource.has_resources(amount: int) -> bool
resource.is_starving: bool
resource.starvation_steps: int

# CombatComponent
combat = agent.get_component("combat")
combat.health: float
combat.max_health: float
combat.health_ratio: float
combat.attack(target: AgentCore) -> dict  # Returns: success, damage_dealt, target_killed, target_health, error
combat.take_damage(damage: float) -> float  # Returns actual damage taken
combat.heal(amount: float) -> float  # Returns actual healing done
combat.start_defense(duration: Optional[int] = None) -> None
combat.is_defending: bool

# PerceptionComponent
perception = agent.get_component("perception")
perception.get_nearby_entities(types: Optional[List[str]], radius: Optional[float] = None) -> dict
perception.get_nearest_entity(types: Optional[List[str]]) -> dict
perception.get_observation() -> Optional[AgentObservation]  # Multi-channel observation
perception.update_observation(world_layers: Dict[str, Any], **kwargs) -> None
perception.create_perception_grid() -> np.ndarray  # [DEPRECATED] Use get_observation()
perception.can_see(position: tuple) -> bool
perception.count_nearby(entity_type: str, radius: Optional[float] = None) -> int

# ReproductionComponent
reproduction = agent.get_component("reproduction")
reproduction.can_reproduce() -> bool  # Returns True if agent can reproduce
reproduction.reproduce() -> dict  # Returns: success, offspring_id, cost, error
reproduction.get_reproduction_info() -> dict  # Returns reproduction status info
reproduction.reproduction_count: int  # Number of times agent has reproduced
```

---

## Examples

### Example 1: Simple Agent

```python
# Create agent
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig()  # Optional
)
agent = factory.create_default_agent(
    agent_id="simple_001",
    position=(0, 0),
    initial_resources=100
)

# Use agent
for _ in range(100):
    agent.act()
```

### Example 2: Learning Agent

```python
from farm.core.decision.decision import DecisionModule
from farm.core.decision.config import DecisionConfig

# Create decision module
decision_config = DecisionConfig(algorithm_type="dqn")
decision_module = DecisionModule(...)  # Setup your RL

# Create learning agent
agent = factory.create_learning_agent(
    agent_id="learner_001",
    position=(50, 50),
    initial_resources=100,
    decision_module=decision_module
)

# Agent learns from experience
for episode in range(100):
    for _ in range(1000):
        agent.act()
```

### Example 3: Custom Configuration

```python
from farm.core.agent.config import (
    AgentConfig, MovementConfig, CombatConfig, ResourceConfig
)

# Create custom config
config = AgentConfig(
    movement=MovementConfig(max_movement=20.0),
    resource=ResourceConfig(
        base_consumption_rate=2,
        starvation_threshold=50
    ),
    combat=CombatConfig(
        starting_health=200.0,
        base_attack_strength=25.0
    )
)

# Use config
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=config
)

agent = factory.create_default_agent(
    agent_id="custom_001",
    position=(0, 0)
)
```

### Example 4: Agent vs Agent Combat

```python
# Create two agents
agent1 = factory.create_default_agent(agent_id="fighter_1", position=(0, 0))
agent2 = factory.create_default_agent(agent_id="fighter_2", position=(5, 5))

# Combat simulation
combat1 = agent1.get_component("combat")
combat2 = agent2.get_component("combat")

while agent1.alive and agent2.alive:
    # Agent 1 attacks
    result = combat1.attack(agent2)
    print(f"Agent 1 dealt {result['damage_dealt']} damage")
    
    if not agent2.alive:
        print("Agent 2 defeated!")
        break
    
    # Agent 2 defends and counter-attacks
    combat2.start_defense()
    result = combat2.attack(agent1)
    print(f"Agent 2 dealt {result['damage_dealt']} damage (defending)")
    
    if not agent1.alive:
        print("Agent 1 defeated!")
        break

print(f"Victor: {agent1.agent_id if agent1.alive else agent2.agent_id}")
```

---

## Testing Your Code

### Unit Test Components

```python
import pytest
from unittest.mock import Mock
from farm.core.agent.components import MovementComponent
from farm.core.agent.config import MovementConfig

def test_movement():
    """Test movement component."""
    # Create component
    config = MovementConfig(max_movement=10.0)
    movement = MovementComponent(config)
    
    # Create mock agent
    agent = Mock()
    agent.state_manager = Mock()
    agent.state_manager.position = (0.0, 0.0)
    
    # Attach and test
    movement.attach(agent)
    movement.move_to((5.0, 5.0))
    
    # Verify
    agent.state_manager.set_position.assert_called()
```

### Integration Test Agents

```python
def test_agent_integration():
    """Test complete agent system."""
    factory = AgentFactory(spatial_service=mock_spatial_service)
    
    agent = factory.create_default_agent(
        agent_id="test",
        position=(0, 0),
        initial_resources=100
    )
    
    # Test components work together
    movement = agent.get_component("movement")
    resource = agent.get_component("resource")
    
    movement.move_to((10, 10))
    resource.consume(20)
    
    agent.act()
    
    assert agent.alive is True
```

---

## Performance

Performance characteristics:

- **Component-based architecture**: Minimal overhead from composition
- **Type-safe configuration**: Compile-time validation, no runtime config parsing
- **Efficient component access**: O(1) component lookup by name
- **Memory efficient**: Components only created when needed
- **Scalable**: Supports large populations with minimal memory overhead

**Note**: For specific performance benchmarks, run the benchmark suite in `tests/benchmarks/`.

---

## Troubleshooting

### Component Not Found

**Problem**: `agent.get_component("xyz")` returns `None`

**Solution**: Check component was added
```python
if not agent.has_component("movement"):
    agent.add_component(MovementComponent(config.movement))
```

### Type Errors

**Problem**: Type checker complains

**Solution**: Add type hints
```python
from farm.core.agent import AgentCore

def process(agent: AgentCore) -> None:
    movement = agent.get_component("movement")
    if movement:  # Type guard
        movement.move_to((100, 100))
```

### Performance Issues

**Problem**: Simulation is slow

**Solution**: Profile and optimize
```python
# Use performance benchmarks
python tests/benchmarks/test_agent_performance.py

# Profile your code
python -m cProfile -o profile.stats your_simulation.py
```

---

## Resources

- **Quick Start**: `docs/QUICK_START.md`
- **Complete Design**: `docs/design/agent_refactoring_design.md`
- **Recommended Usage**: `docs/design/RECOMMENDED_USAGE.md`
- **Phase Summaries**: `docs/design/agent_refactoring_phase*_summary.md`
- **Examples**: `examples/new_agent_system_demo.py`
- **Tests**: `tests/agent/`

---

## Summary

The new agent system provides:

✅ **Modular architecture** - Easy to understand and extend
✅ **SOLID principles** - Clean, maintainable code
✅ **Comprehensive testing** - 195 tests ensuring correctness
✅ **Type safety** - Catch errors early
✅ **Performance** - No regression, some improvements
✅ **Component-based** - Compose capabilities as needed
✅ **Well-documented** - Guides and examples

**Ready for production!** 🚀

Start using it today with just a few lines of code:

```python
from farm.core.agent import AgentFactory

factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig()  # Optional
)
agent = factory.create_default_agent(agent_id="001", position=(0, 0))
agent.act()
```

Happy coding! 🎉