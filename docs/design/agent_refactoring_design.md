# Agent Module Refactoring Design

## Executive Summary

This document outlines a comprehensive refactoring of the agent module to follow SOLID principles and modern design patterns. The refactoring transforms the current monolithic `BaseAgent` class (1571 lines) into a modular, composable architecture that is easier to test, extend, and maintain.

## Current Architecture Problems

### 1. Single Responsibility Principle (SRP) Violations

The `BaseAgent` class has **13+ distinct responsibilities**:

```python
# Current BaseAgent responsibilities:
- State management (position, resources, health)
- Decision making (action selection)
- Action execution coordination
- Resource management (consumption, gathering)
- Combat mechanics (attack, defend, damage)
- Reproduction (offspring creation)
- Perception (environment observation)
- Memory management (Redis integration)
- Genome serialization/deserialization
- Service initialization and wiring
- Curriculum learning
- Reward calculation
- Starvation checking
```

**Impact**: 
- Difficult to test individual behaviors in isolation
- Hard to understand which part of code does what
- Changes in one area can break unrelated functionality

### 2. Open-Closed Principle (OCP) Violations

Adding new behaviors requires modifying `BaseAgent` directly:
- New action types require changes to the core class
- New decision algorithms need modifications to initialization logic
- New resource mechanics require editing existing methods

**Impact**:
- Risk of breaking existing functionality when adding features
- Cannot easily create specialized agent types
- Violates "open for extension, closed for modification"

### 3. DRY (Don't Repeat Yourself) Violations

Config fetching pattern repeated 15+ times:
```python
self.max_movement = get_nested_then_flat(
    config=self.config,
    nested_parent_attr="agent_behavior",
    nested_attr_name="max_movement",
    flat_attr_name="max_movement",
    default_value=8,
    expected_types=(int, float),
)
```

**Impact**:
- Verbose, hard to read code
- Inconsistent defaults across methods
- Configuration changes require many updates

### 4. Dependency Inversion Principle (DIP) Violations

Direct dependencies on concrete implementations:
- Tightly coupled to `Environment` class
- Direct instantiation of `DecisionModule`
- Hard-coded action execution logic

### 5. God Object Anti-Pattern

The `BaseAgent` class:
- **1571 lines** in a single file
- **40+ public methods**
- **20+ private methods**
- **30+ instance attributes**

## Proposed Architecture

### Core Design Principles

1. **Composition over Inheritance**: Use component-based architecture
2. **Dependency Injection**: All dependencies provided via constructor or setters
3. **Single Responsibility**: Each class has one clear purpose
4. **Strategy Pattern**: Pluggable algorithms for behavior variation
5. **Observer Pattern**: Event-driven communication for lifecycle events

### New Module Structure

```
farm/core/agent/
├── __init__.py
├── core.py                    # AgentCore - minimal agent identity
├── components/                # Behavior components (composition)
│   ├── __init__.py
│   ├── base.py               # IAgentComponent interface
│   ├── movement.py           # MovementComponent
│   ├── resource.py           # ResourceComponent
│   ├── combat.py             # CombatComponent
│   ├── reproduction.py       # ReproductionComponent
│   ├── perception.py         # PerceptionComponent
│   └── memory.py             # MemoryComponent
├── state/                     # State management
│   ├── __init__.py
│   ├── agent_state.py        # AgentState (moved from core/state.py)
│   └── state_manager.py      # StateManager component
├── behaviors/                 # Complete agent behaviors (strategies)
│   ├── __init__.py
│   ├── base_behavior.py      # IAgentBehavior interface
│   ├── default_behavior.py   # DefaultAgentBehavior
│   └── learning_behavior.py  # LearningAgentBehavior
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── agent_config.py       # AgentConfig value object
│   └── config_loader.py      # Configuration loading utilities
├── lifecycle/                 # Lifecycle management
│   ├── __init__.py
│   ├── events.py             # Lifecycle events (birth, death, etc.)
│   └── observers.py          # Event observers/listeners
└── factory.py                 # AgentFactory for construction

# Keep existing decision module separate (already well-designed)
farm/core/decision/
├── ... (existing structure)

# Keep existing action module but refactor
farm/core/action/
├── __init__.py
├── base.py                    # IAction interface
├── actions/                   # Individual action implementations
│   ├── __init__.py
│   ├── movement.py
│   ├── gathering.py
│   ├── combat.py
│   ├── sharing.py
│   └── reproduction.py
├── registry.py                # Action registry (keep existing)
└── validators.py              # Action validation
```

---

## Detailed Component Design

### 1. AgentCore - Minimal Agent Identity

**Responsibility**: Maintain only essential agent identity and coordinate components

```python
# farm/core/agent/core.py

from typing import Dict, Optional, List
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.state.state_manager import StateManager
from farm.core.agent.behaviors.base_behavior import IAgentBehavior
from farm.core.services.interfaces import ISpatialQueryService

class AgentCore:
    """
    Minimal agent core that holds identity and coordinates components.
    
    Follows SRP: Only manages agent identity and component coordination.
    Follows OCP: New behaviors added via components, not modification.
    Follows DIP: Depends on abstractions (interfaces), not concrete classes.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        spatial_service: ISpatialQueryService,
        behavior: IAgentBehavior,
        components: Optional[List[IAgentComponent]] = None,
    ):
        """Initialize agent with identity and pluggable components."""
        # Essential identity
        self.agent_id = agent_id
        self.alive = True
        
        # Core services (injected dependencies)
        self._spatial_service = spatial_service
        self._behavior = behavior
        
        # Component registry (composition)
        self._components: Dict[str, IAgentComponent] = {}
        if components:
            for component in components:
                self.add_component(component)
        
        # State management (single responsibility)
        self._state_manager = StateManager(self)
        self._state_manager.set_position(position)
    
    def add_component(self, component: IAgentComponent) -> None:
        """Add a behavior component to this agent (OCP compliance)."""
        component.attach(self)
        self._components[component.name] = component
    
    def get_component(self, name: str) -> Optional[IAgentComponent]:
        """Get a component by name."""
        return self._components.get(name)
    
    def act(self) -> None:
        """Execute one simulation step using the behavior strategy."""
        if not self.alive:
            return
        
        # Pre-step: notify all components
        for component in self._components.values():
            component.on_step_start()
        
        # Execute behavior strategy
        self._behavior.execute_turn(self)
        
        # Post-step: notify all components
        for component in self._components.values():
            component.on_step_end()
    
    def terminate(self) -> None:
        """Handle agent death and cleanup."""
        if not self.alive:
            return
            
        self.alive = False
        
        # Notify all components of termination
        for component in self._components.values():
            component.on_terminate()
    
    @property
    def position(self) -> tuple[float, float]:
        """Get current position."""
        return self._state_manager.position
    
    @property
    def state_manager(self) -> StateManager:
        """Get state manager for direct access when needed."""
        return self._state_manager
```

**Benefits**:
- ✅ **70 lines** vs 1571 lines (95% reduction)
- ✅ **Single responsibility**: Identity and coordination only
- ✅ **Open for extension**: Add components without modifying core
- ✅ **Testable**: Easy to mock components and behavior

---

### 2. Component System - Pluggable Behaviors

**Responsibility**: Each component handles one specific capability

#### Base Component Interface

```python
# farm/core/agent/components/base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore

class IAgentComponent(ABC):
    """
    Interface for agent components.
    
    Follows ISP: Small, focused interface.
    Follows SRP: Each component has one responsibility.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this component type."""
        pass
    
    def attach(self, agent: "AgentCore") -> None:
        """Attach this component to an agent."""
        self._agent = agent
    
    def on_step_start(self) -> None:
        """Called at the start of each simulation step."""
        pass
    
    def on_step_end(self) -> None:
        """Called at the end of each simulation step."""
        pass
    
    def on_terminate(self) -> None:
        """Called when the agent is terminated."""
        pass
```

#### Movement Component Example

```python
# farm/core/agent/components/movement.py

from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import MovementConfig

class MovementComponent(IAgentComponent):
    """
    Handles agent movement in the environment.
    
    Single Responsibility: Only movement logic.
    """
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._agent = None
    
    @property
    def name(self) -> str:
        return "movement"
    
    def move_to(self, target_position: tuple[float, float]) -> bool:
        """
        Move agent towards target position.
        
        Returns:
            bool: True if move was successful, False otherwise
        """
        if self._agent is None:
            return False
        
        current_pos = self._agent.position
        
        # Calculate movement within max_movement distance
        import math
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return True  # Already at target
        
        # Limit movement to max_movement
        if distance > self._config.max_movement:
            scale = self._config.max_movement / distance
            dx *= scale
            dy *= scale
        
        new_position = (
            current_pos[0] + dx,
            current_pos[1] + dy
        )
        
        # Update position through state manager
        self._agent.state_manager.set_position(new_position)
        return True
    
    def random_move(self) -> bool:
        """Move in a random direction."""
        import random
        import math
        
        if self._agent is None:
            return False
        
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, self._config.max_movement)
        
        current_pos = self._agent.position
        new_position = (
            current_pos[0] + distance * math.cos(angle),
            current_pos[1] + distance * math.sin(angle)
        )
        
        self._agent.state_manager.set_position(new_position)
        return True
```

#### Resource Component Example

```python
# farm/core/agent/components/resource.py

from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import ResourceConfig

class ResourceComponent(IAgentComponent):
    """
    Handles agent resource management.
    
    Single Responsibility: Resource tracking and consumption.
    """
    
    def __init__(self, initial_resources: int, config: ResourceConfig):
        self._resources = initial_resources
        self._config = config
        self._agent = None
        self._starvation_counter = 0
    
    @property
    def name(self) -> str:
        return "resource"
    
    @property
    def level(self) -> int:
        """Current resource level."""
        return self._resources
    
    def add(self, amount: int) -> None:
        """Add resources."""
        self._resources += amount
    
    def consume(self, amount: int) -> bool:
        """
        Consume resources.
        
        Returns:
            bool: True if consumption was successful, False if insufficient
        """
        if self._resources >= amount:
            self._resources -= amount
            return True
        return False
    
    def on_step_end(self) -> None:
        """Consume base resources each step and check starvation."""
        # Base consumption
        self._resources -= self._config.base_consumption_rate
        
        # Check starvation
        if self._resources <= 0:
            self._starvation_counter += 1
            if self._starvation_counter >= self._config.starvation_threshold:
                # Trigger agent death
                if self._agent:
                    self._agent.terminate()
        else:
            self._starvation_counter = 0
```

#### Combat Component Example

```python
# farm/core/agent/components/combat.py

from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.config.agent_config import CombatConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore

class CombatComponent(IAgentComponent):
    """
    Handles agent combat mechanics.
    
    Single Responsibility: Combat and defense logic.
    """
    
    def __init__(self, config: CombatConfig):
        self._config = config
        self._agent = None
        self._health = config.starting_health
        self._is_defending = False
        self._defense_timer = 0
    
    @property
    def name(self) -> str:
        return "combat"
    
    @property
    def health(self) -> float:
        """Current health."""
        return self._health
    
    @property
    def is_defending(self) -> bool:
        """Whether agent is currently defending."""
        return self._is_defending
    
    def attack(self, target: "AgentCore") -> float:
        """
        Attack another agent.
        
        Returns:
            float: Damage dealt
        """
        # Get target's combat component
        target_combat = target.get_component("combat")
        if not target_combat:
            return 0.0
        
        # Calculate damage
        damage = self._calculate_attack_damage()
        
        # Apply damage to target
        actual_damage = target_combat.take_damage(damage)
        
        return actual_damage
    
    def take_damage(self, damage: float) -> float:
        """
        Take damage from an attack.
        
        Returns:
            float: Actual damage taken after defense
        """
        # Reduce damage if defending
        if self._is_defending:
            damage *= self._config.defense_reduction
        
        self._health -= damage
        
        # Check for death
        if self._health <= 0:
            self._health = 0
            if self._agent:
                self._agent.terminate()
        
        return damage
    
    def start_defense(self, duration: int = 1) -> None:
        """Start defending for a number of turns."""
        self._is_defending = True
        self._defense_timer = duration
    
    def on_step_end(self) -> None:
        """Update defense timer."""
        if self._defense_timer > 0:
            self._defense_timer -= 1
            if self._defense_timer <= 0:
                self._is_defending = False
    
    def _calculate_attack_damage(self) -> float:
        """Calculate attack damage based on health."""
        health_ratio = self._health / self._config.starting_health
        return self._config.base_attack_strength * health_ratio
```

**Benefits**:
- ✅ Each component: **50-100 lines** vs mixed in 1571-line class
- ✅ **Single responsibility**: Movement, resources, combat separate
- ✅ **Testable**: Mock agent core, test component in isolation
- ✅ **Reusable**: Components can be shared across agent types
- ✅ **Composable**: Mix and match components for different agent types

---

### 3. Behavior Strategy - Pluggable Decision Logic

**Responsibility**: Define how agents make decisions and act

```python
# farm/core/agent/behaviors/base_behavior.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore

class IAgentBehavior(ABC):
    """
    Strategy interface for agent behavior.
    
    Follows Strategy Pattern: Different algorithms interchangeable.
    Follows OCP: New behaviors without modifying existing code.
    """
    
    @abstractmethod
    def execute_turn(self, agent: "AgentCore") -> None:
        """Execute one turn of agent behavior."""
        pass
```

```python
# farm/core/agent/behaviors/learning_behavior.py

from farm.core.agent.behaviors.base_behavior import IAgentBehavior
from farm.core.decision.decision import DecisionModule
from farm.core.action.registry import ActionRegistry

class LearningAgentBehavior(IAgentBehavior):
    """
    Reinforcement learning based agent behavior.
    
    Single Responsibility: Coordinate learning-based decision making.
    """
    
    def __init__(
        self,
        decision_module: DecisionModule,
        action_registry: ActionRegistry,
    ):
        self._decision_module = decision_module
        self._action_registry = action_registry
        self._previous_state = None
        self._previous_action = None
    
    def execute_turn(self, agent: "AgentCore") -> None:
        """Execute one turn using reinforcement learning."""
        # Get current state
        current_state = self._create_state(agent)
        
        # Select action
        action_index = self._decision_module.decide_action(current_state)
        action = self._action_registry.get_action(action_index)
        
        # Execute action
        result = action.execute(agent)
        
        # Calculate reward
        reward = self._calculate_reward(agent, result)
        
        # Update learning
        if self._previous_state is not None:
            next_state = self._create_state(agent)
            self._decision_module.update(
                state=self._previous_state,
                action=self._previous_action,
                reward=reward,
                next_state=next_state,
                done=not agent.alive
            )
        
        # Store for next iteration
        self._previous_state = current_state
        self._previous_action = action_index
    
    def _create_state(self, agent: "AgentCore"):
        """Create state representation for decision module."""
        # Use agent's perception component
        perception = agent.get_component("perception")
        if perception:
            return perception.get_observation()
        return None
    
    def _calculate_reward(self, agent: "AgentCore", action_result: dict) -> float:
        """Calculate reward for the action."""
        # Get resource component to check resource changes
        resource = agent.get_component("resource")
        combat = agent.get_component("combat")
        
        reward = 0.0
        
        # Resource rewards
        if resource:
            # Reward based on resource level
            reward += resource.level * 0.1
        
        # Health rewards
        if combat:
            health_ratio = combat.health / combat._config.starting_health
            reward += health_ratio * 0.5
        
        # Survival reward
        reward += 0.1 if agent.alive else -10.0
        
        return reward
```

**Benefits**:
- ✅ **Swappable strategies**: Change behavior without modifying agent
- ✅ **Testable**: Test behavior logic independently
- ✅ **Clear separation**: Decision logic separate from agent state

---

### 4. Configuration Value Objects

**Responsibility**: Type-safe configuration with defaults

```python
# farm/core/agent/config/agent_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)  # Immutable value object
class MovementConfig:
    """Configuration for movement component."""
    max_movement: float = 8.0

@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for resource component."""
    base_consumption_rate: int = 1
    starvation_threshold: int = 100

@dataclass(frozen=True)
class CombatConfig:
    """Configuration for combat component."""
    starting_health: float = 100.0
    base_attack_strength: float = 10.0
    base_defense_strength: float = 5.0
    defense_reduction: float = 0.5  # 50% damage reduction when defending

@dataclass(frozen=True)
class ReproductionConfig:
    """Configuration for reproduction component."""
    offspring_cost: int = 5
    offspring_initial_resources: int = 10
    reproduction_threshold: int = 20  # Minimum resources to reproduce

@dataclass(frozen=True)
class AgentConfig:
    """Complete agent configuration."""
    movement: MovementConfig = MovementConfig()
    resource: ResourceConfig = ResourceConfig()
    combat: CombatConfig = CombatConfig()
    reproduction: ReproductionConfig = ReproductionConfig()
    
    @staticmethod
    def from_dict(config_dict: dict) -> "AgentConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return AgentConfig(
            movement=MovementConfig(**config_dict.get("movement", {})),
            resource=ResourceConfig(**config_dict.get("resource", {})),
            combat=CombatConfig(**config_dict.get("combat", {})),
            reproduction=ReproductionConfig(**config_dict.get("reproduction", {})),
        )
```

**Benefits**:
- ✅ **Type safety**: No more `get_nested_then_flat` everywhere
- ✅ **Immutable**: Configuration can't be accidentally modified
- ✅ **Clear defaults**: All defaults in one place
- ✅ **Easy to test**: Pass mock configs easily

---

### 5. Agent Factory - Clean Construction

**Responsibility**: Construct agents with proper dependency injection

```python
# farm/core/agent/factory.py

from typing import Optional
from farm.core.agent.core import AgentCore
from farm.core.agent.components import (
    MovementComponent,
    ResourceComponent,
    CombatComponent,
    ReproductionComponent,
    PerceptionComponent,
)
from farm.core.agent.behaviors.learning_behavior import LearningAgentBehavior
from farm.core.agent.config.agent_config import AgentConfig
from farm.core.decision.decision import DecisionModule
from farm.core.decision.config import DecisionConfig
from farm.core.services.interfaces import ISpatialQueryService

class AgentFactory:
    """
    Factory for creating agents with proper dependency injection.
    
    Follows SRP: Only responsible for agent construction.
    Follows DIP: Depends on interfaces, not concrete classes.
    """
    
    def __init__(
        self,
        spatial_service: ISpatialQueryService,
        default_config: Optional[AgentConfig] = None,
    ):
        self._spatial_service = spatial_service
        self._default_config = default_config or AgentConfig()
    
    def create_learning_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 10,
        config: Optional[AgentConfig] = None,
        decision_config: Optional[DecisionConfig] = None,
    ) -> AgentCore:
        """
        Create a learning agent with reinforcement learning behavior.
        
        Args:
            agent_id: Unique identifier
            position: Starting position
            initial_resources: Starting resources
            config: Agent configuration (uses default if None)
            decision_config: Decision module configuration
        
        Returns:
            Configured AgentCore instance
        """
        config = config or self._default_config
        
        # Create components
        components = [
            MovementComponent(config.movement),
            ResourceComponent(initial_resources, config.resource),
            CombatComponent(config.combat),
            ReproductionComponent(config.reproduction),
            PerceptionComponent(self._spatial_service),
        ]
        
        # Create decision module (temporary - will be injected later)
        # For now, we'll create a placeholder
        decision_module = None  # Will be properly initialized
        
        # Create behavior strategy
        from farm.core.action.registry import action_registry
        behavior = LearningAgentBehavior(
            decision_module=decision_module,
            action_registry=action_registry,
        )
        
        # Create agent core
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
        )
        
        return agent
    
    def create_simple_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 10,
        config: Optional[AgentConfig] = None,
    ) -> AgentCore:
        """Create a simple agent with random behavior."""
        config = config or self._default_config
        
        # Create components (fewer for simple agent)
        components = [
            MovementComponent(config.movement),
            ResourceComponent(initial_resources, config.resource),
        ]
        
        # Create simple behavior
        from farm.core.agent.behaviors.default_behavior import DefaultAgentBehavior
        behavior = DefaultAgentBehavior()
        
        # Create agent core
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
        )
        
        return agent
```

**Benefits**:
- ✅ **Single responsibility**: Only constructs agents
- ✅ **Dependency injection**: All dependencies provided explicitly
- ✅ **Flexible**: Easy to create different agent configurations
- ✅ **Testable**: Mock dependencies for unit tests

---

## Refactored Action System

### Current Issues
- Actions are functions, not objects
- No polymorphism or type safety
- Validation separated from execution

### Proposed Design

```python
# farm/core/action/base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore

class IAction(ABC):
    """
    Interface for agent actions.
    
    Follows SRP: Each action is its own class.
    Follows OCP: New actions extend interface, don't modify existing.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Action name identifier."""
        pass
    
    @abstractmethod
    def can_execute(self, agent: "AgentCore") -> bool:
        """
        Check if this action can be executed.
        
        Returns:
            bool: True if action is valid to execute
        """
        pass
    
    @abstractmethod
    def execute(self, agent: "AgentCore") -> Dict[str, Any]:
        """
        Execute the action.
        
        Returns:
            dict: Result containing success status and details
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, agent: "AgentCore") -> float:
        """
        Estimate the resource cost of this action.
        
        Returns:
            float: Estimated cost in resources
        """
        pass
```

```python
# farm/core/action/actions/movement.py

from farm.core.action.base import IAction
from farm.core.agent.core import AgentCore
from typing import Dict, Any

class MoveAction(IAction):
    """
    Movement action implementation.
    
    Single Responsibility: Only handles movement logic.
    """
    
    @property
    def name(self) -> str:
        return "move"
    
    def can_execute(self, agent: AgentCore) -> bool:
        """Check if agent can move (has movement component)."""
        return agent.get_component("movement") is not None
    
    def execute(self, agent: AgentCore) -> Dict[str, Any]:
        """Execute movement action."""
        movement = agent.get_component("movement")
        if not movement:
            return {
                "success": False,
                "error": "No movement component"
            }
        
        # Find nearest resource and move towards it
        # (This is simplified - real implementation would use perception)
        success = movement.random_move()
        
        return {
            "success": success,
            "details": {
                "new_position": agent.position
            }
        }
    
    def estimate_cost(self, agent: AgentCore) -> float:
        """Movement has minimal resource cost."""
        return 0.1
```

**Benefits**:
- ✅ **Type safety**: Actions are objects with clear contracts
- ✅ **Validation built-in**: `can_execute` method
- ✅ **Testable**: Mock agent, test action in isolation
- ✅ **Extensible**: Add new actions without modifying existing code

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
- [ ] Create new directory structure
- [ ] Implement base interfaces (IAgentComponent, IAgentBehavior, IAction)
- [ ] Create AgentConfig value objects
- [ ] Implement StateManager
- [ ] Write unit tests for new components

### Phase 2: Core Components (Week 2)
- [ ] Implement MovementComponent
- [ ] Implement ResourceComponent
- [ ] Implement CombatComponent
- [ ] Implement PerceptionComponent
- [ ] Write integration tests for components

### Phase 3: Behavior System (Week 3)
- [ ] Implement DefaultAgentBehavior
- [ ] Implement LearningAgentBehavior
- [ ] Update DecisionModule integration
- [ ] Write behavior tests

### Phase 4: Agent Core (Week 4)
- [ ] Implement AgentCore
- [ ] Implement AgentFactory
- [ ] Create compatibility layer for existing code
- [ ] Write integration tests

### Phase 5: Action Refactoring (Week 5)
- [ ] Refactor actions to use IAction interface
- [ ] Update action registry
- [ ] Migrate all existing actions
- [ ] Write action tests

### Phase 6: Migration (Week 6)
- [ ] Create adapter for old BaseAgent API
- [ ] Gradually migrate existing code
- [ ] Update tests to use new architecture
- [ ] Performance benchmarking

### Phase 7: Cleanup (Week 7)
- [ ] Remove old BaseAgent class
- [ ] Remove compatibility layer
- [ ] Update documentation
- [ ] Final performance validation

---

## Testing Strategy

### Unit Tests
Each component, behavior, and action gets isolated unit tests:

```python
# tests/agent/components/test_resource_component.py

import pytest
from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.config.agent_config import ResourceConfig

def test_resource_consumption():
    """Test basic resource consumption."""
    config = ResourceConfig(base_consumption_rate=5)
    component = ResourceComponent(initial_resources=100, config=config)
    
    # Test consumption
    assert component.consume(20) == True
    assert component.level == 80
    
    # Test insufficient resources
    assert component.consume(100) == False
    assert component.level == 80  # Unchanged

def test_starvation():
    """Test starvation mechanics."""
    config = ResourceConfig(
        base_consumption_rate=1,
        starvation_threshold=3
    )
    component = ResourceComponent(initial_resources=0, config=config)
    
    # Mock agent
    from unittest.mock import Mock
    agent = Mock()
    agent.alive = True
    component.attach(agent)
    
    # Simulate steps until starvation
    for _ in range(3):
        component.on_step_end()
    
    # Agent should be terminated
    agent.terminate.assert_called_once()
```

### Integration Tests
Test component interactions:

```python
# tests/agent/test_agent_integration.py

import pytest
from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.agent_config import AgentConfig
from unittest.mock import Mock

def test_agent_with_components():
    """Test agent with multiple components working together."""
    # Create factory with mock spatial service
    spatial_service = Mock()
    factory = AgentFactory(spatial_service=spatial_service)
    
    # Create agent
    agent = factory.create_learning_agent(
        agent_id="test_agent",
        position=(10, 10),
        initial_resources=100
    )
    
    # Test component interaction
    movement = agent.get_component("movement")
    resource = agent.get_component("resource")
    
    # Move agent
    movement.move_to((20, 20))
    assert agent.position == (20, 20)
    
    # Consume resources
    resource.consume(10)
    assert resource.level == 90
```

### Performance Tests
Ensure refactoring doesn't degrade performance:

```python
# tests/benchmarks/test_agent_performance.py

import pytest
import time
from farm.core.agent.factory import AgentFactory

def test_agent_step_performance():
    """Ensure agent step executes within time budget."""
    factory = AgentFactory(spatial_service=Mock())
    agent = factory.create_learning_agent("perf_test", (0, 0))
    
    # Warm up
    for _ in range(100):
        agent.act()
    
    # Measure
    start = time.perf_counter()
    for _ in range(1000):
        agent.act()
    duration = time.perf_counter() - start
    
    # Should be < 1ms per step on average
    assert duration / 1000 < 0.001
```

---

## Benefits Summary

### Code Quality
- ✅ **95% reduction** in main class size (1571 → 70 lines)
- ✅ **Single Responsibility**: Each class has one clear purpose
- ✅ **Open-Closed**: Add features without modifying existing code
- ✅ **Dependency Inversion**: Depend on abstractions, not concrete classes

### Maintainability
- ✅ **Easier to understand**: Small, focused classes
- ✅ **Easier to modify**: Changes isolated to specific components
- ✅ **Easier to test**: Mock dependencies, test in isolation
- ✅ **Easier to extend**: Add new components/behaviors without risk

### Flexibility
- ✅ **Composable**: Mix and match components for different agent types
- ✅ **Reusable**: Components shared across multiple agent types
- ✅ **Swappable**: Change behaviors without touching core logic
- ✅ **Configurable**: Type-safe configuration with clear defaults

### Testing
- ✅ **Unit testable**: Each component tested independently
- ✅ **Integration testable**: Component interactions verified
- ✅ **Mockable**: Easy to create test doubles
- ✅ **Fast**: Isolated tests run quickly

---

## Comparison: Before vs After

### Before: Monolithic Agent

```python
class BaseAgent:
    """1571 lines of tightly coupled code"""
    
    def __init__(self, ...):  # 20+ parameters
        # Service initialization
        self._initialize_services(...)
        # State initialization
        self._initialize_agent_state()
        # Decision module
        self._initialize_decision_module()
        # Memory
        self._init_memory(...)
        # ... 100+ lines of initialization
    
    def act(self):
        """Complex method mixing concerns"""
        # Resource consumption
        self.resource_level -= consumption
        # Starvation check
        if self.check_starvation(): return
        # Decision making
        action = self.decide_action()
        # Action execution
        result = action.execute(self)
        # Reward calculation
        reward = self._calculate_reward(...)
        # Learning update
        self.decision_module.update(...)
        # ... 100+ lines of mixed logic
```

**Problems**:
- ❌ 1571 lines in one file
- ❌ 13+ responsibilities
- ❌ Hard to test individual behaviors
- ❌ Changes risk breaking unrelated code
- ❌ Difficult to understand and maintain

### After: Modular Agent

```python
class AgentCore:
    """70 lines of clean coordination"""
    
    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        spatial_service: ISpatialQueryService,
        behavior: IAgentBehavior,
        components: List[IAgentComponent],
    ):
        self.agent_id = agent_id
        self.alive = True
        self._spatial_service = spatial_service
        self._behavior = behavior
        self._components = {c.name: c for c in components}
        self._state_manager = StateManager(self)
    
    def act(self):
        """Clean delegation to behavior strategy"""
        if not self.alive:
            return
        
        for component in self._components.values():
            component.on_step_start()
        
        self._behavior.execute_turn(self)
        
        for component in self._components.values():
            component.on_step_end()
```

**Benefits**:
- ✅ 70 lines vs 1571 lines
- ✅ Single responsibility: coordination only
- ✅ Easy to test: mock components and behavior
- ✅ Changes isolated to specific components
- ✅ Clear, understandable structure

---

## Conclusion

This refactoring transforms the agent module from a monolithic, tightly-coupled design into a modular, composable architecture that follows SOLID principles and modern design patterns.

### Key Improvements
1. **95% code reduction** in core class (1571 → 70 lines)
2. **Component-based architecture** for flexibility and reusability
3. **Strategy pattern** for swappable behaviors
4. **Dependency injection** for testability
5. **Type-safe configuration** eliminating repeated boilerplate
6. **Clear separation of concerns** making code easier to understand

### Next Steps
1. Review and approve this design
2. Create detailed implementation tickets for each phase
3. Set up new directory structure and base interfaces
4. Begin Phase 1 implementation with foundation components

The refactoring can be done incrementally with a compatibility layer, ensuring the system continues to work throughout the migration process.