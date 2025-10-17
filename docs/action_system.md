
# Hybrid Action System

The hybrid action system manages agent behaviors in the multi-agent simulation environment. It combines the environment's action registry (for PettingZoo compatibility) with agent component-based execution, providing a clean separation between external interfaces and internal capabilities.

## Overview

The system uses a hybrid architecture where:
- **Environment Layer**: Maintains the action registry for external compatibility (PettingZoo, RL frameworks)
- **Agent Layer**: Uses component-based architecture for actual action execution
- **Delegation Pattern**: Action registry functions delegate to agent components

Agents can perform various actions such as moving, gathering resources, attacking, sharing, reproducing, defending, or passing. The system supports both learning-based selection (via behaviors) and direct component interaction.

Key features:
- **Hybrid Architecture**: Environment registry + agent components
- **Component-Based Execution**: Actions delegate to specialized agent components
- **PettingZoo Compatibility**: Maintains external RL framework interfaces
- **Clean Separation**: Environment owns registry, agents own capabilities
- **Flexible Behaviors**: Both registry-based and direct component access

## Core Components

### ActionType Enum

Defined in `farm/core/action.py`:

```python
class ActionType(IntEnum):
    DEFEND = 0     # Enter defensive stance, reducing incoming damage
    ATTACK = 1     # Attack nearby agents
    GATHER = 2     # Collect resources from nearby nodes
    SHARE = 3      # Share resources with nearby allies
    MOVE = 4       # Move to a new position
    REPRODUCE = 5  # Create offspring if conditions met
    PASS = 6       # Take no action
```

### Action Class

The base class for all actions, encapsulating name, selection weight, and execution function.

```python
class Action:
    def __init__(self, name, weight, function):
        self.name = name
        self.weight = weight
        self.function = function

    def execute(self, agent, *args, **kwargs):
        self.function(agent, *args, **kwargs)
```

### Action Registry

A global registry for managing available actions:

```python
class action_registry:
    _registry = {}

    @classmethod
    def register(cls, name: str, weight: float, function: Callable) -> None:
        cls._registry[name] = Action(name, weight, function)

    @classmethod
    def get(cls, name: str) -> Action | None:
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> List[Action]:
        return list(cls._registry.values())
```

Actions are registered with default weights:

- attack: 0.1
- move: 0.4
- gather: 0.3
- reproduce: 0.15
- share: 0.2
- defend: 0.25
- pass: 0.05

## Action Selection Process

Action selection is handled by the `DecisionModule` in `farm/core/decision/decision.py`, which uses reinforcement learning (default: DQN via Stable Baselines3) to choose actions based on the current state.

In agent behavior strategies:

1. Create state representation (position, resources, health, etc.)
2. Determine enabled actions based on curriculum phase
3. Use DecisionModule to select action index
4. Map index to Action object from enabled actions

The DecisionModule supports:
- Epsilon-greedy exploration
- Shared feature encoding
- Multiple algorithms (DQN default)
- Per-agent models for personalized learning

Curriculum learning progressively enables actions through phases defined in config.

## Action Execution

The hybrid system uses a delegation pattern where action functions in `farm/core/action.py` delegate to agent components:

### Action-Component Mapping

- **move_action** → `MovementComponent.move_by()`
- **attack_action** → `CombatComponent.attack()`
- **defend_action** → `CombatComponent.start_defense()`
- **gather_action** → `ResourceComponent.add()` + spatial queries
- **share_action** → `ResourceComponent.consume()` + target `ResourceComponent.add()`
- **reproduce_action** → `ReproductionComponent.reproduce()`
- **pass_action** → No-op (minimal implementation)

### Execution Flow

1. **Environment**: `Environment.step(action_index)` calls action registry
2. **Registry**: Action function delegates to agent component
3. **Component**: Performs actual behavior (movement, combat, etc.)
4. **Result**: Standardized result format returned

### Component Integration

Actions access agent components via `agent.get_component(component_name)`:
- Components manage their own state and configuration
- Actions validate component availability before execution
- Spatial services accessed through `agent._spatial_service`
- Lifecycle services for reproduction and agent creation

## Learning Integration

- Actions generate rewards used for RL training
- State transitions stored for experience replay
- Specialized DQN modules in `farm/core/decision/` for behaviors like movement
- Training occurs after each action via `DecisionModule.update()`

## Usage Examples

### Environment Integration (PettingZoo)

```python
# Environment calls action registry
action = action_registry.get("move")
result = action.execute(agent)
# Result: {"success": bool, "error": str, "details": dict}
```

### Agent Behavior Integration

```python
# Behaviors can call components directly (bypassing registry)
class DefaultAgentBehavior:
    def select_action(self, agent):
        movement = agent.get_component("movement")
        if movement:
            movement.move_by(5.0, 0.0)  # Direct component call
```

### Component-Based Execution

```python
# Actions delegate to components
def move_action(agent):
    movement = agent.get_component("movement")
    if not movement:
        return {"success": False, "error": "No movement component", "details": {}}
    
    return movement.move_by(5.0, 0.0)  # Component handles the logic
```

## Related Components

- **Decision Modules**: See `farm/core/decision/` for DQN implementations
- **Agent State**: `farm/core/state.py` for state representations
- **Perception**: Local grid view for decision making
- **Memory**: Redis-based experience storage for advanced learning
- **Analysis**: See `docs/action_data.md` for action analytics

For implementation details, refer to `farm/core/action.py` and `farm/core/agent.py`.

## Adding New Actions

To extend the hybrid action system with new behaviors:

### 1. Create Agent Component (if needed)

If the action requires new capabilities, create a component:

```python
# farm/core/agent/components/new_capability.py
class NewCapabilityComponent(AgentComponent):
    def __init__(self, config: NewCapabilityConfig):
        super().__init__("new_capability", config)
        # Initialize component state
    
    def perform_new_action(self, param1, param2):
        # Implement the behavior
        return {"success": True, "details": {...}}
```

### 2. Define the Action Function

Create a function that delegates to the component:

```python
# farm/core/action.py
def new_action(agent: "AgentCore") -> Dict[str, Any]:
    """New action that delegates to NewCapabilityComponent."""
    component = agent.get_component("new_capability")
    if not component:
        return {
            "success": False,
            "error": "Agent has no new_capability component",
            "details": {}
        }
    
    # Delegate to component
    return component.perform_new_action(param1, param2)
```

### 3. Register the Action

Register with the action registry:

```python
# farm/core/action.py
action_registry.register("new_action", 0.1, new_action)
```

### 4. Update ActionType Enum (Optional)

Add to the enum if it's a fundamental action type:

```python
class ActionType(IntEnum):
    # ... existing ...
    NEW_ACTION = 7
```

### 5. Add Agent Configuration

Update agent configuration to include the new component:

```python
# farm/core/agent/config/agent_config.py
@dataclass
class NewCapabilityConfig:
    param1: float = 1.0
    param2: int = 10

# In AgentConfig
new_capability: NewCapabilityConfig = field(default_factory=NewCapabilityConfig)
```

### 6. Test the Integration

Add comprehensive tests:

```python
# tests/test_hybrid_action_system.py
def test_new_action_delegates_to_component(self):
    agent = self.create_test_agent(components=[
        NewCapabilityComponent(NewCapabilityConfig()),
    ])
    
    result = new_action(agent)
    assert result["success"] is True
```

### 7. Update Documentation

- Update this documentation with the new action
- Add examples of usage
- Document any special requirements

The new action will be automatically available through both the action registry (for environment integration) and direct component access (for behaviors).
