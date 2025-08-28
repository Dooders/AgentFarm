
# Action System

The action system manages agent behaviors in the multi-agent simulation environment. It defines possible actions agents can take, how they are selected using learning algorithms, and how they are executed in the environment.

## Overview

Agents in the system can perform various actions such as moving, gathering resources, attacking, sharing, reproducing, defending, or passing. The system uses a combination of rule-based execution and learning-based selection (primarily Deep Q-Learning) to enable intelligent agent behavior.

Key features:
- Modular action definitions with weights for selection
- Deep Q-Learning for action selection via DecisionModule
- Curriculum learning support for progressive capability unlocking
- Integration with agent state, perception, and memory systems
- Reward-based learning for adaptive behavior

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

In `BaseAgent.decide_action()`:

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

Each action has a dedicated function in `farm/core/action.py` that implements the behavior:

- **defend_action**: Enter defensive stance, reduce damage, optional healing
- **attack_action**: Find closest target in range, calculate and apply damage
- **gather_action**: Collect from nearest resource node
- **share_action**: Share resources with neediest nearby agent
- **move_action**: Move in random direction within bounds
- **reproduce_action**: Create offspring if resources sufficient
- **pass_action**: Do nothing, small reward

Actions use environment services like spatial indexing for efficiency and log interactions.

## Learning Integration

- Actions generate rewards used for RL training
- State transitions stored for experience replay
- Specialized DQN modules in `farm/core/decision/` for behaviors like movement
- Training occurs after each action via `DecisionModule.update()`

## Usage Example

In agent lifecycle:

```python
# In BaseAgent.act()
current_state_tensor = create_decision_state(self)
action = self.decide_action()  # Uses DecisionModule
action.execute(self)  # Performs the action
reward = self._calculate_reward()
next_state_tensor = create_decision_state(self)
self.decision_module.update(
    state=current_state_tensor,
    action=action_index,
    reward=reward,
    next_state=next_state_tensor,
    done=not self.alive
)
```

## Related Components

- **Decision Modules**: See `farm/core/decision/` for DQN implementations
- **Agent State**: `farm/core/state.py` for state representations
- **Perception**: Local grid view for decision making
- **Memory**: Redis-based experience storage for advanced learning
- **Analysis**: See `docs/action_data.md` for action analytics

For implementation details, refer to `farm/core/action.py` and `farm/core/agent.py`.

## Adding New Actions

To extend the action system with new behaviors:

1. **Define the Action Function**:
   Create a function that implements the behavior. It must take `agent: BaseAgent` as the first parameter.

   Example:
   ```python
   def new_action(agent: "BaseAgent") -> None:
       # Implementation here
       pass
   ```

2. **Register the Action**:
   In `farm/core/action.py`, register the action with a unique name and selection weight (0.0-1.0).

   ```python
   action_registry.register("new_action", 0.1, new_action)
   ```

3. **Update ActionType Enum** (Optional):
   If the action represents a new fundamental type, add it to `ActionType` in `farm/core/action.py`.

   ```python
   class ActionType(IntEnum):
       # ... existing ...
       NEW_ACTION = 7
   ```

4. **Integrate with Learning**:
   - The DecisionModule will automatically handle the new action as its `num_actions` is based on registered actions.
   - Update reward calculations in `BaseAgent._calculate_reward()` if needed.
   - If specialized learning is required, create a new module in `farm/core/decision/`.

5. **Test and Balance**:
   - Add unit tests in `tests/core/test_actions.py`.
   - Adjust weight to balance selection probability.
   - Monitor in simulations to ensure proper integration.

New actions are automatically available to all agents via the registry.
