# Action System

## Overview

The action system in AgentFarm implements a hierarchical reinforcement learning approach where agents learn both high-level action selection and low-level action execution. This design follows the Options Framework (Sutton et al., 1999) where "options" are sub-policies chosen by a high-level policy.

## Hierarchical Action System Optimizations

### Shared Feature Extraction

The system uses a `SharedEncoder` to extract common features across all action modules, reducing computational redundancy and improving learning efficiency.

```python
class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_size)
    
    def forward(self, x):
        return F.relu(self.fc(x))  # Shared features
```

**Benefits:**
- Reduces parameter count across modules
- Enables shared learning of common features (position, health, resources)
- Improves training stability through shared representations

### Unified Training Loop

The `BaseAgent` class implements a centralized training mechanism that updates all modules simultaneously:

```python
def train_all_modules(self):
    """Unified training for all action modules."""
    for module in [self.move_module, self.attack_module, self.share_module, 
                   self.gather_module, self.select_module]:
        if hasattr(module, 'train') and len(module.memory) >= module.config.batch_size:
            batch = random.sample(module.memory, module.config.batch_size)
            module.train(batch)
```

### Curriculum Learning

Actions are gradually enabled based on simulation progress to ease training complexity:

```python
curriculum_phases = [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
]
```

### Rule-Based Simplification

Simple actions like reproduction use rule-based logic instead of DQN to reduce complexity:

```python
def reproduce_action(agent: "BaseAgent") -> None:
    if random.random() < 0.5 and agent.resource_level >= agent.config.min_reproduction_resources:
        agent.reproduce()
```

## Action Modules

### Movement Module (`move.py`)
- **Purpose**: Learn optimal navigation policies
- **Action Space**: 4 discrete directions (right, left, up, down)
- **State**: Position, resource proximity, health
- **Reward**: Movement cost + resource approach bonus

### Attack Action (`action.py`)
- **Purpose**: Simple closest-agent combat using spatial index
- **Method**: Finds nearest valid target within range and attacks
- **Features**: Health-based damage, defensive reduction, efficient spatial queries
- **No DQN**: Direct spatial index-based implementation for simplicity

### Gather Module (`gather.py`)
- **Purpose**: Learn resource collection strategies
- **Action Space**: 3 actions (gather, wait, skip)
- **State**: Resource availability, distance, efficiency
- **Reward**: Resource amount collected minus costs

### Share Module (`share.py`)
- **Purpose**: Learn cooperative resource sharing
- **Action Space**: Share amounts and target selection
- **State**: Own resources, nearby agents, cooperation history
- **Reward**: Altruism bonuses and reciprocity

### Select Module (`select.py`)
- **Purpose**: High-level action selection policy
- **Action Space**: Choose which action to execute
- **State**: Current situation, available actions
- **Reward**: Success of chosen action

## Configuration

Each module uses `BaseDQNConfig` with module-specific parameters:

```python
class MoveConfig(BaseDQNConfig):
    move_base_cost: float = -0.1
    move_resource_approach_reward: float = 0.3
    move_resource_retreat_penalty: float = -0.2
```

## Usage

Actions are typically called from the agent's decision loop:

```python
# In agent update method
state = agent.get_state()
selected_action = agent.select_module.select_action(agent, actions, state)
selected_action.execute(agent)
```

## Dependencies

- PyTorch: Neural network implementation
- NumPy: Numerical computations
- farm.core: Base agent and environment classes

## Development Notes

- Each action module extends `BaseDQNModule` for consistent learning
- State spaces are action-specific but follow similar normalization patterns
- Rewards are designed to promote balanced agent behaviors
- Shared encoder reduces computational overhead
- Curriculum learning eases training complexity