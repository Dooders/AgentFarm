
# Actions Module

## Overview

The `actions` module implements a hierarchical reinforcement learning system for agent behaviors in the AgentFarm simulation. The system follows the Options Framework where agents learn both high-level action selection and low-level action execution using Deep Q-Learning (DQN) with shared feature extraction.

## Hierarchical Architecture

### High-Level Policy (Select Module)
- **Purpose**: Choose which action to execute
- **Method**: DQN-based meta-controller
- **Input**: Current situation and available actions
- **Output**: Selected action type

### Low-Level Policies (Action Modules)
- **Purpose**: Execute specific actions optimally
- **Method**: DQN-based sub-policies
- **Input**: Action-specific state representation
- **Output**: Action parameters (direction, amount, etc.)

## Shared Feature Extraction

The system uses a `SharedEncoder` to extract common features across all modules:

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
- Enables shared learning of common features
- Improves training stability
- Better computational efficiency

## Components

### Core Infrastructure
- **`__init__.py`**: Exports main action functions for easy import
- **`base_dqn.py`**: Provides base classes for DQN implementation with shared encoder support
- **`action_space.md`**: Documents the action space design and philosophy

### Action Modules
- **`attack.py`**: Handles aggressive interactions between agents using DQN
- **`gather.py`**: Manages resource collection with learning-based decisions
- **`move.py`**: Controls agent movement and navigation with shared encoder
- **`reproduce.py`**: **Rule-based** reproduction (simplified from DQN)
- **`select.py`**: High-level action selection policy
- **`share.py`**: Manages cooperative resource sharing between agents

## Curriculum Learning

Actions are gradually enabled based on simulation progress:

```python
curriculum_phases = [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
]
```

**Benefits:**
- Easier training with reduced complexity
- Stable learning progression
- Progressive introduction of sophisticated behaviors

## Unified Training

The `BaseAgent` class implements centralized training for all modules:

```python
def train_all_modules(self):
    """Unified training for all action modules."""
    for module in [self.move_module, self.attack_module, self.share_module, 
                   self.gather_module, self.select_module]:
        if hasattr(module, 'train') and len(module.memory) >= module.config.batch_size:
            batch = random.sample(module.memory, module.config.batch_size)
            module.train(batch)
```

## Rule-Based Simplification

Simple actions like reproduction use rule-based logic instead of DQN:

```python
def reproduce_action(agent: "BaseAgent") -> None:
    if random.random() < 0.5 and agent.resource_level >= agent.config.min_reproduction_resources:
        agent.reproduce()
```

**Benefits:**
- Reduced complexity for simple behaviors
- Faster execution
- Easier debugging and analysis
- Predictable behavior

## Key Features

- **Hierarchical Learning**: Separate policies for selection and execution
- **Shared Feature Extraction**: Common encoder reduces computational overhead
- **Curriculum Learning**: Progressive action enablement
- **Unified Training**: Centralized learning across all modules
- **Rule-Based Simplification**: Simple actions use deterministic logic
- **State Representation**: Normalized state vectors for decision making
- **Reward Systems**: Custom rewards encourage desired behaviors
- **Configuration**: Extensible config classes for tuning behavior
- **Logging**: Integration with simulation database for action tracking

## Usage

Actions are typically called from the agent's decision loop:

```python
from farm.actions import move_action, gather_action, reproduce_action

# In agent update method
state = agent.get_state()
if condition:
    move_action(agent)
elif other_condition:
    gather_action(agent)
else:
    reproduce_action(agent)  # Rule-based
```

For custom configurations with shared encoder:

```python
from farm.actions.base_dqn import BaseDQNConfig, SharedEncoder
from farm.actions.move import MoveModule

# Initialize shared encoder
shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)

# Create module with shared encoder
config = BaseDQNConfig(learning_rate=0.001)
move_module = MoveModule(config, shared_encoder=shared_encoder)
```

## Dependencies

- PyTorch: For neural network implementation
- NumPy: For numerical computations
- farm.core: For base agent and environment classes

## Development Notes

- Each action module extends `BaseDQNModule` for consistent learning implementation
- State spaces are action-specific but follow similar normalization patterns
- Rewards are designed to promote balanced agent behaviors
- Shared encoder reduces computational overhead across modules
- Curriculum learning eases training complexity
- Rule-based actions provide predictable behavior for simple tasks

For more details, see individual module docstrings and the documentation in `docs/`. 

## Adding New Actions

To add a new action easily:

1. Create newaction.py in farm/actions/

2. Define def newaction_action(agent): # implementation

3. At the end: from farm.core.action import action_registry; action_registry.register('newaction', 0.1, newaction_action)

That's it! It will be automatically included in agents' action sets. 