
# Actions Module

## Overview

The `actions` module implements various behaviors for agents in the AgentFarm simulation using Deep Q-Learning (DQN). Each action is contained in its own file and provides learning-based decision making for agent interactions in the environment.

This module is part of the broader AgentFarm system, which simulates multi-agent interactions with learning capabilities.

## Components

- **`__init__.py`**: Exports main action functions for easy import.
- **`attack.py`**: Handles aggressive interactions between agents, including target selection and damage calculation.
- **`base_dqn.py`**: Provides base classes for DQN implementation used across all action modules.
- **`gather.py`**: Manages resource collection from the environment.
- **`move.py`**: Controls agent movement and navigation.
- **`reproduce.py`**: Implements reproduction mechanics for agent population growth.
- **`select.py`**: Handles action selection and prioritization.
- **`share.py`**: Manages cooperative resource sharing between agents.

## Key Features

- **Deep Q-Learning Integration**: Each action uses DQN for learning optimal policies.
- **State Representation**: Actions use normalized state vectors for decision making.
- **Reward Systems**: Custom rewards encourage desired behaviors.
- **Configuration**: Extensible config classes for tuning behavior.
- **Logging**: Integration with simulation database for action tracking.

## Usage

Actions are typically called from the agent's decision loop. Example:

```python
from farm.actions import move_action, gather_action

# In agent update method
state = agent.get_state()
if condition:
    move_action(agent)
else:
    gather_action(agent)
```

For custom configurations:

```python
from farm.actions.base_dqn import BaseDQNConfig
from farm.actions.move import MoveModule

config = BaseDQNConfig(learning_rate=0.001)
move_module = MoveModule(config)
```

## Dependencies

- PyTorch: For neural network implementation
- NumPy: For numerical computations
- farm.core: For base agent and environment classes

## Development Notes

- Each action module extends `BaseDQNModule` for consistent learning implementation.
- State spaces are action-specific but follow similar normalization patterns.
- Rewards are designed to promote balanced agent behaviors.

For more details, see individual module docstrings. 