# Select Module

## Overview

The Select Module implements a hierarchical action selection system that serves as the high-level policy for choosing which action to execute. This follows the Options Framework where the select module acts as the meta-controller choosing between different sub-policies (action modules).

## Hierarchical Action Selection

### High-Level Policy

The select module acts as a meta-controller that chooses between different action types:

```python
class SelectModule(BaseDQNModule):
    def __init__(self, num_actions: int, config: SelectConfig, device: torch.device, 
                 shared_encoder: Optional[SharedEncoder] = None):
        super().__init__(input_dim=8, output_dim=num_actions, config=config, 
                        device=device, shared_encoder=shared_encoder)
```

### Action Space

The select module chooses from available actions based on curriculum learning phases:

```python
def decide_action(self):
    current_step = self.environment.time
    enabled_actions = self.actions  # Default all
    
    # Apply curriculum learning
    for phase in self.config.curriculum_phases:
        if current_step < phase["steps"] or phase["steps"] == -1:
            enabled_actions = [a for a in self.actions if a.name in phase["enabled_actions"]]
            break
    
    selected_action = self.select_module.select_action(agent=self, actions=enabled_actions, 
                                                     state=self._cached_selection_state)
    return selected_action
```

## Curriculum Learning Integration

### Progressive Action Enablement

Actions are gradually enabled based on simulation progress:

```python
curriculum_phases = [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
]
```

### Benefits

- **Easier Training**: Agents learn basic survival before complex behaviors
- **Stable Learning**: Reduces action space complexity early in training
- **Progressive Complexity**: Gradually introduces more sophisticated behaviors

## State Representation

The select module uses a comprehensive state representation:

```python
def get_selection_state(self) -> torch.Tensor:
    """Get state for action selection decision."""
    state = [
        self.resource_level / self.config.max_resources,  # Resource ratio
        self.health / self.config.starting_health,        # Health ratio
        len(self.environment.agents) / self.config.max_population,  # Population ratio
        self._get_resource_proximity(),                   # Resource proximity
        self._get_agent_proximity(),                      # Agent proximity
        self._get_starvation_risk(),                      # Starvation risk
        self._get_combat_risk(),                          # Combat risk
        self._get_cooperation_opportunity()               # Cooperation opportunity
    ]
    return torch.FloatTensor(state).to(self.device)
```

## Action Selection Process

### 1. State Assessment

The module evaluates the current situation:

```python
def select_action(self, agent: "BaseAgent", actions: List[Action], 
                 state: torch.Tensor) -> Action:
    """Select action based on current state and available actions."""
    # Get Q-values for all available actions
    q_values = self.q_network(state)
    
    # Filter Q-values for enabled actions
    enabled_q_values = q_values[:len(actions)]
    
    # Select action using epsilon-greedy
    if random.random() < self.epsilon:
        return random.choice(actions)
    else:
        action_idx = enabled_q_values.argmax().item()
        return actions[action_idx]
```

### 2. Epsilon-Greedy Exploration

Balances exploration and exploitation:

```python
def _update_epsilon(self):
    """Update exploration rate."""
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### 3. Reward Calculation

Rewards are based on the success of the chosen action:

```python
def calculate_selection_reward(self, chosen_action: Action, 
                             action_success: bool, reward: float) -> float:
    """Calculate reward for action selection."""
    base_reward = reward if action_success else -0.1
    
    # Add bonus for appropriate action selection
    if self._is_action_appropriate(chosen_action):
        base_reward += 0.2
    
    return base_reward
```

## Integration with Action Modules

### Unified Training

The select module participates in unified training with other modules:

```python
def train_all_modules(self):
    """Unified training for all action modules including selection."""
    modules = [self.move_module, self.attack_module, self.share_module, 
               self.gather_module, self.select_module]
    
    for module in modules:
        if hasattr(module, 'train') and len(module.memory) >= module.config.batch_size:
            batch = random.sample(module.memory, module.config.batch_size)
            module.train(batch)
```

### Shared Feature Extraction

Uses the same `SharedEncoder` as other modules:

```python
def __init__(self, num_actions: int, config: SelectConfig, device: torch.device, 
             shared_encoder: Optional[SharedEncoder] = None):
    super().__init__(input_dim=8, output_dim=num_actions, config=config, 
                    device=device, shared_encoder=shared_encoder)
```

## Configuration

### SelectConfig

```python
class SelectConfig(BaseDQNConfig):
    # Selection-specific parameters
    selection_success_bonus: float = 0.2
    selection_failure_penalty: float = -0.1
    inappropriate_action_penalty: float = -0.3
```

### Curriculum Configuration

```python
class SimulationConfig:
    curriculum_phases: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"steps": 100, "enabled_actions": ["move", "gather"]},
        {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
        {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
    ])
```

## Performance Optimizations

### State Caching

```python
def _cache_selection_state(self, state: torch.Tensor):
    """Cache state for reuse in action selection."""
    self._cached_selection_state = state
```

### Batch Processing

```python
def train_batch(self, batch: List[Tuple]) -> float:
    """Train on batch of selection experiences."""
    if len(batch) < self.config.batch_size:
        return None
    
    # Process batch for training
    states = torch.stack([x[0] for x in batch])
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([x[2] for x in batch])
    next_states = torch.stack([x[3] for x in batch])
    dones = torch.tensor([x[4] for x in batch])
    
    return self._train_step(states, actions, rewards, next_states, dones)
```

## Usage Examples

### Basic Usage

```python
# Initialize select module
select_module = SelectModule(
    num_actions=len(actions),
    config=SelectConfig(),
    device=device,
    shared_encoder=shared_encoder
)

# Select action
state = agent.get_selection_state()
selected_action = select_module.select_action(agent, available_actions, state)
```

### With Curriculum Learning

```python
# Actions are automatically filtered based on curriculum phase
enabled_actions = agent.get_enabled_actions()  # Based on current phase
selected_action = agent.select_module.select_action(agent, enabled_actions, state)
```

## Best Practices

1. **Curriculum Design**: Start with essential actions and gradually add complexity
2. **State Design**: Include relevant information for action selection
3. **Reward Structure**: Balance immediate and long-term rewards
4. **Exploration**: Maintain appropriate exploration rates for meta-learning
5. **Integration**: Ensure select module coordinates well with action modules

## Troubleshooting

### Common Issues

1. **Poor Action Selection**: Check state representation and reward structure
2. **Curriculum Issues**: Verify phase transitions and action enablement
3. **Training Instability**: Adjust learning rates and batch sizes
4. **Exploration Problems**: Monitor epsilon decay and exploration behavior

### Debugging Tips

- Monitor action selection frequencies
- Check curriculum phase transitions
- Verify state representation quality
- Analyze reward distributions 