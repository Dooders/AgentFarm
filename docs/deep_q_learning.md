# Deep Q-Learning in AgentFarm

## Overview

AgentFarm uses Deep Q-Learning (DQN) for learning optimal policies across multiple action types. The system implements a hierarchical approach with shared feature extraction to improve learning efficiency and reduce computational overhead.

## Shared Feature Extraction

### SharedEncoder Architecture

The system uses a `SharedEncoder` to extract common features across all action modules, reducing redundancy and improving learning efficiency:

```python
class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_size)
    
    def forward(self, x):
        return F.relu(self.fc(x))  # Shared features
```

### Benefits of Shared Encoding

- **Reduced Parameter Count**: Common features (position, health, resources) are learned once
- **Improved Training Stability**: Shared representations provide consistent feature extraction
- **Better Generalization**: Common patterns are learned across all action types
- **Computational Efficiency**: Avoids redundant feature extraction

### Integration with BaseQNetwork

The `BaseQNetwork` class has been modified to optionally use the shared encoder:

```python
class BaseQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        effective_input = hidden_size if shared_encoder else input_dim
        # ... rest of network architecture
```

## Core DQN Components

### BaseDQNConfig

Configuration class for all DQN modules with sensible defaults:

```python
class BaseDQNConfig:
    target_update_freq: int = 100
    memory_size: int = 10000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    dqn_hidden_size: int = 64
    batch_size: int = 32
    tau: float = 0.005
```

### BaseQNetwork

Neural network architecture with layer normalization and dropout:

```python
class BaseQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_dim),
        )
```

### BaseDQNModule

Core DQN functionality with experience replay and target networks:

```python
class BaseDQNModule:
    def __init__(self, input_dim: int, output_dim: int, config: BaseDQNConfig):
        # Initialize networks, optimizer, and memory
        self.q_network = BaseQNetwork(input_dim, output_dim, config.dqn_hidden_size)
        self.target_network = BaseQNetwork(input_dim, output_dim, config.dqn_hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=config.memory_size)
```

## Training Process

### Experience Replay

Experiences are stored in a replay buffer for stable learning:

```python
def store_experience(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
```

### Training Step

The training process uses Double Q-Learning to reduce overestimation bias:

```python
def train(self, batch):
    # Unpack batch
    states = torch.stack([x[0] for x in batch])
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([x[2] for x in batch])
    next_states = torch.stack([x[3] for x in batch])
    dones = torch.tensor([x[4] for x in batch])
    
    # Get current Q values
    current_q_values = self.q_network(states).gather(1, actions)
    
    # Compute target Q values using Double Q-Learning
    with torch.no_grad():
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions)
        target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
    
    # Compute loss and update
    loss = self.criterion(current_q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Soft Target Updates

Target network is updated using soft updates for stability:

```python
def _soft_update_target_network(self):
    for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
        target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
```

## Action Selection

### Epsilon-Greedy Strategy

Action selection uses epsilon-greedy exploration with state caching:

```python
def select_action(self, state_tensor: torch.Tensor, epsilon: Optional[float] = None) -> int:
    if epsilon is None:
        epsilon = self.epsilon
    
    if random.random() < epsilon:
        return random.randint(0, self.output_dim - 1)
    
    with torch.no_grad():
        return self.q_network(state_tensor).argmax().item()
```

### State Caching

Performance optimization through state caching:

```python
# Cache tensor hash for repeated states
state_hash = hash(state_tensor.cpu().numpy().tobytes())
if state_hash in self._state_cache:
    return self._state_cache[state_hash]
```

## Module-Specific Implementations

### Movement Module

```python
class MoveQNetwork(BaseQNetwork):
    def __init__(self, input_dim: int, hidden_size: int = 64, shared_encoder: Optional[SharedEncoder] = None):
        super().__init__(input_dim, output_dim=4, hidden_size=hidden_size, shared_encoder=shared_encoder)
```

### Attack Module

```python
class AttackQNetwork(BaseQNetwork):
    def __init__(self, input_dim: int, hidden_size: int = 64, shared_encoder: Optional[SharedEncoder] = None):
        super().__init__(input_dim, output_dim=5, hidden_size=hidden_size, shared_encoder=shared_encoder)
```

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

## Performance Optimizations

### Gradient Clipping

Prevents exploding gradients during training:

```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
```

### Device Management

Automatic GPU/CPU device selection:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Management

Efficient experience replay with size limits:

```python
self.memory: Deque = deque(maxlen=config.memory_size)
```

## Configuration Examples

### Basic Configuration

```python
config = BaseDQNConfig(
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)
```

### Module-Specific Configuration

```python
class MoveConfig(BaseDQNConfig):
    move_base_cost: float = -0.1
    move_resource_approach_reward: float = 0.3
    move_resource_retreat_penalty: float = -0.2
```

## Best Practices

1. **Use Shared Encoder**: Initialize one `SharedEncoder` per agent and pass to all modules
2. **Curriculum Learning**: Start with simpler actions and gradually add complexity
3. **Rule-Based Simplification**: Use rule-based logic for simple actions like reproduction
4. **Unified Training**: Train all modules together for better coordination
5. **State Caching**: Enable state caching for performance-critical applications

## Troubleshooting

### Common Issues

1. **High Loss Values**: Check learning rate and batch size
2. **Poor Convergence**: Verify reward structure and exploration parameters
3. **Memory Issues**: Reduce memory_size or batch_size
4. **Slow Training**: Enable GPU acceleration and state caching

### Debugging Tips

- Monitor epsilon decay to ensure proper exploration
- Check reward distributions for balanced learning
- Verify target network updates are occurring
- Use gradient clipping to prevent instability