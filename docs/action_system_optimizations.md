# Action System Optimizations

## Overview

This document summarizes the key optimizations implemented in the AgentFarm action system to improve learning efficiency, reduce complexity, and enhance performance.

## Key Optimizations Implemented

### 1. Shared Feature Extraction

**What**: Added `SharedEncoder` class to extract common features across all action modules.

**Implementation**:
```python
class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_size)
    
    def forward(self, x):
        return F.relu(self.fc(x))
```

**Benefits**:
- Reduces parameter count across modules
- Enables shared learning of common features (position, health, resources)
- Improves training stability through shared representations
- Better computational efficiency

### 2. Unified Training Loop

**What**: Centralized training mechanism in `BaseAgent` that updates all modules simultaneously.

**Implementation**:
```python
def train_all_modules(self):
    """Unified training for all action modules."""
    for module in [self.move_module, self.attack_module, self.share_module, 
                   self.gather_module, self.select_module]:
        if hasattr(module, 'train') and len(module.memory) >= module.config.batch_size:
            batch = random.sample(module.memory, module.config.batch_size)
            module.train(batch)
```

**Benefits**:
- Coordinated learning across all modules
- Shared experience where possible
- Better resource utilization
- Simplified training management

### 3. Curriculum Learning

**What**: Progressive action enablement based on simulation progress.

**Implementation**:
```python
curriculum_phases = [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
]
```

**Benefits**:
- Easier training with reduced complexity early on
- Stable learning progression
- Progressive introduction of sophisticated behaviors
- Better exploration of action space

### 4. Rule-Based Simplification

**What**: Simplified reproduction from DQN to rule-based logic.

**Implementation**:
```python
def reproduce_action(agent: "BaseAgent") -> None:
    if random.random() < 0.5 and agent.resource_level >= agent.config.min_reproduction_resources:
        agent.reproduce()
```

**Benefits**:
- Reduced complexity for simple behaviors
- Faster execution
- Easier debugging and analysis
- Predictable behavior

## Architecture Changes

### Before Optimization
- Each action module had independent DQN networks
- No shared feature extraction
- Separate training loops for each module
- All actions used DQN regardless of complexity
- No progressive learning structure

### After Optimization
- Shared encoder for common features
- Unified training across all modules
- Curriculum learning for progressive complexity
- Rule-based logic for simple actions
- Hierarchical action selection system

## Performance Improvements

### Computational Efficiency
- **Shared Encoder**: Reduces redundant feature extraction
- **Unified Training**: Better resource utilization
- **Rule-Based Actions**: Faster execution for simple behaviors

### Learning Stability
- **Curriculum Learning**: Easier initial training
- **Shared Representations**: More stable feature learning
- **Coordinated Training**: Better module coordination

### Memory Usage
- **Shared Encoder**: Reduced parameter count
- **Efficient Training**: Better memory management
- **Simplified Actions**: Less memory for rule-based actions

## Configuration Changes

### Added to SimulationConfig
```python
curriculum_phases: List[Dict[str, Any]] = field(default_factory=lambda: [
    {"steps": 100, "enabled_actions": ["move", "gather"]},
    {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
    {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
])
```

### Modified BaseQNetwork
```python
def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, 
             shared_encoder: Optional[SharedEncoder] = None) -> None:
    super().__init__()
    self.shared_encoder = shared_encoder
    effective_input = hidden_size if shared_encoder else input_dim
    # ... rest of network architecture
```

## Usage Examples

### Initializing with Shared Encoder
```python
# Initialize shared encoder
shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)

# Pass to all modules
move_module = MoveModule(config, shared_encoder=shared_encoder)
attack_module = AttackModule(config, shared_encoder=shared_encoder)
# ... etc
```

### Curriculum Learning Integration
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

## Best Practices

1. **Use Shared Encoder**: Initialize one `SharedEncoder` per agent and pass to all modules
2. **Curriculum Learning**: Start with simpler actions and gradually add complexity
3. **Rule-Based Simplification**: Use rule-based logic for simple actions like reproduction
4. **Unified Training**: Train all modules together for better coordination
5. **State Caching**: Enable state caching for performance-critical applications

## Migration Guide

### For Existing Code
1. Update module initialization to include shared encoder
2. Modify action selection to use curriculum learning
3. Replace DQN-based reproduction with rule-based logic
4. Update training calls to use unified training

### For New Development
1. Start with curriculum learning phases
2. Use shared encoder for all modules
3. Implement rule-based logic for simple actions
4. Use unified training from the beginning

## Future Enhancements

1. **Adaptive Curriculum**: Dynamic curriculum based on learning progress
2. **Advanced Shared Encoder**: More sophisticated shared representations
3. **Hybrid Approaches**: Combine rule-based and learning-based methods
4. **Meta-Learning**: Learn to adapt the curriculum and shared encoder

## References

- **Options Framework**: Sutton et al., 1999
- **Hierarchical RL**: Dayan & Hinton, 1993
- **Curriculum Learning**: Bengio et al., 2009
- **Shared Representations**: Caruana, 1997 