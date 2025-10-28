# SeedController Documentation

The SeedController is a centralized system for managing deterministic random number generation in AgentFarm simulations. It ensures that each agent has its own isolated random number generator while maintaining full determinism across simulation runs.

## Overview

The SeedController solves the critical problem of non-deterministic behavior caused by agent processing order. Previously, agents were processed in dictionary iteration order (`_agent_objects.values()`), which could vary between runs, especially when agents are added or removed during simulation. This led to different random sequences being generated for the same agents, breaking determinism.

## Key Features

- **Per-Agent RNG Isolation**: Each agent gets its own deterministic RNG instances
- **Order Independence**: Agent behavior is independent of processing order
- **Service-Based Architecture**: Integrated via AgentServices for clean dependency injection
- **Backward Compatibility**: Falls back to global random when SeedController is not available
- **Multi-RNG Support**: Provides Python, NumPy, and PyTorch RNG instances

## Architecture

### Core Components

1. **SeedController**: Main class that generates per-agent RNG instances
2. **AgentServices**: Service container that includes SeedController
3. **AgentFactory**: Injects per-agent RNGs into created agents
4. **Agent Behaviors**: Use per-agent RNGs for decision making

### Integration Flow

```
Environment (with seed) 
    ↓
create_services_from_environment()
    ↓
AgentServices (includes SeedController)
    ↓
AgentFactory (uses SeedController)
    ↓
AgentCore (with _py_rng, _np_rng, _torch_gen)
    ↓
Behaviors (use per-agent RNGs)
```

## Usage

### Basic Usage

```python
from farm.core.seed_controller import SeedController

# Create controller with global seed
seed_controller = SeedController(42)

# Get per-agent RNG instances
py_rng, np_rng, torch_gen = seed_controller.get_agent_rng("agent_001")

# Use the RNGs
random_value = py_rng.random()
numpy_value = np_rng.random()
torch_value = torch_gen.random()
```

### Integration with Simulation

The SeedController is automatically integrated when you create an environment with a seed:

```python
from farm.core.environment import Environment
from farm.core.simulation import create_services_from_environment
from farm.core.agent.factory import AgentFactory

# Create environment with seed
environment = Environment(width=100, height=100, seed=42)

# Create services (includes SeedController)
services = create_services_from_environment(environment)

# Create factory
factory = AgentFactory(services)

# Create agents (automatically get per-agent RNGs)
agent = factory.create_default_agent("agent_001", (50.0, 50.0))

# Agent now has per-agent RNGs
assert hasattr(agent, '_py_rng')
assert hasattr(agent, '_np_rng')
assert hasattr(agent, '_torch_gen')
```

### Component-Specific RNGs

For components that need their own randomness streams:

```python
# Get component-specific RNG
py_rng, np_rng, torch_gen = seed_controller.get_component_rng("agent_001", "movement")

# Different components get different seeds
movement_rng = seed_controller.get_component_rng("agent_001", "movement")
perception_rng = seed_controller.get_component_rng("agent_001", "perception")
# movement_rng and perception_rng will produce different sequences
```

## Implementation Details

### Seed Derivation Algorithm

The SeedController uses deterministic hashing to derive per-agent seeds:

```python
def get_agent_rng(self, agent_id: str):
    # Derive agent-specific seed using hash for determinism
    agent_seed = hash((self.global_seed, agent_id)) % (2**32)
    
    # Create seeded RNG instances
    py_rng = random.Random(agent_seed)
    np_rng = np.random.default_rng(agent_seed)
    torch_gen = torch.Generator().manual_seed(agent_seed)
    
    return py_rng, np_rng, torch_gen
```

### Properties

- **Deterministic**: Same agent ID + same global seed = same RNG sequence
- **Unique**: Different agent IDs = different RNG sequences  
- **Reproducible**: Identical sequences across simulation runs
- **Isolated**: Each agent's randomness is independent

### Hash Function Choice

The implementation uses Python's built-in `hash()` function because:
- It's deterministic within a Python session
- It's fast and efficient
- It provides good distribution of seed values
- It's consistent across different Python versions

## API Reference

### SeedController Class

#### `__init__(global_seed: int)`

Initialize the SeedController with a global seed.

**Parameters:**
- `global_seed`: Global seed value for deterministic behavior

#### `get_agent_rng(agent_id: str) -> Tuple[random.Random, np.random.Generator, torch.Generator]`

Get deterministic RNG instances for a specific agent.

**Parameters:**
- `agent_id`: Unique identifier for the agent

**Returns:**
- Tuple of (python_rng, numpy_rng, torch_generator) instances seeded with agent-specific values

#### `get_component_rng(agent_id: str, component_name: str) -> Tuple[random.Random, np.random.Generator, torch.Generator]`

Get deterministic RNG instances for a specific component.

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `component_name`: Name of the component (e.g., 'movement', 'perception')

**Returns:**
- Tuple of (python_rng, numpy_rng, torch_generator) instances seeded with component-specific values

## Testing

### Unit Tests

The SeedController includes comprehensive unit tests:

```python
# Test deterministic behavior
def test_get_agent_rng_deterministic():
    controller = SeedController(42)
    agent_id = "test_agent_001"
    
    py_rng1, np_rng1, torch_gen1 = controller.get_agent_rng(agent_id)
    py_rng2, np_rng2, torch_gen2 = controller.get_agent_rng(agent_id)
    
    # Should produce identical sequences
    values1 = [py_rng1.random() for _ in range(10)]
    values2 = [py_rng2.random() for _ in range(10)]
    assert values1 == values2

# Test different agents get different sequences
def test_different_agents_different_seeds():
    controller = SeedController(42)
    
    agent1_rng = controller.get_agent_rng("agent_001")
    agent2_rng = controller.get_agent_rng("agent_002")
    
    # Different agents should produce different sequences
    py_values1 = [agent1_rng[0].random() for _ in range(10)]
    py_values2 = [agent2_rng[0].random() for _ in range(10)]
    assert py_values1 != py_values2
```

### Integration Tests

Integration tests verify that the SeedController works correctly with:
- AgentFactory
- Agent behaviors
- DecisionModule
- Environment initialization

### Deterministic Simulation Tests

The deterministic test script validates that simulations with SeedController are fully deterministic:

```bash
python tests/test_deterministic.py --steps 100 --seed 42
```

## Best Practices

### 1. Always Use Seeds

Always provide a seed when creating environments:

```python
# Good
environment = Environment(width=100, height=100, seed=42)

# Avoid
environment = Environment(width=100, height=100)  # Non-deterministic
```

### 2. Consistent Agent IDs

Use consistent agent ID patterns for reproducible behavior:

```python
# Good - deterministic IDs
agent_id = f"agent_{i:03d}"

# Avoid - random IDs
agent_id = f"agent_{random.randint(1000, 9999)}"
```

### 3. Test Determinism

Regularly test that your simulations are deterministic:

```python
def test_simulation_determinism():
    # Run simulation twice with same seed
    results1 = run_simulation(seed=42)
    results2 = run_simulation(seed=42)
    
    # Should be identical
    assert results1 == results2
```

### 4. Document Seed Usage

Keep track of which seeds were used for which experiments:

```python
# Document seed usage
EXPERIMENT_SEEDS = {
    "baseline_experiment": 42,
    "learning_experiment": 123,
    "performance_test": 456,
}
```

## Troubleshooting

### Non-Deterministic Behavior

If your simulation is still non-deterministic:

1. **Check SeedController Integration**: Ensure SeedController is being created and injected
2. **Verify Agent IDs**: Make sure agent IDs are consistent across runs
3. **Check Global RNG Usage**: Look for code that uses global `random` or `np.random` instead of per-agent RNGs
4. **Test Incrementally**: Test determinism with simpler configurations first

### Performance Considerations

The SeedController has minimal performance impact:
- Hash computation is very fast
- RNG creation is lightweight
- No significant memory overhead

### Debugging

Enable debug logging to see SeedController activity:

```python
import logging
logging.getLogger("farm.core.seed_controller").setLevel(logging.DEBUG)
```

This will log when RNG instances are created and which seeds are used.

## Migration Guide

### From Global Random to SeedController

If you have existing code using global random:

**Before:**
```python
import random
action = random.choice(actions)
```

**After:**
```python
# Use per-agent RNG if available
if hasattr(agent, '_py_rng'):
    action = agent._py_rng.choice(actions)
else:
    action = random.choice(actions)  # Fallback
```

### Updating Custom Behaviors

If you have custom behaviors that use random numbers:

```python
class CustomBehavior(IAgentBehavior):
    def decide_action(self, core, state, enabled_actions=None):
        # Use per-agent RNG
        if hasattr(core, '_py_rng'):
            rng = core._py_rng
        else:
            rng = random  # Fallback
        
        # Use rng instead of global random
        action = rng.choice(enabled_actions)
        return action
```

## Future Enhancements

Potential future improvements to the SeedController:

1. **RNG State Persistence**: Save and restore RNG states for checkpointing
2. **Custom Hash Functions**: Allow custom seed derivation algorithms
3. **RNG Pool Management**: Reuse RNG instances for better memory efficiency
4. **Advanced Seeding**: Support for hierarchical seeding strategies

## Conclusion

The SeedController provides a robust, efficient solution for deterministic random number generation in AgentFarm simulations. By isolating each agent's randomness while maintaining determinism, it enables reproducible experiments and reliable debugging while preserving the natural randomness that makes agent-based simulations interesting and realistic.
