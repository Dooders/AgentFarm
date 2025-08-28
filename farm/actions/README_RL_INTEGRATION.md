# Stable Baselines RL Integration

This document describes the integration of Stable Baselines3 algorithms with AgentFarm's action selection system, providing enhanced reinforcement learning capabilities as requested in issue #269.

## Overview

The integration adds support for state-of-the-art RL algorithms while maintaining backward compatibility with existing systems. Key features include:

- **Multiple RL Algorithms**: PPO, SAC, A2C, TD3 implementations
- **Unified Interface**: Consistent API across all algorithms
- **Benchmarking Tools**: Compare algorithms performance
- **Flexible Configuration**: Easy switching between algorithms
- **Experience Replay**: Built-in replay buffers for stable learning

## Dependencies

The following packages have been added to `requirements.txt`:

```bash
stable-baselines3>=2.0.0  # Modern RL algorithms (PPO, SAC, A2C, TD3, etc.)
gymnasium>=0.28.0        # Modern Gym replacement for environments
shimmy>=1.0.0           # Environment wrappers for Stable Baselines compatibility
```

Install with:
```bash
pip install stable-baselines3 gymnasium shimmy
```

## Architecture

### Core Components

1. **`RLAlgorithm`**: Abstract base class extending `ActionAlgorithm` for RL-specific methods
2. **`StableBaselinesWrapper`**: Base wrapper class for all Stable Baselines algorithms
3. **Algorithm Wrappers**: Specific implementations for PPO, SAC, A2C, TD3
4. **`AlgorithmBenchmark`**: Benchmarking and comparison utilities
5. **Enhanced Configuration**: Extended `SelectConfig` for RL parameters

### File Structure

```
farm/actions/algorithms/
├── __init__.py
├── base.py                 # AlgorithmRegistry (updated)
├── rl_base.py             # RLAlgorithm interface & SimpleReplayBuffer
├── stable_baselines.py    # Stable Baselines wrappers
├── benchmark.py           # Benchmarking utilities
└── examples/
    └── rl_algorithm_usage.py  # Usage examples
```

## Usage

### Basic Usage with SelectModule

```python
from farm.actions.config import SelectConfig
from farm.actions.select import SelectModule

# Configure for PPO
config = SelectConfig(
    algorithm_type="ppo",
    rl_state_dim=8,
    rl_buffer_size=10000,
    rl_batch_size=32,
    algorithm_params={
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64
    }
)

# Create SelectModule with PPO
select_module = SelectModule(
    num_actions=6,  # Number of action types
    config=config
)

# Use in action selection
action = select_module.select_action(agent, available_actions, state)
```

### Using RL Algorithms Directly

```python
from farm.actions.algorithms import PPOWrapper

# Create PPO algorithm directly
ppo = PPOWrapper(
    num_actions=6,
    state_dim=8,
    algorithm_kwargs={
        'learning_rate': 3e-4,
        'n_steps': 128
    }
)

# Select action
state = np.random.randn(8)
action = ppo.select_action(state)

# Store experience
ppo.store_experience(state, action, reward, next_state, done)
```

### Algorithm Comparison

```python
from farm.actions.algorithms import AlgorithmBenchmark, AlgorithmComparison
from pathlib import Path

# Define algorithms to compare
algorithms = [
    ("ppo", {"learning_rate": 3e-4}),
    ("sac", {"learning_rate": 3e-4}),
    ("a2c", {"learning_rate": 7e-4}),
    ("td3", {"learning_rate": 1e-3}),
]

# Run benchmark
benchmark = AlgorithmBenchmark(
    algorithms=algorithms,
    num_actions=6,
    state_dim=8,
    num_episodes=100,
    save_path=Path("benchmark_results")
)

results = benchmark.run_benchmark()

# Compare results
comparison_df = AlgorithmComparison.compare_results(results)
print(comparison_df)

# Find best algorithm
best_algo, score = AlgorithmComparison.find_best_algorithm(results)
print(f"Best algorithm: {best_algo} (score: {score:.3f})")

# Create comparison plots
AlgorithmComparison.plot_comparison(results, save_path=Path("plots"))
```

## Configuration Options

### SelectConfig Extensions

```python
config = SelectConfig(
    algorithm_type="ppo",  # 'ppo', 'sac', 'a2c', 'td3'

    # RL-specific parameters
    rl_state_dim=8,
    rl_buffer_size=10000,
    rl_batch_size=32,
    rl_train_freq=4,

    # Algorithm-specific parameters
    algorithm_params={
        'learning_rate': 3e-4,
        'n_steps': 2048,      # PPO-specific
        'batch_size': 64,     # PPO-specific
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2
    }
)
```

### Supported Algorithms

#### PPO (Proximal Policy Optimization)
```python
algorithm_params = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0
}
```

#### SAC (Soft Actor-Critic)
```python
algorithm_params = {
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'ent_coef': 'auto'
}
```

#### A2C (Advantage Actor-Critic)
```python
algorithm_params = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.0,
    'vf_coef': 0.5
}
```

#### TD3 (Twin Delayed DDPG)
```python
algorithm_params = {
    'learning_rate': 1e-3,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 100,
    'tau': 0.005,
    'gamma': 0.99,
    'policy_delay': 2
}
```

## Benchmarking and Comparison

### Running Benchmarks

```python
from farm.actions.algorithms import AlgorithmBenchmark

benchmark = AlgorithmBenchmark(
    algorithms=[
        ("ppo", {"learning_rate": 3e-4}),
        ("sac", {"learning_rate": 3e-4}),
        ("dqn", {})  # Include traditional DQN for comparison
    ],
    num_actions=6,
    state_dim=8,
    num_episodes=100,
    max_steps_per_episode=1000
)

results = benchmark.run_benchmark()
```

### Analyzing Results

```python
import pandas as pd
from farm.actions.algorithms import AlgorithmComparison

# Create comparison DataFrame
df = AlgorithmComparison.compare_results(results)
print(df)

# Statistical comparison between two algorithms
stats = AlgorithmComparison.statistical_test(
    results['ppo'],
    results['sac'],
    metric='episode_rewards'
)
print(f"P-value: {stats['p_value']:.4f}")

# Visualize results
AlgorithmComparison.plot_comparison(results, save_path=Path("plots"))
```

## Integration with Existing Systems

### Backward Compatibility

The integration maintains full backward compatibility:
- Existing DQN implementations continue to work unchanged
- Traditional ML algorithms (MLP, SVM, etc.) remain available
- Configuration system supports both old and new options

### Enhanced Action Modules

To use RL algorithms in action modules, simply update the configuration:

```python
# Before (traditional DQN)
config = AttackConfig()

# After (with RL algorithm)
config = AttackConfig(
    algorithm_type="ppo",
    algorithm_params={'learning_rate': 3e-4}
)
```

### State Representation

RL algorithms expect numerical state vectors. The existing state creation functions work seamlessly:

```python
from farm.actions.select import create_selection_state

state = create_selection_state(agent)
action = select_module.select_action(agent, actions, state)
```

## Examples and Tutorials

See `farm/actions/examples/rl_algorithm_usage.py` for comprehensive examples including:

- Basic RL algorithm usage
- Configuration examples
- Algorithm comparison
- Standalone algorithm usage

Run examples with:
```bash
cd farm/actions/examples
python rl_algorithm_usage.py
```

## Performance Considerations

### Memory Usage
- RL algorithms may use more memory than traditional approaches
- Adjust `rl_buffer_size` based on available memory
- Use `rl_batch_size` to control memory usage during training

### Training Frequency
- `rl_train_freq` controls how often training occurs
- Higher values reduce computational overhead
- Lower values may improve learning speed

### State Dimension
- Ensure `rl_state_dim` matches your actual state representation
- Use feature engineering to reduce dimensionality if needed

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **State Dimension Mismatch**: Verify `rl_state_dim` matches your state vectors
3. **Memory Issues**: Reduce buffer size or batch size
4. **Poor Performance**: Adjust algorithm hyperparameters

### Debugging

Enable logging to debug issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation

Test algorithm compatibility:
```python
from farm.actions.algorithms import AlgorithmRegistry

# Test algorithm creation
try:
    algo = AlgorithmRegistry.create("ppo", num_actions=6)
    print("PPO algorithm created successfully")
except Exception as e:
    print(f"Error: {e}")
```

## Future Extensions

The architecture supports easy addition of new algorithms:

1. **Custom Algorithms**: Implement `RLAlgorithm` interface
2. **New Libraries**: Add wrappers for other RL libraries
3. **Hybrid Approaches**: Combine multiple algorithms
4. **Advanced Features**: Add curriculum learning, meta-learning, etc.

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Issue #269: Integrate Stable Baselines](https://github.com/Dooders/AgentFarm/issues/269)
- [AgentFarm Action System](docs/action_system.md)

---

This integration successfully addresses issue #269 by providing:
- ✅ Access to state-of-the-art RL algorithms (PPO, SAC, A2C, TD3)
- ✅ Standardized implementations with proven performance
- ✅ Easy algorithm comparison and benchmarking
- ✅ Reduced maintenance burden through library integration
- ✅ Flexibility to use custom PyTorch models when needed
