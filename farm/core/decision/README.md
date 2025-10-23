
# Decision Module Documentation

## Overview

The decision module provides a flexible framework for agent action selection in the AgentFarm simulation. It supports both traditional machine learning (ML) algorithms and reinforcement learning (RL) algorithms through integration with Stable Baselines3. The module enables agents to make intelligent decisions based on their state and environment, with support for training, benchmarking, and easy algorithm switching.

Key features:
- Unified interface for ML and RL algorithms
- Experience replay and training mechanisms
- Benchmarking and comparison tools
- Flexible configuration system
- Integration with agent and environment states

## Architecture

### Core Components
- **ActionAlgorithm**: Abstract base class for all algorithms, defining select_action, train, and predict_proba methods.
- **AlgorithmRegistry**: Factory for creating algorithm instances by name.
- **RLAlgorithm**: Extension for RL-specific methods like store_experience and train_on_batch.
- **StableBaselinesWrapper**: Base wrapper for integrating Stable Baselines3 algorithms.
- **FeatureEngineer**: Utility for extracting normalized features from agent and environment states.
- **AlgorithmBenchmark**: Tool for comparing algorithm performance.
- **DecisionModule**: High-level module that uses configured algorithms for action selection.

### File Structure
```
farm/core/decision/
├── algorithms/
│   ├── base.py          # Core interface and registry
│   ├── ensemble.py      # Tree-based and ensemble ML algorithms
│   ├── mlp.py           # Neural network classifier
│   ├── rl_base.py       # RL base classes
│   ├── stable_baselines.py # Stable Baselines3 wrappers
│   ├── svm.py           # Support vector machine
│   └── benchmark.py     # Benchmarking utilities
├── config.py            # Configuration classes
├── decision.py          # Main decision module
├── feature_engineering.py # State feature extraction
├── training/            # Data collection and training utilities
└── README.md            # This documentation
```

## Available Algorithms

### Traditional ML Algorithms
- **MLP** (Multi-Layer Perceptron): Neural network classifier
- **SVM** (Support Vector Machine): Kernel-based classification
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree boosting (XGBoost/LightGBM)
- **Naive Bayes**: Probabilistic classifier
- **KNN** (K-Nearest Neighbors): Instance-based learning

### Reinforcement Learning Algorithms
- **PPO** (Proximal Policy Optimization): Policy gradient method
- **SAC** (Soft Actor-Critic): Entropy-regularized actor-critic
- **A2C** (Advantage Actor-Critic): Synchronous actor-critic
- **TD3** (Twin Delayed DDPG): Deterministic policy gradient
- **DQN** (Deep Q-Network): Value-based RL (existing implementation)

All algorithms implement probabilistic action selection for exploration.

## Usage

### Creating a Decision Module
```python
from farm.core.decision import DecisionConfig, DecisionModule
from farm.core.agent import AgentCore  # Component-based agent class

# Create config for PPO
config = DecisionConfig(
    algorithm_type="ppo",
    rl_state_dim=8,
    algorithm_params={'learning_rate': 3e-4}
)

# Create mock agent
agent = AgentCore(agent_id="test_agent")  # Simplified

# Create decision module
decision = DecisionModule(agent=agent, config=config)
```

### Selecting Actions
```python
# Create state tensor (example)
state = torch.randn(8)  # Replace with actual state creation

# Select action
action_index = decision.decide_action(state)
```

### Updating with Experience
```python
reward = 1.0
next_state = torch.randn(8)
done = False

decision.update(state, action_index, reward, next_state, done)
```

### Saving/Loading Models
```python
decision.save_model("agent_model.zip")
decision.load_model("agent_model.zip")
```

## Configuration

### Basic Configuration
```python
config = DecisionConfig(
    algorithm_type="random_forest",  # Or 'ppo', 'svm', etc.
    
    # ML parameters (for traditional algorithms)
    algorithm_params={
        'n_estimators': 100,  # Random Forest specific
        'random_state': 42
    },
    
    # RL parameters (for RL algorithms)
    rl_state_dim=8,
    rl_buffer_size=10000,
    rl_batch_size=32,
    rl_train_freq=4
)
```

See README_RL_INTEGRATION.md and README_ML_INTEGRATION.md for algorithm-specific parameters.

## Benchmarking

### Running Benchmarks
```python
from farm.core.decision.algorithms import AlgorithmBenchmark

benchmark = AlgorithmBenchmark(
    algorithms=[
        ("ppo", {"learning_rate": 3e-4}),
        ("random_forest", {"n_estimators": 50})
    ],
    num_actions=6,
    state_dim=8,
    num_episodes=100
)

results = benchmark.run_benchmark()
```

### Analyzing Results
```python
from farm.core.decision.algorithms import AlgorithmComparison

df = AlgorithmComparison.compare_results(results)
best = AlgorithmComparison.find_best_algorithm(results)
AlgorithmComparison.plot_comparison(results)
```

## Integration with AgentFarm

The decision module integrates seamlessly with AgentFarm's agent system:
- Use in AgentCore's decision loop
- Automatic state creation from agent/environment
- Supports curriculum learning
- Compatible with existing DQN implementations

For complete examples, see examples/rl_algorithm_usage.py and examples/ml_algorithm_usage.py.

## Future Extensions
- Additional RL algorithms (e.g., Rainbow DQN)
- Hybrid ML+RL approaches
- Advanced feature engineering
- Multi-agent coordination
- Transfer learning support

For issues or contributions, see CONTRIBUTING.md. 