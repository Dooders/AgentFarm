# Traditional ML Algorithms Integration

This document describes the integration of traditional machine learning algorithms with AgentFarm's action selection system, providing enhanced decision-making capabilities as requested in issue #270.

## Overview

The integration adds support for multiple traditional ML algorithms while maintaining backward compatibility with existing systems. Key features include:

- **Multiple ML Algorithms**: MLP, SVM, Random Forest, Gradient Boosting, Naive Bayes, KNN
- **Unified Interface**: Consistent API across all algorithms
- **Feature Engineering**: Comprehensive feature extraction from agent/environment state
- **Training Utilities**: Data collection and algorithm training tools
- **Flexible Configuration**: Easy switching between algorithms
- **Benchmarking**: Built-in comparison and performance evaluation

## Current Implementation Status: ✅ COMPLETE

**Issue #270 is already fully resolved!** All requested ML algorithms have been implemented and integrated into the system.

## Available Algorithms

### 1. **Multi-Layer Perceptron (MLP)**
- **Class**: `MLPActionSelector`
- **Description**: Neural network-based action selection
- **Use Case**: Complex pattern recognition in agent behavior
- **Parameters**: Hidden layers, activation functions, regularization

### 2. **Support Vector Machine (SVM)**
- **Class**: `SVMActionSelector`
- **Description**: Kernel-based classification for action selection
- **Use Case**: High-dimensional feature spaces with clear boundaries
- **Parameters**: Kernel type, regularization, gamma

### 3. **Random Forest**
- **Class**: `RandomForestActionSelector`
- **Description**: Ensemble learning with decision trees
- **Use Case**: Robust classification with feature importance
- **Parameters**: Number of trees, max depth, feature selection

### 4. **Gradient Boosting**
- **Class**: `GradientBoostActionSelector`
- **Description**: XGBoost/LightGBM implementation
- **Use Case**: High-performance classification with feature engineering
- **Parameters**: Learning rate, number of estimators, regularization

### 5. **Naive Bayes**
- **Class**: `NaiveBayesActionSelector`
- **Description**: Probabilistic classifier based on Bayes theorem
- **Use Case**: Fast baseline with interpretable probabilities
- **Parameters**: Smoothing, prior probabilities

### 6. **K-Nearest Neighbors**
- **Class**: `KNNActionSelector`
- **Description**: Instance-based learning for action selection
- **Use Case**: Non-parametric learning from similar situations
- **Parameters**: Number of neighbors, distance metric, weighting

## Architecture

### Core Components

1. **`ActionAlgorithm`**: Abstract base class defining the interface
2. **`AlgorithmRegistry`**: Registry for creating algorithm instances
3. **`SelectConfig`**: Configuration system supporting all algorithms
4. **`FeatureEngineer`**: Feature extraction from agent/environment state
5. **`ExperienceCollector`**: Data collection from agent interactions
6. **`AlgorithmTrainer`**: Training utilities for ML algorithms

### File Structure

```
farm/actions/
├── algorithms/
│   ├── __init__.py
│   ├── base.py                 # ActionAlgorithm & AlgorithmRegistry
│   ├── mlp.py                  # MLP implementation
│   ├── svm.py                  # SVM implementation
│   ├── ensemble.py             # Random Forest, Gradient Boosting, etc.
│   └── benchmark.py            # Benchmarking utilities (shared)
├── feature_engineering.py      # Feature extraction
├── training/
│   ├── collector.py            # Experience collection
│   └── trainer.py              # Algorithm training
├── config.py                   # Configuration (updated)
├── select.py                   # SelectModule (updated)
└── examples/
    └── ml_algorithm_usage.py   # Usage examples
```

## Usage

### Basic Usage with SelectModule

```python
from farm.actions.config import SelectConfig
from farm.actions.select import SelectModule

# Configure for Random Forest
config = SelectConfig(
    algorithm_type="random_forest",
    algorithm_params={
        'n_estimators': 100,
        'random_state': 42
    }
)

# Create SelectModule with Random Forest
select_module = SelectModule(
    num_actions=6,  # Number of action types
    config=config
)

# Use in action selection
action = select_module.select_action(agent, available_actions, state)
```

### Using ML Algorithms Directly

```python
from farm.actions.algorithms import RandomForestActionSelector

# Create algorithm directly
rf = RandomForestActionSelector(
    num_actions=6,
    n_estimators=100,
    random_state=42
)

# Train on data
states = np.array([...])  # Training states
actions = np.array([...]) # Training actions
rf.train(states, actions)

# Select action
action = rf.select_action(state)
probabilities = rf.predict_proba(state)
```

### Training and Data Collection

```python
from farm.actions.training import ExperienceCollector, AlgorithmTrainer
from farm.actions.algorithms import MLPActionSelector

# Collect training data
collector = ExperienceCollector()
trainer = AlgorithmTrainer()

# Collect episode data
episode_data = collector.collect_episode(agent, environment)

# Train algorithm
mlp = MLPActionSelector(num_actions=6)
trainer.train_algorithm(mlp, episode_data)
```

### Algorithm Comparison

```python
from farm.actions.algorithms import AlgorithmBenchmark, AlgorithmComparison

# Define algorithms to compare
algorithms = [
    ("mlp", {"hidden_layer_sizes": (32, 16)}),
    ("svm", {"kernel": "rbf"}),
    ("random_forest", {"n_estimators": 50}),
    ("knn", {"n_neighbors": 5}),
]

# Run benchmark
benchmark = AlgorithmBenchmark(
    algorithms=algorithms,
    num_actions=6,
    state_dim=8,
    num_episodes=100
)

results = benchmark.run_benchmark()

# Compare results
comparison_df = AlgorithmComparison.compare_results(results)
best_algo, score = AlgorithmComparison.find_best_algorithm(results)
```

## Configuration Options

### SelectConfig Extensions

```python
config = SelectConfig(
    algorithm_type="mlp",  # 'mlp', 'svm', 'random_forest', 'gradient_boost', 'naive_bayes', 'knn'

    # Algorithm-specific parameters
    algorithm_params={
        # MLP parameters
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'max_iter': 1000,

        # SVM parameters
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',

        # Random Forest parameters
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,

        # Gradient Boosting parameters
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 3,

        # KNN parameters
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
    },

    # Feature engineering flags (for future extension)
    feature_engineering=[],
    ensemble_size=1,
    use_exploration_bonus=True
)
```

## Feature Engineering

The system includes comprehensive feature engineering capabilities:

```python
from farm.actions.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer()
features = feature_engineer.extract_features(agent, environment)

# Features include:
# - Health and resource ratios (normalized)
# - Position coordinates (normalized by environment size)
# - Resource density in vicinity
# - Agent density in vicinity
# - Starvation status
# - Defensive status
# - Time progression (bounded)
```

## Dependencies

All required dependencies are already included in `requirements.txt`:

```bash
scikit-learn>=1.3.0  # Core ML algorithms
xgboost>=1.7.0       # Gradient boosting
lightgbm>=3.3.0      # Alternative gradient boosting
pandas>=1.5.0        # Data manipulation
joblib>=1.2.0        # Model serialization
```

## Integration with Existing Systems

### Backward Compatibility

- Existing DQN implementations work unchanged
- Traditional action selection methods remain available
- Configuration system supports both old and new options

### Enhanced Action Modules

All action modules can now use ML algorithms:

```python
# Before (rule-based or DQN)
attack_module = AttackModule(config=attack_config)

# After (with ML algorithm)
attack_config.algorithm_type = "random_forest"
attack_module = AttackModule(config=attack_config)
```

### Hybrid Approaches

Combine multiple algorithms:

```python
# Ensemble of different algorithms
algorithms = [
    SelectConfig(algorithm_type="random_forest"),
    SelectConfig(algorithm_type="mlp"),
    SelectConfig(algorithm_type="ppo"),  # RL algorithm
]

# Use benchmarking to find best performing algorithm
```

## Performance Considerations

### Algorithm Selection Guidelines

- **MLP**: Best for complex patterns, requires more training data
- **SVM**: Effective with clear decision boundaries, good with high dimensions
- **Random Forest**: Robust, handles missing data, provides feature importance
- **Gradient Boosting**: High performance, but slower training
- **Naive Bayes**: Fast training/inference, good baseline
- **KNN**: Simple, works well with small datasets

### Memory and Computational Requirements

- **MLP/SVM**: Moderate memory usage, can be computationally intensive
- **Random Forest**: Higher memory for large forests
- **Gradient Boosting**: Memory efficient, but slower training
- **Naive Bayes**: Minimal memory and computation
- **KNN**: High memory for large datasets

## Examples and Tutorials

See `farm/actions/examples/ml_algorithm_usage.py` for comprehensive examples:

- Basic ML algorithm usage
- Training and data collection
- Algorithm comparison
- Feature engineering demonstration
- Hybrid ML + RL approaches

Run examples with:
```bash
cd farm/actions/examples
python ml_algorithm_usage.py
```

## Testing and Validation

The implementation includes comprehensive testing:

- Unit tests for each algorithm implementation
- Integration tests with existing action modules
- Performance benchmarks comparing algorithms
- Feature engineering validation
- Training pipeline tests

## Future Extensions

The architecture supports easy addition of new algorithms:

1. **Custom Algorithms**: Implement `ActionAlgorithm` interface
2. **Deep Learning**: Integration with PyTorch/TensorFlow models
3. **Ensemble Methods**: Voting, stacking, boosting combinations
4. **Online Learning**: Algorithms that learn from streaming data
5. **Meta-Learning**: Learning to learn across different environments

## Related Issues

- **RL Integration**: Issue #269 - Stable Baselines integration
- **Benchmarking**: Enhanced benchmarking capabilities
- **Feature Engineering**: Advanced feature extraction methods

## Summary

**Issue #270 is completely resolved** with the following achievements:

✅ **All Requested ML Algorithms Implemented:**
- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Naive Bayes
- K-Nearest Neighbors

✅ **Complete Infrastructure:**
- Unified algorithm interface
- Algorithm registry and configuration
- Feature engineering system
- Training and data collection utilities
- Benchmarking and comparison tools

✅ **Full Integration:**
- Seamless integration with existing action modules
- Backward compatibility maintained
- Support for hybrid ML + RL approaches
- Comprehensive documentation and examples

The system now provides researchers and developers with a comprehensive toolkit for exploring different machine learning approaches to agent decision-making, enabling algorithm comparison and facilitating advanced research in agent behavior and learning strategies.
