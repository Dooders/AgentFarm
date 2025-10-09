# Implement Static Physics Engine

## Overview

Implement a `StaticPhysics` class that implements the `IPhysicsEngine` protocol for static/fixed position environments, enabling parameter optimization, decision-making scenarios, and other non-spatial simulations.

## Background

The physics abstraction layer has been successfully implemented with `Grid2DPhysics` for 2D grid-based environments. The next step is to implement `StaticPhysics` to support environments where spatial movement is not relevant, such as parameter optimization, decision trees, or abstract problem-solving scenarios.

## Requirements

### Core Functionality

- **Position representation**: Any hashable identifier (string, integer, tuple, etc.)
- **Position validation**: Check if position is in predefined valid set
- **Distance computation**: Custom distance metric (e.g., parameter space distance)
- **Spatial queries**: Find entities by position identifiers
- **Observation space**: Abstract observation spaces for non-spatial data
- **Random sampling**: Sample from valid position set

### Use Cases

1. **Parameter Optimization**: Agents optimize parameters in a fixed parameter space
2. **Decision Making**: Agents make decisions from a fixed set of options
3. **Abstract Problem Solving**: Agents work on non-spatial problems
4. **Resource Allocation**: Agents allocate resources from fixed pools
5. **Strategy Games**: Turn-based games with fixed positions

### Configuration

```python
@dataclass
class StaticPhysicsConfig:
    valid_positions: List[Any]  # List of valid position identifiers
    distance_metric: str = "euclidean"  # "euclidean", "manhattan", "custom"
    custom_distance_func: Optional[Callable] = None
    observation_type: str = "parameter"  # "parameter", "decision", "abstract"
    enable_position_sampling: bool = True
    max_entities_per_position: int = 1  # Allow multiple entities per position
```

## Implementation Plan

### Phase 1: Basic Static Physics Engine

1. **Create `StaticPhysics` class** in `farm/core/physics/static.py`
2. **Implement core IPhysicsEngine methods**:
   - `validate_position()`: Check if position is in valid set
   - `compute_distance()`: Custom distance metric between positions
   - `get_bounds()`: Return min/max positions from valid set
   - `sample_position()`: Random sampling from valid positions
   - `get_state_shape()`: Return size of valid position set
   - `get_observation_space()`: Abstract observation space
   - `get_config()`: Return configuration

### Phase 2: Entity Management

1. **Implement entity storage** using position-based dictionaries
2. **Add support for multiple entities per position**
3. **Implement efficient entity queries** by position
4. **Add entity counting and statistics**

### Phase 3: Advanced Features

1. **Custom distance metrics** for different problem types
2. **Position clustering** for similar positions
3. **Dynamic position sets** (add/remove valid positions)
4. **Position weighting** for biased sampling

### Phase 4: Integration

1. **Update physics factory** to support static physics
2. **Add configuration support** in `PhysicsConfig`
3. **Create example environments** using static physics
4. **Write comprehensive tests**

## API Design

```python
class StaticPhysics(IPhysicsEngine):
    def __init__(
        self,
        valid_positions: List[Any],
        distance_metric: str = "euclidean",
        custom_distance_func: Optional[Callable] = None,
        config: Optional[StaticPhysicsConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize static physics engine.
        
        Args:
            valid_positions: List of valid position identifiers
            distance_metric: Distance metric to use ("euclidean", "manhattan", "custom")
            custom_distance_func: Custom distance function if metric is "custom"
            config: Additional configuration
            seed: Random seed for deterministic behavior
        """
    
    def validate_position(self, position: Any) -> bool:
        """Check if position is in valid set."""
        
    def get_nearby_entities(
        self, 
        position: Any, 
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Find entities within distance radius."""
        
    def compute_distance(self, pos1: Any, pos2: Any) -> float:
        """Compute distance between positions using configured metric."""
        
    def get_bounds(self) -> Tuple[Any, Any]:
        """Return min and max positions from valid set."""
        
    def sample_position(self) -> Any:
        """Sample random position from valid set."""
        
    def add_valid_position(self, position: Any) -> None:
        """Add new valid position."""
        
    def remove_valid_position(self, position: Any) -> None:
        """Remove position from valid set."""
```

## Example Usage

### Parameter Optimization

```python
from farm.core.physics import create_physics_engine
from farm.config import SimulationConfig, EnvironmentConfig, PhysicsConfig

# Define parameter space
parameter_positions = [
    (0.1, 0.2, 0.3),  # (learning_rate, momentum, dropout)
    (0.2, 0.3, 0.4),
    (0.3, 0.4, 0.5),
    # ... more parameter combinations
]

# Configure static physics
config = SimulationConfig(
    environment=EnvironmentConfig(
        physics=PhysicsConfig(
            type="static",
            valid_positions=parameter_positions,
            distance_metric="euclidean"
        )
    )
)

# Create environment
physics = create_physics_engine(config)
env = Environment(physics_engine=physics, ...)

# Use parameter positions
position = (0.15, 0.25, 0.35)
is_valid = env.is_valid_position(position)
nearby_params = env.get_nearby_entities(position, radius=0.1)
```

### Decision Making

```python
# Define decision options
decision_positions = [
    "invest_in_stocks",
    "invest_in_bonds", 
    "invest_in_real_estate",
    "save_cash",
    "pay_debt"
]

config = SimulationConfig(
    environment=EnvironmentConfig(
        physics=PhysicsConfig(
            type="static",
            valid_positions=decision_positions,
            distance_metric="custom",
            custom_distance_func=lambda a, b: 1.0 if a != b else 0.0
        )
    )
)
```

## Testing Requirements

### Unit Tests
- Position validation with various valid sets
- Distance computation with different metrics
- Entity storage and retrieval
- Random sampling distribution
- Configuration validation

### Integration Tests
- Agent decision-making scenarios
- Parameter optimization workflows
- Resource allocation problems
- Performance with many positions

### Example Scenarios
- **Hyperparameter optimization**: Agents optimize neural network parameters
- **Portfolio management**: Agents choose investment strategies
- **Resource allocation**: Agents allocate limited resources
- **Game theory**: Agents make strategic decisions

## Acceptance Criteria

- [ ] `StaticPhysics` implements all `IPhysicsEngine` methods
- [ ] Supports arbitrary position identifiers
- [ ] Configurable distance metrics
- [ ] Efficient entity storage and queries
- [ ] Comprehensive test coverage (>95%)
- [ ] Integration with existing Environment class
- [ ] Documentation and examples
- [ ] Configuration support through `PhysicsConfig`

## Dependencies

- `numpy` for numerical operations (optional)
- `scipy.spatial.distance` for distance metrics (optional)
- Standard library for basic functionality

## Related Issues

- #XXX: Implement Continuous Physics Engine
- #XXX: Add custom distance metrics to physics engines
- #XXX: Create example environments for static physics

## Labels

- `enhancement`
- `physics-engine`
- `static-space`
- `parameter-optimization`
- `good-first-issue`

## Priority

**Medium** - This extends the physics abstraction layer for non-spatial scenarios.

## Estimated Effort

- **Basic implementation**: 2-3 days
- **Entity management**: 1-2 days
- **Testing and integration**: 2-3 days
- **Documentation**: 1 day

**Total**: 6-9 days

## Example Implementations

### Hyperparameter Optimization Environment

```python
class HyperparameterOptimizationEnv(Environment):
    def __init__(self, param_space, objective_func):
        # Define parameter positions
        positions = generate_parameter_grid(param_space)
        
        # Create static physics
        physics = StaticPhysics(
            valid_positions=positions,
            distance_metric="euclidean"
        )
        
        super().__init__(physics_engine=physics, ...)
        self.objective_func = objective_func
    
    def evaluate_position(self, position):
        """Evaluate objective function at parameter position."""
        return self.objective_func(position)
```

### Decision Tree Environment

```python
class DecisionTreeEnv(Environment):
    def __init__(self, decision_nodes):
        # Create static physics with decision nodes
        physics = StaticPhysics(
            valid_positions=decision_nodes,
            distance_metric="custom",
            custom_distance_func=tree_distance
        )
        
        super().__init__(physics_engine=physics, ...)
```

This implementation will enable a wide range of non-spatial simulation scenarios while maintaining the same clean interface as the spatial physics engines.
