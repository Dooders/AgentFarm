# Physics Engine API Reference

This document provides detailed API reference for the physics abstraction layer.

## Overview

The physics abstraction layer provides a clean interface for different types of environments through the `IPhysicsEngine` protocol. This allows the same Environment class to work with 2D grids, static environments, continuous spaces, and custom physics implementations.

## Core Interfaces

### IPhysicsEngine

The main protocol that all physics engines must implement.

```python
from farm.core.physics.interface import IPhysicsEngine
```

#### Methods

##### `validate_position(position: Any) -> bool`

Check if a position is valid in this environment.

**Parameters:**
- `position`: Position representation (format depends on implementation)
  - Grid2D: `(x: float, y: float)` tuple
  - Static: Any hashable position identifier
  - Continuous: `numpy.ndarray`

**Returns:**
- `bool`: True if position is valid, False otherwise

**Example:**
```python
physics = Grid2DPhysics(width=100, height=100)
physics.validate_position((50, 50))  # True
physics.validate_position((-1, 50))  # False
```

##### `get_nearby_entities(position: Any, radius: float, entity_type: str = "agents") -> List[Any]`

Find entities near a position.

**Parameters:**
- `position`: Center position to search from
- `radius`: Search radius (interpretation depends on implementation)
- `entity_type`: Type of entities to search for ("agents", "resources", "objects")

**Returns:**
- `List[Any]`: List of nearby entities

**Example:**
```python
nearby_agents = physics.get_nearby_entities((50, 50), radius=10.0, entity_type="agents")
```

##### `compute_distance(pos1: Any, pos2: Any) -> float`

Compute distance between two positions.

**Parameters:**
- `pos1`: First position
- `pos2`: Second position

**Returns:**
- `float`: Distance value (metric depends on implementation)

**Example:**
```python
distance = physics.compute_distance((0, 0), (3, 4))  # 5.0 (Euclidean)
```

##### `get_state_shape() -> Tuple[int, ...]`

Get shape of state representation.

**Returns:**
- `Tuple[int, ...]`: Tuple describing state dimensions
  - Grid2D: `(width, height)`
  - Static: `(state_dim,)`
  - Continuous: `(n_dimensions,)`

**Example:**
```python
shape = physics.get_state_shape()  # (100, 100) for Grid2D
```

##### `get_observation_space(agent_id: str) -> spaces.Space`

Get observation space for an agent.

**Parameters:**
- `agent_id`: Identifier of agent (may affect observation space)

**Returns:**
- `spaces.Space`: Gymnasium space describing observations

**Example:**
```python
obs_space = physics.get_observation_space("agent_1")
# Box(low=0, high=1, shape=(channels, h, w))
```

##### `sample_position() -> Any`

Sample a random valid position.

**Returns:**
- `Any`: Random valid position in this environment

**Example:**
```python
position = physics.sample_position()  # (45.3, 67.8)
```

##### `update(dt: float = 1.0) -> None`

Update physics simulation.

**Parameters:**
- `dt`: Time step (default 1.0)

**Example:**
```python
physics.update(dt=0.1)  # Update with smaller time step
```

##### `reset() -> None`

Reset physics state.

**Example:**
```python
physics.reset()
```

##### `get_bounds() -> Tuple[Any, Any]`

Get environment bounds.

**Returns:**
- `Tuple[Any, Any]`: Tuple of (min_bounds, max_bounds)
  - Grid2D: `((0, 0), (width, height))`
  - Static: `(min_position, max_position)`
  - Continuous: `(min_bounds_array, max_bounds_array)`

**Example:**
```python
bounds = physics.get_bounds()  # ((0, 0), (100, 100))
```

##### `get_config() -> Dict[str, Any]`

Get configuration dictionary for this physics engine.

**Returns:**
- `Dict[str, Any]`: Dictionary describing physics configuration

**Example:**
```python
config = physics.get_config()
# {'type': 'grid_2d', 'width': 100, 'height': 100, ...}
```

## Implementations

### Grid2DPhysics

2D grid-based physics engine for traditional grid environments.

```python
from farm.core.physics.grid_2d import Grid2DPhysics
```

#### Constructor

```python
Grid2DPhysics(
    width: int,
    height: int,
    spatial_config: Optional[Any] = None,
    observation_config: Optional[ObservationConfig] = None,
    seed: Optional[int] = None
)
```

**Parameters:**
- `width`: Environment width in grid units
- `height`: Environment height in grid units
- `spatial_config`: Configuration for spatial indexing (optional)
- `observation_config`: Configuration for observation spaces (optional)
- `seed`: Random seed for deterministic behavior (optional)

#### Additional Methods

##### `set_entity_references(agents: List[Any], resources: List[Any]) -> None`

Set entity references for spatial indexing.

**Parameters:**
- `agents`: List of agent objects
- `resources`: List of resource objects

##### `get_nearby_agents(position: Tuple[float, float], radius: float) -> List[Any]`

Find all agents within radius of position (convenience method).

##### `get_nearby_resources(position: Tuple[float, float], radius: float) -> List[Any]`

Find all resources within radius of position (convenience method).

##### `get_nearest_resource(position: Tuple[float, float]) -> Optional[Any]`

Find nearest resource to position.

## Factory Functions

### create_physics_engine

Create a physics engine from configuration.

```python
from farm.core.physics import create_physics_engine

physics = create_physics_engine(config: SimulationConfig, seed: Optional[int] = None) -> IPhysicsEngine
```

**Parameters:**
- `config`: Simulation configuration containing physics settings
- `seed`: Random seed for deterministic behavior (optional)

**Returns:**
- `IPhysicsEngine`: Physics engine implementing the protocol

**Example:**
```python
from farm.config import SimulationConfig
from farm.core.physics import create_physics_engine

config = SimulationConfig(
    environment=EnvironmentConfig(width=100, height=100)
)
physics = create_physics_engine(config, seed=42)
```

### get_available_physics_types

Get list of available physics engine types.

```python
from farm.core.physics import get_available_physics_types

types = get_available_physics_types() -> List[str]
```

**Returns:**
- `List[str]`: List of supported physics engine type names

### validate_physics_config

Validate physics configuration.

```python
from farm.core.physics import validate_physics_config

is_valid = validate_physics_config(config: Any) -> bool
```

**Parameters:**
- `config`: Configuration object to validate

**Returns:**
- `bool`: True if configuration is valid, False otherwise

## Configuration

### PhysicsConfig

Configuration class for physics engines.

```python
from farm.config import PhysicsConfig

@dataclass
class PhysicsConfig:
    type: str = "grid_2d"  # Physics engine type
    width: Optional[int] = None  # Override environment width
    height: Optional[int] = None  # Override environment height
    spatial_config: Optional[SpatialIndexConfig] = None
    observation_config: Optional[ObservationConfig] = None
```

**Fields:**
- `type`: Physics engine type ("grid_2d", "static", "continuous")
- `width`: Override environment width if specified
- `height`: Override environment height if specified
- `spatial_config`: Configuration for spatial indexing
- `observation_config`: Configuration for observation spaces

**Example:**
```python
from farm.config import PhysicsConfig, SpatialIndexConfig

physics_config = PhysicsConfig(
    type="grid_2d",
    width=200,
    height=200,
    spatial_config=SpatialIndexConfig(
        enable_batch_updates=True,
        region_size=50.0
    )
)
```

## Usage Examples

### Basic Usage

```python
from farm.config import SimulationConfig
from farm.core.physics import create_physics_engine
from farm.core.environment import Environment

# Create configuration
config = SimulationConfig(
    environment=EnvironmentConfig(width=100, height=100)
)

# Create physics engine
physics = create_physics_engine(config, seed=42)

# Create environment
env = Environment(
    physics_engine=physics,
    resource_distribution={"type": "random", "amount": 10},
    config=config
)

# Use environment
position = (50, 50)
is_valid = env.is_valid_position(position)
nearby_agents = env.get_nearby_agents(position, 10.0)
```

### Custom Configuration

```python
from farm.config import PhysicsConfig, SpatialIndexConfig

# Create custom physics configuration
physics_config = PhysicsConfig(
    type="grid_2d",
    spatial_config=SpatialIndexConfig(
        enable_batch_updates=True,
        region_size=25.0,
        max_batch_size=50
    )
)

# Apply to environment config
config = SimulationConfig(
    environment=EnvironmentConfig(
        width=100,
        height=100,
        physics=physics_config
    )
)

# Create physics engine
physics = create_physics_engine(config)
```

### Custom Physics Engine

```python
from farm.core.physics.interface import IPhysicsEngine
from gymnasium import spaces
import numpy as np

class MyCustomPhysics(IPhysicsEngine):
    def __init__(self, custom_param: float):
        self.custom_param = custom_param
    
    def validate_position(self, position):
        # Custom validation logic
        return True
    
    def get_nearby_entities(self, position, radius, entity_type="agents"):
        # Custom spatial query logic
        return []
    
    def compute_distance(self, pos1, pos2):
        # Custom distance metric
        return 0.0
    
    def get_state_shape(self):
        return (10,)
    
    def get_observation_space(self, agent_id):
        return spaces.Box(low=0, high=1, shape=(10,))
    
    def sample_position(self):
        return np.random.random(2)
    
    def update(self, dt=1.0):
        pass
    
    def reset(self):
        pass
    
    def get_bounds(self):
        return (np.array([0, 0]), np.array([1, 1]))
    
    def get_config(self):
        return {"type": "custom", "custom_param": self.custom_param}

# Use custom physics
physics = MyCustomPhysics(custom_param=1.5)
env = Environment(physics_engine=physics, ...)
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid physics type or configuration
- `AttributeError`: Missing required methods in custom physics engine
- `TypeError`: Incorrect parameter types

### Best Practices

1. **Always validate configuration** before creating physics engines
2. **Use type hints** for better IDE support and error detection
3. **Test custom physics engines** thoroughly before production use
4. **Handle edge cases** in position validation and spatial queries
5. **Use appropriate data types** for positions (tuples for Grid2D, arrays for continuous)

## Performance Considerations

- **Protocol overhead**: Minimal, protocols are zero-cost abstractions
- **Spatial queries**: Performance depends on implementation (Grid2D uses optimized spatial indexing)
- **Memory usage**: Similar to original system, physics engines are lightweight
- **Batch updates**: Use spatial index batch updates for better performance with many entities

## Testing

### Unit Tests

```python
import unittest
from farm.core.physics.grid_2d import Grid2DPhysics

class TestGrid2DPhysics(unittest.TestCase):
    def setUp(self):
        self.physics = Grid2DPhysics(width=100, height=100)
    
    def test_validate_position(self):
        self.assertTrue(self.physics.validate_position((50, 50)))
        self.assertFalse(self.physics.validate_position((-1, -1)))
```

### Integration Tests

```python
from farm.core.environment import Environment
from farm.core.physics import create_physics_engine

def test_environment_integration():
    config = SimulationConfig(...)
    physics = create_physics_engine(config)
    env = Environment(physics_engine=physics, ...)
    
    # Test environment operations
    assert env.width == 100
    assert env.is_valid_position((50, 50))
```

## Migration from Old API

See [Migration Guide](migrations/physics_abstraction_migration.md) for detailed migration instructions.

## See Also

- [Core Architecture](core_architecture.md) - Overall system architecture
- [Environment Design](design/environment_module_design_report.md) - Detailed design rationale
- [Examples](../examples/) - Working code examples
- [Tests](../tests/) - Test implementations
