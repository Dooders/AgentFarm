# Implement Continuous Physics Engine

## Overview

Implement a `ContinuousPhysics` class that implements the `IPhysicsEngine` protocol for continuous space environments, enabling robotics, navigation, and other continuous control scenarios.

## Background

The physics abstraction layer has been successfully implemented with `Grid2DPhysics` for 2D grid-based environments. The next step is to implement `ContinuousPhysics` to support continuous space environments where agents can move in continuous 2D or 3D space with floating-point precision.

## Requirements

### Core Functionality

- **Position representation**: `numpy.ndarray` with shape `(n_dimensions,)` where n_dimensions can be 2 or 3
- **Position validation**: Check if position is within continuous bounds (e.g., `[0, width] × [0, height]` for 2D)
- **Distance computation**: Euclidean distance in n-dimensional space
- **Spatial queries**: Efficient nearby entity queries using spatial data structures
- **Observation space**: Continuous observation spaces suitable for neural networks
- **Random sampling**: Sample valid positions from continuous space

### Spatial Indexing

Implement efficient spatial indexing for continuous space:

- **KD-Tree**: For efficient nearest neighbor and range queries
- **R-Tree**: For bounding box queries
- **Grid-based spatial hashing**: For approximate spatial queries
- **Octree**: For 3D continuous spaces

### Configuration

```python
@dataclass
class ContinuousPhysicsConfig:
    dimensions: int = 2  # 2D or 3D space
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((0.0, 0.0), (1.0, 1.0))
    spatial_index_type: str = "kdtree"  # "kdtree", "rtree", "grid_hash", "octree"
    grid_cell_size: Optional[float] = None  # For grid-based spatial hashing
    max_entities_per_leaf: int = 10  # For tree-based structures
    enable_dynamic_updates: bool = True
```

## Implementation Plan

### Phase 1: Basic Continuous Physics Engine

1. **Create `ContinuousPhysics` class** in `farm/core/physics/continuous.py`
2. **Implement core IPhysicsEngine methods**:
   - `validate_position()`: Check bounds for n-dimensional arrays
   - `compute_distance()`: Euclidean distance in n-dimensional space
   - `get_bounds()`: Return continuous bounds
   - `sample_position()`: Uniform random sampling in bounds
   - `get_state_shape()`: Return dimension tuple
   - `get_observation_space()`: Continuous Box space
   - `get_config()`: Return configuration

### Phase 2: Spatial Indexing

1. **Implement KD-Tree spatial index** for efficient queries
2. **Add support for dynamic updates** (add/remove entities)
3. **Implement range queries** for nearby entity searches
4. **Add nearest neighbor queries** for closest entity finding

### Phase 3: Advanced Features

1. **Support for 3D spaces** with octree indexing
2. **Collision detection** for continuous objects
3. **Path planning integration** (A*, RRT, etc.)
4. **Physics simulation** (velocity, acceleration, forces)

### Phase 4: Integration

1. **Update physics factory** to support continuous physics
2. **Add configuration support** in `PhysicsConfig`
3. **Create example environments** using continuous physics
4. **Write comprehensive tests**

## API Design

```python
class ContinuousPhysics(IPhysicsEngine):
    def __init__(
        self,
        bounds: Tuple[Tuple[float, ...], Tuple[float, ...]],
        dimensions: int = 2,
        spatial_index_type: str = "kdtree",
        config: Optional[ContinuousPhysicsConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize continuous physics engine.
        
        Args:
            bounds: ((min_x, min_y, ...), (max_x, max_y, ...)) bounds
            dimensions: Number of dimensions (2 or 3)
            spatial_index_type: Type of spatial indexing to use
            config: Additional configuration
            seed: Random seed for deterministic behavior
        """
    
    def validate_position(self, position: np.ndarray) -> bool:
        """Check if position is within bounds."""
        
    def get_nearby_entities(
        self, 
        position: np.ndarray, 
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Find entities within radius using spatial index."""
        
    def compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute Euclidean distance in n-dimensional space."""
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds as numpy arrays."""
        
    def sample_position(self) -> np.ndarray:
        """Sample random position within bounds."""
```

## Example Usage

```python
from farm.core.physics import create_physics_engine
from farm.config import SimulationConfig, EnvironmentConfig, PhysicsConfig

# Configure continuous physics
config = SimulationConfig(
    environment=EnvironmentConfig(
        physics=PhysicsConfig(
            type="continuous",
            bounds=((0.0, 0.0), (100.0, 100.0)),  # 2D space
            dimensions=2,
            spatial_index_type="kdtree"
        )
    )
)

# Create environment
physics = create_physics_engine(config)
env = Environment(physics_engine=physics, ...)

# Use continuous positions
position = np.array([50.5, 75.3])  # Continuous coordinates
is_valid = env.is_valid_position(position)
nearby_agents = env.get_nearby_agents(position, radius=10.0)
```

## Testing Requirements

### Unit Tests
- Position validation with various bounds
- Distance computation accuracy
- Spatial query correctness
- Random sampling distribution
- Configuration validation

### Integration Tests
- Agent movement in continuous space
- Resource placement and queries
- Observation generation
- Performance with many entities

### Performance Tests
- Spatial query performance with 1000+ entities
- Memory usage for large environments
- Comparison with grid-based physics

## Acceptance Criteria

- [ ] `ContinuousPhysics` implements all `IPhysicsEngine` methods
- [ ] Supports 2D and 3D continuous spaces
- [ ] Efficient spatial indexing (KD-Tree, R-Tree, or Octree)
- [ ] Comprehensive test coverage (>95%)
- [ ] Performance benchmarks show acceptable query times
- [ ] Integration with existing Environment class
- [ ] Documentation and examples
- [ ] Configuration support through `PhysicsConfig`

## Dependencies

- `numpy` for array operations
- `scipy.spatial` for KD-Tree implementation
- `rtree` for R-Tree spatial indexing (optional)
- `sklearn.neighbors` for advanced spatial queries (optional)

## Related Issues

- #XXX: Implement Static Physics Engine
- #XXX: Add collision detection to physics engines
- #XXX: Integrate path planning with continuous physics

## Labels

- `enhancement`
- `physics-engine`
- `continuous-space`
- `spatial-indexing`
- `good-first-issue` (for basic implementation)

## Priority

**Medium** - This extends the physics abstraction layer but is not required for basic functionality.

## Estimated Effort

- **Basic implementation**: 3-5 days
- **Spatial indexing**: 2-3 days  
- **Testing and integration**: 2-3 days
- **Documentation**: 1 day

**Total**: 8-12 days
