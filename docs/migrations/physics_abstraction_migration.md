# Physics Abstraction Layer Migration Guide

This guide helps you migrate your code from the old Environment API to the new physics abstraction layer.

## Overview

The physics abstraction layer introduces a clean separation between environment orchestration and spatial/physics operations. This makes the system more flexible and allows for different types of environments (2D grid, static, continuous, etc.).

## Breaking Changes

### Environment Constructor

**Old API:**
```python
env = Environment(
    width=100,
    height=100,
    resource_distribution=config.resources,
    config=config
)
```

**New API:**
```python
from farm.core.physics import create_physics_engine

physics = create_physics_engine(config, seed=config.seed)
env = Environment(
    physics_engine=physics,
    resource_distribution=config.resources,
    config=config
)
```

### Direct Spatial Index Access

**Old API:**
```python
# Direct access to spatial index
env.spatial_index.get_nearby_agents(position, radius)
env.spatial_index.mark_positions_dirty()
```

**New API:**
```python
# Access through physics engine
env.physics.get_nearby_entities(position, radius, "agents")
env.physics.mark_positions_dirty()
```

## Migration Steps

### Step 1: Update Environment Creation

Replace direct Environment instantiation with physics engine creation:

```python
# Before
env = Environment(width=100, height=100, ...)

# After
from farm.core.physics import create_physics_engine
physics = create_physics_engine(config, seed=config.seed)
env = Environment(physics_engine=physics, ...)
```

### Step 2: Update Spatial Operations

Replace direct spatial index calls with physics engine calls:

```python
# Before
nearby_agents = env.spatial_index.get_nearby_agents(position, radius)
env.spatial_index.mark_positions_dirty()

# After
nearby_agents = env.physics.get_nearby_entities(position, radius, "agents")
env.physics.mark_positions_dirty()
```

### Step 3: Update Configuration (Optional)

If you want to customize physics engine behavior, update your configuration:

```python
# Before
config = SimulationConfig(
    environment=EnvironmentConfig(
        width=100,
        height=100,
        spatial_index=SpatialIndexConfig(...)
    )
)

# After
config = SimulationConfig(
    environment=EnvironmentConfig(
        width=100,
        height=100,
        physics=PhysicsConfig(
            type="grid_2d",
            width=100,  # Optional override
            height=100,  # Optional override
            spatial_config=SpatialIndexConfig(...)
        )
    )
)
```

## Backward Compatibility

The new system maintains backward compatibility for most operations:

- `env.width` and `env.height` properties still work
- `env.is_valid_position()` still works
- `env.get_nearby_agents()` and `env.get_nearby_resources()` still work
- `env.observation_space()` still works

## Common Migration Patterns

### Simulation Runners

**Before:**
```python
def run_simulation(config):
    env = Environment(
        width=config.environment.width,
        height=config.environment.height,
        resource_distribution=config.resources,
        config=config
    )
    # ... rest of simulation
```

**After:**
```python
def run_simulation(config):
    from farm.core.physics import create_physics_engine
    
    physics = create_physics_engine(config, seed=config.seed)
    env = Environment(
        physics_engine=physics,
        resource_distribution=config.resources,
        config=config
    )
    # ... rest of simulation
```

### Custom Physics Engines

If you need custom physics behavior, implement the `IPhysicsEngine` protocol:

```python
from farm.core.physics.interface import IPhysicsEngine

class MyCustomPhysics(IPhysicsEngine):
    def validate_position(self, position):
        # Custom position validation
        pass
    
    def get_nearby_entities(self, position, radius, entity_type):
        # Custom spatial queries
        pass
    
    # ... implement all required methods

# Use custom physics
physics = MyCustomPhysics()
env = Environment(physics_engine=physics, ...)
```

### Testing

**Before:**
```python
def test_environment():
    env = Environment(width=100, height=100, ...)
    # ... tests
```

**After:**
```python
def test_environment():
    from farm.core.physics import create_physics_engine
    
    config = SimulationConfig(...)
    physics = create_physics_engine(config)
    env = Environment(physics_engine=physics, ...)
    # ... tests
```

## Performance Considerations

The physics abstraction layer adds minimal overhead:

- Protocol calls are zero-cost abstractions
- Spatial operations maintain the same performance characteristics
- Memory usage is similar to the original system

## Troubleshooting

### Common Issues

1. **ImportError: cannot import name 'create_physics_engine'**
   - Make sure you're using the latest version of the codebase
   - Check that the physics module is properly installed

2. **TypeError: Environment() missing 1 required positional argument: 'physics_engine'**
   - Update your Environment instantiation to use the new API
   - Use `create_physics_engine()` to create the physics engine

3. **AttributeError: 'Environment' object has no attribute 'spatial_index'**
   - Replace `env.spatial_index` calls with `env.physics` calls
   - Use the physics engine methods instead of direct spatial index access

### Getting Help

If you encounter issues during migration:

1. Check the [API Reference](api/physics.md) for detailed method documentation
2. Look at the [examples](examples/) for working code samples
3. Review the [test files](tests/) for usage patterns
4. Open an issue on the project repository

## Benefits of Migration

After migration, you'll have:

- **Flexibility**: Easy to swap between different physics engines
- **Testability**: Mock physics engines for unit tests
- **Extensibility**: Add custom physics engines without modifying core code
- **Maintainability**: Clear separation of concerns
- **Future-proofing**: Ready for new environment types (continuous, static, etc.)

## Examples

See the following files for complete examples:

- `examples/static_catapult_example.py` - Static physics example
- `tests/integration/test_physics_environment.py` - Integration tests
- `tests/validation/test_grid2d_equivalence.py` - Validation tests

## Timeline

- **Phase 1**: Update Environment creation (required)
- **Phase 2**: Update spatial operations (if using direct access)
- **Phase 3**: Update configuration (optional)
- **Phase 4**: Add custom physics engines (optional)

Most code can be migrated in a few hours. The new API is designed to be intuitive and similar to the old API.
