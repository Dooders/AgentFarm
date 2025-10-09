# Physics Engine Comparison

## Quick Reference

| Feature | Grid2DPhysics | StaticPhysics | ContinuousPhysics |
|---------|---------------|---------------|-------------------|
| **Position Type** | (x, y) float tuple | Any hashable | numpy array |
| **Space Type** | Bounded 2D grid | Discrete states | Unbounded/bounded continuous |
| **Distance Metric** | Euclidean 2D | State difference | Configurable (Euclidean, Manhattan, etc.) |
| **Spatial Queries** | KD-tree, Quadtree, Hash | Dictionary lookup | Brute force or KD-tree |
| **Use Cases** | Grid-based games, cellular automata | Fixed positions, discrete problems | Robotics, continuous control |
| **Observations** | Multi-channel 2D grids | State vectors | Feature vectors, sensor readings |
| **Performance** | Optimized (existing system) | Very fast (O(1) lookups) | Moderate (needs spatial indexing at scale) |

## Detailed Comparison

### Grid2DPhysics (Default)

**Best For:**
- Traditional grid-based simulations
- Games with tile-based movement
- Cellular automata
- Multi-agent resource gathering

**Characteristics:**
```python
Position: (x: float, y: float)
State Shape: (width, height)
Observation: Box(channels, 2*R+1, 2*R+1)
Distance: sqrt((x1-x2)^2 + (y1-y2)^2)
```

**Pros:**
- ✅ Highly optimized (existing codebase)
- ✅ Efficient spatial indexing (KD-tree, Quadtree, Hash)
- ✅ Works with all existing agents and code
- ✅ Good for large numbers of agents

**Cons:**
- ❌ Locked to 2D
- ❌ Not suitable for non-spatial problems
- ❌ Fixed grid structure

**Example:**
```python
physics = Grid2DPhysics(width=100, height=100)
env = Environment(physics_engine=physics)
```

---

### StaticPhysics

**Best For:**
- Fixed position problems (catapult, aiming)
- Discrete state spaces
- Optimization problems mapped to RL
- Problems where position doesn't change

**Characteristics:**
```python
Position: Any hashable (int, tuple, str, etc.)
State Shape: (state_dim,)
Observation: Box(n,) custom shape
Distance: Discrete hops or state difference
```

**Pros:**
- ✅ Very fast (O(1) position validation)
- ✅ Minimal memory overhead
- ✅ Flexible position representation
- ✅ Good for non-spatial problems

**Cons:**
- ❌ No built-in spatial relationships
- ❌ Requires custom distance metric
- ❌ Not suitable for movement-heavy simulations

**Example:**
```python
positions = [(angle, power) for angle in range(90) for power in range(100)]
physics = StaticPhysics(
    valid_positions=positions,
    state_dim=2,
    observation_space_config={'shape': (3,)}
)
env = Environment(physics_engine=physics)
```

---

### ContinuousPhysics

**Best For:**
- Robotics simulations
- Continuous control problems
- Physics-based environments
- Unbounded navigation

**Characteristics:**
```python
Position: numpy.ndarray(n,)
State Shape: (n_dimensions,)
Observation: Box(low, high, shape) custom
Distance: Configurable metric
```

**Pros:**
- ✅ True continuous space
- ✅ Arbitrary dimensions (not just 2D)
- ✅ Flexible distance metrics
- ✅ Good for robotics/control

**Cons:**
- ❌ Slower spatial queries (need KD-tree at scale)
- ❌ More memory for high-dimensional spaces
- ❌ Requires careful tuning

**Example:**
```python
bounds = (np.array([0, 0]), np.array([100, 100]))
physics = ContinuousPhysics(
    state_dim=2,
    bounds=bounds,
    distance_metric="euclidean"
)
env = Environment(physics_engine=physics)
```

---

## Performance Characteristics

### Spatial Query Performance

| Operation | Grid2D | Static | Continuous |
|-----------|--------|--------|------------|
| Position Validation | O(1) | O(1) | O(1) |
| Nearby Entities (KD-tree) | O(log n + k) | O(1) | O(log n + k) |
| Nearby Entities (brute force) | O(n) | O(n) | O(n) |
| Distance Calculation | O(1) | O(1) | O(d) |
| Update Spatial Index | O(n log n) | O(1) | O(n log n) |

*n = number of entities, k = results returned, d = dimensionality*

### Memory Usage

| Physics Type | Per Agent | Spatial Index | Total (1000 agents) |
|--------------|-----------|---------------|---------------------|
| Grid2D | ~200 bytes | ~50 KB | ~250 KB |
| Static | ~100 bytes | ~10 KB | ~110 KB |
| Continuous | ~150 bytes | ~100 KB | ~250 KB |

*Estimates based on typical configurations*

---

## When to Use Each

### Use Grid2DPhysics When:

- ✅ You have a grid-based world
- ✅ You need 2D spatial relationships
- ✅ You want to use existing AgentFarm features
- ✅ You need high performance with many agents
- ✅ You're doing traditional multi-agent RL

**Example Domains:**
- Resource gathering games
- Territory control
- Multi-agent navigation
- Grid-based combat
- Cellular automata

### Use StaticPhysics When:

- ✅ Positions are fixed/discrete
- ✅ Space is not important
- ✅ You have a discrete state space
- ✅ You're solving optimization problems
- ✅ You want minimal overhead

**Example Domains:**
- Catapult aiming
- Parameter optimization
- Discrete choice problems
- Tower defense (fixed tower positions)
- Turn-based strategy (fixed locations)

### Use ContinuousPhysics When:

- ✅ You need true continuous space
- ✅ You're doing robotics or control
- ✅ You need arbitrary dimensions
- ✅ You want flexible distance metrics
- ✅ Positions are not grid-aligned

**Example Domains:**
- Robot arm manipulation
- Drone navigation
- Vehicle control
- Continuous navigation
- Physics simulations

---

## Hybrid Approaches

You can also combine physics engines for complex scenarios:

### Example: Grid + Continuous

```python
# Use grid for agents, continuous for projectiles
class HybridPhysics:
    def __init__(self):
        self.grid = Grid2DPhysics(100, 100)
        self.continuous = ContinuousPhysics(state_dim=2)
    
    def validate_position(self, position):
        # Choose based on entity type
        if isinstance(position, tuple):
            return self.grid.validate_position(position)
        else:
            return self.continuous.validate_position(position)
```

### Example: Static + Grid (Tower Defense)

```python
# Towers in fixed positions, enemies move on grid
class TowerDefensePhysics:
    def __init__(self):
        self.tower_positions = StaticPhysics(...)
        self.enemy_grid = Grid2DPhysics(...)
```

---

## Migration Guide

### From Current System to Grid2DPhysics

**Step 1:** No changes needed! Backward compatible.

```python
# This still works
env = Environment(width=100, height=100, config=config)

# Equivalent new way
physics = Grid2DPhysics(width=100, height=100)
env = Environment(physics_engine=physics, config=config)
```

### From Grid2D to StaticPhysics

**What Changes:**
- Position validation
- Spatial queries
- Distance calculations
- Observations

**Example Migration:**

```python
# OLD: Grid-based
env = Environment(width=100, height=100, config=config)

# NEW: Static positions
positions = generate_discrete_positions()
physics = StaticPhysics(valid_positions=positions, ...)
env = Environment(physics_engine=physics, config=config)
```

### From Grid2D to ContinuousPhysics

**What Changes:**
- Position representation (tuple -> numpy array)
- Spatial index type
- Movement logic

**Example Migration:**

```python
# OLD: Grid-based
env = Environment(width=100, height=100, config=config)

# NEW: Continuous
bounds = (np.array([0, 0]), np.array([100, 100]))
physics = ContinuousPhysics(state_dim=2, bounds=bounds)
env = Environment(physics_engine=physics, config=config)
```

---

## Testing Different Physics

```python
def test_environment_with_different_physics():
    """Test same agent logic with different physics."""
    
    configs = [
        ("Grid2D", Grid2DPhysics(100, 100)),
        ("Static", StaticPhysics(...)),
        ("Continuous", ContinuousPhysics(...)),
    ]
    
    for name, physics in configs:
        print(f"\nTesting with {name} physics:")
        env = Environment(physics_engine=physics, config=config)
        
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space().sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        print(f"  Completed {env.time} steps")
```

---

## Conclusion

Choose your physics engine based on:
1. **Problem domain** (spatial vs. non-spatial)
2. **Performance needs** (number of agents, query frequency)
3. **Observation requirements** (grid-based vs. vector)
4. **Existing code** (backward compatibility)

Start with Grid2DPhysics (default) and switch only if you need specific features from other engines.
