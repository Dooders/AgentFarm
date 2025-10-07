# Environment Module Design Report & Recommendations

**Date:** 2025-10-07  
**Purpose:** Design flexible environment module architecture to support multiple environment types

## Executive Summary

The current AgentFarm environment module is tightly coupled to a 2D grid-based structure, making it difficult to swap in alternative environments (e.g., static catapult, robotic arm, continuous physics simulations). This report analyzes the current architecture and provides recommendations for a flexible, extensible environment system that maintains backward compatibility while enabling easy environment swapping.

**Recommended Approach:** Abstract environment interface with pluggable implementations using the Strategy pattern and dependency injection.

---

## Current Architecture Analysis

### Current Implementation

The `Environment` class in `farm/core/environment.py` extends PettingZoo's `AECEnv` and implements a 2D grid-based simulation:

**Key characteristics:**
- **Dimensions**: Fixed `width` and `height` define a 2D grid
- **Spatial System**: KD-tree, Quadtree, and Spatial Hash indices for proximity queries
- **Position Representation**: `(x, y)` tuples with float coordinates
- **Observations**: Multi-channel 2D tensor grids (channels × height × width)
- **Resources**: Positioned in 2D space with spatial indexing
- **Agents**: Navigate 2D space with position validation

### Tight Coupling Issues

1. **Hardcoded Dimensionality**
   ```python
   # From environment.py __init__
   self.width = width
   self.height = height
   
   # Used throughout for validation
   def is_valid_position(self, position: Tuple[float, float]) -> bool:
       x, y = position
       return 0 <= x <= self.width and 0 <= y <= self.height
   ```

2. **2D-Specific Spatial Indexing**
   ```python
   # Assumes 2D coordinates
   self.spatial_index = SpatialIndex(self.width, self.height, ...)
   ```

3. **Observation System Assumptions**
   - Multi-channel observations assume 2D grid structure
   - Channels represent local 2D patches around agent
   - FOV (field of view) calculations assume Euclidean 2D space

4. **Position Validation Spread Throughout**
   - Environment: `is_valid_position()`
   - Spatial Index: `_is_valid_position()`
   - Services: `IValidationService.is_valid_position()`
   - Agents: Position validation in movement actions

### What Works Well

1. **Service-Based Architecture**: Already uses dependency injection for validation, metrics, logging
2. **Protocol-Based Interfaces**: Good use of Python protocols for loose coupling
3. **PettingZoo Integration**: Standard RL interface (observation_space, action_space, step, reset)
4. **Configurable Components**: ObservationConfig, SpatialIndexConfig allow customization

---

## Recommended Architecture

### Design Principles

1. **Interface Segregation**: Define minimal, focused interfaces
2. **Open-Closed**: Open for extension (new environments) but closed for modification
3. **Dependency Inversion**: Depend on abstractions, not concrete implementations
4. **Composition Over Inheritance**: Use composition and delegation

### Core Design Pattern: Strategy + Abstract Factory

```
┌─────────────────────────────────────────────────────────────┐
│                     Environment (PettingZoo AECEnv)         │
│  - Orchestrates simulation                                   │
│  - Delegates physics/space to IPhysicsEngine                 │
│  - Delegates observations to IObservationBuilder             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ delegates to
                              ▼
         ┌──────────────────────────────────────────┐
         │        IPhysicsEngine (Protocol)         │
         │  - validate_position()                   │
         │  - get_nearby_entities()                 │
         │  - compute_distance()                    │
         │  - get_state_shape()                     │
         │  - get_observation_space()               │
         └──────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         ┌──────────────────┐  ┌───────────────────┐
         │  Grid2DPhysics   │  │ ContinuousPhysics │
         │  - 2D grid space │  │ - Unbounded space │
         │  - Discrete cells│  │ - Continuous pos  │
         └──────────────────┘  └───────────────────┘
                                         
         ┌──────────────────┐  ┌───────────────────┐
         │ StaticPhysics    │  │  CustomPhysics    │
         │ - Fixed objects  │  │ - User-defined    │
         │ - No movement    │  │ - Domain-specific │
         └──────────────────┘  └───────────────────┘
```

### Proposed Interface

```python
# farm/core/physics/interface.py

from typing import Protocol, Tuple, List, Any, Optional, Dict
from gymnasium import spaces
import numpy as np

class IPhysicsEngine(Protocol):
    """Protocol defining physics and spatial operations for environments.
    
    This abstraction allows different environment types (2D grid, continuous,
    static, etc.) to be swapped transparently.
    """
    
    def validate_position(self, position: Any) -> bool:
        """Check if a position is valid in this environment.
        
        Args:
            position: Position representation (format depends on implementation)
            
        Returns:
            True if position is valid, False otherwise
        """
        ...
    
    def get_nearby_entities(
        self, 
        position: Any, 
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Find entities near a position.
        
        Args:
            position: Center position to search from
            radius: Search radius (interpretation depends on implementation)
            entity_type: Type of entities to search for ("agents", "resources", etc.)
            
        Returns:
            List of nearby entities
        """
        ...
    
    def compute_distance(self, pos1: Any, pos2: Any) -> float:
        """Compute distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Distance value (metric depends on implementation)
        """
        ...
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get shape of state representation.
        
        Returns:
            Tuple describing state dimensions
        """
        ...
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Get observation space for an agent.
        
        Args:
            agent_id: Identifier of agent
            
        Returns:
            Gymnasium space describing observations
        """
        ...
    
    def sample_position(self) -> Any:
        """Sample a random valid position.
        
        Returns:
            Random valid position in this environment
        """
        ...
    
    def update(self, dt: float = 1.0) -> None:
        """Update physics simulation.
        
        Args:
            dt: Time step
        """
        ...
    
    def reset(self) -> None:
        """Reset physics state."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for this physics engine."""
        ...


class IObservationBuilder(Protocol):
    """Protocol for building observations from environment state.
    
    Different environment types may require different observation formats.
    """
    
    def build_observation(
        self, 
        agent_id: str,
        physics_state: Any,
        entities: Dict[str, List[Any]]
    ) -> np.ndarray:
        """Build observation for an agent.
        
        Args:
            agent_id: Agent to build observation for
            physics_state: Current physics state
            entities: Dict of entity lists by type
            
        Returns:
            Observation array
        """
        ...
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space definition."""
        ...
```

### Implementation Examples

#### Example 1: Grid2D Physics (Current System)

```python
# farm/core/physics/grid_2d.py

from typing import Tuple, List, Any, Optional, Dict
import numpy as np
from gymnasium import spaces
from farm.core.spatial import SpatialIndex

class Grid2DPhysics:
    """2D grid-based physics engine (maintains current behavior)."""
    
    def __init__(
        self,
        width: int,
        height: int,
        spatial_index: Optional[SpatialIndex] = None,
        config: Optional[Any] = None
    ):
        self.width = width
        self.height = height
        self.config = config
        
        # Use existing spatial index infrastructure
        if spatial_index is None:
            self.spatial_index = SpatialIndex(width, height)
        else:
            self.spatial_index = spatial_index
    
    def validate_position(self, position: Tuple[float, float]) -> bool:
        """Validate 2D grid position."""
        x, y = position
        return 0 <= x <= self.width and 0 <= y <= self.height
    
    def get_nearby_entities(
        self,
        position: Tuple[float, float],
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Use spatial index for efficient queries."""
        nearby = self.spatial_index.get_nearby(position, radius, [entity_type])
        return nearby.get(entity_type, [])
    
    def compute_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Euclidean distance in 2D."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_state_shape(self) -> Tuple[int, int]:
        """Return (width, height)."""
        return (self.width, self.height)
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Return Box space for 2D multi-channel observations."""
        from farm.core.observations import ObservationConfig
        from farm.core.channels import NUM_CHANNELS
        
        obs_config = self.config.observation if self.config else ObservationConfig()
        obs_size = 2 * obs_config.R + 1
        
        return spaces.Box(
            low=0,
            high=1,
            shape=(NUM_CHANNELS, obs_size, obs_size),
            dtype=np.float32
        )
    
    def sample_position(self) -> Tuple[float, float]:
        """Sample random position in grid."""
        return (
            np.random.uniform(0, self.width),
            np.random.uniform(0, self.height)
        )
    
    def update(self, dt: float = 1.0) -> None:
        """Update spatial index if needed."""
        if hasattr(self.spatial_index, 'update'):
            self.spatial_index.update()
    
    def reset(self) -> None:
        """Reset spatial index."""
        if hasattr(self.spatial_index, 'rebuild'):
            self.spatial_index.rebuild()
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "grid_2d",
            "width": self.width,
            "height": self.height
        }
```

#### Example 2: Static Physics (For Catapult-like Scenarios)

```python
# farm/core/physics/static.py

from typing import Any, List, Tuple, Dict
import numpy as np
from gymnasium import spaces

class StaticPhysics:
    """Physics engine for static/fixed position environments.
    
    Useful for:
    - Catapult aiming problems (fixed launch point)
    - Robotic arm manipulation (fixed base)
    - Tower defense (fixed tower positions)
    - Stateless RL problems mapped to environment framework
    """
    
    def __init__(
        self,
        valid_positions: List[Any],
        state_dim: int,
        observation_space_config: Dict[str, Any],
        config: Optional[Any] = None
    ):
        """
        Args:
            valid_positions: List of valid discrete positions/states
            state_dim: Dimension of state vector
            observation_space_config: Dict with 'low', 'high', 'shape'
        """
        self.valid_positions = valid_positions
        self.state_dim = state_dim
        self.obs_config = observation_space_config
        self.config = config
        
        # Map positions to indices
        self.position_map = {pos: idx for idx, pos in enumerate(valid_positions)}
        
        # Entities are indexed by their position ID
        self.entities: Dict[str, Dict[int, List[Any]]] = {
            "agents": {},
            "resources": {},
            "objects": {}
        }
    
    def validate_position(self, position: Any) -> bool:
        """Check if position is in valid set."""
        return position in self.position_map
    
    def get_nearby_entities(
        self,
        position: Any,
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """For static environments, 'nearby' may not be spatial.
        
        This could return entities in same state cluster, similar
        observations, etc. depending on problem domain.
        """
        if not self.validate_position(position):
            return []
        
        pos_idx = self.position_map[position]
        return self.entities.get(entity_type, {}).get(pos_idx, [])
    
    def compute_distance(self, pos1: Any, pos2: Any) -> float:
        """Distance is abstract - could be state difference, discrete hops, etc."""
        if pos1 == pos2:
            return 0.0
        
        # For discrete positions, use Hamming-like distance
        idx1 = self.position_map.get(pos1, -1)
        idx2 = self.position_map.get(pos2, -1)
        
        if idx1 == -1 or idx2 == -1:
            return float('inf')
        
        return abs(idx1 - idx2)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Return state vector dimension."""
        return (self.state_dim,)
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Return configured observation space."""
        return spaces.Box(
            low=self.obs_config.get('low', -np.inf),
            high=self.obs_config.get('high', np.inf),
            shape=self.obs_config['shape'],
            dtype=np.float32
        )
    
    def sample_position(self) -> Any:
        """Sample from valid positions."""
        return self.valid_positions[
            np.random.randint(len(self.valid_positions))
        ]
    
    def update(self, dt: float = 1.0) -> None:
        """No physics update needed for static environments."""
        pass
    
    def reset(self) -> None:
        """Clear entity registry."""
        for entity_type in self.entities:
            self.entities[entity_type].clear()
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "static",
            "num_positions": len(self.valid_positions),
            "state_dim": self.state_dim
        }
    
    def register_entity(self, position: Any, entity: Any, entity_type: str = "agents"):
        """Register an entity at a position."""
        if not self.validate_position(position):
            return
        
        pos_idx = self.position_map[position]
        if pos_idx not in self.entities[entity_type]:
            self.entities[entity_type][pos_idx] = []
        
        self.entities[entity_type][pos_idx].append(entity)
    
    def unregister_entity(self, position: Any, entity: Any, entity_type: str = "agents"):
        """Remove entity from position."""
        if not self.validate_position(position):
            return
        
        pos_idx = self.position_map[position]
        if pos_idx in self.entities[entity_type]:
            if entity in self.entities[entity_type][pos_idx]:
                self.entities[entity_type][pos_idx].remove(entity)
```

#### Example 3: Continuous Physics

```python
# farm/core/physics/continuous.py

from typing import Tuple, List, Any, Optional, Dict
import numpy as np
from gymnasium import spaces

class ContinuousPhysics:
    """Continuous unbounded physics engine.
    
    Useful for:
    - Continuous control problems
    - Robotics simulations
    - Physics-based games
    - Open-world navigation
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        distance_metric: str = "euclidean",
        config: Optional[Any] = None
    ):
        """
        Args:
            state_dim: Dimensionality of continuous space
            bounds: Optional (low, high) bounds as numpy arrays
            distance_metric: 'euclidean', 'manhattan', 'cosine', etc.
        """
        self.state_dim = state_dim
        self.bounds = bounds
        self.distance_metric = distance_metric
        self.config = config
        
        # Simple list-based storage (could use KD-tree for efficiency)
        self.entities: Dict[str, List[Tuple[np.ndarray, Any]]] = {
            "agents": [],
            "resources": [],
            "objects": []
        }
    
    def validate_position(self, position: np.ndarray) -> bool:
        """Check if position is within bounds (if bounded)."""
        if self.bounds is None:
            return True
        
        low, high = self.bounds
        return np.all(position >= low) and np.all(position <= high)
    
    def get_nearby_entities(
        self,
        position: np.ndarray,
        radius: float,
        entity_type: str = "agents"
    ) -> List[Any]:
        """Brute force search for nearby entities."""
        nearby = []
        
        for pos, entity in self.entities.get(entity_type, []):
            if self.compute_distance(position, pos) <= radius:
                nearby.append(entity)
        
        return nearby
    
    def compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute distance based on configured metric."""
        if self.distance_metric == "euclidean":
            return np.linalg.norm(pos1 - pos2)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(pos1 - pos2))
        elif self.distance_metric == "cosine":
            return 1 - np.dot(pos1, pos2) / (np.linalg.norm(pos1) * np.linalg.norm(pos2))
        else:
            return np.linalg.norm(pos1 - pos2)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Return state dimension."""
        return (self.state_dim,)
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Return continuous observation space."""
        if self.bounds is not None:
            low, high = self.bounds
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            )
    
    def sample_position(self) -> np.ndarray:
        """Sample random position."""
        if self.bounds is not None:
            low, high = self.bounds
            return np.random.uniform(low, high)
        else:
            return np.random.randn(self.state_dim)
    
    def update(self, dt: float = 1.0) -> None:
        """Could implement physics integration here."""
        pass
    
    def reset(self) -> None:
        """Clear entities."""
        for entity_type in self.entities:
            self.entities[entity_type].clear()
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "state_dim": self.state_dim,
            "bounded": self.bounds is not None,
            "distance_metric": self.distance_metric
        }
```

---

## Implementation Strategy

### Phase 1: Create Abstraction Layer (Non-Breaking)

1. **Define Protocols**
   - Create `farm/core/physics/interface.py` with `IPhysicsEngine` protocol
   - Create `farm/core/physics/__init__.py` for exports

2. **Implement Grid2D Adapter**
   - Create `farm/core/physics/grid_2d.py`
   - Wrap existing 2D grid logic
   - Maintain backward compatibility

3. **Add Configuration**
   ```python
   @dataclass
   class PhysicsConfig:
       """Configuration for physics engine."""
       engine_type: str = "grid_2d"  # "grid_2d", "static", "continuous", "custom"
       engine_params: Dict[str, Any] = field(default_factory=dict)
   ```

### Phase 2: Refactor Environment (Incremental)

1. **Modify Environment.__init__**
   ```python
   def __init__(
       self,
       config: Optional[SimulationConfig] = None,
       physics_engine: Optional[IPhysicsEngine] = None,
       **kwargs  # Backward compatibility
   ):
       # Support old API
       if physics_engine is None:
           if 'width' in kwargs and 'height' in kwargs:
               # Legacy 2D grid
               physics_engine = Grid2DPhysics(
                   width=kwargs['width'],
                   height=kwargs['height'],
                   config=config
               )
           elif config and config.physics:
               # New config-based
               physics_engine = create_physics_engine(config.physics)
           else:
               # Default
               physics_engine = Grid2DPhysics(100, 100, config=config)
       
       self.physics = physics_engine
   ```

2. **Delegate Spatial Operations**
   ```python
   def is_valid_position(self, position: Any) -> bool:
       return self.physics.validate_position(position)
   
   def get_nearby_agents(self, position: Any, radius: float) -> List[Any]:
       return self.physics.get_nearby_entities(position, radius, "agents")
   ```

3. **Update Services**
   ```python
   # farm/core/services/implementations.py
   class EnvironmentValidationService(IValidationService):
       def is_valid_position(self, position: Any) -> bool:
           return self._env.physics.validate_position(position)
   ```

### Phase 3: Add New Physics Engines

1. **Create Static Physics**
   - Implement `farm/core/physics/static.py`
   - Add tests

2. **Create Continuous Physics**
   - Implement `farm/core/physics/continuous.py`
   - Add tests

3. **Add Factory**
   ```python
   # farm/core/physics/factory.py
   
   def create_physics_engine(config: PhysicsConfig) -> IPhysicsEngine:
       """Factory function for physics engines."""
       if config.engine_type == "grid_2d":
           return Grid2DPhysics(**config.engine_params)
       elif config.engine_type == "static":
           return StaticPhysics(**config.engine_params)
       elif config.engine_type == "continuous":
           return ContinuousPhysics(**config.engine_params)
       elif config.engine_type == "custom":
           # Load custom physics engine by module path
           module_path = config.engine_params.get("module")
           class_name = config.engine_params.get("class")
           # Dynamic import logic
           ...
       else:
           raise ValueError(f"Unknown physics engine: {config.engine_type}")
   ```

### Phase 4: Documentation & Examples

1. **Create Examples**
   - `examples/grid_2d_environment.py` (existing system)
   - `examples/static_catapult_environment.py` (new)
   - `examples/continuous_navigation_environment.py` (new)

2. **Update Documentation**
   - Migration guide for existing code
   - Physics engine development guide
   - Comparison of different physics engines

---

## Usage Examples

### Example 1: Using Existing 2D Grid (Backward Compatible)

```python
# Old API still works
from farm.core.environment import Environment
from farm.config import SimulationConfig

config = SimulationConfig()
env = Environment(
    width=100,
    height=100,
    resource_distribution={"amount": 20},
    config=config
)

# New API also works
from farm.core.physics import Grid2DPhysics

physics = Grid2DPhysics(width=100, height=100)
env = Environment(config=config, physics_engine=physics)
```

### Example 2: Static Catapult Environment

```python
# examples/static_catapult_environment.py

from farm.core.environment import Environment
from farm.core.physics import StaticPhysics
from farm.config import SimulationConfig, PhysicsConfig
import numpy as np

# Define the catapult problem
# State: [angle, power]
# Goal: Hit target at specific distance

valid_angles = np.linspace(0, 90, 91)  # 0-90 degrees
valid_powers = np.linspace(0, 100, 101)  # 0-100% power

# Create all combinations as "positions"
positions = [(angle, power) for angle in valid_angles for power in valid_powers]

# Configure physics
physics_config = PhysicsConfig(
    engine_type="static",
    engine_params={
        "valid_positions": positions,
        "state_dim": 2,
        "observation_space_config": {
            "low": np.array([0, 0, -100]),  # [angle, power, distance_to_target]
            "high": np.array([90, 100, 100]),
            "shape": (3,)
        }
    }
)

# Configure simulation
sim_config = SimulationConfig(
    physics=physics_config,
    max_steps=100
)

# Create environment
physics = StaticPhysics(
    valid_positions=positions,
    state_dim=2,
    observation_space_config=physics_config.engine_params["observation_space_config"]
)

env = Environment(config=sim_config, physics_engine=physics)

# Custom catapult dynamics
class CatapultEnvironment(Environment):
    def __init__(self, target_distance: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        self.target_distance = target_distance
    
    def _calculate_distance(self, angle: float, power: float) -> float:
        """Physics calculation for projectile distance."""
        angle_rad = np.radians(angle)
        velocity = power  # Simplified
        g = 9.8
        
        # Projectile motion formula
        distance = (velocity ** 2 * np.sin(2 * angle_rad)) / g
        return distance
    
    def _calculate_reward(self, agent_id: str) -> float:
        """Reward based on how close to target."""
        agent = self.get_agent(agent_id)
        angle, power = agent.position
        
        distance = self._calculate_distance(angle, power)
        error = abs(distance - self.target_distance)
        
        # Inverse error reward
        reward = 10.0 / (1.0 + error)
        
        # Bonus for hitting target exactly
        if error < 1.0:
            reward += 100.0
        
        return reward

# Use the environment
env = CatapultEnvironment(
    target_distance=50.0,
    config=sim_config,
    physics_engine=physics
)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space().sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Example 3: Continuous Navigation

```python
# examples/continuous_navigation_environment.py

from farm.core.environment import Environment
from farm.core.physics import ContinuousPhysics
from farm.config import SimulationConfig, PhysicsConfig
import numpy as np

# Configure continuous physics
bounds = (
    np.array([0.0, 0.0]),  # Lower bounds
    np.array([100.0, 100.0])  # Upper bounds
)

physics = ContinuousPhysics(
    state_dim=2,
    bounds=bounds,
    distance_metric="euclidean"
)

config = SimulationConfig(
    max_steps=1000
)

env = Environment(config=config, physics_engine=physics)

# Agents can now move continuously
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space().sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

---

## Migration Path

### For Existing Code

**No changes required!** The old API is preserved:

```python
# This still works
env = Environment(
    width=100,
    height=100,
    resource_distribution={"amount": 20},
    config=config
)
```

### For New Environments

```python
# Use new physics system
from farm.core.physics import StaticPhysics, Grid2DPhysics, ContinuousPhysics

# Pick your physics engine
physics = StaticPhysics(...)
# OR
physics = Grid2DPhysics(...)
# OR
physics = ContinuousPhysics(...)

# Create environment
env = Environment(config=config, physics_engine=physics)
```

### For Custom Environments

```python
# Implement IPhysicsEngine protocol
class MyCustomPhysics:
    def validate_position(self, position): ...
    def get_nearby_entities(self, position, radius, entity_type): ...
    def compute_distance(self, pos1, pos2): ...
    def get_state_shape(self): ...
    def get_observation_space(self, agent_id): ...
    def sample_position(self): ...
    def update(self, dt): ...
    def reset(self): ...
    def get_config(self): ...

# Use it
physics = MyCustomPhysics()
env = Environment(config=config, physics_engine=physics)
```

---

## Benefits of This Approach

### 1. **Backward Compatibility**
- Existing code continues to work without changes
- Gradual migration path
- No breaking changes to API

### 2. **Flexibility**
- Easy to swap physics engines
- Custom physics engines via protocol
- Domain-specific optimizations

### 3. **Separation of Concerns**
- Physics/spatial logic isolated
- Environment focuses on RL interface
- Clear boundaries between components

### 4. **Testability**
- Mock physics engines for testing
- Test each physics engine independently
- Unit test environment without physics complexity

### 5. **Extensibility**
- Add new physics engines without modifying core
- Custom observation builders
- Domain-specific extensions

### 6. **Performance**
- Optimize each physics engine independently
- Choose appropriate data structures per engine
- Grid2D keeps efficient spatial indexing

---

## Testing Strategy

### Unit Tests

```python
# tests/physics/test_grid_2d.py

def test_grid_2d_position_validation():
    physics = Grid2DPhysics(width=100, height=100)
    
    assert physics.validate_position((50, 50))
    assert not physics.validate_position((-1, 50))
    assert not physics.validate_position((50, 101))

def test_grid_2d_nearby_entities():
    physics = Grid2DPhysics(width=100, height=100)
    # ... register entities ...
    nearby = physics.get_nearby_entities((50, 50), radius=10)
    assert len(nearby) > 0

# tests/physics/test_static.py

def test_static_valid_positions():
    positions = [(0, 0), (1, 1), (2, 2)]
    physics = StaticPhysics(
        valid_positions=positions,
        state_dim=2,
        observation_space_config={'shape': (2,)}
    )
    
    assert physics.validate_position((1, 1))
    assert not physics.validate_position((5, 5))

# tests/physics/test_continuous.py

def test_continuous_unbounded():
    physics = ContinuousPhysics(state_dim=3, bounds=None)
    
    # Any position valid when unbounded
    assert physics.validate_position(np.array([100, 200, 300]))
```

### Integration Tests

```python
# tests/test_environment_physics_integration.py

def test_environment_with_grid_2d():
    physics = Grid2DPhysics(width=50, height=50)
    env = Environment(physics_engine=physics, config=config)
    
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert obs is not None
    assert isinstance(reward, float)

def test_environment_with_static():
    physics = StaticPhysics(...)
    env = Environment(physics_engine=physics, config=config)
    
    obs, info = env.reset()
    # ...

def test_backward_compatibility():
    # Old API should still work
    env = Environment(width=100, height=100, config=config)
    assert env.physics is not None
    assert isinstance(env.physics, Grid2DPhysics)
```

---

## Performance Considerations

### Grid2D Physics
- **Strengths**: Optimized spatial indexing (KD-tree, Quadtree), efficient for 2D
- **Use When**: Grid-based simulations, spatial queries are common

### Static Physics
- **Strengths**: Minimal overhead, fast position lookups
- **Use When**: Fixed positions, discrete state spaces, no spatial relationships

### Continuous Physics
- **Strengths**: Flexible, supports arbitrary dimensions
- **Use When**: Continuous control, robotics, physics simulations
- **Note**: May need KD-tree for efficient queries at scale

---

## Open Questions & Future Extensions

### Questions to Resolve

1. **Observation Builder Integration**: Should observations also be delegated to physics engine?
2. **Action Space**: Should physics engine define valid actions too?
3. **Multi-Agent Interactions**: How do physics engines handle agent-agent interactions?

### Future Extensions

1. **Physics Engine Registry**
   ```python
   # Register custom physics engines
   register_physics_engine("my_physics", MyPhysicsEngine)
   
   # Use by name
   env = Environment(physics_engine="my_physics", config=config)
   ```

2. **Composite Physics**
   ```python
   # Combine multiple physics engines
   physics = CompositePhysics([
       Grid2DPhysics(...),
       GravityPhysics(...),
       CollisionPhysics(...)
   ])
   ```

3. **Physics Visualization**
   ```python
   # Visualize physics state
   physics.render(mode="human")
   ```

4. **Performance Profiling**
   ```python
   # Built-in profiling
   stats = physics.get_performance_stats()
   ```

---

## Recommendation Summary

### Immediate Actions (Priority 1)

1. ✅ Create `farm/core/physics/` module
2. ✅ Define `IPhysicsEngine` protocol
3. ✅ Implement `Grid2DPhysics` wrapper (maintains current behavior)
4. ✅ Add `PhysicsConfig` to `SimulationConfig`

### Short Term (Priority 2)

5. ✅ Refactor `Environment` to use `physics_engine`
6. ✅ Update services to delegate to physics
7. ✅ Implement `StaticPhysics` for catapult example
8. ✅ Create migration guide and examples

### Medium Term (Priority 3)

9. ✅ Implement `ContinuousPhysics`
10. ✅ Add physics engine factory
11. ✅ Update all documentation
12. ✅ Comprehensive test suite

### Long Term (Priority 4)

13. ✅ Physics engine registry system
14. ✅ Composite physics engines
15. ✅ Custom observation builders per physics type
16. ✅ Performance optimization per engine type

---

## Conclusion

The recommended architecture provides:
- **Flexibility**: Easy to swap environment types
- **Simplicity**: Clean abstractions via protocols
- **Compatibility**: No breaking changes
- **Extensibility**: Custom physics engines supported
- **Performance**: Each engine can be optimized independently

This design aligns with SOLID principles, maintains backward compatibility, and enables the use cases mentioned (2D grid, static catapult, etc.) through a unified interface.

The implementation can be done incrementally with minimal risk to existing functionality.
