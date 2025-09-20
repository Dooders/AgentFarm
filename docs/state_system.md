## Overview
The state system provides a standardized way to represent and manage different types of states in the simulation. It uses Pydantic for validation and type safety, with a hierarchical structure of state classes.

## Base State (`BaseState`)
The foundation class for normalized state representations.

### Key Features
- Enforces value normalization (0-1 range)
- Immutable after creation
- Required tensor conversion interface
- Standard dictionary serialization
- Built-in validation

### Core Methods
```python
def to_tensor(self, device: torch.device) -> torch.Tensor:
    """Convert state to tensor format for neural networks"""

def to_dict(self) -> Dict[str, Any]:
    """Convert state to dictionary representation"""
```

## State Types

### 1. Agent State (`AgentState`)
Captures the current state of an individual agent including its position, resources, health, and other key attributes.

#### Attributes
- `agent_id`: Unique identifier for the agent
- `step_number`: Current simulation step
- `position_x`: Normalized X coordinate in environment (0-1)
- `position_y`: Normalized Y coordinate in environment (0-1)
- `position_z`: Normalized Z coordinate (usually 0 for 2D)
- `resource_level`: Current resource amount (0-1)
- `current_health`: Current health level (0-1)
- `is_defending`: Whether agent is in defensive stance
- `total_reward`: Cumulative reward received
- `age`: Number of steps agent has existed

#### Factory Method
```python
@classmethod
def from_raw_values(
    cls,
    agent_id: str,
    step_number: int,
    position_x: float,
    position_y: float,
    position_z: float,
    resource_level: float,
    current_health: float,
    is_defending: bool,
    total_reward: float,
    age: int
) -> "AgentState"
```

### 2. Environment State (`EnvironmentState`)
Captures global simulation state.

#### Attributes
- `normalized_resource_density`: Resource concentration (0-1)
- `normalized_agent_density`: Agent population density (0-1)
- `normalized_system_ratio`: System vs Independent agents (0-1)
- `normalized_resource_availability`: Resource levels (0-1)
- `normalized_time`: Simulation progress (0-1)

#### Factory Method
```python
@classmethod
def from_environment(cls, env: "Environment") -> "EnvironmentState"
```

### 3. Model State (`ModelState`)
Tracks ML model parameters and performance.

#### Attributes
- `learning_rate`: Current learning rate
- `epsilon`: Exploration rate
- `latest_loss`: Most recent training loss
- `latest_reward`: Most recent reward
- `memory_size`: Experience buffer size
- `memory_capacity`: Maximum memory size
- `steps`: Training steps taken
- `architecture`: Network structure
- `training_metrics`: Performance metrics

#### Factory Method
```python
@classmethod
def from_move_module(cls, move_module: "MoveModule") -> "ModelState"
```

### 4. Simulation State (`SimulationState`)
Captures the current state of the entire simulation including time progression, population metrics, resource metrics, and performance indicators.

#### Attributes
- `normalized_time_progress`: Current simulation progress (0-1)
- `normalized_population_size`: Current total population relative to capacity (0-1)
- `normalized_survival_rate`: Portion of original agents still alive (0-1)
- `normalized_resource_efficiency`: Resource utilization effectiveness (0-1)
- `normalized_system_performance`: System agents' performance metric (0-1)

#### Factory Method
```python
@classmethod
def from_environment(cls, environment: "Environment", num_steps: int) -> "SimulationState"
```

## Usage Examples

### Creating Agent State
```python
state = AgentState.from_raw_values(
    agent_id="agent_1",
    step_number=100,
    position_x=0.5,
    position_y=0.3,
    position_z=0.0,
    resource_level=0.7,
    current_health=0.9,
    is_defending=False,
    total_reward=10.5,
    age=50
)
```

### Creating Environment State
```python
env_state = EnvironmentState.from_environment(environment)
print(f"Resource density: {env_state.normalized_resource_density}")
print(f"Agent density: {env_state.normalized_agent_density}")
```

### Getting Model State
```python
model_state = ModelState.from_move_module(move_module)
print(f"Current epsilon: {model_state.epsilon}")
print(f"Training metrics: {model_state.training_metrics}")
```

### Creating Simulation State
```python
sim_state = SimulationState.from_environment(environment, total_steps=1000)
print(f"Time progress: {sim_state.normalized_time_progress}")
print(f"Survival rate: {sim_state.normalized_survival_rate}")
```

### Converting to Tensor
```python
tensor = state.to_tensor(device)
# Use tensor for neural network input
```

## Key Benefits

1. **Standardization**
   - Consistent state representation
   - Normalized values for stable learning
   - Standard interfaces across state types

2. **Type Safety**
   - Pydantic validation
   - Clear attribute definitions
   - Runtime type checking

3. **Flexibility**
   - Easy conversion between formats
   - Support for different state types
   - Extensible base class

4. **Monitoring**
   - Training metrics tracking
   - Performance monitoring
   - State serialization

5. **Documentation**
   - Comprehensive docstrings
   - Usage examples
   - Clear attribute descriptions

## Best Practices

1. Always use factory methods for creating states
2. Handle None/missing values appropriately
3. Validate state normalization
4. Use type hints consistently
5. Document state transformations

## Common Patterns

1. **State Creation**
```python
state = StateClass.from_raw_values(...)
```

2. **Neural Network Input**
```python
tensor = state.to_tensor(device)
```

3. **Logging/Serialization**
```python
state_dict = state.to_dict()
```

4. **Monitoring**
```python
print(f"Model State: {model_state}")
```
