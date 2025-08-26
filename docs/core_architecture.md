# Core Architecture Deep Dive

## Overview

This document provides an in-depth exploration of AgentFarm's core architecture, focusing on the fundamental components that make up the simulation framework. Understanding these components is essential for extending the system and developing custom agents, environments, and analysis tools.

## Core Module Structure

### `farm.core` Package

The core package contains the fundamental building blocks of the simulation system:

```
farm.core/
├── __init__.py
├── action.py              # Action enumeration and definitions
├── analysis.py            # Core analysis functionality
├── channels.py            # Dynamic channel system
├── cli.py                 # Command-line interface utilities
├── collector.py           # Data collection mechanisms
├── config.py              # Configuration management
├── environment.py         # Main simulation environment
├── experiment_tracker.py  # Experiment tracking and logging
├── genome.py              # Genetic algorithm components
├── metrics_tracker.py     # Performance metrics collection
├── observations.py        # Observation system
├── perception.py          # Agent perception mechanisms
├── resource_manager.py    # Resource distribution and management
├── resources.py           # Resource definitions and types
├── senses.py              # Sensory input processing
├── simulation.py          # Simulation orchestration
├── spatial_index.py       # Spatial indexing for performance
├── state.py               # Environment state management
└── visualization.py       # Core visualization components
```

## Key Architectural Patterns

### 1. Component-Based Design

AgentFarm uses a component-based architecture where complex systems are built from smaller, focused components:

- **Modular Components**: Each system (observations, channels, actions) is self-contained
- **Dependency Injection**: Components are wired together through configuration
- **Interface Contracts**: Clear interfaces allow component substitution
- **Extension Points**: Well-defined hooks for customization

### 2. Observer Pattern for Events

The simulation uses an event-driven architecture:

```python
# Events are propagated through the system
environment.fire_event("agent_moved", agent, old_position, new_position)
environment.fire_event("resource_consumed", agent, resource_position, amount)
environment.fire_event("combat_occurred", attacker, defender, damage)
```

### 3. Factory Pattern for Agent Creation

Agents are created through a factory system allowing runtime configuration:

```python
# Agent factory in environment.py
def create_agent(self, agent_type: str, **kwargs) -> BaseAgent:
    """Create agent of specified type with configuration."""
    if agent_type == "system":
        return SystemAgent(**kwargs)
    elif agent_type == "independent":
        return IndependentAgent(**kwargs)
    elif agent_type == "control":
        return ControlAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
```

## Deep Dive: Environment System

### Environment Class Architecture

The `Environment` class extends PettingZoo's `AECEnv` and serves as the central coordinator:

```python
class Environment(AECEnv):
    def __init__(self, width: int, height: int, resource_distribution: str, **kwargs):
        super().__init__()

        # Core components
        self._grid = np.zeros((height, width))  # Spatial world representation
        self._agents = {}                       # Agent registry
        self._resources = ResourceManager()     # Resource system
        self._spatial_index = SpatialIndex()    # Performance optimization
        self._metrics_tracker = MetricsTracker() # Data collection
        self._observation_config = kwargs.get('obs_config', ObservationConfig())

        # Simulation state
        self._current_step = 0
        self._terminated_agents = set()
        self._action_space = self._create_action_space()
        self._observation_space = self._create_observation_space()
```

### Simulation Loop

The core simulation loop follows a structured pattern:

```python
def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one simulation step."""

    # 1. Pre-step preparations
    self._clear_instant_channels()
    self._update_resource_regeneration()

    # 2. Agent action execution
    for agent_id, action in actions.items():
        agent = self._agents[agent_id]
        self._execute_agent_action(agent, action)

    # 3. Environmental updates
    self._update_agent_states()
    self._handle_agent_interactions()
    self._process_combat_resolution()

    # 4. Observation updates
    self._update_agent_observations()

    # 5. Post-step processing
    self._collect_metrics()
    self._update_database_logs()

    # 6. Cleanup and preparation for next step
    self._cleanup_terminated_agents()
    self._apply_channel_decay()

    return self._get_step_results()
```

### Spatial Management

The environment uses a spatial indexing system for efficient queries:

```python
class SpatialIndex:
    def __init__(self, grid_width: int, grid_height: int):
        self._buckets = defaultdict(list)  # Spatial partitioning
        self._agent_positions = {}         # Agent location tracking

    def update_position(self, agent_id: str, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """Update agent's position in spatial index."""
        # Remove from old bucket
        if old_pos in self._buckets:
            self._buckets[old_pos] = [aid for aid in self._buckets[old_pos] if aid != agent_id]

        # Add to new bucket
        self._buckets[new_pos].append(agent_id)
        self._agent_positions[agent_id] = new_pos

    def get_nearby_agents(self, position: Tuple[int, int], radius: int) -> List[str]:
        """Find agents within radius using spatial partitioning."""
        nearby = []
        x, y = position

        # Check surrounding buckets within radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self._buckets:
                    nearby.extend(self._buckets[check_pos])

        return nearby
```

## Deep Dive: Observation System

### Local Observation Design

The observation system is built around the concept of **local perception**:

```python
class AgentObservation:
    def __init__(self, config: ObservationConfig):
        self.config = config
        self.radius = config.R
        self.fov_radius = config.fov_radius

        # Observation buffer: (channels, height, width)
        obs_size = 2 * self.radius + 1
        self._observation = torch.zeros(NUM_CHANNELS, obs_size, obs_size)

        # Field-of-view mask for visibility
        self._fov_mask = self._generate_fov_mask()
```

### Channel Processing Pipeline

Observations are built through a multi-stage pipeline:

```python
def update_observation(self, agent_position: Tuple[int, int], environment: Environment):
    """Update agent's observation from current environment state."""

    # 1. Clear previous instant observations
    self._clear_instant_channels()

    # 2. Extract local view from world state
    world_view = environment.get_world_view()
    local_view = crop_local(world_view, agent_position, self.config.R)

    # 3. Process each channel
    for channel_name, channel_handler in self._channel_handlers.items():
        channel_idx = get_channel_registry().get_index(channel_name)
        channel_handler.process(
            observation=self._observation,
            channel_idx=channel_idx,
            config=self.config,
            agent_world_pos=agent_position,
            environment=environment,
            agent=self._agent
        )

    # 4. Apply field-of-view masking
    self._apply_fov_mask()

    # 5. Apply temporal decay to dynamic channels
    self._apply_decay()
```

### Channel Handler Interface

All channel processors implement a common interface:

```python
class ChannelHandler(ABC):
    def __init__(self, name: str, behavior: ChannelBehavior, gamma: float = 1.0):
        self.name = name
        self.behavior = behavior
        self.gamma = gamma  # Decay factor for dynamic channels

    @abstractmethod
    def process(self, observation: torch.Tensor, channel_idx: int,
                config: ObservationConfig, agent_world_pos: Tuple[int, int],
                **kwargs) -> None:
        """Process and update channel observation."""
        pass

    def clear(self, observation: torch.Tensor, channel_idx: int) -> None:
        """Clear channel if it's INSTANT behavior."""
        if self.behavior == ChannelBehavior.INSTANT:
            observation[channel_idx].zero_()

    def decay(self, observation: torch.Tensor, channel_idx: int,
              config: ObservationConfig) -> None:
        """Apply temporal decay to dynamic channels."""
        if self.behavior == ChannelBehavior.DYNAMIC:
            observation[channel_idx].mul_(self.gamma)
```

## Deep Dive: Dynamic Channel System

### Channel Registry Architecture

The channel system uses a global registry for dynamic channel management:

```python
class ChannelRegistry:
    def __init__(self):
        self._handlers = {}                    # name -> handler
        self._name_to_index = {}              # name -> index
        self._index_to_name = {}              # index -> name
        self._next_index = 0                  # Next available index

    def register(self, handler: ChannelHandler, index: Optional[int] = None) -> int:
        """Register a channel handler."""
        # Validation and conflict checking
        if handler.name in self._handlers:
            raise ValueError(f"Channel '{handler.name}' already registered")

        # Assign index
        if index is None:
            index = self._next_index
            self._next_index += 1
        else:
            # Custom index assignment with validation
            if index in self._index_to_name:
                raise ValueError(f"Channel index {index} already assigned")
            self._next_index = max(self._next_index, index + 1)

        # Register mappings
        self._handlers[handler.name] = handler
        self._name_to_index[handler.name] = index
        self._index_to_name[index] = handler.name

        return index
```

### Batch Operations

The registry provides efficient batch operations for performance:

```python
def apply_decay(self, observation: torch.Tensor, config: ObservationConfig) -> None:
    """Apply decay to all DYNAMIC channels in batch."""
    for name, handler in self._handlers.items():
        channel_idx = self._name_to_index[name]
        handler.decay(observation, channel_idx, config)

def clear_instant(self, observation: torch.Tensor) -> None:
    """Clear all INSTANT channels in batch."""
    for name, handler in self._handlers.items():
        channel_idx = self._name_to_index[name]
        handler.clear(observation, channel_idx)
```

### Memory Management

The system includes sophisticated memory management for dynamic channels:

```python
def optimize_memory_layout(self):
    """Optimize channel ordering for memory access patterns."""
    # Sort channels by access frequency
    access_counts = self._get_channel_access_counts()

    # Reorder channels for better cache performance
    reordered_channels = sorted(
        self._handlers.items(),
        key=lambda x: access_counts.get(x[0], 0),
        reverse=True
    )

    # Update mappings with optimized ordering
    self._rebuild_mappings(reordered_channels)
```

## State Management

### Environment State

The environment maintains comprehensive state information:

```python
class EnvironmentState:
    def __init__(self):
        self.current_step = 0
        self.agent_states = {}          # Agent positions, health, etc.
        self.resource_states = {}       # Resource locations and amounts
        self.channel_states = {}        # Dynamic channel values
        self.metrics_history = []       # Historical metrics
        self.event_log = []            # Event history
        self.random_state = None       # For reproducible simulations
```

### State Persistence

State can be serialized for analysis and checkpointing:

```python
def save_state(self, filepath: str):
    """Save complete environment state to file."""
    state_dict = {
        'step': self._current_step,
        'agents': {aid: agent.get_state() for aid, agent in self._agents.items()},
        'resources': self._resources.get_state(),
        'channels': self._get_channel_states(),
        'metrics': self._metrics_tracker.get_history(),
        'random_state': random.getstate()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(state_dict, f)

def load_state(self, filepath: str):
    """Load environment state from file."""
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)

    # Restore all components
    self._current_step = state_dict['step']
    # ... restore agents, resources, etc.
```

## Performance Optimizations

### 1. Spatial Indexing

The spatial index uses bucketing for O(1) neighbor queries:

```python
def get_neighbors(self, position: Tuple[int, int], radius: int) -> List[str]:
    """Get agents within radius using spatial partitioning."""
    x, y = position
    neighbors = []

    # Only check relevant buckets
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            bucket_pos = (x + dx, y + dy)
            if bucket_pos in self._buckets:
                # Check distance for agents in bucket
                for agent_id in self._buckets[bucket_pos]:
                    agent_pos = self._agent_positions[agent_id]
                    if self._distance(position, agent_pos) <= radius:
                        neighbors.append(agent_id)

    return neighbors
```

### 2. Batch Processing

Observation updates are batched for efficiency:

```python
def update_all_observations(self):
    """Update observations for all agents in batch."""
    # Pre-compute world state snapshot
    world_snapshot = self._get_world_snapshot()

    # Update agents in parallel if possible
    for agent in self._agents.values():
        if not agent.is_terminated:
            agent.observation.update_from_snapshot(world_snapshot)
```

### 3. Memory Pooling

Tensor operations use memory pooling to reduce allocations:

```python
class TensorPool:
    def __init__(self):
        self._pool = {}  # size -> available tensors

    def get_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Get tensor from pool or create new one."""
        size_key = tuple(size)
        if size_key in self._pool and self._pool[size_key]:
            return self._pool[size_key].pop()

        return torch.zeros(size)

    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        size_key = tuple(tensor.shape)
        if size_key not in self._pool:
            self._pool[size_key] = []
        self._pool[size_key].append(tensor.zero_())  # Clear and store
```

## Integration Points

### Reinforcement Learning Integration

The system provides clean interfaces for RL algorithms:

```python
def get_observation(self, agent_id: str) -> torch.Tensor:
    """Get observation for RL agent."""
    agent = self._agents[agent_id]
    return agent.observation.get_tensor()

def apply_action(self, agent_id: str, action: Any) -> float:
    """Apply action and return reward."""
    agent = self._agents[agent_id]
    reward = self._execute_agent_action(agent, action)

    # Update observation for next step
    agent.observation.update_observation(agent.position, self)

    return reward
```

### Database Integration

Metrics and events are automatically logged:

```python
def log_step_metrics(self):
    """Log comprehensive metrics for current step."""
    metrics = {
        'step': self._current_step,
        'agent_count': len(self._agents),
        'resource_count': self._resources.total_count(),
        'avg_agent_health': self._calculate_avg_health(),
        'combat_events': len(self._combat_events),
        'birth_events': len(self._birth_events),
        'death_events': len(self._death_events)
    }

    self._database.log_metrics(metrics)
    self._metrics_tracker.record(metrics)
```

## Extension Mechanisms

### 1. Custom Channel Creation

New observation channels can be added without modifying core code:

```python
class WeatherChannel(ChannelHandler):
    def __init__(self):
        super().__init__("WEATHER", ChannelBehavior.DYNAMIC, gamma=0.95)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        # Implement weather simulation logic
        weather_intensity = self._simulate_weather(agent_world_pos)
        observation[channel_idx].fill_(weather_intensity)
```

### 2. Environment Hooks

The environment provides hooks for customization:

```python
def pre_step_hook(self):
    """Called before each simulation step."""
    pass  # Override in subclass

def post_step_hook(self):
    """Called after each simulation step."""
    pass  # Override in subclass

def on_agent_death(self, agent):
    """Called when an agent dies."""
    pass  # Override in subclass
```

### 3. Custom Metrics

New metrics can be easily added:

```python
def register_custom_metric(self, name: str, calculator: Callable):
    """Register a custom metric calculator."""
    self._custom_metrics[name] = calculator

def calculate_custom_metrics(self):
    """Calculate all custom metrics."""
    results = {}
    for name, calculator in self._custom_metrics.items():
        results[name] = calculator(self)
    return results
```

## Testing and Validation

### Unit Testing Architecture

The system includes comprehensive testing:

```python
def test_channel_registration():
    """Test dynamic channel registration."""
    registry = ChannelRegistry()

    # Test basic registration
    handler = MockChannelHandler("TEST")
    idx = registry.register(handler)

    assert registry.get_index("TEST") == idx
    assert registry.get_name(idx) == "TEST"
    assert registry.get_handler("TEST") is handler

def test_observation_pipeline():
    """Test complete observation processing pipeline."""
    config = ObservationConfig(R=3, fov_radius=2)
    observation = AgentObservation(config)

    # Test channel processing
    world_state = create_mock_world_state()
    observation.update_observation((5, 5), world_state)

    # Verify all channels updated
    assert observation.get_tensor().shape == (NUM_CHANNELS, 7, 7)
```

This architecture provides a solid foundation for building complex multi-agent simulations while maintaining flexibility, performance, and extensibility.
