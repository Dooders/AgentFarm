# AgentFarm Awareness Component System Design

## Overview

This document presents the design for an awareness component system that extends AgentFarm's existing perception architecture to support configurable environmental sensing capabilities like weather patterns, spatial objects, light, smell, and other environmental factors.

## Current Architecture Analysis

AgentFarm already has excellent foundations for this system:

1. **Component-based Agent Architecture**: Agents use pluggable components following SOLID principles
2. **Dynamic Channel System**: Extensible observation channels with different behaviors (INSTANT, DYNAMIC, PERSISTENT)
3. **PerceptionComponent**: Handles spatial awareness and observation generation
4. **World Layers**: Support for environmental data like RESOURCES, OBSTACLES, TERRAIN_COST
5. **Service-oriented Design**: Clean dependency injection for environmental services

## Awareness Component System Design

### 1. Core Architecture

The awareness system will be built as a new agent component that works alongside the existing `PerceptionComponent`:

```python
# New component structure
class AwarenessComponent(IAgentComponent):
    """
    Component handling environmental awareness and sensory perception.
    
    Responsibilities:
    - Configure and manage sensory capabilities
    - Process environmental data into awareness channels
    - Provide sensory information to decision-making
    - Handle sensory range and sensitivity settings
    """
```

### 2. Awareness Channel Types

The system will support multiple types of environmental awareness:

#### **Environmental Channels**
- **WEATHER**: Temperature, precipitation, wind patterns
- **LIGHT**: Illumination levels, day/night cycles, shadows
- **TEMPERATURE**: Heat/cold zones, thermal gradients
- **HUMIDITY**: Moisture levels affecting movement/behavior
- **AIR_QUALITY**: Pollution, toxins, breathability

#### **Spatial Awareness Channels**
- **SMELL**: Scent trails, pheromones, odors
- **SOUND**: Audio cues, vibrations, communication signals
- **PRESSURE**: Atmospheric pressure, depth (for underwater scenarios)
- **MAGNETIC**: Magnetic fields, navigation cues

#### **Biological Channels**
- **PHEROMONES**: Chemical communication between agents
- **BIOMARKERS**: Health indicators, disease spread
- **NUTRIENTS**: Soil quality, food sources, toxicity

### 3. Channel Behavior Patterns

Each awareness channel can have different temporal behaviors:

```python
class AwarenessChannelBehavior(Enum):
    INSTANT = "instant"      # Current environmental state
    DYNAMIC = "dynamic"      # Persists with decay (like scent trails)
    PERSISTENT = "persistent" # Permanent features (like landmarks)
    CYCLIC = "cyclic"        # Periodic patterns (day/night, seasons)
    EVENT_DRIVEN = "event"   # Triggered by specific events
```

### 4. Configuration System

The awareness system will be highly configurable:

```python
@dataclass(frozen=True)
class AwarenessConfig:
    """Configuration for agent awareness capabilities."""
    
    # Sensory ranges (in world units)
    weather_range: int = 10
    light_range: int = 15
    smell_range: int = 8
    sound_range: int = 12
    temperature_range: int = 6
    
    # Sensitivity settings (0.0 to 1.0)
    weather_sensitivity: float = 1.0
    light_sensitivity: float = 1.0
    smell_sensitivity: float = 1.0
    sound_sensitivity: float = 1.0
    temperature_sensitivity: float = 1.0
    
    # Enabled sensory capabilities
    enabled_senses: Set[str] = field(default_factory=lambda: {
        "weather", "light", "smell", "sound", "temperature"
    })
    
    # Channel behaviors
    channel_behaviors: Dict[str, AwarenessChannelBehavior] = field(default_factory=lambda: {
        "weather": AwarenessChannelBehavior.DYNAMIC,
        "light": AwarenessChannelBehavior.CYCLIC,
        "smell": AwarenessChannelBehavior.DYNAMIC,
        "sound": AwarenessChannelBehavior.INSTANT,
        "temperature": AwarenessChannelBehavior.DYNAMIC,
    })
    
    # Decay rates for dynamic channels
    decay_rates: Dict[str, float] = field(default_factory=lambda: {
        "weather": 0.95,
        "smell": 0.85,
        "temperature": 0.98,
    })
```

### 5. Environmental Service Integration

The awareness system will integrate with environmental services:

```python
class IEnvironmentalService(ABC):
    """Service for providing environmental data to agents."""
    
    @abstractmethod
    def get_weather_at(self, position: Tuple[float, float]) -> float:
        """Get weather intensity at position (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_light_level_at(self, position: Tuple[float, float]) -> float:
        """Get light level at position (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_smell_at(self, position: Tuple[float, float]) -> float:
        """Get smell intensity at position (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_temperature_at(self, position: Tuple[float, float]) -> float:
        """Get temperature at position (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def update_environment(self) -> None:
        """Update environmental conditions for current time step."""
        pass
```

### 6. Awareness Channel Handlers

Each sensory capability will have its own channel handler:

```python
class WeatherAwarenessHandler(ChannelHandler):
    """Handles weather awareness channel processing."""
    
    def __init__(self, environmental_service: IEnvironmentalService):
        super().__init__("WEATHER", ChannelBehavior.DYNAMIC, gamma=0.95)
        self.environmental_service = environmental_service
    
    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Process weather awareness data."""
        x, y = agent_world_pos
        
        # Get weather data from environmental service
        weather_intensity = self.environmental_service.get_weather_at((x, y))
        
        # Apply agent's weather sensitivity
        sensitivity = kwargs.get('weather_sensitivity', 1.0)
        adjusted_intensity = weather_intensity * sensitivity
        
        # Store in observation channel
        obs_size = observation.shape[-1]
        center = obs_size // 2
        observation[channel_idx, center, center] = adjusted_intensity
        
        # Add spatial variation for nearby areas
        self._add_spatial_weather_variation(observation, channel_idx, agent_world_pos, sensitivity)
```

### 7. Integration with Existing Systems

The awareness system will integrate seamlessly with your existing architecture:

#### **Agent Configuration**
```python
# Extend AgentConfig to include awareness
@dataclass(frozen=True)
class AgentConfig:
    movement: MovementConfig = field(default_factory=MovementConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    awareness: AwarenessConfig = field(default_factory=AwarenessConfig)  # NEW
```

#### **Agent Factory Integration**
```python
# AgentFactory will create awareness components
def create_default_agent(self, agent_id: str, position: Tuple[float, float], **kwargs):
    components = [
        MovementComponent(self._default_config.movement),
        ResourceComponent(initial_resources, self._default_config.resource),
        CombatComponent(self._default_config.combat),
        PerceptionComponent(spatial_service, self._default_config.perception),
        AwarenessComponent(environmental_service, self._default_config.awareness),  # NEW
    ]
    # ... rest of agent creation
```

#### **Observation Integration**
The awareness channels will be registered with the existing dynamic channel system:

```python
# Register awareness channels
def _register_awareness_channels():
    """Register awareness channels with the global registry."""
    register_channel(WeatherAwarenessHandler(env_service), 13)
    register_channel(LightAwarenessHandler(env_service), 14)
    register_channel(SmellAwarenessHandler(env_service), 15)
    register_channel(SoundAwarenessHandler(env_service), 16)
    register_channel(TemperatureAwarenessHandler(env_service), 17)
```

### 8. Environmental System Implementation

The environmental service will manage world-wide environmental conditions:

```python
class EnvironmentalSystem:
    """Manages environmental conditions across the simulation world."""
    
    def __init__(self, width: int, height: int, config: EnvironmentalConfig):
        self.width = width
        self.height = height
        self.config = config
        
        # Environmental data grids
        self.weather_map = torch.zeros((height, width))
        self.light_map = torch.zeros((height, width))
        self.smell_map = torch.zeros((height, width))
        self.temperature_map = torch.zeros((height, width))
        
        # Environmental patterns and cycles
        self.weather_patterns = WeatherPatternGenerator(config.weather)
        self.light_cycle = DayNightCycle(config.light_cycle)
        self.smell_sources = SmellSourceManager(config.smell_sources)
    
    def update_environment(self) -> None:
        """Update all environmental conditions for current time step."""
        self.weather_patterns.update(self.weather_map)
        self.light_cycle.update(self.light_map)
        self.smell_sources.update(self.smell_map)
        self._update_temperature_patterns()
    
    def get_weather_at(self, position: Tuple[float, float]) -> float:
        """Get weather intensity at specific position."""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.weather_map[y, x])
        return 0.0
```

### 9. Usage Examples

Here's how the awareness system would be used:

```python
# Create environment with environmental system
environmental_system = EnvironmentalSystem(width=100, height=100, config=env_config)
environment = Environment(
    width=100, height=100,
    environmental_system=environmental_system,  # NEW
    # ... other parameters
)

# Create agent with awareness capabilities
agent_config = AgentConfig(
    awareness=AwarenessConfig(
        enabled_senses={"weather", "light", "smell"},
        weather_range=15,
        light_range=20,
        smell_range=10,
        weather_sensitivity=0.8,
        light_sensitivity=1.2,  # Enhanced light sensitivity
    )
)

agent = factory.create_agent(
    agent_id="agent_001",
    position=(50, 50),
    config=agent_config
)

# Agent can now sense environmental conditions
awareness = agent.get_component("awareness")
weather_data = awareness.get_weather_awareness()
light_data = awareness.get_light_awareness()
smell_data = awareness.get_smell_awareness()
```

## Benefits of This Design

1. **Modularity**: Each sensory capability is independently configurable
2. **Extensibility**: Easy to add new environmental factors
3. **Performance**: Leverages existing sparse/dense channel optimization
4. **Integration**: Seamlessly works with existing perception system
5. **Flexibility**: Agents can have different sensory capabilities
6. **Realism**: Supports complex environmental interactions
7. **Configurability**: Fine-grained control over sensory parameters

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `AwarenessComponent` class
2. Implement `AwarenessConfig` configuration system
3. Create `IEnvironmentalService` interface
4. Extend `AgentConfig` to include awareness configuration

### Phase 2: Environmental System
1. Implement `EnvironmentalSystem` class
2. Create basic environmental data grids (weather, light, smell, temperature)
3. Implement environmental update mechanisms
4. Add environmental service to Environment class

### Phase 3: Channel Handlers
1. Implement awareness channel handlers (Weather, Light, Smell, etc.)
2. Register awareness channels with dynamic channel system
3. Integrate with existing observation generation pipeline

### Phase 4: Agent Integration
1. Update `AgentFactory` to create awareness components
2. Integrate awareness component with agent lifecycle
3. Add awareness data to observation generation

### Phase 5: Advanced Features
1. Implement cyclic environmental patterns (day/night, seasons)
2. Add event-driven environmental changes
3. Create environmental interaction effects
4. Add environmental visualization tools

## Conclusion

This design extends your existing architecture without breaking changes, providing a powerful and flexible awareness system that can simulate complex environmental interactions while maintaining the performance and modularity of your current system. The awareness component system will enable agents to have sophisticated environmental perception capabilities, making simulations more realistic and behaviorally rich.
