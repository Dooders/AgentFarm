# Implement Observer Pattern for Events System

## 🎯 Summary

Implement a comprehensive event system using the observer pattern to enable decoupled, event-driven communication throughout the AgentFarm simulation framework. This will replace the current basic event logging with a robust publish-subscribe system that supports both synchronous and asynchronous event handling.

## 📋 Background & Motivation

Currently, the AgentFarm codebase has:
- Basic event logging in the Environment class (agent births/deaths, etc.)
- No formal observer pattern implementation
- Limited ability for components to communicate through events
- Architectural recommendations document already outlines event system design

The proposed event system will:
- Enable loose coupling between simulation components
- Support real-time analysis and monitoring
- Allow plugins to hook into simulation events
- Provide a foundation for event-driven agent behaviors
- Improve debugging and introspection capabilities

## 🔍 Detailed Analysis

### Current Event Handling
- **Environment class**: Basic logging of significant events (`record_birth`, `record_death`, etc.)
- **Database logging**: Events are stored but not broadcast to interested components
- **No observer pattern**: Components cannot subscribe to events or react in real-time
- **Limited extensibility**: Hard to add new event types or handlers

### Existing Architecture Context
- Component-based design with clear separation of concerns
- Dependency injection framework being planned
- Plugin system with hooks being designed
- Strong typing with Pydantic/dataclasses throughout

## 🏗️ Proposed Design

### Core Components

#### 1. Event System Infrastructure
```python
# farm/core/events/__init__.py
from .event_system import EventSystem, EventPriority
from .event_types import EventTypes, SimulationEvents
from .event_data import EventData, AgentEventData, ResourceEventData
```

#### 2. Event System Class
```python
class EventSystem:
    """Central event manager supporting both sync/async handlers."""

    def __init__(self):
        self._listeners: Dict[str, List[EventHandler]] = defaultdict(list)
        self._async_listeners: Dict[str, List[AsyncEventHandler]] = defaultdict(list)
        self._event_history: Deque[EventData] = deque(maxlen=1000)

    def subscribe(self, event_type: str, handler: EventHandler,
                  priority: EventPriority = EventPriority.NORMAL) -> None:
        """Subscribe to an event type with optional priority."""

    def subscribe_async(self, event_type: str, handler: AsyncEventHandler,
                       priority: EventPriority = EventPriority.NORMAL) -> None:
        """Subscribe to an event type with async handler."""

    def publish(self, event_type: str, event_data: EventData) -> None:
        """Publish an event synchronously."""

    async def publish_async(self, event_type: str, event_data: EventData) -> None:
        """Publish an event asynchronously."""

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type."""
```

#### 3. Event Data Structures
```python
# farm/core/events/event_data.py
from pydantic import BaseModel
from typing import Any, Dict, Optional
from datetime import datetime

class EventData(BaseModel):
    """Base event data structure."""
    event_type: str
    timestamp: datetime = datetime.now()
    simulation_id: str
    step_number: int
    source_component: str
    metadata: Dict[str, Any] = {}

class AgentEventData(EventData):
    """Event data for agent-related events."""
    agent_id: str
    agent_type: str
    position: Tuple[int, int]
    health: float
    resources: float

class ResourceEventData(EventData):
    """Event data for resource-related events."""
    resource_id: str
    position: Tuple[int, int]
    amount: float
    resource_type: str
```

#### 4. Event Types Enum
```python
# farm/core/events/event_types.py
from enum import Enum

class SimulationEvents(Enum):
    # Agent Lifecycle Events
    AGENT_CREATED = "agent.created"
    AGENT_DESTROYED = "agent.destroyed"
    AGENT_MOVED = "agent.moved"
    AGENT_HEALTH_CHANGED = "agent.health_changed"

    # Action Events
    ACTION_EXECUTED = "action.executed"
    ACTION_FAILED = "action.failed"
    COMBAT_OCCURRED = "combat.occurred"
    RESOURCE_GATHERED = "resource.gathered"

    # Environmental Events
    RESOURCE_SPAWNED = "resource.spawned"
    RESOURCE_CONSUMED = "resource.consumed"
    ENVIRONMENT_CHANGED = "environment.changed"

    # Simulation Events
    SIMULATION_STARTED = "simulation.started"
    SIMULATION_STEP_COMPLETED = "simulation.step_completed"
    SIMULATION_FINISHED = "simulation.finished"

    # Analysis Events
    METRICS_UPDATED = "metrics.updated"
    ANALYSIS_COMPLETED = "analysis.completed"
```

### Integration Points

#### 1. Environment Class Integration
```python
# farm/core/environment.py
class Environment(AECEnv):
    def __init__(self, config: EnvironmentConfig):
        # ... existing initialization ...
        self.event_system = EventSystem()
        self._setup_event_handlers()

    def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        # Pre-step events
        self.event_system.publish(
            SimulationEvents.SIMULATION_STEP_STARTED.value,
            EventData(
                event_type=SimulationEvents.SIMULATION_STEP_STARTED.value,
                simulation_id=self.simulation_id,
                step_number=self.time,
                source_component="environment"
            )
        )

        # ... existing step logic ...

        # Post-step events
        self.event_system.publish(
            SimulationEvents.SIMULATION_STEP_COMPLETED.value,
            EventData(
                event_type=SimulationEvents.SIMULATION_STEP_COMPLETED.value,
                simulation_id=self.simulation_id,
                step_number=self.time,
                source_component="environment"
            )
        )
```

#### 2. Agent System Integration
```python
# farm/agents/base_agent.py
class BaseAgent:
    def __init__(self, agent_id: str, environment: Environment):
        self.agent_id = agent_id
        self.environment = environment
        self.event_system = environment.event_system

    def move(self, new_position: Tuple[int, int]) -> bool:
        old_position = self.position
        success = self._execute_move(new_position)

        if success:
            self.event_system.publish(
                SimulationEvents.AGENT_MOVED.value,
                AgentEventData(
                    event_type=SimulationEvents.AGENT_MOVED.value,
                    simulation_id=self.environment.simulation_id,
                    step_number=self.environment.time,
                    source_component=f"agent.{self.agent_id}",
                    agent_id=self.agent_id,
                    agent_type=self.__class__.__name__,
                    position=new_position,
                    health=self.health,
                    resources=self.resources,
                    metadata={"old_position": old_position}
                )
            )
        return success
```

#### 3. Plugin System Integration
```python
# farm/core/plugins/plugin_interface.py
class PluginInterface(ABC):
    @abstractmethod
    def get_event_handlers(self) -> Dict[str, EventHandler]:
        """Return event handlers provided by this plugin."""
        pass

    @abstractmethod
    def get_event_publishers(self) -> List[str]:
        """Return event types this plugin may publish."""
        pass
```

### Error Handling & Resilience

#### 1. Event Handler Wrapper
```python
# farm/core/events/error_handling.py
class EventHandlerWrapper:
    """Wrapper that provides error handling for event handlers."""

    def __init__(self, handler: EventHandler, logger: logging.Logger):
        self.handler = handler
        self.logger = logger

    def __call__(self, event_data: EventData) -> None:
        try:
            self.handler(event_data)
        except Exception as e:
            self.logger.error(
                f"Event handler failed for event {event_data.event_type}: {e}",
                exc_info=True
            )
            # Continue processing other handlers
```

#### 2. Event System Error Handling
```python
# farm/core/events/event_system.py
def publish(self, event_type: str, event_data: EventData) -> None:
    """Publish an event with error handling."""
    if event_type not in self._listeners:
        return

    failed_handlers = []
    for handler_wrapper in self._listeners[event_type]:
        try:
            handler_wrapper(event_data)
        except Exception as e:
            failed_handlers.append((handler_wrapper, e))
            # Log error but continue with other handlers

    if failed_handlers:
        self._logger.warning(
            f"{len(failed_handlers)} event handlers failed for {event_type}"
        )
```

## 🚀 Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create event system module structure**
   - `farm/core/events/__init__.py`
   - `farm/core/events/event_system.py`
   - `farm/core/events/event_data.py`
   - `farm/core/events/event_types.py`

2. **Implement base EventSystem class**
   - Synchronous event publishing
   - Handler subscription/unsubscription
   - Basic error handling

3. **Define core event data structures**
   - Base EventData class
   - AgentEventData, ResourceEventData
   - EventTypes enum

### Phase 2: Integration & Event Types (Week 3-4)
1. **Integrate with Environment class**
   - Add event publishing to simulation loop
   - Publish step start/end events
   - Add agent lifecycle events

2. **Add agent-related events**
   - Movement events
   - Health change events
   - Action execution events

3. **Implement resource events**
   - Resource spawn/consumption events
   - Transfer events

### Phase 3: Advanced Features (Week 5-6)
1. **Add asynchronous event support**
   - Async event handlers
   - Event batching for performance

2. **Implement event filtering and prioritization**
   - Event priority system
   - Conditional event handlers
   - Event filtering by criteria

3. **Add monitoring and debugging**
   - Event logging and metrics
   - Event replay capabilities
   - Performance monitoring

### Phase 4: Plugin Integration & Testing (Week 7-8)
1. **Plugin system integration**
   - Plugin event handler registration
   - Plugin event publishing permissions

2. **Comprehensive testing**
   - Unit tests for event system
   - Integration tests with Environment
   - Performance benchmarking

3. **Documentation and examples**
   - API documentation
   - Usage examples
   - Plugin development guide

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_event_system.py
def test_event_subscription():
    """Test basic event subscription and publishing."""
    event_system = EventSystem()
    received_events = []

    def test_handler(event_data):
        received_events.append(event_data)

    event_system.subscribe("test.event", test_handler)

    test_event = EventData(
        event_type="test.event",
        simulation_id="test",
        step_number=1,
        source_component="test"
    )

    event_system.publish("test.event", test_event)

    assert len(received_events) == 1
    assert received_events[0] == test_event

def test_error_handling():
    """Test that event handler errors don't crash the system."""
    event_system = EventSystem()

    def failing_handler(event_data):
        raise ValueError("Handler failed")

    def working_handler(event_data):
        pass

    event_system.subscribe("test.event", failing_handler)
    event_system.subscribe("test.event", working_handler)

    test_event = EventData(
        event_type="test.event",
        simulation_id="test",
        step_number=1,
        source_component="test"
    )

    # Should not raise exception despite failing handler
    event_system.publish("test.event", test_event)
```

### Integration Tests
```python
# tests/test_environment_events.py
def test_environment_step_events():
    """Test that environment publishes step events."""
    config = EnvironmentConfig(width=10, height=10)
    env = Environment(config)
    event_system = env.event_system

    received_events = []

    def capture_events(event_data):
        received_events.append(event_data)

    event_system.subscribe(SimulationEvents.SIMULATION_STEP_COMPLETED.value, capture_events)

    # Run a simulation step
    actions = {}
    env.step(actions)

    # Check that step completion event was published
    assert len(received_events) == 1
    assert received_events[0].event_type == SimulationEvents.SIMULATION_STEP_COMPLETED.value
```

## 📊 Success Metrics

### Functional Metrics
- [ ] All major simulation events are published
- [ ] Event handlers can be registered from plugins
- [ ] Events are delivered reliably without data loss
- [ ] Event system handles high-frequency events efficiently

### Performance Metrics
- [ ] Event publishing adds < 5% overhead to simulation step
- [ ] Memory usage remains stable during long simulations
- [ ] Event handlers complete within reasonable time bounds
- [ ] System handles 1000+ concurrent event handlers

### Quality Metrics
- [ ] 95%+ test coverage for event system
- [ ] All event data structures are properly typed
- [ ] Comprehensive error handling prevents crashes
- [ ] Clear documentation and examples provided

## 🎁 Benefits

### For Developers
- **Decoupled Architecture**: Components can communicate without tight dependencies
- **Extensibility**: Easy to add new event types and handlers
- **Debugging**: Rich event logging for troubleshooting
- **Plugin Ecosystem**: Foundation for powerful plugin capabilities

### For Researchers
- **Real-time Analysis**: Subscribe to simulation events for live analysis
- **Custom Metrics**: Build analysis tools that react to simulation events
- **Experiment Monitoring**: Track specific events during experiments
- **Behavioral Insights**: Understand agent behaviors through event patterns

### For System Performance
- **Efficient Communication**: Event-driven architecture reduces polling
- **Scalability**: Better resource utilization through event-driven design
- **Monitoring**: Built-in observability for system health
- **Optimization**: Identify performance bottlenecks through event analysis

## 🔗 Related Issues

- [ ] Refactor Environment class modularity (#xxx)
- [ ] Implement plugin system foundation (#xxx)
- [ ] Add comprehensive logging system (#xxx)
- [ ] Performance optimization initiative (#xxx)

## 📝 Checklist

### Pre-Implementation
- [ ] Review and approve design with team
- [ ] Identify all required event types
- [ ] Plan integration points with existing code
- [ ] Set up development environment

### Implementation
- [ ] Create event system module structure
- [ ] Implement core EventSystem class
- [ ] Define event data structures
- [ ] Add error handling and logging
- [ ] Integrate with Environment class
- [ ] Add agent-related events
- [ ] Implement resource events
- [ ] Add async event support
- [ ] Implement plugin integration
- [ ] Add monitoring and debugging

### Testing & Validation
- [ ] Write comprehensive unit tests
- [ ] Create integration tests
- [ ] Performance benchmarking
- [ ] Error handling validation
- [ ] Documentation review

### Deployment & Documentation
- [ ] Update API documentation
- [ ] Create usage examples
- [ ] Update architectural documentation
- [ ] Create migration guide for existing code

## 🏷️ Labels

`enhancement`, `architecture`, `events`, `observer-pattern`, `plugin-system`, `high-priority`

## 📋 Assignee

@architectural-lead

## 📅 Estimated Timeline

8 weeks total:
- Phase 1 (Core Infrastructure): Weeks 1-2
- Phase 2 (Integration & Event Types): Weeks 3-4
- Phase 3 (Advanced Features): Weeks 5-6
- Phase 4 (Plugin Integration & Testing): Weeks 7-8

## 💡 Additional Context

This event system will serve as the communication backbone for the entire AgentFarm ecosystem, enabling:

1. **Real-time Analysis**: Components can react to simulation events as they happen
2. **Plugin Ecosystem**: Third-party plugins can integrate deeply with the simulation
3. **Debugging Tools**: Rich event streams for understanding system behavior
4. **Research Capabilities**: Fine-grained event data for behavioral analysis
5. **System Monitoring**: Health monitoring and performance insights through events

The design follows the existing architectural patterns in the codebase and integrates seamlessly with the planned dependency injection and plugin systems.
