# Agent Memory Hooks

## Overview

The Agent Memory Hooks system provides a non-intrusive way to add memory capabilities to agent classes and instances. These hooks integrate with the agent lifecycle, capturing and storing states and actions in the hierarchical memory system without requiring changes to the agent's implementation.

## Features

- **Seamless Integration**: Hooks into agent lifecycle methods (`__init__`, `act`, `get_state`) without modifying their behavior
- **Graceful Degradation**: Falls back to original behavior if memory system is unavailable
- **Intelligent Prioritization**: Automatically calculates importance scores for memory entries
- **Error Handling**: Robust error recovery with rate-limited logging
- **Configuration Support**: Respects memory system configuration parameters
- **Performance Tracking**: Measures execution time and state differences

## Installation Methods

### Class Decorator

Apply memory hooks to all instances of an agent class:

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import install_memory_hooks

@install_memory_hooks
class MyAgent(SimpleAgent):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Your custom initialization
```

### Instance Wrapper

Apply memory hooks to a specific agent instance:

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import with_memory

# Create agent
agent = SimpleAgent(config)

# Add memory capabilities
agent_with_memory = with_memory(agent)
```

## Configuration

Memory hooks can be configured through the agent's configuration:

```python
from memory.agent_memory.config import MemoryConfig, RedisSTMConfig

# Create memory configuration
memory_config = MemoryConfig(
    enable_memory_hooks=True,  # Enable/disable hooks globally
    stm_config=RedisSTMConfig(
        host="redis.example.com",
        port=6379,
        ttl=3600  # 1 hour TTL for short-term memory
    )
)

# Pass to agent
agent_config = {
    # Other agent config...
    "memory_config": memory_config
}

agent = MyAgent(agent_config)
```

## Usage Examples

### Basic Integration with Existing Agents

```python
from farm.agents import ExplorerAgent
from memory.agent_memory.api.hooks import install_memory_hooks
from memory.agent_memory.config import MemoryConfig

# Configure memory system
memory_config = MemoryConfig(
    cleanup_interval=50,  # Check for cleanup every 50 insertions
    memory_priority_decay=0.9  # Slightly slower priority decay
)

# Apply hooks to agent class
@install_memory_hooks
class MemoryAwareExplorer(ExplorerAgent):
    pass

# Create agent with memory config
agent = MemoryAwareExplorer({
    "agent_id": "explorer-1",
    "memory_config": memory_config
})

# Use agent normally - memory operations happen automatically
state = agent.get_state()  # State is automatically stored
action_result = agent.act()  # Action is recorded with states before/after
```

### Retrieving Memories from the Agent

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import with_memory

# Create and enhance agent
agent = with_memory(SimpleAgent({"agent_id": "agent-1"}))

# Simulate some actions
for _ in range(10):
    agent.act()

# Now retrieve similar states to current state
current_state = agent.get_state()
similar_states = agent.memory_system.retrieve_similar_states(
    agent.agent_id, 
    current_state,
    k=5
)

# Analyze action history
action_history = agent.memory_system.retrieve_by_time_range(
    agent.agent_id,
    start_step=0,
    end_step=10,
    memory_type="action"
)

# Calculate average reward
avg_reward = sum(a["contents"]["reward"] for a in action_history) / len(action_history)
```

### Disabling Memory for Specific Agents

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import install_memory_hooks
from memory.agent_memory.config import MemoryConfig

@install_memory_hooks
class MemoryAwareAgent(SimpleAgent):
    pass

# Create agent with memory hooks disabled
agent_no_memory = MemoryAwareAgent({
    "agent_id": "agent-no-mem",
    "memory_config": MemoryConfig(enable_memory_hooks=False)
})

# Memory operations will be skipped for this agent
```

### Adding Custom Memory Entries

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import with_memory

agent = with_memory(SimpleAgent({"agent_id": "agent-1"}))

# Store custom interaction data
agent.memory_system.store_agent_interaction(
    agent.agent_id,
    {
        "interaction_type": "conversation",
        "other_agent_id": "agent-2",
        "content": "Hello, how are you?",
        "sentiment": 0.8
    },
    step_number=agent.step_number,
    priority=0.7  # Custom importance score
)
```

## Advanced Usage

### Custom State Difference Calculation

```python
from farm.agents import SimpleAgent
from memory.agent_memory.api.hooks import install_memory_hooks

@install_memory_hooks
class CustomDiffAgent(SimpleAgent):
    # Override the default state difference calculation
    def _calculate_state_difference(self, state_before, state_after):
        """Custom implementation focusing on position changes"""
        if not state_before or not state_after:
            return 1.0
            
        # Calculate distance moved
        pos_before = (state_before.get("position_x", 0), state_before.get("position_y", 0))
        pos_after = (state_after.get("position_x", 0), state_after.get("position_y", 0))
        
        # Euclidean distance
        distance = ((pos_after[0] - pos_before[0])**2 + 
                   (pos_after[1] - pos_before[1])**2)**0.5
                   
        # Normalize by maximum expected movement distance
        max_distance = 10.0  # Maximum expected movement in one step
        return min(1.0, distance / max_distance)
```

### Integration with Multiple Agent Types

```python
from farm.agents import BaseAgent, ExplorerAgent, CombatAgent
from memory.agent_memory.api.hooks import install_memory_hooks

# Create a common memory-aware base class
@install_memory_hooks
class MemoryAwareAgent(BaseAgent):
    pass

# Extend different agent types with memory capabilities
class MemoryAwareExplorer(MemoryAwareAgent, ExplorerAgent):
    pass
    
class MemoryAwareCombatant(MemoryAwareAgent, CombatAgent):
    pass
```

## Performance Considerations

- **Memory Usage**: The hooks store agent states and actions, which can consume significant memory in Redis. Configure TTL and memory limits appropriately.
- **Error Rate Limiting**: Errors are logged at most once per minute to avoid log flooding.
- **Compression**: States are stored with importance scores, allowing the memory system to compress or discard less important memories.
- **Hook Avoidance**: Set `enable_memory_hooks=False` for high-performance critical agents.

## Error Handling

Memory hooks are designed to fail gracefully:

1. If the memory system initialization fails, agents will continue to function without memory capabilities
2. Errors during state/action storage are caught and logged, falling back to original agent behavior
3. Rate limiting prevents error log flooding when the memory system is unavailable

## Related Documentation

- [Core Concepts](../../../core_concepts.md): Fundamental architecture and data structures
- [AgentMemory API](../../../agent_memory_api.md): Full API specification
- [Redis Integration](../../../redis_integration.md): Redis backend details
- [Memory Agent](../../../memory_agent.md): Memory agent implementation

# Memory Hooks Documentation

## Overview

Memory hooks provide a flexible event-based system for customizing memory formation and processing. They allow the agent memory system to automatically respond to important events and modify memory storage behavior based on custom rules.

## Hook System Architecture

### Event Types

The system supports several built-in event types:

```python
EVENT_TYPES = {
    "critical_resource_change",  # Significant resource level changes
    "health_threshold",         # Health crosses important thresholds
    "novel_observation",        # New or unexpected observations
    "goal_achievement",         # Goal-related milestones
    "unexpected_outcome",       # Outcomes differing from predictions
    "agent_interaction",        # Interactions with other agents
    "environment_change"        # Significant environmental changes
}
```

### Hook Registration

```python
def register_hook(
    event_type: str,
    hook_function: callable,
    priority: int = 5
) -> bool:
    """
    Register a new hook function for a specific event type.
    
    Args:
        event_type: Type of event to monitor
        hook_function: Function to call when event occurs
        priority: Execution priority (1-10, 10 highest)
        
    Returns:
        True if registration successful
    """
```

### Hook Function Interface

Hook functions must follow this interface:

```python
def hook_function(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Union[bool, Dict[str, Any]]:
    """
    Process a memory event.
    
    Args:
        event_data: Data related to the event
        memory_agent: Reference to the memory agent
        
    Returns:
        bool: True if event is critical
        dict: Configuration for memory storage
    """
```

## Built-in Hooks

### 1. Resource Change Hook

```python
def _hook_significant_resource_change(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Dict[str, Any]:
    """Monitor significant resource level changes."""
    resource_delta = event_data.get("resource_delta", 0)
    
    return {
        "store_memory": abs(resource_delta) > RESOURCE_THRESHOLD,
        "importance": min(1.0, abs(resource_delta) / MAX_RESOURCE_DELTA),
        "memory_data": event_data
    }
```

### 2. Health Critical Hook

```python
def _hook_health_critical(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Dict[str, Any]:
    """Monitor critical health state changes."""
    health = event_data.get("health", 1.0)
    
    return {
        "store_memory": health < HEALTH_CRITICAL_THRESHOLD,
        "importance": 1.0 - health,  # Lower health = higher importance
        "memory_data": event_data
    }
```

### 3. Novelty Detection Hook

```python
def _hook_novelty_detection(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Dict[str, Any]:
    """Detect novel or unexpected observations."""
    novelty_score = _calculate_novelty(event_data, memory_agent)
    
    return {
        "store_memory": novelty_score > NOVELTY_THRESHOLD,
        "importance": novelty_score,
        "memory_data": event_data
    }
```

## Custom Hook Examples

### Goal Achievement Hook

```python
def custom_goal_hook(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Dict[str, Any]:
    """Track progress towards goals."""
    goal_progress = event_data.get("goal_progress", 0)
    goal_achieved = event_data.get("goal_achieved", False)
    
    return {
        "store_memory": goal_achieved or goal_progress > 0.8,
        "importance": goal_progress,
        "memory_data": {
            **event_data,
            "achievement_time": time.time()
        }
    }
```

### Interaction Pattern Hook

```python
def interaction_pattern_hook(
    event_data: Dict[str, Any],
    memory_agent: MemoryAgent
) -> Dict[str, Any]:
    """Monitor patterns in agent interactions."""
    interaction_type = event_data.get("interaction_type")
    interaction_result = event_data.get("result")
    
    # Check if this interaction pattern is novel
    pattern_score = _analyze_interaction_pattern(
        interaction_type,
        interaction_result,
        memory_agent
    )
    
    return {
        "store_memory": pattern_score > PATTERN_THRESHOLD,
        "importance": pattern_score,
        "memory_data": event_data
    }
```

## Hook Execution Flow

1. **Event Triggering**
```python
memory_agent.trigger_event(
    event_type="critical_resource_change",
    event_data={
        "resource_type": "energy",
        "previous_value": 100,
        "new_value": 20,
        "resource_delta": -80
    }
)
```

2. **Hook Processing**
```python
def trigger_event(
    self,
    event_type: str,
    event_data: Dict[str, Any]
) -> bool:
    """
    1. Validate event type and data
    2. Sort hooks by priority
    3. Execute hooks in priority order
    4. Process hook results
    5. Store memories if needed
    """
```

## Performance Considerations

### Hook Execution
- Hooks are executed asynchronously
- Priority system prevents bottlenecks
- Timeout mechanism for long-running hooks

### Memory Impact
```python
def _process_hook_result(
    self,
    result: Dict[str, Any],
    event_data: Dict[str, Any]
) -> None:
    """
    1. Check result validity
    2. Apply importance threshold
    3. Merge with event data
    4. Store if criteria met
    """
```

### Optimization Strategies
- Hook result caching
- Batch processing of similar events
- Periodic hook cleanup

## Error Handling

### Hook Execution Errors
```python
try:
    result = hook["function"](event_data, self)
except Exception as e:
    logger.error(f"Hook execution failed: {e}")
    return False
```

### Data Validation
```python
def _validate_hook_result(
    self,
    result: Dict[str, Any]
) -> bool:
    """
    1. Check required fields
    2. Validate data types
    3. Verify importance range
    4. Ensure memory data integrity
    """
```

## Configuration

### Hook Settings
```python
HookConfig(
    execution_timeout=1.0,  # seconds
    max_hooks_per_event=10,
    min_priority_threshold=3,
    async_execution=True
)
```

### Priority Levels
```python
PRIORITY_LEVELS = {
    "CRITICAL": 10,  # System-critical events
    "HIGH": 7,      # Important agent states
    "MEDIUM": 5,    # Regular interactions
    "LOW": 3,       # Background events
    "TRACE": 1      # Debug/monitoring
}
```

## See Also
- [Agent Memory System](agent_memory_system.md)
- [Memory Tiers](memory_tiers.md)
- [Agent Memory API](agent_memory_api.md) 