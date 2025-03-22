# Agent Memory API Documentation

## Overview

The Agent Memory API provides a clean, standardized interface for interacting with the agent memory system. It abstracts away the implementation details of the underlying storage mechanisms, allowing developers to focus on using the memory system rather than understanding its internals.

The API enables storing and retrieving agent states, actions, and interactions across the hierarchical memory architecture (Short-Term Memory, Intermediate Memory, and Long-Term Memory). It provides methods for both exact and similarity-based retrieval, as well as management operations for the memory system.

## Key Features

- **Hierarchical Memory Access**: Unified access to all memory tiers (STM, IM, LTM)
- **Vector Similarity Search**: Find semantically similar memories using embedding vectors
- **Attribute-Based Retrieval**: Find memories based on specific attribute values
- **Temporal Queries**: Retrieve memories based on simulation steps or time ranges
- **Memory Management**: Configure and maintain the memory system
- **State Change Tracking**: Track attribute changes over time
- **Memory Statistics**: Get insights about memory usage and distribution

## Installation

The Agent Memory API is available as part of the agent state memory system. Import it directly:

```python
from memory.agent_memory.api import AgentMemoryAPI
```

## Basic Usage

### Initialization

```python
from memory.agent_memory.api import AgentMemoryAPI
from memory.agent_memory.config import MemoryConfig

# Initialize with default configuration
api = AgentMemoryAPI()

# Or, initialize with custom configuration
config = MemoryConfig(
    cleanup_interval=100,
    stm_config={"memory_limit": 10000}
)
api = AgentMemoryAPI(config)
```

### Storing Agent Information

```python
# Store an agent's state
api.store_agent_state(
    agent_id="agent-123",
    state_data={
        "position": [10, 20],
        "health": 0.85,
        "inventory": {"wood": 5, "stone": 2}
    },
    step_number=1234,
    priority=0.75  # Important memory (0.0-1.0)
)

# Store an interaction
api.store_agent_interaction(
    agent_id="agent-123",
    interaction_data={
        "interaction_type": "conversation",
        "other_agent_id": "agent-456",
        "content": "Hello, do you have any wood to trade?",
        "sentiment": 0.6
    },
    step_number=1235
)

# Store an action
api.store_agent_action(
    agent_id="agent-123",
    action_data={
        "action_type": "trade",
        "target_agent": "agent-456",
        "items_given": {"wood": 2},
        "items_received": {"stone": 1},
        "outcome": "success"
    },
    step_number=1236
)
```

### Retrieving Memories

```python
# Get specific memory by ID
memory = api.retrieve_state_by_id(
    agent_id="agent-123",
    memory_id="agent-123-1234-1679233344"
)

# Get recent states
recent_states = api.retrieve_recent_states(
    agent_id="agent-123",
    count=5,
    memory_type="state"  # Optional filter
)

# Find similar states
current_state = {"position": [12, 22], "health": 0.8}
similar_states = api.retrieve_similar_states(
    agent_id="agent-123",
    query_state=current_state,
    k=5
)

# Get memories within a time range
memories = api.retrieve_by_time_range(
    agent_id="agent-123",
    start_step=1000,
    end_step=2000,
    memory_type="action"  # Optional filter
)

# Find memories with specific attributes
trading_memories = api.retrieve_by_attributes(
    agent_id="agent-123",
    attributes={"action_type": "trade", "outcome": "success"},
    memory_type="action"
)
```

### Memory Management

```python
# Force memory maintenance (tier transitions)
api.force_memory_maintenance(agent_id="agent-123")

# Clear agent memory
api.clear_agent_memory(
    agent_id="agent-123",
    memory_tiers=["stm", "im"]  # Optional: specific tiers to clear
)

# Update importance score for a memory
api.set_importance_score(
    agent_id="agent-123",
    memory_id="agent-123-1234-1679233344",
    importance_score=0.9  # New importance (0.0-1.0)
)

# Update memory system configuration
api.configure_memory_system({
    "cleanup_interval": 150,
    "stm_config": {
        "memory_limit": 15000,
        "ttl": 7200  # 2 hours
    }
})
```

### Analytics and Insights

```python
# Get memory statistics
stats = api.get_memory_statistics(agent_id="agent-123")
print(f"Total memories: {stats['total_memories']}")
print(f"Memory distribution: {stats['memory_type_distribution']}")

# Get agent state snapshots at specific steps
snapshots = api.get_memory_snapshots(
    agent_id="agent-123",
    steps=[1000, 2000, 3000]
)

# Track attribute changes over time
health_history = api.get_attribute_change_history(
    agent_id="agent-123",
    attribute_name="health",
    start_step=1000,
    end_step=2000
)
```

## Advanced Usage

### Raw Vector Search

```python
# Get embedding from a state using your embedding logic
embedding = [0.1, 0.2, 0.3, ...]  # Your embedding vector

# Search using the raw embedding
similar_memories = api.search_by_embedding(
    agent_id="agent-123",
    query_embedding=embedding,
    k=10,
    memory_tiers=["stm", "im"]  # Optional: specific tiers to search
)
```

### Content-Based Search

```python
# Search by text content
conversation_memories = api.search_by_content(
    agent_id="agent-123",
    content_query="trade wood",
    k=5
)

# Or search by attribute patterns
attribute_matches = api.search_by_content(
    agent_id="agent-123",
    content_query={"health": {"$lt": 0.5}},  # Find low health states
    k=5
)
```

## API Reference

### Core Storage Methods

| Method | Description |
|--------|-------------|
| `store_agent_state(agent_id, state_data, step_number, priority=1.0)` | Store an agent's state in memory |
| `store_agent_interaction(agent_id, interaction_data, step_number, priority=1.0)` | Store information about an interaction |
| `store_agent_action(agent_id, action_data, step_number, priority=1.0)` | Store information about an action |

### Retrieval Methods

| Method | Description |
|--------|-------------|
| `retrieve_state_by_id(agent_id, memory_id)` | Retrieve a specific memory by ID |
| `retrieve_recent_states(agent_id, count=10, memory_type=None)` | Get most recent agent states |
| `retrieve_similar_states(agent_id, query_state, k=5, memory_type=None)` | Find states similar to the query state |
| `retrieve_by_time_range(agent_id, start_step, end_step, memory_type=None)` | Get memories in a step range |
| `retrieve_by_attributes(agent_id, attributes, memory_type=None)` | Find memories matching attributes |

### Advanced Retrieval Methods

| Method | Description |
|--------|-------------|
| `search_by_embedding(agent_id, query_embedding, k=5, memory_tiers=None)` | Search using a raw embedding vector |
| `search_by_content(agent_id, content_query, k=5)` | Search by content pattern |
| `get_memory_snapshots(agent_id, steps)` | Get agent states at specific steps |
| `get_attribute_change_history(agent_id, attribute_name, start_step=None, end_step=None)` | Track attribute changes over time |

### Memory Management Methods

| Method | Description |
|--------|-------------|
| `force_memory_maintenance(agent_id=None)` | Force tier transitions and cleanup |
| `clear_agent_memory(agent_id, memory_tiers=None)` | Clear agent memory |
| `set_importance_score(agent_id, memory_id, importance_score)` | Update memory importance |
| `configure_memory_system(config)` | Update system configuration |
| `get_memory_statistics(agent_id)` | Get memory usage statistics |

## Memory Structure

Each memory entry stored by the API has the following standardized structure:

```json
{
  "memory_id": "agent-123-1234-1679233344",
  "agent_id": "agent-123",
  "step_number": 1234,
  "timestamp": 1679233344,
  
  "contents": {
    // The actual state/action/interaction data
    "position": [10, 20],
    "health": 0.85,
    // ...
  },
  
  "metadata": {
    "creation_time": 1679233344,
    "last_access_time": 1679233400,
    "compression_level": 0,
    "importance_score": 0.75,
    "retrieval_count": 3,
    "memory_type": "state" // "interaction", "action", etc.
  },
  
  "embeddings": {
    "full_vector": [...],  // STM embedding
    "compressed_vector": [...],  // IM embedding
    "abstract_vector": [...]  // LTM embedding
  }
}
```

## Hierarchical Memory Architecture

The API transparently interacts with the three-tier memory architecture:

1. **Short-Term Memory (STM)**: Recent, detailed memories stored in Redis
2. **Intermediate Memory (IM)**: Medium-term memories with some compression in Redis
3. **Long-Term Memory (LTM)**: Historical, highly compressed memories in SQLite

Each tier has different characteristics:

| Tier | Storage | Resolution | Typical Retention | Access Speed |
|------|---------|------------|-------------------|--------------|
| STM  | Redis   | Full       | ~1000 steps       | Very Fast    |
| IM   | Redis   | Medium     | ~10,000 steps     | Fast         |
| LTM  | SQLite  | Low        | Entire history    | Medium       |

The API automatically queries the appropriate tiers based on the requested information and combines results when needed.

## Error Handling

The API includes robust error handling to ensure operation continuity even when components fail:

- Redis connection errors are caught and logged
- Invalid parameter types or values are validated
- Error information includes context for debugging

Error recovery behaviors include:

- Falling back to available tiers when one tier is unavailable
- Returning partial results when complete results can't be obtained
- Providing meaningful error information in logs

## Performance Considerations

When using the Agent Memory API, consider these performance factors:

- **Storage Volume**: Each stored state increases memory usage, especially in Redis
- **Embedding Generation**: Vector similarity searches require embedding computation
- **Cross-Tier Queries**: Queries spanning multiple tiers incur additional overhead
- **Redis Capacity**: Monitor Redis memory usage, especially with many agents
- **Tier Maintenance**: Regular maintenance (automatic or via `force_memory_maintenance`) helps manage memory growth

## Configuration Options

The API can be configured using the `MemoryConfig` object:

```python
from memory.agent_memory.config import MemoryConfig, RedisSTMConfig

config = MemoryConfig(
    # System-wide settings
    cleanup_interval=100,  # Check for maintenance every N insertions
    memory_priority_decay=0.95,  # Priority decay factor for older memories
    
    # Tier-specific settings
    stm_config=RedisSTMConfig(
        host="localhost",
        port=6379,
        memory_limit=10000,  # Max entries in STM
        ttl=3600  # 1 hour TTL
    )
)

api = AgentMemoryAPI(config)
```

## Integration with Agent Systems

The API is designed to integrate seamlessly with agent systems:

- Can be used directly in agent implementations
- Works well with hook-based memory integration (see [Memory Hooks](memory_hooks.md))
- Can be used as a standalone system for memory analytics

## Further Reading

- [Core Concepts](../../../core_concepts.md): Fundamental architecture and data structures
- [Memory Hooks](memory_hooks.md): Non-intrusive agent integration
- [Redis Integration](../../../redis_integration.md): Redis backend details
- [Memory Agent](../../../memory_agent.md): Memory agent implementation 