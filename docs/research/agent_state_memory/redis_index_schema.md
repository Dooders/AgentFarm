# **Redis Index Schema for AgentMemory**

## **1. Introduction**

This document formalizes the Redis key structure and index design for the AgentMemory system. It specifies the Redis data structures, key naming conventions, and index patterns used for efficient memory storage and retrieval operations.

## **2. Key Naming Conventions**

All Redis keys follow a hierarchical naming pattern to ensure logical organization and to avoid key collisions:

| Pattern | Description | Example |
|---------|-------------|---------|
| `agent:{agent_id}:stm:{entity}` | Short-Term Memory (STM) for a specific agent | `agent:a123:stm:timeline` |
| `agent:{agent_id}:im:{entity}` | Intermediate Memory (IM) for a specific agent | `agent:a123:im:timeline` |
| `{simulation_id}:{entity}` | Simulation-wide data | `sim456:actions` |

## **3. Primary Data Structures**

### **3.1 Memory Storage**

| Key Pattern | Redis Type | Purpose | Fields/Format |
|-------------|------------|---------|---------------|
| `agent:{agent_id}:stm` | Hash | Primary STM storage | memory_id → JSON string |
| `agent:{agent_id}:im` | Hash | Primary IM storage | memory_id → JSON string |
| `agent:{agent_id}:stm:embedding:{memory_id}` | String | Vector embeddings | Binary encoded vector |
| `{simulation_id}:{entity}` | List | Buffered simulation data | JSON strings |

### **3.2 Index Structures**

| Key Pattern | Redis Type | Purpose | Format |
|-------------|------------|---------|--------|
| `agent:{agent_id}:stm:timeline` | Sorted Set | Temporal index | memory_id → step_number |
| `agent:{agent_id}:im:timeline` | Sorted Set | Temporal index for IM | memory_id → step_number |
| `agent:{agent_id}:stm:type:{memory_type}` | Set | Type-based index | Set of memory_ids |
| `agent:{agent_id}:stm:action_type:{action_type}` | Set | Action-specific index | Set of memory_ids |
| `agent:{agent_id}:stm:importance` | Sorted Set | Importance-based index | memory_id → importance_score |

### **3.4 Relative Time Index**

To efficiently access agent states by their relative temporal position (where 0 is the most recent state, -1 is the previous state, etc.), we implement a specialized index:

```
agent:{agent_id}:state:relative_index (Sorted Set)
```

This index maintains a fixed-size sorted set of state IDs ordered by their step number:
- Score: step_number
- Member: state_id

The relative index is updated with each new state storage operation, maintaining only the N most recent states (where N is typically configured to match the short-term memory size). This allows for O(1) access to any relative state position.

Example operations for maintaining and using the relative index:

```python
def update_relative_index(redis_client, agent_id, state_id, step_number, max_size=20):
    """Update the relative time index with a new state."""
    # Add the state to the relative index
    redis_client.zadd(f"agent:{agent_id}:state:relative_index", {state_id: step_number})
    
    # Trim the index to maintain only the most recent states
    redis_client.zremrangebyrank(f"agent:{agent_id}:state:relative_index", 0, -max_size-1)

def get_state_by_relative_position(redis_client, agent_id, relative_position):
    """Get a state by its relative position (0 = current, -1 = previous, etc.)."""
    # Convert relative position to rank (negative positions start from the highest rank)
    rank = -1 - relative_position if relative_position <= 0 else relative_position
    
    # Get the state ID at the specified rank
    state_ids = redis_client.zrange(
        f"agent:{agent_id}:state:relative_index", 
        rank, 
        rank
    )
    
    if not state_ids:
        return None
    
    state_id = state_ids[0]
    
    # Retrieve the state data
    state_data = redis_client.hgetall(f"agent:{agent_id}:state:{state_id}")
    return state_data if state_data else None
```

Usage example:

```python
# Get the current state (relative position 0)
current_state = get_state_by_relative_position(redis_client, "agent1", 0)

# Get the previous state (relative position -1)
previous_state = get_state_by_relative_position(redis_client, "agent1", -1)

# Get the state from 3 steps ago (relative position -3)
older_state = get_state_by_relative_position(redis_client, "agent1", -3)
```

This relative indexing approach provides an intuitive and efficient way to access an agent's recent history without needing to track the absolute step numbers.

## **4. Vector Similarity Index**

For vector similarity operations, we use a composite approach without specialized Redis modules:

```
agent:{agent_id}:stm:vector_buckets:{bucket_id}  (Set)
```

Where `bucket_id` is derived from the vector using a space-partitioning technique like LSH (Locality-Sensitive Hashing) or feature-based bucketing.

## **5. Index Operations**

### **5.1 Adding to Indices**

When a new memory is stored, it is indexed in multiple ways:

```python
# Temporal indexing
redis_client.zadd(f"agent:{agent_id}:stm:timeline", {memory_id: step_number})

# Type-based indexing
redis_client.sadd(f"agent:{agent_id}:stm:type:{memory_type}", memory_id)

# Importance-based indexing
redis_client.zadd(f"agent:{agent_id}:stm:importance", {memory_id: importance_score})

# Vector similarity indexing
bucket_ids = calculate_vector_buckets(embedding)
for bucket_id in bucket_ids:
    redis_client.sadd(f"agent:{agent_id}:stm:vector_buckets:{bucket_id}", memory_id)
```

### **5.2 Memory Cleanup**

When memories are transitioned from STM to IM or removed:

```python
# Remove from all indices
redis_client.zrem(f"agent:{agent_id}:stm:timeline", memory_id)
redis_client.srem(f"agent:{agent_id}:stm:type:{memory_type}", memory_id)
redis_client.zrem(f"agent:{agent_id}:stm:importance", memory_id)

# Remove from vector buckets
bucket_ids = get_memory_buckets(agent_id, memory_id)
for bucket_id in bucket_ids:
    redis_client.srem(f"agent:{agent_id}:stm:vector_buckets:{bucket_id}", memory_id)
```

## **6. Query Patterns**

### **6.1 Temporal Queries**

```python
# Get memories in time range
memory_ids = redis_client.zrangebyscore(
    f"agent:{agent_id}:stm:timeline", 
    min_step, 
    max_step
)
```

### **6.2 Type-Based Queries**

```python
# Get all memories of a specific type
memory_ids = redis_client.smembers(f"agent:{agent_id}:stm:type:state")
```

### **6.3 Combined Queries**

For compound queries, set operations are used:

```python
# Get important state memories in a time range
time_range_ids = redis_client.zrangebyscore(
    f"agent:{agent_id}:stm:timeline", 
    min_step, 
    max_step
)
redis_client.sadd("temp:time_range", *time_range_ids)

# Intersect with state memories
redis_client.sinterstore(
    "temp:result",
    "temp:time_range",
    f"agent:{agent_id}:stm:type:state"
)

# Sort by importance
high_importance_ids = redis_client.zrevrangebyscore(
    f"agent:{agent_id}:stm:importance",
    "+inf", 
    importance_threshold
)
redis_client.sadd("temp:importance", *high_importance_ids)

# Final intersection
redis_client.sinterstore(
    "temp:final_result",
    "temp:result",
    "temp:importance"
)

result_ids = redis_client.smembers("temp:final_result")
```

## **7. Vector Similarity Search**

Without specialized Redis modules, approximate vector similarity search uses bucket-based approach:

```python
def search_similar_vectors(agent_id, query_vector, k=5):
    # Calculate buckets for query vector
    query_buckets = calculate_vector_buckets(query_vector)
    
    # Get candidate memory IDs from all matching buckets
    candidates = set()
    for bucket_id in query_buckets:
        bucket_members = redis_client.smembers(
            f"agent:{agent_id}:stm:vector_buckets:{bucket_id}"
        )
        candidates.update(bucket_members)
    
    # Retrieve full vectors for candidates
    candidate_vectors = {}
    for memory_id in candidates:
        vector_data = redis_client.get(
            f"agent:{agent_id}:stm:embedding:{memory_id}"
        )
        if vector_data:
            candidate_vectors[memory_id] = decode_vector(vector_data)
    
    # Compute actual similarities
    similarities = []
    for memory_id, vector in candidate_vectors.items():
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((memory_id, similarity))
    
    # Return top k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]
```

## **7. Agent Integration Methods**

### **7.1 BaseAgent Methods for State Memory**

The BaseAgent class is extended with methods that use the Redis schema for state memory:

```python
class BaseAgent:
    # ... existing code ...
    
    def get_current_state(self):
        """Get the agent's current (most recent) state."""
        if not self.use_memory or not self.memory_client:
            return None
            
        return get_state_by_relative_position(self.memory_client, self.agent_id, 0)
    
    def get_previous_state(self):
        """Get the agent's previous state."""
        if not self.use_memory or not self.memory_client:
            return None
            
        return get_state_by_relative_position(self.memory_client, self.agent_id, -1)
    
    def get_state_at_relative_position(self, relative_position):
        """Get a state by its relative position to the current one."""
        if not self.use_memory or not self.memory_client:
            return None
            
        return get_state_by_relative_position(self.memory_client, self.agent_id, relative_position)
    
    def get_state_history(self, steps=5):
        """Get the last N states as a chronological history."""
        if not self.use_memory or not self.memory_client:
            return []
            
        states = []
        for i in range(steps):
            # Start from the most recent state (0) and go back in time
            relative_pos = -i if i > 0 else 0
            state = get_state_by_relative_position(self.memory_client, self.agent_id, relative_pos)
            if state:
                states.append(state)
            else:
                # No more states available
                break
                
        return states
    
    def calculate_state_changes(self, steps=1):
        """Calculate changes between current state and N steps ago."""
        if not self.use_memory or not self.memory_client:
            return None
            
        current = self.get_current_state()
        previous = self.get_state_at_relative_position(-steps)
        
        if not current or not previous:
            return None
            
        # Calculate differences
        changes = {}
        for key in current:
            if key in previous and key not in ['state_id', 'step_number', 'stored_at']:
                try:
                    # Try numeric comparison
                    curr_val = float(current[key])
                    prev_val = float(previous[key])
                    changes[key] = {
                        'previous': prev_val,
                        'current': curr_val,
                        'change': curr_val - prev_val,
                        'percent_change': ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else None
                    }
                except (ValueError, TypeError):
                    # Non-numeric comparison
                    changes[key] = {
                        'previous': previous[key],
                        'current': current[key],
                        'changed': current[key] != previous[key]
                    }
        
        return changes
    
    # ... additional methods ...
```

### **7.2 Example Usage in Agent Behavior**

The relative state indexing provides powerful capabilities for implementing agent behaviors that depend on recent history:

```python
def decide_next_action(self):
    """Determine next action based on state changes."""
    # Get current state and changes
    changes = self.calculate_state_changes(steps=3)
    
    if not changes:
        return "explore"  # Default action
    
    # Check if resources have been decreasing
    if 'resource_level' in changes:
        resource_change = changes['resource_level']['change']
        
        if resource_change < -0.2:  # Significant resource decrease
            # Resources dropping, prioritize resource gathering
            return "gather"
        elif resource_change > 0.5:  # Significant resource increase
            # Resources growing, can afford to explore or socialize
            return "socialize"
    
    # Check position changes to see if we've been moving
    if 'position_x' in changes and 'position_y' in changes:
        pos_x_change = abs(changes['position_x']['change'])
        pos_y_change = abs(changes['position_y']['change'])
        
        if pos_x_change < 0.1 and pos_y_change < 0.1:
            # We've been stationary, time to move
            return "explore"
    
    # Default behavior
    return "explore"
```

This approach allows agents to make decisions based on their recent history without needing to explicitly track state changes in agent code. All the temporal information is efficiently stored and retrieved through the Redis relative time index.

## **8. Performance Considerations**

### **8.1 Memory Usage**

The index structures consume additional memory beyond the primary data storage:

- Timeline indices: O(n) where n is number of memories
- Type indices: O(n) total across all types
- Vector bucket indices: O(b*n) where b is average buckets per vector

### **8.2 Expiration and Cleanup**

For IM data with TTL-based expiration:

```python
# Set expiration for IM entries
redis_client.expire(f"agent:{agent_id}:im:{memory_id}", ttl_seconds)
```

### **8.3 Memory Management**

Periodic maintenance tasks:

```python
def cleanup_orphaned_indices():
    # Find all memory IDs in primary storage
    valid_ids = redis_client.hkeys(f"agent:{agent_id}:stm")
    
    # Check each index for orphaned entries
    timeline_ids = redis_client.zrange(
        f"agent:{agent_id}:stm:timeline", 
        0, 
        -1
    )
    
    # Remove orphaned timeline entries
    orphaned_timeline = set(timeline_ids) - set(valid_ids)
    if orphaned_timeline:
        redis_client.zrem(
            f"agent:{agent_id}:stm:timeline", 
            *orphaned_timeline
        )
```

## **9. Integration with Memory System**

This Redis index schema supports all memory operations described in the [Core Concepts](core_concepts.md) document, providing:

1. Efficient temporal-based memory retrieval
2. Type-based filtering for specific memory operations
3. Approximate vector similarity search without specialized Redis modules
4. Support for the hierarchical memory transition process

For further details on implementation, see [Redis Integration](redis_integration.md). 

def store_agent_state(redis_client, agent_id, state_data, simulation_id, step_number, ttl=None):
    """Store an agent state in Redis with appropriate indexing."""
    # Generate unique state ID
    state_id = f"{agent_id}-{step_number}"
    
    # Add metadata fields
    state_data["state_id"] = state_id
    state_data["agent_id"] = agent_id
    state_data["simulation_id"] = simulation_id
    state_data["step_number"] = step_number
    state_data["stored_at"] = int(time.time())
    
    # Store the state hash
    redis_client.hset(f"agent:{agent_id}:state:{state_id}", mapping=state_data)
    
    # Add to timeline index
    redis_client.zadd(f"agent:{agent_id}:state:timeline", {state_id: step_number})
    
    # Add to relative time index and maintain its size
    update_relative_index(redis_client, agent_id, state_id, step_number)
    
    # Add to spatial index if position data exists
    if "position_x" in state_data and "position_y" in state_data:
        spatial_key = get_spatial_key(float(state_data["position_x"]), float(state_data["position_y"]))
        redis_client.sadd(f"spatial:{spatial_key}:states", state_id)
        redis_client.sadd(f"agent:{agent_id}:state:spatial_keys", spatial_key)
    
    # Add to type-specific index if type exists
    if "agent_type" in state_data:
        redis_client.sadd(f"type:{state_data['agent_type']}:agents", agent_id)
        redis_client.sadd(f"type:{state_data['agent_type']}:states", state_id)
    
    # Set expiration time for automatic cleanup (TTL in seconds)
    if ttl:
        redis_client.expire(f"agent:{agent_id}:state:{state_id}", ttl)
    
    # Add to simulation index
    redis_client.sadd(f"simulation:{simulation_id}:agents", agent_id)
    redis_client.sadd(f"simulation:{simulation_id}:states", state_id)
    redis_client.zadd(f"simulation:{simulation_id}:timeline", {state_id: step_number})
    
    return state_id 