# **Agent State Redis Schema**

## **1. Introduction**

This document defines the Redis schema for storing and retrieving agent state information within the AgentMemory system. It provides specific data structures and access patterns optimized for the existing AgentState class and AgentStateModel database schema.

## **2. State Structure Alignment**

This schema aligns with the existing state structures defined in:

1. **`farm/core/state.py:AgentState`** - In-memory state representation
2. **`farm/database/models.py:AgentStateModel`** - SQLite database model
3. **`farm/agents/base_agent.py:get_state()`** - State generation method

## **3. Redis Schema for Agent State**

### **3.1 Key Structure**

All agent state keys follow these patterns:

| Pattern | Purpose | Example |
|---------|---------|---------|
| `agent:{agent_id}:state:{memory_id}` | Individual state snapshots | `agent:a123:state:a123-1000` |
| `agent:{agent_id}:state:current` | Most recent state | `agent:a123:state:current` |
| `agent:{agent_id}:state:timeline` | Chronological state index | `agent:a123:state:timeline` |
| `simulation:{sim_id}:agents:states` | States in a simulation | `simulation:sim01:agents:states` |

### **3.2 Primary State Hash Structure**

Each agent state is stored as a Redis hash with normalized field names matching the AgentState class:

```
agent:{agent_id}:state:{memory_id} (Hash)
```

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `agent_id` | String | Agent identifier | AgentState.agent_id |
| `step_number` | Integer | Simulation step | AgentState.step_number |
| `position_x` | Float | X coordinate | AgentState.position_x |
| `position_y` | Float | Y coordinate | AgentState.position_y |
| `position_z` | Float | Z coordinate | AgentState.position_z |
| `resource_level` | Float | Current resources | AgentState.resource_level |
| `current_health` | Float | Current health | AgentState.current_health |
| `is_defending` | Boolean | Defense stance | AgentState.is_defending |
| `total_reward` | Float | Cumulative reward | AgentState.total_reward |
| `age` | Integer | Agent age in steps | AgentState.age |
| `embedding` | Binary | State vector embedding | Generated from state values |
| `stored_at` | Integer | Timestamp of storage | System time when stored |

### **3.3 Index Structures**

#### **Timeline Index**

```
agent:{agent_id}:state:timeline (Sorted Set)
```

Maps memory_ids to step numbers for time-based queries:
- Score: `step_number`
- Member: `memory_id`

#### **Position Index**

```
agent:{agent_id}:state:position:{grid_cell} (Set)
```

For spatial queries, where `grid_cell` is a spatial partition identifier (e.g., "10:15"):
- Members: memory_ids of states in that grid cell

#### **Resource Level Index**

```
agent:{agent_id}:state:resource_level (Sorted Set)
```

For resource-level queries:
- Score: normalized resource level (0-1)
- Member: memory_id

#### **Health Index**

```
agent:{agent_id}:state:health (Sorted Set)
```

For health-based queries:
- Score: normalized health (0-1)
- Member: memory_id

### **3.4 Position Grid Implementation**

For efficient spatial queries, we partition the environment into a grid:

```python
def position_to_grid_cell(x, y, grid_size=10):
    """Convert position coordinates to grid cell identifier."""
    grid_x = int(x * 100) // grid_size
    grid_y = int(y * 100) // grid_size
    return f"{grid_x}:{grid_y}"
```

## **4. Storage Operations**

### **4.1 Storing a New State**

```python
def store_agent_state(redis_client, agent_id, state, step_number):
    """Store an agent state in Redis."""
    # Generate unique memory ID
    memory_id = f"{agent_id}-{step_number}"
    
    # Convert state to hash fields
    state_hash = {
        "agent_id": agent_id,
        "step_number": step_number,
        "position_x": state.position_x,
        "position_y": state.position_y,
        "position_z": state.position_z,
        "resource_level": state.resource_level,
        "current_health": state.current_health,
        "is_defending": "1" if state.is_defending else "0",
        "total_reward": state.total_reward,
        "age": state.age,
        "stored_at": int(time.time())
    }
    
    # Generate embedding if supported
    if hasattr(state, "to_tensor"):
        embedding = state.to_tensor(torch.device("cpu")).numpy().tobytes()
        state_hash["embedding"] = embedding
    
    # Store state hash
    redis_client.hset(f"agent:{agent_id}:state:{memory_id}", mapping=state_hash)
    
    # Update current state pointer
    redis_client.hset(f"agent:{agent_id}:state:current", mapping=state_hash)
    
    # Add to timeline index
    redis_client.zadd(f"agent:{agent_id}:state:timeline", {memory_id: step_number})
    
    # Add to position index
    grid_cell = position_to_grid_cell(state.position_x, state.position_y)
    redis_client.sadd(f"agent:{agent_id}:state:position:{grid_cell}", memory_id)
    
    # Add to resource level index
    redis_client.zadd(f"agent:{agent_id}:state:resource_level", 
                     {memory_id: state.resource_level})
    
    # Add to health index
    redis_client.zadd(f"agent:{agent_id}:state:health", 
                     {memory_id: state.current_health})
    
    # Add to simulation index
    redis_client.zadd(f"simulation:{state.simulation_id}:agents:states",
                     {memory_id: step_number})
    
    # Set TTL for state in STM
    redis_client.expire(f"agent:{agent_id}:state:{memory_id}", STM_TTL)
    
    return memory_id
```

### **4.2 Retrieving States by Time Range**

```python
def get_agent_states_by_time_range(redis_client, agent_id, min_step, max_step):
    """Retrieve agent states within a time range."""
    # Get memory IDs in the time range
    memory_ids = redis_client.zrangebyscore(
        f"agent:{agent_id}:state:timeline",
        min_step,
        max_step
    )
    
    # Retrieve each state
    states = []
    for memory_id in memory_ids:
        state_hash = redis_client.hgetall(f"agent:{agent_id}:state:{memory_id}")
        if state_hash:
            # Convert to AgentState object
            state = convert_hash_to_agent_state(state_hash)
            states.append(state)
    
    return states
```

### **4.3 Finding Similar States by Position**

```python
def find_states_by_position(redis_client, agent_id, x, y, radius=0.1):
    """Find agent states near a given position."""
    # Get nearby grid cells
    center_cell = position_to_grid_cell(x, y)
    cell_x, cell_y = map(int, center_cell.split(':'))
    
    # Check surrounding cells based on radius
    nearby_cells = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nearby_cells.append(f"{cell_x + dx}:{cell_y + dy}")
    
    # Get memory IDs from nearby cells
    candidate_ids = set()
    for cell in nearby_cells:
        cell_ids = redis_client.smembers(f"agent:{agent_id}:state:position:{cell}")
        candidate_ids.update(cell_ids)
    
    # Filter by exact distance
    states = []
    for memory_id in candidate_ids:
        state_hash = redis_client.hgetall(f"agent:{agent_id}:state:{memory_id}")
        if state_hash:
            px = float(state_hash["position_x"])
            py = float(state_hash["position_y"])
            
            # Calculate Euclidean distance
            distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            
            if distance <= radius:
                states.append(convert_hash_to_agent_state(state_hash))
    
    return states
```

## **5. State Conversion Utilities**

### **5.1 Converting Hash to AgentState**

```python
def convert_hash_to_agent_state(state_hash):
    """Convert Redis hash to AgentState object."""
    return AgentState(
        agent_id=state_hash["agent_id"],
        step_number=int(state_hash["step_number"]),
        position_x=float(state_hash["position_x"]),
        position_y=float(state_hash["position_y"]),
        position_z=float(state_hash["position_z"]),
        resource_level=float(state_hash["resource_level"]),
        current_health=float(state_hash["current_health"]),
        is_defending=state_hash["is_defending"] == "1",
        total_reward=float(state_hash["total_reward"]),
        age=int(state_hash["age"])
    )
```

### **5.2 Agent State Memory Integration**

```python
def remember_agent_state(agent, step_number):
    """Remember an agent's state at a specific step."""
    # Get current state
    state = agent.get_state()
    
    # Store in Redis
    memory_id = store_agent_state(
        redis_client=agent.memory_client,
        agent_id=agent.agent_id,
        state=state,
        step_number=step_number
    )
    
    return memory_id
```

## **6. Indexing Optimizations**

### **6.1 Importance-Based Retention**

To determine which states to keep longer in memory:

```python
def calculate_state_importance(state_hash):
    """Calculate importance score for state retention decisions."""
    importance = 0.0
    
    # Health-based importance (lower health = higher importance)
    health = float(state_hash["current_health"])
    importance += max(0, 0.5 - health)
    
    # Resource-based importance (very low or very high resources are important)
    resources = float(state_hash["resource_level"])
    if resources < 0.2 or resources > 0.8:
        importance += 0.3
    
    # Recent states are more important
    recency = time.time() - int(state_hash["stored_at"])
    importance += max(0, 0.5 - (recency / (24 * 60 * 60)))  # Decay over 24 hours
    
    return min(importance, 1.0)
```

### **6.2 Memory Tier Transition**

```python
def transition_state_to_intermediate_memory(redis_client, agent_id, memory_id):
    """Move state from STM to IM with compression."""
    # Get original state
    state_key = f"agent:{agent_id}:state:{memory_id}"
    state_hash = redis_client.hgetall(state_key)
    
    if not state_hash:
        return False
    
    # Create compressed version (fewer fields)
    compressed_hash = {
        "agent_id": state_hash["agent_id"],
        "step_number": state_hash["step_number"],
        "position_x": state_hash["position_x"],
        "position_y": state_hash["position_y"],
        "resource_level": state_hash["resource_level"],
        "current_health": state_hash["current_health"],
        "importance": calculate_state_importance(state_hash)
    }
    
    # Store in IM
    im_key = f"agent:{agent_id}:im:state:{memory_id}"
    redis_client.hset(im_key, mapping=compressed_hash)
    
    # Add to IM timeline
    redis_client.zadd(f"agent:{agent_id}:im:timeline", {memory_id: int(state_hash["step_number"])})
    
    # Set longer TTL for intermediate memory
    redis_client.expire(im_key, IM_TTL)
    
    # Remove from STM
    redis_client.delete(state_key)
    
    # Remove from STM indices
    redis_client.zrem(f"agent:{agent_id}:state:timeline", memory_id)
    
    grid_cell = position_to_grid_cell(
        float(state_hash["position_x"]), 
        float(state_hash["position_y"])
    )
    redis_client.srem(f"agent:{agent_id}:state:position:{grid_cell}", memory_id)
    redis_client.zrem(f"agent:{agent_id}:state:resource_level", memory_id)
    redis_client.zrem(f"agent:{agent_id}:state:health", memory_id)
    
    return True
```

## **7. Integration with BaseAgent**

To integrate this Redis schema with the `BaseAgent` class:

```python
def _init_memory(self, memory_config=None):
    """Initialize agent's memory system."""
    if not memory_config:
        memory_config = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "max_stm_states": 1000,
            "stm_ttl": 3600,  # 1 hour
            "im_ttl": 86400   # 1 day
        }
    
    # Initialize Redis client for memory operations
    self.memory_client = redis.Redis(
        host=memory_config.get("host", "localhost"),
        port=memory_config.get("port", 6379),
        db=memory_config.get("db", 0)
    )
    
    self.memory_config = memory_config
    self.use_memory = True
```

## **8. Query Patterns for Agent Behavior**

### **8.1 Finding Prior Similar States**

```python
def recall_similar_situations(self, position=None, limit=5):
    """Recall similar past situations based on position."""
    if not self.use_memory:
        return []
    
    # Use current position if none provided
    if position is None:
        position = (self.position_x, self.position_y)
    
    # Find past states in similar positions
    similar_states = find_states_by_position(
        redis_client=self.memory_client,
        agent_id=self.agent_id,
        x=position[0],
        y=position[1],
        radius=0.1
    )
    
    # Sort by recency
    similar_states.sort(key=lambda s: s.step_number, reverse=True)
    
    return similar_states[:limit]
```

### **8.2 Learning From Past Experiences**

```python
def learn_from_past_experiences(self, current_state):
    """Learn from past similar experiences to inform current decision."""
    if not self.use_memory:
        return None
    
    # Get similar past states
    similar_states = self.recall_similar_situations(
        position=(current_state.position_x, current_state.position_y),
        limit=10
    )
    
    if not similar_states:
        return None
    
    # Calculate average reward in similar situations
    avg_reward = sum(s.total_reward for s in similar_states) / len(similar_states)
    
    # Get the most rewarding state
    best_state = max(similar_states, key=lambda s: s.total_reward)
    
    return {
        "avg_reward": avg_reward,
        "best_state": best_state,
        "similar_count": len(similar_states)
    }
```

## **9. Performance Considerations**

### **9.1 Memory Usage**

Approximate memory usage per agent with 1,000 states in STM:
- Base state data: ~200 bytes per state × 1,000 = 200 KB
- Indices: ~100 bytes per state × 1,000 = 100 KB
- Embeddings (optional): ~400 bytes per state × 1,000 = 400 KB
- Total per agent: ~700 KB (with embeddings)

### **9.2 Redis Configuration Recommendations**

For optimal performance with agent state storage:

```
# redis.conf settings
maxmemory 4gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

### **9.3 Connection Pooling**

For multi-agent environments:

```python
# Create a shared connection pool
redis_pool = redis.ConnectionPool(
    host=memory_config.get("host", "localhost"),
    port=memory_config.get("port", 6379),
    db=memory_config.get("db", 0),
    max_connections=50
)

# Use in agent initialization
self.memory_client = redis.Redis(connection_pool=redis_pool)
``` 