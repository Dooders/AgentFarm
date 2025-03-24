# **Agent Interaction Redis Schema**

## **1. Introduction**

This document defines the Redis schema for storing and retrieving agent social interaction information within the AgentMemory system. It provides specific data structures and access patterns optimized for the existing SocialInteractionModel database schema and social interaction tracking requirements.

## **2. Interaction Structure Alignment**

This schema aligns with the existing interaction structures defined in:

1. **`farm/database/models.py:SocialInteractionModel`** - SQLite database model for social interactions
2. **`farm/agents/base_agent.py`** - Interaction-related methods in the agent class

## **3. Redis Schema for Social Interactions**

### **3.1 Key Structure**

All social interaction keys follow these patterns:

| Pattern | Purpose | Example |
|---------|---------|---------|
| `interaction:{interaction_id}` | Individual interaction records | `interaction:a123-b456-1000` |
| `agent:{agent_id}:interactions:initiated` | Interactions initiated by agent | `agent:a123:interactions:initiated` |
| `agent:{agent_id}:interactions:received` | Interactions received by agent | `agent:a123:interactions:received` |
| `agent:{agent_id}:interactions:with:{other_id}` | Interactions between specific agents | `agent:a123:interactions:with:b456` |
| `simulation:{sim_id}:interactions` | Interactions in a simulation | `simulation:sim01:interactions` |
| `simulation:{sim_id}:interactions:types:{type}` | Interactions by type | `simulation:sim01:interactions:types:cooperation` |

### **3.2 Primary Interaction Hash Structure**

Each social interaction is stored as a Redis hash with normalized field names matching the SocialInteractionModel schema:

```
interaction:{interaction_id} (Hash)
```

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `interaction_id` | String | Unique interaction identifier | `{initiator_id}-{recipient_id}-{step}` |
| `simulation_id` | String | Simulation identifier | SocialInteractionModel.simulation_id |
| `step_number` | Integer | Step when interaction occurred | SocialInteractionModel.step_number |
| `initiator_id` | String | Agent that initiated interaction | SocialInteractionModel.initiator_id |
| `recipient_id` | String | Agent that received interaction | SocialInteractionModel.recipient_id |
| `interaction_type` | String | Type of interaction | SocialInteractionModel.interaction_type |
| `subtype` | String | Specific subtype of interaction | SocialInteractionModel.subtype |
| `outcome` | String | Result of interaction | SocialInteractionModel.outcome |
| `resources_transferred` | Float | Resources exchanged | SocialInteractionModel.resources_transferred |
| `distance` | Float | Distance between agents | SocialInteractionModel.distance |
| `initiator_resources_before` | Float | Initiator's resources before | SocialInteractionModel.initiator_resources_before |
| `initiator_resources_after` | Float | Initiator's resources after | SocialInteractionModel.initiator_resources_after |
| `recipient_resources_before` | Float | Recipient's resources before | SocialInteractionModel.recipient_resources_before |
| `recipient_resources_after` | Float | Recipient's resources after | SocialInteractionModel.recipient_resources_after |
| `group_id` | String | Group identifier if applicable | SocialInteractionModel.group_id |
| `details` | String | JSON-encoded interaction details | SocialInteractionModel.details |
| `timestamp` | Integer | Unix timestamp | SocialInteractionModel.timestamp |
| `stored_at` | Integer | Timestamp of storage | System time when stored |
| `embedding` | Binary | Interaction embedding vector | Generated from interaction data |

### **3.3 Index Structures**

#### **Timeline Indices**

```
agent:{agent_id}:interactions:timeline (Sorted Set)
```

Maps interaction_ids to step numbers for time-based queries:
- Score: `step_number`
- Member: `interaction_id`

#### **Type and Subtype Indices**

```
agent:{agent_id}:interactions:types:{interaction_type} (Set)
```

For interaction type-based queries:
- Members: interaction_ids of that type

```
agent:{agent_id}:interactions:subtypes:{subtype} (Set)
```

For subtype-based queries:
- Members: interaction_ids of that subtype

#### **Outcome Index**

```
agent:{agent_id}:interactions:outcomes:{outcome} (Set)
```

For outcome-based queries:
- Members: interaction_ids with that outcome

#### **Group Index**

```
group:{group_id}:interactions (Set)
```

For group-based queries:
- Members: interaction_ids involving that group

#### **Resource Transfer Index**

```
agent:{agent_id}:interactions:resource_transfers (Sorted Set)
```

For resource transfer-based queries:
- Score: amount of resources transferred
- Member: interaction_id

#### **Relative Time Index**

```
agent:{agent_id}:interactions:relative_index (Sorted Set)
```

This index maintains a fixed-size sorted set of interaction IDs ordered by their step number, enabling efficient access to interactions by their relative temporal position (0 for most recent, -1 for previous, etc.):
- Score: step_number
- Member: interaction_id

### **3.4 Relationship Indices**

These indices track relationships between agents based on their interaction history:

```
agent:{agent_id}:relationships (Sorted Set)
```

For identifying important relationships:
- Score: relationship strength score
- Member: other agent ID

```
agent:{agent_id}:relationships:{other_id}:interactions (Sorted Set)
```

For tracking all interactions with a specific agent:
- Score: step_number
- Member: interaction_id

## **4. Storage Operations**

### **4.1 Storing a New Interaction**

```python
def store_social_interaction(
    redis_client,
    simulation_id,
    step_number,
    initiator_id,
    recipient_id,
    interaction_type,
    subtype=None,
    outcome=None,
    resources_transferred=None,
    distance=None,
    initiator_resources_before=None,
    initiator_resources_after=None,
    recipient_resources_before=None,
    recipient_resources_after=None,
    group_id=None,
    details=None
):
    """Store a social interaction in Redis."""
    # Generate unique interaction ID
    interaction_id = f"{initiator_id}-{recipient_id}-{step_number}"
    timestamp = int(time.time())
    
    # Convert interaction to hash fields
    interaction_hash = {
        "interaction_id": interaction_id,
        "simulation_id": simulation_id,
        "step_number": step_number,
        "initiator_id": initiator_id,
        "recipient_id": recipient_id,
        "interaction_type": interaction_type,
        "timestamp": timestamp,
        "stored_at": timestamp
    }
    
    # Add optional fields
    if subtype:
        interaction_hash["subtype"] = subtype
    if outcome:
        interaction_hash["outcome"] = outcome
    if resources_transferred is not None:
        interaction_hash["resources_transferred"] = resources_transferred
    if distance is not None:
        interaction_hash["distance"] = distance
    if initiator_resources_before is not None:
        interaction_hash["initiator_resources_before"] = initiator_resources_before
    if initiator_resources_after is not None:
        interaction_hash["initiator_resources_after"] = initiator_resources_after
    if recipient_resources_before is not None:
        interaction_hash["recipient_resources_before"] = recipient_resources_before
    if recipient_resources_after is not None:
        interaction_hash["recipient_resources_after"] = recipient_resources_after
    if group_id:
        interaction_hash["group_id"] = group_id
    if details:
        interaction_hash["details"] = json.dumps(details) if isinstance(details, dict) else details
    
    # Store interaction hash
    redis_client.hset(f"interaction:{interaction_id}", mapping=interaction_hash)
    
    # Add to agent timeline indices
    redis_client.zadd(f"agent:{initiator_id}:interactions:timeline", {interaction_id: step_number})
    redis_client.zadd(f"agent:{recipient_id}:interactions:timeline", {interaction_id: step_number})
    
    # Add to relative time indices for both agents
    update_interaction_relative_index(redis_client, initiator_id, interaction_id, step_number)
    update_interaction_relative_index(redis_client, recipient_id, interaction_id, step_number)
    
    # Add to initiated and received indices
    redis_client.zadd(f"agent:{initiator_id}:interactions:initiated", {interaction_id: step_number})
    redis_client.zadd(f"agent:{recipient_id}:interactions:received", {interaction_id: step_number})
    
    # Add to mutual interaction index
    redis_client.zadd(f"agent:{initiator_id}:interactions:with:{recipient_id}", {interaction_id: step_number})
    redis_client.zadd(f"agent:{recipient_id}:interactions:with:{initiator_id}", {interaction_id: step_number})
    
    # Add to type and subtype indices
    redis_client.sadd(f"agent:{initiator_id}:interactions:types:{interaction_type}", interaction_id)
    redis_client.sadd(f"agent:{recipient_id}:interactions:types:{interaction_type}", interaction_id)
    
    if subtype:
        redis_client.sadd(f"agent:{initiator_id}:interactions:subtypes:{subtype}", interaction_id)
        redis_client.sadd(f"agent:{recipient_id}:interactions:subtypes:{subtype}", interaction_id)
    
    # Add to outcome index if outcome exists
    if outcome:
        redis_client.sadd(f"agent:{initiator_id}:interactions:outcomes:{outcome}", interaction_id)
        redis_client.sadd(f"agent:{recipient_id}:interactions:outcomes:{outcome}", interaction_id)
    
    # Add to group index if group exists
    if group_id:
        redis_client.sadd(f"group:{group_id}:interactions", interaction_id)
    
    # Add to resource transfer index if resources transferred
    if resources_transferred is not None and resources_transferred != 0:
        redis_client.zadd(
            f"agent:{initiator_id}:interactions:resource_transfers", 
            {interaction_id: resources_transferred}
        )
        redis_client.zadd(
            f"agent:{recipient_id}:interactions:resource_transfers", 
            {interaction_id: resources_transferred}
        )
    
    # Add to simulation indices
    redis_client.rpush(f"simulation:{simulation_id}:interactions", interaction_id)
    redis_client.sadd(f"simulation:{simulation_id}:interactions:types:{interaction_type}", interaction_id)
    
    # Update relationship strength
    update_relationship_strength(
        redis_client, 
        initiator_id, 
        recipient_id, 
        interaction_type, 
        outcome, 
        resources_transferred
    )
    
    # Set TTL for interaction in STM
    redis_client.expire(f"interaction:{interaction_id}", INTERACTION_STM_TTL)
    
    return interaction_id
```

### **4.2 Retrieving Interactions Between Agents**

```python
def get_interactions_between_agents(redis_client, agent1_id, agent2_id, limit=10):
    """Retrieve interactions between two specific agents."""
    # Get interaction IDs from the relationship index
    interaction_ids = redis_client.zrevrange(
        f"agent:{agent1_id}:interactions:with:{agent2_id}",
        0,
        limit - 1
    )
    
    # Retrieve each interaction
    interactions = []
    for interaction_id in interaction_ids:
        interaction_hash = redis_client.hgetall(f"interaction:{interaction_id}")
        if interaction_hash:
            interactions.append(convert_hash_to_interaction(interaction_hash))
    
    return interactions
```

### **4.3 Finding Successful Cooperation**

```python
def find_successful_cooperation(redis_client, agent_id, limit=5):
    """Find successful cooperation interactions involving an agent."""
    # First get cooperation interaction IDs
    cooperation_ids = redis_client.sinter(
        f"agent:{agent_id}:interactions:types:cooperation",
        f"agent:{agent_id}:interactions:outcomes:successful"
    )
    
    # Limit and retrieve each interaction
    interaction_ids = list(cooperation_ids)[:limit]
    interactions = []
    
    for interaction_id in interaction_ids:
        interaction_hash = redis_client.hgetall(f"interaction:{interaction_id}")
        if interaction_hash:
            interactions.append(convert_hash_to_interaction(interaction_hash))
    
    return interactions
```

## **5. Interaction Conversion Utilities**

### **5.1 Converting Hash to Interaction Object**

```python
def convert_hash_to_interaction(interaction_hash):
    """Convert Redis hash to Interaction object."""
    # Parse details JSON if present
    details = None
    if "details" in interaction_hash:
        try:
            details = json.loads(interaction_hash["details"])
        except:
            details = interaction_hash["details"]
    
    # Create interaction object (can be a simple dict or custom class)
    interaction = {
        "interaction_id": interaction_hash["interaction_id"],
        "simulation_id": interaction_hash["simulation_id"],
        "step_number": int(interaction_hash["step_number"]),
        "initiator_id": interaction_hash["initiator_id"],
        "recipient_id": interaction_hash["recipient_id"],
        "interaction_type": interaction_hash["interaction_type"],
        "subtype": interaction_hash.get("subtype"),
        "outcome": interaction_hash.get("outcome"),
        "timestamp": int(interaction_hash["timestamp"]),
        "details": details
    }
    
    # Add numeric fields if present
    for field in [
        "resources_transferred", "distance", 
        "initiator_resources_before", "initiator_resources_after",
        "recipient_resources_before", "recipient_resources_after"
    ]:
        if field in interaction_hash:
            interaction[field] = float(interaction_hash[field])
    
    # Add group ID if present
    if "group_id" in interaction_hash:
        interaction["group_id"] = interaction_hash["group_id"]
    
    return interaction
```

### **5.2 Agent Interaction Memory Integration**

```python
def remember_social_interaction(
    self, 
    recipient_id, 
    interaction_type, 
    subtype=None, 
    outcome=None, 
    resources_transferred=None, 
    details=None
):
    """Remember a social interaction with another agent."""
    if not self.use_memory:
        return None
    
    # Calculate distance to recipient
    recipient = None
    distance = None
    
    for agent in self.environment.agents:
        if agent.agent_id == recipient_id:
            recipient = agent
            distance = ((self.position_x - agent.position_x) ** 2 + 
                       (self.position_y - agent.position_y) ** 2) ** 0.5
            break
    
    # Store in Redis
    interaction_id = store_social_interaction(
        redis_client=self.memory_client,
        simulation_id=self.environment.simulation_id,
        step_number=self.environment.time,
        initiator_id=self.agent_id,
        recipient_id=recipient_id,
        interaction_type=interaction_type,
        subtype=subtype,
        outcome=outcome,
        resources_transferred=resources_transferred,
        distance=distance,
        initiator_resources_before=self.resource_level,
        details=details
    )
    
    return interaction_id
```

## **6. Relationship Strength Calculation**

```python
def update_relationship_strength(
    redis_client, 
    agent1_id, 
    agent2_id, 
    interaction_type, 
    outcome, 
    resources_transferred
):
    """Update relationship strength between two agents based on interaction."""
    # Get current relationship strength
    strength1 = redis_client.zscore(f"agent:{agent1_id}:relationships", agent2_id) or 0
    strength2 = redis_client.zscore(f"agent:{agent2_id}:relationships", agent1_id) or 0
    
    # Calculate interaction value
    interaction_value = 0
    
    # Base interaction value by type
    type_values = {
        "cooperation": 0.5,
        "competition": 0.2,
        "sharing": 0.7,
        "conflict": -0.3,
        "group_formation": 0.8
    }
    interaction_value += type_values.get(interaction_type, 0.1)
    
    # Adjust for outcome
    if outcome == "successful":
        interaction_value *= 1.5
    elif outcome == "rejected":
        interaction_value *= 0.5
    elif outcome == "conflict":
        interaction_value *= -0.5
    
    # Adjust for resource transfers
    if resources_transferred is not None:
        # Positive transfers strengthen relationship, negative transfers weaken
        interaction_value += min(0.3, abs(resources_transferred) * 0.1) * (
            1 if resources_transferred > 0 else -1
        )
    
    # Apply decay to existing strength (recent interactions matter more)
    decay_factor = 0.95
    new_strength1 = (strength1 * decay_factor) + interaction_value
    new_strength2 = (strength2 * decay_factor) + interaction_value
    
    # Ensure strength stays in reasonable range [-1, 1]
    new_strength1 = max(-1, min(1, new_strength1))
    new_strength2 = max(-1, min(1, new_strength2))
    
    # Update relationship strengths
    redis_client.zadd(f"agent:{agent1_id}:relationships", {agent2_id: new_strength1})
    redis_client.zadd(f"agent:{agent2_id}:relationships", {agent1_id: new_strength2})
    
    return new_strength1, new_strength2
```

## **7. Memory Tier Transition**

```python
def transition_interaction_to_intermediate_memory(redis_client, interaction_id):
    """Move interaction from STM to IM with compression."""
    # Get original interaction
    interaction_key = f"interaction:{interaction_id}"
    interaction_hash = redis_client.hgetall(interaction_key)
    
    if not interaction_hash:
        return False
    
    # Calculate importance score
    importance = calculate_interaction_importance(interaction_hash)
    
    # Extract agent IDs
    initiator_id = interaction_hash["initiator_id"]
    recipient_id = interaction_hash["recipient_id"]
    
    # Create compressed version (fewer fields)
    compressed_hash = {
        "interaction_id": interaction_hash["interaction_id"],
        "simulation_id": interaction_hash["simulation_id"],
        "step_number": interaction_hash["step_number"],
        "initiator_id": initiator_id,
        "recipient_id": recipient_id,
        "interaction_type": interaction_hash["interaction_type"],
        "outcome": interaction_hash.get("outcome", "unknown"),
        "importance": str(importance)
    }
    
    # Add resource transfer if it exists (important for social interactions)
    if "resources_transferred" in interaction_hash:
        compressed_hash["resources_transferred"] = interaction_hash["resources_transferred"]
    
    # Store in IM
    im_key = f"interaction:im:{interaction_id}"
    redis_client.hset(im_key, mapping=compressed_hash)
    
    # Add to IM indices
    redis_client.zadd(
        f"agent:{initiator_id}:im:interactions:timeline", 
        {interaction_id: int(interaction_hash["step_number"])}
    )
    redis_client.zadd(
        f"agent:{recipient_id}:im:interactions:timeline", 
        {interaction_id: int(interaction_hash["step_number"])}
    )
    
    # Set TTL for intermediate memory
    redis_client.expire(im_key, INTERACTION_IM_TTL)
    
    # Remove from STM if this is a true transition (not just duplication)
    redis_client.delete(interaction_key)
    
    # Remove from STM indices (partial list for brevity)
    redis_client.zrem(f"agent:{initiator_id}:interactions:timeline", interaction_id)
    redis_client.zrem(f"agent:{recipient_id}:interactions:timeline", interaction_id)
    redis_client.zrem(f"agent:{initiator_id}:interactions:initiated", interaction_id)
    redis_client.zrem(f"agent:{recipient_id}:interactions:received", interaction_id)
    
    # Keep relationship indices even after transition
    # This preserves the social graph even as individual interactions are compressed
    
    return True
```

## **8. Interaction Importance Calculation**

```python
def calculate_interaction_importance(interaction_hash):
    """Calculate importance score for interaction retention decisions."""
    importance = 0.0
    
    # Type-based importance
    interaction_type = interaction_hash["interaction_type"]
    type_importance = {
        "cooperation": 0.6,
        "competition": 0.4,
        "sharing": 0.7,
        "conflict": 0.8,  # Conflicts are important to remember
        "group_formation": 0.9  # Group formation is highly important
    }
    importance += type_importance.get(interaction_type, 0.3)
    
    # Outcome-based importance
    if "outcome" in interaction_hash:
        outcome = interaction_hash["outcome"]
        if outcome == "successful":
            importance += 0.2
        elif outcome == "conflict":
            importance += 0.3  # Conflicts are important to remember
    
    # Resource transfer importance
    if "resources_transferred" in interaction_hash:
        transfer = float(interaction_hash["resources_transferred"])
        # Larger transfers (positive or negative) are more important
        importance += min(abs(transfer) * 0.1, 0.3)
    
    # Recency-based importance
    if "stored_at" in interaction_hash:
        recency = time.time() - int(interaction_hash["stored_at"])
        # More recent interactions are more important
        importance += max(0, 0.2 - (recency / (7 * 24 * 60 * 60)) * 0.2)  # Decay over 7 days
    
    return min(importance, 1.0)
```

## **9. Social Network Analysis**

### **9.1 Finding Key Relationships**

```python
def get_agent_key_relationships(redis_client, agent_id, threshold=0.3, limit=5):
    """Find the most important relationships for an agent."""
    # Get agents with relationship strength above threshold
    relationships = redis_client.zrangebyscore(
        f"agent:{agent_id}:relationships",
        threshold,
        float('inf'),
        withscores=True
    )
    
    # Sort by strength and limit
    relationships.sort(key=lambda x: x[1], reverse=True)
    
    # Format results
    result = [
        {"agent_id": r[0].decode() if isinstance(r[0], bytes) else r[0], 
         "strength": r[1]}
        for r in relationships[:limit]
    ]
    
    return result
```

### **9.2 Identifying Social Groups**

```python
def identify_social_groups(redis_client, simulation_id, min_strength=0.5):
    """Identify social groups based on relationship strengths."""
    # Get all agents in simulation
    agent_ids = redis_client.smembers(f"simulation:{simulation_id}:agents")
    
    # Build adjacency list of strong relationships
    adjacency = {}
    for agent_id in agent_ids:
        agent_id = agent_id.decode() if isinstance(agent_id, bytes) else agent_id
        relationships = redis_client.zrangebyscore(
            f"agent:{agent_id}:relationships",
            min_strength,
            float('inf'),
            withscores=True
        )
        
        adjacency[agent_id] = [
            r[0].decode() if isinstance(r[0], bytes) else r[0]
            for r in relationships
        ]
    
    # Use simple connected components algorithm to find groups
    groups = []
    visited = set()
    
    for agent_id in adjacency:
        if agent_id in visited:
            continue
            
        # BFS to find connected component
        group = []
        queue = [agent_id]
        visited.add(agent_id)
        
        while queue:
            current = queue.pop(0)
            group.append(current)
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(group) > 1:  # Only consider groups with at least 2 agents
            groups.append(group)
    
    return groups
```

### **9.3 Interaction History Analysis**

```python
def analyze_interaction_history(self, other_agent_id=None, length=5):
    """Analyze recent interaction history, optionally with a specific agent."""
    if not self.use_memory:
        return None
    
    interactions = []
    
    if other_agent_id:
        # Get interactions with specific agent
        interaction_ids = self.memory_client.zrevrange(
            f"agent:{self.agent_id}:interactions:with:{other_agent_id}",
            0,
            length - 1
        )
        
        for interaction_id in interaction_ids:
            interaction_hash = self.memory_client.hgetall(f"interaction:{interaction_id}")
            if interaction_hash:
                interactions.append(convert_hash_to_interaction(interaction_hash))
    else:
        # Get recent interactions using relative index
        for i in range(length):
            interaction = get_interaction_by_relative_position(self.memory_client, self.agent_id, -i)
            if interaction:
                interactions.append(interaction)
            else:
                break
    
    if not interactions:
        return None
    
    # Analyze interaction data
    interaction_types = {}
    outcomes = {}
    resources_transferred = 0
    
    for interaction in interactions:
        # Count by type
        interaction_type = interaction["interaction_type"]
        interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
        
        # Count by outcome
        if "outcome" in interaction:
            outcome = interaction["outcome"]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        # Sum resource transfers
        if "resources_transferred" in interaction:
            resources_transferred += interaction["resources_transferred"]
    
    # Determine most common type and outcome
    most_common_type = max(interaction_types.items(), key=lambda x: x[1])[0] if interaction_types else None
    most_common_outcome = max(outcomes.items(), key=lambda x: x[1])[0] if outcomes else None
    
    # Detect any trend in outcomes
    trend = None
    if len(interactions) >= 3:
        if all(i.get("outcome") == "successful" for i in interactions[:3]):
            trend = "positive"
        elif all(i.get("outcome") in ["rejected", "conflict"] for i in interactions[:3]):
            trend = "negative"
    
    return {
        "interaction_count": len(interactions),
        "interaction_types": interaction_types,
        "most_common_type": most_common_type,
        "outcomes": outcomes,
        "most_common_outcome": most_common_outcome,
        "total_resources_transferred": resources_transferred,
        "trend": trend,
        "interactions": interactions
    }
```

## **10. Performance Considerations**

### **10.1 Memory Usage**

Approximate memory usage for interaction storage:
- Base interaction data: ~250 bytes per interaction × 1,000 = 250 KB
- Indices: ~150 bytes per interaction × 1,000 = 150 KB
- Relationship graph: ~30 bytes per relationship × 100 = 3 KB
- Total per agent: ~400 KB for 1,000 interactions

### **10.2 Index Optimization**

For large simulations with many interactions, consider:

1. **Selective indexing**: Only create indices for the most common query patterns
2. **Time-based pruning**: Aggressively move old interactions to IM tier
3. **Relationship-based summarization**: For long-running relationships, store periodic summaries instead of all interactions

### **10.3 Group Interaction Aggregation**

For group interactions, store aggregate statistics to reduce memory usage:

```python
def aggregate_group_interactions(redis_client, group_id, time_window=1000):
    """Create aggregate statistics for group interactions over a time window."""
    # Get all interactions for this group
    interaction_ids = redis_client.smembers(f"group:{group_id}:interactions")
    
    # Group by time windows
    windows = {}
    
    for interaction_id in interaction_ids:
        interaction = redis_client.hgetall(f"interaction:{interaction_id}")
        if not interaction:
            continue
            
        step = int(interaction["step_number"])
        window_key = step // time_window
        
        if window_key not in windows:
            windows[window_key] = {
                "count": 0,
                "types": {},
                "outcomes": {},
                "resources_total": 0
            }
        
        # Update counts
        windows[window_key]["count"] += 1
        
        # Update type counts
        interaction_type = interaction["interaction_type"]
        windows[window_key]["types"][interaction_type] = (
            windows[window_key]["types"].get(interaction_type, 0) + 1
        )
        
        # Update outcome counts if present
        if "outcome" in interaction:
            outcome = interaction["outcome"]
            windows[window_key]["outcomes"][outcome] = (
                windows[window_key]["outcomes"].get(outcome, 0) + 1
            )
        
        # Sum resources if present
        if "resources_transferred" in interaction:
            windows[window_key]["resources_total"] += float(interaction["resources_transferred"])
    
    # Store aggregates
    for window_key, stats in windows.items():
        time_start = window_key * time_window
        time_end = time_start + time_window - 1
        
        # Store as JSON
        redis_client.hset(
            f"group:{group_id}:interaction_stats",
            f"{time_start}-{time_end}",
            json.dumps(stats)
        )
    
    return windows
```

### **10.4 Relative Time Index Operations**

```python
def update_interaction_relative_index(redis_client, agent_id, interaction_id, step_number, max_size=20):
    """Update the relative time index with a new interaction."""
    # Add the interaction to the relative index
    redis_client.zadd(f"agent:{agent_id}:interactions:relative_index", {interaction_id: step_number})
    
    # Trim the index to maintain only the most recent interactions
    redis_client.zremrangebyrank(f"agent:{agent_id}:interactions:relative_index", 0, -max_size-1)

def get_interaction_by_relative_position(redis_client, agent_id, relative_position):
    """Get an interaction by its relative position (0 = current, -1 = previous, etc.)."""
    # Convert relative position to rank
    rank = -1 - relative_position if relative_position <= 0 else relative_position
    
    # Get the interaction ID at the specified rank
    interaction_ids = redis_client.zrange(
        f"agent:{agent_id}:interactions:relative_index", 
        rank, 
        rank
    )
    
    if not interaction_ids:
        return None
    
    interaction_id = interaction_ids[0]
    
    # Retrieve the interaction data
    interaction_data = redis_client.hgetall(f"interaction:{interaction_id}")
    return convert_hash_to_interaction(interaction_data) if interaction_data else None 