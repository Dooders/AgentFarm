# **Agent Action Redis Schema**

## **1. Introduction**

This document defines the Redis schema for storing and retrieving agent action information within the AgentMemory system. It provides specific data structures and access patterns optimized for the existing ActionModel database schema and action tracking requirements.

## **2. Action Structure Alignment**

This schema aligns with the existing action structures defined in:

1. **`farm/database/models.py:ActionModel`** - SQLite database model for actions
2. **`farm/agents/base_agent.py:act()`** - Action execution method
3. **`farm/database/data_logging.py:log_agent_action()`** - Action logging method

## **3. Redis Schema for Agent Actions**

### **3.1 Key Structure**

All agent action keys follow these patterns:

| Pattern | Purpose | Example |
|---------|---------|---------|
| `agent:{agent_id}:action:{action_id}` | Individual action records | `agent:a123:action:1000` |
| `agent:{agent_id}:action:timeline` | Chronological action index | `agent:a123:action:timeline` |
| `agent:{agent_id}:action:types:{action_type}` | Actions by type | `agent:a123:action:types:move` |
| `agent:{agent_id}:action:targets:{target_id}` | Actions by target | `agent:a123:action:targets:a456` |
| `simulation:{sim_id}:actions` | Actions in a simulation | `simulation:sim01:actions` |
| `simulation:{sim_id}:actions:{step}` | Actions by step | `simulation:sim01:actions:1050` |

### **3.2 Primary Action Hash Structure**

Each agent action is stored as a Redis hash with normalized field names matching the ActionModel schema:

```
agent:{agent_id}:action:{action_id} (Hash)
```

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `action_id` | String | Unique action identifier | `{agent_id}-{step}-{type}` |
| `simulation_id` | String | Simulation identifier | ActionModel.simulation_id |
| `step_number` | Integer | Step when action occurred | ActionModel.step_number |
| `agent_id` | String | Agent that performed action | ActionModel.agent_id |
| `action_type` | String | Type of action performed | ActionModel.action_type |
| `action_target_id` | String | Target of action (if any) | ActionModel.action_target_id |
| `state_before_id` | String | Agent state before action | ActionModel.state_before_id |
| `state_after_id` | String | Agent state after action | ActionModel.state_after_id |
| `resources_before` | Float | Resources before action | ActionModel.resources_before |
| `resources_after` | Float | Resources after action | ActionModel.resources_after |
| `reward` | Float | Reward received | ActionModel.reward |
| `details` | String | JSON-encoded action details | ActionModel.details |
| `position_x` | Float | X position when action occurred | From state data |
| `position_y` | Float | Y position when action occurred | From state data |
| `embedding` | Binary | Action embedding vector | Generated from action data |
| `stored_at` | Integer | Timestamp of storage | System time when stored |

### **3.3 Index Structures**

#### **Timeline Index**

```
agent:{agent_id}:action:timeline (Sorted Set)
```

Maps action_ids to step numbers for time-based queries:
- Score: `step_number`
- Member: `action_id`

#### **Action Type Index**

```
agent:{agent_id}:action:types:{action_type} (Set)
```

For action type-based queries, where `action_type` is the type of action (e.g., "move", "attack", "consume"):
- Members: action_ids of that type

#### **Target Index**

```
agent:{agent_id}:action:targets:{target_id} (Set)
```

For target-based queries, where `target_id` is the target agent ID:
- Members: action_ids targeting that agent

#### **Reward Index**

```
agent:{agent_id}:action:rewards (Sorted Set)
```

For reward-based queries:
- Score: normalized reward value
- Member: action_id

### **3.4 Simulation-Level Indices**

```
simulation:{sim_id}:actions:steps (Sorted Set)
```

For tracking steps with actions across the simulation:
- Score: step number
- Member: step number (used as a uniqueness constraint)

```
simulation:{sim_id}:actions:by_type (Hash)
```

For counting action types across the simulation:
- Field: action type
- Value: count of actions of that type

### **3.5 Relative Time Index**

```
agent:{agent_id}:action:relative_index (Sorted Set)
```

This index provides efficient access to actions by their relative temporal position (0 is the most recent action, -1 is the previous action, etc.):
- Score: step_number
- Member: action_id

The relative index is maintained with each new action storage operation:

```python
def update_action_relative_index(redis_client, agent_id, action_id, step_number, max_size=20):
    """Update the relative time index with a new action."""
    # Add the action to the relative index
    redis_client.zadd(f"agent:{agent_id}:action:relative_index", {action_id: step_number})
    
    # Trim the index to maintain only the most recent actions
    redis_client.zremrangebyrank(f"agent:{agent_id}:action:relative_index", 0, -max_size-1)

def get_action_by_relative_position(redis_client, agent_id, relative_position):
    """Get an action by its relative position (0 = current, -1 = previous, etc.)."""
    # Convert relative position to rank
    rank = -1 - relative_position if relative_position <= 0 else relative_position
    
    # Get the action ID at the specified rank
    action_ids = redis_client.zrange(
        f"agent:{agent_id}:action:relative_index", 
        rank, 
        rank
    )
    
    if not action_ids:
        return None
    
    action_id = action_ids[0]
    
    # Retrieve the action data
    action_data = redis_client.hgetall(f"agent:{agent_id}:action:{action_id}")
    return convert_hash_to_action(action_data) if action_data else None
```

## **4. Storage Operations**

### **4.1 Storing a New Action**

```python
def store_agent_action(
    redis_client, 
    agent_id, 
    simulation_id, 
    step_number, 
    action_type, 
    action_target_id=None, 
    state_before_id=None,
    state_after_id=None,
    resources_before=None,
    resources_after=None,
    reward=None,
    details=None,
    position=None
):
    """Store an agent action in Redis."""
    # Generate unique action ID
    action_id = f"{agent_id}-{step_number}-{action_type}"
    
    # Convert action to hash fields
    action_hash = {
        "action_id": action_id,
        "simulation_id": simulation_id,
        "step_number": step_number,
        "agent_id": agent_id,
        "action_type": action_type,
        "stored_at": int(time.time())
    }
    
    # Add optional fields
    if action_target_id:
        action_hash["action_target_id"] = action_target_id
    if state_before_id:
        action_hash["state_before_id"] = state_before_id
    if state_after_id:
        action_hash["state_after_id"] = state_after_id
    if resources_before is not None:
        action_hash["resources_before"] = resources_before
    if resources_after is not None:
        action_hash["resources_after"] = resources_after
    if reward is not None:
        action_hash["reward"] = reward
    if details:
        action_hash["details"] = json.dumps(details)
    if position:
        action_hash["position_x"] = position[0]
        action_hash["position_y"] = position[1]
    
    # Generate action embedding if needed for similarity search
    if hasattr(action_type, "to_embedding"):
        embedding = action_type.to_embedding().tobytes()
        action_hash["embedding"] = embedding
    
    # Store action hash
    redis_client.hset(f"agent:{agent_id}:action:{action_id}", mapping=action_hash)
    
    # Add to timeline index
    redis_client.zadd(f"agent:{agent_id}:action:timeline", {action_id: step_number})
    
    # Add to relative time index
    update_action_relative_index(redis_client, agent_id, action_id, step_number)
    
    # Add to action type index
    redis_client.sadd(f"agent:{agent_id}:action:types:{action_type}", action_id)
    
    # Add to target index if target exists
    if action_target_id:
        redis_client.sadd(f"agent:{agent_id}:action:targets:{action_target_id}", action_id)
    
    # Add to reward index if reward exists
    if reward is not None:
        redis_client.zadd(f"agent:{agent_id}:action:rewards", {action_id: reward})
    
    # Add to simulation indices
    redis_client.rpush(f"simulation:{simulation_id}:actions", action_id)
    redis_client.sadd(f"simulation:{simulation_id}:actions:{step_number}", action_id)
    redis_client.zadd(f"simulation:{simulation_id}:actions:steps", {step_number: step_number})
    redis_client.hincrby(f"simulation:{simulation_id}:actions:by_type", action_type, 1)
    
    # Set TTL for action in STM
    redis_client.expire(f"agent:{agent_id}:action:{action_id}", ACTION_STM_TTL)
    
    return action_id
```

### **4.2 Retrieving Actions by Type**

```python
def get_agent_actions_by_type(redis_client, agent_id, action_type, limit=None):
    """Retrieve agent actions of a specific type."""
    # Get action IDs of the specified type
    action_ids = redis_client.smembers(f"agent:{agent_id}:action:types:{action_type}")
    
    # Sort by step number if needed
    if limit:
        # Get step numbers for each action
        actions_with_steps = []
        for action_id in action_ids:
            step = redis_client.hget(f"agent:{agent_id}:action:{action_id}", "step_number")
            if step:
                actions_with_steps.append((action_id, int(step)))
        
        # Sort by step number (most recent first) and limit
        actions_with_steps.sort(key=lambda x: x[1], reverse=True)
        action_ids = [a[0] for a in actions_with_steps[:limit]]
    
    # Retrieve each action
    actions = []
    for action_id in action_ids:
        action_hash = redis_client.hgetall(f"agent:{agent_id}:action:{action_id}")
        if action_hash:
            actions.append(convert_hash_to_action(action_hash))
    
    return actions
```

### **4.3 Retrieving High-Reward Actions**

```python
def get_agent_high_reward_actions(redis_client, agent_id, min_reward=0.5, limit=10):
    """Retrieve agent actions with rewards above the specified threshold."""
    # Get action IDs with rewards above threshold
    action_ids = redis_client.zrangebyscore(
        f"agent:{agent_id}:action:rewards",
        min_reward,
        float('inf'),
        start=0,
        num=limit
    )
    
    # Retrieve each action
    actions = []
    for action_id in action_ids:
        action_hash = redis_client.hgetall(f"agent:{agent_id}:action:{action_id}")
        if action_hash:
            actions.append(convert_hash_to_action(action_hash))
    
    return actions
```

## **5. Action Conversion Utilities**

### **5.1 Converting Hash to Action Object**

```python
def convert_hash_to_action(action_hash):
    """Convert Redis hash to Action object."""
    # Parse details JSON if present
    details = None
    if "details" in action_hash:
        try:
            details = json.loads(action_hash["details"])
        except:
            details = action_hash["details"]
    
    # Create action object (can be a simple dict or custom class)
    action = {
        "action_id": action_hash["action_id"],
        "simulation_id": action_hash["simulation_id"],
        "step_number": int(action_hash["step_number"]),
        "agent_id": action_hash["agent_id"],
        "action_type": action_hash["action_type"],
        "action_target_id": action_hash.get("action_target_id"),
        "state_before_id": action_hash.get("state_before_id"),
        "state_after_id": action_hash.get("state_after_id"),
        "resources_before": float(action_hash["resources_before"]) if "resources_before" in action_hash else None,
        "resources_after": float(action_hash["resources_after"]) if "resources_after" in action_hash else None,
        "reward": float(action_hash["reward"]) if "reward" in action_hash else None,
        "details": details
    }
    
    # Add position if available
    if "position_x" in action_hash and "position_y" in action_hash:
        action["position"] = (float(action_hash["position_x"]), float(action_hash["position_y"]))
    
    return action
```

### **5.2 Agent Action Memory Integration**

```python
def remember_agent_action(self, action_type, target_id=None, reward=None, details=None):
    """Remember an action performed by the agent."""
    if not self.use_memory:
        return None
    
    # Get current state information
    current_state = self.get_state()
    
    # Store in Redis
    action_id = store_agent_action(
        redis_client=self.memory_client,
        agent_id=self.agent_id,
        simulation_id=self.environment.simulation_id,
        step_number=self.environment.time,
        action_type=action_type,
        action_target_id=target_id,
        resources_before=self.resource_level,
        reward=reward,
        details=details,
        position=(self.position_x, self.position_y)
    )
    
    return action_id
```

## **6. Memory Tier Transition**

```python
def transition_action_to_intermediate_memory(redis_client, agent_id, action_id):
    """Move action from STM to IM with compression."""
    # Get original action
    action_key = f"agent:{agent_id}:action:{action_id}"
    action_hash = redis_client.hgetall(action_key)
    
    if not action_hash:
        return False
    
    # Calculate importance score
    importance = calculate_action_importance(action_hash)
    
    # Create compressed version (fewer fields)
    compressed_hash = {
        "action_id": action_hash["action_id"],
        "simulation_id": action_hash["simulation_id"],
        "step_number": action_hash["step_number"],
        "agent_id": action_hash["agent_id"],
        "action_type": action_hash["action_type"],
        "reward": action_hash.get("reward", "0"),
        "importance": str(importance)
    }
    
    # Add target ID if it exists (important for social interactions)
    if "action_target_id" in action_hash:
        compressed_hash["action_target_id"] = action_hash["action_target_id"]
    
    # Store in IM
    im_key = f"agent:{agent_id}:im:action:{action_id}"
    redis_client.hset(im_key, mapping=compressed_hash)
    
    # Add to IM timeline
    redis_client.zadd(
        f"agent:{agent_id}:im:action:timeline", 
        {action_id: int(action_hash["step_number"])}
    )
    
    # Set TTL for intermediate memory
    redis_client.expire(im_key, ACTION_IM_TTL)
    
    # Remove from STM
    redis_client.delete(action_key)
    
    # Remove from STM indices
    redis_client.zrem(f"agent:{agent_id}:action:timeline", action_id)
    redis_client.srem(f"agent:{agent_id}:action:types:{action_hash['action_type']}", action_id)
    
    if "action_target_id" in action_hash:
        redis_client.srem(
            f"agent:{agent_id}:action:targets:{action_hash['action_target_id']}", 
            action_id
        )
    
    if "reward" in action_hash:
        redis_client.zrem(f"agent:{agent_id}:action:rewards", action_id)
    
    return True
```

## **7. Action Importance Calculation**

```python
def calculate_action_importance(action_hash):
    """Calculate importance score for action retention decisions."""
    importance = 0.0
    
    # Reward-based importance
    if "reward" in action_hash:
        reward = float(action_hash["reward"])
        # Higher absolute rewards (positive or negative) are more important
        importance += min(abs(reward) * 0.5, 0.5)
    
    # Action type importance
    action_type = action_hash["action_type"]
    if action_type in ["attack", "reproduce", "share"]:
        # Social interactions are more important
        importance += 0.3
    
    # Target-based importance
    if "action_target_id" in action_hash:
        # Actions involving other agents are more important
        importance += 0.2
    
    # Resource change importance
    if "resources_before" in action_hash and "resources_after" in action_hash:
        resources_before = float(action_hash["resources_before"])
        resources_after = float(action_hash["resources_after"])
        resource_change = abs(resources_after - resources_before)
        
        # Significant resource changes are more important
        if resource_change > 0.2:
            importance += 0.2
    
    return min(importance, 1.0)
```

## **8. Query Patterns for Agent Behavior**

### **8.1 Learning From Past Actions**

```python
def learn_from_past_actions(self, action_type, current_position=None, limit=5):
    """Learn from past actions of a specific type."""
    if not self.use_memory:
        return None
    
    # Get past actions of this type
    past_actions = get_agent_actions_by_type(
        redis_client=self.memory_client,
        agent_id=self.agent_id,
        action_type=action_type,
        limit=limit
    )
    
    if not past_actions:
        return None
    
    # Filter by position similarity if position provided
    if current_position:
        # Calculate distances
        for action in past_actions:
            if "position" in action:
                action["distance"] = (
                    (action["position"][0] - current_position[0]) ** 2 +
                    (action["position"][1] - current_position[1]) ** 2
                ) ** 0.5
            else:
                action["distance"] = float('inf')
        
        # Filter to nearby actions
        nearby_actions = [a for a in past_actions if a.get("distance", float('inf')) < 0.2]
        if nearby_actions:
            past_actions = nearby_actions
    
    # Calculate average reward
    rewards = [a["reward"] for a in past_actions if a["reward"] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    
    # Find best action
    best_action = max(past_actions, key=lambda a: a["reward"] if a["reward"] is not None else float('-inf'))
    
    return {
        "avg_reward": avg_reward,
        "best_action": best_action,
        "action_count": len(past_actions)
    }
```

### **8.2 Finding Successful Interactions With Target**

```python
def find_successful_interactions_with(self, target_id, limit=3):
    """Find past successful interactions with a specific target agent."""
    if not self.use_memory:
        return []
    
    # Get actions targeting this agent
    action_ids = self.memory_client.smembers(f"agent:{self.agent_id}:action:targets:{target_id}")
    
    # Retrieve each action
    actions = []
    for action_id in action_ids:
        action_hash = self.memory_client.hgetall(f"agent:{self.agent_id}:action:{action_id}")
        if action_hash and "reward" in action_hash:
            # Only include positive reward actions
            if float(action_hash["reward"]) > 0:
                actions.append(convert_hash_to_action(action_hash))
    
    # Sort by reward and limit
    actions.sort(key=lambda a: a["reward"] if a["reward"] is not None else 0, reverse=True)
    return actions[:limit]
```

### **8.3 Action History Analysis**

```python
def analyze_action_sequence(self, length=5):
    """Analyze the recent sequence of actions to identify patterns."""
    if not self.use_memory:
        return None
        
    # Get recent actions using relative index
    actions = []
    for i in range(length):
        action = get_action_by_relative_position(self.memory_client, self.agent_id, -i)
        if action:
            actions.append(action)
        else:
            break
            
    if not actions:
        return None
        
    # Count action types
    action_counts = {}
    for action in actions:
        action_type = action["action_type"]
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
    # Calculate average reward
    rewards = [a["reward"] for a in actions if a["reward"] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    
    # Detect cycles
    cycle_detected = False
    if len(actions) >= 4:
        # Check for ABAB pattern
        if (actions[0]["action_type"] == actions[2]["action_type"] and
            actions[1]["action_type"] == actions[3]["action_type"]):
            cycle_detected = True
            
    return {
        "action_counts": action_counts,
        "most_common": max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None,
        "avg_reward": avg_reward,
        "cycle_detected": cycle_detected,
        "sequence": [a["action_type"] for a in actions]
    }
```

## **9. Performance Considerations**

### **9.1 Memory Usage**

Approximate memory usage for action storage:
- Base action data: ~150 bytes per action × 1,000 = 150 KB
- Indices: ~80 bytes per action × 1,000 = 80 KB
- Total per agent: ~230 KB for 1,000 actions

### **9.2 Cleanup and Maintenance**

```python
def cleanup_stale_action_indices(redis_client, agent_id):
    """Remove orphaned action indices where the primary action is gone."""
    # Get all valid action IDs from timeline
    valid_ids = set(redis_client.zrange(f"agent:{agent_id}:action:timeline", 0, -1))
    
    # Check action type indices
    action_types = [
        t.decode().split(":")[-1] 
        for t in redis_client.keys(f"agent:{agent_id}:action:types:*")
    ]
    
    for action_type in action_types:
        # Get all actions of this type
        type_key = f"agent:{agent_id}:action:types:{action_type}"
        type_ids = set(redis_client.smembers(type_key))
        
        # Find orphaned IDs
        orphaned = type_ids - valid_ids
        if orphaned:
            redis_client.srem(type_key, *orphaned)
    
    # Check target indices similarly
    # ... and other indices
``` 