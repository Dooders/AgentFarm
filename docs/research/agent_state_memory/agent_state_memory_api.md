# **Agent State Memory API Specification**

## **1. Introduction**

This document formalizes the API for the Agent State Memory system, providing interfaces for storing, retrieving, and managing agent states. The API is designed to work with the hierarchical memory architecture described in [Core Concepts](core_concepts.md).

For details on memory structure, data formats, and architectural concepts, please refer to:
- [Hierarchical Memory Architecture](core_concepts.md#2-hierarchical-memory-architecture)
- [Memory Entry Structure](core_concepts.md#3-memory-entry-structure)
- [Memory Retrieval Methods](core_concepts.md#5-memory-retrieval-methods)

## **2. Core Data API**

The primary interface for storing and retrieving agent memory states:

```python
class AgentStateMemoryAPI:
    """Interface for storing and retrieving agent states in the hierarchical memory system."""
    
    def store_agent_state(self, agent_id, state_data, step_number):
        """
        Store an agent's state in short-term memory.
        
        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
        """
        pass
        
    def store_agent_interaction(self, agent_id, interaction_data, step_number):
        """
        Store information about an agent's interaction with environment or other agents.
        
        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
        """
        pass
        
    def store_agent_action(self, agent_id, action_data, step_number):
        """
        Store information about an action taken by an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary containing action details (type, parameters, context, outcome)
            step_number: Current simulation step number
        """
        pass
        
    def retrieve_similar_states(self, agent_id, query_state, k=5, memory_type=None):
        """
        Retrieve most similar past states to the provided query state.
        
        Args:
            agent_id: Unique identifier for the agent
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries sorted by similarity to query state
        """
        pass
        
    def retrieve_by_time_range(self, agent_id, start_step, end_step, memory_type=None):
        """
        Retrieve memories within a specific time/step range.
        
        Args:
            agent_id: Unique identifier for the agent
            start_step: Beginning of time range
            end_step: End of time range
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries within the specified time range
        """
        pass
        
    def retrieve_by_attributes(self, agent_id, attributes, memory_type=None):
        """
        Retrieve memories matching specific attribute values.
        
        Args:
            agent_id: Unique identifier for the agent
            attributes: Dictionary of attribute-value pairs to match
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries matching the specified attributes
        """
        pass
        
    def get_memory_statistics(self, agent_id):
        """
        Get statistics about an agent's memory usage.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dictionary containing memory statistics
        """
        pass
```

## **3. State Change Tracking API**

Interface for tracking changes to agent states over time:

```python
class StateChangeTrackingAPI:
    """Interface for tracking changes to agent states over time."""
    
    def get_attribute_change_history(self, agent_id, attribute_name, start_step=None, end_step=None):
        """
        Get history of changes for a specific attribute.
        
        Args:
            agent_id: Unique identifier for the agent
            attribute_name: Name of the attribute to track
            start_step: Optional start step for filtering
            end_step: Optional end step for filtering
            
        Returns:
            List of change records for the specified attribute
        """
        pass
        
    def get_significant_changes(self, agent_id, magnitude_threshold=0.3, start_step=None, end_step=None):
        """
        Find significant state changes based on magnitude.
        
        Args:
            agent_id: Unique identifier for the agent
            magnitude_threshold: Minimum change magnitude to consider significant
            start_step: Optional start step for filtering
            end_step: Optional end step for filtering
            
        Returns:
            List of significant change records sorted by time
        """
        pass
        
    def get_agent_change_statistics(self, agent_id):
        """
        Calculate statistics about an agent's state changes.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dictionary containing change statistics including most volatile attributes
        """
        pass
```

## **4. Memory Management API**

Interface for managing the agent memory system:

```python
class MemoryManagementAPI:
    """Interface for managing the agent memory system."""
    
    def force_memory_maintenance(self, agent_id=None):
        """
        Force memory tier transitions and cleanup operations.
        
        Args:
            agent_id: Optional agent ID to restrict maintenance to a single agent
        """
        pass
        
    def clear_agent_memory(self, agent_id, memory_tiers=None):
        """
        Clear an agent's memory in specified tiers.
        
        Args:
            agent_id: Identifier for the agent
            memory_tiers: Optional list of tiers to clear (e.g., ["stm", "im"])
                          If None, clears all tiers
        """
        pass
        
    def set_importance_score(self, agent_id, memory_id, importance_score):
        """
        Update the importance score for a specific memory.
        
        Args:
            agent_id: Identifier for the agent
            memory_id: Unique identifier for the memory entry
            importance_score: New importance score (0.0 to 1.0)
        """
        pass
        
    def configure_memory_system(self, config):
        """
        Update configuration parameters for the memory system.
        
        Args:
            config: Dictionary of configuration parameters
        """
        pass
```

## **5. Memory Query API**

Interface for advanced memory queries:

```python
class MemoryQueryAPI:
    """Interface for advanced memory queries."""
    
    def search_by_embedding(self, agent_id, query_embedding, k=5, memory_tiers=None):
        """
        Find memories by raw embedding vector similarity.
        
        Args:
            agent_id: Identifier for the agent
            query_embedding: Embedding vector to search with
            k: Number of results to return
            memory_tiers: Optional list of tiers to search
            
        Returns:
            List of memory entries sorted by similarity
        """
        pass
        
    def search_by_content(self, agent_id, content_query, k=5):
        """
        Search for memories based on content text/attributes.
        
        Args:
            agent_id: Identifier for the agent
            content_query: String or dict to search for in memory contents
            k: Number of results to return
            
        Returns:
            List of memory entries matching the content query
        """
        pass
        
    def get_memory_snapshots(self, agent_id, steps):
        """
        Get agent memory snapshots at specific steps.
        
        Args:
            agent_id: Identifier for the agent
            steps: List of step numbers to get snapshots for
            
        Returns:
            Dictionary mapping step numbers to memory snapshots
        """
        pass
```

## **6. Implementation Classes**

The API interfaces are implemented by the following classes:

### **6.1 RedisAgentMemoryManager**

Primary implementation class that leverages Redis for STM and IM tiers:

```python
class RedisAgentMemoryManager(AgentStateMemoryAPI, StateChangeTrackingAPI, MemoryManagementAPI, MemoryQueryAPI):
    """Redis-based implementation of the Agent State Memory API."""
    
    def __init__(self, config=None):
        self.config = config or DefaultMemoryConfig()
        self.redis_client = redis.Redis(**self.config.redis_connection_params)
        self.sqlite_connection = sqlite3.connect(self.config.sqlite_path)
        self.initialize_storage()
        
    def initialize_storage(self):
        """Set up the necessary Redis data structures and SQLite tables."""
        # Create SQLite tables for LTM if they don't exist
        # Set up Redis indices and data structures
        pass
        
    # ... API method implementations ...
```

### **6.2 Configuration**

Configuration class for the memory system:

```python
@dataclass
class MemoryConfig:
    """Configuration for the agent memory system."""
    # Memory tier settings
    stm_capacity: int = 1000
    im_ttl: int = 3600  # 1 hour in seconds
    
    # Redis connection settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # SQLite settings
    sqlite_path: str = "agent_memory.sqlite"
    
    # Compression settings
    stm_vector_dim: int = 512
    im_vector_dim: int = 128
    ltm_vector_dim: int = 32
    
    @property
    def redis_connection_params(self):
        """Get Redis connection parameters."""
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db
        }
```

## **7. Usage Examples**

### **7.1 Basic Store and Retrieve**

```python
# Initialize the memory system
memory_api = RedisAgentMemoryManager()

# Store an agent state
memory_api.store_agent_state(
    agent_id="agent-123",
    state_data={
        "position": [10, 20],
        "resources": 150,
        "health": 0.75,
        "observations": ["obstacle_ahead", "resource_nearby"]
    },
    step_number=1234
)

# Retrieve similar states
similar_states = memory_api.retrieve_similar_states(
    agent_id="agent-123",
    query_state={"position": [11, 19], "resources": 140},
    k=3
)
```

### **7.2 Tracking State Changes**

```python
# Get attribute history
health_history = memory_api.get_attribute_change_history(
    agent_id="agent-123",
    attribute_name="health",
    start_step=1000,
    end_step=2000
)

# Find significant changes
significant_changes = memory_api.get_significant_changes(
    agent_id="agent-123",
    magnitude_threshold=0.2
)
```

### **7.3 Storing and Retrieving Action States**

```python
# Store an agent action
memory_api.store_agent_action(
    agent_id="agent-123",
    action_data={
        "action_type": "move",
        "action_params": {"direction": "north", "distance": 5},
        "initial_context": {
            "position": [10, 20],
            "obstacles": ["tree", "rock"],
            "energy": 0.85
        },
        "outcome": {
            "success": True,
            "new_position": [10, 25],
            "energy_cost": 0.05
        }
    },
    step_number=1235
)

# Retrieve similar actions
similar_actions = memory_api.retrieve_similar_states(
    agent_id="agent-123",
    query_state={"action_type": "move", "action_params": {"direction": "north"}},
    k=3,
    memory_type="action"
)

# Analyze action outcomes
success_rate = sum(1 for action in similar_actions if action["contents"]["outcome"]["success"]) / len(similar_actions)
avg_energy_cost = sum(action["contents"]["outcome"]["energy_cost"] for action in similar_actions) / len(similar_actions)
```

## **8. Error Handling**

The API follows these error handling principles:

1. **Graceful Degradation**: When a memory tier is unavailable, fall back to available tiers
2. **Explicit Exceptions**: Raise specific exception types for different error conditions
3. **Data Validation**: Validate inputs before processing to prevent data corruption

```python
class MemoryAccessError(Exception):
    """Base exception for memory access issues."""
    pass

class MemoryTierUnavailableError(MemoryAccessError):
    """Raised when a memory tier cannot be accessed."""
    pass

class MemoryNotFoundError(MemoryAccessError):
    """Raised when a requested memory doesn't exist."""
    pass
```

---

**See Also:**
- [Core Concepts](core_concepts.md) - Fundamental architecture and data structures 
- [Memory Agent](memory_agent.md) - Memory agent implementation
- [Agent State Storage](agent_state_storage.md) - State storage implementation
- [Redis Integration](redis_integration.md) - Redis caching implementation
- [Glossary](glossary.md) - Terminology reference 