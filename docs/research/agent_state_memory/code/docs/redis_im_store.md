# **RedisIMStore Documentation**

## **1. Overview**

The `RedisIMStore` is a Redis-based implementation of the Intermediate Memory (IM) storage tier in the agent memory system. It provides medium-resolution storage with TTL-based expiration for memories that have transitioned from Short-Term Memory but aren't yet ready for Long-Term Memory storage.

## **2. Key Features**

- **Level 1 Compression**: Enforces medium-resolution storage with level 1 compression
- **TTL-Based Expiration**: Automatic expiry of memories after configured duration (default 7 days)
- **Multi-Index Design**: Maintains specialized indices for different query patterns
- **Resilient Redis Operations**: Uses circuit breaker pattern and retry mechanisms
- **Access-Based Importance**: Updates importance scores based on retrieval patterns
- **Comprehensive Error Handling**: Graceful degradation under failure conditions

## **3. Class Structure**

### **3.1 RedisIMStore**

```python
class RedisIMStore:
    def __init__(self, config: RedisIMConfig):
        """Initialize the Redis IM store.
        
        Args:
            config: Configuration for IM Redis storage
        """
```

### **3.2 Configuration (RedisIMConfig)**

```python
@dataclass
class RedisIMConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 1  # IM uses database 1
    password: Optional[str] = None
    
    # Memory settings
    ttl: int = 604800  # 7 days
    memory_limit: int = 10000  # Max entries per agent
    compression_level: int = 1  # Level 1 compression
    
    namespace: str = "agent_memory:im"
```

## **4. Key Methods**

### **4.1 Memory Storage and Retrieval**

| Method | Purpose |
|--------|---------|
| `store(agent_id, memory_entry, priority)` | Store a level 1 compressed memory entry |
| `get(agent_id, memory_id)` | Retrieve a memory entry by ID |
| `get_by_timerange(agent_id, start_time, end_time, limit)` | Retrieve memories within a time range |
| `get_by_importance(agent_id, min_importance, max_importance, limit)` | Retrieve memories by importance score |
| `delete(agent_id, memory_id)` | Delete a memory entry |
| `count(agent_id)` | Get the number of memories for an agent |
| `clear(agent_id)` | Clear all memories for an agent |

### **4.2 Internal Methods**

| Method | Purpose |
|--------|---------|
| `_store_memory_entry(agent_id, memory_entry)` | Internal storage implementation |
| `_update_access_metadata(agent_id, memory_id, memory_entry)` | Update access statistics and importance |

## **5. Data Organization**

### **5.1 Redis Key Structure**

| Key Pattern | Purpose |
|-------------|---------|
| `{namespace}:{agent_id}:memory:{memory_id}` | Individual memory entries |
| `{namespace}:{agent_id}:memories` | Sorted set of all agent memories |
| `{namespace}:{agent_id}:timeline` | Chronological index |
| `{namespace}:{agent_id}:importance` | Importance score index |

### **5.2 Memory Entry Structure**

```json
{
    "memory_id": "unique-id",
    "agent_id": "agent-id",
    "timestamp": 1234567890,
    "contents": {
        // Level 1 compressed memory contents
    },
    "metadata": {
        "compression_level": 1,
        "importance_score": 0.5,
        "retrieval_count": 0,
        "creation_time": 1234567890,
        "last_access_time": 1234567890
    }
}
```

## **6. Error Handling**

### **6.1 Redis Operation Errors**

- Uses `ResilientRedisClient` with circuit breaker pattern
- Retries failed operations based on priority
- Graceful degradation when Redis is unavailable

### **6.2 Data Validation**

- Enforces level 1 compression requirement
- Validates memory entry structure
- Ensures proper TTL settings on all keys

## **7. Performance Considerations**

### **7.1 Memory Usage**

- Maintains medium-resolution data with level 1 compression
- Automatic TTL-based cleanup
- Configurable memory limits per agent

### **7.2 Access Patterns**

- Optimized for time-based queries
- Efficient importance-based retrieval
- Access frequency affects importance scores

### **7.3 Optimization Strategies**

- Redis pipelining for batch operations
- TTL on indices to prevent orphaned data
- Automatic importance score adjustments

## **8. Integration Points**

### **8.1 Memory Transition**

- Accepts memories transitioned from STM
- Verifies compression level before storage
- Updates metadata during transitions

### **8.2 Importance Scoring**

- Base importance from memory creation
- Dynamic updates based on access patterns
- Influences memory retention decisions

## **9. Best Practices**

1. **Memory Entry Validation**
   - Always verify compression level
   - Ensure required fields are present
   - Validate importance scores (0.0-1.0)

2. **Error Handling**
   - Handle Redis connection issues gracefully
   - Log errors with appropriate context
   - Use retry mechanisms for critical operations

3. **Performance Optimization**
   - Monitor memory usage
   - Implement batch operations where possible
   - Regular maintenance of indices

## **10. See Also**

- [Memory Tiers](memory_tiers.md)
- [Redis STM Store](redis_stm_store.md)
- [SQLite LTM Store](sqlite_ltm_store.md)
- [Agent Memory System](agent_memory_system.md) 