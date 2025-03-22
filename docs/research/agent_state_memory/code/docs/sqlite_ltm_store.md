# **SQLiteLTMStore Documentation**

## **1. Overview**

The `SQLiteLTMStore` is a SQLite-based implementation of the Long-Term Memory (LTM) storage tier in the agent memory system. It provides persistent, highly-compressed storage for historical agent memory entries with comprehensive error handling, efficient vector similarity search, and robust database operations.

## **2. Key Features**

- **Resilient SQLite Operations**: Contextual connection management with error categorization
- **Persistent Storage**: Disk-based storage for long-term retention of memories
- **High Compression**: Level 2 (highest) compression for efficient storage
- **Vector Similarity Search**: Native support for embedding-based memory retrieval
- **Comprehensive Error Handling**: Distinction between temporary and permanent errors
- **Transactional Operations**: ACID-compliant batch operations
- **Foreign Key Integrity**: Ensures data consistency across tables
- **Health Monitoring**: Database integrity checking and performance metrics

## **3. Class Structure**

### **3.1 SQLiteLTMStore**

```python
class SQLiteLTMStore:
    def __init__(self, agent_id: str, config: SQLiteLTMConfig):
        """Initialize the SQLite LTM store.
        
        Args:
            agent_id: ID of the agent
            config: Configuration for LTM SQLite storage
        """
        # ...
```

## **4. Key Methods**

### **4.1 Memory Storage and Retrieval**

| Method | Purpose |
|--------|---------|
| `store(memory_entry)` | Store a memory entry in LTM |
| `store_batch(memory_entries)` | Store multiple memory entries in one transaction |
| `get(memory_id)` | Retrieve a memory entry by ID |
| `get_by_timerange(start_time, end_time, limit)` | Retrieve memories within a time range |
| `get_by_importance(min_importance, max_importance, limit)` | Retrieve memories by importance score |
| `get_most_similar(query_vector, top_k)` | Find memories most similar to query vector |
| `get_all(limit)` | Get all memories for the agent |
| `count()` | Get the number of memories for an agent |
| `delete(memory_id)` | Delete a memory entry |
| `clear()` | Clear all memories for an agent |
| `check_health()` | Check SQLite database health |

### **4.2 Internal Methods**

| Method | Purpose |
|--------|---------|
| `_get_connection()` | Get a SQLite connection with error handling |
| `_init_database()` | Initialize the database schema |
| `_update_access_metadata(memory_id)` | Update access statistics |

## **5. Data Organization**

### **5.1 Memory Entry Structure**

Each memory entry is stored with this structure:

```json
{
  "memory_id": "unique-identifier",
  "agent_id": "agent-123",
  "step_number": 1234,
  "timestamp": 1679233344,
  
  "content": {
    // Agent state or action data
  },
  
  "metadata": {
    "creation_time": 1679233344,
    "last_access_time": 1679233400,
    "compression_level": 2,
    "importance_score": 0.75,
    "retrieval_count": 3,
    "memory_type": "state" 
  },
  
  "embeddings": {
    "compressed_vector": [...]  // 32d LTM embedding vector
  }
}
```

### **5.2 Database Schema**

#### **Main Memory Table**

```sql
CREATE TABLE {table_prefix}_memories (
    memory_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    
    content_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    
    compression_level INTEGER DEFAULT 2,
    importance_score REAL DEFAULT 0.0,
    retrieval_count INTEGER DEFAULT 0,
    memory_type TEXT NOT NULL,
    
    created_at INTEGER NOT NULL,
    last_accessed INTEGER NOT NULL
)
```

#### **Embeddings Table**

```sql
CREATE TABLE {table_prefix}_embeddings (
    memory_id TEXT PRIMARY KEY,
    vector_blob BLOB NOT NULL,
    vector_dim INTEGER NOT NULL,
    
    FOREIGN KEY (memory_id) REFERENCES {table_prefix}_memories (memory_id) 
    ON DELETE CASCADE
)
```

### **5.3 Database Indices**

| Index | Purpose | Implementation |
|-------|---------|----------------|
| `idx_{table_prefix}_agent_id` | Agent-based filtering | `CREATE INDEX ... ON {table_prefix}_memories (agent_id)` |
| `idx_{table_prefix}_step` | Step-based filtering | `CREATE INDEX ... ON {table_prefix}_memories (step_number)` |
| `idx_{table_prefix}_type` | Type-based filtering | `CREATE INDEX ... ON {table_prefix}_memories (memory_type)` |
| `idx_{table_prefix}_importance` | Importance-based sorting | `CREATE INDEX ... ON {table_prefix}_memories (importance_score)` |
| `idx_{table_prefix}_timestamp` | Temporal sorting | `CREATE INDEX ... ON {table_prefix}_memories (timestamp)` |

## **6. Error Handling Strategy**

### **6.1 Error Classification**

The store classifies SQLite errors into two categories:

1. **Temporary Errors** (`SQLiteTemporaryError`): 
   - Database locks and timeouts
   - Retries may succeed after waiting

2. **Permanent Errors** (`SQLitePermanentError`):
   - Database corruption
   - Structural issues requiring intervention

### **6.2 Connection Management**

The store uses Python's context manager pattern to:

1. Automatically handle connection cleanup
2. Properly categorize errors
3. Enforce transaction boundaries
4. Set consistent connection parameters

### **6.3 Error Recovery**

All public methods include comprehensive try/except blocks that:

1. Log appropriate error details
2. Return reasonable defaults on failure
3. Preserve database consistency

## **7. Vector Similarity Search Implementation**

### **7.1 Embedding Storage**

Embeddings are stored as binary blobs for space efficiency:

```python
# Convert vector to bytes for blob storage
vector_blob = np.array(vector, dtype=np.float32).tobytes()
```

### **7.2 Similarity Calculation**

Vector similarity is computed using cosine similarity:

```python
# Calculate cosine similarity
similarity = np.dot(query_array, vector) / (
    np.linalg.norm(query_array) * np.linalg.norm(vector)
)
```

## **8. Usage Examples**

### **8.1 Basic Memory Storage and Retrieval**

```python
from memory.agent_memory.config import SQLiteLTMConfig
from memory.agent_memory.storage.sqlite_ltm import SQLiteLTMStore

# Initialize store
config = SQLiteLTMConfig(
    db_path="agent_memory.db",
    table_prefix="agent_ltm",
    compression_level=2
)
ltm_store = SQLiteLTMStore("agent123", config)

# Store a memory
memory_entry = {
    "memory_id": "mem123",
    "agent_id": "agent123",
    "step_number": 5000,
    "timestamp": int(time.time()),
    "type": "state",
    "content": {"position": [10, 20], "health": 0.9},
    "metadata": {
        "importance_score": 0.75,
        "compression_level": 2
    },
    "embeddings": {
        "compressed_vector": [0.1, 0.2, 0.3, 0.4]  # Small example vector
    }
}
ltm_store.store(memory_entry)

# Retrieve the memory
retrieved_memory = ltm_store.get("mem123")
```

### **8.2 Batch Storage**

```python
# Prepare multiple memories
memories = []
for i in range(10):
    memories.append({
        "memory_id": f"mem{i}",
        "agent_id": "agent123",
        "step_number": 5000 + i,
        "timestamp": int(time.time()),
        "type": "state",
        "content": {"position": [10 + i, 20 + i], "health": 0.9 - (i * 0.05)},
        "metadata": {"importance_score": 0.75 - (i * 0.05)}
    })

# Store in a single transaction
ltm_store.store_batch(memories)
```

### **8.3 Vector Similarity Search**

```python
# Query vector
query_vector = [0.1, 0.2, 0.3, 0.4]  # Sample query vector

# Find similar memories
similar_memories = ltm_store.get_most_similar(query_vector, top_k=5)

# Print results
for memory, similarity in similar_memories:
    print(f"Memory ID: {memory['memory_id']}, Similarity: {similarity:.4f}")
```

### **8.4 Time-Based Retrieval**

```python
# Get memories from a specific time range
start_time = 1677000000  # Example timestamp
end_time = 1677100000    # Example timestamp

time_range_memories = ltm_store.get_by_timerange(
    start_time,
    end_time,
    limit=20
)
```

### **8.5 Health Check**

```python
# Check database health
health_status = ltm_store.check_health()
print(f"Database status: {health_status['status']}")
print(f"Latency: {health_status['latency_ms']:.2f} ms")
print(f"Integrity: {health_status['integrity']}")
```

## **9. Configuration Options**

### **9.1 SQLiteLTMConfig Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `db_path` | Path to SQLite database file | "agent_memory.db" |
| `compression_level` | Compression level (0-2) | 2 |
| `batch_size` | Number of entries to batch write | 100 |
| `table_prefix` | Prefix for table names | "agent_ltm" |

## **10. Performance Considerations**

### **10.1 Optimization Techniques**

1. **Indexing Strategy**: Indices on frequently queried fields
2. **Connection Pooling**: Context manager for connection reuse
3. **Batch Operations**: Transactional batch writes for efficiency
4. **BLOB Storage**: Efficient binary storage for vector embeddings

### **10.2 Performance Recommendations**

1. **Database Location**: Store on local SSD for best performance
2. **Vacuum Regularly**: Run VACUUM periodically on large databases
3. **Index Tuning**: Additional indices for specific query patterns
4. **Connection Timeout**: Adjust timeout for network/disk conditions
5. **Batch Size**: Tune batch size based on memory requirements

## **11. Integration with Memory Architecture**

The `SQLiteLTMStore` is designed to be the final persistence tier in the memory hierarchy:

```
Agent State → Redis STM → Redis IM → SQLite LTM
           (full detail)   (compressed)  (highly compressed)
```

This class handles the long-term storage requirements with efficient compression and retrieval mechanisms.

## **12. Future Enhancements**

1. **Full-Text Search**: Add text-based search capabilities for content
2. **Partitioning Strategy**: Implement sharding for large agent populations
3. **Optimized Vector Search**: Implement approximate nearest neighbor algorithms
4. **Auto-Vacuum**: Implement automatic database maintenance
5. **Query Optimization**: Prepare specialized queries for common access patterns 