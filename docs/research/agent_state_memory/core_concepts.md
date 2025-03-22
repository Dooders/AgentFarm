# **Agent State Memory: Core Concepts**

## **1. Hierarchical Memory Architecture**

### **1.1 Memory Tiers**

The system employs a three-tier hierarchical memory architecture that mimics cognitive processes:

| Memory Tier | Implementation | Resolution | Purpose | Typical Retention |
|-------------|----------------|------------|---------|-------------------|
| **Short-Term Memory (STM)** | Redis in-memory store | Full resolution | Recent, detailed states | Last ~1000 steps |
| **Intermediate Memory (IM)** | Redis with TTL + compression | Medium resolution | Medium-term trends | Last ~10,000 steps |
| **Long-Term Memory (LTM)** | SQLite with high compression | Low resolution | Historical patterns | Entire simulation history |

### **1.2 Memory Transition Flow**

```
Agent State → Redis STM → Redis IM → SQLite LTM
           (full detail)   (compressed)  (highly compressed)
```

As memory ages or as capacity limits are reached, information transitions between tiers with progressive compression applied to reduce storage requirements while preserving essential patterns.

## **2. Memory Entry Structure**

Each memory record contains the following standardized structure:

```json
{
  "memory_id": "unique-identifier",
  "agent_id": "agent-123",
  "simulation_id": "sim-456",
  "step_number": 1234,
  "timestamp": 1679233344,
  
  "contents": {
    "position": [x, y],
    "resources": 42,
    "health": 0.85,
    "perception": {...},
    "other_attributes": {...}
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
    "full_vector": [...],  // 300-500d for STM
    "compressed_vector": [...],  // 100d for IM
    "abstract_vector": [...]  // 20-30d for LTM
  }
}
```

## **3. Memory Compression Techniques**

The system uses several techniques to compress memories while preserving semantic meaning:

1. **Dimensionality Reduction**: Reducing vector dimensions while preserving semantic relationships
2. **Autoencoder Compression**: Neural network-based encoding to a smaller latent space
3. **Importance-Based Filtering**: Retaining only the most significant features based on importance scores
4. **Temporal Aggregation**: Combining multiple sequential states into summary representations

Compression levels vary by memory tier:
- **STM**: No compression (level 0)
- **IM**: Moderate compression (level 1)
- **LTM**: High compression (level 2)

For detailed implementation of the autoencoder-based compression approach, see [Custom Autoencoder](custom_autoencoder.md).

## **4. Memory Retrieval Methods**

### **4.1 Similarity-Based Retrieval**

Retrieval of memories based on semantic similarity to a query state:

1. **Vector Similarity**: Using cosine similarity between embedding vectors
2. **KNN Search**: Finding k-nearest neighbors in the embedding space
3. **Contextual Relevance**: Weighting by relevance to current context

### **4.2 Attribute-Based Retrieval**

Retrieval based on specific attributes or conditions:

1. **Exact Matching**: Finding memories with exact attribute values
2. **Range Queries**: Finding memories with attributes in specified ranges
3. **Compound Queries**: Combining multiple attribute conditions

### **4.3 Temporal Retrieval**

Retrieval based on time or sequence:

1. **Time Range**: Memories within specific time/step ranges
2. **Recency**: Most recent memories first
3. **Sequential**: Memories in chronological order

## **5. System Architecture Components**

The system consists of these primary components:

1. **Memory Storage Layer**: Handles persistence across memory tiers
2. **Memory Agent**: Manages memory operations and retrieval logic
3. **Redis Cache Layer**: Provides high-performance access to recent memories
4. **API Interface**: Standardized methods for interacting with the system

These components work together to provide a unified memory system that efficiently stores, manages, and retrieves agent state information.

## **6. Key Performance Metrics**

The system's performance is evaluated using:

1. **Retrieval Latency**: Time to retrieve memories
2. **Compression Ratio**: Storage reduction through compression
3. **Reconstruction Fidelity**: Accuracy of decompressed memories
4. **Memory Relevance**: How well retrieved memories match query intent
5. **System Throughput**: Rate of memory operations per second

---

**See Also:**
- For implementation details, see [Agent State Storage](agent_state_storage.md)
- For memory agent functionality, see [Memory Agent](memory_agent.md)
- For Redis integration, see [Redis Integration](redis_integration.md)
- For API specifications, see [Agent State Memory API](agent_state_memory_api.md) 