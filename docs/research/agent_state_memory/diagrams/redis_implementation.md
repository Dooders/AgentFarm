# Redis Implementation Diagrams

## Redis Memory Tier Architecture

```mermaid
graph TB
    subgraph "Redis Memory System"
        STM[Short-Term Memory]
        IM[Intermediate Memory]
        IDX[Memory Indices]
        META[Metadata Store]
    end
    
    Agent[Agent] --> |Write memory| WRITE[Memory Write Handler]
    Agent --> |Query memory| QUERY[Memory Query Handler]
    
    WRITE --> STM
    WRITE --> |Update indices| IDX
    WRITE --> |Update metadata| META
    
    QUERY --> IDX
    IDX --> STM
    IDX --> IM
    IDX --> LTM[Long-Term Memory]
    
    STM --> |Importance-based transfer| PROC[Memory Processor]
    PROC --> IM
    IM --> |Age-based compression| COMP[Compression Service]
    COMP --> LTM
    
    LTM --> |Stored in| PS[Persistent Storage]
    META --> |Tracked in| PS
```

This diagram details the Redis-based memory tier architecture. It shows how agents interact with the system through Write and Query handlers. New memories are stored in Short-Term Memory (STM) with their indices and metadata updated accordingly. The Memory Processor evaluates STM entries and transfers important ones to Intermediate Memory (IM). A Compression Service compresses older IM entries and moves them to Long-Term Memory (LTM). Both LTM and metadata are persisted to storage. Query operations leverage indices to efficiently retrieve memories from any tier.

## Redis Data Structure

```mermaid
graph LR
    subgraph "Redis Key Structure"
        STM_K["stm:{agent_id}:{memory_id}"]
        IM_K["im:{agent_id}:{memory_id}"]
        LTM_K["ltm:{agent_id}:{memory_id}"]
        IDX_K["idx:{agent_id}:{index_type}"]
        META_K["meta:{agent_id}:{memory_id}"]
    end
    
    STM_K --> |Hash| STM_V[Memory Entry JSON]
    IM_K --> |Hash| IM_V[Memory Entry JSON]
    LTM_K --> |Hash| LTM_V[Compressed Memory JSON]
    
    IDX_K --> |Sorted Set| IDX_V["{memory_id}" score]
    META_K --> |Hash| META_V[Metadata Fields]
    
    subgraph "Index Types"
        TEMP["temporal:step"]
        SPAT["spatial:region"]
        IMPO["importance:score"]
        TYPE["type:{state|interaction}"]
    end
    
    IDX_K --> TEMP
    IDX_K --> SPAT
    IDX_K --> IMPO
    IDX_K --> TYPE
```

This diagram illustrates the Redis key structure used to implement the memory system. Memory entries are stored as Redis hashes with keys prefixed by their tier (stm/im/ltm) followed by agent_id and memory_id. Indices are implemented as sorted sets that enable efficient querying by various dimensions (temporal, spatial, importance, and type). Metadata is stored in separate hashes for quick access. This structure leverages Redis's data types to provide fast access patterns while maintaining the relationships between different aspects of the memory system.

## Memory Access Patterns

```mermaid
sequenceDiagram
    participant A as Agent
    participant MS as Memory System
    participant R as Redis
    participant PS as Persistent Storage
    
    A->>MS: Request memory (query)
    MS->>R: Check cache for matching entries
    R-->>MS: Return cached entries
    
    alt Cache Miss
        MS->>PS: Query persistent storage
        PS-->>MS: Return matching entries
        MS->>R: Update cache
    end
    
    MS->>MS: Apply relevance scoring
    MS-->>A: Return relevant memories
    
    A->>MS: Store new memory
    MS->>R: Store in STM
    MS->>R: Update indices
    
    loop Memory Management (Background)
        MS->>R: Process STM entries
        MS->>R: Move important memories to IM
        MS->>R: Compress older IM entries to LTM
        MS->>PS: Persist LTM entries
    end
```

This sequence diagram shows the memory access patterns and interactions between system components. When an agent requests memories, the system first checks Redis for cached entries. If there's a cache miss, it falls back to persistent storage and updates the cache. Relevance scoring is applied before returning memories to the agent. For storing new memories, entries are placed in STM with index updates. A background process continuously manages memory transfers between tiers based on importance and age, ensuring optimal memory organization and persistence.

## Data Consistency Management

```mermaid
graph TB
    subgraph "Redis Cache"
        STM[Short-Term Memory]
        IM[Intermediate Memory]
        IDX[Memory Indices]
    end
    
    subgraph "Persistent Storage"
        P_IM[Persisted IM]
        P_LTM[Long-Term Memory]
        P_IDX[Persisted Indices]
    end
    
    STM --> |Write-through| SYNC1[Sync Service]
    IM --> |Periodic sync| SYNC1
    IDX --> |Index updates| SYNC1
    
    SYNC1 --> P_IM
    SYNC1 --> P_LTM
    SYNC1 --> P_IDX
    
    P_IM --> |Cache miss| SYNC2[Data Loader]
    P_LTM --> |Cache miss| SYNC2
    P_IDX --> |Index rebuild| SYNC2
    
    SYNC2 --> IM
    SYNC2 --> IDX
```

This diagram outlines how data consistency is maintained between Redis and persistent storage. The Sync Service manages data flow to storage using different strategies: write-through for STM and periodic syncing for IM and indices. This ensures data durability while optimizing performance. The Data Loader handles cache misses by loading data from persistent storage back into Redis, and can rebuild indices if needed. This two-way synchronization mechanism ensures the system can recover from failures and maintain data integrity across the caching and persistence layers.
