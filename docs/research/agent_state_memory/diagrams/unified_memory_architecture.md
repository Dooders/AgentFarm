# Unified Memory Architecture Diagrams

## High-Level System Architecture

```mermaid
graph TD
    Agent[Agent] --> |Interacts with| MS[Memory System]
    
    subgraph "Unified Memory Architecture"
        MS --> |Stores/Retrieves| STM[Short-Term Memory]
        MS --> |Stores/Retrieves| IM[Intermediate Memory]
        MS --> |Stores/Retrieves| LTM[Long-Term Memory]
        
        STM --> |Importance-based transfer| IM
        IM --> |Age/importance-based transfer| LTM
    end
    
    MS --> |Uses| Redis[Redis Cache]
    MS --> |Persists to| DB[Persistent Storage]
    
    Redis --> |Syncs with| DB
```

This diagram illustrates the high-level architecture of the unified memory system. It shows how agents interact with the memory system, which manages three tiers of memory: Short-Term Memory (STM), Intermediate Memory (IM), and Long-Term Memory (LTM). The memory system uses Redis for high-performance caching and persists data to a database. Information flows between memory tiers based on importance and age, with Redis syncing to persistent storage to ensure data durability.

## Memory Entry Structure

```mermaid
classDiagram
    class MemoryEntry {
        +string memory_id
        +string agent_id
        +string simulation_id
        +int step_number
        +timestamp timestamp
        +StateData state_data
        +Metadata metadata
        +Embeddings embeddings
    }
    
    class StateData {
        +array position
        +int resources
        +float health
        +array action_history
        +array perception_data
        +object other_attributes
    }
    
    class Metadata {
        +timestamp creation_time
        +timestamp last_access_time
        +int compression_level
        +float importance_score
        +int retrieval_count
        +string memory_type
    }
    
    class Embeddings {
        +array full_vector
        +array compressed_vector
        +array abstract_vector
    }
    
    MemoryEntry *-- StateData
    MemoryEntry *-- Metadata
    MemoryEntry *-- Embeddings
```

This class diagram depicts the structure of a memory entry in the system. Each memory entry contains identifiers, a timestamp, state data (position, resources, health, etc.), metadata (creation time, importance score, etc.), and embeddings used for semantic retrieval. The unified design allows for both state data and interaction data to be stored in the same structure, with the memory_type field distinguishing between them. This schema supports the efficient storage and retrieval of different types of agent memories.

## Memory Flow and Transitions

```mermaid
flowchart LR
    A[New Memory Entry] --> STM[Short-Term Memory]
    STM --> |Low importance| Discard[Discarded]
    STM --> |High importance| IM[Intermediate Memory]
    IM --> |Low retrieval frequency| Compress[Compressed]
    Compress --> LTM[Long-Term Memory]
    IM --> |High retrieval frequency| Retain[Retained in IM]
    LTM --> |Retrieval request| Decompress[Decompressed]
    Decompress --> IM
```

This flowchart shows how memory entries move through the system's memory tiers. New memories enter Short-Term Memory (STM), where low-importance memories are discarded while important ones move to Intermediate Memory (IM). From IM, memories that are accessed frequently remain there for quick retrieval, while less-frequently accessed memories are compressed and moved to Long-Term Memory (LTM). When an LTM entry needs to be accessed, it's decompressed and moved back to IM for faster access. This mimics human memory processes where important and frequently accessed memories remain readily available.

## Component Integration

```mermaid
graph TB
    subgraph "Main Components"
        AS[Agent State Storage]
        MA[Memory Agent]
        RC[Redis Cache]
    end
    
    AS --> |State updates| UMS[Unified Memory System]
    MA --> |Memory operations| UMS
    RC --> |Caching layer| UMS
    
    subgraph "Unified Memory System"
        SM[State Management]
        MM[Memory Management]
        QE[Query Engine]
        PM[Persistence Manager]
    end
    
    UMS --> SM
    UMS --> MM
    UMS --> QE
    UMS --> PM
    
    SM --> |Uses| Redis[Redis]
    MM --> |Uses| Redis
    QE --> |Queries| Redis
    PM --> |Persists| DB[Database/File Storage]
    
    Redis --> |Syncs with| DB
```

This diagram illustrates how the main components integrate within the unified system. The three key components (Agent State Storage, Memory Agent, and Redis Cache) work with the Unified Memory System, which consists of four subsystems: State Management, Memory Management, Query Engine, and Persistence Manager. Each subsystem interfaces with Redis for caching and with the database for persistence. This design unifies what would otherwise be separate systems into a cohesive architecture with clear responsibilities and interfaces.

## Data Types and Storage Tiers

```mermaid
graph TD
    subgraph "Memory Types"
        ST[State Data]
        IN[Interaction Data]
    end
    
    ST --> |Stored in| UMS[Unified Memory System]
    IN --> |Stored in| UMS
    
    subgraph "Storage Tiers"
        STM[Short-Term Memory Tier]
        IM[Intermediate Memory Tier]
        LTM[Long-Term Memory Tier]
    end
    
    UMS --> STM
    UMS --> IM
    UMS --> LTM
    
    STM --> |In-memory, Redis| RC[Redis Cache]
    IM --> |Partially cached| RC
    IM --> |Partially persisted| PS[Persistent Storage]
    LTM --> |Fully persisted| PS
```

This diagram shows how different types of data (State Data and Interaction Data) are stored within the unified system. Both data types share the same memory architecture with three tiers: Short-Term Memory (STM), Intermediate Memory (IM), and Long-Term Memory (LTM). STM is kept entirely in Redis for fast access, IM is partially cached and partially persisted, and LTM is fully persisted to storage. This unified approach enables consistent memory management regardless of data type while allowing for optimized storage strategies based on memory tier. 