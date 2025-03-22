# Memory Entry Lifecycle Diagrams

## Memory Entry Transformation

```mermaid
graph TD
    subgraph "Memory Entry Creation"
        A[New Agent Experience] --> |Generate| ME[Memory Entry]
        ME --> |Initial Storage| STM[Short-Term Memory]
    end
    
    subgraph "Memory Structure - STM Stage"
        STM_E[STM Entry]
        STM_E --> |Contains| Full[Full State Data]
        STM_E --> |Contains| Comp[Complete Metadata]
        STM_E --> |Contains| FV[Full Embeddings Vector]
    end
    
    subgraph "Memory Structure - IM Stage"
        IM_E[IM Entry]
        IM_E --> |Contains| Sel[Selected State Data]
        IM_E --> |Contains| UpdM[Updated Metadata]
        IM_E --> |Contains| CV[Compressed Vector]
    end
    
    subgraph "Memory Structure - LTM Stage"
        LTM_E[LTM Entry]
        LTM_E --> |Contains| Abs[Abstract State Summary]
        LTM_E --> |Contains| ArchM[Archival Metadata]
        LTM_E --> |Contains| AV[Abstract Vector]
    end
    
    STM --> |Importance Analysis| IM[Intermediate Memory]
    IM --> |Age/Usage Analysis| LTM[Long-Term Memory]
    
    STM_E --> |Transfer & Process| IM_E
    IM_E --> |Compress & Summarize| LTM_E
```

This diagram illustrates how memory entries transform as they move through the memory tiers. When an agent has a new experience, a memory entry is created and stored in Short-Term Memory (STM) with full state data, complete metadata, and full embedding vectors. As important memories transfer to Intermediate Memory (IM), they undergo processing where only selected state data is retained and vectors are compressed. When memories move to Long-Term Memory (LTM), they're further compressed to abstract summaries with archival metadata and abstract vectors. This progressive transformation mimics human memory, where details fade over time but core concepts remain.

## Memory Importance Scoring System

```mermaid
graph LR
    subgraph "Importance Scoring Factors"
        E1[Emotional Impact]
        E2[Novelty Score]
        E3[Action Relevance]
        E4[Goal Alignment]
        E5[Retrieval Frequency]
    end
    
    E1 --> |Weight & Sum| IS[Importance Score]
    E2 --> |Weight & Sum| IS
    E3 --> |Weight & Sum| IS
    E4 --> |Weight & Sum| IS
    E5 --> |Weight & Sum| IS
    
    IS --> |Score > Threshold| Retain[Retain in Memory]
    IS --> |Score < Threshold| Discard[Discard from Memory]
    
    Retain --> |Very High Score| IM_P[IM - Priority]
    Retain --> |High Score| IM_S[IM - Standard]
    Retain --> |Medium Score| LTM_A[LTM - Active]
    Retain --> |Low Score| LTM_D[LTM - Deep]
```

This diagram shows the importance scoring system that determines which memories to retain and where to store them. Five key factors contribute to a memory's importance: emotional impact, novelty, action relevance, goal alignment, and retrieval frequency. These factors are weighted and summed to produce an importance score. Memories scoring below a threshold are discarded, while those above are retained. The final score determines the memory's destination: very high-scoring memories go to priority IM, high scores to standard IM, medium scores to active LTM, and low (but above threshold) scores to deep LTM. This nuanced scoring system ensures the most relevant memories are most easily accessible.

## Memory Access and Retrieval Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant Q as Query Engine
    participant STM as Short-Term Memory
    participant IM as Intermediate Memory
    participant LTM as Long-Term Memory
    
    A->>Q: Query for relevant memories
    
    par Query all memory tiers
        Q->>STM: Search recent memories
        STM-->>Q: Return matches (high detail)
        
        Q->>IM: Search relevant memories
        IM-->>Q: Return matches (medium detail)
        
        Q->>LTM: Search historical memories
        LTM-->>Q: Return matches (low detail)
    end
    
    Q->>Q: Rank and merge results
    
    alt Detailed information needed
        Q->>LTM: Request decompression of entry
        LTM-->>IM: Restore memory to IM
        IM-->>Q: Return detailed entry
    end
    
    Q-->>A: Return consolidated memories
    
    A->>Q: Select memory for focus
    Q->>Q: Update retrieval count
    Q->>IM: Promote memory if needed
```

This sequence diagram illustrates the memory access and retrieval process. When an agent queries for memories, the Query Engine searches all memory tiers in parallel: STM for recent memories with high detail, IM for relevant memories with medium detail, and LTM for historical memories with low detail. Results are ranked and merged based on relevance. If detailed information is needed for an LTM entry, it's decompressed and restored to IM before being returned. When the agent selects a specific memory for focus, its retrieval count is updated, potentially promoting frequently accessed memories to more accessible tiers. This parallel querying with intelligent promotion ensures efficient memory retrieval.

## State Data Compression Techniques

```mermaid
graph TD
    subgraph "Raw State Data"
        RS1[Exact Position]
        RS2[Resource Levels]
        RS3[Health Status]
        RS4[All Perception Data]
        RS5[Detailed Action History]
    end
    
    subgraph "Compressed State Data"
        CS1[Region Only]
        CS2[Resource Summary]
        CS3[Health Trend]
        CS4[Important Perceptions]
        CS5[Key Actions]
    end
    
    subgraph "Abstract State Data"
        AS1[Area Description]
        AS2[Resource State]
        AS3[Overall Condition]
        AS4[Critical Insights]
        AS5[Behavior Pattern]
    end
    
    RS1 --> |Spatial Compression| CS1
    RS2 --> |Numerical Binning| CS2
    RS3 --> |Trend Analysis| CS3
    RS4 --> |Relevance Filtering| CS4
    RS5 --> |Action Clustering| CS5
    
    CS1 --> |Semantic Abstraction| AS1
    CS2 --> |State Classification| AS2
    CS3 --> |Condition Summary| AS3
    CS4 --> |Insight Extraction| AS4
    CS5 --> |Pattern Recognition| AS5
```

This diagram details the progressive compression techniques applied to different aspects of state data as memories move through tiers. Raw state data in STM contains precise information like exact positions and detailed resource levels. In IM, this data is compressed: positions become regions, resources are summarized, and only important perceptions and key actions are retained. In LTM, the data becomes highly abstract: regions become area descriptions, resource details become general states, and specific actions become behavior patterns. Each type of data undergoes specialized compression algorithms appropriate to its nature, preserving essential information while reducing storage requirements.

## Memory Index and Query System

```mermaid
graph TB
    subgraph "Memory Indices"
        TI[Temporal Index]
        SI[Spatial Index]
        EI[Embedding Index]
        TYI[Type Index]
        II[Importance Index]
    end
    
    subgraph "Query Types"
        QT1[Time-based Query]
        QT2[Location-based Query]
        QT3[Semantic Query]
        QT4[Type-specific Query]
        QT5[Importance-based Query]
        QT6[Combined Query]
    end
    
    QT1 --> |Uses| TI
    QT2 --> |Uses| SI
    QT3 --> |Uses| EI
    QT4 --> |Uses| TYI
    QT5 --> |Uses| II
    
    QT6 --> |Uses| TI
    QT6 --> |Uses| SI
    QT6 --> |Uses| EI
    QT6 --> |Uses| TYI
    QT6 --> |Uses| II
    
    TI --> |Indexes| MEM[Memory Entries]
    SI --> |Indexes| MEM
    EI --> |Indexes| MEM
    TYI --> |Indexes| MEM
    II --> |Indexes| MEM
```

This diagram shows the indexing and query system that enables efficient memory retrieval. Five types of indices are maintained: Temporal (time-based), Spatial (location-based), Embedding (semantic), Type (state vs. interaction), and Importance (priority-based). These correspond to different types of queries an agent might make, such as "What happened yesterday?" (temporal), "What did I see in the forest?" (spatial), or "What do I know about resources?" (semantic). Combined queries can leverage multiple indices simultaneously for complex memory retrieval. This multi-dimensional indexing system allows for flexible, efficient querying across all memory tiers regardless of compression level. 