# Unified Agent State and Memory Storage

## Issue

We needed to determine whether agent interaction data (managed by the Memory Agent) and agent state data should be stored separately or unified in the same storage system.

## What We Considered

1. **Separate Storage Approach**:
   - Maintain distinct systems for interaction memory and state data
   - Allow specialized optimization for each data type
   - Enable independent scaling and retention policies
   - More complex integration points between the systems
   - Potential for data duplication or inconsistency

2. **Unified Storage Approach**:
   - Store both data types in the same Redis/SQLite structures
   - Share common infrastructure across data types
   - Use a single hierarchical memory architecture with specialized handling where needed
   - Consistent memory transition mechanisms (STM → IM → LTM)
   - Common indexing system with type-specific extensions

## Decision

We decided to implement a **unified storage approach** where both interaction data and state data share the same hierarchical memory architecture (STM, IM, LTM tiers) with specialized indexing and compression strategies for each data type.

The memory entry structure already accommodates both use cases:

```json
{
  "memory_id": "unique-identifier",
  "agent_id": "agent-123",
  "simulation_id": "sim-456",
  "step_number": 1234,
  "timestamp": 1679233344,
  
  "state_data": {
    "position": [x, y],
    "resources": 42,
    "health": 0.85,
    "action_history": [...],
    "perception_data": [...],
    "other_attributes": {}
  },
  
  "metadata": {
    "creation_time": 1679233344,
    "last_access_time": 1679233400,
    "compression_level": 0,
    "importance_score": 0.75,
    "retrieval_count": 3,
    "memory_type": "state" // or "interaction"
  },
  
  "embeddings": {
    "full_vector": [...],
    "compressed_vector": [...],
    "abstract_vector": [...]
  }
}
```

## Benefits

- **Reduced Complexity**: Simpler implementation with less duplication of code and infrastructure
- **Consistent Access Patterns**: Uniform API for retrieving both state and interaction data
- **Shared Optimizations**: Performance improvements benefit both data types
- **Flexible Structure**: The memory entry structure already accommodates both state data (position, resources, health) and interaction data (action_history, perception_data)
- **Efficient Resource Usage**: Single Redis instance and persistence mechanism
- **Combined Retrieval**: Ability to query across both data types when needed
- **Simplified Development**: Engineers work with a single system rather than two separate ones

## What to Watch For

- **Performance Bottlenecks**: Monitor whether certain data types cause slowdowns that affect the entire system
- **Retrieval Complexity**: Ensure that the unified indexing system handles the different retrieval patterns effectively
- **Storage Efficiency**: Watch for cases where one data type dominates storage, potentially requiring separate compression strategies
- **Schema Evolution**: Be cautious when evolving the schema to ensure backward compatibility for both data types
- **Query Patterns**: Different query patterns between state and interaction data might require specialized indexing strategies
- **Storage Growth**: Monitor the growth rate of different data types, which might require type-specific retention policies despite the unified architecture

## Implementation Notes

- Add a "memory_type" field to distinguish between state and interaction data
- Implement specialized compression strategies for different data types while using the same overall architecture
- Create type-specific indices to optimize retrieval for common query patterns
- Establish different importance scoring mechanisms based on memory type
- Consider asynchronous processing of less time-critical memory operations 