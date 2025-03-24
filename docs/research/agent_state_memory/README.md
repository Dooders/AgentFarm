# **AgentMemory**

## **Overview**
This project implements a comprehensive system for agent state persistence, memory management, and high-performance caching. It provides autonomous agents with reliable state tracking, context-aware memory, and efficient data retrieval capabilities.

## **Documentation Structure**
This documentation is organized to minimize duplication and provide clear navigation between related concepts:

| Document | Purpose |
|----------|---------|
| [Core Concepts](core_concepts.md) | Single source of truth for key concepts and shared structures |
| [Agent State Storage](agent_state_storage.md) | Implementation of persistent state management |
| [Memory Agent](memory_agent.md) | Dynamic memory system with human-like recall capabilities |
| [Redis Integration](redis_integration.md) | High-performance caching implementation |
| [API Specification](agent_memory_api.md) | Formal API definitions for the memory system |
| [Glossary](glossary.md) | Definitions of key terminology used across documents |
| [Future Enhancements](future_enhancements.md) | Planned improvements and extensions |
| [Custom Autoencoder](custom_autoencoder.md) | Neural network-based embeddings and compression for agent states |
| [Implementation Plan](implementation_plan.md) | Phased approach for development with milestones |
| [Dev Setup](dev_setup.md) | Environment setup for development and testing |
| [Testing Strategy](testing_strategy.md) | Comprehensive testing plan with unit and integration tests |
| [Integration Notes](integration_notes.md) | Guidelines for integrating with existing codebase |
| [Code Structure](CODE_STRUCTURE.MD) | Detailed module structure and code organization |

### **Redis Schema Documentation**
| Document | Purpose |
|----------|---------|
| [Agent State Redis Schema](agent_state_redis_schema.md) | Schema design for agent state storage in Redis |
| [Agent Action Redis Schema](agent_action_redis_schema.md) | Schema design for agent action recording in Redis |
| [Agent Interaction Redis Schema](agent_interaction_redis_schema.md) | Schema design for agent interaction tracking in Redis |
| [Redis Index Schema](redis_index_schema.md) | Index design for efficient Redis retrieval |
| [Index Optimization Strategies](index_optimization_strategies.md) | Performance techniques for Redis indices |

### **Diagram Documentation**
| Document | Purpose |
|----------|---------|
| [Documentation Map](diagrams/documentation_map.md) | Visual guide to documentation organization |
| [Unified Memory Architecture](diagrams/unified_memory_architecture.md) | Comprehensive system architecture and data flows |
| [Redis Implementation](diagrams/redis_implementation.md) | Redis-specific implementation details |
| [Memory Entry Lifecycle](diagrams/memory_entry_lifecycle.md) | How memories flow through the system |

## **System Components**

### **1. Agent State Storage**
Persistent storage mechanism for reliable agent state tracking.
[Learn more →](agent_state_storage.md)

### **2. Memory Agent**
Dynamic memory system with context-awareness and learning capabilities.
[Learn more →](memory_agent.md)

### **3. Redis Integration**
High-performance caching to reduce latency and improve throughput.
[Learn more →](redis_integration.md)

## **Key Features**

- **Hierarchical Memory**: Three-tier architecture with varying resolution levels
- **Performance Optimization**: Redis-based caching with batch persistence
- **Contextual Retrieval**: Semantic similarity and attribute-based lookups
- **Flexible API**: Comprehensive interfaces for memory operations
- **Action State Memory**: Records of agent actions with context and outcomes for improved decision-making
- **Error Handling**: Robust error recovery and graceful degradation strategies
- **Optimized Indices**: Specialized Redis index structures for efficient retrieval

For detailed explanations of these features, see the [Core Concepts](core_concepts.md) document.

## **Implementation Roadmap**

### **Phase 1: Foundation**
- Redis client setup with connection pooling
- Base data structures and storage classes implementation
- Key management and naming conventions

### **Phase 2: Core Memory Operations**
- Core Memory Agent implementation
- Simple embedding generation with similarity search
- Intermediate Memory Store with TTL configuration

### **Phase 3: Long-Term Storage**
- SQLite integration for long-term memory
- Advanced compression with importance-based filtering
- Background workers for batch operations

### **Phase 4: Advanced Retrieval**
- Complex query support with attribute-based filtering
- Memory relevance ranking with context-aware results
- API refinement and comprehensive documentation

### **Phase 5: ML Integration**
- Autoencoder embedding engine implementation
- Multi-resolution compression for different memory tiers
- Production optimization for inference performance

### **Phase 6: Integration & Optimization**
- Agent integration with memory hooks
- Performance optimization for identified bottlenecks
- Error recovery and monitoring implementation

See the complete [Implementation Plan](implementation_plan.md) for detailed milestones and feature prioritization.

## **Getting Started**

For developers looking to use or contribute to this system:

1. Review the [Core Concepts](core_concepts.md) document first
2. Follow the [Dev Setup](dev_setup.md) guide to set up your development environment
3. Explore the [API Specification](agent_memory_api.md) for integration details
4. Check implementation specifics in relevant component documents
5. Review the [Testing Strategy](testing_strategy.md) for quality assurance guidelines
6. Refer to the [Glossary](glossary.md) for term definitions
7. See the [Documentation Map](diagrams/documentation_map.md) for navigation assistance
8. Understand the [Code Structure](CODE_STRUCTURE.MD) for module organization

## **Architectural Diagram**

```
┌─────────────────────────────────────────────────┐
│                  Agent System                   │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│               AgentMemory                       │
├─────────────┬─────────────────────┬─────────────┤
│  Memory     │   State Storage     │    Redis    │
│   Agent     │                     │   Cache     │
└─────┬───────┴──────────┬──────────┴──────┬──────┘
      │                  │                 │
┌─────▼──────┐    ┌──────▼───────┐   ┌────▼─────┐
│    STM     │───▶│      IM      │──▶│   LTM    │
│ (Redis)    │    │  (Redis+TTL) │   │ (SQLite) │
└────────────┘    └──────────────┘   └──────────┘
```

For more detailed architectural views, please refer to the following diagrams:
- [Unified Memory Architecture](diagrams/unified_memory_architecture.md) - Comprehensive system architecture and data flows
- [Redis Implementation](diagrams/redis_implementation.md) - Redis-specific implementation details
- [Memory Entry Lifecycle](diagrams/memory_entry_lifecycle.md) - How memories flow through the system

---

*For additional technical details, see the [Core Concepts](core_concepts.md) document and [Glossary](glossary.md) for terminology.*
