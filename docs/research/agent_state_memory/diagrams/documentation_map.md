# **AgentMemory Documentation Map**

The following diagram illustrates the relationships between the various documentation files in the AgentMemory system. Use this map to navigate the documentation more effectively.

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│                        README.md                          │
│                  (Navigation Hub & Overview)              │
│                                                           │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                                                           │
│                     core_concepts.md                      │
│                (Single Source of Truth)                   │
│                                                           │
└─┬─────────────┬─────────────┬────────────┬──────────────┬─┘
  │             │             │            │              │
  ▼             ▼             ▼            ▼              ▼
┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│        │  │        │  │         │  │          │  │          │
│ agent_ │  │ memory │  │ redis_  │  │ agent_   │  │ future_  │
│ state_ │  │ agent  │  │ integra │  │ memory_  │  │ enhance  │
│ storage│  │  .md   │  │ tion.md │  │ api.md   │  │ ments.md │
│  .md   │  │        │  │         │  │          │  │          │
│        │  │        │  │         │  │          │  │          │
└────────┘  └────────┘  └─────────┘  └──────────┘  └──────────┘
     │           │           │             │              │
     │           │           │             │              │
     └───────────┴───────────┴─────────────┴──────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │             │
                      │  glossary   │
                      │    .md      │
                      │             │
                      └─────────────┘
```

## **Document Relationships**

### **Core Documents**

- **README.md**: Entry point and navigation hub
- **core_concepts.md**: Central source of truth for all shared concepts

### **Implementation Documents**

- **agent_state_storage.md**: State persistence implementation
- **memory_agent.md**: Memory management implementation
- **redis_integration.md**: Redis caching implementation
- **agent_memory_api.md**: API specification

### **Supporting Documents**

- **glossary.md**: Terminology reference used by all documents
- **future_enhancements.md**: Planned improvements

## **Navigation Guidelines**

1. Start with **README.md** for a high-level overview
2. Read **core_concepts.md** to understand the fundamental architecture
3. Explore specific implementation documents based on your area of interest
4. Refer to the **glossary.md** for any unfamiliar terms
5. Check **future_enhancements.md** for upcoming features

## **Cross-Document References**

All implementation documents reference the core_concepts.md file for shared architectural concepts, data structures, and memory operations. This approach eliminates duplication while ensuring consistency across the documentation.

The glossary.md file serves as a centralized reference for terminology, further enhancing consistency and clarity throughout the documentation. 