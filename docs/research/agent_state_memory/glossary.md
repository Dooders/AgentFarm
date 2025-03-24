# **AgentMemory: Glossary of Terms**

This document provides definitions for terminology used throughout the AgentMemory documentation. Consistent terminology helps maintain clarity across all documentation files.

## **Memory Architecture Terms**

### **STM (Short-Term Memory)**
Redis-based storage tier containing full-resolution, recent agent state information. Characterized by fast access and no compression. Typically retains the most recent ~1000 steps.

### **IM (Intermediate Memory)**
Middle tier in the memory hierarchy with moderate compression. Implemented using Redis with TTL (Time-To-Live) for automatic expiration. Typically retains ~10,000 steps of agent history.

### **LTM (Long-Term Memory)**
Persistent storage tier implemented in SQLite with high compression. Contains highly abstracted representations of agent states spanning the entire simulation history.

### **Memory Transition**
The process of moving memories between tiers (STM → IM → LTM) with increasing compression as they age or become less important.

### **Memory Embedding**
Vector representation of agent states that enables semantic similarity comparisons and efficient retrieval. Implemented using neural network autoencoders for optimized dimensionality reduction while preserving meaningful relationships between states. See [Custom Autoencoder](custom_autoencoder.md) for implementation details.

## **Data Structure Terms**

### **Memory Entry**
Standard data structure for storing agent state information across all memory tiers. Contains core data, metadata, and embeddings for retrieval. See [Core Concepts: Memory Entry Structure](core_concepts.md#3-memory-entry-structure).

### **Memory ID**
Unique identifier assigned to each memory entry when created, persisting across all memory tiers.

### **Importance Score**
Numerical value (typically 0-1) assigned to memories to prioritize retention. Higher scores indicate greater importance.

### **Compression Level**
Indicator of how compressed a memory entry is:
- Level 0: No compression (STM)
- Level 1: Moderate compression (IM)
- Level 2: High compression (LTM)

## **Memory Types**

### **State Memory**
The basic memory type that stores an agent's internal state attributes such as position, resources, health, etc., at a specific point in time.

### **Interaction Memory**
Memory records capturing interactions between agents or between an agent and the environment. Includes details about participants, interaction type, and outcomes.

### **Action State**
Memory records specifically focused on actions an agent has taken, including the action type, parameters, context in which the action was performed, and the resulting outcomes. Action states help agents learn from past behaviors and improve decision-making.

## **Memory Operations**

### **Vector Similarity Search**
Retrieval method that finds memories semantically similar to a query vector, typically using cosine similarity or other distance metrics.

### **Memory Compression**
Process of reducing the dimensionality or detail level of a memory while preserving its most important characteristics.

### **Batch Flushing**
Process of moving accumulated data from Redis to SQLite in batches to optimize performance.

### **Memory Reconstruction**
Process of restoring compressed memories to a more detailed form, potentially with some information loss.

## **System Components**

### **Memory Agent**
Component responsible for managing memory operations, including storage, transition, compression, and retrieval.

### **RedisAgentMemoryLogger**
Implementation class that handles logging of agent states to the Redis-based STM tier.

### **SQLitePersistenceWorker**
Background process responsible for moving data from Redis to SQLite for long-term storage.

### **AgentMemoryAPI**
Interface for interacting with the memory system, providing methods for storing and retrieving agent states.

## **Performance Metrics**

### **Retrieval Latency**
Time required to retrieve memories from storage, typically measured in milliseconds.

### **Compression Ratio**
Measure of data size reduction achieved through compression, expressed as original size : compressed size.

### **Reconstruction Fidelity**
Measure of how accurately a compressed memory can be reconstructed, typically quantified as error rate.

### **Memory Relevance**
Measure of how well retrieved memories match the intended query context.

---

**See Also:**
- [Core Concepts](core_concepts.md) - Fundamental architecture and data structures
- [README](README.md) - Project overview and documentation structure 