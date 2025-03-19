
# **Memory Agent: Implementation Details**

## Memory Architecture

1. **Memory System Classes**:
   - `MemoryAgent` class extending `BaseAgent`
   - `HierarchicalMemory` core class to manage the 3-tier memory
   - `MemoryEntry` class to represent a single memory
   - Tier-specific classes: `STMemory`, `IMMemory`, `LTMemory`

2. **Memory Compression**:
   - Autoencoder-based compression for STM → IM transition
   - Further dimensionality reduction for IM → LTM transition
   - Configurable compression ratios between tiers

3. **Memory Retrieval**:
   - Vector similarity search for relevant memories
   - Reconstruction mechanisms for compressed memories
   - Context-aware filtering based on current agent state

## Implementation Details

1. **Memory Structure**:
   ```python
   class MemoryEntry:
       """Single memory unit with metadata."""
       memory_id: str  # Unique ID for this memory
       creation_time: int  # When this memory was created
       last_access_time: int  # When this memory was last accessed
       importance: float  # Calculated importance score
       embedding: np.ndarray  # Vector representation
       original_data: Any  # Original uncompressed memory content
       compressed_data: Optional[Any]  # Compressed representation
       memory_type: str  # E.g., "perception", "action", "reward", etc.
   ```

2. **Hierarchical Memory Manager**:
   ```python
   class HierarchicalMemory:
       """Manages the three-tier memory system."""
       stm: List[MemoryEntry]  # Short-term memory (full fidelity)
       im: List[MemoryEntry]  # Intermediate memory (medium compression)
       ltm: List[MemoryEntry]  # Long-term memory (high compression)
       
       # Capacities and compression configurations
       stm_capacity: int
       im_capacity: int
       ltm_capacity: int
       stm_to_im_compression_ratio: float
       im_to_ltm_compression_ratio: float
       
       # Encoder models for compression
       stm_encoder: AutoEncoder
       ltm_encoder: AutoEncoder
   ```

3. **Memory Agent Integration**:
   ```python
   class MemoryAgent(BaseAgent):
       """Agent with hierarchical memory capabilities."""
       memory: HierarchicalMemory
       
       # Core memory operations
       def store_memory(self, data: Any, memory_type: str) -> str:
           """Store new memory in STM."""
       
       def retrieve_relevant_memories(self, query: np.ndarray, k: int = 5) -> List[MemoryEntry]:
           """Retrieve k most relevant memories across all tiers."""
       
       def consolidate_memories(self) -> None:
           """Move memories between tiers based on age/importance."""
           
       # Extended agent functions that use memory
       def get_state_with_memory_context(self) -> AgentState:
           """Get current state augmented with relevant memory context."""
       
       def decide_action_with_memory(self) -> Action:
           """Use memory to make more informed decisions."""
   ```

4. **Database Integration**:
   - Add `AgentMemoryModel` to track memory entries in database
   - Implement memory consolidation metrics in `SimulationStepModel`
   - Track memory retrieval success in `LearningExperienceModel`

## Implementation Phases

1. **Phase 1: Basic Memory Structure**
   - Implement `MemoryEntry` and `HierarchicalMemory` classes
   - Add simple storage/retrieval without compression
   - Test with basic agent state memories

2. **Phase 2: Memory Compression**
   - Implement autoencoder for STM → IM compression
   - Add dimensionality reduction for IM → LTM
   - Test reconstruction quality at different compression levels

3. **Phase 3: Memory Retrieval & Relevance**
   - Implement vector similarity search across memory tiers
   - Add importance scoring based on reward/surprise
   - Test retrieval accuracy with known memory patterns

4. **Phase 4: Agent Integration**
   - Extend `BaseAgent` to create `MemoryAgent`
   - Use memories for decision-making
   - Add forgetting mechanisms and retrieval functions

## Key Considerations

1. **Compression Technique Selection**:
   - Does the environment have PyTorch already? Using autoencoders would be optimal
     -  yes, we have PyTorch
   - Alternative: PCA/SVD for simpler compression if deep learning is too heavy

2. **Memory Transition Policies**:
   - When should memories move from STM → IM → LTM?
   - Options: age-based, importance-based, or hybrid approach
     - Experiment with all three

3. **Retrieval Efficiency**:
   - Fast vector similarity search implementation needed
   - Consider approximate nearest neighbor algorithms for large memory stores

4. **Memory Representation**:
   - What's stored in memory: raw perceptions or higher-level abstractions?
   - How to represent different memory types uniformly