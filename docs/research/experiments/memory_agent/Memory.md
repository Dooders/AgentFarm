## Memory Representation Structure

For the MemoryAgent, each memory entry should capture:

1. **Input Context** - What the agent perceived
2. **Action Taken** - What the agent did
3. **Outcome** - What resulted from the action
4. **Temporal Information** - When this occurred

Here's a more detailed specification:

```python
class MemoryEntry:
    """Represents a single episodic memory in the agent's memory system."""
    
    memory_id: str  # Unique identifier
    
    # Temporal metadata
    creation_time: int  # Simulation step when memory was created
    last_access_time: int  # Last time this memory was retrieved
    
    # Agent state when memory was formed
    agent_state: AgentState  # Agent's state before action
    perception: PerceptionData  # Environmental perception at that moment
    
    # Action and outcome
    action_taken: Action  # The action agent chose to take
    action_result: Any  # Result of the action (resource gain, position change, etc.)
    reward: float  # Reward received for this action
    
    # Memory management metadata
    importance: float  # Calculated importance score
    retrieval_count: int  # How many times this memory has been accessed
    
    # Embeddings for searching
    embedding: np.ndarray  # Vector representation for similarity search
    
    # Compression data
    compression_level: int  # 0=none, 1=IM, 2=LTM
    compressed_data: Optional[np.ndarray]  # Compressed representation if not in STM
```

## Memory Types to Store

I recommend storing several types of memory:

1. **Perception-Action Memories**
   - What the agent saw and did
   - Important for learning behavioral patterns

2. **Resource-Related Memories**
   - Where resources were found
   - Successful gathering events

3. **Social Interaction Memories**
   - Encounters with other agents
   - Combat and sharing outcomes

4. **Significant Events**
   - Near-death experiences
   - Large rewards or penalties
   - Reproduction events

## Encoding Strategy

For efficiency and effective similarity search:

1. **Joint Embedding**:
   - Encode both state and action information in a single vector
   - Use a concatenation of encoded perception and encoded state

2. **Vectorization Process**:
   ```python
   def create_memory_embedding(agent_state, perception, action, reward):
       # Convert agent state to tensor
       state_tensor = agent_state.to_tensor(device)
       
       # Flatten and normalize perception grid
       perception_tensor = torch.tensor(perception.grid.flatten(), device=device) / 3.0
       
       # One-hot encode the action
       action_tensor = torch.zeros(len(agent.actions), device=device)
       action_tensor[agent.actions.index(action)] = 1.0
       
       # Combine into single embedding
       combined = torch.cat([state_tensor, perception_tensor, action_tensor, 
                            torch.tensor([reward], device=device)])
       
       return combined.cpu().numpy()
   ```

3. **Compression Approach**:
   - STM: Store full embeddings (~300-500 dimensions)
   - IM: Compress to ~100 dimensions using autoencoder
   - LTM: Further compress to ~20-30 dimensions

## Integration with Agent Decision-Making

The memory system should be used in the agent's decision process:

```python
def decide_action_with_memory(self):
    # Get current state and perception
    current_state = self.get_state()
    perception = self.get_perception()
    
    # Create query embedding from current state
    query = self.create_query_embedding(current_state, perception)
    
    # Retrieve relevant memories
    relevant_memories = self.memory.retrieve_relevant_memories(query, k=5)
    
    # Augment decision making with past experiences
    if relevant_memories:
        # Extract patterns from similar past situations
        similar_actions = [mem.action_taken for mem in relevant_memories]
        rewards = [mem.reward for mem in relevant_memories]
        
        # Bias action selection based on past successes
        action = self.select_module.select_action_with_memory(
            self, self.actions, current_state, similar_actions, rewards)
    else:
        # Fall back to standard decision process if no relevant memories
        action = self.select_module.select_action(self, self.actions, current_state)
    
    # Store this new experience in STM
    self.store_memory(current_state, perception, action, 0)  # 0 reward initially
    
    return action
```

## Memory Transition Strategy

For determining when to move memories between tiers:

1. **Hybrid Approach (recommended)**:
   - Age: Move oldest memories when STM/IM reach capacity
   - Importance: Retain memories with high importance scores longer
   - Formula: `transition_score = age * (1 - importance_factor)`

2. **Importance Calculation**:
   ```python
   def calculate_importance(memory):
       # Base importance from reward magnitude
       reward_importance = min(1.0, abs(memory.reward) / 10.0)
       
       # Retrieval frequency factor
       retrieval_factor = min(1.0, memory.retrieval_count / 5.0)
       
       # Recency factor (inverse of age)
       recency = max(0.0, 1.0 - ((current_time - memory.creation_time) / 1000))
       
       # Surprise factor (difference from expected outcome)
       surprise = calculate_surprise(memory)
       
       # Combined importance score
       importance = (0.4 * reward_importance + 
                     0.3 * retrieval_factor + 
                     0.2 * recency + 
                     0.1 * surprise)
       
       return importance
   ```

This memory representation design allows the agent to store and retrieve meaningful experiences while efficiently managing memory capacity through compression tiers. It integrates well with the existing agent framework while adding the hierarchical memory capabilities we want to experiment with.
