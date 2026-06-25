# **Memory Agent: Implementation**

A **Memory Agent** needs a structured way to **store**, **retrieve**, and **use** past experiences to influence future decisions. This involves:

1. **Encoding and Storing New Experiences** (Adding to memory)  
2. **Retrieving Relevant Memories** (Using memory for decision-making)  
3. **Applying Retrieved Memories to Decision-Making** (Choosing actions based on memory)  

---

## **1. Adding New Experiences to Memory**
When an agent experiences an event (e.g., interacting with an environment or another agent), it needs to store the relevant information in its memory.

### **Steps for Memory Addition**
1. **Perceive the environment** → Collect relevant features from the current situation.
2. **Encode the experience** into a high-dimensional vector.
3. **Store it in Short-Term Memory (STM)**.
4. **Periodically compress and move old STM entries** into Intermediate Memory (IM) and Long-Term Memory (LTM).

### **Implementation Approach**
```python
def add_experience(self, state, action, reward, next_state):
    """Encodes and adds an experience to STM."""
    experience = np.concatenate([state, [action, reward], next_state])  # Encode experience as a vector
    self.stm.append(experience)  # Add to STM

    if len(self.stm) > self.stm_capacity:
        self.transition_memory()  # Move older experiences to IM and LTM
```

---

## **2. Retrieving Relevant Memories**
When the agent needs to make a decision, it should retrieve past experiences that are most relevant to the current situation.

### **Steps for Memory Retrieval**
1. **Encode the current state** as a vector.
2. **Search for the most similar memories** (using cosine similarity or a learned retrieval function).
3. **Prioritize STM** (most recent and highest-resolution memories).
4. **If STM lacks useful data, query IM and LTM** for general patterns.

### **Implementation Approach**
```python
def retrieve_memory(self, query_state):
    """Finds the most relevant past experience based on similarity."""
    query_vector = np.array(query_state)
    best_match = None
    best_score = float('-inf')

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # Search STM first
    for memory in self.stm:
        similarity = cosine_similarity(query_vector, memory[:len(query_state)])
        if similarity > best_score:
            best_score = similarity
            best_match = memory

    # Search IM if no good match in STM
    if best_match is None:
        for memory in self.im:
            similarity = cosine_similarity(query_vector, memory[:len(query_state)])
            if similarity > best_score:
                best_score = similarity
                best_match = memory

    # Search LTM if no match in IM
    if best_match is None:
        for memory in self.ltm:
            similarity = cosine_similarity(query_vector, memory[:len(query_state)])
            if similarity > best_score:
                best_score = similarity
                best_match = memory

    return best_match  # Return the most similar past experience
```

---

## **3. Using Memory for Decision-Making**
Once a past experience is retrieved, the agent can use it to inform its next action.

### **Decision-Making Process**
1. **Retrieve similar past experience** based on the current state.
2. **Extract the best action taken in that past experience**.
3. **Decide whether to reuse the past action or modify it** based on updated conditions.

### **Implementation Approach**
```python
def make_decision(self, state):
    """Uses retrieved memory to decide on an action."""
    retrieved_memory = self.retrieve_memory(state)

    if retrieved_memory is None:
        # No relevant memory found → take a random action
        return np.random.choice(self.action_space)

    # Extract the action that was taken in the past similar situation
    past_action = retrieved_memory[len(state)]
    return past_action  # Reuse past successful action
```

---

## **4. Enhancements for Smarter Decision-Making**
### **A. Action Adjustment**
Instead of directly copying past actions, the agent can:
- **Weigh the past action based on memory similarity** (use softmax over top-k memories).
- **Modify the past action slightly** (to account for changes in environment dynamics).

```python
def make_decision_with_adjustment(self, state):
    """Retrieves similar memories and adjusts actions using a weighted approach."""
    retrieved_memories = [self.retrieve_memory(state) for _ in range(3)]  # Get top 3 memories
    actions = [mem[len(state)] for mem in retrieved_memories if mem is not None]
    
    if not actions:
        return np.random.choice(self.action_space)

    # Choose most frequent or best-rated action from similar memories
    adjusted_action = max(set(actions), key=actions.count)
    return adjusted_action
```

### **B. Reinforcement Learning Hybrid**
- Use the memory retrieval function as a **lookup table for Q-learning**.
- Store **(state, action, reward, next state) tuples** and refine decisions using reinforcement learning.

---

## **5. Summary of Memory Integration in Agent Decision-Making**
| Step | Description | Implementation |
|------|------------|----------------|
| **1. Add Memory** | Store new experiences in STM and gradually compress older ones. | `add_experience()` |
| **2. Retrieve Memory** | Find the most relevant past experience. | `retrieve_memory()` |
| **3. Decision-Making** | Use memory to determine the best action. | `make_decision()` |
| **4. Adjust Actions (Optional)** | Modify actions based on memory weighting. | `make_decision_with_adjustment()` |
| **5. Reinforcement Learning (Optional)** | Use memory as a Q-learning lookup table. | Extend to RL framework |

---

## **6. Potential Applications**
- **Agent-based simulations**: Helps agents remember past strategies in long-term simulations.
- **Game AI**: Allows AI players to recall past moves and adjust strategies.
- **Reinforcement Learning**: Enhances learning efficiency by providing a structured memory system.
- **Autonomous decision-making**: Enables AI assistants and robots to adapt based on experience.

---

### **Next Steps**
- Test the memory-based decision-making in a simple **game simulation**.
- Evaluate the **retrieval accuracy and decision efficiency**.
- Extend to **reinforcement learning environments** to compare with traditional learning techniques.