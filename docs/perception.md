# Designing Perception Models for an Artificial Agent in a 2D Grid Environment

## Overview
When creating an artificial agent in a 2D environment, the perception system plays a crucial role. Rather than giving the agent direct, global information (like the nearest resource’s exact coordinates), it’s often more realistic and beneficial to model perception as a local, noisy, and limited view. This encourages more naturalistic and adaptable behavior.

## Key Principles

### Local, Agent-Centric Perception
- **Local Perspective Fields:** Provide the agent with a small window (e.g., 7x7 cells) centered on its position.
- **Orientation Alignment:** Align the local perspective field so that the direction the agent faces is always “up” in the perception. Turning left or right rotates the perceived perspective field, ensuring a stable, agent-centric perspective.
- **Limited Field of View (FOV):** Allow the agent to “see” only cells within a certain distance or in front of it, leaving other areas as unknown or masked. This simulates partial observability and encourages the agent to explore.

### Representations
- **Raw Grid Encoding:**
  - Encode each cell with a numeric value:  
    - `0` = Empty  
    - `1` = Energy  
    - `2` = Obstacle/Wall  
    - `-1` (or similar) = Unknown/Not Visible
  - This creates a 2D array (e.g., a 7x7 perspective field) that the agent’s perception network interprets.
  - **Future Extensions:** These representations can become dynamic and agent-defined through experience:
    - Learned embeddings for different cell types
    - Emergent categorization of novel objects
    - Continuous rather than discrete encodings
    - Hierarchical representations that develop through interaction

- **Distance/Ray Sensors (Optional):**
  - Instead of a raw grid, use sets of distance measurements (like LIDAR rays) at fixed angles around the agent.
  - Each ray returns the distance to the nearest object of interest.
  - This is more compact but requires the agent to reconstruct spatial patterns mentally:
    - For example, with 8 rays at 45° intervals, you might only need 8 values instead of a 7x7=49 cell grid
    - The agent must learn to infer spatial relationships between objects from these distance measurements
    - Two objects at similar distances but different angles require the agent to understand geometric relationships
    - Pattern recognition becomes more challenging:
      - A U-shaped wall might be represented as similar distances in three directions
      - The agent needs to learn that this pattern implies an enclosed space
    - Memory becomes more critical as the agent must integrate readings over time to build a complete picture
    - This approach more closely mimics biological systems like whiskers or echolocation

### Noise and Uncertainty
- **Noisy Observations:** Introduce randomness so that sensor readings are not always perfect. This leads to more robust, generalizable policies.
- **Partial Observability:** Limit how far the agent can see or obscure certain areas of the perspective field, forcing the agent to move and remember previous observations to build a mental map.

### Processing the Perception
- **Convolutional Neural Networks (CNNs):**
  - When using a grid perspective field, a small CNN can extract spatial features (like clusters of energy or obstacles).
- **Recurrent Neural Networks (RNNs):**
  - Integrate over time by feeding the CNN features into an LSTM/GRU. This helps the agent remember where it saw energy previously and form a mental representation of the unseen parts of the map.
  
### Hierarchical Structure
- **Perception Module → Decision Module:**
  - Separate the perception process from the decision-making policy.
  - The perception network (CNN/RNN) transforms raw sensor data (grid perspective fields or rays) into a meaningful feature vector.
  - The policy network uses these features plus agent internal states (e.g., its own energy level or orientation) to decide on actions.

### Incremental Complexity
- **Start Simple:**
  - Begin by providing a straightforward local grid with no orientation changes.
- **Add Orientation:**
  - Rotate the local perspective field so “forward” always aligns with the top of the array.
- **Add Limited FOV:**
  - Mask out areas behind the agent or beyond a certain radius.
- **Add Noise and Partial Observability:**
  - Make the environment more challenging and realistic.

## Potential Approaches

1. **Local Grid Perspective Field (Vision-Like):**
   - Extract a small area around the agent.
   - Process with a CNN.
   - Ideal for straightforward spatial reasoning.

2. **Ray-Based Sensors (LIDAR):**
   - Provide a vector of distances to objects at various angles.
   - More compact, but requires more inference by the agent to understand layout.

3. **Hybrid Approach:**
   - Use both local grids and a few distance measurements.
   - The agent gains both a raw spatial “image” and structured distance info.

4. **Preprocessing Steps:**
   - Could use an autoencoder on perspective fields to get a compressed representation.
   - Could pre-train a perception network to identify objects before feeding features into the policy.

## Research and Experimentation Directions
- **Vary Perspective Field Size & FOV:** See how performance changes with smaller or larger views, narrower or wider fields of view.
- **Noise Levels:** Introduce different noise distributions and measure how quickly the agent adapts.
- **Temporal Integration:** Add recurrent layers to handle partial observability and track moving resources or obstacles over time.
- **Scaling Up Complexity:**
  - Add multiple object types (e.g., different resources, enemies).
  - Experiment with dynamic lighting or “fog” where certain areas are always unknown until visited.
  
## Summary
By using local, agent-centric, and limited perception models, you move closer to realistic scenarios where the agent must interpret its environment rather than rely on global cues. Experimenting with grid perspective fields, sensor rays, noise, orientation-dependent views, and memory-based models (RNNs) will yield insights into how robust and adaptive your agent can become.