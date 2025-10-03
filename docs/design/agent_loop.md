# AgentFarm Agent Loop Design

This document describes the design of the **Agent Step Loop** in AgentFarm.
The loop provides a structured way to model how agents interact with their environment, transform observations into internal states, and act back on the world.

> **Note**: This design is still aspirational and not fully implemented in the current codebase.  

---

## Overview

At its core, the agent loop follows four stages:

**Observation → Perception → Cognition → Action**

This decomposition provides both clarity and modularity:
- **Observation** handles raw environment input.
- **Perception** transforms input into useful latent features.
- **Cognition** integrates memory, world modeling, and decision-making.
- **Action** executes changes in the environment.

---

## Detailed Stages

### 1. Observation
- **Description**: Direct data from the environment, unprocessed and possibly noisy.  
- **Examples**:  
  - Grid cells within field-of-view (FOV)  
  - Resource amounts, agent positions  
  - Audio, messages, or event signals  
- **Outputs**: `obs_raw`

---

### 2. Perception
- **Description**: Encodes raw observations into meaningful representations.  
- **Operations**:  
  - Normalization and denoising  
  - Salience detection and attention  
  - Modality-specific encoders (CNN for vision, GNN for relationships, MLP for symbolic data)  
- **Outputs**: `z_percept` (latent embedding)

---

### 3. Cognition
- **Description**: The agent’s reasoning and decision-making stage.  
- **Components**:  
  - **State Estimation**: World model update (`z_state`)  
  - **Memory**: Query episodic/semantic/genetic stores  
  - **Goals & Drives**: Update needs, intrinsic motivation (entropy, curiosity)  
  - **Planning**: Rollouts in latent space, candidate evaluation  
  - **Decision**: Policy head selects an action distribution  
- **Outputs**: `policy_out`, `action_dist`

---

### 4. Action
- **Description**: Externalization of cognition into environment-affecting moves.  
- **Operations**:  
  - Synthesize and filter action commands  
  - Apply safety constraints or arbitration  
  - Commit to environment step  
- **Outputs**: `action_cmd`

---

### Feedback Loop
The cycle is **recursive**:
- Action changes the environment.  
- The environment produces new observations.  
- Observation begins the cycle again.  

This makes the loop **continuous** and suitable for multi-agent interaction.

---

## Comparative Mapping

| **Stage** | **AgentFarm (O–P–C–A)** | **OODA Loop** (Boyd) | **Sense–Plan–Act** (Robotics) | **World Models** (Ha & Schmidhuber / Dreamer) |
|-----------|--------------------------|-----------------------|-------------------------------|-----------------------------------------------|
| **Observation** | Raw data from environment | **Observe** | **Sense** | Raw obs |
| **Perception** | Encoders transform input into latent features | **Orient** | (folded into Sense) | Encoder → latent |
| **Cognition** | World model, memory, goals, planning, policy | **Decide** | **Plan** | Latent dynamics + controller |
| **Action** | Synthesized and committed action | **Act** | **Act** | Decoder/actor produces action |
| **Feedback** | Explicit cyclical loop | Iterative O–O–D–A | Sequential SPA cycles | Closed loop: encode → rollout → act |

---

## Design Principles

1. **Explicit Observation vs. Perception**
   - Keeps raw environment data separate from learned feature extraction.
   - Supports ablation: symbolic envs may skip perception entirely.

2. **Composable Cognition**
   - Cognition is not a black box.  
   - Submodules: memory, world model, goals, planning, and decision heads.  
   - Allows experiments with modular swaps (e.g., memory on/off).

3. **First-Class Memory**
   - Explicit read/write interface with episodic, semantic, and genetic tiers.  
   - Agents can learn to use or ignore memory as needed.

4. **Intrinsic Motivation**
   - Entropy, curiosity, and empowerment are baked into cognition.  
   - Goes beyond extrinsic task rewards.

5. **Feedback and Recursion**
   - Action is not terminal: it’s part of a closed loop.  
   - Agents can reorient perception and cognition based on their own prior actions.

---

## Diagram

```mermaid
flowchart TD
    O[Observation<br/>Raw Input] --> P[Perception<br/>Encoding & Attention]
    P --> C[Cognition<br/>Memory • World Model • Policy]
    C --> A[Action<br/>Actuation & Safety]
    A --> O
````

---

## Minimal Implementation

For early experiments, a **minimal viable loop** can be implemented:

1. **Sense** → 2. **Encode** → 3. **Policy** → 4. **Act**

Then expand with:

* Attention/salience
* World model
* Memory
* Intrinsic reward modules
* Rollout-based planning

---

## Future Work

* **Attention Mechanisms**: Add adaptive focus on relevant observations.
* **Multi-Agent Cognition**: Shared memory pools, communication protocols.
* **Hierarchical Cognition**: Split cognition into fast reactive vs. slow deliberative heads.
* **Emotion/Affect Layer**: Valence and drive modulation across the loop.
* **Meta-Learning**: Agents that evolve their own loop structure over time.

---

## Summary

The AgentFarm loop is both **classical and novel**:

* Classical in its resemblance to OODA, SPA, and World Models.
* Novel in its explicit observation-perception split, modular cognition, and entropy-driven design.

This structure ensures AgentFarm agents are both **research-friendly** (fine-grained, ablatable) and **scalable** (suited to large populations in simulation).

