# Metric Documentation: 

## **Population Momentum**

## Definition:
**Population Momentum (momentum)** is a composite metric designed to capture the relationship between population growth and simulation duration. It measures population sustainability and growth efficiency by combining three key factors:
1. **Survival Time**: How long the population lasts in the simulation (measured by `final_step`).
2. **Peak Population**: The highest number of agents present during the simulation (`max_population`).
3. **Initial Population**: The starting population size for normalization (`initial_population`).

### Formula:
\[
\text{momentum} = \frac{\text{final\_step} \cdot \text{max\_population}}{\text{initial\_population}}
\]
- **`final_step`**: The simulation step where the last agent perishes, marking the end of the simulation.
- **`max_population`**: The maximum population size achieved at any step during the simulation.
- **`initial_population`**: The population count at the first step of the simulation.

---

## Purpose:
The **Population Momentum** metric quantifies population sustainability and growth efficiency by normalizing for initial population size. It is useful for:
1. **Comparing Simulations**:
   - Identifying which configurations, environments, or parameters result in more sustainable populations.
2. **Evaluating Strategies**:
   - Understanding how changes in agent behavior, resources, or other variables affect population growth efficiency.
3. **Normalized Analysis**:
   - Comparing populations with different starting sizes on an equal basis.

---

## Value:
### Interpretation:
- **Higher Momentum**: Indicates a population that achieved high growth efficiency relative to its starting size, with either long survival time or significant population growth.
- **Lower Momentum**: Reflects a population with poor growth efficiency, either dying out quickly or failing to grow significantly relative to its initial size.

### Examples:
| **Simulation** | **final_step** | **max_population** | **initial_population** | **Momentum** |
|-----------------|----------------|-------------------|----------------------|--------------|
| Simulation A    | 1000           | 50                | 10                    | 5,000        |
| Simulation B    | 500            | 200               | 20                    | 5,000        |
| Simulation C    | 200            | 20                | 5                     | 800          |

---

## Limitations:
While **Population Momentum** is a useful normalized metric, it may not fully capture:
1. **Population Stability**: Simulations with erratic population fluctuations may have the same momentum as stable ones.
2. **Resource Efficiency**: Momentum does not reflect resource consumption or resource utilization efficiency.
3. **Population Dynamics**: It focuses only on peak size, end time, and initial size, ignoring the shape of the population curve or growth patterns.

---

## Usage:
**Population Momentum** provides a normalized measure of population growth efficiency that accounts for different starting conditions. It can be visualized alongside other simulation metrics to provide a meaningful assessment of population sustainability. For deeper insights, consider pairing it with additional metrics like **average population size**, **resource utilization**, or **stability measures**. 

---

Here’s how you can calculate and interpret the suggested metrics:

---

### 1. **Average Population Size**
**Definition**: The average number of agents present at each simulation step.

**Formula**:
\[
\text{average population size} = \frac{\sum_{t=1}^{\text{death\_step}} \text{population}(t)}{\text{death\_step}}
\]
Where:
- \( \text{population}(t) \) is the population size at step \( t \).
- \( \text{death\_step} \) is the last step where any agents are alive.

**Purpose**:
- Reflects the overall activity level of the population throughout the simulation.
- Provides insight into how "full" the simulation world is on average.

**Implementation**:
- Maintain a running total of the population size at each step.
- Divide by the total number of steps (`death_step`).

---

### 2. **Resource Utilization**
**Definition**: The proportion of resources consumed by the population during the simulation relative to the total available resources.

**Formula**:
\[
\text{resource utilization} = \frac{\text{resources consumed}}{\text{resources available}}
\]
Where:
- **`resources consumed`**: Total amount of resources used by agents during the simulation.
- **`resources available`**: Total amount of resources generated or present in the simulation.

**Purpose**:
- Indicates how efficiently the population is using available resources.
- Can highlight whether resource scarcity played a role in population decline.

**Implementation**:
- Track resource consumption per step or action.
- Sum the consumption over the simulation.
- Normalize by the total available resources.

**Advanced Consideration**:
- Add a **per-agent utilization metric**:
  \[
  \text{utilization per agent} = \frac{\text{resources consumed}}{\text{total agent-steps}}
  \]
  where agent-steps = total population size across all simulation steps.

---

### 3. **Stability Measures**
**Definition**: Metrics that capture fluctuations in population size over time.

#### (a) **Population Variance**:
Measures how much the population size fluctuates around its mean.

**Formula**:
\[
\text{population variance} = \frac{\sum_{t=1}^{\text{death\_step}} \left(\text{population}(t) - \text{average population size}\right)^2}{\text{death\_step}}
\]

**Purpose**:
- High variance indicates instability (e.g., booms and crashes).
- Low variance suggests a steady, sustainable population.

#### (b) **Coefficient of Variation (CV)**:
Standardized measure of stability (dimensionless).

**Formula**:
\[
\text{CV} = \frac{\text{standard deviation of population size}}{\text{average population size}}
\]

**Purpose**:
- Allows comparison of stability across simulations with different population sizes.

---

Here are additional valuable metrics to further analyze and understand population dynamics and simulation performance:

---

### **Population Metrics**
1. **Peak-to-End Ratio**:
   - Measures how much the population declines from its peak to its final size.
   - Formula:
     \[
     \text{Peak-to-End Ratio} = \frac{\text{max\_count}}{\text{population\_end}}
     \]
   - Purpose:
     - Indicates resilience or catastrophic collapse towards the end.

2. **Population Growth Rate**:
   - Average rate of change in population size over time.
   - Formula:
     \[
     \text{Growth Rate} = \frac{\text{population(end)} - \text{population(start)}}{\text{death\_step}}
     \]
   - Purpose:
     - Quantifies the speed and direction of population change (growth vs. decline).

3. **Extinction Threshold Time**:
   - Time (step) at which the population first falls below a critical threshold (e.g., 10% of the peak size).
   - Purpose:
     - Highlights critical periods of vulnerability or collapse.

---

### **Behavioral Metrics**
4. **Agent Diversity**:
   - Measures the diversity of agent states, types, or actions during the simulation.
   - Formula (Shannon Entropy of agent types):
     \[
     H = -\sum_{i} p_i \log(p_i)
     \]
     Where \( p_i \) is the proportion of agents of type \( i \).
   - Purpose:
     - Indicates robustness and adaptability of the population.

5. **Interaction Rate**:
   - Average number of interactions per agent per step.
   - Purpose:
     - Tracks how actively agents are engaging with each other or their environment.

6. **Conflict/Cooperation Ratio**:
   - Ratio of combative to collaborative interactions between agents.
   - Purpose:
     - Reveals behavioral dynamics and population harmony.

---

### **Resource Metrics**
7. **Resource Scarcity Index**:
   - Measures how often resources fall below a critical threshold relative to agent demand.
   - Formula:
     \[
     \text{Scarcity Index} = \frac{\text{steps with scarce resources}}{\text{total steps}}
     \]
   - Purpose:
     - Highlights resource-driven pressures on the population.

8. **Unequal Resource Distribution**:
   - Tracks inequality in resource possession among agents (e.g., Gini coefficient).
   - Purpose:
     - Identifies fairness or disparity in resource allocation.

---

### **Environmental Metrics**
9. **Resource Recovery Time**:
   - Measures how long it takes for depleted resources to regenerate.
   - Purpose:
     - Indicates the resilience of the environment and its ability to sustain life.

10. **Environmental Stress Index**:
    - Tracks how much stress agents place on the environment over time.
    - Formula:
      \[
      \text{Stress Index} = \frac{\text{resources consumed}}{\text{resource regeneration rate}}
      \]
    - Purpose:
      - Helps determine sustainability of agent behaviors.

---

### **Evolutionary Metrics**
11. **Mutation Rate Success**:
    - Proportion of beneficial mutations leading to higher fitness (e.g., better resource acquisition, survival).
    - Purpose:
      - Tracks the effectiveness of evolutionary mechanisms.

12. **Gene Flow**:
    - Tracks how frequently genetic material (e.g., parameters) is shared among agents.
    - Purpose:
      - Highlights the spread of traits or strategies in the population.

13. **Generational Turnover**:
    - Measures how quickly the population cycles through generations.
    - Purpose:
      - Captures the pace of evolution and adaptation.

---

### **System-Wide Metrics**
14. **Carrying Capacity Utilization**:
    - Ratio of the average population size to the environment’s carrying capacity.
    - Purpose:
      - Indicates how close the system is operating to its limits.

15. **Population-Resource Feedback**:
    - Tracks correlations between population size and resource availability over time.
    - Purpose:
      - Reveals systemic feedback loops, such as overpopulation leading to resource depletion.

16. **Energy Efficiency**:
    - Measures how efficiently agents convert resources into productive actions.
    - Formula:
      \[
      \text{Energy Efficiency} = \frac{\text{reward earned by agents}}{\text{resources consumed}}
      \]
    - Purpose:
      - Highlights optimization of resource use.

---

### **Temporal Metrics**
17. **Critical Event Timing**:
    - Tracks key moments like the first resource shortage, peak population, or onset of collapse.
    - Purpose:
      - Pinpoints pivotal phases in the simulation timeline.

18. **Population Oscillation Frequency**:
    - Analyzes periodic fluctuations in population size.
    - Purpose:
      - Identifies stability or cyclic dynamics in the system.

---

### **Health and Well-being Metrics**
19. **Average Health**:
    - Tracks the average health of agents over time.
    - Purpose:
      - Indicates overall well-being and resilience.

20. **Survivor Ratio**:
    - Percentage of agents that survive to the end of the simulation.
    - Formula:
      \[
      \text{Survivor Ratio} = \frac{\text{agents alive at death\_step}}{\text{total agents created}}
      \]
    - Purpose:
      - Reflects survival success under current conditions.


---

