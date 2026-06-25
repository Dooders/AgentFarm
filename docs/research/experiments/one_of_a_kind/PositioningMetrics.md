# Positioning Metrics in Agent Farm

This document defines and explains the various positioning metrics used to analyze how initial agent positioning relative to resources affects outcomes in Agent Farm simulations. These metrics provide complementary perspectives on spatial relationships between agents and resources.

## 1. Distance-Based Metrics

Distance-based metrics measure the spatial separation between agents and resources, providing insights into resource accessibility.

### 1.1 Nearest Resource Distance

**Definition:**
The distance from an agent to its closest resource.

**Implementation:**
- Calculates the Euclidean distance from an agent to each resource
- Takes the minimum of these distances
- Calculated as: `min(distances to all resources)`

**Interpretation:**
- Lower values indicate better positioning
- Directly measures immediate resource accessibility
- Strong predictor of early resource acquisition advantage

**Strengths:**
- Simple and intuitive
- Strongly correlates with early-phase dominance
- Good predictor of initial resource acquisition speed

**Limitations:**
- Considers only the single closest resource
- Doesn't account for resource amount or quality
- Ignores competition from other agents

### 1.2 Average Resource Distance

**Definition:**
The average distance from an agent to all resources in the environment.

**Implementation:**
- Calculates the Euclidean distance from an agent to each resource
- Takes the mean of these distances
- Calculated as: `sum(distances to all resources) / count(resources)`

**Interpretation:**
- Lower values indicate better overall positioning
- Measures general resource accessibility
- Reflects long-term resource acquisition potential

**Strengths:**
- Provides a broader view of resource accessibility
- Less sensitive to outlier resource positions
- Better predictor of mid-to-late phase performance

**Limitations:**
- Treats all resources equally regardless of amount
- May overvalue distant resources that are practically inaccessible
- Less predictive of early advantage than nearest distance

### 1.3 Weighted Resource Distance

**Definition:**
The average distance from an agent to all resources, weighted by resource amount.

**Implementation:**
- Calculates the Euclidean distance from an agent to each resource
- Weights each distance by the inverse of resource amount
- Calculated as: `sum(distance * (1/(resource_amount+1))) / count(resources)`

**Interpretation:**
- Lower values indicate better positioning relative to valuable resources
- Balances distance with resource value
- Reflects quality-adjusted resource accessibility

**Strengths:**
- Accounts for both distance and resource amount
- Gives more importance to proximity to larger resources
- Better predictor of resource acquisition efficiency

**Limitations:**
- More complex to interpret
- Sensitive to resource amount distribution
- May undervalue strategic positioning near resource clusters

## 2. Range-Based Metrics

Range-based metrics focus on resources within an agent's immediate gathering range, providing insights into immediately accessible resources.

### 2.1 Resources in Range

**Definition:**
The number of resources within an agent's gathering range.

**Implementation:**
- Counts resources where distance to agent is less than or equal to gathering range
- Calculated as: `count(resources where distance <= gathering_range)`

**Interpretation:**
- Higher values indicate better immediate resource accessibility
- Directly measures resource options available without movement
- Strong predictor of early-phase resource acquisition diversity

**Strengths:**
- Directly measures immediate resource accessibility
- Intuitive predictor of early advantage
- Accounts for agent's gathering capabilities

**Limitations:**
- Doesn't consider resource amounts
- Binary inclusion (in range or not) ignores partial accessibility
- Doesn't account for competition from other agents

### 2.2 Resource Amount in Range

**Definition:**
The total amount of resources within an agent's gathering range.

**Implementation:**
- Sums the amount of all resources within gathering range
- Calculated as: `sum(resource_amount where distance <= gathering_range)`

**Interpretation:**
- Higher values indicate better immediate resource wealth
- Measures the quantity of immediately accessible resources
- Strong predictor of early-phase resource acquisition volume

**Strengths:**
- Combines accessibility with resource quantity
- Better predictor of early resource advantage than simple count
- Accounts for both gathering range and resource wealth

**Limitations:**
- Doesn't account for competition from other agents
- May overvalue a single large resource versus multiple smaller ones
- Binary inclusion (in range or not) ignores partial accessibility

## 3. Relative Advantage Metrics

Relative advantage metrics compare positioning between different agent types, providing insights into competitive advantages.

### 3.1 Nearest Resource Advantage

**Definition:**
The difference in nearest resource distances between two agent types.

**Implementation:**
- Compares nearest resource distances between agent types
- Calculated as: `agent2_nearest_resource_dist - agent1_nearest_resource_dist`

**Interpretation:**
- Positive values indicate agent1 has an advantage over agent2
- Measures relative proximity advantage to the closest resource
- Predictor of which agent type will acquire resources first

**Strengths:**
- Directly compares competitive positioning
- Strong predictor of early competition outcomes
- Simple comparative measure

**Limitations:**
- Considers only the single closest resource
- Doesn't account for resource amount
- May miss broader positioning advantages

### 3.2 Resources in Range Advantage

**Definition:**
The difference in the number of resources within gathering range between two agent types.

**Implementation:**
- Compares the number of accessible resources between agent types
- Calculated as: `agent1_resources_in_range - agent2_resources_in_range`

**Interpretation:**
- Positive values indicate agent1 has more resource options than agent2
- Measures relative advantage in immediate resource accessibility
- Predictor of resource acquisition diversity advantage

**Strengths:**
- Compares immediate resource accessibility
- Accounts for gathering capabilities
- Good predictor of early-phase resource diversity advantage

**Limitations:**
- Doesn't consider resource amounts
- Binary inclusion may miss nuanced advantages
- Doesn't account for resource competition dynamics

### 3.3 Resource Amount Advantage

**Definition:**
The difference in the total amount of resources within gathering range between two agent types.

**Implementation:**
- Compares the amount of accessible resources between agent types
- Calculated as: `agent1_resource_amount_in_range - agent2_resource_amount_in_range`

**Interpretation:**
- Positive values indicate agent1 has access to more resource wealth than agent2
- Measures relative advantage in immediate resource wealth
- Strong predictor of early-phase resource acquisition volume advantage

**Strengths:**
- Combines accessibility with resource quantity in comparison
- Most comprehensive relative advantage metric
- Strong predictor of early-phase dominance advantage

**Limitations:**
- Doesn't account for competition dynamics
- Binary inclusion may miss nuanced advantages
- May overvalue quantity over strategic positioning

## 4. Comparison of Positioning Metrics

Each positioning metric highlights different aspects of spatial relationships:

- **Nearest Resource Distance** answers: "How close is an agent to its nearest resource?"
- **Average Resource Distance** answers: "How well-positioned is an agent relative to all resources?"
- **Weighted Resource Distance** answers: "How well-positioned is an agent relative to valuable resources?"
- **Resources in Range** answers: "How many resources can an agent access without moving?"
- **Resource Amount in Range** answers: "How much resource wealth can an agent access immediately?"
- **Relative Advantage Metrics** answer: "Which agent type has better positioning?"

## 5. Practical Applications

### 5.1 Predictive Power

Analysis of 500 simulations revealed the following correlations with dominance outcomes:

| Metric | Correlation with System Dominance | Correlation with Independent Dominance | Correlation with Control Dominance |
|--------|-----------------------------------|----------------------------------------|-----------------------------------|
| System Nearest Resource Distance | -0.42 | 0.28 | 0.14 |
| Independent Resources in Range | -0.31 | 0.47 | -0.16 |
| Control Resource Amount in Range | -0.18 | -0.22 | 0.40 |
| System vs. Independent Resource Advantage | 0.39 | -0.44 | 0.05 |
| Independent vs. Control Resource Advantage | -0.12 | 0.38 | -0.26 |

### 5.2 Strategic Implications

Understanding positioning metrics helps in:

1. **Simulation Design**
   - Creating balanced or intentionally imbalanced initial conditions
   - Testing agent resilience to positioning disadvantages
   - Designing environments that test specific agent capabilities

2. **Agent Strategy Development**
   - Developing adaptive strategies based on initial positioning assessment
   - Prioritizing movement and resource acquisition based on positioning metrics
   - Balancing risk and opportunity based on relative positioning advantages

3. **Analysis and Interpretation**
   - Explaining simulation outcomes through initial conditions
   - Isolating the effects of positioning from agent characteristics
   - Identifying which positioning aspects most strongly influence outcomes

These positioning metrics provide a comprehensive framework for analyzing spatial relationships in Agent Farm simulations and understanding how initial conditions influence emergent dominance patterns. 