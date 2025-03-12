# Dominance Measures in Agent Farm

This document describes the different dominance measures used to analyze agent performance in the Agent Farm simulations. These measures provide complementary perspectives on how different agent types establish and maintain dominance throughout simulations.

## 1. Population Dominance

Population dominance is the simplest measure, focusing solely on which agent type has the highest population count at the end of a simulation.

**Implementation:**
- Examines only the final simulation step
- Counts the number of agents of each type
- The agent type with the highest count is considered dominant

**Strengths:**
- Simple and intuitive
- Directly reflects which agent type was most successful at surviving and reproducing

**Limitations:**
- Ignores the entire history of the simulation
- Can be heavily influenced by late-stage random events
- Doesn't account for survival efficiency or resource utilization

## 2. Survival Dominance

Survival dominance measures which agent type has the highest average survival time across all agents born during the simulation.

**Implementation:**
- For each agent, calculates survival time as (death_time - birth_time)
- For agents still alive at simulation end, uses the final step as a proxy for death_time
- Computes the average survival time for each agent type
- The agent type with the highest average survival time is considered dominant

**Strengths:**
- Focuses on longevity and survival efficiency
- Rewards agent types that make better use of resources
- Less influenced by reproduction rates

**Limitations:**
- Doesn't directly account for population growth
- An agent type could have excellent survival but fail to reproduce effectively
- May favor conservative strategies over expansive ones

## 3. Comprehensive Dominance

Comprehensive dominance is a composite measure that considers multiple factors throughout the entire simulation history.

**Implementation:**
Combines five weighted metrics:

1. **Area Under the Curve (AUC)** (20% weight)
   - Sums the total agent-steps throughout the simulation
   - Represents overall population persistence

2. **Recency-weighted AUC** (30% weight)
   - Similar to AUC but gives more weight to later steps in the simulation
   - Emphasizes sustained or growing dominance

3. **Dominance Duration** (20% weight)
   - Counts how many steps each agent type was the most numerous
   - Rewards consistent leadership

4. **Growth Trend** (10% weight)
   - Measures positive growth trends in the latter half of simulation
   - Rewards agent types that are improving over time

5. **Final Population Ratio** (20% weight)
   - The proportion of agents of each type at the end of simulation
   - Similar to population dominance but as a ratio

All metrics are normalized to a [0,1] scale before being combined into the final score.

**Strengths:**
- Provides the most holistic view of agent performance
- Balances immediate success with long-term sustainability
- Considers both historical performance and final outcomes
- Less susceptible to simulation anomalies

**Limitations:**
- More complex to interpret
- Weighting of components introduces subjective priorities
- May obscure trade-offs between different success strategies

## Comparison of Dominance Measures
Each dominance measure highlights different aspects of agent performance:

- **Population Dominance** answers: "Which agent type had the most individuals at the end?"
- **Survival Dominance** answers: "Which agent type lived the longest on average?"
- **Comprehensive Dominance** answers: "Which agent type performed best overall throughout the simulation?"

The comprehensive measure is generally preferred for overall analysis, while the other measures provide valuable insights into specific aspects of agent performance.

## Comprehensive Score Breakdown Analysis

The comprehensive dominance score can be further analyzed by examining the contribution of each component to the final score for each agent type. This breakdown reveals which specific aspects of performance drive an agent type's overall success.

![Comprehensive Score Breakdown](/docs/experiments/one_of_a_kind/images/comprehensive_score_breakdown.png)

### Key Insights from Score Breakdown

1. **System Agents**
   - Typically excel in **Dominance Duration** and **Recency-weighted AUC**, indicating they maintain numerical superiority for longer periods and are particularly strong in later simulation stages
   - Their high scores in these areas reflect their ability to establish and maintain stable dominance once achieved
   - Often show moderate to strong **Growth Trend** scores, suggesting continued improvement over time

2. **Independent Agents**
   - Often score highest in **AUC** relative to their other metrics, indicating they maintain a consistent presence throughout simulations
   - Typically show lower **Final Ratio** scores, suggesting they struggle to maintain population advantages at simulation end
   - Their **Growth Trend** scores are frequently lower, indicating difficulty in improving their position in later simulation stages

3. **Control Agents**
   - Generally show balanced contributions across all five components
   - Often perform well in **Final Ratio**, indicating strong end-game performance
   - Their **Recency-weighted AUC** scores typically reflect steady performance in later simulation stages

### Interpreting Component Contributions

The relative contribution of each component to an agent type's comprehensive score provides insights into their simulation strategy:

- **High AUC + Low Recency-weighted AUC**: 
  - Strong early but declining performance
- **Low AUC + High Recency-weighted AUC**: 
  - Weak start but improving performance
- **High Dominance Duration + Low Final Ratio**: 
  - Maintained leadership for long periods but lost it by the end
- **Low Dominance Duration + High Final Ratio**: 
  - Achieved dominance late in the simulation
- **High Growth Trend**: 
  - Demonstrated improving performance in the latter half of the simulation

This breakdown helps identify not just which agent type performed best overall, but also the specific aspects of performance that contributed to their success or failure, providing deeper insights into agent behavior and competitive dynamics.