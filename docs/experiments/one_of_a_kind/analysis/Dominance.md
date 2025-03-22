# Dominance Analysis

1. **Population Dominance**
   - Determines which agent type dominates by having the highest population count at the end of a simulation
   - Simple measure based on final step counts

2. **Survival Dominance**
   - Identifies which agent type has the highest average survival time
   - Calculates how long agents of each type survive on average (death_time - birth_time)
   - For agents still alive at the end, uses the final step as a proxy

3. **Comprehensive Dominance**
   - A sophisticated measure that considers the entire simulation history using multiple metrics:
     - Area Under the Curve (AUC): Total agent-steps throughout the simulation
     - Recency-weighted AUC: Gives more weight to later steps in the simulation
     - Dominance duration: How many steps each agent type was dominant
     - Growth trend: Positive growth trends in the latter half of simulation
     - Final population ratio: The proportion of agents at the end of simulation

4. **Dominance Switches**
   - Tracks how often agent types switch dominance during a simulation
   - Calculates:
     - Total number of dominance switches
     - Average duration of dominance periods for each agent type
     - Volatility of dominance (frequency of switches in different phases)
     - Transition matrix showing which agent types tend to take over from others

## Data Analysis Capabilities

1. **Final Population Counts** - End-state agent counts by type
2. **Agent Survival Statistics** - Detailed survival metrics for each agent type
3. **Initial Positions and Resources** - Starting conditions analysis
4. **Reproduction Statistics** - Analysis of reproduction events and patterns

## Visualization Capabilities

1. **Dominance Distribution** - Shows percentage distribution of different dominance types
2. **Feature Importance** - Visualizes which factors contribute most to dominance
3. **Resource Proximity vs. Dominance** - Analyzes how resource proximity affects dominance
4. **Reproduction vs. Dominance** - Shows relationship between reproduction rates and dominance
5. **Correlation Matrix** - Visualizes correlations between various factors and dominance
6. **Dominance Comparison** - Compares different dominance measures
7. **Dominance Switches** - Visualizes frequency and patterns of dominance changes
8. **Dominance Stability** - Analyzes how stable dominance is across simulations
9. **Reproduction Advantage Stability** - Shows stability of reproduction advantages
10. **Comprehensive Score Breakdown** - Visualizes components of the comprehensive dominance score

This dominance analysis framework provides a robust set of tools for understanding which agent types dominate in your simulations, why they dominate, and how stable that dominance is across different simulation conditions.
