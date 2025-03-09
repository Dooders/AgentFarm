# Agent Dominance Analysis Report

## Executive Summary

This report analyzes the dominance patterns observed across 250 simulations in the Agent Farm environment. Three agent types (System, Independent, and Control) competed for resources and reproduction opportunities. The analysis reveals:

- **System agents** emerged as the most dominant type overall (44% comprehensive dominance), followed by Control agents (36.4%) and Independent agents (19.6%).
- **Dominance switching** occurred on average 8.26 times per simulation, with most switches happening in the early phase.
- **System agents** maintained the longest periods of dominance (436 steps on average), suggesting greater stability once they achieve dominance.
- **Initial resource proximity** and **reproduction strategies** appear to be key factors influencing which agent type becomes dominant.

## 1. Dominance Distribution

Three different measures of dominance were analyzed:

| Dominance Measure | System | Independent | Control |
|-------------------|--------|-------------|---------|
| Population Dominance | 42.8% | 21.2% | 36.0% |
| Survival Dominance | 24.4% | 48.0% | 27.6% |
| Comprehensive Dominance | 44.0% | 19.6% | 36.4% |

Key observations:
- **System agents** excel at population growth, leading in both population and comprehensive dominance measures.
- **Independent agents** show superior survival skills (48% survival dominance) but struggle to convert this into overall dominance.
- **Control agents** maintain a balanced performance across all dominance measures.

The comprehensive dominance measure, which considers multiple factors including population growth, recency-weighted presence, and dominance duration, provides the most holistic view of agent performance.

![Dominance Distribution](images/dominance_distribution.png)
*Figure 1: Distribution of dominance across agent types for different dominance measures.*

![Dominance Comparison](images/dominance_comparison.png)
*Figure 2: Comparison of different dominance measures and their relationships.*

## 2. Dominance Switching Patterns

### 2.1 Frequency and Timing

- **Average switches per simulation**: 8.26
- **Average switches per step**: 0.0041 (approximately 1 switch every 244 steps)

Switches by simulation phase:
- **Early phase**: 4.91 switches (59.4% of all switches)
- **Middle phase**: 1.84 switches (22.3% of all switches)
- **Late phase**: 1.52 switches (18.4% of all switches)

This pattern suggests that dominance is typically established early and becomes more stable as simulations progress.

![Dominance Switches Distribution](images/dominance_switches_distribution.png)
*Figure 3: Distribution of the number of dominance switches across simulations.*

![Phase Switches](images/phase_switches.png)
*Figure 4: Average number of dominance switches by simulation phase.*

### 2.2 Dominance Period Duration

Average number of steps each agent type maintained dominance when they were dominant:

- **System**: 436.47 steps
- **Independent**: 195.17 steps
- **Control**: 360.20 steps

System agents demonstrate the most stable dominance, maintaining control for more than twice as long as Independent agents on average.

![Average Dominance Period](images/avg_dominance_period.png)
*Figure 5: Average duration of dominance periods by agent type.*

### 2.3 Transition Probabilities

When dominance switches occur, the following transition patterns were observed:

From System agents:
- To Independent agents: 42%
- To Control agents: 44%

From Independent agents:
- To System agents: 29%
- To Control agents: 22%

From Control agents:
- To System agents: 32%
- To Independent agents: 19%

These probabilities suggest:
- System agents lose dominance almost equally to both other agent types
- Independent agents are more likely to lose dominance to System agents
- Control agents are more likely to lose dominance to System agents

![Dominance Transitions](images/dominance_transitions.png)
*Figure 6: Transition probabilities between agent types when dominance switches occur.*

### 2.4 Dominance Switching by Final Dominant Type

Average number of dominance switches in simulations where each agent type was ultimately dominant:

- System: 7.54 switches
- Independent: 9.12 switches
- Control: 8.68 switches

Simulations where System agents ultimately dominated showed fewer switches, suggesting they establish stable dominance more efficiently.

## 3. Factors Influencing Dominance

### 3.1 Resource Proximity

The analysis identified a correlation between initial resource proximity and dominance outcomes:

- **Independent agents** benefit most from having more resources in range at the start
- The factor most associated with increased dominance switching was `independentagent_resource_amount_in_range` (correlation: 0.117)

This suggests that when Independent agents start with more resources nearby, it creates more competitive and volatile simulations with more frequent dominance changes.

### 3.2 Predictive Features for Population Dominance

The most important features for predicting population dominance were:

1. Control agent final ratio (14.1%)
2. System agent final ratio (10.3%)
3. Independent agent final ratio (8.2%)
4. Control agent dominance score (7.1%)
5. System agent dominance score (6.6%)

The classifier achieved 96% accuracy in predicting population dominance, indicating that the identified features are highly reliable predictors.

![Population Dominance Feature Importance](images/population_dominance_feature_importance.png)
*Figure 7: Importance of features for predicting population dominance.*

![Population Dominance Correlation Matrix](images/population_dominance_correlation_matrix.png)
*Figure 8: Correlation matrix showing relationships between features and population dominance.*

### 3.3 Predictive Features for Survival Dominance

The most important features for predicting survival dominance were:

1. Independent agent dominance score (6.5%)
2. Independent agent recency-weighted AUC (5.3%)
3. Independent agent AUC (4.8%)
4. System agent nearest resource distance (3.6%)
5. Control agent dominance score (3.3%)

The survival dominance classifier achieved 64% accuracy, suggesting that survival patterns are more complex and influenced by a wider range of factors than population dominance.

![Survival Dominance Feature Importance](images/survival_dominance_feature_importance.png)
*Figure 9: Importance of features for predicting survival dominance.*

![Survival Dominance Correlation Matrix](images/survival_dominance_correlation_matrix.png)
*Figure 10: Correlation matrix showing relationships between features and survival dominance.*

## 4. Stability and Final Dominance

The analysis of dominance stability (inverse of switches per step) revealed:

- Simulations with higher stability tend to have clearer dominance outcomes
- System agents benefit most from stable conditions
- Independent agents show more success in volatile environments with frequent dominance changes

![Dominance Stability Analysis](images/dominance_stability_analysis.png)
*Figure 11: Relationship between dominance stability and final dominance scores for each agent type.*

## 5. Conclusions and Implications

1. **System agents** demonstrate superior overall performance, particularly in establishing and maintaining dominance for extended periods.

2. **Independent agents** excel at survival but struggle to convert this advantage into population dominance, suggesting they may prioritize individual survival over reproduction.

3. **Control agents** show balanced performance across metrics, maintaining competitive presence throughout simulations.

4. **Early simulation dynamics** are critical in determining the ultimate dominant agent type, with most dominance switches occurring in the early phase.

5. **Resource proximity** at simulation start significantly influences competitive dynamics and the frequency of dominance switches.

6. **Stability favors System agents**, while volatility creates opportunities for Independent agents to temporarily gain dominance.

These findings suggest that in competitive multi-agent environments:

- Early advantage is crucial for long-term dominance
- Balanced resource allocation strategies outperform purely survival-focused approaches
- The ability to maintain dominance once achieved is as important as the ability to achieve it initially

## 6. Recommendations for Future Analysis

1. Investigate the specific mechanisms that allow System agents to maintain longer periods of dominance

2. Analyze the relationship between reproduction strategies and dominance switching patterns

3. Explore how different environmental configurations might alter the dominance dynamics between agent types

4. Examine whether hybrid strategies combining strengths of different agent types could outperform pure strategies 