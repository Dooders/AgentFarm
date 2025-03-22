# Social Behavior Analysis

## Core Analysis Categories

### 1. Social Network Analysis
- **Network Structure**: Analyzes the connections between agents based on their interactions
- **Connection Metrics**: Calculates metrics like:
  - Network density (how interconnected agents are)
  - Average outgoing/incoming connections per agent type
  - Unique interaction pairs
- **Agent Type Relationships**: Examines how different agent types interact with each other

### 2. Resource Sharing Patterns
- **Sharing Metrics**: Tracks:
  - Total resources shared between agents
  - Number of sharing actions
  - Average resources per sharing action
- **Altruistic Sharing**: Identifies instances where agents share resources without immediate benefit
- **Sharing Matrix**: Analyzes which agent types share with which other types
- **Temporal Distribution**: Examines how resource sharing evolves over time

### 3. Cooperation vs. Competition
- **Action Classification**: Categorizes agent actions as cooperative or competitive
- **Cooperation-Competition Ratio**: Calculates the balance between cooperative and competitive behaviors
- **Agent Type Analysis**: Compares how different agent types balance cooperation and competition
- **Temporal Trends**: Tracks how cooperation/competition patterns change throughout the simulation

### 4. Spatial Clustering
- **Cluster Detection**: Identifies groups of agents that cluster together spatially
- **Cluster Composition**: Analyzes the agent type makeup of each cluster
- **Diversity Index**: Calculates how diverse each cluster is (using Shannon entropy)
- **Isolation Metrics**: Tracks which agents tend to remain isolated vs. clustered
- **Agent Type Clustering**: Examines which agent types tend to form or join clusters

### 5. Reproduction Social Patterns
- **Social Context**: Analyzes whether reproduction occurs in:
  - Isolation
  - Homogeneous groups (same agent type)
  - Heterogeneous groups (mixed agent types)
- **Agent Type Reproduction**: Compares reproduction rates across different agent types
- **Social Influence**: Examines how social context affects reproduction success

## Analysis Capabilities

### Cross-Simulation Analysis
- Aggregates social behavior metrics across multiple simulations
- Identifies consistent patterns that emerge regardless of initial conditions
- Calculates average metrics and their variance across simulations

### Pattern Extraction
- Identifies emergent social patterns like:
  - Cooperation networks
  - Resource sharing communities
  - Competitive hierarchies
  - Spatial segregation or integration

### Insight Generation
- Automatically extracts key insights about social dynamics
- Identifies which agent types are most social, cooperative, or competitive
- Recognizes unusual or unexpected social behaviors

### Visualization
1. **Social Network Visualizations**:
   - Agent connection graphs
   - Network density charts
   - Connection metrics by agent type

2. **Resource Sharing Visualizations**:
   - Sharing patterns by agent type
   - Sharing matrix heatmaps
   - Temporal distribution of sharing

3. **Cooperation/Competition Visualizations**:
   - Cooperation vs. competition pie charts
   - Ratio analysis by agent type
   - Temporal trends in cooperation/competition

4. **Spatial Clustering Visualizations**:
   - Cluster composition pie charts
   - Diversity vs. cluster size scatter plots
   - Clustering ratios by agent type

5. **Reproduction Pattern Visualizations**:
   - Social context of reproduction
   - Reproduction events by agent type
   - Social context breakdown by agent type

### Reporting
The system generates comprehensive reports that include:
- Key findings and insights
- Detailed metrics across all social dimensions
- Agent type-specific insights
- Emergent patterns
- Recommendations for further investigation

## Applications

This social behavior analysis framework allows you to:

1. **Understand Emergent Social Structures**: See how complex social networks and behaviors emerge from simple agent rules

2. **Compare Agent Types**: Analyze how different agent types (system, independent, control) interact socially and which develop advantages

3. **Identify Successful Strategies**: Determine which social behaviors correlate with survival and reproduction success

4. **Track Social Evolution**: Observe how social structures and behaviors evolve over the course of simulations

5. **Detect Unexpected Patterns**: Identify surprising or emergent social behaviors that weren't explicitly programmed

The analysis script provides a command-line interface to run this analysis on your simulation data, generating visualizations and a comprehensive report that summarizes all social behavior patterns observed in your simulations.
