# AI & Machine Learning

![Feature](https://img.shields.io/badge/feature-AI%20%26%20ML-purple)

## Table of Contents

1. [Overview](#overview)
   - [Why AI & ML in Agent-Based Modeling?](#why-ai--ml-in-agent-based-modeling)
2. [Core Capabilities](#core-capabilities)
   - [1. Reinforcement Learning for Agent Adaptation](#1-reinforcement-learning-for-agent-adaptation)
     - [Deep Q-Learning (DQN)](#deep-q-learning-dqn)
     - [Additional RL Algorithms](#additional-rl-algorithms)
     - [Curriculum Learning](#curriculum-learning)
   - [2. Automated Data Analysis and Insight Generation](#2-automated-data-analysis-and-insight-generation)
     - [Automated Statistical Analysis](#automated-statistical-analysis)
     - [Machine Learning-Based Analysis](#machine-learning-based-analysis)
     - [Behavioral Clustering](#behavioral-clustering)
     - [Automated Report Generation](#automated-report-generation)
   - [3. Pattern Recognition and Behavior Prediction](#3-pattern-recognition-and-behavior-prediction)
     - [Sequence Pattern Analysis](#sequence-pattern-analysis)
     - [Predictive Modeling](#predictive-modeling)
     - [Survival Prediction](#survival-prediction)
   - [4. Evolutionary Algorithms and Genetic Modeling](#4-evolutionary-algorithms-and-genetic-modeling)
     - [Genome System](#genome-system)
     - [Genetic Operators](#genetic-operators)
     - [Evolutionary Simulation](#evolutionary-simulation)
     - [Genome Embeddings](#genome-embeddings)
3. [Advanced ML Techniques](#advanced-ml-techniques)
   - [Transfer Learning](#transfer-learning)
   - [Multi-Agent Reinforcement Learning (MARL)](#multi-agent-reinforcement-learning-marl)
   - [Imitation Learning](#imitation-learning)
4. [Performance Optimization](#performance-optimization)
   - [GPU Acceleration](#gpu-acceleration)
   - [Batch Training](#batch-training)
5. [Example: Complete ML-Enhanced Simulation](#example-complete-ml-enhanced-simulation)
6. [Additional Resources](#additional-resources)
   - [Documentation](#documentation)
   - [Examples](#examples)
   - [Research Resources](#research-resources)
7. [Support](#support)

---

## Overview

AgentFarm integrates advanced AI and machine learning capabilities to enable intelligent agent behaviors, automated analysis, and sophisticated pattern recognition. From reinforcement learning for adaptive agents to evolutionary algorithms for population dynamics, AgentFarm provides a comprehensive ML toolkit for agent-based modeling research.

### Why AI & ML in Agent-Based Modeling?

Machine learning enhances agent-based simulations by:
- **Adaptive Behavior**: Agents learn optimal strategies through experience
- **Complex Decision-Making**: Handle high-dimensional state spaces
- **Pattern Discovery**: Automatically identify emergent behaviors
- **Predictive Modeling**: Forecast system dynamics and outcomes
- **Automated Analysis**: Extract insights from large-scale simulations

---

## Core Capabilities

### 1. Reinforcement Learning for Agent Adaptation

AgentFarm provides multiple reinforcement learning algorithms that enable agents to learn and adapt through interaction with their environment.

#### Deep Q-Learning (DQN)

The primary RL algorithm in AgentFarm uses Deep Q-Networks with several enhancements:

**Key Features:**
- **Double Q-Learning**: Reduces overestimation bias
- **Experience Replay**: Breaks correlation in training data
- **Target Networks**: Stabilizes learning
- **Soft Updates**: Gradual target network synchronization
- **Shared Feature Extraction**: Efficient multi-task learning

**Basic DQN Usage:**

```python
from farm.core.decision.base_dqn import BaseDQNConfig, BaseDQNModule

# Configure DQN
config = BaseDQNConfig(
    learning_rate=0.001,
    gamma=0.99,              # Discount factor
    epsilon_start=1.0,       # Initial exploration rate
    epsilon_min=0.01,        # Minimum exploration rate
    epsilon_decay=0.995,     # Exploration decay
    memory_size=10000,       # Experience replay buffer size
    batch_size=32,           # Training batch size
    target_update_freq=100,  # Target network update frequency
    tau=0.005               # Soft update parameter
)

# Create DQN module
dqn = BaseDQNModule(
    input_dim=state_dimension,
    output_dim=num_actions,
    config=config
)

# Training loop
for episode in range(num_episodes):
    state = environment.reset()
    done = False
    
    while not done:
        # Select action
        action = dqn.select_action(state)
        
        # Execute action
        next_state, reward, done = environment.step(action)
        
        # Store experience
        dqn.store_experience(state, action, reward, next_state, done)
        
        # Train if enough experiences
        if len(dqn.memory) >= config.batch_size:
            batch = random.sample(dqn.memory, config.batch_size)
            dqn.train(batch)
        
        state = next_state
```

**Hierarchical Action Selection:**

AgentFarm uses specialized DQN modules for different action types:

```python
from farm.core.agent import BaseAgent

class LearningAgent(BaseAgent):
    """Agent with hierarchical RL decision-making."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Specialized modules for different actions
        self.move_module = MoveDQNModule(config)
        self.attack_module = AttackDQNModule(config)
        self.gather_module = GatherDQNModule(config)
        self.share_module = ShareDQNModule(config)
        
    def decide_action(self):
        """Multi-level decision making."""
        # Level 1: Select action type
        state = self.get_state_tensor()
        action_type = self.select_action_type(state)
        
        # Level 2: Select specific action parameters
        if action_type == "move":
            direction = self.move_module.select_action(state)
            return {"action_type": "move", "direction": direction}
        elif action_type == "attack":
            target = self.attack_module.select_target(state)
            return {"action_type": "attack", "target": target}
        # ... other action types
        
    def update_learning(self, action, reward, next_state):
        """Update all relevant modules."""
        # Update the module that was used
        if action["action_type"] == "move":
            self.move_module.store_experience(
                self.last_state, 
                action["direction"], 
                reward, 
                next_state, 
                False
            )
            if len(self.move_module.memory) >= self.move_module.config.batch_size:
                batch = random.sample(
                    self.move_module.memory, 
                    self.move_module.config.batch_size
                )
                self.move_module.train(batch)
```

**Shared Feature Extraction:**

Improve learning efficiency with shared encoders:

> **Note**: The `SharedEncoder` class is planned for a future release. Currently, shared feature extraction can be implemented using PyTorch's module sharing or custom neural network architectures.

```python
import torch.nn as nn

# Create shared encoder manually
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.encoder(x)

# Create shared encoder
shared_encoder = SharedEncoder(state_dimension, 64)

# Use across multiple modules (manually sharing the encoder)
move_module = MoveDQNModule(
    input_dim=state_dimension,
    output_dim=4,
    config=config,
    shared_encoder=shared_encoder
)

attack_module = AttackDQNModule(
    input_dim=state_dimension,
    output_dim=5,
    config=config,
    shared_encoder=shared_encoder  # Reuse same encoder
)
```

#### Additional RL Algorithms

AgentFarm supports multiple RL algorithms through integration with Stable Baselines3:

**Proximal Policy Optimization (PPO):**
```python
from farm.core.decision import DecisionConfig, DecisionModule

# Configure PPO
config = DecisionConfig(
    algorithm_type="ppo",
    rl_state_dim=8,
    algorithm_params={
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01
    }
)

# Create decision module
decision = DecisionModule(agent=agent, config=config)

# PPO automatically handles:
# - Policy optimization
# - Value function learning
# - Advantage estimation
# - Entropy regularization
```

**Soft Actor-Critic (SAC):**
```python
config = DecisionConfig(
    algorithm_type="sac",
    rl_state_dim=8,
    algorithm_params={
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'ent_coef': 'auto',  # Automatic entropy tuning
    }
)

# SAC features:
# - Entropy-regularized learning
# - Off-policy training
# - Continuous action support
# - Maximum entropy objective
```

**Available RL Algorithms:**
- **PPO** (Proximal Policy Optimization): On-policy, good for most tasks
- **SAC** (Soft Actor-Critic): Off-policy, maximum entropy
- **A2C** (Advantage Actor-Critic): Synchronous actor-critic
- **TD3** (Twin Delayed DDPG): Deterministic policy gradient
- **DQN** (Deep Q-Network): Value-based, discrete actions

#### Curriculum Learning

Gradually increase task complexity for better learning:

```python
from farm.config import SimulationConfig

# Define curriculum phases
config = SimulationConfig(
    curriculum_phases=[
        {
            'steps': 100,
            'enabled_actions': ['move', 'gather'],
            'description': 'Learn basic movement and resource gathering'
        },
        {
            'steps': 200,
            'enabled_actions': ['move', 'gather', 'share'],
            'description': 'Add cooperative sharing'
        },
        {
            'steps': 300,
            'enabled_actions': ['move', 'gather', 'share', 'attack'],
            'description': 'Introduce competition'
        },
        {
            'steps': -1,  # Unlimited
            'enabled_actions': ['move', 'gather', 'share', 'attack', 'reproduce'],
            'description': 'Full action space'
        }
    ]
)

# Curriculum is automatically applied during simulation
# - Agents focus on simpler tasks first
# - Gradually build complex behaviors
# - Better convergence and stability
```

---

### 2. Automated Data Analysis and Insight Generation

AgentFarm provides AI-powered analysis tools that automatically extract insights from simulation data.

#### Automated Statistical Analysis

```python
from farm.analysis.service import AnalysisService, AnalysisRequest
from pathlib import Path

# Initialize analysis service
service = AnalysisService(config_service)

# Request automated analysis
request = AnalysisRequest(
    module_name="comprehensive",
    experiment_path=Path("experiments/cooperation_study"),
    output_path=Path("results/automated_analysis"),
    group="all"  # Run all analysis modules
)

# Run analysis
result = service.run(request)

# Automated analysis includes:
# - Population dynamics
# - Resource utilization patterns
# - Agent interaction networks
# - Behavioral clustering
# - Statistical summaries
# - Comparative metrics
```

#### Machine Learning-Based Analysis

Use ML to discover patterns in simulation data:

```python
from farm.analysis.dominance.ml import train_classifier, prepare_features_for_classification
import pandas as pd

# Load simulation data
df = pd.read_csv("simulation_results.csv")

# Prepare features
X, feature_cols, exclude_cols = prepare_features_for_classification(df)
y = df['agent_dominance']  # Target variable

# Train classifier to predict dominance
classifier, feature_importances = train_classifier(X, y, "Agent Dominance")

# Automatically generates:
# - Classification report
# - Confusion matrix
# - Feature importance rankings
# - Predictions for new data

# Top features influencing dominance
print("\nKey Factors for Dominance:")
for feature, importance in feature_importances[:10]:
    print(f"  {feature}: {importance:.3f}")
```

#### Behavioral Clustering

Automatically group agents by behavior:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cluster_agent_behaviors(simulation_db: str, n_clusters: int = 5):
    """Automatically identify behavioral patterns."""
    
    # Extract behavioral features
    features = extract_behavioral_features(simulation_db)
    # Features include:
    # - Action frequency distributions
    # - Movement patterns
    # - Resource gathering efficiency
    # - Social interaction rates
    # - Combat engagement patterns
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster agents
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    for i in range(n_clusters):
        cluster_agents = features[clusters == i]
        print(f"\nCluster {i} ({len(cluster_agents)} agents):")
        print(f"  Characteristics:")
        for feature_name, values in cluster_agents.items():
            print(f"    {feature_name}: {values.mean():.3f} ± {values.std():.3f}")
    
    return clusters, kmeans

# Use clustering
clusters, model = cluster_agent_behaviors("simulation.db", n_clusters=4)

# Possible cluster interpretations:
# Cluster 0: "Aggressive competitors" - High attack, low sharing
# Cluster 1: "Cooperative gatherers" - High gathering, high sharing
# Cluster 2: "Cautious explorers" - High movement, low combat
# Cluster 3: "Resource hoarders" - High gathering, low sharing, low movement
```

#### Automated Report Generation

Generate comprehensive analysis reports automatically:

```python
from farm.charts.chart_analyzer import ChartAnalyzer

# Initialize analyzer with LLM
analyzer = ChartAnalyzer(
    model_name="gpt-4",
    api_key="your-api-key"
)

# Generate automated insights
insights = analyzer.analyze_simulation_results(
    simulation_db="simulation.db",
    charts_dir="charts/",
    analysis_focus=[
        "population_dynamics",
        "resource_competition",
        "emergent_cooperation",
        "survival_strategies"
    ]
)

# Report includes:
# - Natural language summaries
# - Key findings and trends
# - Statistical significance tests
# - Visualizations with explanations
# - Recommendations for follow-up experiments

print(insights['executive_summary'])
print("\nKey Findings:")
for finding in insights['key_findings']:
    print(f"  - {finding}")
```

---

### 3. Pattern Recognition and Behavior Prediction

Use ML to identify patterns and predict future behaviors.

#### Sequence Pattern Analysis

Identify common behavioral sequences:

> **Note**: Advanced sequence pattern analysis with automatic discovery of behavioral motifs is planned for a future release. The current implementation provides basic action frequency analysis.

```python
from farm.analysis.social_behavior.analyze import analyze_interaction_sequences

def find_behavioral_patterns(simulation_db: str):
    """Discover common action sequences."""
    
    # Extract action sequences for each agent
    sequences = get_agent_action_sequences(simulation_db)
    
    # Use sequence mining to find patterns
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Convert sequences to text format
    sequence_texts = [' '.join(seq) for seq in sequences]
    
    # Find frequent patterns
    vectorizer = CountVectorizer(ngram_range=(2, 5))
    X = vectorizer.fit_transform(sequence_texts)
    
    # Get most common patterns
    pattern_frequencies = X.sum(axis=0).A1
    patterns = vectorizer.get_feature_names_out()
    
    top_patterns = sorted(
        zip(patterns, pattern_frequencies),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    print("Most Common Behavioral Patterns:")
    for pattern, frequency in top_patterns:
        print(f"  {pattern}: {frequency:.0f} occurrences")
        
    return top_patterns

# Example output:
# Most Common Behavioral Patterns:
#   move gather move: 1523 occurrences
#   gather move gather: 1289 occurrences
#   move move attack: 876 occurrences
#   share move gather: 654 occurrences
```

#### Predictive Modeling

Predict agent actions and outcomes:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_action_predictor(simulation_db: str):
    """Train model to predict next action."""
    
    # Prepare training data
    states, actions = extract_state_action_pairs(simulation_db)
    
    # Features: current state
    # - Position
    # - Resources
    # - Health
    # - Nearby agents
    # - Nearby resources
    # - Recent action history
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        states, actions, test_size=0.2, random_state=42
    )
    
    # Train predictor
    predictor = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    predictor.fit(X_train, y_train)
    
    # Evaluate
    accuracy = predictor.score(X_test, y_test)
    print(f"Action Prediction Accuracy: {accuracy:.2%}")
    
    # Feature importance
    importances = predictor.feature_importances_
    feature_names = get_feature_names()
    
    print("\nMost Predictive Features:")
    for name, importance in sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"  {name}: {importance:.3f}")
    
    return predictor

# Use predictor
predictor = train_action_predictor("simulation.db")

# Predict future actions
current_state = get_agent_state(agent_id)
predicted_action = predictor.predict([current_state])[0]
action_probabilities = predictor.predict_proba([current_state])[0]

print(f"Predicted next action: {predicted_action}")
print("Action probabilities:")
for action, prob in zip(action_names, action_probabilities):
    print(f"  {action}: {prob:.2%}")
```

#### Survival Prediction

Predict which agents will survive:

```python
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

def train_survival_predictor(simulation_db: str):
    """Predict agent survival probability."""
    
    # Extract features at various time points
    df = load_agent_timeseries(simulation_db)
    
    # Features for survival prediction:
    features = [
        'initial_resources',
        'avg_resource_level',
        'resource_variance',
        'total_actions',
        'combat_participation',
        'sharing_frequency',
        'movement_distance',
        'agent_type',
        'starting_position_x',
        'starting_position_y'
    ]
    
    X = df[features]
    y = df['survived']  # Binary: survived to end or not
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import classification_report, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Survival Prediction Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
    
    return model

# Use for early prediction
survival_model = train_survival_predictor("simulation.db")

# Predict survival after just 100 steps
early_state = get_agent_state_at_step(agent_id, step=100)
survival_prob = survival_model.predict_proba([early_state])[0][1]

print(f"Agent {agent_id} survival probability: {survival_prob:.2%}")
```

---

### 4. Evolutionary Algorithms and Genetic Modeling

AgentFarm supports evolutionary approaches to agent development.

#### Genome System

Agents have genomes that can evolve:

```python
from farm.core.genome import Genome

# Extract genome from agent
genome = Genome.from_agent(agent)

# Genome contains:
# {
#     'action_set': [(action_name, weight), ...],
#     'module_states': {module_name: state_dict, ...},
#     'agent_type': 'SystemAgent',
#     'resource_level': 100,
#     'current_health': 85.5
# }

# Create new agent from genome
offspring = Genome.to_agent(
    genome=genome,
    agent_id="offspring_001",
    position=(50, 50),
    environment=environment,
    agent_factory=BaseAgent
)
```

#### Genetic Operators

Implement evolution through genetic operators:

```python
def mutate_genome(genome: dict, mutation_rate: float = 0.1) -> dict:
    """Apply random mutations to genome."""
    mutated = genome.copy()
    
    # Mutate action weights
    if random.random() < mutation_rate:
        action_idx = random.randint(0, len(mutated['action_set']) - 1)
        action_name, weight = mutated['action_set'][action_idx]
        # Randomly adjust weight
        new_weight = weight * random.uniform(0.8, 1.2)
        mutated['action_set'][action_idx] = (action_name, new_weight)
    
    # Mutate module parameters
    for module_name, state in mutated['module_states'].items():
        if random.random() < mutation_rate:
            # Mutate learning rate
            if 'learning_rate' in state:
                state['learning_rate'] *= random.uniform(0.9, 1.1)
    
    return mutated

def crossover_genomes(parent1: dict, parent2: dict) -> tuple[dict, dict]:
    """Create two offspring through crossover."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Single-point crossover for action weights
    crossover_point = len(parent1['action_set']) // 2
    
    child1['action_set'] = (
        parent1['action_set'][:crossover_point] +
        parent2['action_set'][crossover_point:]
    )
    
    child2['action_set'] = (
        parent2['action_set'][:crossover_point] +
        parent1['action_set'][crossover_point:]
    )
    
    return child1, child2
```

#### Evolutionary Simulation

Run simulations with evolutionary dynamics:

```python
def run_evolutionary_simulation(
    config: SimulationConfig,
    num_generations: int = 50,
    population_size: int = 100,
    selection_pressure: float = 0.3
):
    """Run simulation with genetic evolution."""
    
    # Initialize population
    environment = create_environment(config)
    population = create_initial_population(population_size, environment)
    
    for generation in range(num_generations):
        print(f"\n=== Generation {generation} ===")
        
        # Run simulation for this generation
        for step in range(config.max_steps):
            # Agents act
            for agent in population:
                if agent.alive:
                    action = agent.decide_action()
                    agent.execute_action(action)
        
        # Evaluate fitness
        fitness_scores = [
            evaluate_fitness(agent) for agent in population
        ]
        
        # Selection: keep top performers
        sorted_agents = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        num_survivors = int(population_size * selection_pressure)
        survivors = [agent for agent, _ in sorted_agents[:num_survivors]]
        
        print(f"Generation {generation} fitness:")
        print(f"  Best: {fitness_scores[0]:.2f}")
        print(f"  Average: {np.mean(fitness_scores):.2f}")
        print(f"  Worst: {fitness_scores[-1]:.2f}")
        
        # Create next generation
        new_population = survivors.copy()
        
        while len(new_population) < population_size:
            # Select parents (tournament selection)
            parent1 = tournament_select(survivors, fitness_scores)
            parent2 = tournament_select(survivors, fitness_scores)
            
            # Get genomes
            genome1 = Genome.from_agent(parent1)
            genome2 = Genome.from_agent(parent2)
            
            # Crossover
            child1_genome, child2_genome = crossover_genomes(genome1, genome2)
            
            # Mutation
            child1_genome = mutate_genome(child1_genome)
            child2_genome = mutate_genome(child2_genome)
            
            # Create offspring
            child1 = Genome.to_agent(
                child1_genome,
                f"gen{generation}_child{len(new_population)}",
                random_position(),
                environment,
                BaseAgent
            )
            child2 = Genome.to_agent(
                child2_genome,
                f"gen{generation}_child{len(new_population)+1}",
                random_position(),
                environment,
                BaseAgent
            )
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Reset environment for next generation
        environment = create_environment(config)
    
    # Return best agent from final generation
    best_agent = max(population, key=evaluate_fitness)
    return best_agent, population

def evaluate_fitness(agent):
    """Calculate agent fitness."""
    fitness = 0
    
    # Survival bonus
    if agent.alive:
        fitness += 100
    
    # Resource accumulation
    fitness += agent.resource_level
    
    # Longevity
    fitness += agent.age * 2
    
    # Reproduction success
    fitness += len(agent.offspring) * 50
    
    return fitness
```

#### Genome Embeddings

Use neural networks to analyze genome evolution:

```python
from farm.genome_embeddings.encoder import GenomeEncoder
from farm.genome_embeddings.training import train_genome_encoder
import torch

# Create genome encoder
encoder = GenomeEncoder(embedding_dim=32)

# Train on genome data
train_genome_encoder(
    encoder,
    genome_dataset,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# Use encoder to analyze genomes
agent_genomes = [Genome.from_agent(agent) for agent in population]
genome_vectors = []

for genome in agent_genomes:
    # Convert genome to features
    generation = genome['generation']
    parent_hash = hash_genome(genome['parent'])
    trait_hash = hash_traits(genome)
    
    # Encode
    embedding = encoder(
        torch.tensor([generation]),
        torch.tensor([parent_hash]),
        torch.tensor([trait_hash])
    )
    genome_vectors.append(embedding)

# Cluster genomes by similarity
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.3, min_samples=5)
genome_clusters = clustering.fit_predict(
    torch.stack(genome_vectors).detach().numpy()
)

print(f"Found {len(set(genome_clusters))} distinct genome lineages")
```

---

## Advanced ML Techniques

### Transfer Learning

Transfer knowledge between simulations:

```python
# Train agent in simple environment
simple_config = SimulationConfig(width=50, height=50, max_steps=1000)
simple_env = create_environment(simple_config)
agent = train_agent(simple_env, episodes=500)

# Save learned knowledge
agent.move_module.save_model("move_policy.pth")
agent.gather_module.save_model("gather_policy.pth")

# Transfer to complex environment
complex_config = SimulationConfig(width=200, height=200, max_steps=5000)
complex_env = create_environment(complex_config)
transfer_agent = create_agent(complex_env)

# Load pre-trained modules
transfer_agent.move_module.load_model("move_policy.pth")
transfer_agent.gather_module.load_model("gather_policy.pth")

# Fine-tune in new environment
fine_tune_agent(transfer_agent, complex_env, episodes=100)

# Benefits:
# - Faster learning in complex environments
# - Better initial performance
# - Reduced training time
```

### Multi-Agent Reinforcement Learning (MARL)

Coordinate multiple agents:

```python
class CoordinatedLearningAgent(BaseAgent):
    """Agent that learns with awareness of other agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Shared experience pool
        self.shared_memory = kwargs.get('shared_memory')
        
        # Communication network
        self.message_encoder = nn.Linear(state_dim, 16)
        self.message_decoder = nn.Linear(16, state_dim)
        
    def share_experience(self, state, action, reward, next_state):
        """Share experience with nearby agents."""
        if self.shared_memory:
            experience = (state, action, reward, next_state, self.agent_id)
            self.shared_memory.add(experience)
            
    def learn_from_others(self):
        """Learn from other agents' experiences."""
        if self.shared_memory and len(self.shared_memory) > 32:
            # Sample from shared memory
            batch = self.shared_memory.sample(32)
            
            # Train on others' experiences
            for state, action, reward, next_state, source_id in batch:
                if source_id != self.agent_id:
                    # Weight by similarity to source agent
                    weight = self.calculate_similarity(source_id)
                    self.update_with_weight(
                        state, action, reward, next_state, weight
                    )
```

### Imitation Learning

Learn from expert demonstrations:

```python
from sklearn.neural_network import MLPClassifier

def train_from_expert(expert_agent, novice_agent, num_demonstrations=1000):
    """Train novice agent to imitate expert."""
    
    # Collect expert demonstrations
    states = []
    actions = []
    
    for _ in range(num_demonstrations):
        state = environment.get_random_state()
        action = expert_agent.decide_action_from_state(state)
        
        states.append(state)
        actions.append(action)
    
    # Train imitation model
    imitation_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200
    )
    
    imitation_model.fit(states, actions)
    
    # Use imitation model to initialize novice
    novice_agent.policy_network.load_state_dict(
        convert_sklearn_to_torch(imitation_model)
    )
    
    # Fine-tune with RL
    train_agent_with_rl(novice_agent, num_episodes=100)
```

---

## Performance Optimization

### GPU Acceleration

Enable GPU training for faster learning:

```python
from farm.core.device_utils import create_device_from_config
import torch

# Configure device
config = SimulationConfig(
    device_preference="cuda",  # Use GPU if available
    device_fallback=True,      # Fall back to CPU if needed
    device_memory_fraction=0.8 # Use 80% of GPU memory
)

# Create device
device = create_device_from_config(config)

# Move models to device
agent.move_module.q_network.to(device)
agent.move_module.target_network.to(device)

# Training automatically uses GPU
# Speed improvements:
# - 5-10x faster for large networks
# - Essential for deep networks
# - Enables larger batch sizes
```

### Batch Training

Train multiple agents simultaneously:

```python
def batch_train_agents(agents: List[BaseAgent], batch_size: int = 32):
    """Train multiple agents in parallel."""
    
    # Collect experiences from all agents
    all_experiences = []
    for agent in agents:
        if len(agent.memory) > 0:
            all_experiences.extend(list(agent.memory))
    
    # Sample mini-batch
    if len(all_experiences) >= batch_size:
        batch = random.sample(all_experiences, batch_size)
        
        # Batch prepare data
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        
        # Single forward/backward pass for all
        for agent in agents:
            agent.train_on_batch(states, actions, rewards, next_states)
```

---

## Example: Complete ML-Enhanced Simulation

```python
#!/usr/bin/env python3
"""
Complete example: ML-enhanced agent simulation with automated analysis.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.core.agent import BaseAgent
from farm.analysis.service import AnalysisService, AnalysisRequest
import torch

def main():
    # Configure simulation with ML
    config = SimulationConfig(
        width=100,
        height=100,
        system_agents=25,
        independent_agents=25,
        max_steps=2000,
        
        # ML parameters
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        
        # Curriculum learning
        curriculum_phases=[
            {'steps': 200, 'enabled_actions': ['move', 'gather']},
            {'steps': 500, 'enabled_actions': ['move', 'gather', 'share']},
            {'steps': -1, 'enabled_actions': ['move', 'gather', 'share', 'attack', 'reproduce']}
        ],
        
        # GPU acceleration
        device_preference="cuda",
        
        # Seed for reproducibility
        seed=42
    )
    
    print("=== ML-Enhanced Agent Simulation ===\n")
    
    # Run simulation
    print("Running simulation with RL agents...")
    results = run_simulation(config)
    
    print(f"\nSimulation complete!")
    print(f"  Simulation ID: {results['simulation_id']}")
    print(f"  Final step: {results['final_step']}")
    print(f"  Surviving agents: {results['surviving_agents']}")
    
    # Automated ML-based analysis
    print("\nRunning automated analysis...")
    
    analysis_service = AnalysisService(config_service)
    analysis_request = AnalysisRequest(
        module_name="comprehensive",
        experiment_path=Path(results['db_path']).parent,
        output_path=Path("ml_analysis_results"),
        group="all"
    )
    
    analysis_result = analysis_service.run(analysis_request)
    
    if analysis_result.success:
        print(f"Analysis complete! Results in: {analysis_result.output_path}")
        
        # ML-based insights
        print("\n=== Automated Insights ===")
        
        # Behavioral clustering
        print("\nDiscovered behavioral patterns:")
        clusters = cluster_agent_behaviors(results['db_path'])
        
        # Survival prediction
        print("\nSurvival factors:")
        survival_model = train_survival_predictor(results['db_path'])
        
        # Pattern recognition
        print("\nCommon action sequences:")
        patterns = find_behavioral_patterns(results['db_path'])
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Deep Q-Learning](deep_q_learning.md) - Detailed DQN documentation
- [Decision Module](core/decision/README.md) - Algorithm implementations
- [Analysis System](analysis/README.md) - ML-based analysis tools
- [Genome System](api_reference.md#genome) - Evolutionary algorithms

### Examples
- [RL Algorithm Usage](examples/rl_algorithm_usage.py)
- [ML Algorithm Usage](examples/ml_algorithm_usage.py)
- [Evolutionary Simulation](examples/evolutionary_simulation.py)

### Research Resources
- [Genome Embeddings](genome_embeddings/) - Neural genome analysis
- [Behavioral Clustering](research/behavioral_clustering.py)
- [Pattern Recognition](analysis/social_behavior/)

---

## Support

For ML-related questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Examples**: Check `examples/` and `research/` directories

---

**Ready to add intelligence to your agents?** Start with [Basic RL Training](#deep-q-learning-dqn) or explore our [Automated Analysis](#automated-data-analysis-and-insight-generation) tools!
