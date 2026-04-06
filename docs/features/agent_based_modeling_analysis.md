# Agent-Based Modeling & Analysis

![Project Status](https://img.shields.io/badge/feature-agent%20modeling-blue)

> **Documentation note:** Many narrative examples below use **illustrative pseudocode** (for example `BaseAgent`, `run_simulation_batch`, and `SimulationConfig(width=...)` flat constructors). The real API uses **nested** `SimulationConfig` fields (`environment`, `population`, `resources`, …), **`run_simulation(...)`** which returns an **`Environment`**, and **`farm.core.analysis.SimulationAnalyzer`** as implemented in `farm/core/analysis.py` (see [System Dynamics Analysis](#system-dynamics-analysis)). For sweeps and multi-run studies use **`farm.runners.experiment_runner.ExperimentRunner`**. See [Usage examples](../usage_examples.md).

## Table of Contents

1. [Overview](#overview)
   - [What is Agent-Based Modeling?](#what-is-agent-based-modeling)
2. [Core Capabilities](#core-capabilities)
   - [1. Run Complex Simulations with Interacting, Adaptive Agents](#1-run-complex-simulations-with-interacting-adaptive-agents)
     - [Agent Types](#agent-types)
     - [Agent Capabilities](#agent-capabilities)
     - [Creating Agents](#creating-agents)
     - [Agent Interactions](#agent-interactions)
     - [Adaptive Behaviors](#adaptive-behaviors)
   - [2. Study Emergent Behaviors and System Dynamics](#2-study-emergent-behaviors-and-system-dynamics)
     - [What are Emergent Behaviors?](#what-are-emergent-behaviors)
     - [System Dynamics Analysis](#system-dynamics-analysis)
     - [Example: Observing Emergence](#example-observing-emergence)
   - [3. Track Agent Interactions and Environmental Influences](#3-track-agent-interactions-and-environmental-influences)
     - [Agent Interaction Tracking](#agent-interaction-tracking)
     - [Environmental Influence Analysis](#environmental-influence-analysis)
     - [Interaction Networks](#interaction-networks)
   - [4. Analyze Trends and Patterns Over Time](#4-analyze-trends-and-patterns-over-time)
     - [Time Series Analysis](#time-series-analysis)
     - [Pattern Recognition](#pattern-recognition)
     - [Comparative Analysis](#comparative-analysis)
3. [Practical Examples](#practical-examples)
   - [Example 1: Basic agent-based simulation](#example-1-basic-agent-based-simulation)
   - [Examples 2–4 (replaced)](#examples-24-replaced)
   - [Example 2: Order vs. Chaos scenario](#example-2-order-vs-chaos-scenario)
4. [Advanced features](#advanced-features)
5. [Performance optimization](#performance-optimization)
   - [Efficient spatial queries](#efficient-spatial-queries)
   - [Multiple simulations](#multiple-simulations)
   - [Memory](#memory)
6. [Best Practices](#best-practices)
   - [1. Simulation Design](#1-simulation-design)
   - [2. Analysis Workflow](#2-analysis-workflow)
   - [3. Performance Considerations](#3-performance-considerations)
7. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
8. [Additional Resources](#additional-resources)
   - [Documentation](#documentation)
   - [Tutorials](#tutorials)
   - [Research Examples](#research-examples)
9. [Research Applications](#research-applications)
   - [Ecology & Biology](#ecology--biology)
   - [Social Sciences](#social-sciences)
   - [Computer Science](#computer-science)
   - [Economics](#economics)
10. [Contributing](#contributing)
11. [Citation](#citation)
12. [Support](#support)

---

## Overview

Agent-Based Modeling (ABM) is a core feature of AgentFarm that enables researchers and developers to explore complex systems through computational simulations. This powerful framework allows you to create, execute, and analyze simulations where autonomous agents interact with each other and their environment, producing emergent behaviors and revealing system dynamics that would be difficult to study otherwise.

### What is Agent-Based Modeling?

Agent-Based Modeling is a computational approach that simulates the actions and interactions of autonomous agents to understand the behavior of complex systems. Unlike traditional analytical approaches, ABM focuses on individual entities (agents) and their behaviors, allowing emergent patterns to arise naturally from agent interactions.

**Key Characteristics:**
- **Individual-Level Focus**: Model each agent's unique attributes, behaviors, and decision-making
- **Emergent Behavior**: System-level patterns emerge from local interactions
- **Heterogeneity**: Agents can have diverse properties and behaviors
- **Adaptive Learning**: Agents can learn and adapt based on experience
- **Environmental Context**: Agents interact with a shared environment and each other

---

## Core Capabilities

### 1. Run Complex Simulations with Interacting, Adaptive Agents

AgentFarm enables you to create sophisticated multi-agent simulations where agents autonomously interact, adapt, and evolve over time.

#### Agent Types

AgentFarm supports multiple agent types, each with distinct behaviors:

- **System Agents** (`agent_type="system"`): Cooperative agents that prioritize collective goals and resource sharing. They have high sharing weights and low attack weights, making them well-suited for studying cooperative emergence.
- **Independent Agents** (`agent_type="independent"`): Self-oriented agents focused on individual survival and resource acquisition. They gather resources aggressively but rarely share, making them useful for studying competitive dynamics.
- **Control Agents** (`agent_type="control"`): Baseline agents with balanced parameters for experimental comparison. They serve as a neutral reference point in mixed-type experiments.
- **Order Agents** (`agent_type="order"`): Structure-seeking agents that favor stability, predictable resource gathering, and cautious behavior. They maintain high resource reserves, share moderately with neighbors, and avoid combat. Use them to study the emergence of organized, rule-following societies.
- **Chaos Agents** (`agent_type="chaos"`): Disruption-oriented agents that act recklessly, attack frequently, and ignore cooperative norms. They keep minimal resource reserves and rarely share. Use them to study instability, adversarial dynamics, and the breakdown of cooperative strategies.
- **Custom Agents**: Define your own agent types with specialized behaviors

#### Agent Capabilities

Each agent in AgentFarm possesses a rich set of capabilities:

```
Agent Core Attributes
- Autonomous decision-making
- Spatial awareness and navigation
- Resource gathering and management
- Health and survival mechanics
- Combat and defense capabilities
- Communication with nearby agents  ← implemented via CommunicationComponent
- Learning through reinforcement
- Memory and experience tracking    ← see scope note below
```

#### Memory and Experience Tracking

**Current scope (implemented):**

| Component | Description |
|---|---|
| **RL experience replay** | `SimpleReplayBuffer` / `ExperienceReplayBuffer` in `farm/core/decision/algorithms/rl_base.py` store `(state, action, reward, next_state, done)` tuples used for training DQN-style policies. This is the primary form of "experience tracking" in production agents. |
| **Redis-backed episodic memory** (optional) | `farm/memory/redis_memory.py` and `farm/memory/base_memory.py` provide an optional, agent-scoped episodic store that records per-step states, actions, rewards, and perceptions in Redis. Supports timeline retrieval, spatial search, and metadata filtering. Requires a running Redis instance and is not enabled by default. See [Redis agent memory](../redis_agent_memory.md). |

There is **no** general-purpose long-term cognitive memory (episodic, semantic, or associative) active in the default simulation stack today. The `Memory and experience tracking` capability listed above refers exclusively to the two components described here.

**Planned / experimental:**

A hierarchical memory architecture (Short-Term Memory → Intermediate Memory → Long-Term Memory with progressive compression) is under active research in the `memory_agent` experiment. This would give agents biologically-inspired, multi-tier memory with importance-weighted retention and cross-tier experience replay. See the [Memory Agent experiment docs](../experiments/memory_agent/README.md) for the full design. A roadmap issue will be filed to track promotion of this experiment into the core framework.

#### Agent-to-Agent (A2A) Communication

**Status: implemented** — `farm/core/agent/components/communication.py`

AgentFarm implements an asynchronous **inbox/outbox message-passing** model inspired by the [FIPA ACL](http://www.fipa.org/repository/aclspecs.html) standard and classic publish/subscribe paradigms.

##### Design overview

| Concept | Implementation |
|---|---|
| **Asynchronous delivery** | Senders push to an outbox; recipients read from a bounded inbox. Queued sends survive across simulation steps until the agent runs **communicate** (which calls `flush_outbox`). The outbox is **not** cleared at step boundaries. |
| **Proximity-limited broadcast** | The `communicate` action delivers messages only to agents within `communication_range` (default 50 units). |
| **Typed messages** | Each `Message` carries a `MessageType` so recipients can filter on semantics. |
| **Bounded inbox** | `inbox_capacity` (default 20) prevents memory growth; oldest messages are dropped when full. |
| **Bounded outbox** | `outbox_capacity` (default 20); oldest queued outbound messages are dropped when the queue is full. |
| **Optional component** | Agents without `CommunicationComponent` simply cannot send or receive messages. |

##### Supported message types

| `MessageType` | Meaning |
|---|---|
| `RESOURCE_REQUEST` | Sender is asking nearby agents for resources. |
| `RESOURCE_OFFER` | Sender is proactively offering to share resources. |
| `THREAT_ALERT` | Sender is warning nearby agents of a danger (e.g. an attacker). |
| `INFO` | General-purpose informational exchange. |
| `CUSTOM` | Arbitrary user-defined payload. |

##### Usage example

```python
from farm.core.agent import CommunicationComponent, MessageType

# Access the component on any full-featured agent
comm = agent.get_component("communication")

# ── Sending ─────────────────────────────────────────────────────
# Queue a broadcast (all nearby agents within communication_range)
comm.send(MessageType.RESOURCE_OFFER, content={"amount": 10})

# Queue a direct message to a specific agent (unicast).
# Only the agent with agent_id="ally_003" will receive it,
# provided it is within communication_range and alive.
comm.send(MessageType.THREAT_ALERT,
          content={"attacker_id": "enemy_007", "position": (40, 60)},
          recipient_id="ally_003")

# Messages are delivered when the 'communicate' action executes (flush).
# You may call send() in an earlier step and still deliver on a later communicate,
# within outbox_capacity. Broadcast messages (recipient_id=None) go to every
# eligible neighbour; unicast messages (recipient_id set) are routed only to the
# named agent.

# ── Receiving ────────────────────────────────────────────────────
# Read all inbox messages
for msg in comm.get_messages():
    print(msg.sender_id, msg.message_type, msg.content)

# Filter by type
alerts = comm.get_messages(MessageType.THREAT_ALERT)
offers  = comm.get_messages(MessageType.RESOURCE_OFFER)

comm.clear_inbox()   # discard processed messages
```

##### Configuration (`CommunicationConfig`)

```python
from farm.core.agent.config import CommunicationConfig

cfg = CommunicationConfig(
    communication_range=80.0,   # wider broadcast radius
    inbox_capacity=50,           # larger inbox
    outbox_capacity=50,          # more queued outbound messages before oldest drop
    reward_per_message=0.02,     # higher reward for communicating
)
# Pass via AgentComponentConfig.communication
```

##### The `communicate` action

When an agent selects the **communicate** action it:

1. Queries the spatial index for agents within `communication_range`.
2. Composes an `INFO` broadcast carrying `resource_level`, `position`, `health`, and `agent_type`.
3. Flushes the outbox (including any messages queued earlier via `send()`), delivering each to eligible neighbours' inboxes. Broadcast payloads are shallow-copied per recipient so neighbours cannot mutate each other's copy.
4. Earns a small reward proportional to successful deliveries.

The action weight in the global registry is **0.1** (lower than move/gather so communication does not dominate agent policy by default).

##### A2A communication protocols — research context

The implementation covers the foundational layer.  Researchers wishing to build richer protocols may draw on these established paradigms:

| Paradigm | Key idea | Applicable to AgentFarm |
|---|---|---|
| **FIPA ACL** | Performative-based language: `inform`, `request`, `propose`, `accept`, `refuse` | Extend `MessageType` with FIPA performatives |
| **Publish/Subscribe** | Agents subscribe to "topics"; publishers push updates | Use a shared topic registry + `CommunicationComponent.receive` |
| **Blackboard** | Shared read/write workspace; agents post and read problem data | Implement a per-environment dict; agents read in `on_step_start` |
| **Contract Net Protocol (CNP)** | Manager broadcasts task; contractors bid; winner is selected | Multi-message dialogue using `RESOURCE_REQUEST` / `RESOURCE_OFFER` |
| **Stigmergy** | Indirect communication via environment signals (pheromones) | Write signals to the resource grid; agents detect via perception |

> **TODO / future work** — see the checklist below.

##### Known limitations and future work

- [ ] **Dialogue protocols** — multi-turn conversations (e.g. CNP negotiation) require a correlation ID and state machine per dialogue.
- [ ] **Performative semantics** — messages currently carry raw `content` dicts; adding FIPA-style performatives would allow generic middleware.
- [x] **Direct unicast routing** — unicast messages (with `recipient_id` set) are routed by the `communicate` action to the named agent only if it is within range and alive.
- [ ] **Message persistence / logging** — sent messages are not written to the simulation database; add a `CommunicationEvent` row to the data model.
- [ ] **Observation integration** — the perception system (channels) does not yet include inbox-message features; consider adding a `message_channel` to the observation pipeline.
- [ ] **Scalability** — for very large populations (>10 000 agents), per-message spatial queries may become expensive; consider a region-based message bus.

#### Creating Agents

Agents are built by **`AgentFactory`** as **`AgentCore`** instances with composed components and behaviors (see `farm/core/agent/`). You normally do **not** construct agents directly in user code; the simulation runner creates them from `SimulationConfig.population` and related settings. For extension points, see **`IAgentBehavior`**, **`AgentComponent`**, and **`AgentServices`** in the same package.

#### Agent Interactions

Agents can interact in multiple ways:

1. **Resource Sharing**: Agents can share resources with nearby agents
2. **Combat**: Agents can engage in offensive or defensive actions
3. **Communication**: Agents exchange typed messages via `CommunicationComponent` and the `communicate` action
4. **Cooperation**: System agents can coordinate for collective benefit
5. **Competition**: Independent agents compete for limited resources

**Example Interaction:**

```python
# During simulation step
for agent in environment.agents.values():
    if agent.alive:
        # Agent perceives its environment
        perception = agent.perceive()
        
        # Agent decides on action based on observation
        action = agent.decide_action()
        
        # Agent executes action (move, gather, share, attack, communicate, etc.)
        result = agent.execute_action(action)
        
        # Agent learns from experience
        agent.update_learning(result)
```

#### Adaptive Behaviors

Agents adapt through multiple mechanisms:

**Reinforcement Learning:**
```python
from farm.core.decision.config import DecisionConfig

# Configure learning parameters
decision_config = DecisionConfig(
    algorithm="dqn",           # Deep Q-Network
    learning_rate=0.001,
    discount_factor=0.99,
    exploration_rate=0.1
)

# Agent learns optimal policies through experience
```

**Evolutionary Adaptation:**
```python
# Agents reproduce when conditions are met
if agent.resource_level > reproduction_threshold:
    offspring = agent.reproduce()
    # Offspring inherits parent's genome with mutations
    offspring.genome = mutate(agent.genome)
```

---

### 2. Study Emergent Behaviors and System Dynamics

One of the most powerful aspects of agent-based modeling is observing emergent behaviors—patterns that arise from agent interactions without being explicitly programmed.

#### What are Emergent Behaviors?

Emergent behaviors are system-level patterns that result from local agent interactions. In AgentFarm simulations, you might observe:

- **Resource Clustering**: Agents naturally form groups around resource-rich areas
- **Cooperation Networks**: System agents develop sharing relationships
- **Territorial Behavior**: Agents establish and defend resource territories
- **Migration Patterns**: Population movements in response to resource depletion
- **Social Hierarchies**: Dominance structures emerge through repeated interactions
- **Adaptive Strategies**: Population-level strategy shifts in response to environmental pressures

#### System Dynamics Analysis

`SimulationAnalyzer` (`farm/core/analysis.py`) runs SQL-backed summaries on a simulation SQLite file. Pass the database path and, when the file stores multiple runs, the same `simulation_id` you used with `run_simulation`:

```python
from farm.core.analysis import SimulationAnalyzer

analyzer = SimulationAnalyzer(simulation_db_path, simulation_id=optional_simulation_id)
```

**Population and survival**

`calculate_survival_rates()` returns one row per step with living counts by coarse agent type: `step`, `system_alive`, `independent_alive`.

```python
survival = analyzer.calculate_survival_rates()
```

**Resource dynamics**

`analyze_resource_distribution()` returns per-step, per-`agent_type` aggregates (`avg_resources`, `min_resources`, `max_resources`, `agent_count`). `analyze_resource_efficiency()` returns per-step `efficiency` from `simulation_steps`.

```python
resource_dist = analyzer.analyze_resource_distribution()
efficiency = analyzer.analyze_resource_efficiency()
```

**Combat intensity**

`analyze_competitive_interactions()` counts `attack` actions per step (`step`, `competitive_interactions`). Finer-grained action histograms, sharing networks, and cooperation metrics are not methods on this class; load `agent_actions` with `SimulationDatabase` / pandas or use repositories under `farm.database.repositories`.

#### Example: Observing Emergence

```python
from farm.config import SimulationConfig
from farm.core.analysis import SimulationAnalyzer
from farm.core.simulation import run_simulation

config = SimulationConfig.from_centralized_config(environment="development")
# Configure nested fields, e.g. config.environment.width, config.population.system_agents, …

env = run_simulation(
    num_steps=config.max_steps,
    config=config,
    path="simulations",
    save_config=True,
)

analyzer = SimulationAnalyzer(env.db.db_path, simulation_id=env.simulation_id)
survival = analyzer.calculate_survival_rates()
resource_dist = analyzer.analyze_resource_distribution()
combat_by_step = analyzer.analyze_competitive_interactions()
efficiency = analyzer.analyze_resource_efficiency()
# Use pandas/plots to relate combat spikes to resource_dist / efficiency trends
```

---

### 3. Track Agent Interactions and Environmental Influences

Understanding how agents interact with each other and respond to environmental conditions is crucial for ABM research.

#### Agent Interaction Tracking

Actions and state are persisted to SQLite (`agent_actions`, `agent_states`, `resource_states`, `health_incidents`, …). `SimulationAnalyzer` exposes **attack counts per step**; detailed event lists and spatial encounter mining are done with SQL or application code.

**Combat (built-in summary):**
```python
from farm.core.analysis import SimulationAnalyzer

analyzer = SimulationAnalyzer(simulation_db_path, simulation_id=optional_simulation_id)
combat_by_step = analyzer.analyze_competitive_interactions()
# Columns: step, competitive_interactions
```

**Combat (row-level via SQL):**
```python
import pandas as pd
from farm.database.database import SimulationDatabase

db = SimulationDatabase(simulation_db_path, simulation_id=optional_simulation_id)
attack_rows = pd.read_sql(
    """
    SELECT step_number, agent_id, action_target_id, reward
    FROM agent_actions
    WHERE action_type = 'attack'
    ORDER BY step_number
    """,
    db.engine,
)
```

**Resource sharing, proximity, and environment–behavior correlations**

There are no `SimulationAnalyzer` helpers for sharing graphs, encounter history, or spatial heatmaps. Filter `agent_actions` (`action_type='share'`, etc.), join `agent_states` / `resource_states`, or build analysis on top of `SimulationDatabase` and pandas.

#### Interaction Networks

Visualize and analyze agent interaction networks:

> **Note**: The `InteractionNetworkAnalyzer` class is planned for a future release. Currently, network analysis can be performed using custom SQL queries or external network analysis libraries.

```python
# Use custom SQL queries for network analysis
from farm.database.database import SimulationDatabase

db = SimulationDatabase(simulation_db_path)

# Example: Query combat interactions (add AND simulation_id = :id when sharing one DB across runs)
combat_query = """
SELECT agent_id AS attacker_id, action_target_id AS defender_id, COUNT(*) AS interactions
FROM agent_actions
WHERE action_type = 'attack'
GROUP BY agent_id, action_target_id
"""

# Custom network analysis implementation using pandas/networkx
import pandas as pd
combat_df = pd.read_sql(combat_query, db.engine)
# Build network graph from combat_df...

# Use networkx for analysis
import networkx as nx
G = nx.from_pandas_edgelist(combat_df, 'attacker_id', 'defender_id', 'interactions')
network_metrics = {
    'degree_distribution': dict(G.degree()),
    'clustering_coefficient': nx.average_clustering(G),
    'betweenness_centrality': nx.betweenness_centrality(G)
}
```

---

### 4. Analyze Trends and Patterns Over Time

Temporal analysis reveals how simulations evolve and helps identify long-term trends and cyclical patterns.

#### Time Series Analysis

AgentFarm provides comprehensive time series analysis tools:

> **Note**: The `TimeSeriesAnalyzer` class is planned for a future release. Currently, time series analysis can be performed using custom SQL queries and external libraries like pandas or statsmodels.

**Population Trends:**
```python
# Use custom time series analysis
from farm.database.database import SimulationDatabase
import pandas as pd

db = SimulationDatabase(simulation_db_path)

# Population over time from per-step state (add AND simulation_id = :id for multi-run DBs)
population_query = """
SELECT step_number AS step, COUNT(*) AS population
FROM agent_states
WHERE current_health > 0
GROUP BY step_number
ORDER BY step_number
"""

population_df = pd.read_sql(population_query, db.engine)
# Analyze trends with pandas/statsmodels...
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform trend analysis
result = seasonal_decompose(population_df.set_index('step')['population'], model='additive', period=100)
population_trends = {
    'trend': result.trend,
    'seasonal': result.seasonal,
    'residual': result.resid
}
```

**Resource Dynamics:**
```python
# Total resource amount over time (add AND simulation_id = :id for multi-run DBs)
resource_query = """
SELECT step_number AS step, SUM(amount) AS total_resources
FROM resource_states
GROUP BY step_number
ORDER BY step_number
"""

resource_df = pd.read_sql(resource_query, db.engine)
# Analyze resource trends with pandas
```

**Behavioral Evolution:**
```python
# Study how agent behaviors change over time
behavior_query = """
SELECT step, action_type, COUNT(*) as frequency
FROM agent_actions
GROUP BY step, action_type
ORDER BY step, action_type
"""

behavior_df = pd.read_sql(behavior_query, db.engine)
# Analyze behavioral evolution with pandas
```

#### Pattern Recognition

There is no `TimeSeriesAnalyzer` in the package. After loading series into pandas (see above), use pandas rolling windows, `statsmodels`, or your own thresholds (for example flag steps where a metric exceeds three standard deviations from its mean) for cycles, regime shifts, and anomalies.

#### Comparative Analysis

Compare two recorded simulations that live in the **same** experiment database using SQLAlchemy session helpers:

```python
from farm.database.simulation_comparison import compare_simulations, summarize_comparison

# session: sqlalchemy.orm.Session bound to your DB
# sim1_id / sim2_id: string simulation_id values (primary key of simulations table)
diff = compare_simulations(session, sim1_id="sim_001", sim2_id="sim_002")
summary = summarize_comparison(session, sim1_id="sim_001", sim2_id="sim_002")
```

For directory-style outputs and higher-level workflows, see **`farm.analysis.comparative_analysis.compare_simulations`** (writes a placeholder summary today) and the pointers in [Practical Examples](#practical-examples).

---

## Practical Examples

### Example 1: Basic agent-based simulation

```python
from farm.config import SimulationConfig
from farm.core.analysis import SimulationAnalyzer
from farm.core.simulation import run_simulation


def run_basic_abm():
    config = SimulationConfig.from_centralized_config(environment="development")
    config.environment.width = 50
    config.environment.height = 50
    config.population.system_agents = 10
    config.population.independent_agents = 10
    config.resources.initial_resources = 200
    config.resources.resource_regen_rate = 0.02
    config.max_steps = 500
    config.seed = 42

    env = run_simulation(
        num_steps=config.max_steps,
        config=config,
        path="simulations",
        save_config=True,
    )

    analyzer = SimulationAnalyzer(env.db.db_path, simulation_id=env.simulation_id)
    survival = analyzer.calculate_survival_rates()
    resources = analyzer.analyze_resource_distribution()
    print(survival.head())
    print(resources.head())


if __name__ == "__main__":
    run_basic_abm()
```

### Examples 2–4 (replaced)

Longer tutorials for cooperation studies, multi-run comparisons, and sweeps belong in **[Usage examples](../usage_examples.md)** and the **`tests/`** suite. For multiple runs with a single driver, use **`ExperimentRunner`** (`farm.runners.experiment_runner`) and read `_create_iteration_config` for how variation dicts map to `SimulationConfig`. For comparing SQLite outputs, use **`farm.database.simulation_comparison`** (session-based helpers) or **`farm.analysis.comparative_analysis.compare_simulations`**, depending on your workflow.

### Example 2: Order vs. Chaos scenario

```python
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.core.analysis import SimulationAnalyzer


def run_order_vs_chaos():
    config = SimulationConfig.from_centralized_config(environment="development")
    config.environment.width = 50
    config.environment.height = 50

    # Mix of all agent types
    config.population.system_agents = 5
    config.population.independent_agents = 5
    config.population.control_agents = 5
    config.population.order_agents = 10   # Structure-seeking, cooperative
    config.population.chaos_agents = 10   # Disruptive, aggressive

    config.resources.initial_resources = 300
    config.resources.resource_regen_rate = 0.03
    config.max_steps = 500
    config.seed = 42

    env = run_simulation(
        num_steps=config.max_steps,
        config=config,
        path="simulations",
        save_config=True,
    )

    analyzer = SimulationAnalyzer(env.db.db_path, simulation_id=env.simulation_id)
    survival = analyzer.calculate_survival_rates()
    print(survival.head())


if __name__ == "__main__":
    run_order_vs_chaos()
```

**Order Agent characteristics** (`agent_type="order"`):
- High minimum resource threshold (0.3) — maintains stable reserves
- Moderate sharing weight (0.25) — cooperative with neighbors
- Very low attack weight (0.02) — avoids conflict
- Moderate gather efficiency (0.6) — consistent and reliable

**Chaos Agent characteristics** (`agent_type="chaos"`):
- Very low minimum resource threshold (0.03) — reckless resource management
- Very low sharing weight (0.02) — non-cooperative
- Very high attack weight (0.45) — aggressive and disruptive
- Moderate gather efficiency (0.5) — unpredictable gathering

---

## Advanced features

- **Multi-run experiments:** `ExperimentRunner` + optional `chart_analyzer`; nested config changes are easiest with an explicit Python loop over `run_simulation` that clones/tweaks `SimulationConfig`.
- **Custom pipelines:** compose `SimulationDatabase`, repositories under `farm.database.repositories`, `farm.core.analysis.SimulationAnalyzer`, and `farm.analysis.service.AnalysisService` / domain modules (see [analysis modules](../analysis/modules/README.md)).
- **ML-heavy workflows:** export tensors or DB features to your own training code; decision/RL configuration lives under `farm.core.decision` and learning-related config on `SimulationConfig.learning`.

---

## Performance Optimization

### Efficient spatial queries

Use **`Environment.spatial_index`** and **`Environment.spatial_service`** after construction; enable extra indices with **`enable_quadtree_indices`** / **`enable_spatial_hash_indices`**. See [Spatial indexing](../spatial/spatial_indexing.md).

### Multiple simulations

Call **`run_simulation`** in a loop or use **`ExperimentRunner`**. There is **no** `run_simulation_batch` helper in `farm.core.simulation`.

### Memory

Tune **`SimulationConfig.database`** (in-memory SQLite, persistence flags) and optional **Redis** settings on **`SimulationConfig.redis`** / `farm.memory.redis_memory` per [Redis agent memory](../redis_agent_memory.md). Resource memmap options live on **`ResourceConfig`** (`memmap_delete_on_close`, etc.), not a top-level `use_memmap` flag on `SimulationConfig`.

---

## Best Practices

### 1. Simulation Design

**Define Clear Objectives:**
- What research questions are you addressing?
- What outcomes are you measuring?
- What constitutes success or failure?

**Start Simple:**
- Begin with minimal agent types
- Use small populations for initial testing
- Add complexity incrementally

**Control Randomness:**
- Always use seeds for reproducibility
- Document random parameters
- Run multiple replications

### 2. Analysis Workflow

**Systematic data collection:** persist `SimulationConfig.to_dict()`, seeds, git revision, and paths to SQLite outputs; use `SimulationAnalyzer` / pandas for summaries rather than assuming a `get_summary_statistics()` helper.

**Validation:** after `env = run_simulation(...)`, assert on `env.time`, `len(env.agents)`, or query the DB — not on a fictional `results` dict.

**Reproducibility:** set `PYTHONHASHSEED=0` where required (see `run_simulation.py`), fix `config.seed`, and log environment metadata yourself.

### 3. Performance Considerations

**Profile before optimizing:**
```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()

run_simulation(num_steps=config.max_steps, config=config, path="simulations")

profiler.disable()
profiler.print_stats(sort="cumtime")
```

**Monitor Resource Usage:**
> **Note**: The `SimulationMonitor` class is planned for a future release. Currently, resource monitoring can be implemented using external profiling libraries like `memory_profiler` or `psutil`.

```python
# Use external monitoring libraries
import psutil
import os

def monitor_simulation():
    process = psutil.Process(os.getpid())

    # Monitor memory usage
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

    # Monitor CPU usage
    cpu_percent = process.cpu_percent(interval=1.0)
    print(f"CPU usage: {cpu_percent:.1f}%")

# Call during simulation
monitor_simulation()
```

---

## Troubleshooting

### Common Issues

**Issue: simulation runs too slowly** — shrink `ObservationConfig.R`, use `storage_mode`/sparse options, tune `SpatialIndexConfig` batch settings, or reduce `max_steps` while profiling.

**Issue: memory usage too high** — prefer `SimulationConfig.database` in-memory + persist flags, observation HYBRID/sparse settings, and resource memmap options on `ResourceConfig`.

**Issue: agents die too quickly** — raise `config.resources.initial_resources`, `config.resources.resource_regen_rate`, or adjust `config.agent_parameters` / combat and behavior config blocks.

---

## Additional Resources

### Documentation
- [Core Architecture](../core_architecture.md) - System design and components
- [Agents](../agents.md) - Detailed agent system documentation
- [Experiments](../experiments.md) - Running experiments guide
- [Usage Examples](../usage_examples.md) - More practical examples
- [API Reference](../api_reference.md) - Complete API documentation

### Tutorials
- [Basic Simulation Setup](../usage_examples.md#tutorial-1-basic-simulation-setup)
- [Custom Agent Implementation](../usage_examples.md#tutorial-2-custom-agent-behaviors)
- [Experiment Design](../ExperimentQuickStart.md)

### Research examples
- [Experiment case studies](../experiments/)
- [Analysis modules](../analysis/modules/README.md)
- [Benchmark reports](../../benchmarks/reports/)

---

## Research Applications

Agent-Based Modeling in AgentFarm is suitable for various research domains:

### Ecology & Biology
- Population dynamics
- Predator-prey relationships
- Resource competition
- Evolutionary adaptation
- Disease spread models

### Social Sciences
- Cooperation emergence
- Social network formation
- Collective decision-making
- Cultural evolution
- Market dynamics

### Computer Science
- Multi-agent systems
- Reinforcement learning
- Swarm intelligence
- Distributed systems
- Emergent computation

### Economics
- Market simulations
- Resource allocation
- Agent-based macroeconomics
- Trading strategies
- Network effects

---

## Contributing

We welcome contributions to improve AgentFarm's agent-based modeling capabilities:

- **Bug Reports**: Report issues with simulations or analysis
- **Feature Requests**: Suggest new agent types or analysis methods
- **Documentation**: Improve examples and tutorials
- **Research**: Share your research findings and use cases

See [Contributing Guidelines](../../CONTRIBUTING.md) for more information.

---

## Citation

If you use AgentFarm for your research, please cite:

```bibtex
@software{agentfarm2024,
  title={AgentFarm: A Platform for Agent-Based Modeling and Analysis},
  author={Dooders Research Team},
  year={2024},
  url={https://github.com/Dooders/AgentFarm}
}
```

---

## Support

For questions and support:
- **GitHub Issues**: [https://github.com/Dooders/AgentFarm/issues](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [https://github.com/Dooders/AgentFarm/docs](https://github.com/Dooders/AgentFarm/docs)
- **Discussions**: Use GitHub Discussions for questions and community interaction

---

**Ready to explore complex systems?** Start with the [Basic Simulation Example](#example-1-basic-agent-based-simulation) or check out our [Quick Start Guide](../README.md#quick-start) to begin your agent-based modeling journey!
