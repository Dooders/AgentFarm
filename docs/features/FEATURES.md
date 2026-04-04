# AgentFarm Features

**A Comprehensive Platform for Agent-Based Modeling & Simulation**

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

---

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Feature Comparison Matrix](#feature-comparison-matrix)
4. [Getting Started by Use Case](#getting-started-by-use-case)
5. [Feature Roadmap](#feature-roadmap)
6. [Additional Resources](#additional-resources)

---

## Overview

AgentFarm is a comprehensive platform for agent-based modeling and simulation, providing researchers and developers with powerful tools for creating, running, and analyzing complex multi-agent systems. From basic simulations to advanced AI-driven models, AgentFarm offers the flexibility and performance needed for cutting-edge research.

### Why AgentFarm?

- **🎯 Research-Focused**: Built specifically for scientific investigation and experimentation
- **🚀 High Performance**: Optimized spatial indexing and batch processing for large-scale simulations
- **🔬 Comprehensive Analysis**: Deep analytical tools from basic statistics to ML-powered insights
- **🎨 Highly Customizable**: Flexible architecture supporting custom agents, environments, and behaviors
- **📊 Data-Driven**: Complete data pipeline from collection to visualization and reporting
- **🔧 Developer-Friendly**: Clean APIs, extensive documentation, and practical examples

---

## Core Features

### 1. Agent-Based Modeling & Analysis

**Run sophisticated multi-agent simulations with emergent behaviors**

Build complex simulations where autonomous agents interact, adapt, and evolve, producing emergent system-level behaviors that would be impossible to predict from individual rules alone.

**Key Capabilities:**
- 🤖 **Interacting Agents**: Autonomous entities with perception, decision-making, and action
- 🌊 **Emergent Behaviors**: Observe complex patterns arising from simple local interactions
- 🔗 **Interaction Tracking**: Monitor all agent-agent and agent-environment interactions
- 📈 **Trend Analysis**: Identify patterns and dynamics over time

**Quick example:**
```python
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation

config = SimulationConfig.from_centralized_config(environment="development")
config.environment.width = 100
config.environment.height = 100
config.population.system_agents = 25
config.population.independent_agents = 25
config.max_steps = 1000

env = run_simulation(num_steps=config.max_steps, config=config, path="simulations")
# Inspect env.agents, env.db, and logs for emergent dynamics
```

**Use Cases:**
- Social dynamics and cooperation studies
- Resource competition and allocation
- Evolutionary dynamics and adaptation
- Collective decision-making
- Ecosystem modeling

📖 **[Full Documentation →](agent_based_modeling_analysis.md)**

---

### 2. Customization & Flexibility

**Tailor every aspect of your simulations to your research needs**

From simple parameter adjustments to complete behavioral overhauls, customize agents, environments, rules, and scenarios without touching core code.

**Key Capabilities:**
- ⚙️ **Custom Parameters**: Configure via YAML files with hierarchical overrides
- 🎭 **Custom Agents**: Create specialized agent types with unique behaviors
- 🌍 **Custom Environments**: Define rules, zones, and environmental dynamics
- 🧪 **Custom Experiments**: Design complete experimental protocols

**Quick example (directional):** extend **`IAgentBehavior`** / **`AgentComponent`** under `farm/core/agent/`, or subclass **`Environment`** in `farm/core/environment.py` if you need different world dynamics. See [Usage examples](../usage_examples.md) and the agent package README patterns before copying large snippets.

**Use Cases:**
- Domain-specific simulations (ecology, economics, social science)
- Testing theoretical models
- Exploring alternative rule sets
- Rapid prototyping of agent behaviors

📖 **[Full Documentation →](customization_flexibility.md)**

---

### 3. AI & Machine Learning

**Intelligent agents that learn and adapt through experience**

Integrate state-of-the-art reinforcement learning, evolutionary algorithms, and machine learning analytics to create truly adaptive agents and extract deep insights from simulation data.

**Key Capabilities:**
- 🧠 **Reinforcement Learning**: DQN, PPO, SAC, A2C, TD3 algorithms
- 📊 **Automated Analysis**: ML-powered pattern recognition and clustering
- 🔮 **Behavior Prediction**: Forecast agent actions and outcomes
- 🧬 **Evolutionary Algorithms**: Genetic evolution and natural selection

**Quick example:**
```python
from farm.core.decision.config import DecisionConfig

# DQN / action-selection parameters (see farm.core.decision.config for fields)
decision_cfg = DecisionConfig(
    algorithm_type="dqn",
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
)

# Wire decision_cfg through learning / agent configuration per your scenario YAML or SimulationConfig.
# Post-run charts and tables: farm.charts.chart_analyzer.ChartAnalyzer + farm.core.analysis.SimulationAnalyzer
```

**Use Cases:**
- Adaptive agent strategies
- Multi-agent reinforcement learning
- Evolutionary simulations
- Behavioral pattern discovery
- Predictive modeling

📖 **[Full Documentation →](ai_machine_learning.md)**

---

### 4. Data & Visualization

**Comprehensive data collection with powerful visualization tools**

Capture every detail of your simulations with automatic data logging, then visualize and analyze with professional-grade tools.

**Key Capabilities:**
- 💾 **Comprehensive Collection**: Every action, state change, and event logged
- 📊 **Real-Time Visualization**: Watch simulations unfold with interactive viewers
- 📈 **Rich Charting**: 20+ chart types for population, resources, actions, and more
- 📝 **Automated Reports**: AI-powered insights and analysis generation

**Quick Example:**
```python
from farm.charts.chart_analyzer import ChartAnalyzer
from farm.database.database import SimulationDatabase

# Automatic data collection (no setup needed)
results = run_simulation(config)
db_path = results['db_path']

# Generate comprehensive analysis
db = SimulationDatabase(db_path)
analyzer = ChartAnalyzer(database=db, save_charts=True)
analyses = analyzer.analyze_all_charts()

# Generates:
# - Population dynamics charts
# - Resource utilization graphs  
# - Action distribution plots
# - Behavioral heatmaps
# - AI-powered insights for each chart
```

**Use Cases:**
- Result visualization and presentation
- Real-time simulation monitoring
- Publication-quality figures
- Exploratory data analysis
- Automated reporting

📖 **[Full Documentation →](data_visualization.md)**

---

### 5. Research Tools

**Rigorous scientific investigation through systematic experimentation**

Run parameter sweeps, comparative studies, and replicate experiments with tools designed for reproducible research.

**Key Capabilities:**
- 🔬 **Parameter Sweeps**: Grid search, random search, Bayesian optimization
- ⚖️ **Comparative Analysis**: Statistical tests and effect size analysis
- 🔁 **Replication Tools**: Ensure reproducibility with validation and verification
- 📋 **Structured Logging**: Professional-grade logs with rich context

**Quick example:**
```python
from pathlib import Path

from farm.config import SimulationConfig
from farm.runners.experiment_runner import ExperimentRunner

base_config = SimulationConfig.from_centralized_config(environment="development")

# `_create_iteration_config` applies dict keys with setattr() on SimulationConfig — use **top-level**
# fields only (e.g. max_steps, seed). For nested changes (population/resources), clone the config
# per iteration in your own loop or extend ExperimentRunner.
variations = [
    {"max_steps": 400, "seed": 1},
    {"max_steps": 400, "seed": 2},
    {"max_steps": 400, "seed": 3},
]

runner = ExperimentRunner(base_config, "repeatability_demo")
runner.run_iterations(
    num_iterations=len(variations),
    config_variations=variations,
    path=Path("experiments_out"),
)
# Inspect per-iteration folders under experiments/repeatability_demo/ and experiments_out/
```

**Use Cases:**
- Parameter space exploration
- Hypothesis testing
- A/B testing
- Sensitivity analysis
- Reproducible research

📖 **[Full Documentation →](research_tools.md)**

---

### 6. Data System

**Layered architecture for efficient data management and analysis**

A comprehensive 4-layer data system providing clean abstractions from raw storage to high-level analytics.

**Key Capabilities:**
- 🏗️ **Layered Architecture**: Database → Repositories → Analyzers → Services
- 🔍 **Advanced Analytics**: 10+ specialized analyzers for deep insights
- 🔌 **Flexible Access**: Multiple ways to query and retrieve data
- 🚀 **High-Level Services**: Coordinated analysis with error handling

**Quick example:**
```python
from farm.database.database import SimulationDatabase
from farm.database.repositories.agent_repository import AgentRepository

db = SimulationDatabase("simulations/simulation.db")
agent_repo = AgentRepository(db.session_manager)
row = agent_repo.get_agent_by_id("agent_001")

# Higher-level analysis: farm.core.analysis.SimulationAnalyzer, farm.analysis.*, farm.charts.chart_analyzer
```

**Use Cases:**
- Complex data queries
- Multi-faceted analysis
- Custom analytics
- Data pipeline development
- Cross-simulation studies

📖 **[Full Documentation →](data_system.md)**

---

### 7. Spatial Indexing & Performance

**State-of-the-art spatial data structures for maximum performance**

Multiple spatial indexing strategies optimized for different query patterns, with batch update processing for incremental improvements.

**Key Capabilities:**
- 🌳 **Three Index Types**: KD-tree, Quadtree, Spatial Hash Grid
- ⚡ **Batch Updates**: Incremental reduction in computational overhead
- 🎯 **Query Optimization**: Choose optimal index per query type
- 📊 **Performance Monitoring**: Real-time metrics and benchmarking

**Quick example:**
```python
from farm.config import SimulationConfig
from farm.config.config import SpatialIndexConfig
from farm.core.environment import Environment

spatial_config = SpatialIndexConfig(
    enable_batch_updates=True,
    region_size=50.0,
    max_batch_size=100,
)
sim_config = SimulationConfig.from_centralized_config(environment="development")
sim_config.spatial_index = spatial_config
sim_config.environment.width = 200
sim_config.environment.height = 200

env = Environment(
    width=sim_config.environment.width,
    height=sim_config.environment.height,
    resource_distribution="uniform",
    config=sim_config,
)
env.enable_quadtree_indices()
env.enable_spatial_hash_indices(15.0)

pos = (50.0, 50.0)
bounds = (40.0, 40.0, 20.0, 20.0)  # (x, y, width, height)
env.spatial_index.get_nearby(pos, 5.0, ["agents"])
env.spatial_index.get_nearby_range(bounds, ["agents_quadtree"])
env.spatial_index.get_nearby(pos, 3.0, ["agents_hash"])
```

**Performance:**
- **Build Time**: 1.26ms for 1,000 entities
- **Query Time**: 4.85μs average (beats Scikit-learn by 5x)
- **Memory**: <0.1MB per 1,000 entities
- **Scalability**: Handles hundreds of entities efficiently

📖 **[Full Documentation →](spatial_indexing_performance.md)**

---

## Feature Comparison Matrix

### At a Glance

| Feature | Complexity | Performance | Flexibility | Documentation |
|---------|-----------|-------------|-------------|---------------|
| **Agent-Based Modeling** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Customization** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **AI & ML** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Data & Visualization** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Research Tools** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Data System** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Spatial Indexing** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Feature Dependencies

```
Agent-Based Modeling (Core)
    ↓
    ├─→ Spatial Indexing (Required for proximity)
    ├─→ Data System (Required for persistence)
    └─→ Data & Visualization (Optional for analysis)

Customization (Standalone)
    ↓
    └─→ Works with all features

AI & ML (Advanced)
    ↓
    ├─→ Agent-Based Modeling (Builds upon)
    └─→ Data System (Uses for training data)

Research Tools (Orchestration)
    ↓
    └─→ Uses all features for experiments
```

---

## Getting Started by Use Case

### 🎓 Academic Research

**Goal**: Publish rigorous scientific studies

**Recommended Features:**
1. ✅ Agent-Based Modeling - Core simulations
2. ✅ Research Tools - Parameter sweeps and replication
3. ✅ Data & Visualization - Publication-quality figures
4. ✅ Data System - Comprehensive analysis

**Quick Start:**
```python
# 1. Run systematic parameter sweep
experiment = ExperimentRunner(base_config, "my_study")
experiment.run_iterations(num_iterations=100, config_variations=variations)

# 2. Analyze results
comparison = experiment.compare_results()

# 3. Generate publication figures
analyzer = ChartAnalyzer(database, save_charts=True)
analyses = analyzer.analyze_all_charts()

# 4. Create reproducibility report
create_reproducibility_report(analysis_params, results)
```

---

### 🏢 Industry Applications

**Goal**: Model real-world systems and optimize processes

**Recommended Features:**
1. ✅ Agent-Based Modeling - System simulation
2. ✅ Customization - Domain-specific rules
3. ✅ Spatial Indexing - Performance at scale
4. ✅ Data System - Analytics and reporting

**Quick start:** combine nested `SimulationConfig`, `run_simulation`, and `SpatialIndexConfig` (see section 7) for scale; add domain-specific **behaviors/components** under `farm/core/agent/` instead of legacy “subclass BaseAgent” patterns.

---

### 🤖 AI/ML Research

**Goal**: Develop and test learning algorithms

**Recommended Features:**
1. ✅ AI & ML - Learning algorithms
2. ✅ Agent-Based Modeling - Test environment
3. ✅ Research Tools - Systematic testing
4. ✅ Data System - Training data

**Quick start:** tune `DecisionConfig` / learning settings, run repeated `run_simulation` calls or `ExperimentRunner`, then analyze SQLite outputs with `SimulationAnalyzer`, `ChartAnalyzer`, or your own notebooks.

---

### 🔬 Exploratory Studies

**Goal**: Understand emergent phenomena

**Recommended Features:**
1. ✅ Agent-Based Modeling - Explore dynamics
2. ✅ Customization - Test hypotheses
3. ✅ Data & Visualization - Discover patterns
4. ✅ Data System - Deep analysis

**Quick start:** drive `run_simulation` + structured logging/metrics, then explore the resulting database with analysis modules under `farm/analysis/` and `farm/core/analysis.py`.

---

### 🎮 Interactive Simulations

**Goal**: Create responsive, user-interactive models

**Recommended Features:**
1. ✅ Agent-Based Modeling - Simulation core
2. ✅ Spatial Indexing - Real-time performance
3. ✅ Data & Visualization - Interactive UI
4. ✅ Customization - User controls

**Quick start:** tune `SpatialIndexConfig` batch settings, call `Environment.process_batch_spatial_updates()` between frames when you need fresh spatial queries, and pair with `farm.core.visualization` if you render frames.

---

## Feature Roadmap

### ✅ Completed (v0.1.0)

- [x] Core agent-based modeling framework
- [x] Three spatial indexing strategies
- [x] Batch update system with incremental improvements
- [x] Comprehensive data architecture
- [x] Multiple RL algorithms (DQN, PPO, SAC, A2C, TD3)
- [x] 20+ chart types and visualizations
- [x] Parameter sweep tools
- [x] Structured logging system
- [x] Multi-simulation support

### 🚧 In Development (v0.2.0)

- [ ] GPU-accelerated spatial queries [^gpu-note]
- [ ] Distributed simulation support [^distributed-note]
- [ ] Real-time collaborative features [^realtime-note]
- [ ] Advanced visualization (3D, WebGL) [^3d-viz-note]
- [ ] Interactive configuration UI [^config-ui-note]
- [ ] Cloud deployment tools [^cloud-note]
- [ ] Enhanced ML integration [^ml-integration-note]

### 🔮 Planned (v0.3.0+)

- [ ] Multi-agent communication protocols [^comm-protocols-note]
- [ ] Neural architecture search [^neural-arch-note]
- [ ] Federated learning support [^federated-note]
- [ ] Real-time analytics dashboard [^analytics-dashboard-note]
- [ ] Plugin system for extensions [^plugin-system-note]
- [ ] Mobile visualization app [^mobile-app-note]
- [ ] Integration with Unity/Unreal [^unity-unreal-note]

---

## Performance Highlights

### Simulation Scale

| Metric | Performance | Notes |
|--------|-------------|-------|
| **Max Agents** | 200+ | Tested with spatial indexing |
| **Steps per Second** | 2.5-8.6 | Depends on agent count (fewer = faster) |
| **Memory Usage** | ~200MB | 1,000 agents, full logging |
| **Spatial Query** | 4.85μs | Average KD-tree query |
| **Batch Updates** | 2-3% faster | vs. individual updates |

### Scalability

- **Linear Scaling**: Memory usage scales linearly with agent count
- **Sub-Linear Queries**: Spatial queries scale O(log n)
- **Efficient Updates**: Batch processing provides incremental overhead reduction
- **Multi-Core**: Parallel experiment execution supported

---

## Installation & Quick Start

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- pip package manager
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies and the package (required for `farm` imports)
pip install -r requirements.txt
pip install -e .

# Optional: Install Redis for enhanced memory
# Ubuntu/Debian: sudo apt-get install redis-server
# macOS: brew install redis
```

### Your first simulation

```python
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation

config = SimulationConfig.from_centralized_config(environment="development")
config.environment.width = 100
config.environment.height = 100
config.population.system_agents = 25
config.population.independent_agents = 25
config.max_steps = 1000
config.seed = 42

env = run_simulation(
    num_steps=config.max_steps,
    config=config,
    path="simulations",
    save_config=True,
)
print(f"Final agent count: {len(env.agents)}")
```

**Next steps:**
- [README](../../README.md) — simulation CLI, API server, benchmarks
- [User Guide](../user-guide.md) — Config Explorer UI (`farm/editor`)
- [Developer Guide](../developer-guide.md) — Config Explorer / frontend dev
- [Contributing](../../CONTRIBUTING.md) — Python package and tests
- [API Reference](../api_reference.md) — module reference
- [Usage examples](../usage_examples.md) — tutorials and snippets

---

## Additional Resources

### 📚 Documentation

**Feature Documentation:**
- [Agent-Based Modeling & Analysis](agent_based_modeling_analysis.md)
- [Customization & Flexibility](customization_flexibility.md)
- [AI & Machine Learning](ai_machine_learning.md)
- [Data & Visualization](data_visualization.md)
- [Research Tools](research_tools.md)
- [Data System](data_system.md)
- [Spatial Indexing & Performance](spatial_indexing_performance.md)

**Technical Guides:**
- [Core Architecture](../core_architecture.md)
- [Configuration Guide](../config/configuration_guide.md)
- [Database Schema](../data/database_schema.md)
- [Logging Guide](../logging_guide.md)
- [Benchmarks](../../benchmarks/README.md)

### Examples

Primary entry points for runnable patterns:

- [Usage examples](../usage_examples.md)
- `benchmarks/examples/` (repository root)
- `tests/` (repository root)

### 📊 Benchmarks

**Performance Reports:**
- [Spatial Benchmark Report](../../benchmarks/reports/0.1.0/spatial_benchmark_report.md)
- [Database Benchmark Report](../../benchmarks/reports/0.1.0/database_benchmark_report.md)
- [Benchmark & Profiling Summary](../../benchmarks/reports/0.1.0/benchmark_profiling_summary_report.md)

### 🔬 Research

**Use Cases & Studies:**
- [Experiment write-ups](../experiments/)
- [Analysis modules](../analysis/modules/README.md)
- Simulation comparison API: `farm/database/simulation_comparison.py`

---

## Support & Community

### Getting Help

- **📖 Documentation**: Start with the [User Guide](../user-guide.md)
- **💬 GitHub Discussions**: Ask questions and share ideas
- **🐛 GitHub Issues**: Report bugs and request features
- **📧 Email**: Contact the development team

### Contributing

We welcome contributions! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for:
- Code contribution guidelines
- Development setup
- Testing requirements
- Documentation standards

### Citation

If you use AgentFarm in your research, please cite:

```bibtex
@software{agentfarm2024,
  title={AgentFarm: A Platform for Agent-Based Modeling and Analysis},
  author={Dooders Research Team},
  year={2024},
  url={https://github.com/Dooders/AgentFarm}
}
```

---

## Feature Implementation Notes

[^gpu-note]: GPU-accelerated spatial queries are planned for v0.2.0 to enable large-scale simulations with thousands of agents.

[^distributed-note]: Distributed simulation support will allow simulations to run across multiple machines for massive agent populations.

[^realtime-note]: Real-time collaborative features will enable multiple researchers to interact with and modify simulations simultaneously.

[^3d-viz-note]: Advanced 3D and WebGL visualization will provide immersive simulation experiences and better spatial understanding.

[^config-ui-note]: Interactive configuration UI will simplify experiment setup and parameter exploration for non-programmers.

[^cloud-note]: Cloud deployment tools will enable running simulations on cloud infrastructure with automatic scaling.

[^ml-integration-note]: Enhanced ML integration will include advanced algorithms like evolutionary learning and federated learning approaches.

[^comm-protocols-note]: Multi-agent communication protocols will enable complex social interactions and language-based coordination.

[^neural-arch-note]: Neural architecture search will automatically optimize agent neural networks for specific tasks.

[^federated-note]: Federated learning support will allow training across distributed datasets while preserving privacy.

[^analytics-dashboard-note]: Real-time analytics dashboard will provide live monitoring and alerting for long-running simulations.

[^plugin-system-note]: Plugin system will enable third-party extensions and custom simulation components.

[^mobile-app-note]: Mobile visualization app will allow monitoring simulations on smartphones and tablets.

[^unity-unreal-note]: Unity/Unreal integration will enable using AgentFarm simulations in game engines for interactive experiences.

---

## License

AgentFarm is part of the [Dooders](https://github.com/Dooders) research initiative exploring complex adaptive systems through computational modeling.

---

**🚀 Ready to start?** Choose your path:
- 🎓 [Academic Research](#-academic-research)
- 🏢 [Industry Applications](#-industry-applications)
- 🤖 [AI/ML Research](#-aiml-research)
- 🔬 [Exploratory Studies](#-exploratory-studies)
- 🎮 [Interactive Simulations](#-interactive-simulations)

Or dive right in with the [Quick Start Guide](user-guide.md)!
