# AgentFarm

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

**AgentFarm** is an advanced simulation and computational modeling platform for exploring complex systems. Designed as a comprehensive workbench, it enables users to run, analyze, and compare simulations with ease.

This repository is being developed to support research in the [Dooders](https://github.com/Dooders) project, focusing on complex adaptive systems and agent-based modeling approaches.

> **Note**: This project is currently in active development. APIs and features may change between releases. See the [Contributing Guidelines](CONTRIBUTING.md) for information on getting involved.

## Key Features

### Agent-Based Modeling & Analysis
- Run complex simulations with interacting, adaptive agents
- Study emergent behaviors and system dynamics
- Track agent interactions and environmental influences
- Analyze trends and patterns over time

### Customization & Flexibility
- Define custom parameters, rules, and environments
- Create specialized agent behaviors and properties
- Configure simulation parameters and conditions
- Design custom experiments and scenarios

### AI & Machine Learning
- Reinforcement learning for agent adaptation
- Automated data analysis and insight generation
- Pattern recognition and behavior prediction
- Evolutionary algorithms and genetic modeling

### Data & Visualization
- Comprehensive data collection and metrics
- Interactive results dashboard
- Real-time visualization tools
- Automated report generation

### Research Tools
- Parameter sweep experiments
- Comparative analysis framework
- Experiment replication tools
- Detailed logging and tracking

### Data System
- **Comprehensive Data Architecture**: Layered system with database, repositories, analyzers, and services
- **Advanced Analytics**: Action statistics, behavioral clustering, causal analysis, and pattern recognition
- **Flexible Data Access**: Repository pattern for efficient data retrieval and querying
- **High-Level Services**: Coordinated analysis operations with built-in error handling
- **Multi-Simulation Support**: Experiment database for comparing multiple simulation runs

### Additional Tools
- **Interactive Notebooks**: Jupyter notebooks for data exploration and analysis
- **Web Dashboard**: Browser-based interface for monitoring and visualization
- **Benchmarking Suite**: Performance testing and optimization tools
- **Research Tools**: Advanced analysis modules for academic research
- **Genome Embeddings**: Machine learning tools for agent evolution analysis

## Quick Start

### Prerequisites
- Python 3.8 or higher (3.9+ recommended for best performance)
- pip (Python package installer)
- Git
- Redis (optional, for enhanced memory management)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Optional: Install Redis for enhanced memory management
# Note: Redis is used for agent memory storage and can improve performance
# On Ubuntu/Debian: sudo apt-get install redis-server
# On macOS: brew install redis
# On Windows: Download from https://redis.io/download
# Then start Redis: redis-server
```

### Running Your First Simulation

AgentFarm provides multiple ways to run simulations:

**Command Line (Simple)**
```bash
python run_simulation.py --config config.yaml --steps 1000
```

**Command Line Interface (Advanced)**
```bash
# Run simulation with various options
python farm/core/cli.py --mode simulate --config config.yaml --steps 1000

# Run experiments with parameter variations
python farm/core/cli.py --mode experiment --config config.yaml --experiment-name test --iterations 3

# Visualize existing simulation results
python farm/core/cli.py --mode visualize --db-path simulations/simulation.db

# Generate analysis reports
python farm/core/cli.py --mode analyze --db-path simulations/simulation.db
```

**GUI Interface**
```bash
python main.py
```

**Results**
All simulation results are saved in the `simulations` directory with database files, logs, and analysis reports.

## Documentation

For detailed documentation and advanced usage:
- [Simulation Guide](docs/SimulationQuickStart.md)
- [Experiment Guide](docs/ExperimentQuickStart.md)
- [Core Architecture](docs/core_architecture.md)
- [Decision Module](farm/core/decision/README.md)
- [Configuration Guide](docs/configuration_guide.md)
- [Data System Architecture](docs/data/data_api.md)
- [API Reference](docs/api_reference.md)
- [Module Overview](docs/module_overview.md)
- [Research Documentation](docs/research.md)
- [Full Documentation Index](docs/README.md)

## Contributing

Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is appreciated.

Please see [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## Design Principles

- **Single Responsibility Principle (SRP)**: A class or module should have only one reason to change, focusing on a single responsibility to reduce complexity.
- **Open-Closed Principle (OCP)**: Entities should be open for extension (e.g., via new subclasses) but closed for modification, allowing behavior addition without altering existing code.
- **Liskov Substitution Principle (LSP)**: Subclasses must be substitutable for their base classes without breaking program behavior, honoring the base class's contract.
- **Interface Segregation Principle (ISP)**: Clients should not depend on interfaces they don't use; prefer small, specific interfaces over large, general ones.
- **Dependency Inversion Principle (DIP)**: High-level modules should depend on abstractions (e.g., interfaces), not concrete implementations, to decouple components.
- **Don't Repeat Yourself (DRY)**: Avoid duplicating code or logic; centralize shared functionality to improve maintainability and reduce errors.
- **Keep It Simple, Stupid (KISS)**: Favor simple, straightforward solutions over complex ones to enhance readability and reduce bugs.
- **Composition Over Inheritance**: Prefer composing objects (e.g., via dependencies) to achieve behavior rather than relying on inheritance hierarchies, for greater flexibility.

## Support

If you encounter any issues, please check [issues page](https://github.com/Dooders/AgentFarm/issues) or open a new issue.