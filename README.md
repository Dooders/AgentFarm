# AgentFarm

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

**AgentFarm** is an advanced simulation and computational modeling platform for exploring complex systems. Designed as a comprehensive workbench, it enables users to run, analyze, and compare simulations with ease.

This repository is being developed to support research in the [Dooders](https://github.com/Dooders) project, focusing on complex adaptive systems and agent-based modeling approaches.

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

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

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
```

### Running Your First Simulation

1. **Copy the template configuration**
   ```bash
   cd config
   cp config_template.yaml my_simulation.yaml
   ```

2. **Run the simulation**
   ```bash
   cd ..
   python run_simulation.py --config config/my_simulation.yaml
   ```

Results will be saved in the `results` directory.

## Documentation

For detailed documentation and advanced usage:
- [Simulation Guide](docs/SimulationQuickStart.md)
- [Experiment Guide](docs/ExperimentQuickStart.md)
- [Data System Architecture](docs/data/data_api.md)
- [Interaction Edge Logging](docs/data/interaction_edges.md)
- [Full Documentation](docs/README.md)

## Contributing

Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is appreciated.

Please see [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## Support

If you encounter any issues, please check [issues page](https://github.com/Dooders/AgentFarm/issues) or open a new issue.