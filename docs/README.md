## Documentation

Welcome to the Live Simulation Config Explorer documentation.

- User Guide: `docs/user-guide.md`
- Developer Guide: `docs/developer-guide.md`
- IPC API Reference: `docs/ipc-api.md`
- Deployment Guide: `docs/deployment.md`
- Monitoring & Performance: `docs/monitoring.md`

If you are new, start with the User Guide, then see the Developer Guide for setup and contribution.
# AgentFarm Documentation

Welcome to the comprehensive AgentFarm documentation. This guide will help you navigate the available resources to get the most out of this powerful multi-agent reinforcement learning simulation platform.

## 📚 Documentation Overview

AgentFarm is a sophisticated platform for researching complex adaptive systems through agent-based modeling. This documentation provides everything from basic tutorials to advanced API references.

### 🗂️ Documentation Structure

- **[Module Overview](module_overview.md)** - High-level introduction to AgentFarm's architecture and capabilities
- **[Core Architecture](core_architecture.md)** - Deep dive into fundamental components and design patterns
- **[Electron Config Explorer Architecture](electron/config_explorer_architecture.md)** - Electron renderer/main boundaries, IPC, and migration plan
- **[Usage Examples](usage_examples.md)** - Practical tutorials and code examples
- **[Configuration Guide](configuration_guide.md)** - Comprehensive configuration system documentation (includes Electron Config Explorer compare/diff/presets/status bar)
- **[API Reference](api_reference.md)** - Complete API documentation for all modules

## 🚀 Quick Start Guides

- [Experiment QuickStart Guide](ExperimentQuickStart.md) - Running parameter studies

## 🏗️ Core System Components

### Agent System
- [Agents](agents.md) - Agent architecture, types, and behaviors
- [Perception](perception.md) - Sensory systems and observation processing
- [State System](state_system.md) - Agent state management and transitions

### Action System
- [Action System](action_system.md) - Core action execution framework
- [Action Data](action_data.md) - Action data structures and management

### Observation & Channels
- [Dynamic Channel System](dynamic_channel_system.md) - Flexible observation channel framework
- Custom channel implementation examples (see [Usage Examples](usage_examples.md))

#### Sparse Observation Storage (HYBRID mode)
The observation system supports tensor-backed sparse point storage via `SparsePoints` for channels that are naturally sparse (e.g., allies/enemies/trajectories). This reduces Python dict overhead and improves GPU transfers.

- Configuration (in `ObservationConfig`):
  - `storage_mode`: `HYBRID` (default) uses `SparsePoints` for point-sparse channels; `DENSE` writes directly to a dense tensor.
  - `sparse_backend`: `"scatter"` (default) or `"coo"`. Use `coo` when `sum` reduction with many duplicates is common.
  - `default_point_reduction`: `"max"` (default), `"sum"`, or `"overwrite"`.
  - `channel_reduction_overrides`: per-channel overrides by channel name.

- Reductions:
  - `max`: keep maximum per index (deterministic, good for presence maps)
  - `sum`: accumulate contributions (good for intensities)
  - `overwrite`: last write wins (order-dependent; not deterministic with duplicates)

- Metrics via `AgentObservation.get_metrics()`:
  - `dense_bytes`, `sparse_points`, `sparse_logical_bytes`, `memory_reduction_percent`
  - `cache_hits`, `cache_misses`, `dense_rebuilds`, `dense_rebuild_time_s_total`
  - `sparse_apply_calls`, `sparse_apply_time_s_total`

Example:
```python
config = ObservationConfig(
  R=6,
  sparse_backend="scatter",
  default_point_reduction="max",
  channel_reduction_overrides={"TRAILS": "sum"},
)
obs = AgentObservation(config)
tensor = obs.tensor()
metrics = obs.get_metrics()
```

### Data & Analysis
- [Data API](data/data_api.md) - Interfaces for data access and manipulation
- [Data Services](data/data_services.md) - Data processing and management services
- [Data Retrieval](data/data_retrieval.md) - Methods for accessing simulation data
- [Database Schema](data/database_schema.md) - Data structure and organization
- [Metrics](metrics.md) - Performance and behavior measurement systems
- [Repositories](data/repositories.md) - Data storage and retrieval patterns
- [Health Resource Analysis](health_resource_analysis.md) - Agent health and resource utilization analysis
- [Experiment Analysis](experiment_analysis.md) - Tools for analyzing experiment results

### AI & Learning
- [Deep Q Learning](deep_q_learning.md) - Reinforcement learning implementation
- [Redis Agent Memory](redis_agent_memory.md) - Distributed agent memory system

## 🔬 Research & Experiments

- [Experiments](experiments.md) - Experiment design and implementation
- Experiment case studies (see [Usage Examples](usage_examples.md))

## 📖 Specialized Guides & Tutorials

### Tutorials
Step-by-step guides for specific use cases:
- Basic simulation setup
- Custom agent implementation
- Extending observation channels
- Experiment management
- Analysis and visualization

### Analysis Techniques
- [Analysis](analysis/) - Advanced analysis techniques and examples
- [Comparative Analysis](analysis/comparative_analysis.md)
- [Agent Behavior Analysis](analysis/agent_analysis.md)
- [Experiment Analysis](analysis/experiment_analysis.md)

### Experiments
- [Experiments](experiments/) - Detailed experiment configurations
- Parameter sweep examples
- Comparative studies
- Replication techniques

## 🔧 Technical Reference

### API Documentation
- **[Complete API Reference](api_reference.md)** - All classes, methods, and functions
- Module-specific API documentation
- Type hints and signatures
- Usage examples for each API

### Configuration System
- **[Configuration Guide](configuration_guide.md)** - Comprehensive configuration reference
- YAML configuration format
- Parameter validation
- Configuration management tools

### Database & Persistence
- [Interaction Edge Logging](data/interaction_edges.md)
- Database utilities and maintenance
- Data export and import procedures

## 🎯 Use Cases & Examples

### Basic Usage Patterns
1. **Simple Simulation**: Load config → Create environment → Add agents → Run simulation
2. **Custom Agents**: Use AgentFactory → Create custom components → Implement custom behaviors
3. **Extended Observations**: Create ChannelHandler → Register channel → Process observations
4. **Parameter Studies**: Define parameter ranges → Run experiment → Analyze results

### Advanced Scenarios
- Multi-agent cooperation studies
- Resource competition dynamics
- Learning algorithm comparison
- Emergent behavior analysis
- Scalability testing

## 🤝 Contributing

We welcome contributions to both the platform and its documentation:

### Ways to Contribute
- **Bug Reports**: Use [GitHub Issues](https://github.com/Dooders/AgentFarm/issues) for bugs
- **Feature Requests**: Propose new features via GitHub Issues
- **Documentation**: Improve existing docs or add new guides
- **Code Contributions**: Submit pull requests for enhancements

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/AgentFarm.git`
3. Set up development environment: `pip install -r requirements.txt`
4. Run tests: `python -m pytest`
5. Submit pull request

### Documentation Guidelines
- Use clear, concise language
- Include code examples where helpful
- Follow existing documentation structure
- Test examples to ensure they work
- Update this README when adding new documentation

## 📞 Support & Community

### Getting Help
- **Documentation**: Start with this README and the guides listed above
- **Issues**: Check existing [GitHub Issues](https://github.com/Dooders/AgentFarm/issues)
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Examples**: Review the `examples/` directory for working code samples

### Community Resources
- **Research Papers**: See `docs/research/` for related academic work
- **Tutorials**: Community-contributed tutorials and guides
- **Case Studies**: Real-world applications and results

## 📋 Roadmap & Future Development

### Planned Features
- Enhanced visualization tools
- Distributed simulation support
- Additional agent types and behaviors
- Advanced analysis frameworks
- Integration with popular RL libraries

### Research Directions
- Complex social dynamics modeling
- Evolutionary algorithm integration
- Multi-objective optimization
- Real-time adaptive systems

## 📄 License & Attribution

This project is part of the [Dooders](https://github.com/Dooders) research initiative exploring complex adaptive systems through computational modeling.

---

**🎓 Learning Path Recommendation:**
1. Start with [Module Overview](module_overview.md) for high-level understanding
2. Follow [Usage Examples](usage_examples.md) for hands-on experience
3. Dive into [Configuration Guide](configuration_guide.md) for customization
4. Reference [API Documentation](api_reference.md) for development
5. Explore specialized guides for advanced topics

**Happy simulating! 🚀** 