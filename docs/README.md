# AgentFarm Documentation

Welcome to the comprehensive AgentFarm documentation. This guide will help you navigate the available resources to get the most out of this powerful multi-agent reinforcement learning simulation platform.

## 📚 Documentation Overview

AgentFarm is a sophisticated platform for researching complex adaptive systems through agent-based modeling. This documentation provides everything from basic tutorials to advanced API references.

### 🗂️ Documentation Structure

- **[Module Overview](module_overview.md)** - High-level introduction to AgentFarm's architecture and capabilities
- **[Core Architecture](core_architecture.md)** - Deep dive into fundamental components and design patterns
- **[Usage Examples](usage_examples.md)** - Practical tutorials and code examples
- **[Configuration Guide](configuration_guide.md)** - Comprehensive configuration system documentation
- **[API Reference](api_reference.md)** - Complete API documentation for all modules

## 🚀 Quick Start Guides

- [Simulation QuickStart Guide](SimulationQuickStart.md) - Beginner-friendly setup guide
- [Experiment QuickStart Guide](ExperimentQuickStart.md) - Running parameter studies

## 🏗️ Core System Components

### Agent System
- [Agents](agents.md) - Agent architecture, types, and behaviors
- [Perception](perception.md) - Sensory systems and observation processing
- [State System](state_system.md) - Agent state management and transitions

### Action System
- [Action System](action_system.md) - Core action execution framework
- [Action Data](action_data.md) - Action data structures and management
- [Move Module](move_module.md) - Agent movement and navigation
- [Gather Module](gather_module.md) - Resource collection mechanisms
- [Attack Module](attack_module.md) - Conflict resolution systems
- [Reproduce Module](reproduce_module.md) - Agent reproduction and inheritance
- [Select Module](select_module.md) - Decision-making mechanisms
- [Share Module](share_module.md) - Resource sharing and cooperation

### Observation & Channels
- [Dynamic Channel System](dynamic_channel_system.md) - Flexible observation channel framework
- Custom channel implementation examples (see [Usage Examples](usage_examples.md))

### Data & Analysis
- [Data API](data/data_api.md) - Interfaces for data access and manipulation
- [Data Services](data/data_services.md) - Data processing and management services
- [Data Retrieval](data/data_retrieval.md) - Methods for accessing simulation data
- [Database Schema](data/database_schema.md) - Data structure and organization
- [Metrics](metrics.md) - Performance and behavior measurement systems
- [Repositories](repositories.md) - Data storage and retrieval patterns
- [Health Resource Analysis](health_resource_analysis.md) - Agent health and resource utilization analysis
- [Experiment Analysis](experiment_analysis.md) - Tools for analyzing experiment results

### AI & Learning
- [Deep Q Learning](deep_q_learning.md) - Reinforcement learning implementation
- [Redis Agent Memory](redis_agent_memory.md) - Distributed agent memory system

## 🔬 Research & Experiments

- [Research](research.md) - Research methodologies and frameworks
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
2. **Custom Agents**: Extend BaseAgent → Implement custom decision logic → Register behaviors
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