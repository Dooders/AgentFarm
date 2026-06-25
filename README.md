# AgentFarm

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

Simulation and analysis platform for agent-based modeling, reinforcement learning experiments, and complex adaptive systems research in the [Dooders](https://github.com/Dooders) project.

> **Note:** Active development — APIs may change between releases. See [CHANGELOG](CHANGELOG.md).

## Features

- Multi-agent simulations with configurable genomes, actions, and spatial indexing
- Experiment runner, SQLite-backed metrics, and analysis pipeline
- RL/decision stack (DQN, distillation, evolution experiments)
- REST/WebSocket API and structured logging with `structlog`

## Quick start

```bash
git clone https://github.com/Dooders/AgentFarm.git && cd AgentFarm
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python run_simulation.py --environment development --steps 1000
```

Results appear under `simulations/`. See [Installation](docs/getting-started/installation.md) and [First simulation](docs/getting-started/first-simulation.md) for prerequisites, Redis, API server, and benchmarks.

## Documentation

- **Docs site:** [dooders.github.io/AgentFarm](https://dooders.github.io/AgentFarm/)
- **Hub:** [docs/README.md](docs/README.md)
- **Research:** [Devlog](docs/devlog/index.md) · [Experiments](docs/experiments.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Support

[Open an issue](https://github.com/Dooders/AgentFarm/issues) on GitHub.
