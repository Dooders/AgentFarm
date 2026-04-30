# AgentFarm

![Project Status](https://img.shields.io/badge/status-in%20development-orange)

**AgentFarm** is an advanced simulation and computational modeling platform for exploring complex systems. Designed as a comprehensive workbench, it enables users to run, analyze, and compare simulations with ease.

This repository is being developed to support research in the [Dooders](https://github.com/Dooders) project, focusing on complex adaptive systems and agent-based modeling approaches.

> **Note**: This project is currently in active development. APIs and features may change between releases. See the [Contributing Guidelines](CONTRIBUTING.md) for information on getting involved.

## Key Features

### [Agent-Based Modeling & Analysis](docs/features/agent_based_modeling_analysis.md)
- Run complex simulations with interacting, adaptive agents
- Study emergent behaviors and system dynamics
- Track agent interactions and environmental influences
- Analyze trends and patterns over time

### [Customization & Flexibility](docs/features/customization_flexibility.md)
- Define custom parameters, rules, and environments
- Create specialized agent behaviors and properties
- Configure simulation parameters and conditions
- Design custom experiments and scenarios
- Opt-in [initial genotype diversity](docs/initial_diversity.md) for any simulation - seed the starting population with `independent_mutation`, `unique`, or `min_distance` modes via `SimulationConfig.initial_diversity`

### [AI & Machine Learning](docs/features/ai_machine_learning.md)
- Reinforcement learning for agent adaptation
- Automated data analysis and insight generation
- Pattern recognition and behavior prediction
- Evolutionary algorithms and genetic modeling

### [Data & Visualization](docs/features/data_visualization.md)
- Comprehensive data collection and metrics
- Simulation visualization tools
- Charting and plotting utilities
- Automated report generation

### [Research Tools](docs/features/research_tools.md)
- Parameter sweep experiments
- Comparative analysis framework
- Experiment replication tools
- Professional-grade logging with structlog for rich, contextual, machine-readable logs

### [Data System](docs/features/data_system.md)
- Layered system with database, repositories, analyzers, and services
- Action statistics, behavioral clustering, causal analysis, and pattern recognition
- Repository pattern for efficient data retrieval and querying
- Coordinated analysis operations with built-in error handling
- Experiment database for comparing multiple simulation runs

### [Spatial Indexing & Performance](docs/features/spatial_indexing_performance.md)
- KD-tree, Quadtree, and Spatial Hash Grid implementations for efficient proximity queries
- Dirty region tracking system that only updates changed regions, reducing computational overhead by up to 70%
- Choose optimal spatial index type for different query patterns (radial, range, neighbor queries)
- Comprehensive metrics and statistics for spatial query optimization
- Efficiently handles thousands of agents with minimal performance degradation


## Logging & Observability ✨

AgentFarm now includes **professional-grade structured logging** with `structlog`:

**Key Features:**
- 🔍 Rich contextual logs (simulation_id, step, agent_id, etc.)
- 📊 Machine-readable JSON output for analysis
- 🎨 Multiple formats: Console (colored), JSON, plain text
- ⚡ Performance-optimized with log sampling
- 🛡️ Automatic sensitive data censoring

**Quick Example:**
```python
from farm.utils import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger(__name__)

logger.info("simulation_started", num_agents=100, num_steps=1000)
```

**Learn More:**
- [Quick Reference](docs/LOGGING_QUICK_REFERENCE.md)
- [Complete Documentation](docs/logging_guide.md)
- [Logging module overview](farm/utils/logging/README.md) (inline examples)

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

# Install dependencies and the package in editable mode (required for `farm` imports)
pip install -r requirements.txt
pip install -e .

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
python run_simulation.py --environment development --steps 1000
```

**Experiments and Tools**
```bash
# Run experiments with parameter variations
python farm/core/cli.py --mode experiment --environment development --experiment-name test --iterations 3

# Visualize existing simulation results
python farm/core/cli.py --mode visualize --db-path simulations/simulation.db

# Generate analysis reports
python farm/core/cli.py --mode analyze --db-path simulations/simulation.db
```

**Results**
All simulation results are saved in the `simulations` directory with database files, logs, and analysis reports.

### API Server

Run the REST/WebSocket API for managing simulations and analysis:

```bash
# From repository root
python -m farm.api.server
# or
python farm/api/server.py
```

Defaults:
- Binds to port 5000
- Writes logs to `logs/api.log`

Key endpoints:
- `POST /api/simulation/new` — create and run a simulation
- `GET /api/simulation/<sim_id>/step/<step>` — fetch a specific step
- `GET /api/simulation/<sim_id>/analysis` — run analysis
- `GET /api/simulation/<sim_id>/export` — export data
- `WS /ws/<client_id>` — WebSocket for streaming-style client messages

### Benchmarks

```bash
# List spec-driven experiments
python -m benchmarks.run_benchmarks --list

# Run from a YAML spec (example: in-memory DB benchmark)
python -m benchmarks.run_benchmarks --spec benchmarks/specs/memory_db_baseline.yaml
```
See `benchmarks/README.md` for details and recommended configurations.

### Distillation, PTQ, and optional QAT

After distilling student Q-networks you can apply **8-bit post-training quantization (PTQ)** and, if accuracy is not good enough, **quantization-aware training (QAT)**. Full behavior, PyTorch version notes, and CPU/CUDA limits are in [`farm/core/decision/training/quantize_ptq.py`](farm/core/decision/training/quantize_ptq.py) and [`farm/core/decision/training/quantize_qat.py`](farm/core/decision/training/quantize_qat.py).

**Typical flow**

1. **Distill** float students (`student_A.pt` / `student_B.pt`):

   ```bash
   python scripts/run_distillation.py --help
   ```

2. **PTQ** (default: dynamic weight-only `qint8`; static mode needs calibration states):

   ```bash
   python scripts/quantize_distilled.py \
       --checkpoint-dir checkpoints/distillation \
       --output-dir checkpoints/quantized
   ```

   For static PTQ, use `--states-file` or synthetic `--n-states` / `--seed`; calibration volume uses `--calibration-batches` and `--calibration-batch-size` (defaults 10 / 64). Match distillation architecture with `--input-dim`, `--output-dim`, `--parent-hidden` (defaults **8**, **4**, **64**).

3. **Validate** float students (optional): `python scripts/validate_distillation.py --help`

4. **Validate quantized vs float** (CPU): `python scripts/validate_quantized.py --help`. The validator loads quantized checkpoints as full-model pickles, so pass `--allow-unsafe-unpickle` only for trusted artifacts. The JSON report includes median/mean/p95 single-sample latency, optional **throughput** (`--throughput-batch-size`), **memory** RSS snapshots, float–quant **MSE/KL/top-k** agreement, and optional **teacher** metrics if `parent_*.pt` is found under `--float-dir` / `--teacher-dir` or via `--teacher-*-ckpt`.

5. **Evaluate a crossover child vs both parents** (offline Q metrics, versioned JSON): `python scripts/validate_recombination.py --help`. Baselines: **child vs parent A**, **child vs parent B**, optional **parent A vs parent B** (`--include-parent-baseline`), plus **oracle** agreement in the report summary. Use the same `--states-file` / `--seed` / `--n-states` pattern as other validation scripts. For **quantized** full-model checkpoints (PTQ or post-QAT `torch.save` exports), add `--parent-a-quantized`, `--parent-b-quantized`, and/or `--child-quantized` together with `--allow-unsafe-unpickle`; those roles are loaded with `load_quantized_checkpoint` and run on **CPU**.

6. **Search many crossover + fine-tune combinations** (leaderboard + manifest): `python scripts/run_crossover_search.py --help`. Presets include `minimal` / `default`, plus **`minimal-qat`** / **`default-qat`** (adds a `short_qat` / `ptq_dynamic` regime). Use **`--workers N`** for process-parallel children (float `BaseQNetwork` parents only). Quick check: `make crossover-search-smoke`. Design notes: `docs/design/crossover_search_space.md`, strategy semantics: `docs/design/crossover_strategies.md`.

**Crossover from PTQ parent paths (Python):** [`initialize_child_from_crossover`](farm/core/decision/training/crossover.py) can auto-detect a **dynamic** PTQ sidecar next to a `.pt` file (same JSON shape as `PostTrainingQuantizer.save_checkpoint`) and load via `load_quantized_checkpoint`. That path uses full-model unpickling (`weights_only=False`); pass **`allow_unsafe_unpickle=True` only for trusted checkpoints**. Static PTQ sidecars are not auto-loaded here—use float state dicts or in-memory modules. Details: [`docs/design/crossover_strategies.md`](docs/design/crossover_strategies.md).

**Optional QAT** (after PTQ if action agreement or Q-error is unacceptable): weight-only fake quant on linear layers, same int8 export format as PTQ after convert.

```bash
python scripts/qat_distilled.py \
    --checkpoint-dir checkpoints/distillation \
    --output-dir checkpoints/qat
```

Use `--teacher-a-ckpt` / `--student-a-ckpt` (and `*-b-*` for pair B) when paths are not under a single `--checkpoint-dir`; see `python scripts/qat_distilled.py --help` for epochs, learning rate, and `--no-convert` (float QAT checkpoint only). Quantized QAT checkpoints work with `scripts/validate_quantized.py` like PTQ outputs.

**Tests:** `pytest tests/decision/test_ptq.py tests/decision/test_validate_quantized.py tests/decision/test_qat.py tests/decision/test_crossover_search.py`

### Testing

```bash
# Using pytest
pytest -q

# Or using the bundled test runner
python run_tests.py
```

## Documentation

For detailed documentation and advanced usage:
- [Changelog](CHANGELOG.md)
- [Agent Loop Design](docs/design/agent_loop.md)
- [Agent Design](docs/design/Agent.md)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [Deployment](docs/deployment.md)
- [Monitoring & Performance](docs/monitoring.md)
- [Benchmarking & Profiling Report](benchmarks/reports/0.1.0/benchmark_profiling_summary_report.md)
- [Core Architecture](docs/core_architecture.md)
- [Hyperparameter Chromosome Design](docs/design/hyperparameter_chromosome.md)
- [Devlog](docs/devlog/index.md)
- [Latest Devlog: Evolving Hyperparameter Genomes in Foraging and Learning Agents](docs/devlog/2026-04-23-evolving-hyperparameter-genomes-foraging-learning-agents.md)
- [Full Documentation Index](docs/README.md)

### GitHub Pages

This repository includes a GitHub Pages workflow that publishes content from `docs/`.

- Docs home: `https://dooders.github.io/AgentFarm/`
- Devlog index: `https://dooders.github.io/AgentFarm/devlog/`

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