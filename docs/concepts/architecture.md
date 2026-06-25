# Architecture

AgentFarm is a Python-first multi-agent simulation platform built on PettingZoo's AECEnv interface. The codebase under `farm/` separates environment orchestration, agent decision-making, persistence, and analysis into composable modules.

For API signatures see [API reference](../reference/api-reference.md). For hands-on patterns see [Usage examples](../guides/usage-examples.md).

## Module map

```
farm/
├── core/           # Environment, agents, actions, observations, spatial, simulation loop
├── config/         # SimulationConfig and YAML loading
├── database/       # SQLite persistence and repositories
├── analysis/       # Post-run analysis modules and services
├── runners/        # ExperimentRunner and batch orchestration
├── api/            # REST/WebSocket server
└── utils/          # Logging and shared helpers
```

Logical layering inside `farm.core`:

| Layer | Packages / modules | Responsibility |
|-------|-------------------|----------------|
| World | `environment`, `resource_manager`, `spatial/` | Grid world, resources, proximity queries |
| Agents | `agent/`, `decision/` | AgentCore, behaviors, RL/decision stack |
| Perception | `observations`, `channels`, `perception` | Local egocentric tensors and channel handlers |
| Actions | `action.py` | Action registry, built-in move/gather/attack/share/reproduce |
| Runtime | `simulation.py`, `metrics_tracker`, `collector` | Stepping, metrics, data capture |

## Simulation loop

Each step follows a fixed pipeline:

1. **Pre-step** — refresh instant channels, regenerate resources
2. **Actions** — each alive agent executes a discrete action from the registry
3. **World updates** — combat, sharing, reproduction side effects; agent state transitions
4. **Observations** — channel handlers write into each agent's local tensor (sparse or dense)
5. **Persistence** — metrics and state logged to SQLite when a database is attached
6. **Cleanup** — remove dead agents, apply temporal decay on dynamic channels

The `Environment` class coordinates this loop and exposes PettingZoo-compatible `reset`, `step`, `observation_space`, and `action_space`.

## Environment and spatial indexing

The environment owns the 2D grid, agent lifecycle, and resource distribution. Proximity queries (nearby agents, nearest resource) go through a **spatial index** — KD-tree, quadtree, or spatial hash — with dirty-region tracking so only changed cells are reindexed.

See [Spatial indexing](spatial/spatial_indexing.md) for backend choice and performance notes.

## Agent system

Agents are built from **AgentCore** plus pluggable **components** (movement, perception, combat, learning) and an **IAgentBehavior** implementation. **AgentFactory** constructs agents for a run from `SimulationConfig`.

Decision-making lives in `farm.core.decision` (DQN and related training/distillation tooling). Hyperparameters can be encoded in a typed **HyperparameterChromosome** inherited at reproduction.

Deep dives: [Agents and decisions](agents-and-decisions.md) · [Actions](actions.md) · [Deep Q-learning](deep-q-learning.md) · [Initial diversity](initial-diversity.md)

## Observation and channel system

Each agent maintains an egocentric `(channels, height, width)` tensor centered on its position. Built-in channels include self/allies/enemies HP, resources, visibility, trails, and damage heat.

Channels are registered through **ChannelRegistry** and implement **ChannelHandler** with one of three behaviors:

| Behavior | Meaning |
|----------|---------|
| `INSTANT` | Cleared and rewritten every step |
| `DYNAMIC` | Persists with configurable gamma decay |
| `PERSISTENT` | Accumulates until explicitly cleared |

`ObservationConfig` controls radius, dtype/device, sparse vs dense storage, and per-channel reduction (`max`, `sum`, `overwrite`).

See [Observation channels](observation-channels.md) · [Dynamic channel system](dynamic-channel-system.md) · [Perception system](perception-system.md)

## Configuration

Runs are driven by **SimulationConfig** (nested fields for environment, population, resources, learning, observation, etc.) loaded from YAML under `farm/config/`. Parameter sweeps use **ExperimentRunner** with a list of top-level config variation dicts.

See [Configuration guide](../reference/config/configuration_guide.md) · [Experiment runner](../guides/experiment-runner.md)

## Data and analysis

Simulation output is stored in SQLite (`simulations/*.db`). The stack is:

```
SimulationDatabase → repositories → analyzers / AnalysisService → reports & plots
```

Schema covers agent states, actions, population metrics, reproduction events, and learning telemetry. Analysis modules (population, spatial, learning, combat, …) live under `farm.analysis` and are documented in [Analysis modules](../reference/analysis/modules/README.md).

See [Data API](../reference/data/data_api.md) · [Database schema](../reference/data/database_schema.md)

## Architectural patterns

**Component composition** — extend behavior via new components or `IAgentBehavior` subclasses rather than editing `Environment` internals.

**Protocol-based DI** — `farm.core.interfaces` defines `DatabaseProtocol`, `RepositoryProtocol`, and related abstractions so tests can swap mocks and backends.

**Registry extensibility** — actions and observation channels register at import time; new types do not require core edits.

**Factory creation** — agents and some services are constructed through factories wired from config.

See [Dependency injection](dependency-injection.md) for wiring details.

## Reinforcement learning integration

- **Observation space** — multi-channel Box tensor, egocentric, configurable radius
- **Action space** — Discrete index into the action registry (dynamic size if actions are added)
- **Rewards** — computed in AgentCore from configurable goal weights (including evolvable chromosome loci)

Training, distillation, PTQ/QAT, and crossover tooling live under `farm/core/decision/training/`; see [Neural recombination guide](../guides/neural-recombination.md).

## Extension points

| Goal | Approach |
|------|----------|
| New observation channel | Subclass `ChannelHandler`, `register_channel()` |
| New agent behavior | Subclass `IAgentBehavior` or add components |
| Custom world rules | Subclass `Environment`, override hooks |
| New analysis | Add module under `farm.analysis` or call `AnalysisService` |
| HTTP/WS control | `uvicorn farm.api.server:app` — see [First simulation](../getting-started/first-simulation.md) |

## Performance considerations

- Tune observation radius and channel count for memory (sparse `HYBRID` mode helps)
- Pick spatial index type for your query pattern (radial vs range vs neighbor-heavy)
- Use benchmark specs under `benchmarks/specs/` to regression-test hot paths
- Long runs: monitor SQLite size and structlog sampling settings

## Related documentation

| Topic | Document |
|-------|----------|
| Getting started | [Installation](../getting-started/installation.md) · [First simulation](../getting-started/first-simulation.md) |
| Design RFCs | [Design index](../design/README.md) |
| Research writeups | [Devlog](../research/devlog/index.md) · [Experiments](../research/experiments-catalog.md) |
| Legacy feature pages | [Archive](../archive/features/FEATURES.md) (deprecated stubs) |
