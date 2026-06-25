# Architecture

High-level map of AgentFarm's structure.

## Overview

AgentFarm is a Python-first multi-agent simulation platform built on PettingZoo's AECEnv framework. The codebase separates environment orchestration, agent decision-making, data persistence, and analysis into composable modules under `farm/`.

## Read next

| Topic | Document |
|-------|----------|
| Module map and component tour | [Module overview](module-overview.md) |
| Core design patterns and data flow | [Core architecture](core-architecture.md) |
| Agent types and decision stack | [Agents and decisions](agents-and-decisions.md) |
| Action execution framework | [Actions](actions.md) |
| Observations and channels | [Observation channels](observation-channels.md) · [Dynamic channels](dynamic-channel-system.md) |
| Spatial indexing | [Spatial indexing](spatial/spatial_indexing.md) |
| Dependency injection | [Dependency injection](dependency-injection.md) |

## Package layout

```
farm/
├── core/          # Environment, agents, actions, observations, spatial
├── config/        # Simulation configuration
├── database/      # Persistence and repositories
├── analysis/      # Post-run analysis modules
├── runners/       # Experiment orchestration
├── api/           # REST/WebSocket server
└── utils/         # Logging and shared helpers
```

See [API reference](../reference/api-reference.md) for entry-point signatures.
