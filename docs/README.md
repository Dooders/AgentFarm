# AgentFarm documentation

Navigation hub for the published docs site and repository guides.

## New here?

1. [Installation](getting-started/installation.md)
2. [First simulation](getting-started/first-simulation.md)
3. [Module overview](module_overview.md) — architecture and components

## By role

| Role | Start here |
|------|------------|
| **User / researcher** | [Experiment QuickStart](ExperimentQuickStart.md) · [Experiments](experiments.md) · [Devlog](devlog/index.md) |
| **Developer** | [CONTRIBUTING](../CONTRIBUTING.md) · [Developer guide](developer-guide.md) · [Core architecture](core_architecture.md) |
| **Maintainer** | [Release process](RELEASE.md) |
| **Operator** | [Deployment](deployment.md) · [Logging guide](logging_guide.md) · [Monitoring](monitoring.md) |

## Guides

- [Configuration](config/configuration_guide.md)
- [Neural recombination](guides/neural-recombination.md) — distillation, PTQ, QAT, crossover
- [Neural recombination runbook](howto/neural_recombination_runbook.md) — full pipeline walkthrough
- [Usage examples](usage_examples.md)
- [API reference](api_reference.md)

## Design & research

- [Design RFCs](design/agent_loop.md) — agent loop, chromosomes, crossover
- [Devlog](devlog/index.md) — build notes and experiment outcomes
- [Experiments catalog](experiments.md)

## Package docs

Module-local READMEs live next to code under `farm/` (for example [logging](../farm/utils/logging/README.md) and [decision](../farm/core/decision/README.md)).
