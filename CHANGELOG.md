# Changelog

All notable changes to this project are documented in this file.

This repository uses an automated Copilot-assisted workflow to draft changelog updates after merges into `main`.

## Format

Each entry should include:

- what changed
- why it matters
- user/developer impact

Suggested grouping:

- Added
- Changed
- Fixed
- Docs
- Performance

## Entries

### 2026-04-18

#### Added

- **Reflective boundary mutation** (`BoundaryMode.REFLECT`) for `mutate_chromosome` — mutated gene values that overshoot a bound now bounce back instead of sticking at the wall, preventing boundary collapse and preserving population diversity. ([#708](https://github.com/Dooders/AgentFarm/pull/708), [#709](https://github.com/Dooders/AgentFarm/pull/709))
- **Soft boundary penalties** (`BoundaryPenaltyConfig` + `compute_boundary_penalty`) — an opt-in, linearly-ramped fitness penalty that discourages genes from crowding near their min/max limits without hard constraints. Subtract the result from raw fitness to apply. ([#708](https://github.com/Dooders/AgentFarm/pull/708))

#### Changed

- Removed redundant boundary handling parameters from `EvolutionExperimentConfig`; boundary settings are now defined exclusively through `BoundaryPenaltyConfig` to avoid duplication. ([#709](https://github.com/Dooders/AgentFarm/pull/709))
- Improved `copilot-changelog-after-merge.yml` workflow based on review feedback. ([#707](https://github.com/Dooders/AgentFarm/pull/707))

#### Fixed

- **Prototype pollution** in Config Explorer editor (`farm/editor`): hardened `deepMerge` and `deepClone` to skip dangerous keys (`__proto__`, `constructor`, `prototype`) and only recurse into own-property destinations. Addresses code scanning alert #5. ([#711](https://github.com/Dooders/AgentFarm/pull/711))
- Added explicit `permissions: contents: read` to `tests.yml` workflow so all test jobs run with least-privilege `GITHUB_TOKEN` scopes. Addresses code scanning alert #3. ([#712](https://github.com/Dooders/AgentFarm/pull/712))
- Added explicit `permissions: contents: read` to `deterministic-simulation.yml` workflow. Addresses code scanning alert #1. ([#713](https://github.com/Dooders/AgentFarm/pull/713))

---

### 2026-04-17

#### Added

- **`HyperparameterChromosome`** — a typed, bounded, evolvable hyperparameter model with encoding/decoding, mutation (Gaussian), and crossover utilities. Each gene can be individually marked evolvable or fixed, with per-gene min/max bounds. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- **`EvolutionExperiment`** — a generation-based hyperparameter evolution runner supporting tournament and roulette selection strategies, lineage tracking, and persisted `evolution_generation_summaries.json` / `evolution_lineage.json` artifacts. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `Genome.tournament_selection` and `Genome.roulette_selection` helpers with optional seeding and index-returning. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- CLI and plotting scripts for hyperparameter evolution runs. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Design document (`docs/design/hyperparameter_chromosome.md`) and experiment note (`docs/experiments/hyperparameter_evolution_convergence.md`) covering the new chromosome model, mutation strategies, and evolution convergence analysis. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Established GitHub Pages publishing for repository docs and devlog. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Added a dedicated devlog section and an initial post covering DNA-style hyperparameter evolution design and outcomes. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Introduced an automated workflow to request Copilot-authored changelog updates after merges to `main`. ([#707](https://github.com/Dooders/AgentFarm/pull/707))

#### Changed

- `AgentCore.reproduce()` now derives offspring configs from a mutated `HyperparameterChromosome`, storing the resulting chromosome on the child agent. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Agent component config construction maps fields from the `learning` config section into the agent's decision config, enabling more flexible agent initialization. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `EnvironmentalFactorsConfig` now raises `ValueError` for any normalized factor outside `[0.0, 1.0]` (or `tolerance_width <= 0`) instead of silently clamping, making misconfigured simulations fail fast and loudly. ([#690](https://github.com/Dooders/AgentFarm/pull/690))

#### Fixed

- Reproduction error handling hardened: a failed resource-cost deduction now aborts the reproduction cleanly rather than leaving partial state. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `Environment.rollback_partial_agent_add` now compensates DB side effects by deleting the `AgentModel` row via a transaction instead of writing a spurious death record, and also purges associated metrics/DB buffers. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `DataLogger` now requires a truthy `simulation_id` and raises `ValueError` when it is missing or empty. ([#690](https://github.com/Dooders/AgentFarm/pull/690))

#### Security

- Clarified usage and security implications of the `allow_unsafe_unpickle` flag for checkpoint loading in crossover/fine-tune pipelines; explicit opt-in is now documented and recorded in audit reports. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
