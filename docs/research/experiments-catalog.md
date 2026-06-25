# Experiments

AgentFarm ships with a growing set of **defined experiments** — concrete
research designs, runners, and case studies that exercise the simulation
framework against a specific question. Each experiment below has its own
dedicated documentation covering motivation, configuration, output
artifacts, and (where available) results.

> Looking for the generic multi-iteration runner instead of a specific
> experiment? See [ExperimentRunner — Running Multi-Iteration Simulations](../guides/experiment-runner.md).

## Evolutionary dynamics

Experiments that study how learning policies, hyperparameters, or
genotypes change over time under selection.

### Intrinsic Evolution

Each agent carries its own `HyperparameterChromosome`. Offspring inherit
it (optionally crossed with a co-parent and mutated) and selection
emerges implicitly from survival and reproduction in the shared resource
environment — no external fitness function, no separate evaluation
runs.

- **Status:** Implemented and reproducible end-to-end.
- **Runner:** `IntrinsicEvolutionExperiment`
  ([`farm/runners/intrinsic_evolution_experiment.py`](../farm/runners/intrinsic_evolution_experiment.py))
- **CLI:** [`scripts/run_intrinsic_evolution_experiment.py`](../scripts/run_intrinsic_evolution_experiment.py)
- **Docs:**
  - [Design and reference](experiments/intrinsic_evolution/intrinsic_evolution.md)
  - [10,000-step results](experiments/intrinsic_evolution/RESULTS.md)
  - [Stable profile comparison](experiments/intrinsic_evolution/stable_profile_comparison.md)
  - [Analysis summary](experiments/intrinsic_evolution/analysis/analysis_summary.md)

### Hyperparameter Evolution Convergence

A classical generational genetic algorithm over `HyperparameterChromosome`
values, evaluated by short simulation rollouts. Documents how to capture
and interpret learning-rate convergence, the `stable_hyper_evo` preset
that prevents lower-bound collapse, and the adaptive-mutation controller.

- **Status:** Implemented and reproducible end-to-end.
- **Runner:** `EvolutionExperiment`
  ([`farm/runners/evolution_experiment.py`](../farm/runners/evolution_experiment.py))
- **CLI:** [`scripts/run_evolution_experiment.py`](../scripts/run_evolution_experiment.py)
- **Docs:** [Hyperparameter Evolution Convergence](experiments/hyperparameter_evolution_convergence.md)

### Multi-Seed Cohort

Wraps any evolution configuration in *N* independent random-seed runs
and aggregates the results into a single summary (mean, standard
deviation, convergence rate, lower-bound occupancy). Designed to remove
single-run seed variance from configuration comparisons.

- **Status:** Implemented and reproducible end-to-end.
- **Runner:** `CohortRunner`
  ([`farm/runners/cohort_runner.py`](../farm/runners/cohort_runner.py))
- **CLI:** [`scripts/run_cohort_experiment.py`](../scripts/run_cohort_experiment.py)
- **Docs:** [Multi-Seed Cohort Runner](experiments/multi_seed_cohort.md)

## Agent cognition & architecture

Experiments that vary the agent's internal machinery rather than the
environment or selection regime.

### Memory Agent

A biologically-inspired three-tier memory system (short-term,
intermediate, long-term) with progressive compression. The experiment
asks how hierarchical memory compression impacts agent learning,
adaptation, and decision quality.

- **Status:** Design and analysis notes; ongoing research.
- **Docs:**
  - [Overview](experiments/memory_agent/README.md)
  - [Memory model](experiments/memory_agent/Memory.md)
  - [Design and considerations](experiments/memory_agent/DesignConsiderations.md)
  - [Implementation](experiments/memory_agent/Implementation.md)
  - [Detailed walkthrough](experiments/memory_agent/Detail.md)
  - [Advanced topics](experiments/memory_agent/Advanced.md)

## Emergent behavior & dominance

Experiments that probe how population-level outcomes (dominance,
cooperation, competition) emerge from agent and environment design.

### One of a Kind

A 500-iteration case study of dominance dynamics between System,
Independent, and Control agents. Investigates how initial conditions —
particularly spatial position relative to resources — determine which
agent type ultimately thrives.

- **Status:** Case study with published findings; data and analysis
  artifacts available in the docs tree.
- **Docs:**
  - [Findings](experiments/one_of_a_kind/Findings.md)
  - [Dominance measures](experiments/one_of_a_kind/Measures.md)
  - [Initial positioning](experiments/one_of_a_kind/InitialPositioning.md)
  - [Positioning metrics](experiments/one_of_a_kind/PositioningMetrics.md)
  - [Competition](experiments/one_of_a_kind/Competition.md)
  - [Cooperation](experiments/one_of_a_kind/Cooperation.md)
  - [Reproduction](experiments/one_of_a_kind/Reproduction.md)
  - [Dominance dynamics](experiments/one_of_a_kind/Dominance.md)

### Rabbit's Foot

Introduces a singular, non-consumable artifact that grants its holder a
persistent RNG advantage. Agents can hold, trade, or steal it via
combat. The experiment asks whether agents learn to retain the artifact
and under what conditions hoarding, trading, or ignoring it becomes a
dominant strategy.

- **Status:** Design proposal; mechanics, configuration, and metrics
  specified ahead of implementation.
- **Docs:** [Design](experiments/rabbits_foot/Design.md)

## Adding a new experiment

When introducing a new experiment, follow the structure used by the
existing entries:

1. **Create a runner** under `farm/runners/` (or extend an existing
   one). Mirror the patterns in
   [`intrinsic_evolution_experiment.py`](../farm/runners/intrinsic_evolution_experiment.py)
   or [`evolution_experiment.py`](../farm/runners/evolution_experiment.py).
2. **Add a CLI driver** under `scripts/run_<experiment>.py` so the
   experiment is reproducible from the command line.
3. **Write documentation** under `docs/research/experiments/<experiment_name>/`
   (or a single `docs/research/experiments/<experiment_name>.md` for smaller
   experiments). Include motivation, configuration reference, output
   artifacts, and at least one reproducible quick-start command.
4. **Link the experiment from this page** under the appropriate
   category, with a one-paragraph summary, status, runner / CLI
   references, and a link to the detailed documentation.
5. **Add tests** under `tests/runners/` and/or `tests/analysis/` for any
   new runner, analysis function, or artifact schema.
