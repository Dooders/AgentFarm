# Accuracy Audit: "Open-Ended Genome Evolution" Post vs. AgentFarm Codebase

This document audits a marketing/devlog post titled **"Open-Ended Genome Evolution
in Foraging and Learning Agents"** against what is actually implemented in the
AgentFarm codebase. It is intended as a reference for keeping public-facing
descriptions of the project honest and concrete.

## Source post (verbatim)

> **Open-Ended Genome Evolution in Foraging and Learning Agents**
>
> In artificial life research, one of the longstanding challenges is creating
> systems capable of truly open-ended evolution — where digital organisms
> continue to generate novelty, complexity, and new behaviors indefinitely,
> rather than converging on fixed solutions. This experiment explores that idea
> through a simple yet powerful mechanism: evolving genomes that directly shape
> agent behavior in a resource-limited world.
>
> Each agent has its own chromosomes, inherited from its parents. Through
> reproduction and selection, the best and most fit genetic code prevails over
> generations.
>
> The genome represents the hyperparameters of the agent, encoded as genes and
> chromosomes. These evolve in value within an open-ended, resource-based
> simulation where agents must forage, reproduce, and learn.
>
> The goal is to observe how complex behaviors and increasingly effective
> genetic encodings can emerge purely from inheritance, mutation, selection
> pressure, and individual learning — without hand-crafted objectives or
> predefined optima.
>
> Key features:
> - Individual chromosomes per agent, passed from parents with variation
> - Genome as evolvable hyperparameters controlling behavior and learning
> - Resource competition that drives natural selection
> - Agents that actively forage for limited resources
> - Reproduction with genetic inheritance
> - Lifetime learning integrated into the agent lifecycle
>
> This is an early-stage personal experiment at the intersection of artificial
> life, evolutionary computation, and agent-based modeling. I'm building it to
> better understand how evolution can sculpt not just parameters, but entire
> adaptive strategies in agents that must survive and thrive in a dynamic,
> competitive environment.

## Verdict at a glance

| Claim | Verdict |
|---|---|
| Open-ended artificial life framing | Overstated |
| Individual chromosomes per agent, inherited with variation | Accurate |
| Genome as evolvable hyperparameters controlling behavior and learning | Partially accurate (narrow gene set) |
| Resource competition driving natural selection | Accurate (emergent, not explicit) |
| Agents actively forage for limited resources | Accurate |
| Reproduction with genetic inheritance | Accurate (with caveat) |
| Lifetime learning integrated into the lifecycle | Accurate, but combination is Baldwinian, not Lamarckian |

## Detailed findings

### 1. "Open-ended evolution / artificial life" framing — Overstated

The repo is primarily an **agent-based modeling + RL + analysis** platform, not
an ALife/open-ended-evolution system. There are no occurrences of "artificial
life", "open-ended evolution", or "alife" in tracked files. The README lists
"Evolutionary algorithms and genetic modeling" as one bullet alongside ABM,
analysis, RL, and visualization. `farm/runners/intrinsic_evolution_experiment.py`
describes itself as "a single long simulation" where selection emerges from the
resource environment — bounded, not open-ended.

### 2. "Individual chromosomes per agent, passed from parents with variation" — Accurate

Each agent gets its own `HyperparameterChromosome` at construction in
`farm/core/agent/core.py` (`AgentCore.__init__` calls
`chromosome_from_learning_config`). On `reproduce()`,
`_derive_child_chromosome` either deep-copies the parent or applies
`crossover_chromosomes(...)` followed by `mutate_chromosome(...)` when
`intrinsic_evolution_policy.enabled` is true. So inheritance + variation is
real.

**Caveat:** there is also a `farm/core/genome.py` `Genome` class (action
weights + module state with its own `mutate`/`crossover`), but
`AgentCore.reproduce()` does **not** use it — only the hyperparameter
chromosome flows to offspring. The post's "chromosomes" really means the
hyperparameter chromosome.

### 3. "Genome as evolvable hyperparameters controlling behavior and learning" — Partially accurate

`DEFAULT_HYPERPARAMETER_GENES` in `farm/core/hyperparameter_chromosome.py`
defines four genes, but only three are evolvable: `learning_rate`, `gamma`,
`epsilon_decay`. `memory_size` is explicitly `evolvable=False`. These do flow
into the offspring's learning config via
`apply_chromosome_to_learning_config(...)` before
`AgentFactory.create_learning_agent(...)` builds the child's `DecisionModule`.

So the claim is true in spirit, but the genome is currently a **narrow
RL-hyperparameter genome** (3 evolvable loci), not a broad "behavior and
learning" genome (no action weights, network shape, sensors, etc., evolve via
this path).

### 4. "Resource competition that drives natural selection" — Accurate (emergent)

- Resources are finite, depletable nodes (`farm/core/resources.py`
  `Resource.consume`).
- Agents pay `base_consumption_rate` per step and are terminated after
  `starvation_threshold` zero-resource steps
  (`farm/core/agent/components/resource.py` `_check_starvation`).
- Reproduction is gated on resource cost in
  `ReproductionComponent.can_reproduce()` and pays `offspring_cost`.

There is no explicit fitness-proportionate selection operator inside the world
loop — selection is **ecologically emergent** (lineages that don't pay the
energy budget die out). That matches the post's wording.

### 5. "Agents actively forage" — Accurate

`gather_action` in `farm/core/action.py` finds the nearest resource node within
`gathering_range` via the spatial service, calls
`env.consume_resource(...)`, and credits the agent.

### 6. "Reproduction with genetic inheritance" — Accurate (with caveat)

`AgentCore.reproduce()` builds the child with a derived chromosome and a fresh
learning agent via `AgentFactory.create_learning_agent`, sets
`generation = parent.generation + 1`, and records `parent_ids`. Crossover
supports `SINGLE_POINT`, `UNIFORM`, `BLEND`, and `MULTI_POINT` modes
(`CrossoverMode` in `farm/core/hyperparameter_chromosome.py`). When no
co-parent is selected, reproduction falls back to **asexual** (mutation-only)
inheritance — see `_select_coparent`'s docstring. The post's plural "parents"
is therefore conditionally true; default behavior is single-parent + mutation
unless crossover is enabled and a co-parent is found.

### 7. "Lifetime learning integrated into the agent lifecycle" — Accurate, with nuance

`AgentFactory.create_learning_agent` instantiates a `DecisionModule` +
`LearningAgentBehavior` so agents do RL during life. However, when offspring
are created in `reproduce()`, a **fresh** `DecisionModule` is built — parent
Q-values/weights are **not** copied. Combined with #3, the system is
**Baldwinian** (evolved learning priors shape how the child learns), not
**Lamarckian** (acquired knowledge is not inherited). The post's wording is
fine since it only asserts integration, not Lamarckian transfer.

## Suggested edits to the post for accuracy

- Soften "open-ended evolution" → "evolutionary dynamics within bounded
  simulations" (or implement an outer loop / no-termination mode that justifies
  the original wording).
- Replace "Genome as evolvable hyperparameters controlling behavior and
  learning" with something like "Evolvable RL hyperparameters (learning rate,
  discount factor, exploration decay) carried per agent." Or expand the gene
  set to include action-weight or module-state loci by wiring
  `farm/core/genome.py` into `reproduce()`.
- Clarify reproduction: single-parent with mutation by default; crossover with
  a co-parent when `intrinsic_evolution_policy.crossover_enabled` is on.
- Optionally call out that learning is Baldwinian — evolved priors, not
  inherited weights — which is a feature, not a bug, but distinguishes it from
  "evolving entire adaptive strategies" (the latter implies inherited
  policy/weights too).

## Key files referenced

- `farm/core/agent/core.py`
- `farm/core/hyperparameter_chromosome.py`
- `farm/core/genome.py`
- `farm/core/agent/factory.py`
- `farm/core/agent/components/resource.py`
- `farm/core/agent/components/reproduction.py`
- `farm/core/resources.py`
- `farm/core/action.py`
- `farm/runners/intrinsic_evolution_experiment.py`
- `docs/design/hyperparameter_chromosome.md`
