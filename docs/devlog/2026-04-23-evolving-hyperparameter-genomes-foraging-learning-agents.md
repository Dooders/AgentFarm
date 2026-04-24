---
layout: page
title: "Evolving Hyperparameter Genomes in Foraging and Learning Agents"
---

# Evolving Hyperparameter Genomes in Foraging and Learning Agents

A recurring question in evolutionary computation and agent-based modeling is
how much adaptive behavior can emerge from ecology alone - finite resources,
costly reproduction, and inherited learning priors - without hand-crafted
fitness functions or predefined optima. This experiment is a small step in
that direction: each agent carries its own hyperparameter chromosome, offspring
inherit it (with optional mutation and crossover), and selection is whatever
the resource environment happens to apply.

It is not a claim of open-ended evolution. It is a bounded simulation in which
evolutionary dynamics are layered on top of reinforcement learning agents that
have to feed themselves to stay alive.

## What's actually evolving

Each agent owns a `HyperparameterChromosome` built from its decision config at
construction time. Today the evolvable loci are the core RL hyperparameters:

- `learning_rate`
- `gamma` (discount factor)
- `epsilon_decay` (exploration schedule)

`memory_size` is represented as a gene but is held fixed for now, pending
proper integer-rounding handling alongside the continuous-gene operators.

These genes don't directly encode behavior; they shape **how the agent learns**
during its lifetime. The learned policy itself is not part of the genome.

## Inheritance, mutation, and crossover

When an agent reproduces, the child's chromosome is derived from the parent's:

- **Default path:** asexual - deep-copy the parent chromosome and apply
  per-gene mutation.
- **Crossover enabled:** sexual - pick a co-parent, combine the two
  chromosomes (`SINGLE_POINT`, `UNIFORM`, `BLEND`, or `MULTI_POINT`), then
  mutate.

The resulting chromosome is written back into the child's decision config
before its `DecisionModule` is constructed, so the offspring starts life with
the inherited learning priors already baked in.

## Selection through ecology, not a fitness function

There is no explicit selection operator in the simulation loop. Selection is
emergent and ecological:

- **Finite resources.** Resource nodes deplete as agents gather from them.
- **Metabolic cost.** Every step debits a base consumption rate from the
  agent's resource pool.
- **Starvation.** After a configurable number of zero-resource steps, the
  agent terminates.
- **Costly reproduction.** Reproduction requires meeting an offspring-cost
  threshold, which is paid out of the parent's resources.

Lineages whose hyperparameters happen to produce agents that forage well
enough to cover both metabolism and reproduction persist. Lineages that don't,
die out. That's the whole selection story.

## Foraging

Agents have an explicit `gather` action: locate the nearest resource node
within range via the spatial index, consume from it, and credit the agent's
resource pool. Foraging is one of the actions the RL policy chooses among, so
the evolved learning hyperparameters and the foraging behavior are coupled
through the decision module.

## Learning during life - Baldwinian, not Lamarckian

Within an agent's lifetime, a `DecisionModule` driven by RL updates from
experience. At reproduction time, however, only the **hyperparameter
chromosome** is passed on; the offspring builds a fresh decision module. No
Q-values, weights, or replay buffers cross the generational boundary.

That makes the system Baldwinian: evolution tunes the *priors* and *learning
dynamics* (how fast to learn, how much to discount the future, how aggressively
to explore), and each generation has to acquire its own experience inside
those priors.

## What this experiment is and isn't

**It is:**

- A per-agent hyperparameter genome with mutation and optional crossover.
- An ecological selection regime driven by resource scarcity and
  reproduction costs.
- RL agents whose learning hyperparameters are themselves under selection.

**It isn't (yet):**

- Open-ended evolution in the ALife sense - runs are bounded simulations.
- A genome that encodes full behavior or network architecture - only a
  narrow set of RL hyperparameters evolve through this path.
- Lamarckian inheritance - learned policies are not transmitted to offspring.

## Where it might go next

A few natural extensions, roughly in order of how invasive they are:

- Make `memory_size` (and other integer-valued knobs) properly evolvable.
- Wire the existing `Genome` representation - which already serializes action
  weights and module state - into the reproduction path, so action
  preferences and module parameters can also be inherited and mutated.
- Optionally support warm-starting offspring from parent weights (Lamarckian
  inheritance) as a configurable policy.
- Run longer, less-bounded experiments to see how much novelty the ecology
  can sustain before populations collapse or stagnate.

This is an early-stage personal experiment at the intersection of
agent-based modeling, evolutionary computation, and reinforcement learning.
The interesting question isn't whether the framing is grand; it's whether
small ecological pressures, applied to the parameters that govern learning,
produce noticeably better foragers over generations than fixed
hyperparameters do.
