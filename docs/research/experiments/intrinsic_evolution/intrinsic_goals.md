# Intrinsic Goals Experiment

The intrinsic-goals experiment lives *inside* the intrinsic-evolution framework
and asks a single, focused question:

> **What happens to a simulation when every agent has a different
> reinforcement-learning goal (loss / reward function)?**

In the baseline AgentFarm world, every agent is trained against the same hand-
written reward: gain resources, keep health up, stay alive, and do *something*
each step. This experiment replaces that single shared objective with a
*per-agent, heritable* objective so that the population contains hoarders,
medics, survivalists, cooperators, aggressors, gatherers, and breeders all at
once — and then lets selection decide which goals persist.

## How goals are represented

Each agent's per-step reward is computed in
`AgentCore._calculate_reward`. That function now reads a set of **Chromosome C**
genes (the `reward_*` loci) defined in
`farm/core/hyperparameter_chromosome.py`:

| Gene | Default | Range | Meaning |
|------|---------|-------|---------|
| `reward_resource_weight` | 0.1 | [0, 2] | value of net resource gain |
| `reward_health_weight` | 0.5 | [0, 2] | value of net health gain |
| `reward_survival_weight` | 0.1 | [0, 1] | per-step bonus for staying alive |
| `reward_death_penalty` | 10.0 | [0, 50] | penalty applied on death |
| `reward_action_bonus` | 0.05 | [0, 1] | bonus for any non-`pass` action |
| `reward_gather_bonus` | 0.0 | [0, 2] | intrinsic bonus for gathering |
| `reward_share_bonus` | 0.0 | [0, 2] | intrinsic bonus for sharing (prosocial) |
| `reward_attack_bonus` | 0.0 | [0, 2] | intrinsic bonus for attacking (aggressive) |
| `reward_reproduce_bonus` | 0.0 | [0, 2] | intrinsic bonus for reproducing (fecund) |

The reward is:

```
reward = resource_delta * reward_resource_weight
       + health_delta   * reward_health_weight
       + (reward_survival_weight if alive else -reward_death_penalty)
       + (reward_action_bonus if action != "pass" else 0)
       + per_action_bonus(action)     # gather/share/attack/reproduce
```

Because these are ordinary genes, they:

- are **heritable** — offspring inherit the parent's goal (with optional
  crossover from a co-parent),
- **mutate** on reproduction like any other gene,
- are **selected** implicitly — a goal that helps its carrier survive and
  reproduce spreads; a self-defeating goal dies out, and
- are **logged** automatically by `GeneTrajectoryLogger` and the speciation
  tooling, since they are evolvable.

> The default chromosome reproduces the historical reward formula exactly, so
> nothing changes for runs that do not opt into goal diversity.

## The experiment

`farm/runners/intrinsic_goals_experiment.py` runs two arms with **identical
seeds and configuration** so the *only* difference is the agents' objectives:

- **`uniform`** (control) — every agent shares the default reward function.
- **`unique`** (treatment) — every initial agent is given an independently
  sampled reward function (each `reward_*` gene drawn uniformly within its
  bounds). Offspring inherit and mutate their parent's goal.

To isolate the manipulated variable, platform-wide initial diversity is turned
**off** in both arms (so learning hyperparameters and action priors stay at
their defaults); only the goal genes differ.

For each arm the runner records, per step:

- population size, births, deaths;
- the **action mix** (fraction of alive agents whose most recent action was
  move / gather / share / attack / reproduce / defend / pass), derived from
  each agent's `last_action_name`; and
- the population mean of every goal gene (to watch goals drift under
  selection).

It then writes:

- `intrinsic_goals_summary.json` — per-arm summaries plus a `comparison` block
  (population deltas, action-mix deltas, start/end goal diversity); and
- `intrinsic_goals_comparison.png` — population trajectories, mean action mix,
  goal-gene drift, and goal-diversity (std) start-vs-end (when matplotlib is
  available).

## Running it

```bash
source venv/bin/activate
python scripts/run_intrinsic_goals_experiment.py \
    --num-steps 600 --seed 42 \
    --output-dir experiments/intrinsic_goals
```

Useful flags:

- `--num-steps` — simulation length per arm.
- `--selection-pressure` — `none` / `low` / `medium` / `high` (or a float in
  `[0, 1]`); density-dependent reproduction cost. A little pressure makes
  selection on goals matter.
- `--mutation-rate`, `--mutation-scale`, `--boundary-mode` — per-reproduction
  mutation of all genes (including goals).
- `--initial-agent-resource-level`, `--initial-resource-count` — startup
  stability knobs (the default dev config is intentionally boom/bust).

## What to look for

- **Behavioural divergence.** With unique goals you should see a different
  action mix — e.g. more sharing or attacking — because some agents are now
  intrinsically rewarded for those actions regardless of their resource/health
  outcome.
- **Population dynamics.** Heterogeneous goals can change carrying capacity and
  birth/death balance versus the uniform control.
- **Goal selection.** Track `goal_gene_mean_*` and the goal-diversity (std)
  start-vs-end values: under selection, the population mean of each goal gene
  drifts toward objectives that survive, and diversity may collapse (one goal
  wins) or persist (multiple niches coexist — visible in the speciation
  tooling).

## Relationship to the other runners

- `farm/runners/intrinsic_evolution_experiment.py` evolves **all** genes at
  once (learning hyperparameters + action priors + goals). Use it when you want
  the full co-evolutionary picture.
- This runner deliberately freezes everything except the goal genes so the
  effect of *unique objectives* is legible and directly comparable to a
  matched control.
