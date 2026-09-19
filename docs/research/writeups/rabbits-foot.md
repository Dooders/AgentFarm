---
layout: experiment
title: Rabbit's Foot
subtitle: A singular, non-consumable artifact with an RNG advantage
status: design
updated: 2026-01-01
category: Emergent behavior
excerpt: >-
  Exactly one Rabbit's Foot exists. The holder gets a persistent luck bias;
  others can trade for it, steal it, or pick it up off a corpse. This writeup
  is the whole planned experiment: six variant runs specified ahead of
  implementation.
variants:
  - id: baseline
    title: Baseline without the foot
  - id: default
    title: Default introduction
    note: luck_bonus = 0.15
  - id: luck-sweep
    title: Luck-bonus sweep
  - id: trade-cost
    title: Trade-cost sweep
  - id: composition
    title: Agent-type composition
  - id: learning
    title: Learning-agent focus
---

Standard AgentFarm experiments study fungible food. The Rabbit's Foot is a
different economic object: singular, non-consumable, transferable, and
asymmetric — it only helps the current holder. The question for the whole
experiment is whether agents learn to retain it, and under what conditions
hoarding, trading, or ignoring it becomes the dominant strategy.

This page is a **design** writeup. Mechanics, configuration, metrics, and
the six variant runs are specified; they have not been run here. Protocol:
[Design](../experiments/rabbits_foot/Design.md).

## Shared mechanics

| Property | Value |
|---|---|
| Count | Exactly 1 |
| Consumable | No |
| Drop on death | Yes, at the holder's cell |
| Visible | Nearby agents can see who holds it |
| Initial placement | Random cell, not held |

States: on the ground, held, or in transit during `TRADE`. Pickup uses the
ordinary gather range; ties go to the agent with more resources. The holder
gets a persistent RNG bias on combat, gather yields, and reproduction.
Feature-gated behind `rabbits_foot.enabled` so other experiments stay
untouched.

## Baseline without the foot {#baseline}

`rabbits_foot.enabled = false`, 2000 steps, 100 repetitions. Ordinary
dominance / cooperation / resource metrics. This is the comparison world:
without it, you cannot say the foot changed anything.

## Default introduction {#default}

`luck_bonus = 0.15`, 2000 steps, 250 repetitions. Moderate advantage, large
enough n to absorb first-pickup luck.

Asks: does holding duration increase within a run (learning to hold)? Which
type holds longest? Does dominance shift relative to baseline?

## Luck-bonus sweep {#luck-sweep}

`luck_bonus ∈ {0.0, 0.05, 0.10, 0.15, 0.25, 0.40}`, 2000 steps, 100
repetitions per value.

Maps the threshold where intentional holding appears, whether a high bonus
makes the holder unkillable, and whether there is an intermediate bonus
where trading stays interesting. Pre-registered hunches: holding emerges at
≥ 0.10; ≥ 0.40 creates stagnant, near-immortal holders.

## Trade-cost sweep {#trade-cost}

`trade_resource_cost ∈ {0.0, 1.0, 3.0, 5.0, 10.0}` at `luck_bonus = 0.15`,
100 repetitions each.

Asks whether a price on `TRADE` cuts frequency, produces desperation trades
(luck for food when starving), or makes the foot effectively non-tradable.

## Agent-type composition {#composition}

Population ratios — majority System, majority Independent, balanced — at
the default bonus, 100 repetitions each.

Tests whether Independents dominate possession, whether Systems over-trade
the foot the way they share food, and whether the holding-vs-trading
equilibrium is a function of who is in the room.

## Learning-agent focus {#learning}

All agents on `LearningAgentBehavior`, `luck_bonus = 0.15`, 5000 steps, 50
repetitions. Longer horizon so a policy can form.

Asks whether RL agents converge to always-hold, always-trade, or
conditional behavior, and on what timescale holding appears.

## Synthesis (pre-registered, not yet results)

The planned reading is: holding should emerge once luck is large enough;
Independents should hold longest; Systems should over-trade; high luck
should freeze dominance; priced trades should be desperation trades; the
foot should amplify whoever already wins the baseline. None of those
sentences is a finding until the variant runs exist. The self-contained
value of this page is that the six runs, and the claims they are allowed to
support, live in one place.

## Reproduce and artifacts

Not implemented as a runner yet. Spec:
[Design](../experiments/rabbits_foot/Design.md).
[Catalog entry](../experiments-catalog.md).
