# The Rabbit's Foot Experiment

## Overview

This experiment introduces a **unique, non-consumable artifact** — the *Rabbit's Foot* — into the AgentFarm resource environment. Unlike standard resources that are gathered and consumed, the Rabbit's Foot is a singular object that can be held by exactly one agent at a time. The holder receives a persistent **RNG advantage** that biases stochastic outcomes in their favor (combat rolls, gather yields, reproduction success). Agents can trade the Rabbit's Foot to others via a new `TRADE` action, steal it through combat, or lose it on death.

The central research question: **Will agents learn to retain the Rabbit's Foot, and under what conditions does hoarding, trading, or ignoring it emerge as a dominant strategy?**

---

## Table of Contents

1. [Motivation](#motivation)
2. [Core Mechanics](#core-mechanics)
3. [Agent Interactions with the Rabbit's Foot](#agent-interactions-with-the-rabbits-foot)
4. [RNG Advantage Model](#rng-advantage-model)
5. [Implementation Plan](#implementation-plan)
6. [Configuration](#configuration)
7. [Metrics and Analysis](#metrics-and-analysis)
8. [Experiment Plan](#experiment-plan)
9. [Hypotheses](#hypotheses)
10. [Risk and Mitigation](#risk-and-mitigation)

---

## Motivation

Standard AgentFarm experiments study emergent behavior around **fungible resources** — food that is gathered, consumed, shared, and regenerated. These resources are interchangeable and their value is linear. The Rabbit's Foot introduces a fundamentally different economic object:

- **Singular**: Only one exists in the world at any time.
- **Non-consumable**: It is never depleted; holding it is costless.
- **Transferable**: It can change hands through trade or combat.
- **Asymmetric**: Its value is entirely positional — it benefits only the current holder.

This creates a game-theoretic landscape where agents must weigh the private benefit of holding the artifact against the social cost of being a target, and where trading it away can be either altruistic or strategic (e.g., trading it for resources when starving).

The experiment connects to broader questions about **property, accumulation, and strategic valuation of unique goods** in multi-agent systems.

---

## Core Mechanics

### The Artifact

| Property | Value |
|----------|-------|
| Name | Rabbit's Foot |
| Count | Exactly 1 per simulation |
| Consumable | No |
| Droppable on death | Yes — drops to the ground at the holder's position |
| Visible | Yes — nearby agents can observe whether a neighbor holds it |
| Initial placement | Random position on the grid (not held by any agent) |

The Rabbit's Foot exists as a distinct entity tracked by the environment. It has three possible states:

1. **On the ground** at a grid position (can be picked up by any adjacent agent).
2. **Held by an agent** (provides the RNG advantage).
3. **In transit** during a `TRADE` action (instantaneous transfer to a neighbor).

### Picking Up the Rabbit's Foot

When the Rabbit's Foot is on the ground, any agent within `pickup_range` (default: gathering range from config) can pick it up as part of a `GATHER` action. If multiple agents attempt to gather from the same position in the same step, the agent with the **highest current resource level** wins the tie (a proxy for strength/priority). Once picked up, the Rabbit's Foot is bound to the holder until traded, stolen, or the holder dies.

### Dropping the Rabbit's Foot

The Rabbit's Foot returns to the ground when:

- The **holder dies** (starvation, combat death) — it drops at the death position.
- The holder executes a **`TRADE` action** targeting a neighbor — it transfers directly (never hits the ground in a successful trade).

---

## Agent Interactions with the Rabbit's Foot

### New Action: `TRADE`

A new action type is added to the `ActionType` enum:

```python
class ActionType(IntEnum):
    DEFEND = 0
    ATTACK = 1
    GATHER = 2
    SHARE = 3
    MOVE = 4
    REPRODUCE = 5
    PASS = 6
    COMMUNICATE = 7
    TRADE = 8  # new
```

**`TRADE` semantics:**

| Aspect | Detail |
|--------|--------|
| Precondition | The acting agent holds the Rabbit's Foot AND a valid trade target is within `trade_range` |
| Target selection | Nearest agent within range, or the agent that most recently sent a `RESOURCE_OFFER` message (via `CommunicationComponent`) |
| Effect | The Rabbit's Foot transfers from holder to target. Optionally, the target transfers `trade_resource_cost` resources back to the holder (configurable, can be 0 for a gift). |
| Cost | Small action cost (same as `SHARE`) |
| Failure | If no valid target is in range, the action becomes a `PASS` |

The `TRADE` action enables agents to **voluntarily give up** the Rabbit's Foot. Whether agents learn to do this — and under what circumstances — is a primary experimental outcome.

### Combat and Theft

When an agent successfully attacks and kills a Rabbit's Foot holder, the attacker **automatically acquires** the Rabbit's Foot. This makes holders targets and creates a risk/reward tension:

- Holding the foot improves your RNG → you win more fights → positive feedback loop.
- But being known to hold it makes you a priority target → negative feedback from increased aggression.

If the attacker does not kill the defender (the defender survives), the Rabbit's Foot stays with the original holder. There is no "pickpocket" mechanic — theft requires lethal combat.

### Observation

The observation space is extended so that agents can perceive:

- Whether **they** hold the Rabbit's Foot (binary feature in self-observation).
- Whether any **visible neighbor** holds it (binary feature per neighbor in the spatial observation).
- The **ground position** of the Rabbit's Foot if it is within perception range and on the ground.

This information allows learning agents to make informed decisions about pursuing, trading, or avoiding the artifact.

---

## RNG Advantage Model

The holder of the Rabbit's Foot receives a **luck multiplier** applied to stochastic outcomes. The advantage is parameterized by a single float `luck_bonus` (default: `0.15`, representing a 15% relative improvement).

| Outcome | Baseline | With Rabbit's Foot |
|---------|----------|---------------------|
| Gather yield | `amount * efficiency` | `amount * efficiency * (1 + luck_bonus)` |
| Combat attack roll | `base_damage * random(0.8, 1.2)` | `base_damage * random(0.8, 1.2) * (1 + luck_bonus)` |
| Combat defense | `defense_value` | `defense_value * (1 + luck_bonus)` |
| Reproduction success | `base_probability` | `base_probability * (1 + luck_bonus)` |
| Share efficiency | Unchanged | Unchanged (no luck bonus on sharing) |

The advantage is deliberately **not** applied to sharing, so that cooperation is not inflated by holding the artifact. The bonus affects competitive and self-interested actions, making the artifact a tool for individual advantage.

### Tuning the Advantage

The `luck_bonus` parameter is a primary experimental variable. Sweeping it across values (e.g., 0.0, 0.05, 0.10, 0.15, 0.25, 0.40) lets us study how the **magnitude of the advantage** affects holding behavior:

- At `luck_bonus = 0.0`, the Rabbit's Foot is inert — a control condition.
- At low values (0.05–0.10), the advantage is subtle; agents may not learn to value it.
- At moderate values (0.15–0.25), the advantage is meaningful; strategic behavior should emerge.
- At high values (0.40+), the advantage is dominant; the holder becomes extremely powerful, potentially creating runaway dynamics.

---

## Implementation Plan

### Phase 1: Rabbit's Foot Entity and Environment Integration

**Files to modify or create:**

| File | Change |
|------|--------|
| `farm/core/rabbits_foot.py` (new) | `RabbitsFoot` class — tracks position, holder, state transitions, pickup/drop logic |
| `farm/core/environment.py` | Register the Rabbit's Foot in the environment; handle ground placement, pickup during gather, drop on agent death; extend `step()` |
| `farm/core/state.py` | Add `has_rabbits_foot: bool` to `AgentState`; add `rabbits_foot_position` / `rabbits_foot_holder_id` to `EnvironmentState` |

**`RabbitsFoot` class sketch:**

```python
class RabbitsFoot:
    def __init__(self, position: tuple[int, int]):
        self.position = position
        self.holder_id: Optional[str] = None

    @property
    def is_held(self) -> bool:
        return self.holder_id is not None

    def pickup(self, agent_id: str) -> None:
        self.holder_id = agent_id
        self.position = None

    def drop(self, position: tuple[int, int]) -> None:
        self.holder_id = None
        self.position = position

    def trade(self, new_holder_id: str) -> None:
        self.holder_id = new_holder_id
```

### Phase 2: TRADE Action

**Files to modify:**

| File | Change |
|------|--------|
| `farm/core/action.py` | Add `TRADE = 8` to `ActionType`; implement `trade_action()` function; register in `action_registry` |
| `farm/core/agent/behaviors/` | Update `DefaultAgentBehavior` and `LearningAgentBehavior` to include `TRADE` in the action space with configurable weight |

**`trade_action()` logic:**

1. Validate agent holds the Rabbit's Foot.
2. Find nearest agent within `trade_range`.
3. Transfer the Rabbit's Foot to the target.
4. Optionally transfer `trade_resource_cost` resources from target to holder.
5. Log the trade event.
6. Return action result dict.

### Phase 3: Luck Multiplier Integration

**Files to modify:**

| File | Change |
|------|--------|
| `farm/core/action.py` | In `gather_action()`, apply `(1 + luck_bonus)` to yield when agent holds the foot. In `attack_action()`, apply to damage. In `defend_action()`, apply to defense. |
| `farm/core/agent/components/reproduction.py` | Apply `(1 + luck_bonus)` to reproduction probability check |
| `farm/core/agent/core.py` or services | Add `luck_multiplier` property that returns `1.0 + luck_bonus` if holding, else `1.0` |

The luck multiplier is read from the environment's `RabbitsFoot` state, not stored on the agent — this keeps the agent stateless with respect to the artifact and avoids sync issues.

### Phase 4: Observation Space Extension

**Files to modify:**

| File | Change |
|------|--------|
| `farm/core/observations.py` | Add `has_rabbits_foot` (self), `neighbor_has_rabbits_foot` (per visible agent), `rabbits_foot_ground_position` (if visible) to the observation vector |
| `farm/core/decision/` | Update feature engineering to include the new observation features |

### Phase 5: Configuration and YAML

**Files to modify:**

| File | Change |
|------|--------|
| `farm/config/config.py` | Add `RabbitsFootConfig` dataclass with `enabled`, `luck_bonus`, `trade_range`, `trade_resource_cost`, `pickup_range`, `initial_position` |
| `farm/config/default.yaml` | Add `rabbits_foot:` section with defaults |
| `farm/config/templates/rabbits_foot_sweep.yaml` | Template for `luck_bonus` parameter sweep |

### Phase 6: Metrics, Logging, and Analysis

**Files to create or modify:**

| File | Change |
|------|--------|
| `farm/core/environment.py` | Log Rabbit's Foot events to the step database: pickups, drops, trades, theft-via-combat |
| `farm/analysis/rabbits_foot.py` (new) | Analysis module: holding duration distribution, trade frequency, holder survival rates, correlation between holding and dominance |
| `tests/test_rabbits_foot.py` (new) | Unit tests for `RabbitsFoot`, `trade_action`, luck multiplier application, observation features |

---

## Configuration

```yaml
# Added to default.yaml under a new top-level key
rabbits_foot:
  enabled: true
  luck_bonus: 0.15          # relative bonus to RNG outcomes
  trade_range: 30.0          # max distance for TRADE action
  trade_resource_cost: 0.0   # resources transferred from target to holder on trade (0 = gift)
  pickup_range: null          # null = use gather range from resource config
  initial_position: null      # null = random; or [x, y] for fixed start
  drop_on_death: true         # whether the foot drops when holder dies
```

**Agent behavior weights (added to `agent_parameters`):**

```yaml
agent_parameters:
  SystemAgent:
    trade_weight: 0.05       # low propensity to trade away the foot
  IndependentAgent:
    trade_weight: 0.02       # very low — hoarding tendency
  ControlAgent:
    trade_weight: 0.10       # higher willingness to trade
```

---

## Metrics and Analysis

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **Holding Duration** | How many consecutive steps an agent holds the Rabbit's Foot before losing it |
| **Hold Ratio by Agent Type** | Fraction of total simulation steps each agent type holds the artifact |
| **Trade Frequency** | Number of voluntary `TRADE` actions per simulation |
| **Theft Frequency** | Number of times the foot changes hands via combat |
| **Holder Survival Rate** | Survival rate of agents while holding vs. not holding |
| **Holder Resource Accumulation** | Average resource gain rate while holding vs. not holding |
| **Dominance Correlation** | Correlation between holding the foot and final dominance score |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Time to First Pickup** | Steps until the Rabbit's Foot is first picked up from the ground |
| **Ground Time Ratio** | Fraction of simulation where the foot is on the ground (unclaimed) |
| **Trade Network** | Graph of trade relationships — who trades to whom |
| **Pursuit Behavior** | Whether agents move toward the Rabbit's Foot ground position |
| **Target Selection Bias** | Whether agents preferentially attack Rabbit's Foot holders |
| **Learning Convergence** | How quickly agents develop stable policies regarding the foot (for `LearningAgentBehavior`) |

### Visualization Ideas

- **Timeline heatmap**: Horizontal bar per simulation step, colored by which agent (or agent type) holds the foot. Gaps represent ground time.
- **Trade network graph**: Directed graph of all trades with edge weights for frequency.
- **Holding duration distribution**: Histogram of consecutive holding streaks, faceted by agent type.
- **Luck bonus sweep plot**: Holding duration and trade frequency as a function of `luck_bonus`.

---

## Experiment Plan

### Experiment 1: Baseline Characterization

**Goal**: Establish baseline simulation dynamics without the Rabbit's Foot.

| Parameter | Value |
|-----------|-------|
| `rabbits_foot.enabled` | `false` |
| `simulation_steps` | 2000 |
| Repetitions | 100 |

Collect standard metrics (dominance, cooperation, resource levels) for comparison.

### Experiment 2: Default Rabbit's Foot Introduction

**Goal**: Observe emergent behavior with a moderate luck bonus.

| Parameter | Value |
|-----------|-------|
| `rabbits_foot.enabled` | `true` |
| `rabbits_foot.luck_bonus` | 0.15 |
| `simulation_steps` | 2000 |
| Repetitions | 250 |

**Key questions:**
- Does holding duration increase over time within a simulation (do agents learn to hold)?
- Which agent type holds the foot the longest?
- How does the foot affect dominance outcomes compared to baseline?

### Experiment 3: Luck Bonus Sweep

**Goal**: Map the relationship between advantage magnitude and holding behavior.

| Parameter | Sweep Values |
|-----------|-------------|
| `rabbits_foot.luck_bonus` | 0.0, 0.05, 0.10, 0.15, 0.25, 0.40 |
| `simulation_steps` | 2000 |
| Repetitions per value | 100 |

**Key questions:**
- At what `luck_bonus` threshold do agents begin to exhibit intentional holding behavior?
- Does a high bonus create runaway dynamics where the holder becomes unkillable?
- Is there an optimal bonus where interesting trading dynamics emerge?

### Experiment 4: Trade Cost Sweep

**Goal**: Investigate whether requiring payment for trades changes behavior.

| Parameter | Sweep Values |
|-----------|-------------|
| `rabbits_foot.trade_resource_cost` | 0.0, 1.0, 3.0, 5.0, 10.0 |
| `rabbits_foot.luck_bonus` | 0.15 (fixed) |
| `simulation_steps` | 2000 |
| Repetitions per value | 100 |

**Key questions:**
- Does a nonzero trade cost reduce trade frequency?
- Do agents learn to trade the foot when starving (trading luck for survival resources)?
- Does a high trade cost make the foot effectively non-tradable?

### Experiment 5: Agent Type Composition

**Goal**: Test how population ratios affect Rabbit's Foot dynamics.

| Parameter | Sweep Values |
|-----------|-------------|
| `agent_type_ratios` | Majority System (0.6/0.2/0.2), Majority Independent (0.2/0.6/0.2), Balanced (0.33/0.33/0.33) |
| `rabbits_foot.luck_bonus` | 0.15 (fixed) |
| `simulation_steps` | 2000 |
| Repetitions per config | 100 |

**Key questions:**
- Does the most aggressive agent type (Independent) dominate foot possession?
- Does the most cooperative type (System) trade the foot away too readily?
- Does agent type composition change whether holding or trading is the dominant strategy?

### Experiment 6: Learning Agent Focus

**Goal**: Test whether RL-trained agents learn to value and retain the Rabbit's Foot.

| Parameter | Value |
|-----------|-------|
| Agent behavior | `LearningAgentBehavior` for all agents |
| `rabbits_foot.luck_bonus` | 0.15 |
| `simulation_steps` | 5000 (longer to allow learning) |
| Repetitions | 50 |

**Key questions:**
- Do learning agents develop a preference for holding the foot over time?
- Does the learned policy converge to always-hold, always-trade, or conditional behavior?
- How many steps does it take for holding behavior to emerge?

---

## Hypotheses

1. **Holding will emerge**: At `luck_bonus >= 0.10`, agents (especially learning agents) will develop longer average holding durations over the course of a simulation, indicating they learn the foot is valuable.

2. **Independent agents will hold longest**: Due to their lower cooperation propensity and higher aggression, Independent agents will accumulate more holding time than System or Control agents.

3. **High luck bonus creates instability**: At `luck_bonus >= 0.40`, the holder becomes so powerful that only combat death breaks their hold, reducing dominance switching and creating stagnant simulations.

4. **Trade cost creates desperation trading**: When `trade_resource_cost > 0`, agents will primarily trade the foot when their resources are critically low, exchanging long-term advantage for immediate survival.

5. **System agents over-trade**: System agents' high sharing propensity will extend to the foot, causing them to trade it away more frequently than other types even when it would be beneficial to hold.

6. **The foot amplifies existing dominance patterns**: The agent type that is already dominant in baseline simulations will benefit disproportionately from the foot, as the RNG advantage compounds their existing strengths.

7. **Ground time decreases over simulation**: As agents learn (either through RL or through selection pressure from reproduction), the foot will spend less time on the ground in later simulation steps.

---

## Risk and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Luck bonus too weak to matter | Agents ignore the foot; experiment produces null results | Run Experiment 3 (sweep) early to identify the threshold |
| Luck bonus too strong; holder never dies | Stagnant simulations with one immortal agent | Cap the bonus at 0.40; add a "jealousy" mechanic where nearby agents get an attack bonus against holders |
| TRADE action never chosen by default behaviors | The foot only changes hands via combat, reducing behavioral variety | Tune `trade_weight` in agent parameters; ensure learning agents have TRADE in their action space |
| Observation space changes break existing models | RL training diverges or crashes | Gate new features behind `rabbits_foot.enabled`; add features as optional dimensions |
| Single artifact creates too much randomness | High variance between simulations depending on who picks it up first | Use 250+ repetitions per experiment; analyze early-pickup bias as a secondary metric |
| Performance overhead from tracking the artifact | Simulation slows down | The artifact is a single object with O(1) state; overhead should be negligible |

---

## Summary

The Rabbit's Foot experiment extends AgentFarm with a **unique tradable artifact** that grants an RNG advantage, creating a natural laboratory for studying **possession, valuation, and strategic trade** in multi-agent systems. The implementation touches the action system, observation space, environment state, and analysis pipeline, but is designed to be **feature-gated** behind a config flag so existing experiments are unaffected.

The six-experiment plan systematically varies the key parameters (luck bonus, trade cost, population mix, agent intelligence) to build a complete picture of how agents interact with a scarce, non-consumable resource — and whether they learn that holding on to luck is the smartest move.
