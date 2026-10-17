# Cooperative Units

`farm.core.cooperation` models cooperative institutions as one configurable
object. A **cooperative unit** forms when agents pool part of their income;
the pool creates a surplus none of them could produce alone, and the unit
develops goals of its own (a reserve, its own survival, dues to a parent
unit). A marriage, a friendship, a business partnership, a family and a tribe
are the same object with different settings.

The package is engine-agnostic: participants are string ids, and the caller
supplies incomes and holdings. It does not depend on the main
`Environment`. The [cooperation experiment](../research/experiments/cooperation/Design.md)
is the first consumer.

## The seven settings

`CooperationSettings` holds the settings; `CooperationRegistry` enforces them.

| Setting | Field | Rule |
| --- | --- | --- |
| Members | `min_members`, `max_members` | Capacity on joining; below the minimum the unit dissolves and splits its pool among the remaining members |
| Scope | `scope` | Fraction of each member's income pooled each step |
| Exit cost | `exit_cost` | Fraction of a leaver's holdings forfeited to the unit, when the exit is enforced |
| Recognition | `recognition` | Probability that outsiders enforce the rules: the exit cost is collected, an outside offer to a member of an exclusive unit is blocked, a parent's veto holds |
| Exclusivity | `exclusive` | One unit of the kind per agent, or up to `nonexclusive_limit` |
| Entry | `entry` | `CHOSEN` (both sides accept), `ARRANGED` (a parent unit picks the match), `BORN` (children join at birth) |
| Symmetry | `symmetry` | Payout shares blend an equal split (1) with a rank split (0, ratio 0.25 between ranks) |

`PRESETS` positions six institutions on the settings: `marriage`,
`cohabitation`, `friendship`, `business_partnership`, `family`, `tribe`. The
values are illustrative calibrations, not estimates.

## One step of pooling

```python
pooled = registry.contribute(agent_id, income, step)   # each member, each step
payouts = registry.settle(step, holdings=energy_by_id)  # once per step
```

`settle` runs leaves-first, then roots-first:

1. Each unit creates a surplus `synergy * inflow * (1 - 1/n)` (none for a
   single participant).
2. A unit with a parent sends `parent.scope * (inflow + surplus)` up as dues.
3. Each unit pays emergency aid (up to `aid_amount`) to agent members whose
   holdings are below `aid_threshold`.
4. Each unit pays out `payout_rate` of its pool above the reserve target
   (`UnitGoals.reserve_target` per participant), split by rank weights; a
   child unit receives its share into its own pool before paying its members.

Pool parameters live in `UnitEconomics`; they describe the environment, not
the institution.

## Members, goals and value

Agents carry a mixture over five motives (`Motive`): political, economic,
emotional, hedonist, religious. Each motive prefers certain settings
(`MOTIVE_SETTING_PREFERENCES`) and weights four value terms: wealth (expected
net transfer), security (support on the agent's own bad-luck steps), affinity
(compatibility with partners) and belonging (what the membership feeds into a
parent unit). `value_of_membership` turns a `MembershipAppraisal` into one
number in resource units:

```
V = H * (w_wealth * net + w_security * insurance + w_belonging * parent_gain)
    + k_fit * fit(settings) + k_aff * w_affinity * compatibility
```

The registry keeps a `MemberLedger` per participant, including a running
estimate of realised net benefit (optionally seeded with the expected value
at entry), so a consumer can let members judge a unit by what it actually
did for them. A typical decision rule: join when `V` clears a threshold; leave
when `V < -settings.expected_exit_payment(holdings)`; otherwise, if `V < 0`,
the member is trapped and the consumer records it with
`registry.record_exit_desire(..., blocked_by_price=True)`.

## Nesting, births and vetoes

Units can be participants of other units (`form(..., children=[...])`,
`attach`). Dues flow up and payouts flow down each step. `register_birth`
enrols a newborn in the BORN-entry unit of its parents' household (the
nearest BORN ancestor of a parent's chosen or arranged unit), falling back to
the parent's own BORN units. A parent vetoes an exit that would dissolve a
child unit with probability equal to the parent's recognition.

## Failure modes

`diagnose_unit` labels a unit from its ledgers and per-step history:

| Mode | Test (default thresholds) |
| --- | --- |
| Fragility | Members' exits dissolved it within 30 steps of forming |
| Override | A parent vetoed an exit, or an arranged member did not consent |
| Capture | A member's payout share exceeds its contribution share by 0.25 and is at least 0.6 (aid excluded) |
| Trap | A member spent 15+ steps wanting out but priced in |
| Drift | Over the last 50 steps members net-lose while the unit or its parent retains resources |
| Hollow | Less than 5% of members' income is pooled |
| Healthy | None of the above, positive surplus, no member losing more than 0.02 per step |

Units younger than 20 steps are `indeterminate`; units that fire nothing but
are not healthy are `unclassified`. `mode_prevalence` aggregates diagnoses.

## Minimal example

```python
import numpy as np
from farm.core.cooperation import CooperationRegistry, diagnose_unit, preset

registry = CooperationRegistry(rng=np.random.default_rng(0))
couple = registry.form(preset("marriage"), ["a", "b"], step=0)
family = registry.form(preset("family"), [], step=0, children=[couple.unit_id])

for step in range(1, 50):
    registry.contribute("a", 2.0, step)
    registry.contribute("b", 0.5, step)
    payouts = registry.settle(step, holdings={"a": 10.0, "b": 1.0})

registry.register_birth("a", "c", step=50)            # c joins the family
result = registry.leave(couple.unit_id, "a", step=51, holdings=10.0)  # may be vetoed by the family
print(diagnose_unit(couple, 52).primary)
```
