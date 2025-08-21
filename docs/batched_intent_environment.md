# Batched-Intent Environment with Context Snapshot and PettingZoo Wrapper

## Overview
This document proposes a context-snapshot + batched-intent stepping model that keeps agents self-contained while improving fairness, determinism, and future parallelization. The environment provides a read-only `Context` snapshot each step. Agents observe this snapshot and produce `ActionIntent`(s). The environment resolves conflicts and applies all results atomically, then advances time.

A PettingZoo ParallelEnv wrapper is included for ecosystem compatibility and easier testing/training loops, without forcing a turn-based (AEC) refactor.

## Goals
- Self-contained agents (own networks, memory, state) that operate: observe → think → act.
- Decouple agents from global mutable state via read-only `Context`.
- Simultaneous-step semantics via batched intents to avoid order bias.
- Determinism with seeded tie-breakers; fairness improvements.
- PettingZoo ParallelEnv wrapper for compatibility with tooling and libraries.
- Smooth migration from the current immediate-mutation step loop.

## Non-goals (for this iteration)
- Turn-based AEC implementation.
- Full parametric action space with all parameters exposed (can follow incrementally).
- Distributed execution (the design enables it later).

## Core Concepts
### Context (Read-only Snapshot)
A compact, immutable snapshot of environment state used by agents for decision-making. Typical fields:
- time, width, height
- agent_positions (alive only)
- resource_positions, resource_amounts

The Context is created once per step to guarantee all agents plan from the same world state.

### ActionIntents
Structured, declarative requests from agents, e.g.:
- MoveIntent(x, y)
- GatherIntent(target_resource_id, amount?)
- ShareIntent(target_agent_id, amount)
- AttackIntent(target_agent_id, power?)
- ReproduceIntent()

Agents may emit one or multiple intents per step (initially one for simplicity).

### Step Pipeline (Batched)
1) build_context()
2) collect intents from all alive agents (can run in parallel)
3) resolve conflicts (seeded tie-breakers)
4) apply all effects atomically to environment
5) commit lifecycle changes (births, deaths, add/remove, persistence flush)
6) update resources/metrics/kd-trees; time += 1

This prevents mid-iteration state mutations from influencing later agents.

### Conflict Resolution Policies (Initial)
- Move collisions: deterministic priority (seeded by agent_id + step); losing agents remain in place.
- Gather contention: split fairly (uniform or proportional) and clamp to available.
- Attack: accumulate damage; apply simultaneously; handle deaths in commit phase.
- Share: clamp to available; break cycles by deterministic ordering.
- Reproduce: enqueue offspring; add during commit.

Policies must be deterministic under fixed seeds.

### Dynamic Add/Remove
- Agents may request reproduction or signal death; environment enqueues add/remove.
- Commit phase applies creation/removal and triggers persistence flush safely.

## Agent API (Incremental)
- `plan(context: Context) -> list[ActionIntent]`
  - Default implementation can map the current module-based `decide_action()` to one intent.
- `act_with_context(context: Context)` (optional/temporary)
  - Calls `plan()` then executes locally; used during migration.
- Existing `act()` remains for backward compatibility until all modules are migrated to intents.

Optional future refinements:
- `get_perception(context: Context)` path that avoids direct environment queries.
- `get_observation_tensor()` standardized feature vector for learning.

## Environment API Additions
- `build_context() -> Context`
- `step_batched()` implementing the pipeline above
- `resolve_conflicts(intents) -> resolved_effects`
- `apply_effects(resolved_effects)`
- `commit_lifecycle_changes()` for births, deaths, and persistence flush

A transitional helper `step_with_context()` may call agents’ `act_with_context()` if present, else fallback to `act()`.

Note: A prototype of `Context` and an `EnvironmentV2` with `step_with_context()` has been added in `farm/environments/context_environment.py` as a stepping stone.

## PettingZoo ParallelEnv Wrapper
### Rationale
The batch-step model aligns with PettingZoo’s ParallelEnv: all agents provide actions each step. This enables:
- Conformance testing and easier evaluation
- Integration with MARL tooling

### Initial Spaces (V1)
- Action space: `spaces.Discrete(5)`: 0=move, 1=gather, 2=share, 3=attack, 4=reproduce
  - Parameters use defaults initially; later evolve to `spaces.Dict` with action parameters.
- Observation space: `spaces.Box(shape=(8,), dtype=float32)` using `AgentState.to_tensor()`
  - Future: richer dicts or perception grids.

### Wrapper Responsibilities
- `reset(seed=None, options=None)` initializes env and returns per-agent observations and infos.
- `step(actions: dict[agent_id -> action])` decodes actions → intents → runs batched pipeline → returns (observations, rewards, terminations, truncations, infos).
- Maintains `agents` and `possible_agents`, handling dynamic births/deaths.

## Benefits
- Fairness: eliminates first-actor advantage within a step.
- Determinism: reproducible outcomes with seeded resolution.
- Parallelization-ready: planning can be run concurrently since all read the same snapshot.
- Modularity: agents depend on `Context`, not mutable environment internals.
- Ecosystem compatibility: PettingZoo enables tooling and baseline algorithms for validation.

## Considerations & Trade-offs
- Conflict logic adds complexity; must be thoroughly tested.
- Behavior might shift vs immediate mutation; mitigate with tests and metrics.
- Parametric actions increase space complexity; staged rollout recommended.
- DB semantics move to end-of-step commit; update log timing expectations.
- Observation design: balance between compact features and richer perception.

## Migration Plan
1) Transitional APIs
   - Add `plan(context)` and optional `act_with_context(context)`
   - Keep `act()` working; `EnvironmentV2.step_with_context()` prefers the new path.
2) Introduce `step_batched()`
   - Implement intents, resolver, commit phase; add config flag to toggle batched vs immediate mode.
3) PettingZoo wrapper
   - Add ParallelEnv adapter and conformance tests.
4) Stabilize and switch defaults
   - Default to batched mode in examples; deprecate immediate mutation path.
5) Extend spaces
   - Move from Discrete actions to Dict parameterized actions as modules mature.

## Testing Strategy
- Unit tests for each conflict policy (move, gather, attack, share)
- Reproducibility tests (fixed seeds)
- Integration tests with dynamic add/remove and reproduction
- PettingZoo conformance tests
- Performance smoke tests

## Open Questions
- Priority policies beyond seeded ordering (e.g., utility-based arbitration)?
- Multi-intent per agent per step; ordering and constraints.
- Parametric action space layout for flexibility vs simplicity.
- Interrupt semantics (out-of-band events) in a batched model.
- Distributed planning with Ray/ProcessPool; batching and timeouts.

## Pseudocode Appendix
### Agent (minimal)
```python
class BaseAgent:
    def plan(self, context: Context) -> list[ActionIntent]:
        action = self.decide_action()  # existing selection
        return [map_action_to_intent(self, action)]
```

### Environment step (batched)
```python
def step_batched(self):
    ctx = self.build_context()
    intents = [a.plan(ctx) for a in self.agents if a.alive]
    resolved = self.resolve_conflicts(intents)
    self.apply_effects(resolved)
    self.commit_lifecycle_changes()
    self.update()  # resources, kd-trees, metrics, time += 1
```

---
For a transitional implementation, see `farm/environments/context_environment.py` which adds `Context` and `EnvironmentV2.step_with_context()` to ease migration before full batched-intent adoption.