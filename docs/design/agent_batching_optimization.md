# Proposal: batch per-agent observation construction and policy inference

> Status: proposed / not yet implemented
>
> This document is the design write-up behind the performance follow-up referenced
> in the observation-pipeline optimization PR. It is intended to seed a GitHub issue
> and guide implementation.

## Summary
Profiling of the development workload (`run_simulation.py --environment development --steps 2000 --seed 1234567890`) shows that roughly **50% of simulation runtime is torch ML compute** — DQN policy inference plus deferred training (`torch.conv2d` ~18s, `run_backward` ~16.7s, `linear` ~5.2s, Adam ~13s in a 600-step cProfile run). This work currently runs **once per agent per step** on tiny per-agent observation tensors (≈13×13 multi-channel), which is dominated by Python/dispatch overhead and is highly inefficient on small tensors.

This proposes **batching observation construction and policy inference (and ideally the deferred training step) across all agents that share a policy**, so each simulation step performs one vectorized forward/backward pass over a batched tensor instead of N small ones. This should meaningfully cut the dominant ML cost **without changing learning numerics** if done carefully.

## Background / current state
- Each agent builds its own observation via `PerceptionComponent.get_observation_tensor` → `AgentObservation.perceive_world` (`farm/core/observations.py`), then runs an individual policy forward in `farm/core/decision/`.
- The step loop in `farm/core/simulation.py` already processes agents in batches (`agent_processing_batch_size`, default 32) but calls `agent.act()` per agent, so inference/observation are not vectorized.
- Recent behavior-preserving optimizations reduced the *observation pipeline* overhead (~12% end-to-end), but the ML compute itself was intentionally left untouched because changing it risks changing results:
  - memoize VISIBILITY disk mask
  - disable benchmark-only observation metrics on the live path
  - NumPy-buffer resource accumulation + redundant spatial-update removal

## Proposed approach
1. **Batch observation tensors:** stack the per-agent observation tensors for all alive agents (sharing the same shape/device/dtype) into a single `(N, C, S, S)` batch.
2. **Single inference pass:** run one policy forward over the batch to get per-agent action distributions, then dispatch actions back to agents.
3. **Batched deferred training:** where agents share a policy/optimizer, accumulate transitions and run one (or few) batched training update(s) per step instead of per-agent updates.
4. **Handle heterogeneity:** group agents by policy/network identity (e.g., system vs. independent vs. control agents, or per-shared-module) and batch within each group; fall back to the per-agent path for singletons.

## Hard constraints (must hold)
- **Numerical results must be identical** (or the change must be explicitly gated behind a flag and validated). The repo is fully deterministic with a fixed seed + `PYTHONHASHSEED=0`; a batched forward must produce the same action selections and the same training gradients/updates as the sequential path.
- **Determinism oracle:** validate against a full-sim state signature (agent ids/alive/positions/health/resources + resources + time + cached totals) — it must remain bitwise-identical, or any divergence must be understood and justified.
- `pytest -q` must stay green (modulo the known pre-existing `tests/analysis/*` matplotlib/genetics failures).

## Risks / open questions
- **Batched vs. sequential float math:** batched matmul/conv may produce slightly different floating-point results than N separate calls (reduction order), which would change action selection and break determinism. Need to measure; may require keeping per-agent math where exactness is required, or accepting a flag-gated "fast/approximate" mode.
- **Training order semantics:** sequential deferred updates apply gradients one agent at a time (each update sees the previous one's weights); a single batched update changes that semantics. Decide whether to (a) preserve exact sequential semantics (limits batching to inference only) or (b) define batched training as the new, intended semantics behind a config flag.
- **Variable agent counts / deaths within a step**, mixed devices, and per-agent enabled-action masks (curriculum) need handling in the batch.

## Suggested scope / phasing
- **Phase A (lower risk):** batch *observation construction + inference only* (action selection), keep training per-agent. Validate determinism.
- **Phase B (higher risk):** batch deferred training for shared-policy groups behind a config flag; document the semantic change and validate learning-curve equivalence.

## Acceptance criteria
- Measurable reduction in step time on the development workload (target: a substantial cut in the ~50% ML share).
- Determinism signature unchanged for Phase A (or flag-gated + documented for Phase B).
- `pytest -q` green; new tests covering batched-vs-sequential equivalence.

## Pointers
- `farm/core/simulation.py` — step loop, `_run_deferred_learning_updates`.
- `farm/core/agent/core.py` — `act()` / `step()` / `_execute_action`.
- `farm/core/decision/` — policy/decision modules, tianshou wrappers.
- `farm/core/agent/components/perception.py`, `farm/core/observations.py` — observation construction.
