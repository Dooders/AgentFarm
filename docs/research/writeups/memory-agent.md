---
layout: experiment
title: Memory Agent
subtitle: Hierarchical compression as the agent's history
status: in-progress
updated: 2026-01-01
category: Agent cognition
excerpt: >-
  A three-tier memory (short-term, intermediate, long-term) with progressive
  compression, cross-tier replay, and reconstructive use at decision time.
  This writeup is the experiment as designed: the architecture variants are
  specified; a published run matrix is not.
variants:
  - id: stm
    title: Short-term memory
  - id: intermediate
    title: Intermediate memory
  - id: ltm
    title: Long-term memory
  - id: replay
    title: Cross-tier reconstructive replay
---

Traditional agents either keep full history (expensive) or a short recency
window (blind to long-term pattern). Memory Agent asks whether a
biologically-inspired compression ladder can keep decision-relevant
information while dropping detail, and whether that changes learning and
adaptation relative to a standard AgentFarm agent.

This is an **in-progress** experiment: design and analysis notes exist;
there is not yet a seed-matched results matrix on this page. The variants
below are the architecture treatments the experiment is built to compare,
not completed confirmatory cells.

Design corpus:
[Overview](../experiments/memory_agent/README.md),
[Memory model](../experiments/memory_agent/Memory.md),
[Design considerations](../experiments/memory_agent/DesignConsiderations.md),
[Implementation](../experiments/memory_agent/Implementation.md),
[Walkthrough](../experiments/memory_agent/Detail.md),
[Advanced](../experiments/memory_agent/Advanced.md).

## Shared design

The agent extends AgentFarm's base agent with a three-tier store. Memories
move STM → IM → LTM by age, importance, and relevance. Compression increases
at each hop. Decisions query all tiers. Experience replay can sample across
tiers, reconstruct compressed traces, and up-weight important memories.

Questions the whole experiment is meant to answer:

1. Can compression preserve decision-relevant information?
2. What is the fidelity vs compute trade-off?
3. Does hierarchical memory change learning, adaptation, and decision quality?
4. Which compression stack is the useful compromise?

## Short-term memory {#stm}

High-dimensional, complete traces of recent experience. This is the
no-compression control inside the hierarchy: if STM-only agents match the
baseline AgentFarm agent, the new machinery has not yet done any work. Cost
and recency bias are the expected failure modes.

## Intermediate memory {#intermediate}

Moderately compressed older traces. The first place the experiment can
show a trade-off: reconstruction error vs reduced memory footprint, and
whether mid-horizon patterns (resource cycles, neighbor identity) survive
the first compression hop.

## Long-term memory {#ltm}

Highly compressed, abstract distant experience. This variant is the reason
the experiment exists — access to long-horizon structure after details are
gone. The risk is semantic drift: reconstructed memories that are fluent
and wrong. Evaluation has to compare reconstructed traces against the
originals on *behaviorally* relevant features, not pixel-level MSE alone.

## Cross-tier reconstructive replay {#replay}

Training-time treatment, orthogonal to which tiers exist:

- sample from STM / IM / LTM, not only recency
- importance-weighted replay
- reconstruct IM/LTM before the update
- adaptive sampling ratios as learning converges

This is the variant that asks whether compressed history is actually *used*
as training signal, or only as a retrieval trick at act time.

## Synthesis

Until a run matrix exists, do not treat the architecture as a result. The
self-contained claim of this writeup is the comparison the experiment is
built for: STM-only vs the full ladder, with reconstructive replay on or
off, scored on memory footprint, decision quality, and adaptation speed in
environments that require temporal pattern.

## Reproduce and artifacts

No single CLI yet; implementation notes are in
[Implementation](../experiments/memory_agent/Implementation.md).
[Catalog entry](../experiments-catalog.md).
