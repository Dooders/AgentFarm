# GitHub issue triage — AgentFarm (`Dooders/AgentFarm`)

Reference list for open issues: what to do with each category. Last updated from codebase review on branch `cursor/existing-issues-assessment-f0df`.

## Close as implemented (verify in issue thread, then close)

These match current code paths; close with a short comment pointing at the implementation if you want history.

| Issue | Title | Implementation / notes |
|-------|--------|-------------------------|
| [#61](https://github.com/Dooders/AgentFarm/issues/61) | Implement `ExperimentDatabase` class | `farm/database/experiment_database.py`, `farm/database/test_experiment_db.py` |
| [#70](https://github.com/Dooders/AgentFarm/issues/70) | Deaths and births not tracked | `birth_time` / `death_time`, step metrics; e.g. `farm/analysis/significant_events/compute.py` |
| [#152](https://github.com/Dooders/AgentFarm/issues/152) | Database sharding for large-scale sims | `ShardedSimulationDatabase`, `ShardedDataLogger` in `farm/database/` |
| [#154](https://github.com/Dooders/AgentFarm/issues/154) | Batch operations / buffering for DB performance | `DataLogger` buffers and bulk inserts in `farm/database/data_logging.py` |
| [#156](https://github.com/Dooders/AgentFarm/issues/156) | Parallel experiment runner | `farm/runners/parallel_experiment_runner.py` |
| [#186](https://github.com/Dooders/AgentFarm/issues/186) | Single database file for multiple simulations | `ExperimentDatabase` / unified DB story |
| [#207](https://github.com/Dooders/AgentFarm/issues/207) | Centralized Pydantic configuration | `farm/config/schema.py`, core state, decision/observation configs, API models |

## Close as obsolete or superseded

| Issue | Title | Action |
|-------|--------|--------|
| [#25](https://github.com/Dooders/AgentFarm/issues/25) | Strategy pattern for action selection in MoveModule | **Close** — layout is now `MovementComponent` + strategy-style decision/behavior code elsewhere; retitle new work if needed. |
| [#231](https://github.com/Dooders/AgentFarm/issues/231) | Multi-modal data logging (Week 1 framing) | **Close** as duplicate / stale milestone — keep [#230](https://github.com/Dooders/AgentFarm/issues/230) or one umbrella multimodal-logging issue. |
| [#350](https://github.com/Dooders/AgentFarm/issues/350) | Ensemble methods for decision-making | **Close** or **narrow** — `farm/core/decision/algorithms/ensemble.py` already provides ensemble-style selectors; open a new issue for specific gaps. |

## Merge or deduplicate

| Issues | Action |
|--------|--------|
| [#230](https://github.com/Dooders/AgentFarm/issues/230) vs [#231](https://github.com/Dooders/AgentFarm/issues/231) | Prefer **one** issue for structured / multimodal / HDF5 logging; close the other after copying any unique acceptance criteria. |

## Reassess or rewrite (not automatic close)

| Issue | Title | What to do |
|-------|--------|------------|
| [#450](https://github.com/Dooders/AgentFarm/issues/450) | Post–dev merge roadmap epic | **Archive**, split into child issues, or replace with a living doc. References (e.g. Electron) may not match `farm/editor`. |
| [#208](https://github.com/Dooders/AgentFarm/issues/208) | Redis agent memory — adaptive batch sizing | **Re-scope** to what is still missing; Redis memory exists under `farm/memory/`, `farm/database/memory.py`. |
| [#209](https://github.com/Dooders/AgentFarm/issues/209) | Redis-buffered data logging | **Re-scope** vs current `DataLogger` / DB paths; close if fully redundant. |
| [#347](https://github.com/Dooders/AgentFarm/issues/347)–[#351](https://github.com/Dooders/AgentFarm/issues/351), [#426](https://github.com/Dooders/AgentFarm/issues/426) | MVP 1.0 performance / training batch | **Re-triage** — drop or update `MVP 1.0` label; memmap already used for resources (`farm/core/resource_manager.py`); GPU / mixed precision / distillation / sharing need explicit acceptance tests. |

## Related optimizations (keep open until verified)

| Issue | Title | Notes |
|-------|--------|--------|
| [#150](https://github.com/Dooders/AgentFarm/issues/150) | I/O and parallelize experiments | Overlaps parallel runner; verify remaining I/O work. |
| [#153](https://github.com/Dooders/AgentFarm/issues/153) | Asynchronous data logging | Distinct from buffered sync logging; confirm if still a goal. |
| [#163](https://github.com/Dooders/AgentFarm/issues/163) | Parallel runner startup time | Still a valid follow-up if slow cold starts matter. |
| [#168](https://github.com/Dooders/AgentFarm/issues/168) | Analysis scripts unified framework | Separate from DB/sim runner triage. |
| [#190](https://github.com/Dooders/AgentFarm/issues/190) | Agent type mapping inconsistency | Bug-style; verify against current mapping code. |
| [#194](https://github.com/Dooders/AgentFarm/issues/194) | Relative advantage analysis module | Verify vs `farm/analysis/advantage/`. |

## Long-standing backlog (2024) — product decision

Many issues from late 2024 (experiments, genetics, perception, GUI, etc.) are **not wrong**, but **untouched for months**. For each: **close (won’t do)**, **icebox**, or **refresh** with a one-line scope and owner.

Examples: [#8](https://github.com/Dooders/AgentFarm/issues/8), [#9](https://github.com/Dooders/AgentFarm/issues/9), [#10](https://github.com/Dooders/AgentFarm/issues/10), [#13](https://github.com/Dooders/AgentFarm/issues/13), [#16](https://github.com/Dooders/AgentFarm/issues/16), [#18](https://github.com/Dooders/AgentFarm/issues/18), [#22](https://github.com/Dooders/AgentFarm/issues/22), [#24](https://github.com/Dooders/AgentFarm/issues/24), [#34](https://github.com/Dooders/AgentFarm/issues/34)–[#99](https://github.com/Dooders/AgentFarm/issues/99), [#111](https://github.com/Dooders/AgentFarm/issues/111), [#117](https://github.com/Dooders/AgentFarm/issues/117), [#123](https://github.com/Dooders/AgentFarm/issues/123), [#218](https://github.com/Dooders/AgentFarm/issues/218), [#271](https://github.com/Dooders/AgentFarm/issues/271), [#315](https://github.com/Dooders/AgentFarm/issues/315), etc.

## Batch close command (local `gh`)

If your token can close issues, you can run:

```bash
REPO=Dooders/AgentFarm
for n in 61 70 152 154 156 186 207 25 231 350; do
  gh issue close "$n" --repo "$REPO" --comment "Closed per docs/github-issue-triage.md — reopen if anything remains."
done
```

Adjust the issue list after you merge or deduplicate #230 / #231.
