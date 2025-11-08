## Research Assistant Tool Usage Benchmark Tiers

This taxonomy defines progressive evaluation bands for measuring how well a research assistant agent deploys its available tools. Each tier introduces additional reasoning depth, evidence requirements, and orchestration complexity.

### Tier Summary

| Tier | Focus | Question Pattern | Expected Tooling | Primary Pass Criteria |
| --- | --- | --- | --- | --- |
| 0 | Warmup retrieval | Single-fact lookup from a known source | One retrieval/read action | Exact answer, ≤2 tool calls, no irrelevant tools |
| 1 | Context stitching | Combine 2–3 related facts | Sequential retrieval or light synthesis | Consistent answer with cited support, limited redundant calls |
| 2 | Cross-document synthesis | Merge heterogeneous evidence (docs/code/data) | Mixed retrieval, parsing utilities | Evidence coverage across sources, efficient tool chain |
| 3 | Analytical reasoning | Perform calculations or code execution | REPL/data tools plus retrieval | Correct computation, explicit verification trace |
| 4 | Research-style exploration | Plan multi-step investigation with risk analysis | Full tool suite with iterative refinement | Complete plan, quality iterations, risk identification |
| 5 | Novel tool orchestration | Learn and operate unfamiliar tools | Doc ingestion plus exploratory runs | Successful onboarding, error recovery, applied execution |

### Tier Details & Metrics

- **Tier 0 · Warmup Retrieval**
  - Tracks: `answer_exact`, `tool_latency_mean`, `tool_calls_total`, `irrelevant_call_rate`
  - Failure signals: incorrect fact, >2 tool invocations, exploratory tools

- **Tier 1 · Context Stitching**
  - Tracks: `answer_consistency`, `supporting_refs_count`, `tool_parallelism`, `time_to_first_tool`
  - Failure signals: missing citation, redundant loops, fabricated links

- **Tier 2 · Cross-Document Synthesis**
  - Tracks: `evidence_coverage`, `chain_depth`, `tool_efficiency`, `response_fidelity`
  - Failure signals: unsupported claims, dead-end tool chains, omitted required tools

- **Tier 3 · Analytical Reasoning**
  - Tracks: `calc_correctness`, `verification_trace`, `runtime_cost`, `artifact_quality`
  - Failure signals: unchecked code execution, manual arithmetic only, partial outputs

- **Tier 4 · Research-Style Exploration**
  - Tracks: `plan_completeness`, `tool_branching_factor`, `iteration_quality`, `risk_identification`
  - Failure signals: shallow plans, no revision, critical tool omissions, missing risk assessment

- **Tier 5 · Novel Tool Orchestration**
  - Tracks: `onboarding_time`, `adaptation_success`, `error_recovery_rate`, `documentation_utilization`
  - Failure signals: repeated identical errors, ignoring docs, incomplete execution logs

### Cross-Tier Rollups

- `tier_pass_rate` — success ratio per tier
- `escalation_resilience` — ability to recover after a failed tier
- `tool_safety_incidents` — unsafe or disallowed tool usage
- `human_override_requests` — interventions required to complete a task

