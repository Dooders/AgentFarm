# Design documents

RFC-style design notes for AgentFarm subsystems. These describe intent and evolution; when in doubt, trust the code and [API reference](../reference/api-reference.md).

## Index

| Document | Status | Summary |
|----------|--------|---------|
| [Agent loop](agent_loop.md) | **Proposed** | Observation → perception → cognition → action loop (aspirational) |
| [Agent model](Agent.md) | **Accepted** | AgentCore, components, and behavior extension |
| [Hyperparameter chromosome](hyperparameter_chromosome.md) | **Accepted** | Typed genes, bounds, reproduction-time mutation |
| [Evolvable loci roadmap](evolvable_loci_roadmap.md) | **Proposed** | Future gene loci and evolution surface |
| [Inherited payload design](inherited_payload_design.md) | **Accepted** | What agents pass to offspring beyond chromosomes |
| [Crossover search space](crossover_search_space.md) | **Accepted** | Neural crossover + fine-tune search dimensions |
| [Crossover strategies](crossover_strategies.md) | **Accepted** | Parent blending semantics and PTQ loading |
| [Distill / quantize / crossover / fine-tune](distill_quantize_crossover_finetune.md) | **Accepted** | End-to-end neural recombination pipeline |
| [On-chain run provenance](onchain_run_provenance.md) | **Proposed** | FarmNotary adapter: hash official artifacts, optional CID + chain stamp |
| [Architectural recommendations](architectural-recommendations.md) | **Reference** | Historical architecture review notes |

## Status key

- **Accepted** — implemented or actively used; may still drift from code
- **Proposed** — direction documented; not fully implemented
- **Reference** — background material, not a normative spec

## Related

- [Architecture overview](../concepts/architecture.md)
- [Neural recombination guide](../guides/neural-recombination.md)
- [FarmNotary guide](../guides/farm-notary.md)
- [Research devlog](../research/devlog/index.md)
