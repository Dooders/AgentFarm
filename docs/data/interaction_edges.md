# Interaction Edge Logging

AgentFarm logs interactions as edges between environment nodes to enable social network analysis, lineage tracing, and resource interaction analytics.

## Table: `interactions`

Columns:
- `interaction_id` (PK)
- `simulation_id` (FK simulations.simulation_id)
- `step_number` (int)
- `source_type` (str): one of `agent`, `resource`
- `source_id` (str)
- `target_type` (str): one of `agent`, `resource`
- `target_id` (str)
- `interaction_type` (str): semantic type such as `share`, `attack`, `gather`, `reproduce`, and their `_attempt`/`_failed` variants
- `action_type` (str, optional): action grouping
- `details` (JSON, optional)
- `timestamp` (datetime)

Indexes are provided on `(step_number)`, `(source_type, source_id)`, `(target_type, target_id)`, and `(interaction_type)`.

## Programmatic API

Use the environment helper to log edges:

```python
env.log_interaction_edge(
    source_type="agent", source_id="A1",
    target_type="resource", target_id="42",
    interaction_type="gather", action_type="gather",
    details={"amount": 3}
)
```

Under the hood this calls `DataLogger.log_interaction_edge`, which buffers and batch-writes to the `interactions` table.

## Automatic Instrumentation

The following actions now log interaction edges automatically:
- Share: `agent -> agent` (`share`, `share_attempt`)
- Attack: `agent -> agent` (`attack`, `attack_failed`, `attack_attempt`)
- Gather: `agent -> resource` (`gather`, `gather_attempt`, `gather_failed`)
- Reproduce: `agent -> agent` parent→offspring (`reproduce`), plus attempts/failures as self-edges

## Notes
- Interaction logs include the current simulation step and optional metadata in `details`.
- Edge logging is a supplement to existing `agent_actions` records and does not replace them.