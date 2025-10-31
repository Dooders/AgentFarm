### Identity Service

The `farm/utils/identity.py` module centralizes all ID generation and parsing logic.
It replaces scattered UUID generation and ad hoc string concatenation with a clear,
testable API. Deterministic sequences are supported when a seed is provided.

#### Key types
- `SimulationId`, `ExperimentId`, `RunId`, `AgentId`, `AgentStateId`, `GenomeIdStr`

#### Configuration
```python
from farm.utils.identity import Identity, IdentityConfig

identity = Identity(IdentityConfig(deterministic_seed=123))
```

#### Methods
- `simulation_id(prefix="sim") -> SimulationId`
- `run_id(length=8) -> RunId`
- `experiment_id() -> ExperimentId`
- `agent_id() -> AgentId` (deterministic sequence when `deterministic_seed` set)
- `agent_state_id(agent_id: str, step: int) -> AgentStateId`
- `genome_id(parent_ids: Sequence[str], existing_genome_checker: Optional[Callable[[str], bool]] = None) -> GenomeIdStr`
  - Format: `parent1:parent2[:counter]`
  - No parents (initial agents): `::` or `::0`, `::1`, etc.
  - Single parent (cloning): `agent_a:` or `agent_a:0`, `agent_a:1`, etc.
  - Two parents (sexual reproduction): `agent_a:agent_b` or `agent_a:agent_b:0`, `agent_a:agent_b:1`, etc.
  - Counter is appended automatically when duplicate base IDs exist
- `parse_agent_state_id(agent_state_id: str) -> tuple[str, int]`

#### Usage in Environment
`Environment` initializes an `Identity` instance with its `seed_value` (if provided) and uses it
for `simulation_id` and `get_next_agent_id()`.

#### Migration notes
- `short_id.generate_simulation_id` is now a thin shim delegating to `Identity`.
- Direct calls to `uuid.uuid4()` for IDs have been replaced by `Identity` factories.

