## Spatial Search Requirements (Filled with Explanations)

### 1. Data Characteristics

- **What type of data are you indexing?** Points: agents and resources.
  - **Explanation (from code)**: `farm/core/agent.py` and `farm/core/resources.py` store `position` as `(x, y)` tuples. No rectangle/polygon geometries are used in queries. Visualization draws points/circles.

- **What is the dimensionality?** 2D (x, y).
  - **Explanation (from code)**: Environment is constructed with `width` and `height` scalars and all spatial ops use 2D tuples.

- **How many items do you typically index?** Defaults: 30 agents (10 per type) and 20 resources; hard cap `max_population` default 300 agents. Configurable via `config.yaml`.
  - **Explanation (from code)**: `SimulationConfig` defaults `system_agents=10`, `independent_agents=10`, `control_agents=10`, and `initial_resources=20` (`farm/core/config.py`, `config.yaml`). The environment enforces `max_population=300`. Actual counts depend on runtime config.

- **Are the items uniform or mixed?** Uniform point geometries with different semantics (agents vs resources).
  - **Explanation (from code)**: Geometry is always point-based; class differences only affect behavior/filters.

- **Do items have additional attributes beyond position?** Yes: agents have `alive: bool`, `agent_type: str`, `current_health: float`, `resource_level: int`; resources have `amount: float`, `max_amount: float`, `regeneration_rate: float`.
  - **Explanation (from code)**: Seen in `farm/core/agent.py` and `farm/core/resources.py`. Queries mainly use spatial positions; filtering by `alive` is handled inside the index for agents.

- **What is the spatial distribution?** Initially random uniform placement for resources and agents; alternative patterns exist for resources (grid/clustered) but default is random.
  - **Explanation (from code)**: `ResourceManager.initialize_resources` with distribution `{"type": "random"}` from `Simulation`; helper methods support grid/clustered creation.

- **Are there bounds or a fixed environment size?** Fixed rectangular world, `width × height`.
  - **Explanation (from code)**: `SpatialIndex._is_valid_position` enforces `0 ≤ x ≤ width` and `0 ≤ y ≤ height`. Queries outside bounds are rejected, not wrapped.

### 2. Update Dynamics

- **How frequently do updates occur?** Once per environment step.
  - **Explanation (from code)**: `Environment.update()` is called each main loop iteration and invokes `spatial_index.update()` once per step.

- **What types of updates?** Position changes, agent additions/removals, resource additions/removals, resource amount changes.
  - **Explanation (from code)**: Agents move in actions (`action.move` updates position). `Environment.add_agent`/`remove_agent` and `ResourceManager.add_resource`/`remove_resource` modify sets; these call `mark_positions_dirty()`.

- **What fraction of items change per update cycle?** Not explicitly tracked.
  - **Explanation**: No direct metric in code; depends on agent behaviors. Tests exercise both small and large changes.

- **Do you need to detect changes efficiently?** Yes, implemented via counts and position hashing.
  - **Explanation (from code)**: `SpatialIndex.update()` checks `_counts_changed()` and `_hash_positions_changed()` to decide KD-tree rebuild. Dirty flag triggers checking, but rebuild only occurs on detected change.

- **Are updates batched or individual?** Batched per step with a single `update()` call.
  - **Explanation (from code)**: Many mutations mark dirty; actual rebuild decision occurs once in `Environment.update()`.

- **How tolerant are you of stale data?** Queries are served against last built KD-trees until `update()` runs.
  - **Explanation (from code)**: There is no partial incremental update; consistency is per-step. Staleness within the same step before `update()` is possible by design.

### 3. Query Types and Patterns

- **Primary query types?** Point-radius range queries and nearest-resource queries.
  - **Explanation (from code)**: `SpatialIndex.get_nearby_agents`, `get_nearby_resources`, and `get_nearest_resource` are the exposed operations; no rectangle or k-NN APIs are present.

- **How frequent are queries?** Per observation update and various decision points per step.
  - **Explanation (from code)**: `AgentObservation._compute_entities_from_spatial_index` calls `get_nearby_agents` with `fov_radius` for each observed agent. Tests and environment methods also issue queries as needed.

- **Typical query parameters?** Radii around 6 (observation `fov_radius` default) and 10–20 in tests; combat/gathering ranges in config are around 20–30 but spatial queries for agents primarily use `fov_radius`.
  - **Explanation (from code)**: `ObservationConfig.fov_radius=6`; tests use radii 5–20; config includes `attack_range=20`, `gathering_range=30` which may inform application-level ranges.

- **Do queries filter on non-spatial attributes?** Agents are auto-filtered by `alive`; no other attribute filters at index-level.
  - **Explanation (from code)**: `SpatialIndex` builds agent KD-tree from alive agents only; resource queries return all resources within radius.

- **Are queries point- or region-based?** Point-based.
  - **Explanation (from code)**: All index methods take a single `(x,y)` and optional radius.

- **Any advanced needs?** Euclidean distance; no toroidal wrapping; no approximate queries in current implementation.
  - **Explanation (from code)**: KDTree uses Euclidean metrics; `_is_valid_position` rejects out-of-bounds; no ANN or custom metrics used.

### 4. Performance and Scalability Requirements

- **Latency targets?** Not explicitly specified; real-time step loop executes `Environment.update()` once per step.
  - **Explanation (from code)**: No profiling/targets in code. Tests include sanity performance checks (e.g., 100 queries loops) but without hard thresholds.

- **Memory constraints?** Not specified.
  - **Explanation**: No memory budgets or constraints detected in code or configs.

- **Throughput needs?** Not specified.
  - **Explanation**: No explicit QPS goals in repo; unit tests perform small batches (e.g., 100 queries) as functionality checks.

- **Expected scale?** Defaults: ~30 agents and 20 resources; configurable. Upper bounds governed by `max_population` default of 300.
  - **Explanation (from code)**: `SimulationConfig` and `config.yaml` set defaults; no large-scale stress targets included.

- **Bottlenecks observed?** KD-tree rebuilds triggered on count/position changes each step; no incremental updates.
  - **Explanation (from code)**: `SpatialIndex.update()` may rebuild both trees when counts or hashes differ; frequency depends on movement and spawn/despawn.

- **Tolerance for approximations?** Not used.
  - **Explanation (from code)**: Exact KDTree queries only; no approximate or custom metrics implemented.

### 5. Environment and Integration Constraints

- **Programming language and ecosystem?** Python 3.10–3.11; NumPy; SciPy used for KDTree.
  - **Explanation (from code/deps)**: `requirements.txt` includes NumPy, scikit-learn, etc.; `SpatialIndex` imports `scipy.spatial.KDTree`.

- **Allowed dependencies?** SciPy present; other spatial libs not used.
  - **Explanation**: No rtree/pyqtree/faiss in requirements; adding them would require updating deps.

- **Integration needs?** `SpatialIndex` is used via `Environment` and an `ISpatialQueryService` adapter.
  - **Explanation (from code)**: `Environment` exposes `get_nearby_agents/resources` and `get_nearest_resource`; `SpatialIndexAdapter` wraps index to service interface; reads happen throughout agent perception; writes are centralized per step.

- **Testing/debugging features?** `SpatialIndex.get_stats()` returns basic info; extensive unit tests exist.
  - **Explanation (from code)**: `get_stats()` includes counts, flags, cache info; tests validate behavior and edge cases.

- **Any domain-specific rules?** Positions must be within bounds; invalid inputs rejected.
  - **Explanation (from code)**: `_is_valid_position` checks bounds; movement clamps to config bounds in `action.move` if set.

### 6. Trade-offs and Goals

- **Primary goal?** Maintain correctness and simplicity while supporting per-step queries for observations and decisions.
  - **Explanation (from code)**: Emphasis on clean APIs and robust tests; no explicit throughput/latency targets.

- **Willingness to customize?** Current implementation relies on SciPy KDTree; adapters exist to abstract the service.
  - **Explanation**: `SpatialIndexAdapter` decouples callers from implementation, allowing swaps with minimal changes.

- **Known pain points?** Potential rebuild cost each step when counts/positions change; per-step hashing O(n).
  - **Explanation (from code)**: Update strategy may rebuild frequently in highly dynamic scenes; no incremental per-entity updates.

- **Inspirations or benchmarks?** Not explicitly stated in repo.
  - **Explanation**: Docs reference general performance considerations but no spatial index alternatives.

- **Any other requirements?** None beyond current API and bounds validation.
  - **Explanation**: No persistence for index; single-process use within environment.

