## Spatial Search Requirements (Filled with Explanations)

### 1. Data Characteristics

- **What type of data are you indexing?** Points (agents and resources), potential small axis-aligned rectangles for viewport/vision.
  - **Explanation**: Current operations center on agent/resource positions; rectangles might appear for visualization or field-of-view queries.

- **What is the dimensionality?** 2D (x, y).
  - **Explanation**: The simulation world is a plane; no z-dimension currently used.

- **How many items do you typically index?** Typical ~20k (range 10k–30k); min ~2k; max observed ~50k; expected growth 100k–500k; stretch ~1M (batch/offline).
  - **Explanation**: Real-time targets are ≤500k; higher counts are for offline/relaxed-latency runs.

- **Are the items uniform or mixed?** Mostly uniform points with mixed semantic types (agent/resource); occasional small AABBs.
  - **Explanation**: Geometry is point-like, but behavior/filters differ by type.

- **Do items have additional attributes beyond position?** Yes: agent.alive (bool), agent.team (enum), agent.radius (float), resource.type (enum), resource.remaining (int).
  - **Explanation**: These are used as filters after spatial candidate generation; pre-filter hints are desirable but optional.

- **What is the spatial distribution?** Mixed/clustered with hotspots; clusters move slowly.
  - **Explanation**: Non-uniformity favors bucketed grids or trees with good partitioning to avoid degenerate performance.

- **Are there bounds or a fixed environment size?** Fixed rectangular world (known width × height). Queries remain within bounds; out-of-bounds inputs are clamped/ignored.
  - **Explanation**: Bounded world enables grid-based indices with fixed extents and simpler hashing.

### 2. Update Dynamics

- **How frequently do updates occur?** Every simulation step; target 30–60 steps/sec.
  - **Explanation**: Index must keep up with real-time frame cadence.

- **What types of updates?** Many position changes per step; frequent inserts/removals (spawns/despawns); alive/dead toggles; infrequent attribute changes.
  - **Explanation**: Favors dynamic indices with efficient move/remove/insert operations.

- **What fraction of items change per update cycle?** Positions: 20–60%; inserts/removals: 0.5–5%; status toggles: 0–2%.
  - **Explanation**: Full rebuilds each step are costly at these change rates; incremental updates preferred.

- **Do you need to detect changes efficiently?** Yes—dirty tracking by ID and batched reindexing; “no-change” steps are rare (<5%).
  - **Explanation**: Avoids touching unaffected items and reduces rebuild time.

- **Are updates batched or individual?** Batched per step; prefer incremental structure updates.
  - **Explanation**: Allows amortized updates and epoch swapping for lock-free reads.

- **How tolerant are you of stale data?** Low for interactions/collisions; up to 1 step stale acceptable for analytics.
  - **Explanation**: Supports dual-mode: strict for gameplay-critical queries, relaxed for non-critical metrics.

### 3. Query Types and Patterns

- **Primary query types?** Radius/range queries; nearest neighbor; k-NN (k ≤ 10); occasional rectangle range.
  - **Explanation**: Drives choice toward spatial hash/grid for range, KD/BallTree for k-NN, or hybrid.

- **How frequent are queries?** ~1–3 queries per active agent per step. For ~20k at 60 Hz: ~1.2–3.6M queries/sec (upper bound; often reduced via throttling/subsampling).
  - **Explanation**: High query volume requires low constant factors and vectorized computations.

- **Typical query parameters?** Radius 10–40 units (max 200); k typical 1–5 (max 10); result counts usually 0–50.
  - **Explanation**: Small neighborhoods favor bucket-based candidates to bound work.

- **Do queries filter on non-spatial attributes?** Yes: alive-only, by team, resource.type.
  - **Explanation**: Post-filtering is fine if candidate sets are small; prefilter partitions can help if skew is high.

- **Are queries point- or region-based?** Mostly point-based; some rectangle region queries.
  - **Explanation**: Point queries dominate; region queries should remain efficient but are secondary.

- **Any advanced needs?** Euclidean distance; approximate k-NN acceptable (≤10% error) for ≥2× speed; optional toroidal wrapping mode; exclude self and recently interacted IDs.
  - **Explanation**: Enables ANN libraries for k-NN and simple wrap-around distance handling.

### 4. Performance and Scalability Requirements

- **Latency targets?** Frame budget 16 ms (60 FPS). Index maintenance ≤ 4–6 ms/step; spatial queries ≤ 6–8 ms/step. Median per-query ≤ 50 µs; tail ≤ 250 µs.
  - **Explanation**: Leaves time for other simulation systems (AI/physics/rendering).

- **Memory constraints?** Prefer O(n). ≤ 1–2 GB at 500k items; avoid large per-item overhead.
  - **Explanation**: Select structures with compact node storage and cache-friendly layouts.

- **Throughput needs?** Sustain 1M+ queries/sec on 8-core desktop CPU (3.0–4.5 GHz) with vectorization.
  - **Explanation**: Encourages batched queries and SIMD-friendly distance calculations.

- **Expected scale?** Real-time up to 100k–500k; batch/offline up to ~1M with relaxed latency.
  - **Explanation**: Solutions should degrade gracefully and permit offline modes.

- **Bottlenecks observed?** Full KD-tree rebuild per step is expensive; Python-level overhead in filters/candidate merges; poor cache locality in pointer-heavy structures.
  - **Explanation**: Suggests grid/SoA layouts and minimizing Python loops.

- **Tolerance for approximations?** Yes for k-NN (90–95% accuracy for ≥2× speed); range queries must be exact.
  - **Explanation**: Mixed exact/approx policy depending on query semantics.

### 5. Environment and Integration Constraints

- **Programming language and ecosystem?** Python 3.10–3.11; NumPy. SciPy optional. Numba acceptable. C/C++-backed libs allowed.
  - **Explanation**: Broad library support; JIT or native code for performance-critical paths.

- **Allowed dependencies?** SciPy (cKDTree), rtree (libspatialindex), pyqtree, scikit-learn (BallTree/KDTree), FAISS (optional ANN), Shapely 2/PyGEOS (optional).
  - **Explanation**: No hard restriction on C extensions in target envs.

- **Integration needs?** Maintain a `SpatialIndex`-like interface; concurrent read-only queries; single-threaded writer per step; prefer reentrant queries with copy-on-write/epoch swapping.
  - **Explanation**: Supports safe concurrent reads during updates.

- **Testing/debugging features?** Expose `get_stats()` (build/update/query counts, timings, average candidate sizes); optional debug sampling and visualization hooks.
  - **Explanation**: Aids performance tuning and correctness checks.

- **Any domain-specific rules?** Clamp to bounds; discard NaN/inf; ignore `invisible` entities; no privacy constraints.
  - **Explanation**: Ensures data cleanliness and predictable behavior.

### 6. Trade-offs and Goals

- **Primary goal?** Maximize end-to-end simulation throughput with predictable latency; minimize rebuild costs; keep code complexity reasonable.
  - **Explanation**: Prioritizes stable real-time performance over algorithmic novelty.

- **Willingness to customize?** Prefer off-the-shelf core indices with custom wrappers/hybrids (grid + KD/R-tree); open to light customization and filter pipelines.
  - **Explanation**: Reduces maintenance while enabling targeted optimizations.

- **Known pain points?** Costly full rebuilds; Python overhead in filtering; redundant distance checks; occasional staleness with deferred rebuilds; memory churn.
  - **Explanation**: Directs efforts toward incremental updates, batching, and SoA data layouts.

- **Inspirations or benchmarks?** Uniform/spatial hash grids (game engines); SciPy cKDTree; libspatialindex R-tree; boids/swarm sims; ANN libs (FAISS/HNSW).
  - **Explanation**: Proven patterns for high query throughput and dynamic scenes.

- **Any other requirements?** No persistence; single-process focus; future multi-process/batch possible; prefer permissive licensing (MIT/BSD).
  - **Explanation**: Keeps integration and distribution simple.

