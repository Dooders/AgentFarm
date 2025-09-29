## Streaming and Chunked Data Processing

This document explains the new streaming/chunked data processing capabilities added to the analysis framework. The goal is to efficiently handle large datasets without exhausting RAM, while maintaining performance.

### What changed

- `iter_data(...)` is now the primary API on `DataLoader` for streaming chunks.
- `load_data(...)` now concatenates streamed chunks and is considered secondary.
- Added `run_streaming(...)` to `AnalysisTask` to process data incrementally.
- Implemented chunked streaming in loaders:
- `CSVLoader.iter_data(...)` uses pandas `chunksize` (and `load_data` leverages it by default).
- `JSONLoader.iter_data(...)` streams JSON Lines (`.jsonl`/`.ndjson`) via pandas; falls back to single-load for non-line-delimited JSON (and `load_data` leverages streaming by default).
  - `SQLiteLoader.execute_query_iter(...)` streams query results; plus `iter_agents/resources/steps/reproduction_events/simulations(...)`.
  - `SimulationLoader.iter_data(...)` dispatches to table-specific iterators; `iter_time_series(...)` streams derived rows.
  - `ExperimentLoader.iter_data(...)` streams across multiple simulation DBs and annotates each chunk with `db_path`.

### Why it helps

- Memory safety: Process datasets larger than available RAM by handling one chunk at a time.
- Performance: Avoids large in-memory DataFrames, reducing GC pressure and peak memory, typically keeping throughput within ~10% of full-load paths (dependent on IO and transforms).
- Composability: Any `DataProcessor` can be applied per-chunk, enabling pipelines that scale.

### How to use

Minimal example with CSV streaming:

```python
from farm.analysis.data.loaders import CSVLoader

loader = CSVLoader(file_path="/path/to/large.csv")
for chunk in loader.iter_data(chunksize=200_000):
    # process each chunk
    do_something(chunk)
```

Streaming an end-to-end analysis task:

```python
from farm.analysis.base import AnalysisTask
from farm.analysis.data.loaders import SimulationLoader
from my_project.processors import MyProcessor
from my_project.analyzers import MyAnalyzer

task = AnalysisTask(
    data_loader=SimulationLoader(db_path="/path/to/sim.db", simulation_id=1),
    data_processor=MyProcessor(),
    analyzer=MyAnalyzer(),
)

def consume(processed_chunk):
    # e.g., aggregate, write to DB, compute online stats
    accumulate(processed_chunk)

results = task.run_streaming(process_chunk=consume, loader_args={"table": "steps", "chunk_size": 10_000})
```

Streaming from multiple simulation databases:

```python
from farm.analysis.data.loaders import ExperimentLoader

exp = ExperimentLoader(db_paths=["/exp/run1.db", "/exp/run2.db"]) 
for df in exp.iter_data(table="steps", chunk_size=5_000):
    # df contains a "db_path" column indicating source database
    process(df)
```

### Configuration tips

- **Chunk sizing**: Start with `chunk_size` or `chunksize` between 2,000 and 50,000 rows depending on row width and memory. Increase if IO-bound, decrease if memory headroom is tight.
- **JSON**: For multi-GB JSON, use line-delimited JSON (`.jsonl`/`.ndjson`) to enable streaming. Non-line-delimited JSON is loaded as a single chunk by design.
- **SQLite**: The `iter_*` methods internally use `yield_per(...)` to reduce memory overhead when loading ORM rows.
- **Downstream processing**: Prefer per-chunk transformations to avoid concatenating all chunks. If a final analysis requires a full DataFrame, the framework will concatenate at the end only when no `process_chunk` callback is supplied.

### Performance tuning

- **Column projection**: Use `columns=[...]` with `SimulationLoader.iter_data(...)`/`SQLiteLoader` iterators to fetch only needed fields. This can significantly improve throughput and reduce memory.
- **Indexes**: Ensure your database has an index on filtering columns. For example, add an index on `simulation_steps(simulation_id)` and optionally `simulation_steps(simulation_id, step_number)` to accelerate range scans:

```sql
CREATE INDEX IF NOT EXISTS idx_sim_steps_sim_id ON simulation_steps(simulation_id);
CREATE INDEX IF NOT EXISTS idx_sim_steps_sim_id_step ON simulation_steps(simulation_id, step_number);
```

- **ORM vs raw SQL**: For maximum streaming performance, prefer raw SQL chunking (`execute_query_iter`) over ORM iteration.

### Backwards compatibility

- Existing code using `load_data(...)` continues to work unchanged.
- New streaming methods are opt-in; you can switch incrementally.

### Notes

- These changes complement existing memmap and database optimizations. No Redis setup is required for streaming; however, you can still combine streaming with external buffers or queues if desired.

