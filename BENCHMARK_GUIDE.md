# AgentFarm Benchmarking Guide (Spec-driven)

This guide explains how to run and interpret experiments (benchmarks and profiles) using the spec-driven framework.

## Quick Start

### List available experiments

```bash
python -m benchmarks.run_benchmarks --list
```

### Run a single experiment

```bash
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_baseline.yaml
```

### Run a sweep

```bash
python -m benchmarks.run_benchmarks --spec benchmarks/specs/observation_sweep.yaml
```

## Available Experiments

### Observation Flow Experiment

Purpose: Tests observation generation throughput under various loads.

Key Params (via spec): `num_agents`, `steps`, `width`, `height`, `radius`, `device`.

Additional experiments can be added under `benchmarks/implementations/` and registered with a slug.

## Understanding Results

- Per run, a JSON result is saved with environment and VCS metadata, iteration metrics, and summary stats.
- A Markdown `README.md` is generated per run with parameters, duration statistics, and artifact links.
- If enabled, artifacts like cProfile `.prof` and psutil JSONL samples are saved alongside results.

### Common Metrics

- Duration (seconds): mean, p50, p95 across measured iterations
- Throughput (if exposed by the experiment): e.g., `observes_per_sec`

## Comparing Two Runs (A/B)

```bash
python -m benchmarks.run_benchmarks --compare path/to/A.json path/to/B.json
```

The tool prints a Markdown summary comparing duration statistics and common throughput metrics.

## Best Practices

1. Start small and scale up parameters.
2. Use `iterations.warmup` in specs to stabilize caches.
3. Pin `seed` for reproducibility when comparing runs.
4. Enable `cprofile` and `psutil` instruments in specs for deeper analysis.

## Creating a New Experiment

1. Create a new file in `benchmarks/implementations/`.
2. Subclass `Experiment`, implement `setup`, `execute_once`, and `teardown`.
3. Register with `@register_experiment("your_slug")` and define `param_schema` defaults.
4. Add a spec under `benchmarks/specs/` to run it.

## References

- `benchmarks/core/*`: Experiment API, Runner, Registry, Spec, Reporting
- `benchmarks/run_benchmarks.py`: Spec-driven CLI
- `benchmarks/specs/*`: Example specs for single-run and sweeps
