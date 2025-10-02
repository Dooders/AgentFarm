import os
import json

from benchmarks.core.spec import load_spec
from benchmarks.core.registry import REGISTRY
from benchmarks.core.runner import Runner


def test_spec_load_minimal(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
experiment: observation_flow
params:
  width: 50
  height: 50
  steps: 1
  num_agents: 1
iterations:
  warmup: 0
  measured: 1
instrumentation:
  - timing
        """
    )
    spec = load_spec(str(spec_path))
    assert spec.experiment == "observation_flow"
    assert spec.iterations["measured"] == 1


def test_registry_and_runner(tmp_path):
    # Discover and create experiment
    REGISTRY.discover_package("benchmarks.implementations")
    exp = REGISTRY.create("observation_flow", {"width": 20, "height": 20, "steps": 1, "num_agents": 1})
    runner = Runner(
        name="observation_flow",
        experiment=exp,
        output_dir=str(tmp_path),
        iterations_warmup=0,
        iterations_measured=1,
        seed=42,
        tags=["test"],
        notes="",
        instruments=["timing"],
    )
    res = runner.run()
    assert res.name == "observation_flow"
    assert "duration_s" in (res.metrics or {})
    saved_files = os.listdir(runner.run_dir)
    assert any(f.endswith(".json") for f in saved_files)


def test_compare_output_markdown(tmp_path):
    # Create two minimal result JSONs to compare
    a = {
        "name": "observation_flow",
        "run_id": "A",
        "metrics": {"duration_s": {"mean": 1.0, "p50": 1.0, "p95": 1.0}},
        "iteration_metrics": [{"metrics": {"observes_per_sec": 100.0}}],
    }
    b = {
        "name": "observation_flow",
        "run_id": "B",
        "metrics": {"duration_s": {"mean": 2.0, "p50": 2.0, "p95": 2.0}},
        "iteration_metrics": [{"metrics": {"observes_per_sec": 200.0}}],
    }
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(a))
    b_path.write_text(json.dumps(b))
    from benchmarks.core.compare import compare_results

    md = compare_results(str(a_path), str(b_path))
    assert "Compare:" in md
    assert "Throughput" in md
