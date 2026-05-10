"""Guardrails for the committed spatial benchmark artifact."""

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFIED_JSON = _REPO_ROOT / "benchmarks" / "results" / "spatial_benchmark_verified.json"
_VERIFIED_MD = _REPO_ROOT / "benchmarks" / "results" / "SPATIAL_BENCHMARK_VERIFIED.md"


@pytest.mark.unit
def test_spatial_benchmark_verified_json_exists_and_schema() -> None:
    assert _VERIFIED_JSON.is_file(), (
        "Missing benchmarks/results/spatial_benchmark_verified.json — run:\n"
        "  PYTHONHASHSEED=0 python benchmarks/implementations/spatial/comprehensive_spatial_benchmark.py --verified"
    )
    payload = json.loads(_VERIFIED_JSON.read_text(encoding="utf-8"))
    assert "verification" in payload
    v = payload["verification"]
    assert v.get("random_seed") == 42
    assert v.get("git_revision")
    assert "results" in payload and isinstance(payload["results"], list)

    standard = [
        r
        for r in payload["results"]
        if "build_time" in r and "distribution" in r and "implementation" in r
    ]
    assert len(standard) >= 24  # 6 impls × 4 sizes × 2 distributions (when scipy/sklearn present)

    for r in standard:
        assert isinstance(r["build_time"], (int, float))
        assert isinstance(r["avg_query_time"], (int, float))
        assert r["entity_count"] in {100, 500, 1000, 2000}
        assert r["distribution"] in {"uniform", "clustered"}

    batch = [r for r in payload["results"] if r.get("implementation") == "AgentFarm Batch Updates"]
    assert len(batch) == 3
    for r in batch:
        assert "batch_update_time" in r and "individual_update_time" in r

    sw = payload.get("step_workload_benchmark")
    assert sw is not None, "JSON should include step_workload_benchmark from --verified run"
    assert "results" in sw and len(sw["results"]) == 2
    for row in sw["results"]:
        assert row["entity_count"] in (500, 1000)
        assert "batch_total_time_mean_s" in row
        assert "immediate_total_time_mean_s" in row
        assert "immediate_over_batch" in row
        assert row["batch_total_time_mean_s"] > 0
        assert row["immediate_total_time_mean_s"] > 0


@pytest.mark.unit
def test_spatial_benchmark_verified_markdown_exists() -> None:
    assert _VERIFIED_MD.is_file(), (
        "Missing benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md — regenerate with --verified (see sibling JSON test)."
    )
    text = _VERIFIED_MD.read_text(encoding="utf-8")
    assert "Verified spatial benchmark" in text
    assert "Distribution:" in text
    assert "Interleaved step workload" in text
