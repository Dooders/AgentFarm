"""Tests for ``scripts/benchmark_inference.py`` — multi-device inference harness.

CI-friendly: all tests run on CPU only with tiny in-memory models (no saved
checkpoints required).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "benchmark_inference.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_inference_mod", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so @dataclass annotation resolution works.
    sys.modules["benchmark_inference_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        _mod = _load_module()
    return _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_ns(**overrides) -> argparse.Namespace:
    base = dict(
        input_dim=8,
        output_dim=4,
        hidden_size=64,
        n_states=32,
        seed=0,
        warmup=2,
        repeats=5,
        throughput_repeats=5,
        devices="cpu",
        batch_sizes="1",
        states_file="",
        checkpoint_dir="",
        parent_ckpt="",
        student_ckpt="",
        int8_ckpt="",
        child_ckpt="",
        allow_unsafe_unpickle=False,
        report_dir="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_states(n: int = 32, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype("float32")


def _bench_meta(states: np.ndarray, **overrides) -> dict:
    """Keyword-only metadata for :func:`run_benchmark` tests."""
    sh: Tuple[int, int] = (int(states.shape[0]), int(states.shape[1]))
    base = {
        "states_source": "pytest",
        "states_loaded_shape": sh,
        "git_commit": "",
        "hostname": "test-host",
    }
    base.update(overrides)
    return base


def _make_tiny_parent(input_dim: int = 8, output_dim: int = 4) -> torch.nn.Module:
    from farm.core.decision.base_dqn import BaseQNetwork

    net = BaseQNetwork(input_dim=input_dim, output_dim=output_dim, hidden_size=16)
    net.eval()
    return net


def _make_tiny_student(input_dim: int = 8, output_dim: int = 4) -> torch.nn.Module:
    from farm.core.decision.base_dqn import StudentQNetwork

    net = StudentQNetwork(input_dim=input_dim, output_dim=output_dim, parent_hidden_size=16)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# _validate_args tests
# ---------------------------------------------------------------------------


def test_validate_args_accepts_defaults():
    mod = _get_mod()
    mod._validate_args(_default_ns())


@pytest.mark.parametrize(
    "field, value",
    [
        ("input_dim", 0),
        ("output_dim", -1),
        ("hidden_size", 0),
        ("n_states", 0),
        ("warmup", -1),
        ("repeats", 0),
        ("throughput_repeats", -1),
    ],
)
def test_validate_args_rejects_invalid(field: str, value: int):
    mod = _get_mod()
    with pytest.raises(ValueError, match=field):
        mod._validate_args(_default_ns(**{field: value}))


# ---------------------------------------------------------------------------
# _parse_int_list tests
# ---------------------------------------------------------------------------


def test_parse_int_list_single():
    mod = _get_mod()
    assert mod._parse_int_list("1", "--batch-sizes") == [1]


def test_parse_int_list_multiple():
    mod = _get_mod()
    assert mod._parse_int_list("1,16,64", "--batch-sizes") == [1, 16, 64]


def test_parse_int_list_rejects_zero():
    mod = _get_mod()
    with pytest.raises(ValueError, match="positive"):
        mod._parse_int_list("1,0", "--batch-sizes")


def test_parse_int_list_rejects_non_int():
    mod = _get_mod()
    with pytest.raises(ValueError, match="not a valid integer"):
        mod._parse_int_list("1,abc", "--batch-sizes")


def test_parse_int_list_rejects_empty():
    mod = _get_mod()
    with pytest.raises(ValueError, match="cannot be empty"):
        mod._parse_int_list("", "--batch-sizes")


# ---------------------------------------------------------------------------
# _resolve_checkpoint tests
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_explicit_wins(tmp_path):
    mod = _get_mod()
    explicit = str(tmp_path / "my.pt")
    result = mod._resolve_checkpoint(explicit, str(tmp_path), "other.pt")
    assert result == explicit


def test_resolve_checkpoint_directory_fallback(tmp_path):
    mod = _get_mod()
    target = tmp_path / "parent.pt"
    target.touch()
    result = mod._resolve_checkpoint("", str(tmp_path), "parent.pt")
    assert result == str(target)


def test_resolve_checkpoint_returns_empty_when_not_found(tmp_path):
    mod = _get_mod()
    result = mod._resolve_checkpoint("", str(tmp_path), "missing.pt")
    assert result == ""


# ---------------------------------------------------------------------------
# _load_int8
# ---------------------------------------------------------------------------


def test_load_int8_requires_checkpoint_path():
    mod = _get_mod()
    with pytest.raises(ValueError, match="--int8-ckpt"):
        mod._load_int8("", allow_unsafe=True)


def test_load_int8_requires_unsafe_flag():
    mod = _get_mod()
    with pytest.raises(ValueError, match="allow-unsafe-unpickle"):
        mod._load_int8("/nonexistent/for_gating_only.pt", allow_unsafe=False)


def test_load_int8_unpacks_checkpoint_and_sets_eval(monkeypatch):
    mod = _get_mod()
    dummy = torch.nn.Linear(4, 2)
    dummy.train()

    def fake_load(path: str):
        assert path == "/fake/int8.pt"
        return dummy, {"meta": True}

    monkeypatch.setattr(mod, "load_quantized_checkpoint", fake_load)
    out = mod._load_int8("/fake/int8.pt", allow_unsafe=True)
    assert out is dummy
    assert not out.training


# ---------------------------------------------------------------------------
# BenchmarkRow / BenchmarkReport serialisation
# ---------------------------------------------------------------------------


def test_benchmark_row_fields():
    mod = _get_mod()
    row = mod.BenchmarkRow(
        model="parent",
        device="cpu",
        batch_size=1,
        median_ms=0.1,
        mean_ms=0.11,
        p95_ms=0.15,
        throughput_batches_per_sec=9000.0,
        n_warmup=5,
        n_repeats=50,
    )
    assert row.model == "parent"
    assert row.batch_size == 1


def test_benchmark_report_to_dict_is_json_serialisable():
    mod = _get_mod()
    row = mod.BenchmarkRow(
        model="parent",
        device="cpu",
        batch_size=1,
        median_ms=0.1,
        mean_ms=0.11,
        p95_ms=0.15,
        throughput_batches_per_sec=9000.0,
        n_warmup=5,
        n_repeats=50,
    )
    report = mod.BenchmarkReport(
        schema_version="1.1",
        torch_version="2.0.0",
        states_shape=(32, 8),
        states_loaded_shape=(16, 8),
        states_source="synthetic",
        devices=["cpu"],
        batch_sizes=[1, 4],
        warmup=5,
        repeats=50,
        throughput_repeats=200,
        git_commit="deadbeef",
        hostname="pytest",
        rows=[row],
    )
    d = report.to_dict()
    serialised = json.dumps(d)
    parsed = json.loads(serialised)
    assert parsed["schema_version"] == "1.1"
    assert isinstance(parsed["states_shape"], list)
    assert parsed["states_loaded_shape"] == [16, 8]
    assert parsed["devices"] == ["cpu"]
    assert parsed["batch_sizes"] == [1, 4]
    assert len(parsed["rows"]) == 1


# ---------------------------------------------------------------------------
# _time_model_single tests
# ---------------------------------------------------------------------------


def test_time_model_single_returns_positive_ms():
    mod = _get_mod()
    model = _make_tiny_parent()
    inp = torch.zeros(1, 8)
    median, mean, p95 = mod._time_model_single(model, inp, n_warmup=2, n_repeats=5)
    assert median > 0
    assert mean > 0
    assert p95 >= median


# ---------------------------------------------------------------------------
# run_benchmark end-to-end (CPU, tiny model)
# ---------------------------------------------------------------------------


def test_run_benchmark_single_model_cpu():
    mod = _get_mod()
    states = _make_states()
    parent = _make_tiny_parent()
    report = mod.run_benchmark(
        named_models=[("parent", parent)],
        states=states,
        devices=["cpu"],
        batch_sizes=[1],
        n_warmup=2,
        n_repeats=5,
        throughput_n_timed=5,
        **_bench_meta(states),
    )
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.model == "parent"
    assert row.device == "cpu"
    assert row.batch_size == 1
    assert row.median_ms > 0
    assert row.throughput_batches_per_sec > 0


def test_run_benchmark_multiple_models_cpu():
    mod = _get_mod()
    states = _make_states()
    parent = _make_tiny_parent()
    student = _make_tiny_student()
    report = mod.run_benchmark(
        named_models=[("parent", parent), ("student", student)],
        states=states,
        devices=["cpu"],
        batch_sizes=[1, 4],
        n_warmup=1,
        n_repeats=3,
        throughput_n_timed=3,
        **_bench_meta(states),
    )
    # 2 models × 1 device × 2 batch sizes = 4 rows
    assert len(report.rows) == 4
    models_seen = {r.model for r in report.rows}
    assert models_seen == {"parent", "student"}
    batch_sizes_seen = {r.batch_size for r in report.rows}
    assert batch_sizes_seen == {1, 4}


def test_run_benchmark_states_shape_matches_probe_not_full_array():
    """``states_shape`` reflects the probe slice (max batch), not ``states.shape``."""
    mod = _get_mod()
    states = _make_states(n=64, dim=8)
    parent = _make_tiny_parent()
    report = mod.run_benchmark(
        named_models=[("parent", parent)],
        states=states,
        devices=["cpu"],
        batch_sizes=[1, 4],
        n_warmup=1,
        n_repeats=2,
        throughput_n_timed=0,
        **_bench_meta(states),
    )
    assert report.states_shape == (4, 8)
    assert report.states_loaded_shape == (64, 8)


def test_run_benchmark_no_throughput():
    mod = _get_mod()
    states = _make_states()
    parent = _make_tiny_parent()
    report = mod.run_benchmark(
        named_models=[("parent", parent)],
        states=states,
        devices=["cpu"],
        batch_sizes=[1],
        n_warmup=1,
        n_repeats=3,
        throughput_n_timed=0,
        **_bench_meta(states),
    )
    assert len(report.rows) == 1
    assert report.rows[0].throughput_batches_per_sec == 0.0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def test_format_table_contains_headers():
    mod = _get_mod()
    row = mod.BenchmarkRow(
        model="parent",
        device="cpu",
        batch_size=1,
        median_ms=0.1,
        mean_ms=0.11,
        p95_ms=0.15,
        throughput_batches_per_sec=9000.0,
        n_warmup=5,
        n_repeats=50,
    )
    table = mod._format_table([row], show_throughput=True)
    assert "Model" in table
    assert "Median" in table
    assert "Throughput" in table
    assert "parent" in table
    assert "cpu" in table


def test_format_table_no_throughput_column():
    mod = _get_mod()
    row = mod.BenchmarkRow(
        model="parent",
        device="cpu",
        batch_size=1,
        median_ms=0.1,
        mean_ms=0.11,
        p95_ms=0.15,
        throughput_batches_per_sec=0.0,
        n_warmup=5,
        n_repeats=50,
    )
    table = mod._format_table([row], show_throughput=False)
    assert "Throughput" not in table


def test_format_table_empty_returns_string():
    mod = _get_mod()
    result = mod._format_table([], show_throughput=False)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def test_write_reports_creates_files(tmp_path):
    mod = _get_mod()
    row = mod.BenchmarkRow(
        model="parent",
        device="cpu",
        batch_size=1,
        median_ms=0.1,
        mean_ms=0.11,
        p95_ms=0.15,
        throughput_batches_per_sec=9000.0,
        n_warmup=5,
        n_repeats=50,
    )
    report = mod.BenchmarkReport(
        schema_version="1.1",
        torch_version=torch.__version__,
        states_shape=(32, 8),
        states_loaded_shape=(32, 8),
        states_source="synthetic",
        devices=["cpu"],
        batch_sizes=[1],
        warmup=5,
        repeats=50,
        throughput_repeats=200,
        git_commit="",
        hostname="pytest",
        rows=[row],
    )
    mod._write_reports(report, str(tmp_path), show_throughput=True)

    json_path = tmp_path / "inference_benchmark.json"
    md_path = tmp_path / "inference_benchmark.md"
    assert json_path.exists()
    assert md_path.exists()

    with open(json_path) as fh:
        data = json.load(fh)
    assert data["schema_version"] == "1.1"
    assert len(data["rows"]) == 1

    md_content = md_path.read_text()
    assert "parent" in md_content
    assert "Median" in md_content


# ---------------------------------------------------------------------------
# Full end-to-end: save tiny checkpoint, run main() on CPU
# ---------------------------------------------------------------------------


def test_main_end_to_end_cpu_random_weights(tmp_path, capsys):
    """Run main() with no checkpoints (random weights) on CPU — CI-safe."""
    mod = _get_mod()
    report_dir = str(tmp_path / "reports")

    # Patch sys.argv and call main().
    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--input-dim", "8",
        "--output-dim", "4",
        "--hidden-size", "16",
        "--n-states", "16",
        "--devices", "cpu",
        "--batch-sizes", "1,4",
        "--warmup", "2",
        "--repeats", "5",
        "--throughput-repeats", "5",
        "--report-dir", report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "Inference Latency Benchmark" in captured.out
    assert "parent" in captured.out
    assert "cpu" in captured.out

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    assert os.path.isfile(json_path)
    with open(json_path) as fh:
        data = json.load(fh)
    # 1 model × 1 device × 2 batch sizes = 2 rows
    assert len(data["rows"]) == 2
    assert data["rows"][0]["device"] == "cpu"
    assert data["schema_version"] == "1.1"
    assert "hostname" in data and "git_commit" in data
    assert data["devices"] == ["cpu"]
    assert data["batch_sizes"] == [1, 4]


def test_main_end_to_end_cpu_with_checkpoint(tmp_path, capsys):
    """Run main() with a tiny saved checkpoint — CI-safe."""
    from farm.core.decision.base_dqn import BaseQNetwork

    mod = _get_mod()

    # Save a tiny parent checkpoint.
    net = BaseQNetwork(input_dim=4, output_dim=2, hidden_size=8)
    ckpt_path = str(tmp_path / "parent.pt")
    torch.save(net.state_dict(), ckpt_path)

    report_dir = str(tmp_path / "reports")

    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--parent-ckpt", ckpt_path,
        "--input-dim", "4",
        "--output-dim", "2",
        "--hidden-size", "8",
        "--n-states", "8",
        "--devices", "cpu",
        "--batch-sizes", "1",
        "--warmup", "1",
        "--repeats", "3",
        "--throughput-repeats", "0",
        "--report-dir", report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "Inference Latency Benchmark" in captured.out

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    assert os.path.isfile(json_path)


def test_main_end_to_end_parent_and_student(tmp_path, capsys):
    """Run main() benchmarking both parent and student on CPU."""
    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork

    mod = _get_mod()

    parent = BaseQNetwork(input_dim=4, output_dim=2, hidden_size=8)
    student = StudentQNetwork(input_dim=4, output_dim=2, parent_hidden_size=8)

    parent_ckpt = str(tmp_path / "parent.pt")
    student_ckpt = str(tmp_path / "student.pt")
    torch.save(parent.state_dict(), parent_ckpt)
    torch.save(student.state_dict(), student_ckpt)

    report_dir = str(tmp_path / "reports")

    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--parent-ckpt", parent_ckpt,
        "--student-ckpt", student_ckpt,
        "--input-dim", "4",
        "--output-dim", "2",
        "--hidden-size", "8",
        "--n-states", "8",
        "--devices", "cpu",
        "--batch-sizes", "1",
        "--warmup", "1",
        "--repeats", "3",
        "--throughput-repeats", "3",
        "--report-dir", report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    with open(json_path) as fh:
        data = json.load(fh)
    # 2 models × 1 device × 1 batch size = 2 rows
    assert len(data["rows"]) == 2
    model_names = {r["model"] for r in data["rows"]}
    assert model_names == {"parent", "student"}


def test_main_cuda_unavailable_skips_gracefully(tmp_path, capsys, monkeypatch):
    """When CUDA is requested but unavailable, CPU results are still produced."""
    mod = _get_mod()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report_dir = str(tmp_path / "reports")

    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--input-dim", "4",
        "--output-dim", "2",
        "--hidden-size", "8",
        "--n-states", "4",
        "--devices", "cpu,cuda",
        "--batch-sizes", "1",
        "--warmup", "1",
        "--repeats", "3",
        "--throughput-repeats", "0",
        "--report-dir", report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "CUDA requested but not available" in captured.out

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    with open(json_path) as fh:
        data = json.load(fh)
    # Only CPU rows should be present.
    assert all(r["device"] == "cpu" for r in data["rows"])


def test_main_cuda_index_unavailable_skips_gracefully(tmp_path, capsys, monkeypatch):
    """``cuda:N`` is skipped when CUDA is unavailable, same as bare ``cuda``."""
    mod = _get_mod()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report_dir = str(tmp_path / "reports")

    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--input-dim", "4",
        "--output-dim", "2",
        "--hidden-size", "8",
        "--n-states", "4",
        "--devices", "cpu,cuda:0",
        "--batch-sizes", "1",
        "--warmup", "1",
        "--repeats", "3",
        "--throughput-repeats", "0",
        "--report-dir", report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "CUDA requested but not available" in captured.out
    assert "cuda:0" in captured.out

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert all(r["device"] == "cpu" for r in data["rows"])


def test_main_states_file_used(tmp_path, capsys):
    """Run main() with a real .npy states file."""
    mod = _get_mod()

    states = np.random.default_rng(7).standard_normal((16, 4)).astype("float32")
    states_path = str(tmp_path / "states.npy")
    np.save(states_path, states)

    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--input-dim", "4",
        "--output-dim", "2",
        "--hidden-size", "8",
        "--n-states", "16",
        "--states-file", states_path,
        "--devices", "cpu",
        "--batch-sizes", "1",
        "--warmup", "1",
        "--repeats", "3",
        "--throughput-repeats", "0",
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "Inference Latency Benchmark" in captured.out
    # The states path should appear in the output.
    assert states_path in captured.out


def test_main_rejects_unsupported_device():
    mod = _get_mod()
    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--devices",
        "metal",
        "--n-states",
        "4",
    ]
    try:
        with pytest.raises(ValueError, match="Unsupported device"):
            mod.main()
    finally:
        sys.argv = old_argv


def test_main_padded_states_preserves_loaded_shape_in_json(tmp_path):
    """Fewer state rows than max batch: JSON records pre-pad and post-pad shapes."""
    mod = _get_mod()
    report_dir = str(tmp_path / "reports")
    old_argv = sys.argv
    sys.argv = [
        "benchmark_inference.py",
        "--input-dim",
        "4",
        "--output-dim",
        "2",
        "--hidden-size",
        "8",
        "--n-states",
        "2",
        "--batch-sizes",
        "8",
        "--devices",
        "cpu",
        "--warmup",
        "1",
        "--repeats",
        "2",
        "--throughput-repeats",
        "0",
        "--report-dir",
        report_dir,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    with open(json_path) as fh:
        data = json.load(fh)
    assert data["states_loaded_shape"] == [2, 4]
    assert data["states_shape"] == [8, 4]
