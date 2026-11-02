"""Gate behavior for scripts/engine_review/engine_baseline.py.

Failed children and a comparison with no shared configuration must not exit 0.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "engine_review" / "engine_baseline.py"
_spec = importlib.util.spec_from_file_location("engine_baseline_harness", _SCRIPT)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = harness
_spec.loader.exec_module(harness)


def _row(label: str, steps: int, ms: float = 1.0, rss: float = 10.0) -> dict:
    return {
        "label": label,
        "opts": {"label": label, "steps": steps},
        "wall_s": 1.0,
        "ms_per_agent_step": ms,
        "peak_rss_mb": rss,
        "kdtree_rebuilds": 0,
    }


def _baseline(*rows: dict) -> dict:
    return {
        "context": {"git_revision": "abc123def456", "platform": "test"},
        "results": list(rows),
    }


def test_compare_rejects_disjoint_labels() -> None:
    """Smoke labels do not overlap the committed review baseline."""
    current = {"results": [_row("smoke_pop_15", 100)]}
    baseline = _baseline(_row("pop_15", 200))
    assert harness.compare(current, baseline, 0.10) == 1


def test_compare_rejects_same_label_with_different_opts() -> None:
    current = {"results": [_row("pop_15", 100, ms=9.0, rss=90.0)]}
    baseline = _baseline(_row("pop_15", 200))
    assert harness.compare(current, baseline, 0.10) == 1


def test_compare_rejects_empty_current_results() -> None:
    current = {"results": []}
    baseline = _baseline(_row("pop_15", 200))
    assert harness.compare(current, baseline, 0.10) == 1


def test_compare_flags_metric_regression() -> None:
    current = {"results": [_row("pop_15", 200, ms=1.2, rss=10.0)]}
    baseline = _baseline(_row("pop_15", 200, ms=1.0, rss=10.0))
    assert harness.compare(current, baseline, 0.10) == 1


def test_compare_passes_within_threshold() -> None:
    current = {"results": [_row("pop_15", 200, ms=1.05, rss=10.4)]}
    baseline = _baseline(_row("pop_15", 200, ms=1.0, rss=10.0))
    assert harness.compare(current, baseline, 0.10) == 0


def test_run_suite_records_failed_child(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="simulation crashed")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    results, failures = harness.run_suite([{"label": "pop_15"}], tmp_path)
    assert results == []
    assert failures == ["pop_15"]


def test_run_suite_records_missing_result_line(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="no metrics\n", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    results, failures = harness.run_suite([{"label": "pop_30"}], tmp_path)
    assert results == []
    assert failures == ["pop_30"]


def test_run_suite_keeps_successful_result(tmp_path: Path, monkeypatch) -> None:
    row = _row("pop_15", 200)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout="RESULT " + json.dumps(row) + "\n", stderr=""
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    results, failures = harness.run_suite([{"label": "pop_15"}], tmp_path)
    assert failures == []
    assert results == [row]


def test_main_failed_child_exits_nonzero_without_compare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "run_suite", lambda configs, work_dir: ([], ["smoke_pop_15"]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["engine_baseline.py", "--suite", "smoke", "--work-dir", str(tmp_path)],
    )
    assert harness.main() == 1


def test_main_failed_child_exits_nonzero_when_other_rows_match(tmp_path: Path, monkeypatch) -> None:
    row = _row("pop_15", 200)
    monkeypatch.setattr(harness, "run_suite", lambda configs, work_dir: ([row], ["pop_30"]))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline(row)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_baseline.py",
            "--suite",
            "review",
            "--compare",
            str(baseline_path),
            "--work-dir",
            str(tmp_path),
        ],
    )
    assert harness.main() == 1


def test_main_disjoint_compare_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        harness,
        "run_suite",
        lambda configs, work_dir: ([_row("smoke_pop_15", 100)], []),
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline(_row("pop_15", 200))), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_baseline.py",
            "--suite",
            "smoke",
            "--compare",
            str(baseline_path),
            "--work-dir",
            str(tmp_path),
        ],
    )
    assert harness.main() == 1


def test_main_matching_compare_exits_zero(tmp_path: Path, monkeypatch) -> None:
    current = _row("pop_15", 200, ms=1.05, rss=10.4)
    baseline_row = _row("pop_15", 200, ms=1.0, rss=10.0)
    monkeypatch.setattr(harness, "run_suite", lambda configs, work_dir: ([current], []))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline(baseline_row)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine_baseline.py",
            "--suite",
            "review",
            "--compare",
            str(baseline_path),
            "--work-dir",
            str(tmp_path),
        ],
    )
    assert harness.main() == 0
