"""Tests for scripts/run_comparison_matrix.py and scripts/summarise_comparison_matrix.py.

Covers:
- run_comparison_matrix: argument parsing helpers, _extract_row_summary,
  _write_summary, and an integration smoke-test using temporary checkpoints.
- summarise_comparison_matrix: _detect_type, _extract_recombination,
  _extract_quant_fidelity, _build_markdown, _load_reports, and main() end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_SCRIPT = _REPO_ROOT / "scripts" / "run_comparison_matrix.py"
_SUMMARY_SCRIPT = _REPO_ROOT / "scripts" / "summarise_comparison_matrix.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_driver = None
_summary = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = _load_module(_DRIVER_SCRIPT, "run_comparison_matrix_mod")
    return _driver


def _get_summary():
    global _summary
    if _summary is None:
        _summary = _load_module(_SUMMARY_SCRIPT, "summarise_comparison_matrix_mod")
    return _summary


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN = 64


def _make_base_qnetwork(seed: int):
    from farm.core.decision.base_dqn import BaseQNetwork
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=HIDDEN)


def _make_student(seed: int):
    from farm.core.decision.base_dqn import StudentQNetwork
    torch.manual_seed(seed)
    return StudentQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, parent_hidden_size=HIDDEN)


def _save_state_dict(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def _save_quantized(model, path: str) -> None:
    from farm.core.decision.training.quantize_ptq import PostTrainingQuantizer, QuantizationConfig
    model.eval()
    q = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
    q_model, result = q.quantize(model)
    q.save_checkpoint(q_model, path, result)


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    """Create a directory with all required checkpoints."""
    d = tmp_path_factory.mktemp("ckpts")
    _save_state_dict(_make_base_qnetwork(1), str(d / "parent_A.pt"))
    _save_state_dict(_make_base_qnetwork(2), str(d / "parent_B.pt"))
    _save_state_dict(_make_base_qnetwork(3), str(d / "child.pt"))
    _save_state_dict(_make_student(4), str(d / "student_A.pt"))
    _save_state_dict(_make_student(5), str(d / "student_B.pt"))
    _save_quantized(_make_student(4), str(d / "student_A_int8.pt"))
    _save_quantized(_make_student(5), str(d / "student_B_int8.pt"))
    return d


@pytest.fixture(scope="module")
def minimal_ckpt_dir(tmp_path_factory):
    """Create a directory with only the minimum required checkpoints (rows A)."""
    d = tmp_path_factory.mktemp("ckpts_min")
    _save_state_dict(_make_base_qnetwork(1), str(d / "parent_A.pt"))
    _save_state_dict(_make_base_qnetwork(2), str(d / "parent_B.pt"))
    _save_state_dict(_make_base_qnetwork(3), str(d / "child.pt"))
    return d


# ---------------------------------------------------------------------------
# run_comparison_matrix helpers
# ---------------------------------------------------------------------------


class TestDriverHelpers:
    def test_resolve_explicit_wins(self):
        mod = _get_driver()
        result = mod._resolve("explicit.pt", "some/dir", "default.pt")
        assert result == "explicit.pt"

    def test_resolve_uses_directory_when_no_explicit(self):
        mod = _get_driver()
        result = mod._resolve("", "some/dir", "default.pt")
        assert result == os.path.join("some/dir", "default.pt")

    def test_resolve_empty_when_no_directory(self):
        mod = _get_driver()
        assert mod._resolve("", "", "default.pt") == ""

    def test_require_file_raises_for_missing_path(self, tmp_path):
        mod = _get_driver()
        with pytest.raises(ValueError, match="Missing"):
            mod._require_file("", "test_label")

    def test_require_file_raises_for_nonexistent_file(self, tmp_path):
        mod = _get_driver()
        with pytest.raises(FileNotFoundError):
            mod._require_file(str(tmp_path / "noexist.pt"), "test_label")

    def test_require_file_passes_for_existing_file(self, tmp_path):
        mod = _get_driver()
        f = tmp_path / "ok.pt"
        f.write_text("x")
        mod._require_file(str(f), "test_label")  # should not raise

    def test_load_base_qnetwork_returns_module(self, minimal_ckpt_dir):
        mod = _get_driver()
        net = mod._load_base_qnetwork(
            str(minimal_ckpt_dir / "parent_A.pt"), INPUT_DIM, OUTPUT_DIM, HIDDEN
        )
        assert net is not None

    def test_load_student_returns_module(self, ckpt_dir):
        mod = _get_driver()
        net = mod._load_student(
            str(ckpt_dir / "student_A.pt"), INPUT_DIM, OUTPUT_DIM, HIDDEN
        )
        assert net is not None

    def test_load_quantized_returns_module(self, ckpt_dir):
        mod = _get_driver()
        net = mod._load_quantized(str(ckpt_dir / "student_A_int8.pt"))
        assert net is not None

    def test_load_base_qnetwork_raises_for_wrong_path(self, tmp_path):
        mod = _get_driver()
        bad = tmp_path / "bad.pt"
        torch.save("not_a_state_dict", str(bad))
        with pytest.raises(ValueError, match="not a state dict"):
            mod._load_base_qnetwork(str(bad), INPUT_DIM, OUTPUT_DIM, HIDDEN)


class TestExtractRowSummary:
    def _recombination_report(self):
        return {
            "summary": {
                "child_agrees_with_parent_a": 0.75,
                "child_agrees_with_parent_b": 0.65,
                "oracle_agreement": 0.85,
            },
            "comparisons": {
                "child_vs_parent_a": {
                    "kl_divergence": 0.12,
                    "mse": 0.5,
                    "mean_cosine_similarity": 0.92,
                },
                "child_vs_parent_b": {
                    "kl_divergence": 0.18,
                    "mse": 0.7,
                    "mean_cosine_similarity": 0.88,
                },
            },
            "passed": True,
        }

    def _quant_report(self):
        return {
            "fidelity": {
                "action_agreement": 0.99,
                "kl_divergence_float_vs_quant": 0.001,
                "mean_cosine_similarity": 0.998,
            },
            "latency": {
                "float_inference_ms": 0.05,
                "quantized_inference_ms": 0.12,
            },
            "size": {"size_ratio": 0.95},
            "passed": True,
        }

    def test_recombination_row_keys(self):
        mod = _get_driver()
        row = mod._extract_row_summary("TestRow", "recombination", self._recombination_report())
        assert row["row"] == "TestRow"
        assert row["type"] == "recombination"
        assert "child_vs_ref_a_agreement" in row
        assert "oracle_agreement" in row
        assert "kl_ref_a" in row
        assert "cosine_ref_b" in row

    def test_quant_row_keys(self):
        mod = _get_driver()
        row = mod._extract_row_summary("TestQ", "quantized_fidelity", self._quant_report())
        assert row["row"] == "TestQ"
        assert row["type"] == "quantized_fidelity"
        assert "action_agreement" in row
        assert "float_ms" in row
        assert "size_ratio" in row

    def test_oracle_agreement_present(self):
        mod = _get_driver()
        row = mod._extract_row_summary("R", "recombination", self._recombination_report())
        assert row["oracle_agreement"] != "n/a"

    def test_oracle_agreement_absent(self):
        mod = _get_driver()
        report = self._recombination_report()
        del report["summary"]["oracle_agreement"]
        row = mod._extract_row_summary("R", "recombination", report)
        assert row["oracle_agreement"] == "n/a"


class TestWriteSummary:
    def test_creates_md_and_csv(self, tmp_path):
        mod = _get_driver()
        rows = [
            {
                "row": "A", "type": "recombination",
                "child_vs_ref_a_agreement": "0.75", "child_vs_ref_b_agreement": "0.65",
                "oracle_agreement": "0.85", "kl_ref_a": "0.12", "kl_ref_b": "0.18",
                "cosine_ref_a": "0.92", "cosine_ref_b": "0.88", "passed": "True",
            }
        ]
        mod._write_summary(str(tmp_path), rows, n_states=100, input_dim=8, states_source="test")
        assert (tmp_path / "comparison_matrix_summary.md").exists()
        assert (tmp_path / "comparison_matrix_summary.csv").exists()

    def test_md_contains_row_label(self, tmp_path):
        mod = _get_driver()
        rows = [{"row": "MyLabel", "type": "recombination"}]
        mod._write_summary(str(tmp_path), rows, n_states=100, input_dim=8, states_source="synth")
        text = (tmp_path / "comparison_matrix_summary.md").read_text()
        assert "MyLabel" in text


# ---------------------------------------------------------------------------
# run_comparison_matrix integration smoke test
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestRunComparisonMatrixIntegration:
    """Light integration test: rows A + B + C, synthetic states, short latency."""

    def test_row_a_only(self, minimal_ckpt_dir, tmp_path):
        mod = _get_driver()
        states = np.random.default_rng(42).standard_normal((50, INPUT_DIM)).astype("float32")
        from farm.core.decision.training.recombination_eval import RecombinationThresholds
        from farm.core.decision.training.quantize_ptq import QuantizedValidationThresholds

        # Load models
        pa = mod._load_base_qnetwork(str(minimal_ckpt_dir / "parent_A.pt"), INPUT_DIM, OUTPUT_DIM, HIDDEN)
        pb = mod._load_base_qnetwork(str(minimal_ckpt_dir / "parent_B.pt"), INPUT_DIM, OUTPUT_DIM, HIDDEN)
        child = mod._load_base_qnetwork(str(minimal_ckpt_dir / "child.pt"), INPUT_DIM, OUTPUT_DIM, HIDDEN)
        thresholds = RecombinationThresholds(report_only=True)

        row_a = mod._run_recombination_row(
            "A: child vs float parents",
            pa, pb, child,
            states, "synthetic",
            thresholds,
            model_paths={"parent_a": "", "parent_b": "", "child": ""},
            model_formats={"parent_a": mod._FMT_FLOAT, "parent_b": mod._FMT_FLOAT, "child": mod._FMT_FLOAT},
            latency_warmup=0, latency_repeats=3,
        )
        assert "comparisons" in row_a
        assert "child_vs_parent_a" in row_a["comparisons"]
        assert "child_vs_parent_b" in row_a["comparisons"]
        assert row_a["matrix_row"] == "A: child vs float parents"

    def test_full_matrix_abc_via_main(self, ckpt_dir, tmp_path):
        """Smoke test the main() entry point for rows A, B, C."""
        script = str(_DRIVER_SCRIPT)
        argv = [
            script,
            "--parent-a-ckpt", str(ckpt_dir / "parent_A.pt"),
            "--parent-b-ckpt", str(ckpt_dir / "parent_B.pt"),
            "--student-a-ckpt", str(ckpt_dir / "student_A.pt"),
            "--student-b-ckpt", str(ckpt_dir / "student_B.pt"),
            "--student-a-int8", str(ckpt_dir / "student_A_int8.pt"),
            "--student-b-int8", str(ckpt_dir / "student_B_int8.pt"),
            "--child-ckpt", str(ckpt_dir / "child.pt"),
            "--n-states", "50",
            "--seed", "0",
            "--latency-warmup", "0",
            "--latency-repeats", "3",
            "--report-only",
            "--report-dir", str(tmp_path),
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            mod = _get_driver()
            mod.main()
        finally:
            sys.argv = old_argv

        assert (tmp_path / "row_A_child_vs_parents.json").exists()
        assert (tmp_path / "row_B_child_vs_students.json").exists()
        assert (tmp_path / "row_C_child_vs_int8_students.json").exists()
        assert (tmp_path / "row_C_quant_fidelity_A.json").exists()
        assert (tmp_path / "row_C_quant_fidelity_B.json").exists()
        assert (tmp_path / "comparison_matrix_summary.md").exists()
        assert (tmp_path / "comparison_matrix_summary.csv").exists()

    def test_row_a_report_schema(self, ckpt_dir, tmp_path):
        """Verify that row A JSON has the expected schema fields."""
        script = str(_DRIVER_SCRIPT)
        argv = [
            script,
            "--parent-a-ckpt", str(ckpt_dir / "parent_A.pt"),
            "--parent-b-ckpt", str(ckpt_dir / "parent_B.pt"),
            "--child-ckpt", str(ckpt_dir / "child.pt"),
            "--n-states", "30", "--seed", "1",
            "--latency-warmup", "0", "--latency-repeats", "3",
            "--report-only",
            "--report-dir", str(tmp_path / "schema_test"),
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            mod = _get_driver()
            mod.main()
        finally:
            sys.argv = old_argv

        with open(tmp_path / "schema_test" / "row_A_child_vs_parents.json") as fh:
            data = json.load(fh)
        assert "schema_version" in data
        assert "comparisons" in data
        assert "summary" in data
        assert data["matrix_row"] == "A: child vs float parents"


# ---------------------------------------------------------------------------
# summarise_comparison_matrix helpers
# ---------------------------------------------------------------------------


class TestSummaryHelpers:
    def _recombination_report(self):
        return {
            "comparisons": {
                "child_vs_parent_a": {"kl_divergence": 0.12, "mean_cosine_similarity": 0.9},
                "child_vs_parent_b": {"kl_divergence": 0.18, "mean_cosine_similarity": 0.85},
            },
            "summary": {
                "child_agrees_with_parent_a": 0.75,
                "child_agrees_with_parent_b": 0.65,
                "oracle_agreement": 0.85,
            },
            "passed": True,
        }

    def _quant_report(self):
        return {
            "fidelity": {
                "action_agreement": 0.99,
                "kl_divergence_float_vs_quant": 0.001,
                "mean_cosine_similarity": 0.998,
            },
            "latency": {
                "float_inference_ms": 0.05,
                "quantized_inference_ms": 0.12,
            },
            "size": {"size_ratio": 0.95},
            "passed": True,
        }

    def test_detect_type_recombination(self):
        mod = _get_summary()
        assert mod._detect_type(self._recombination_report()) == "recombination"

    def test_detect_type_quantized_fidelity(self):
        mod = _get_summary()
        assert mod._detect_type(self._quant_report()) == "quantized_fidelity"

    def test_detect_type_unknown(self):
        mod = _get_summary()
        assert mod._detect_type({"random_key": 1}) == "unknown"

    def test_extract_recombination_keys(self):
        mod = _get_summary()
        row = mod._extract_recombination("TestLabel", self._recombination_report())
        assert row["label"] == "TestLabel"
        assert row["type"] == "recombination"
        assert "child_vs_ref_a_agreement" in row
        assert "oracle_agreement" in row
        assert "kl_ref_a" in row
        assert "passed" in row

    def test_extract_quant_fidelity_keys(self):
        mod = _get_summary()
        row = mod._extract_quant_fidelity("QLabel", self._quant_report())
        assert row["label"] == "QLabel"
        assert row["type"] == "quantized_fidelity"
        assert "action_agreement" in row
        assert "size_ratio" in row
        assert "passed" in row

    def test_extract_quant_kl_uses_float_vs_quant(self):
        mod = _get_summary()
        row = mod._extract_quant_fidelity("Q", self._quant_report())
        # kl_divergence_float_vs_quant = 0.001
        assert row["kl_divergence"] != "n/a"

    def test_build_markdown_contains_headers(self):
        mod = _get_summary()
        recombi = [mod._extract_recombination("A", self._recombination_report())]
        quant = [mod._extract_quant_fidelity("C_fid", self._quant_report())]
        md = mod._build_markdown(recombi, quant)
        assert "Recombination rows" in md
        assert "Quantized fidelity rows" in md
        assert "A" in md
        assert "C_fid" in md

    def test_build_markdown_empty_quant(self):
        mod = _get_summary()
        recombi = [mod._extract_recombination("A", self._recombination_report())]
        md = mod._build_markdown(recombi, [])
        assert "Quantized fidelity" not in md

    def test_load_reports_skips_bad_json(self, tmp_path):
        mod = _get_summary()
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON")
        results = mod._load_reports(str(tmp_path), [])
        assert all(p != str(bad) for p, _ in results)

    def test_load_reports_deduplicates(self, tmp_path):
        mod = _get_summary()
        good = tmp_path / "good.json"
        good.write_text(json.dumps(self._recombination_report()))
        results = mod._load_reports(str(tmp_path), [str(good)])
        paths = [p for p, _ in results]
        assert len(paths) == len(set(os.path.abspath(p) for p in paths))


class TestSummariseMainIntegration:
    """End-to-end test for summarise_comparison_matrix.main()."""

    def _write_json(self, d, name, content):
        p = d / name
        p.write_text(json.dumps(content))
        return str(p)

    def _recombination_report(self, label="Row A"):
        return {
            "matrix_row": label,
            "comparisons": {
                "child_vs_parent_a": {"kl_divergence": 0.1, "mean_cosine_similarity": 0.9},
                "child_vs_parent_b": {"kl_divergence": 0.2, "mean_cosine_similarity": 0.8},
            },
            "summary": {
                "child_agrees_with_parent_a": 0.7,
                "child_agrees_with_parent_b": 0.6,
                "oracle_agreement": 0.8,
            },
            "passed": True,
        }

    def _quant_report(self, label="Row C_fid"):
        return {
            "matrix_row": label,
            "fidelity": {
                "action_agreement": 0.99,
                "kl_divergence_float_vs_quant": 0.001,
                "mean_cosine_similarity": 0.998,
            },
            "latency": {
                "float_inference_ms": 0.05,
                "quantized_inference_ms": 0.12,
            },
            "size": {"size_ratio": 0.95},
            "passed": True,
        }

    def test_creates_outputs(self, tmp_path):
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        self._write_json(report_dir, "row_A.json", self._recombination_report())
        self._write_json(report_dir, "row_C_fid.json", self._quant_report())

        mod = _get_summary()
        old_argv = sys.argv
        try:
            sys.argv = ["summarise", "--report-dir", str(report_dir)]
            mod.main()
        finally:
            sys.argv = old_argv

        assert (report_dir / "comparison_matrix_summary.md").exists()
        assert (report_dir / "comparison_matrix_summary.csv").exists()

    def test_markdown_has_row_labels(self, tmp_path):
        report_dir = tmp_path / "reports2"
        report_dir.mkdir()
        self._write_json(report_dir, "row_A.json", self._recombination_report("Row A test"))

        mod = _get_summary()
        old_argv = sys.argv
        try:
            sys.argv = ["summarise", "--report-dir", str(report_dir)]
            mod.main()
        finally:
            sys.argv = old_argv

        text = (report_dir / "comparison_matrix_summary.md").read_text()
        assert "Row A test" in text

    def test_custom_output_paths(self, tmp_path):
        report_dir = tmp_path / "reports3"
        report_dir.mkdir()
        self._write_json(report_dir, "row_A.json", self._recombination_report())
        out_md = tmp_path / "custom.md"
        out_csv = tmp_path / "custom.csv"

        mod = _get_summary()
        old_argv = sys.argv
        try:
            sys.argv = [
                "summarise",
                "--report-dir", str(report_dir),
                "--output-md", str(out_md),
                "--output-csv", str(out_csv),
            ]
            mod.main()
        finally:
            sys.argv = old_argv

        assert out_md.exists()
        assert out_csv.exists()
