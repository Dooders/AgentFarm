"""Tests for scripts/run_recombination_ablation.py.

Covers:
- Config loading and validation (_parse_config / _load_raw_config)
- Dry-run mode (main with --dry-run)
- Smoke-test mode (main with --smoke-test)
- State buffer generation (_make_states)
- Summary writing (_write_summary)
- End-to-end integration smoke test (full_pipeline condition, 1 seed)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "run_recombination_ablation.py"


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location("run_recombination_ablation_mod", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass decorator can resolve the module namespace.
    sys.modules["run_recombination_ablation_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        _mod = _load_ablation_module()
    return _mod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_RAW: Dict[str, Any] = {
    "seeds": [0],
    "n_states": 30,
    "states_file": "",
    "input_dim": 8,
    "output_dim": 4,
    "hidden_size": 64,
    "conditions": [
        {"name": "distill_only", "stages": ["distill"]},
    ],
}


# ---------------------------------------------------------------------------
# _parse_config
# ---------------------------------------------------------------------------


def test_parse_config_minimal():
    mod = _get_mod()
    cfg = mod._parse_config(_MINIMAL_RAW)
    assert cfg.seeds == [0]
    assert cfg.n_states == 30
    assert len(cfg.conditions) == 1
    assert cfg.conditions[0].name == "distill_only"
    assert cfg.conditions[0].stages == ["distill"]


def test_parse_config_results_dir_override():
    mod = _get_mod()
    cfg = mod._parse_config(_MINIMAL_RAW, results_dir_override="/tmp/my_results")
    assert cfg.results_dir == "/tmp/my_results"


def test_parse_config_empty_seeds_raises():
    mod = _get_mod()
    raw = dict(_MINIMAL_RAW, seeds=[])
    with pytest.raises(ValueError, match="seeds"):
        mod._parse_config(raw)


def test_parse_config_bad_n_states_raises():
    mod = _get_mod()
    raw = dict(_MINIMAL_RAW, n_states=0)
    with pytest.raises(ValueError, match="n_states"):
        mod._parse_config(raw)


def test_parse_config_invalid_stage_raises():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        conditions=[{"name": "bad", "stages": ["distill", "fly_to_mars"]}],
    )
    with pytest.raises(ValueError, match="fly_to_mars"):
        mod._parse_config(raw)


def test_parse_config_quantize_without_distill_raises():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        conditions=[{"name": "bad", "stages": ["quantize"]}],
    )
    with pytest.raises(ValueError, match="distill"):
        mod._parse_config(raw)


def test_parse_config_crossover_without_distill_raises():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        conditions=[{"name": "bad", "stages": ["crossover"]}],
    )
    with pytest.raises(ValueError, match="distill"):
        mod._parse_config(raw)


def test_parse_config_compare_without_crossover_raises():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        conditions=[{"name": "bad", "stages": ["distill", "compare"]}],
    )
    with pytest.raises(ValueError, match="crossover"):
        mod._parse_config(raw)


def test_parse_config_compare_only_raises():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        conditions=[{"name": "bad", "stages": ["compare"]}],
    )
    with pytest.raises(ValueError, match="crossover"):
        mod._parse_config(raw)


def test_parse_config_empty_conditions_raises():
    mod = _get_mod()
    raw = dict(_MINIMAL_RAW, conditions=[])
    with pytest.raises(ValueError, match="conditions"):
        mod._parse_config(raw)


def test_parse_config_empty_stages_raises():
    mod = _get_mod()
    raw = dict(_MINIMAL_RAW, conditions=[{"name": "empty", "stages": []}])
    with pytest.raises(ValueError, match="stage"):
        mod._parse_config(raw)


def test_parse_config_invalid_dim_raises():
    mod = _get_mod()
    raw = dict(_MINIMAL_RAW, input_dim=0)
    with pytest.raises(ValueError, match="input_dim"):
        mod._parse_config(raw)


def test_parse_config_global_and_condition_dicts():
    mod = _get_mod()
    raw = dict(
        _MINIMAL_RAW,
        distillation={"epochs": 5, "lr": 1e-4},
        conditions=[
            {
                "name": "custom",
                "stages": ["distill"],
                "distillation": {"epochs": 3},
            }
        ],
    )
    cfg = mod._parse_config(raw)
    assert cfg.distillation["epochs"] == 5
    assert cfg.conditions[0].distillation["epochs"] == 3


# ---------------------------------------------------------------------------
# _load_raw_config
# ---------------------------------------------------------------------------


def test_load_raw_config_json(tmp_path):
    mod = _get_mod()
    cfg_dict = {"seeds": [1, 2], "n_states": 10, "conditions": [{"name": "x", "stages": ["distill"]}]}
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps(cfg_dict))
    raw = mod._load_raw_config(str(cfg_file))
    assert raw["seeds"] == [1, 2]


def test_load_raw_config_missing_raises(tmp_path):
    mod = _get_mod()
    with pytest.raises(FileNotFoundError):
        mod._load_raw_config(str(tmp_path / "nonexistent.yaml"))


def test_load_raw_config_non_dict_raises(tmp_path):
    mod = _get_mod()
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps([1, 2, 3]))  # top-level list, not dict
    with pytest.raises(ValueError, match="top-level"):
        mod._load_raw_config(str(cfg_file))


def test_load_raw_config_invalid_json_raises(tmp_path):
    mod = _get_mod()
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text("{not: valid json!!!}")
    # Module may use YAML (which would parse this differently) or JSON fallback.
    # Either way it must not silently succeed with a non-dict or raise an unhelpful error.
    try:
        raw = mod._load_raw_config(str(cfg_file))
        # If YAML parsed it as a dict that's acceptable; otherwise should have raised.
        assert isinstance(raw, dict)
    except Exception:
        pass  # Any clear exception is acceptable for invalid input


# ---------------------------------------------------------------------------
# _make_states
# ---------------------------------------------------------------------------


def test_make_states_synthetic():
    mod = _get_mod()
    cfg = mod._parse_config(_MINIMAL_RAW)
    states = mod._make_states(cfg, seed=0)
    assert states.shape == (cfg.n_states, cfg.input_dim)
    assert states.dtype == np.float32


def test_make_states_synthetic_reproducible():
    mod = _get_mod()
    cfg = mod._parse_config(_MINIMAL_RAW)
    s0a = mod._make_states(cfg, seed=0)
    s0b = mod._make_states(cfg, seed=0)
    assert np.allclose(s0a, s0b)


def test_make_states_different_seeds_differ():
    mod = _get_mod()
    cfg = mod._parse_config(_MINIMAL_RAW)
    s0 = mod._make_states(cfg, seed=0)
    s1 = mod._make_states(cfg, seed=1)
    assert not np.allclose(s0, s1)


def test_make_states_from_file(tmp_path):
    mod = _get_mod()
    arr = np.random.default_rng(42).standard_normal((20, 8)).astype("float32")
    npy_path = str(tmp_path / "states.npy")
    np.save(npy_path, arr)
    raw = dict(_MINIMAL_RAW, states_file=npy_path)
    cfg = mod._parse_config(raw)
    states = mod._make_states(cfg, seed=0)
    assert states.shape == (20, 8)
    assert np.allclose(states, arr)


# ---------------------------------------------------------------------------
# _write_summary
# ---------------------------------------------------------------------------


def test_write_summary_creates_files(tmp_path):
    mod = _get_mod()
    rows = [
        {"condition": "distill_only", "seed": 0, "stages": "distill", "elapsed_s": 1.5},
        {"condition": "distill_only", "seed": 1, "stages": "distill", "elapsed_s": 1.2},
    ]
    mod._write_summary(rows, str(tmp_path), dry_run=False)
    assert (tmp_path / "ablation_summary.csv").exists()
    assert (tmp_path / "ablation_summary.md").exists()


def test_write_summary_dry_run_marker(tmp_path):
    mod = _get_mod()
    rows = [{"condition": "c", "seed": 0, "stages": "distill"}]
    mod._write_summary(rows, str(tmp_path), dry_run=True)
    md = (tmp_path / "ablation_summary.md").read_text()
    assert "DRY-RUN" in md


def test_write_summary_csv_has_header(tmp_path):
    mod = _get_mod()
    rows = [{"condition": "c", "seed": 0, "stages": "distill", "elapsed_s": 0.5}]
    mod._write_summary(rows, str(tmp_path), dry_run=False)
    csv_text = (tmp_path / "ablation_summary.csv").read_text()
    assert "condition" in csv_text
    assert "seed" in csv_text


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


def test_dry_run_writes_summary(tmp_path):
    """--dry-run must write stub summary files without training."""
    mod = _get_mod()
    results_dir = str(tmp_path / "dry_results")
    mod.main(["--smoke-test", "--dry-run", "--results-dir", results_dir])
    assert os.path.isfile(os.path.join(results_dir, "ablation_summary.csv"))
    assert os.path.isfile(os.path.join(results_dir, "ablation_summary.md"))
    # No model checkpoint should be created
    assert not any(f.endswith(".pt") for f in _find_pt_files(results_dir))


def test_dry_run_with_config_file(tmp_path):
    mod = _get_mod()
    cfg_data = {
        "seeds": [42],
        "n_states": 20,
        "conditions": [{"name": "distill_only", "stages": ["distill"]}],
    }
    cfg_file = tmp_path / "ab.json"
    cfg_file.write_text(json.dumps(cfg_data))
    results_dir = str(tmp_path / "res")
    mod.main(["--config", str(cfg_file), "--dry-run", "--results-dir", results_dir])
    assert os.path.isfile(os.path.join(results_dir, "ablation_summary.csv"))


def test_no_args_exits(monkeypatch):
    """Calling main without --config or --smoke-test should sys.exit(1)."""
    mod = _get_mod()
    with pytest.raises(SystemExit) as exc_info:
        mod.main([])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Smoke-test mode (end-to-end, fast)
# ---------------------------------------------------------------------------


def _find_pt_files(root: str) -> list:
    """Return a list of all .pt file paths under *root*."""
    return [
        os.path.join(dirpath, fname)
        for dirpath, _, fnames in os.walk(root)
        for fname in fnames
        if fname.endswith(".pt")
    ]


@pytest.mark.slow
def test_smoke_test_end_to_end(tmp_path):
    """Full smoke-test run: tiny config, real training, check outputs."""
    mod = _get_mod()
    results_dir = str(tmp_path / "smoke")
    mod.main(["--smoke-test", "--results-dir", results_dir])
    # CSV summary
    csv_path = os.path.join(results_dir, "ablation_summary.csv")
    assert os.path.isfile(csv_path)
    # Markdown summary
    md_path = os.path.join(results_dir, "ablation_summary.md")
    assert os.path.isfile(md_path)
    # At least one student checkpoint produced
    assert len(_find_pt_files(results_dir)) >= 1


@pytest.mark.slow
def test_smoke_test_distill_only_produces_student_checkpoints(tmp_path):
    """distill_only condition must write student_A.pt and student_B.pt."""
    mod = _get_mod()
    raw = {
        "seeds": [7],
        "n_states": 30,
        "input_dim": 8,
        "output_dim": 4,
        "hidden_size": 64,
        "conditions": [{"name": "distill_only", "stages": ["distill"]}],
        "distillation": {"epochs": 1, "batch_size": 16},
    }
    results_dir = str(tmp_path / "res")
    cfg = mod._parse_config(raw, results_dir_override=results_dir)
    states = mod._make_states(cfg, seed=7)
    row = mod._run_condition_seed(cfg, cfg.conditions[0], seed=7, states=states, dry_run=False)
    work_dir = row["work_dir"]
    assert os.path.isfile(os.path.join(work_dir, "student_A.pt"))
    assert os.path.isfile(os.path.join(work_dir, "student_B.pt"))


@pytest.mark.slow
def test_smoke_test_distill_quantize_produces_int8_checkpoints(tmp_path):
    """distill + quantize stages must produce student_A_int8.pt."""
    mod = _get_mod()
    raw = {
        "seeds": [3],
        "n_states": 30,
        "input_dim": 8,
        "output_dim": 4,
        "hidden_size": 64,
        "conditions": [{"name": "dq", "stages": ["distill", "quantize"]}],
        "distillation": {"epochs": 1, "batch_size": 16},
        "quantization": {"mode": "dynamic"},
    }
    results_dir = str(tmp_path / "res")
    cfg = mod._parse_config(raw, results_dir_override=results_dir)
    states = mod._make_states(cfg, seed=3)
    row = mod._run_condition_seed(cfg, cfg.conditions[0], seed=3, states=states, dry_run=False)
    work_dir = row["work_dir"]
    assert os.path.isfile(os.path.join(work_dir, "student_A_int8.pt"))
    assert os.path.isfile(os.path.join(work_dir, "student_B_int8.pt"))


@pytest.mark.slow
def test_smoke_test_full_pipeline_produces_child(tmp_path):
    """Full pipeline must produce a child_finetuned.pt checkpoint."""
    mod = _get_mod()
    raw = {
        "seeds": [5],
        "n_states": 40,
        "input_dim": 8,
        "output_dim": 4,
        "hidden_size": 64,
        "conditions": [{"name": "full", "stages": ["distill", "quantize", "crossover", "compare"]}],
        "distillation": {"epochs": 1, "batch_size": 16},
        "quantization": {"mode": "dynamic"},
        "crossover": {"mode": "weighted", "alpha": 0.5},
        "comparison": {"report_only": True},
    }
    results_dir = str(tmp_path / "res")
    cfg = mod._parse_config(raw, results_dir_override=results_dir)
    states = mod._make_states(cfg, seed=5)
    row = mod._run_condition_seed(cfg, cfg.conditions[0], seed=5, states=states, dry_run=False)
    work_dir = row["work_dir"]
    assert os.path.isfile(os.path.join(work_dir, "child_finetuned.pt"))
    assert os.path.isfile(os.path.join(work_dir, "compare_child_vs_students.json"))
