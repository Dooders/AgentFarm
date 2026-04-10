#!/usr/bin/env python3
"""Unified publication-style ablation runner for recombination experiments.

This script orchestrates a full ablation study across multiple conditions
(crossover-only, distill-only, quantize-only, or the full pipeline), a list
of random seeds, and optional shared state buffers.  All results land in a
single ``results/`` tree together with CSV and Markdown summary tables suitable
for paper tables and CI regression.

How to run
----------
::

    # Dry-run (validate config, print plan, no training)
    python scripts/run_recombination_ablation.py --config ablation.yaml --dry-run

    # Tiny synthetic run for smoke-testing (no config file needed)
    python scripts/run_recombination_ablation.py --smoke-test --results-dir /tmp/ablation_smoke

    # Full run from a config file
    python scripts/run_recombination_ablation.py --config ablation.yaml

Config format
-------------
YAML (recommended) or JSON.  Example::

    seeds: [0, 1, 2]
    n_states: 2000
    states_file: ""          # leave empty to synthesise from seed
    input_dim: 8
    output_dim: 4
    hidden_size: 64
    results_dir: results/ablation
    conditions:
      - name: distill_only
        stages: [distill]
      - name: distill_quantize
        stages: [distill, quantize]
      - name: full_pipeline
        stages: [distill, quantize, crossover, compare]
    distillation:
      epochs: 10
      temperature: 3.0
      alpha: 1.0
      lr: 0.001
      batch_size: 32
    quantization:
      mode: dynamic
    crossover:
      mode: weighted
      alpha: 0.5
    comparison:
      report_only: true

Valid stages
------------
``distill``, ``quantize``, ``crossover``, ``compare``

``compare`` must include ``crossover`` (it evaluates a crossover child).  When
``quantize`` is present, crossover blends **dequantized** weights extracted from
the int8 checkpoints; ``compare`` loads the same int8 modules as parent
references when reporting agreement against the float fine-tuned child.

Each stage writes its outputs into a per-seed sub-directory under
``<results_dir>/<condition_name>/seed_<seed>/``.

The final summary CSV and Markdown table are written to
``<results_dir>/ablation_summary.csv`` and
``<results_dir>/ablation_summary.md``.

Dry-run / smoke-test mode
--------------------------
Pass ``--dry-run`` to validate the config and print the full execution plan
without running any training.  Pass ``--smoke-test`` to run a tiny end-to-end
exercise using 50 synthetic states and 2 epochs with no checkpoints required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root without pip install -e .
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# Optional YAML support (falls back to JSON if PyYAML is absent)
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ModuleNotFoundError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class _ConditionConfig:
    name: str
    stages: List[str]  # subset of ["distill", "quantize", "crossover", "compare"]
    # Per-condition overrides (merged on top of global settings)
    distillation: Dict[str, Any] = field(default_factory=dict)
    quantization: Dict[str, Any] = field(default_factory=dict)
    crossover: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _AblationConfig:
    seeds: List[int]
    n_states: int
    states_file: str
    input_dim: int
    output_dim: int
    hidden_size: int
    results_dir: str
    conditions: List[_ConditionConfig]
    # Global stage defaults
    distillation: Dict[str, Any] = field(default_factory=dict)
    quantization: Dict[str, Any] = field(default_factory=dict)
    crossover: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)


_VALID_STAGES = {"distill", "quantize", "crossover", "compare"}

_SMOKE_CONFIG_DICT: Dict[str, Any] = {
    "seeds": [0, 1],
    "n_states": 50,
    "states_file": "",
    "input_dim": 8,
    "output_dim": 4,
    "hidden_size": 64,
    "conditions": [
        {"name": "distill_only", "stages": ["distill"]},
        {"name": "distill_quantize", "stages": ["distill", "quantize"]},
        {"name": "full_pipeline", "stages": ["distill", "quantize", "crossover", "compare"]},
    ],
    "distillation": {"epochs": 2, "temperature": 3.0, "alpha": 1.0, "lr": 1e-3, "batch_size": 16},
    "quantization": {"mode": "dynamic"},
    "crossover": {"mode": "weighted", "alpha": 0.5},
    "comparison": {"report_only": True},
}


# ---------------------------------------------------------------------------
# Config loading / validation
# ---------------------------------------------------------------------------


def _load_raw_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file and return a raw dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    # Try YAML first (superset of JSON).
    if _YAML_AVAILABLE:
        try:
            raw = _yaml.safe_load(content)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse config file '{path}' as YAML. "
                "Expected a YAML or JSON object at the top level."
            ) from exc
    else:
        # Fallback to JSON.
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse config file '{path}' as JSON. "
                "Expected a YAML or JSON object at the top level."
            ) from exc

    if not isinstance(raw, dict):
        parsed_type = type(raw).__name__
        raise ValueError(
            f"Config file '{path}' must contain a top-level YAML/JSON object, "
            f"got {parsed_type}."
        )

    return raw


def _parse_config(raw: Dict[str, Any], results_dir_override: str = "") -> _AblationConfig:
    """Parse and validate a raw config dict into an ``_AblationConfig``."""
    seeds = list(raw.get("seeds", [0]))
    if not seeds:
        raise ValueError("'seeds' must be a non-empty list.")
    if not all(isinstance(s, int) for s in seeds):
        raise ValueError("All values in 'seeds' must be integers.")

    n_states = int(raw.get("n_states", 500))
    if n_states < 1:
        raise ValueError(f"'n_states' must be >= 1, got {n_states}.")

    states_file = str(raw.get("states_file", "") or "")
    input_dim = int(raw.get("input_dim", 8))
    output_dim = int(raw.get("output_dim", 4))
    hidden_size = int(raw.get("hidden_size", 64))
    if input_dim < 1 or output_dim < 1 or hidden_size < 1:
        raise ValueError("input_dim, output_dim, hidden_size must all be >= 1.")

    results_dir = results_dir_override or str(raw.get("results_dir", "results/ablation"))

    raw_conditions = raw.get("conditions", [])
    if not raw_conditions:
        raise ValueError("'conditions' must be a non-empty list.")

    conditions: List[_ConditionConfig] = []
    for i, rc in enumerate(raw_conditions):
        name = str(rc.get("name", f"condition_{i}"))
        stages_raw = list(rc.get("stages", []))
        if not stages_raw:
            raise ValueError(f"Condition '{name}' must declare at least one stage.")
        invalid = set(stages_raw) - _VALID_STAGES
        if invalid:
            raise ValueError(
                f"Condition '{name}' has unknown stages: {sorted(invalid)}. "
                f"Valid stages are: {sorted(_VALID_STAGES)}."
            )
        stages = set(stages_raw)
        if "quantize" in stages and "distill" not in stages:
            raise ValueError(
                f"Condition '{name}' includes 'quantize' but is missing required "
                "'distill' stage."
            )
        if "crossover" in stages and "distill" not in stages:
            raise ValueError(
                f"Condition '{name}' includes 'crossover' but is missing required "
                "'distill' stage."
            )
        if "compare" in stages and "crossover" not in stages:
            raise ValueError(
                f"Condition '{name}' includes 'compare' but is missing required "
                "'crossover' stage (compare evaluates a crossover child)."
            )
        conditions.append(
            _ConditionConfig(
                name=name,
                stages=stages_raw,
                distillation=dict(rc.get("distillation", {})),
                quantization=dict(rc.get("quantization", {})),
                crossover=dict(rc.get("crossover", {})),
                comparison=dict(rc.get("comparison", {})),
            )
        )

    return _AblationConfig(
        seeds=seeds,
        n_states=n_states,
        states_file=states_file,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
        results_dir=results_dir,
        conditions=conditions,
        distillation=dict(raw.get("distillation", {})),
        quantization=dict(raw.get("quantization", {})),
        crossover=dict(raw.get("crossover", {})),
        comparison=dict(raw.get("comparison", {})),
    )


# ---------------------------------------------------------------------------
# States generation
# ---------------------------------------------------------------------------


def _make_states(cfg: _AblationConfig, seed: int) -> np.ndarray:
    """Return the shared state buffer for *seed*.

    If ``cfg.states_file`` is set the file is loaded (every seed uses the same
    file).  Otherwise ``cfg.n_states`` synthetic states are drawn from a seeded
    standard-normal distribution.
    """
    if cfg.states_file:
        states = np.load(cfg.states_file).astype("float32")
        if states.ndim != 2 or states.shape[1] != cfg.input_dim:
            raise ValueError(
                f"States file {cfg.states_file!r} has unexpected shape {states.shape}; "
                f"expected (N, {cfg.input_dim})."
            )
        return states
    rng = np.random.default_rng(seed)
    return rng.standard_normal((cfg.n_states, cfg.input_dim)).astype("float32")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _merged(global_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_dict)
    merged.update(override_dict)
    return merged


def _run_distill_stage(
    cfg: _AblationConfig,
    cond: _ConditionConfig,
    seed: int,
    states: np.ndarray,
    work_dir: str,
) -> Dict[str, str]:
    """Run distillation for both pairs and return a dict of checkpoint paths."""
    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
    from farm.core.decision.training.trainer_distill import DistillationConfig, DistillationTrainer

    dcfg_raw = _merged(cfg.distillation, cond.distillation)
    dist_cfg = DistillationConfig(
        temperature=float(dcfg_raw.get("temperature", 3.0)),
        alpha=float(dcfg_raw.get("alpha", 1.0)),
        learning_rate=float(dcfg_raw.get("lr", 1e-3)),
        epochs=int(dcfg_raw.get("epochs", 10)),
        batch_size=int(dcfg_raw.get("batch_size", 32)),
        max_grad_norm=float(dcfg_raw.get("max_grad_norm", 1.0)),
        val_fraction=float(dcfg_raw.get("val_fraction", 0.1)),
        loss_fn=str(dcfg_raw.get("loss_fn", "kl")),
        seed=seed,
    )

    ckpts: Dict[str, str] = {}
    _pair_offset = {"A": 0, "B": 1}

    import torch

    for pair in ("A", "B"):
        torch.manual_seed(seed + _pair_offset[pair])
        teacher = BaseQNetwork(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            hidden_size=cfg.hidden_size,
        )
        student = StudentQNetwork(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            parent_hidden_size=cfg.hidden_size,
        )
        ckpt_path = os.path.join(work_dir, f"student_{pair}.pt")
        trainer = DistillationTrainer(teacher, student, dist_cfg)
        trainer.train(states, checkpoint_path=ckpt_path)
        ckpts[pair] = ckpt_path
        print(f"    [distill] pair {pair} -> {ckpt_path}")
    return ckpts


def _run_quantize_stage(
    cfg: _AblationConfig,
    cond: _ConditionConfig,
    student_ckpts: Dict[str, str],
    states: np.ndarray,
    work_dir: str,
) -> Dict[str, str]:
    """Quantize both students and return a dict of int8 checkpoint paths."""
    import torch
    from farm.core.decision.base_dqn import StudentQNetwork
    from farm.core.decision.training.quantize_ptq import PostTrainingQuantizer, QuantizationConfig

    qcfg_raw = _merged(cfg.quantization, cond.quantization)
    q_config = QuantizationConfig(
        mode=str(qcfg_raw.get("mode", "dynamic")),
    )

    int8_ckpts: Dict[str, str] = {}
    for pair, ckpt_path in student_ckpts.items():
        float_student = StudentQNetwork(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            parent_hidden_size=cfg.hidden_size,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        float_student.load_state_dict(state)
        float_student.eval()

        quantizer = PostTrainingQuantizer(q_config)
        q_model, q_result = quantizer.quantize(float_student, calibration_states=states)
        int8_path = os.path.join(work_dir, f"student_{pair}_int8.pt")
        quantizer.save_checkpoint(q_model, int8_path, q_result)
        int8_ckpts[pair] = int8_path
        print(f"    [quantize] pair {pair} -> {int8_path}")
    return int8_ckpts


def _run_crossover_stage(
    cfg: _AblationConfig,
    cond: _ConditionConfig,
    seed: int,
    student_ckpts: Dict[str, str],
    int8_ckpts: Dict[str, str],
    states: np.ndarray,
    work_dir: str,
) -> str:
    """Run crossover + fine-tuning and return the child checkpoint path.

    When ``quantize`` is in the condition's stages and both int8 checkpoints
    exist, parents are loaded as dynamic PTQ modules and converted to float
    parameter dicts (same helper as ``crossover`` training code) before
    :func:`crossover_quantized_state_dict` blends them.  Otherwise float
    ``student_*.pt`` checkpoints from distillation are blended.

    Fine-tuning still uses the float distilled student A as the soft-label
    teacher so KD targets stay in full precision.
    """
    import torch
    from farm.core.decision.base_dqn import StudentQNetwork
    from farm.core.decision.training.crossover import (
        _float_state_dict_from_dynamic_quantized_module,
        crossover_quantized_state_dict,
    )
    from farm.core.decision.training.finetune import FineTuner, FineTuningConfig
    from farm.core.decision.training.quantize_ptq import load_quantized_checkpoint

    xcfg_raw = _merged(cfg.crossover, cond.crossover)
    dcfg_raw = _merged(cfg.distillation, cond.distillation)

    def _load_float_student(path: str) -> StudentQNetwork:
        net = StudentQNetwork(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            parent_hidden_size=cfg.hidden_size,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        return net

    parent_a_path = student_ckpts.get("A", "")
    parent_b_path = student_ckpts.get("B", "")

    if not parent_a_path or not parent_b_path:
        raise RuntimeError("crossover stage requires 'distill' stage to run first.")

    parent_a_float = _load_float_student(parent_a_path)

    use_quantized_parents = (
        "quantize" in cond.stages
        and bool(int8_ckpts.get("A"))
        and bool(int8_ckpts.get("B"))
    )
    if "quantize" in cond.stages and "crossover" in cond.stages and not use_quantized_parents:
        raise RuntimeError(
            f"Condition {cond.name!r}: crossover after quantize requires both "
            "student_A_int8.pt and student_B_int8.pt; quantize stage did not produce them."
        )

    if use_quantized_parents:
        parent_a_q, _ = load_quantized_checkpoint(int8_ckpts["A"])
        parent_b_q, _ = load_quantized_checkpoint(int8_ckpts["B"])
        # Dynamic PTQ modules use packed params; extract float weights for crossover.
        sd_a = _float_state_dict_from_dynamic_quantized_module(parent_a_q)
        sd_b = _float_state_dict_from_dynamic_quantized_module(parent_b_q)
    else:
        parent_b_float = _load_float_student(parent_b_path)
        sd_a = parent_a_float.state_dict()
        sd_b = parent_b_float.state_dict()

    # Crossover
    crossover_mode = str(xcfg_raw.get("mode", "weighted"))
    crossover_alpha = float(xcfg_raw.get("alpha", 0.5))
    crossover_seed = int(xcfg_raw.get("seed", seed))

    child_state = crossover_quantized_state_dict(
        sd_a,
        sd_b,
        mode=crossover_mode,
        alpha=crossover_alpha,
        seed=crossover_seed,
    )
    child = StudentQNetwork(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        parent_hidden_size=cfg.hidden_size,
    )
    child.load_state_dict(child_state)

    # Fine-tune child against parent A acting as soft-label teacher.
    ft_config = FineTuningConfig(
        learning_rate=float(dcfg_raw.get("lr", 1e-3)),
        epochs=int(dcfg_raw.get("epochs", 10)),
        batch_size=int(dcfg_raw.get("batch_size", 32)),
        max_grad_norm=float(dcfg_raw.get("max_grad_norm", 1.0)),
        val_fraction=float(dcfg_raw.get("val_fraction", 0.1)),
        loss_fn=str(dcfg_raw.get("loss_fn", "kl")),
        temperature=float(dcfg_raw.get("temperature", 3.0)),
        alpha=float(dcfg_raw.get("alpha", 1.0)),
        seed=seed,
    )
    # FineTuner(reference, child, config) — float parent A as soft-label teacher.
    tuner = FineTuner(parent_a_float, child, ft_config)
    child_path = os.path.join(work_dir, "child_finetuned.pt")
    tuner.finetune(states, checkpoint_path=child_path)
    print(f"    [crossover] child -> {child_path}")
    return child_path


def _run_compare_stage(
    cfg: _AblationConfig,
    cond: _ConditionConfig,
    student_ckpts: Dict[str, str],
    int8_ckpts: Dict[str, str],
    child_path: str,
    states: np.ndarray,
    work_dir: str,
) -> Dict[str, Any]:
    """Run comparison matrix and return a summary dict.

    Parents are loaded from int8 checkpoints when ``quantize`` ran and both
    ``student_*_int8.pt`` paths exist; otherwise float ``student_*.pt`` parents
    are used.  The child is always the fine-tuned float checkpoint from
    crossover.
    """
    import torch
    from farm.core.decision.base_dqn import StudentQNetwork
    from farm.core.decision.training.quantize_ptq import load_quantized_checkpoint
    from farm.core.decision.training.recombination_eval import (
        RecombinationEvaluator,
        RecombinationThresholds,
    )

    if not child_path:
        raise RuntimeError(
            "compare stage requires a crossover child checkpoint (run crossover first)."
        )
    if not student_ckpts.get("A") or not student_ckpts.get("B"):
        raise RuntimeError("compare requires distilled student_A.pt and student_B.pt.")

    cmpcfg_raw = _merged(cfg.comparison, cond.comparison)
    report_only = bool(cmpcfg_raw.get("report_only", True))

    thresholds = RecombinationThresholds(
        min_action_agreement=float(cmpcfg_raw.get("min_action_agreement", 0.70)),
        max_kl_divergence=float(cmpcfg_raw.get("max_kl_divergence", 1.0)),
        max_mse=float(cmpcfg_raw.get("max_mse", 5.0)),
        min_cosine_similarity=float(cmpcfg_raw.get("min_cosine_similarity", 0.70)),
        report_only=report_only,
    )

    def _load_student_net(path: str) -> StudentQNetwork:
        net = StudentQNetwork(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            parent_hidden_size=cfg.hidden_size,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        return net

    use_quantized_refs = (
        "quantize" in cond.stages
        and bool(int8_ckpts.get("A"))
        and bool(int8_ckpts.get("B"))
    )
    if "quantize" in cond.stages and not use_quantized_refs:
        raise RuntimeError(
            f"Condition {cond.name!r}: compare after quantize requires int8 checkpoints "
            "for both students."
        )

    child_net = _load_student_net(child_path)
    if use_quantized_refs:
        ref_a, _ = load_quantized_checkpoint(int8_ckpts["A"])
        ref_b, _ = load_quantized_checkpoint(int8_ckpts["B"])
        model_paths = {
            "parent_a": int8_ckpts["A"],
            "parent_b": int8_ckpts["B"],
            "child": child_path,
        }
        model_formats = {
            "parent_a": "quantized_full_model",
            "parent_b": "quantized_full_model",
            "child": "float_state_dict",
        }
    else:
        ref_a = _load_student_net(student_ckpts["A"])
        ref_b = _load_student_net(student_ckpts["B"])
        model_paths = {
            "parent_a": student_ckpts["A"],
            "parent_b": student_ckpts["B"],
            "child": child_path,
        }
        model_formats = {
            "parent_a": "float_state_dict",
            "parent_b": "float_state_dict",
            "child": "float_state_dict",
        }

    evaluator = RecombinationEvaluator(ref_a, ref_b, child_net, thresholds)
    report = evaluator.evaluate(states, model_paths=model_paths, model_formats=model_formats)
    report_dict = report.to_dict()
    report_path = os.path.join(work_dir, "compare_child_vs_students.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)
    summary: Dict[str, Any] = {"child_vs_students": report_dict.get("summary", {})}
    print(f"    [compare] child vs students -> {report_path}")
    return summary


# ---------------------------------------------------------------------------
# Per-seed condition runner
# ---------------------------------------------------------------------------


def _run_condition_seed(
    cfg: _AblationConfig,
    cond: _ConditionConfig,
    seed: int,
    states: np.ndarray,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run all stages for one (condition, seed) pair.

    Returns a flat metrics dict for the summary table.
    """
    work_dir = os.path.join(cfg.results_dir, cond.name, f"seed_{seed}")
    if not dry_run:
        os.makedirs(work_dir, exist_ok=True)

    row: Dict[str, Any] = {
        "condition": cond.name,
        "seed": seed,
        "stages": ",".join(cond.stages),
        "work_dir": work_dir,
    }

    if dry_run:
        return row

    student_ckpts: Dict[str, str] = {}
    int8_ckpts: Dict[str, str] = {}
    child_path: str = ""

    t_start = time.perf_counter()

    if "distill" in cond.stages:
        student_ckpts = _run_distill_stage(cfg, cond, seed, states, work_dir)

    if "quantize" in cond.stages:
        if not student_ckpts:
            print(
                f"    [quantize] WARNING: no student checkpoints for condition "
                f"'{cond.name}' seed {seed} – skipping quantize stage."
            )
        else:
            int8_ckpts = _run_quantize_stage(cfg, cond, student_ckpts, states, work_dir)

    if "crossover" in cond.stages:
        child_path = _run_crossover_stage(
            cfg, cond, seed, student_ckpts, int8_ckpts, states, work_dir
        )

    if "compare" in cond.stages:
        compare_summary = _run_compare_stage(
            cfg, cond, student_ckpts, int8_ckpts, child_path, states, work_dir
        )
        cvs = compare_summary.get("child_vs_students", {})
        row["child_vs_ref_a_agreement"] = cvs.get("child_agrees_with_parent_a", "n/a")
        row["child_vs_ref_b_agreement"] = cvs.get("child_agrees_with_parent_b", "n/a")
        row["oracle_agreement"] = cvs.get("oracle_agreement", "n/a")

    row["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    return row


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------


_SUMMARY_COLUMNS = [
    "condition",
    "seed",
    "stages",
    "child_vs_ref_a_agreement",
    "child_vs_ref_b_agreement",
    "oracle_agreement",
    "elapsed_s",
    "work_dir",
]


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_summary(rows: List[Dict[str, Any]], results_dir: str, dry_run: bool) -> None:
    """Write CSV and Markdown summary tables under *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "ablation_summary.csv")
    md_path = os.path.join(results_dir, "ablation_summary.md")

    # Collect all keys (some rows may have extra keys)
    all_keys: List[str] = list(_SUMMARY_COLUMNS)
    seen = set(all_keys)
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    prefix = "[DRY-RUN] " if dry_run else ""
    md_lines = [
        "# Recombination Ablation Summary",
        "",
        f"> {prefix}Generated by `scripts/run_recombination_ablation.py`.",
        "",
        "## Results",
        "",
    ]
    header = " | ".join(_SUMMARY_COLUMNS)
    sep = " | ".join(["---"] * len(_SUMMARY_COLUMNS))
    md_lines.append(f"| {header} |")
    md_lines.append(f"| {sep} |")
    for r in rows:
        cells = " | ".join(_fmt_cell(r.get(c)) for c in _SUMMARY_COLUMNS)
        md_lines.append(f"| {cells} |")

    md_lines += [
        "",
        "## Notes",
        "",
        "- When `states_file` is set, all seeds share the same state buffer.",
        "  When omitted, synthetic states are generated per seed (cross-seed results",
        "  are not directly comparable without a shared `states_file`).",
        "- Offline metrics only (action agreement, KL, MSE, cosine); no online rollout.",
        "- `child_vs_ref_a/b_agreement` are populated only when the `compare` stage runs.",
        "- Stage order: `distill` → `quantize` → `crossover` → `compare`.",
        "- `compare` requires `crossover`. With `quantize`, crossover and compare use int8 parents.",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    print(f"\nSummary CSV      : {csv_path}")
    print(f"Summary Markdown : {md_path}")


# ---------------------------------------------------------------------------
# Dry-run plan printer
# ---------------------------------------------------------------------------


def _print_plan(cfg: _AblationConfig) -> None:
    total = len(cfg.conditions) * len(cfg.seeds)
    print("=" * 70)
    print("Recombination Ablation — Execution Plan (DRY-RUN)")
    print("=" * 70)
    print(f"  Seeds              : {cfg.seeds}")
    print(f"  States             : {cfg.n_states} synthetic" if not cfg.states_file
          else f"  States             : {cfg.states_file}")
    print(f"  Input / output dim : {cfg.input_dim} / {cfg.output_dim}")
    print(f"  Hidden size        : {cfg.hidden_size}")
    print(f"  Results dir        : {cfg.results_dir}")
    print(f"  Conditions ({len(cfg.conditions)}):")
    for cond in cfg.conditions:
        print(f"    [{cond.name}] stages: {cond.stages}")
    print(f"  Total runs         : {total}")
    print("=" * 70)
    print()
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            work_dir = os.path.join(cfg.results_dir, cond.name, f"seed_{seed}")
            print(f"  {cond.name} / seed={seed}")
            print(f"    Work dir : {work_dir}")
            for stage in cond.stages:
                print(f"    Stage    : {stage}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified ablation runner: seeds × conditions → results/ tree + summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--config",
        default="",
        help="Path to YAML or JSON ablation config file.",
    )
    source.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a tiny built-in smoke-test config (2 seeds, 50 states, 2 epochs). "
            "No config file required."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan without running any training.",
    )
    p.add_argument(
        "--results-dir",
        default="",
        help="Override the results directory from the config (or the smoke-test default).",
    )
    return p


def _parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args() if argv is None else _parse_args_from(argv)

    # ------------------------------------------------------------------
    # Resolve config
    # ------------------------------------------------------------------
    if args.smoke_test:
        raw = dict(_SMOKE_CONFIG_DICT)
        if not args.results_dir:
            args.results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "ablation_smoke",
            )
        print("Running smoke-test config (tiny synthetic run).")
    elif args.config:
        raw = _load_raw_config(args.config)
    else:
        print(
            "Error: provide --config <file> or --smoke-test.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = _parse_config(raw, results_dir_override=args.results_dir)

    # ------------------------------------------------------------------
    # Dry-run: print plan and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        _print_plan(cfg)
        print("[DRY-RUN] No training was performed.")
        # Write stub summary so tests can verify the files exist.
        stub_rows = [
            {
                "condition": cond.name,
                "seed": seed,
                "stages": ",".join(cond.stages),
                "work_dir": os.path.join(cfg.results_dir, cond.name, f"seed_{seed}"),
            }
            for cond in cfg.conditions
            for seed in cfg.seeds
        ]
        _write_summary(stub_rows, cfg.results_dir, dry_run=True)
        return

    # ------------------------------------------------------------------
    # Real run
    # ------------------------------------------------------------------
    all_rows: List[Dict[str, Any]] = []
    shared_states: Optional[np.ndarray] = None

    for cond in cfg.conditions:
        print(f"\n{'=' * 70}")
        print(f"Condition: {cond.name}  (stages: {cond.stages})")
        print("=" * 70)
        for seed in cfg.seeds:
            print(f"\n  Seed {seed} …")
            if cfg.states_file:
                if shared_states is None:
                    shared_states = _make_states(cfg, seed)
                states = shared_states
            else:
                states = _make_states(cfg, seed)
            row = _run_condition_seed(cfg, cond, seed, states, dry_run=False)
            all_rows.append(row)
            print(f"  Done. elapsed={row.get('elapsed_s', '?')}s")

    _write_summary(all_rows, cfg.results_dir, dry_run=False)
    print("\nAblation complete.")


# ---------------------------------------------------------------------------
# Internal helper: parse args from an explicit list (for testing)
# ---------------------------------------------------------------------------


def _parse_args_from(argv: Sequence[str]) -> argparse.Namespace:
    """Parse *argv* using the same parser as ``_parse_args``."""
    return _build_arg_parser().parse_args(list(argv))


if __name__ == "__main__":
    main()
