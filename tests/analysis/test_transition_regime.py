"""Tests for transition-regime analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

from farm.analysis.transition_regime import (
    HIGH_SPECIATION_MODE,
    LOW_SPECIATION_MODE,
    TransitionRunMetrics,
    classify_modes,
    estimate_transition_probabilities,
    extract_transition_run_metrics,
    summarize_transition_regime,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _metric(
    idx: int,
    final_speciation: float,
    *,
    intervention: str = "baseline",
    parameter_value: float = 10.0,
    selection_strength: float = 0.1,
    overshoot: float = 5.0,
) -> TransitionRunMetrics:
    return TransitionRunMetrics(
        run_dir=f"/tmp/run_{intervention}_{idx}",
        seed=idx,
        profile="stable",
        parameter_name="initial_agent_resource_level",
        parameter_value=parameter_value,
        intervention=intervention,
        final_speciation=final_speciation,
        late_speciation_mean=final_speciation,
        late_speciation_slope=0.02 if final_speciation > 0.5 else -0.01,
        mean_population=100.0,
        final_population=95.0,
        peak_population=110.0,
        population_early_overshoot=overshoot,
        late_selection_strength_mean=selection_strength,
    )


def test_extract_transition_run_metrics_from_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "intrinsic_gene_trajectory.jsonl",
        [
            {"step": 0, "speciation_index": 0.2, "n_alive": 10},
            {
                "step": 1,
                "speciation_index": 0.4,
                "n_alive": 12,
                "realized_birth_rate": 0.2,
                "realized_death_rate": 0.0,
                "effective_selection_strength": 0.1,
                "mean_reproduction_cost": 5.0,
            },
            {
                "step": 2,
                "speciation_index": 0.8,
                "n_alive": 11,
                "realized_birth_rate": 0.0,
                "realized_death_rate": 0.1,
                "effective_selection_strength": 0.3,
                "mean_reproduction_cost": 6.0,
            },
        ],
    )
    _write_jsonl(
        run_dir / "intrinsic_gene_snapshots.jsonl",
        [
            {"step": 0, "agents": [{"chromosome": {"learning_rate": 0.01}}]},
            {"step": 2, "agents": [{"chromosome": {"learning_rate": 0.02}}]},
        ],
    )
    _write_jsonl(
        run_dir / "cluster_lineage.jsonl",
        [
            {"transition_type": "split"},
            {"transition_type": "continuation"},
            {"transition_type": "merge"},
        ],
    )
    (run_dir / "intrinsic_evolution_metadata.json").write_text(
        json.dumps({"seed": 42, "initial_conditions": {"profile": "stable"}}),
        encoding="utf-8",
    )

    metrics = extract_transition_run_metrics(
        run_dir,
        {
            "seed": 42,
            "parameter_name": "initial_agent_resource_level",
            "parameter_value": 10.0,
            "intervention": "baseline",
        },
    )

    assert metrics.seed == 42
    assert metrics.final_speciation == 0.8
    assert metrics.population_early_overshoot == 2
    assert metrics.cluster_split_rate == 1 / 3
    assert metrics.cluster_merge_rate == 1 / 3
    assert metrics.gene_pct_shift["learning_rate"] == 100.0


def test_classify_modes_threshold_separates_low_and_high():
    metrics = [
        _metric(0, 0.2),
        _metric(1, 0.3),
        _metric(2, 0.8),
        _metric(3, 0.9),
    ]
    assignments = classify_modes(metrics, threshold=0.5)
    assert [assignment.mode for assignment in assignments[:2]] == [LOW_SPECIATION_MODE, LOW_SPECIATION_MODE]
    assert [assignment.mode for assignment in assignments[2:]] == [HIGH_SPECIATION_MODE, HIGH_SPECIATION_MODE]


def test_estimate_transition_probability_uses_wilson_interval():
    metrics = [
        _metric(0, 0.2),
        _metric(1, 0.8),
        _metric(2, 0.9),
    ]
    assignments = classify_modes(metrics, threshold=0.5)
    probabilities = estimate_transition_probabilities(
        assignments,
        metrics,
        parameter_name="initial_agent_resource_level",
    )
    assert len(probabilities) == 1
    assert probabilities[0].n == 3
    assert probabilities[0].k == 2
    assert probabilities[0].p == 2 / 3
    assert 0.0 <= probabilities[0].ci95[0] < probabilities[0].ci95[1] <= 1.0


def test_summary_generates_exit_paragraph_when_gates_pass():
    baseline = [
        _metric(0, 0.25, selection_strength=0.1, overshoot=3.0),
        _metric(1, 0.26, selection_strength=0.1, overshoot=3.0),
        _metric(2, 0.27, selection_strength=0.1, overshoot=3.0),
        _metric(3, 0.82, selection_strength=1.0, overshoot=12.0),
        _metric(4, 0.83, selection_strength=1.0, overshoot=12.0),
        _metric(5, 0.84, selection_strength=1.0, overshoot=12.0),
    ]
    crossover = [
        _metric(10, 0.80, intervention="crossover_on"),
        _metric(11, 0.81, intervention="crossover_on"),
        _metric(12, 0.82, intervention="crossover_on"),
        _metric(13, 0.83, intervention="crossover_on"),
        _metric(14, 0.84, intervention="crossover_on"),
        _metric(15, 0.85, intervention="crossover_on"),
    ]
    summary = summarize_transition_regime(
        baseline + crossover,
        min_runs_per_range=6,
        mechanism_effect_threshold=0.5,
    )

    assert summary.mode_counts[LOW_SPECIATION_MODE] >= 3
    assert summary.mode_counts[HIGH_SPECIATION_MODE] >= 6
    assert any(mechanism.supported for mechanism in summary.mechanisms)
    assert summary.exit_paragraph is not None
    assert "transitions between mode" in summary.exit_paragraph


def test_summary_refuses_exit_paragraph_when_evidence_is_insufficient():
    summary = summarize_transition_regime([_metric(0, 0.8)], min_runs_per_range=6)
    assert summary.exit_paragraph is None
    assert summary.evidence_gate_reasons
