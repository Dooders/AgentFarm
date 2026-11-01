"""Unit tests for the Layer C union-emergence runner helpers."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome
from farm.core.union_bonds import UnionPolicy
from farm.runners.union_emergence_experiment import (
    ARM_NAMES,
    CellArmResult,
    UnionCell,
    UnionEmergenceExperiment,
    UnionEmergenceExperimentConfig,
    _aggregate,
    _evaluate_win_conditions,
    _fmt_metric,
    _write_figures,
    _write_markdown,
    default_cells,
)


@pytest.mark.unit
def test_default_cells_cover_pre_registered_ablations():
    cells = {cell.name: cell for cell in default_cells()}
    assert cells["baseline"].courtship_steps == 12
    assert cells["baseline"].exit_tax == 1.1
    assert cells["baseline"].social_range == 3.2
    assert cells["no_courtship"].courtship_steps == 0
    assert cells["cheap_exit"].exit_tax == 0.2
    assert cells["costly_exit"].exit_tax == 2.2
    assert cells["tight_range"].social_range == 2.0
    assert cells["wide_range"].social_range == 8.0
    assert set(ARM_NAMES) == {"solo_only", "promiscuous", "optional_union", "forced_union"}


@pytest.mark.unit
def test_aggregate_and_win_conditions_read_directionally():
    optional = CellArmResult(cell="baseline", arm="optional_union", seed=0)
    optional.synergy = [1.2]
    optional.paired_frac = [0.5]
    optional.mean_energy = [20.0]
    optional.extraction = [0.08]
    optional.leave_events = [0, 4]
    optional.steps = [0, 10]
    optional.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    optional.end_genes = {"pair_commitment": 0.49, "fidelity": 0.56}

    forced = CellArmResult(cell="baseline", arm="forced_union", seed=0)
    forced.synergy = [1.4]
    forced.paired_frac = [0.8]
    forced.mean_energy = [16.0]
    forced.extraction = [0.12]
    forced.leave_events = [0, 6]
    forced.steps = [0, 10]
    forced.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    forced.end_genes = {"pair_commitment": 0.5, "fidelity": 0.6}

    promiscuous = CellArmResult(cell="baseline", arm="promiscuous", seed=0)
    promiscuous.mean_energy = [12.0]
    promiscuous.steps = [0, 10]
    promiscuous.leave_events = [0, 0]
    promiscuous.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    promiscuous.end_genes = {"pair_commitment": 0.5, "fidelity": 0.5}

    cheap = CellArmResult(cell="cheap_exit", arm="optional_union", seed=0)
    cheap.synergy = [1.05]
    cheap.mean_energy = [18.0]
    cheap.steps = [0, 10]
    cheap.leave_events = [0, 8]
    cheap.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    cheap.end_genes = {"pair_commitment": 0.5, "fidelity": 0.5}

    noco = CellArmResult(cell="no_courtship", arm="optional_union", seed=0)
    noco.synergy = [1.25]
    noco.mean_energy = [19.0]
    noco.steps = [0, 10]
    noco.leave_events = [0, 3]
    noco.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    noco.end_genes = {"pair_commitment": 0.5, "fidelity": 0.5}

    cells = _aggregate([optional, forced, promiscuous, cheap, noco])
    wins = _evaluate_win_conditions(cells)
    assert wins["passed"] == wins["total"]
    assert UnionCell(name="x").name == "x"


@pytest.mark.unit
def test_config_rejects_invalid_steps_replicates_and_arms():
    with pytest.raises(ValueError, match="num_steps"):
        UnionEmergenceExperimentConfig(num_steps=0)
    with pytest.raises(ValueError, match="num_replicates"):
        UnionEmergenceExperimentConfig(num_replicates=0)
    with pytest.raises(ValueError, match="unknown arms"):
        UnionEmergenceExperimentConfig(arms=("solo_only", "not_real"))


@pytest.mark.unit
def test_fmt_metric_and_win_conditions_handle_missing_rows():
    assert _fmt_metric({"synergy_index": None}, "synergy_index") == "—"
    assert _fmt_metric({"synergy_index": 1.2345}, "synergy_index") == "1.234"
    wins = _evaluate_win_conditions([])
    assert wins["passed"] == 0
    assert wins["headline"]["baseline_optional_synergy"] is None


@pytest.mark.unit
def test_write_markdown_and_figures_for_a_baseline_payload():
    optional = CellArmResult(cell="baseline", arm="optional_union", seed=0)
    optional.steps = [0, 5, 10]
    optional.paired_frac = [0.2, 0.4, 0.6]
    optional.synergy = [1.1, 1.2, 1.15]
    optional.mean_energy = [20.0, 21.0, 22.0]
    optional.extraction = [0.1, 0.12, 0.11]
    optional.leave_events = [0, 2, 4]
    optional.gene_means = {"pair_commitment": [0.5, 0.5, 0.49], "fidelity": [0.5, 0.51, 0.52]}
    optional.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5, "specialize": 0.5}
    optional.end_genes = {"pair_commitment": 0.49, "fidelity": 0.52, "specialize": 0.5}

    forced = CellArmResult(cell="baseline", arm="forced_union", seed=0)
    forced.steps = [0, 5, 10]
    forced.paired_frac = [0.8, 0.85, 0.9]
    forced.synergy = [0.9, 0.95, 1.0]
    forced.mean_energy = [18.0, 18.5, 19.0]
    forced.extraction = [0.2, 0.22, 0.21]
    forced.leave_events = [0, 3, 6]
    forced.gene_means = {"pair_commitment": [0.5, 0.5, 0.5], "fidelity": [0.5, 0.5, 0.5]}
    forced.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5, "specialize": 0.5}
    forced.end_genes = {"pair_commitment": 0.5, "fidelity": 0.5, "specialize": 0.5}

    promiscuous = CellArmResult(cell="baseline", arm="promiscuous", seed=0)
    promiscuous.steps = [0, 5, 10]
    promiscuous.paired_frac = [0.0, 0.0, 0.0]
    promiscuous.mean_energy = [16.0, 16.5, 17.0]
    promiscuous.leave_events = [0, 0, 0]
    promiscuous.start_genes = {"pair_commitment": 0.5, "fidelity": 0.5}
    promiscuous.end_genes = {"pair_commitment": 0.5, "fidelity": 0.5}

    cells = _aggregate([optional, forced, promiscuous])
    payload = {
        "cells": cells,
        "win_conditions": _evaluate_win_conditions(cells),
    }
    with tempfile.TemporaryDirectory() as output_dir:
        md_path = _write_markdown(output_dir, payload)
        assert os.path.isfile(md_path)
        with open(md_path, encoding="utf-8") as handle:
            text = handle.read()
        assert "Win-condition checks" in text
        figures = _write_figures(output_dir, payload, [optional, forced, promiscuous])
        assert "synergy_bar" in figures
        assert "paired_frac_trajectory" in figures
        assert os.path.isfile(figures["synergy_bar"])


def _fake_union_env(n_agents: int = 3):
    agents = []
    for index in range(n_agents):
        agent = SimpleNamespace(
            agent_id=f"a{index}",
            alive=True,
            resource_level=8.0 + index,
            position=(float(index), 0.0),
            partner_id=None,
            pair_age=0,
            bond_strength=0.0,
            role="none",
            last_action_name="gather",
            hyperparameter_chromosome=default_hyperparameter_chromosome(),
            environment=None,
        )
        agents.append(agent)
    env = SimpleNamespace(
        alive_agent_objects=agents,
        _agent_objects={agent.agent_id: agent for agent in agents},
        width=24.0,
        height=24.0,
        union_policy=None,
    )
    for agent in agents:
        agent.environment = env
    return env


@pytest.mark.unit
def test_experiment_run_writes_summary_through_stubbed_simulation():
    env = _fake_union_env()

    def _side_effect(*_args, **kwargs):
        on_ready = kwargs.get("on_environment_ready")
        on_step_end = kwargs.get("on_step_end")
        if on_ready is not None:
            on_ready(env)
        if on_step_end is not None:
            on_step_end(env, 0)
            on_step_end(env, 5)
        return env

    config = UnionEmergenceExperimentConfig(
        num_steps=5,
        num_replicates=1,
        seed=7,
        record_interval=5,
        arms=("optional_union",),
        cells=(UnionCell(name="baseline"),),
        in_memory_db=True,
    )
    with tempfile.TemporaryDirectory() as output_dir:
        config.output_dir = output_dir
        with patch(
            "farm.runners.union_emergence_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            payload = UnionEmergenceExperiment(SimulationConfig(), config).run()
        assert os.path.isfile(payload["summary_path"])
        with open(payload["summary_path"], encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["config"]["arms"] == ["optional_union"]
        assert saved["win_conditions"]["total"] == 8
        assert env.union_policy is not None
        assert isinstance(env.union_policy, UnionPolicy)
        assert env.union_policy.pairing_mode == "optional"

    config.in_memory_db = False
    config.arms = ("solo_only",)
    with tempfile.TemporaryDirectory() as disk_dir:
        config.output_dir = disk_dir
        with patch(
            "farm.runners.union_emergence_experiment.run_simulation",
            side_effect=_side_effect,
        ):
            UnionEmergenceExperiment(SimulationConfig(), config).run()
        assert any(name.endswith("_s7") or "solo_only" in name for name in os.listdir(disk_dir))
