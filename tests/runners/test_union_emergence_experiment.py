"""Unit tests for the Layer C union-emergence runner helpers."""

from __future__ import annotations

import pytest

from farm.runners.union_emergence_experiment import (
    ARM_NAMES,
    CellArmResult,
    UnionCell,
    _aggregate,
    _evaluate_win_conditions,
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
