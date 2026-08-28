"""Tests for consensus experiment plot generation and output helpers."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from farm.experiments.consensus.experiment import (
    ExperimentConfig,
    SweepConfig,
    config_manifest,
    run_sweep,
    sweep_manifest,
    write_outputs,
)
from farm.experiments.consensus.plots import write_figures

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

PARADIGMS = ["party", "individual", "score"]
POPULATIONS = ["two_cluster", "one_cluster"]


def _make_trials(populations=POPULATIONS, paradigms=PARADIGMS, n=6) -> pd.DataFrame:
    """Minimal but realistic trials DataFrame accepted by plot helpers."""
    rng = np.random.default_rng(0)
    rows = []
    for pop in populations:
        for paradigm in paradigms:
            for _ in range(n):
                rows.append(
                    {
                        "population": pop,
                        "paradigm": paradigm,
                        "total_welfare": rng.uniform(0.4, 0.9),
                        "supporter_welfare": rng.uniform(0.5, 1.0),
                        "loser_welfare": rng.uniform(0.1, 0.5),
                        "loser_share": rng.uniform(0.3, 0.7),
                        "gap": rng.uniform(-0.1, 0.3),
                        "lambda_winner": rng.uniform(0.0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trials() -> pd.DataFrame:
    return _make_trials()


# ---------------------------------------------------------------------------
# plots.py
# ---------------------------------------------------------------------------


class TestWriteFigures:
    def test_creates_three_png_files(self, trials, tmp_path):
        figs_dir = tmp_path / "figures"
        write_figures(trials, figs_dir)
        expected = {"welfare_by_paradigm.png", "gap_vs_loser_share.png", "lambda_by_paradigm.png"}
        created = {p.name for p in figs_dir.iterdir()}
        assert expected.issubset(created)

    def test_figures_are_non_empty(self, trials, tmp_path):
        figs_dir = tmp_path / "figures"
        write_figures(trials, figs_dir)
        for png in figs_dir.glob("*.png"):
            assert png.stat().st_size > 0, f"{png.name} is empty"

    def test_creates_output_directory(self, trials, tmp_path):
        figs_dir = tmp_path / "nested" / "figures"
        assert not figs_dir.exists()
        write_figures(trials, figs_dir)
        assert figs_dir.is_dir()

    def test_single_population(self, tmp_path):
        trials_single = _make_trials(populations=["two_cluster"])
        figs_dir = tmp_path / "figures_single"
        write_figures(trials_single, figs_dir)
        assert (figs_dir / "welfare_by_paradigm.png").exists()
        assert (figs_dir / "gap_vs_loser_share.png").exists()
        assert (figs_dir / "lambda_by_paradigm.png").exists()

    def test_unknown_paradigm_uses_fallback_color(self, tmp_path):
        trials_unknown = _make_trials(paradigms=["unknown_paradigm"])
        figs_dir = tmp_path / "figures_unknown"
        write_figures(trials_unknown, figs_dir)
        assert (figs_dir / "welfare_by_paradigm.png").exists()


# ---------------------------------------------------------------------------
# experiment.py missing lines
# ---------------------------------------------------------------------------


TINY_CONFIG = ExperimentConfig(
    trials=3,
    voters=40,
    candidates=4,
    population="two_cluster",
    seed=0,
)


class TestSweepConfig:
    def test_cells_returns_cross_product(self):
        sweep = SweepConfig(
            base=ExperimentConfig(trials=1, voters=20, candidates=4, seed=0),
            populations=["two_cluster", "one_cluster"],
            candidate_counts=(4, 6),
        )
        cells = sweep.cells()
        assert len(cells) == 4
        pops = [c.population for c in cells]
        assert "two_cluster" in pops and "one_cluster" in pops

    def test_cells_single_population_single_count(self):
        sweep = SweepConfig(
            base=ExperimentConfig(trials=1, voters=20, candidates=4, seed=0),
            populations=["two_cluster"],
            candidate_counts=(4,),
        )
        cells = sweep.cells()
        assert len(cells) == 1
        assert cells[0].population == "two_cluster"
        assert cells[0].candidates == 4


class TestRunSweep:
    def test_run_sweep_concatenates_cells(self):
        sweep = SweepConfig(
            base=ExperimentConfig(trials=2, voters=30, candidates=4, seed=1),
            populations=["two_cluster", "one_cluster"],
            candidate_counts=(4,),
        )
        result = run_sweep(sweep)
        assert isinstance(result, pd.DataFrame)
        assert set(result["population"].unique()) == {"two_cluster", "one_cluster"}


class TestWriteOutputs:
    def test_write_outputs_creates_csvs_and_json(self, tmp_path):
        from dataclasses import asdict

        from farm.experiments.consensus.experiment import run_trials

        trials = run_trials(TINY_CONFIG)
        run_config = config_manifest(TINY_CONFIG, "python run_experiment.py")
        write_outputs(trials, tmp_path, run_config)
        assert (tmp_path / "trials.csv").exists()
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "allocation_means.csv").exists()
        assert (tmp_path / "contrasts.csv").exists()
        assert (tmp_path / "run_config.json").exists()
        assert (tmp_path / "figures").is_dir()
        assert (tmp_path / "REPORT.md").exists()


class TestManifestHelpers:
    def test_config_manifest_contains_command_and_config(self):
        cfg = ExperimentConfig(trials=5, voters=100, candidates=6, seed=42)
        manifest = config_manifest(cfg, "python run_experiment.py --seed 42")
        assert manifest["command"] == "python run_experiment.py --seed 42"
        assert manifest["config"]["trials"] == 5
        assert manifest["config"]["seed"] == 42

    def test_sweep_manifest_contains_populations_and_counts(self):
        sweep = SweepConfig(
            base=ExperimentConfig(trials=2, voters=30, candidates=4, seed=0),
            populations=["two_cluster", "one_cluster"],
            candidate_counts=(4, 8),
        )
        manifest = sweep_manifest(sweep, "python run_experiment.py --sweep")
        assert manifest["command"] == "python run_experiment.py --sweep"
        assert "two_cluster" in manifest["populations"]
        assert 4 in manifest["candidate_counts"]
        assert manifest["config"]["trials"] == 2
