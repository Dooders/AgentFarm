"""Tests for the Veil Ceiling metrics, matrix orchestration, analysis and report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farm.experiments.veil_ceiling.analysis import (
    VERDICT_FALSIFIED,
    VERDICT_INCONCLUSIVE,
    VERDICT_SUPPORTED,
    VERDICT_VOID,
    analyze,
    bootstrap_mean_ci,
    compute_noise_band,
    compute_onsets,
    compute_validity,
    hypothesis_verdicts,
    paired_contrast,
    write_analysis,
)
from farm.experiments.veil_ceiling.config import AnalysisThresholds
from farm.experiments.veil_ceiling.experiment import (
    AGENTS_FILENAME,
    CELLS_DIRNAME,
    MANIFEST_FILENAME,
    RUNS_FILENAME,
    WINDOWS_FILENAME,
    MatrixConfig,
    MatrixOutputs,
    load_outputs,
    matrix_manifest,
    run_matrix,
)
from farm.experiments.veil_ceiling.metrics import (
    STRATEGY_CONDITIONAL,
    STRATEGY_COOPERATIVE,
    STRATEGY_DEFECTOR,
    agent_validity_table,
    classify_strategies,
    late_training_windows,
    onset_window,
    predictive_validity,
    safe_rate,
    window_rates,
)
from farm.experiments.veil_ceiling.report import FIGURES_DIRNAME, REPORT_FILENAME, write_report

pytestmark = pytest.mark.unit

FAST_THRESHOLDS = AnalysisThresholds(bootstrap_reps=40)
TINY_MATRIX = MatrixConfig(
    seeds=(1, 2),
    condition_names=("C0", "C1", "C2", "C4"),
    include_robustness=False,
    train_ticks=100,
    eval_ticks=50,
    window_ticks=50,
    workers=1,
)


@pytest.fixture(scope="module")
def tiny_outputs(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("veil_tiny")
    outputs = run_matrix(TINY_MATRIX, out, progress=None)
    return out, outputs


# ── pure metric helpers ───────────────────────────────────────────────────
def test_safe_rate_handles_zero_denominators() -> None:
    assert np.isnan(safe_rate(1.0, 0.0))
    assert safe_rate(1.0, 4.0) == 0.25
    series = safe_rate(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
    assert np.isnan(series.iloc[0]) and series.iloc[1] == 0.5


def test_bootstrap_ci_brackets_mean_and_handles_degenerate_input() -> None:
    mean, lo, hi, n = bootstrap_mean_ci(np.array([1.0, 2.0, 3.0, 4.0]), FAST_THRESHOLDS)
    assert n == 4 and lo <= mean <= hi and mean == 2.5
    assert bootstrap_mean_ci(np.array([]), FAST_THRESHOLDS)[3] == 0
    assert bootstrap_mean_ci(np.array([np.nan, 7.0]), FAST_THRESHOLDS)[:3] == (7.0, 7.0, 7.0)


def test_paired_contrast_matches_on_seed_index() -> None:
    treatment = pd.Series({1: 0.5, 2: 0.6, 3: 0.7})
    baseline = pd.Series({2: 0.1, 3: 0.1, 4: 0.1})
    out = paired_contrast(treatment, baseline, FAST_THRESHOLDS)
    assert out["n_pairs"] == 2
    assert out["mean_diff"] == pytest.approx(0.55)
    assert out["sign_agreement"] == 1.0


def test_onset_requires_consecutive_windows_above_band() -> None:
    rates = pd.DataFrame(
        {
            "phase": ["train"] * 5,
            "window_end_tick": [50, 100, 150, 200, 250],
            "mean_generation": [0, 1, 2, 3, 4],
            "delta_cue": [0.5, 0.0, 0.4, 0.4, 0.4],
        }
    )
    assert onset_window(rates, 0.1, consecutive=2) == (150.0, 2.0)
    assert onset_window(rates, 0.1, consecutive=1) == (50.0, 0.0)
    assert all(np.isnan(v) for v in onset_window(rates, 0.9, consecutive=2))


def test_late_training_windows_uses_last_third() -> None:
    windows = pd.DataFrame({"phase": ["train"] * 6 + ["eval"], "window_end_tick": [50, 100, 150, 200, 250, 300, 350]})
    late = late_training_windows(windows)
    assert list(late["window_end_tick"]) == [250, 300]


def _synthetic_agents(n: int, seed_count: int, conditional: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        obs_def = rng.integers(0, 3) if conditional else rng.integers(0, 40)
        unobs_def = rng.integers(30, 40) if conditional else obs_def + rng.integers(0, 2)
        row = {
            "seed": i % seed_count,
            "agent_id": i,
            "lifespan": 100,
            "offspring": 1,
            "energy_gained": 50.0,
            "penalties_paid": 0.0,
        }
        for stat in ("ticks", "opportunities", "defections", "gathers", "moves", "passes", "penalties", "energy_sum"):
            for phase in ("train", "eval"):
                for region in ("train", "heldout"):
                    for m in (0, 1):
                        for c in (0, 1):
                            row[f"{stat}__{phase}__{region}__m{m}__c{c}"] = 0.0
        for c, defections in ((0, unobs_def), (1, obs_def)):
            row[f"ticks__train__train__m{c}__c{c}"] = 60.0
            row[f"opportunities__train__train__m{c}__c{c}"] = 40.0
            row[f"defections__train__train__m{c}__c{c}"] = float(defections)
            row[f"gathers__train__train__m{c}__c{c}"] = 40.0 - defections
            row[f"energy_sum__train__train__m{c}__c{c}"] = 600.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_validity_table_and_auc_on_synthetic_agents() -> None:
    conditional = _synthetic_agents(80, 4, conditional=True)
    conditional["ticks__train__train__m1__c0"] = 40.0
    table = agent_validity_table(conditional, FAST_THRESHOLDS)
    assert len(table) == 80
    assert (table["unobs_defect_rate"] > table["obs_defect_rate"]).all()
    assert table["obs_log_ticks"].iloc[0] == pytest.approx(np.log1p(60.0))
    honest = _synthetic_agents(80, 4, conditional=False)
    result = predictive_validity(agent_validity_table(honest, FAST_THRESHOLDS), FAST_THRESHOLDS)
    assert result.n_seeds == 4
    assert result.auc_rank > 0.9
    assert result.auc_classifier > 0.8
    assert result.auc_classifier_ci[0] <= result.auc_classifier <= result.auc_classifier_ci[1]


def test_strategy_classification() -> None:
    conditional = classify_strategies(_synthetic_agents(20, 2, conditional=True), FAST_THRESHOLDS)
    assert (conditional["strategy"] == STRATEGY_CONDITIONAL).all()
    honest = classify_strategies(_synthetic_agents(200, 2, conditional=False), FAST_THRESHOLDS)
    assert set(honest["strategy"]) >= {STRATEGY_COOPERATIVE, STRATEGY_DEFECTOR}
    assert np.allclose(honest["fitness"], 0.5)


# ── orchestration ─────────────────────────────────────────────────────────
def test_matrix_config_enumerates_pre_registered_cells() -> None:
    full = MatrixConfig()
    configs = full.run_configs()
    assert len(configs) == (9 + 4) * 2 * 30
    assert len({c.cell_id for c in configs}) == 26
    assert len({c.run_id for c in configs}) == len(configs)
    with pytest.raises(ValueError):
        MatrixConfig(condition_names=("C9",))
    with pytest.raises(ValueError):
        MatrixConfig(inheritance_modes=("genetic",))
    manifest = matrix_manifest(TINY_MATRIX)
    assert manifest["n_runs"] == 16
    assert set(manifest["conditions"]) == {"C0", "C1", "C2", "C4"}
    assert "analysis_thresholds" in manifest and "world" in manifest


def test_run_matrix_writes_raw_outputs_and_resumes(tiny_outputs) -> None:
    out, outputs = tiny_outputs
    cells = sorted(p.name for p in (out / CELLS_DIRNAME).iterdir())
    assert cells == sorted(f"{c}__{m}" for c in ("C0", "C1", "C2", "C4") for m in ("baldwinian", "lamarckian"))
    assert len(outputs.runs) == 16
    assert set(outputs.runs["seed"]) == {1, 2}
    assert (out / MANIFEST_FILENAME).exists()
    for cell in cells:
        agents = pd.read_csv(out / CELLS_DIRNAME / cell / AGENTS_FILENAME)
        assert set(agents["cell_id"]) == {cell}
        assert set(agents["seed"]) == {1, 2}
    reloaded = load_outputs(out)
    assert reloaded.train_ticks == 100
    pd.testing.assert_frame_equal(
        reloaded.runs.sort_values("run_id").reset_index(drop=True),
        outputs.runs.sort_values("run_id").reset_index(drop=True),
    )
    marker = out / CELLS_DIRNAME / cells[0] / AGENTS_FILENAME
    mtime = marker.stat().st_mtime
    run_matrix(TINY_MATRIX, out, progress=None, resume=True)
    assert marker.stat().st_mtime == mtime


def test_run_matrix_rejects_incompatible_resume_manifest(tiny_outputs) -> None:
    out, _ = tiny_outputs
    incompatible = MatrixConfig(
        seeds=(1, 2, 3),
        condition_names=("C0", "C1", "C2", "C4"),
        include_robustness=False,
        train_ticks=100,
        eval_ticks=50,
        window_ticks=50,
        workers=1,
    )
    with pytest.raises(ValueError, match="incompatible output directory for resume"):
        run_matrix(incompatible, out, progress=None, resume=True)


def test_run_matrix_reruns_incomplete_cells_on_resume(tmp_path: Path) -> None:
    run_matrix(TINY_MATRIX, tmp_path, progress=None)
    cell_dir = tmp_path / CELLS_DIRNAME / "C0__baldwinian"
    (cell_dir / RUNS_FILENAME).unlink()
    (cell_dir / WINDOWS_FILENAME).unlink()
    outputs = run_matrix(TINY_MATRIX, tmp_path, progress=None, resume=True)
    assert (cell_dir / RUNS_FILENAME).exists()
    assert (cell_dir / WINDOWS_FILENAME).exists()
    assert len(outputs.runs) == 16


def test_window_rates_from_raw_windows(tiny_outputs) -> None:
    _, outputs = tiny_outputs
    run_id = outputs.runs["run_id"].iloc[0]
    rates = window_rates(outputs.windows[outputs.windows["run_id"] == run_id])
    assert {"window_end_tick", "phase", "rate", "delta_cue", "delta_true"} <= set(rates.columns)
    assert list(rates["phase"]) == ["train", "train", "eval"]
    assert rates["rate"].dropna().between(0.0, 1.0).all()


# ── analysis + report ─────────────────────────────────────────────────────
def test_compute_validity_uses_run_metadata_for_robustness_fidelity() -> None:
    agents = _synthetic_agents(40, 2, conditional=True).assign(
        run_id="C1_p3__baldwinian__s1",
        cell_id="C1_p3__baldwinian",
        condition="C1_p3",
        family="C1",
        inheritance_mode="baldwinian",
    )
    outputs = MatrixOutputs(
        runs=pd.DataFrame(
            [
                {
                    "run_id": "C1_p3__baldwinian__s1",
                    "cell_id": "C1_p3__baldwinian",
                    "condition": "C1_p3",
                    "family": "C1",
                    "inheritance_mode": "baldwinian",
                    "fidelity": 0.0,
                }
            ]
        ),
        windows=pd.DataFrame(),
        agents_by_cell={"C1_p3__baldwinian": agents},
        train_ticks=100,
    )
    validity = compute_validity(outputs, FAST_THRESHOLDS)
    assert validity["fidelity"].iloc[0] == 0.0


def _window_row(run_id: str, tick: int, delta_cue: float) -> dict:
    row = {
        "run_id": run_id,
        "window_end_tick": float(tick),
        "phase": "train",
        "population": 20.0,
        "mean_generation": tick / 50.0,
    }
    for stat in ("opportunities", "defections", "penalties"):
        for m in (0, 1):
            for c in (0, 1):
                row[f"{stat}__train__m{m}__c{c}"] = 0.0
    for m in (0, 1):
        row[f"opportunities__train__m{m}__c0"] = 50.0
        row[f"opportunities__train__m{m}__c1"] = 50.0
        row[f"defections__train__m{m}__c0"] = 50.0 * delta_cue
    return row


def test_compute_onsets_uses_matching_per_window_c4_bands() -> None:
    windows = pd.DataFrame(
        [
            _window_row("C4__baldwinian__s1", 50, 0.10),
            _window_row("C4__baldwinian__s1", 100, 0.80),
            _window_row("C4__baldwinian__s1", 150, 0.80),
            _window_row("C4__baldwinian__s2", 50, 0.20),
            _window_row("C4__baldwinian__s2", 100, 0.90),
            _window_row("C4__baldwinian__s2", 150, 0.90),
            _window_row("C2__baldwinian__s1", 50, 0.35),
            _window_row("C2__baldwinian__s1", 100, 0.00),
            _window_row("C2__baldwinian__s1", 150, 0.00),
        ]
    )
    outputs = MatrixOutputs(runs=pd.DataFrame(), windows=windows, agents_by_cell={}, train_ticks=150)
    run_metrics = pd.DataFrame(
        [
            {
                "run_id": "C4__baldwinian__s1",
                "cell_id": "C4__baldwinian",
                "condition": "C4",
                "family": "C4",
                "inheritance_mode": "baldwinian",
                "seed": 1,
                "delta_cue": 0.80,
                "delta_true": 0.0,
                "eval_heldout_delta_cue": 0.0,
            },
            {
                "run_id": "C4__baldwinian__s2",
                "cell_id": "C4__baldwinian",
                "condition": "C4",
                "family": "C4",
                "inheritance_mode": "baldwinian",
                "seed": 2,
                "delta_cue": 0.90,
                "delta_true": 0.0,
                "eval_heldout_delta_cue": 0.0,
            },
            {
                "run_id": "C2__baldwinian__s1",
                "cell_id": "C2__baldwinian",
                "condition": "C2",
                "family": "C2",
                "inheritance_mode": "baldwinian",
                "seed": 1,
                "rate": 0.2,
            },
        ]
    )
    noise_band = compute_noise_band(run_metrics, outputs, AnalysisThresholds(bootstrap_reps=20, onset_consecutive_windows=1))
    onsets = compute_onsets(outputs, run_metrics, noise_band, AnalysisThresholds(bootstrap_reps=20, onset_consecutive_windows=1))
    c2 = onsets[onsets["cell_id"] == "C2__baldwinian"].iloc[0]
    assert c2["conditional_onset_tick"] == 50.0


def test_h1_is_inconclusive_when_reduced_condition_set_omits_spearman_test() -> None:
    verdicts = hypothesis_verdicts(
        cell_summary=pd.DataFrame(
            [
                {
                    "cell_id": "C2__baldwinian",
                    "eval_heldout_delta_cue_mean": np.nan,
                    "eval_heldout_delta_cue_lo": np.nan,
                    "eval_heldout_delta_cue_hi": np.nan,
                    "eval_train_delta_cue_mean": np.nan,
                }
            ]
        ),
        noise_band=pd.DataFrame(
            [{"inheritance_mode": "baldwinian", "late_band_lo": -0.1, "late_band_hi": 0.1, "heldout_band_hi": 0.1}]
        ),
        paired=pd.DataFrame(
            [
                {
                    "inheritance_mode": "baldwinian",
                    "treatment": "C2",
                    "baseline": "C4",
                    "metric": "delta_cue",
                    "ci_lo": 0.2,
                    "ci_hi": 0.3,
                }
            ]
        ),
        validity=pd.DataFrame(),
        dose_response=pd.DataFrame([{"inheritance_mode": "baldwinian", "spearman_fidelity_delta": np.nan}]),
        concealment=pd.DataFrame(columns=["cell_id"]),
        onset_summary=pd.DataFrame(),
        onsets=pd.DataFrame(),
        falsification={
            "baldwinian": {
                "check_1_c4_null": {"passed": True},
                "check_2_c2_divergence": {"h1_h2_falsified": False},
                "check_3_c2_validity": {"c1_auc": np.nan, "c2_auc": np.nan, "h2_falsified": False, "collapsed": False},
            }
        },
        censor_tick=150.0,
        thresholds=FAST_THRESHOLDS,
    )
    assert verdicts["baldwinian"]["H1"]["verdict"] == VERDICT_INCONCLUSIVE


def test_analyze_and_write_outputs_end_to_end(tiny_outputs) -> None:
    out, outputs = tiny_outputs
    result = analyze(outputs, FAST_THRESHOLDS)

    assert len(result.run_metrics) == 16
    assert set(result.cell_summary["cell_id"]) == set(outputs.cells())
    assert set(result.noise_band["inheritance_mode"]) == {"baldwinian", "lamarckian"}
    c0 = result.cell_summary[result.cell_summary["condition"] == "C0"]
    assert c0["delta_cue_mean"].isna().all()
    assert (c0["expected_penalty_per_defection"] == 0.0).all()

    paired = result.paired
    assert {("C2", "C1"), ("C2", "C4"), ("C1", "C0")} <= set(zip(paired["treatment"], paired["baseline"]))
    assert "lamarckian-baldwinian" in set(paired["inheritance_mode"])
    c2c1 = paired[(paired["treatment"] == "C2") & (paired["baseline"] == "C1") & (paired["metric"] == "rate")]
    assert (c2c1["n_pairs"] == 2).all()

    assert set(result.calibration) == {"baldwinian", "lamarckian"}
    for mode in ("baldwinian", "lamarckian"):
        checks = result.falsification[mode]
        assert set(checks) == {"check_1_c4_null", "check_2_c2_divergence", "check_3_c2_validity"}
        verdicts = result.hypotheses[mode]
        assert set(verdicts) == {"H1", "H2", "H3", "H4"}
        assert all(
            v["verdict"] in {VERDICT_SUPPORTED, VERDICT_FALSIFIED, VERDICT_INCONCLUSIVE, VERDICT_VOID}
            for v in verdicts.values()
        )
    assert result.hypotheses["H5"]["verdict"] in {VERDICT_SUPPORTED, VERDICT_FALSIFIED, VERDICT_INCONCLUSIVE}
    assert result.hypotheses["H5"]["censoring_tick_for_never_onset"] == 150

    analysis_dir = write_analysis(result, out)
    written = {p.name for p in analysis_dir.iterdir()}
    assert {f"{name}.csv" for name in result.tables()} <= written
    checks = json.loads((analysis_dir / "checks.json").read_text(encoding="utf-8"))
    assert set(checks) == {"calibration", "falsification", "hypotheses"}

    report = write_report(result, outputs, out)
    assert report == Path(out) / REPORT_FILENAME
    text = report.read_text(encoding="utf-8")
    for heading in ("Calibration", "Noise band", "Cell summary", "Matched-pair", "Predictive validity", "Hypothesis"):
        assert heading in text
    figures = sorted(p.name for p in (Path(out) / FIGURES_DIRNAME).iterdir())
    assert figures == ["delta_timecourse.png", "dose_response.png", "heldout_transfer.png"]
