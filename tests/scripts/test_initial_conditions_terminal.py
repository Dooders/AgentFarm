"""Unit tests for the initial-conditions terminal-state sweep and its analysis.

These tests build commands and synthetic arm results. They do not run a simulation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.analyze_initial_conditions_terminal import (
    CONTRAST_METRICS,
    RunRow,
    contrasts_for_metric,
    discover_runs,
    mean_interval,
    series_metrics,
    slope_for_offspring,
    write_outputs,
)
from scripts.run_initial_conditions_terminal import (
    ARM,
    DEFAULT_SEEDS,
    FOUNDER_LEVELS,
    INITIAL_RESOURCE_COUNT,
    MAX_POPULATION,
    OFFSPRING_LEVELS,
    POLICY_TEMPERATURE,
    SELECTION_PRESSURE,
    Cell,
    build_cell_command,
    cell_is_complete,
    execute_cell,
    iter_cells,
    main,
)

SEEDS = (42, 7, 19, 101, 137, 256)
FOUNDERS = (5, 8, 12, 20)


def test_registered_grid_is_eight_cells_by_six_seeds():
    cells = iter_cells(output_dir=Path("experiments/initial_conditions_terminal"))
    assert len(cells) == 8 * 6
    assert len({(cell.founder, cell.offspring) for cell in cells}) == 8
    assert {cell.seed for cell in cells} == set(DEFAULT_SEEDS)
    assert {cell.founder for cell in cells} == set(FOUNDER_LEVELS)
    assert {cell.offspring for cell in cells} == set(OFFSPRING_LEVELS)


def test_child_command_is_baseline_only_and_pins_the_food_budget(tmp_path):
    cell = iter_cells(tmp_path, founders=(12,), offsprings=(8,), seeds=(42,))[0]
    command = build_cell_command(cell, tmp_path / "run_intrinsic_cooperation_experiment.py", python="python")
    assert command[command.index("--arms") + 1] == ARM
    assert "placebo" not in command
    assert "cooperation" not in command
    assert command[command.index("--initial-agent-resource-level") + 1] == "12"
    assert command[command.index("--offspring-initial-resources") + 1] == "8"
    assert command[command.index("--initial-resource-count") + 1] == str(INITIAL_RESOURCE_COUNT)
    assert command[command.index("--max-population") + 1] == str(MAX_POPULATION)
    assert command[command.index("--selection-pressure") + 1] == SELECTION_PRESSURE
    assert command[command.index("--policy-temperature") + 1] == str(POLICY_TEMPERATURE)
    assert command[command.index("--num-replicates") + 1] == "1"
    assert cell.run_dir == tmp_path / "founder_12" / "offspring_8" / "seed_42"


def test_dry_run_prints_48_commands_and_writes_nothing(tmp_path, capsys):
    output = tmp_path / "sweep"
    code = main(["--dry-run", "--output-dir", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    assert not output.exists()
    assert captured.err.strip() == "48 cells"
    assert captured.out.count("--arms baseline") == 48
    assert "--offspring-initial-resources 5" in captured.out
    assert "--offspring-initial-resources 8" in captured.out


def test_completed_cell_is_skipped_and_a_changed_plan_is_not(tmp_path):
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "arm = out / 'baseline'\n"
        "arm.mkdir(parents=True, exist_ok=True)\n"
        "(arm / 'arm_result.json').write_text('{}')\n"
        "(out / 'leftover.txt').write_text('kept')\n",
        encoding="utf-8",
    )
    cell = Cell(founder=12, offspring=5, seed=42, output_dir=tmp_path / "sweep", num_steps=10, environment="development")
    junk = cell.run_dir / "partial.db"
    junk.parent.mkdir(parents=True)
    junk.write_text("partial", encoding="utf-8")

    first = execute_cell(cell, script)
    assert first["status"] == "ok"
    assert cell_is_complete(cell)
    assert not junk.exists()
    assert (cell.run_dir / "leftover.txt").is_file()

    second = execute_cell(cell, script)
    assert second["status"] == "skipped"

    changed = Cell(
        founder=12, offspring=5, seed=42, output_dir=cell.output_dir, num_steps=11, environment="development"
    )
    assert not cell_is_complete(changed)


def test_failed_child_is_recorded(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("import sys\nraise SystemExit(1)\n", encoding="utf-8")
    cell = Cell(founder=5, offspring=5, seed=7, output_dir=tmp_path / "sweep", num_steps=10, environment="development")
    record = execute_cell(cell, script)
    assert record["status"] == "failed"
    assert record["returncode"] == 1


def test_series_metrics_use_the_late_quarter_and_the_first_200_steps():
    steps = list(range(400))
    population = [10.0] * 400
    population[50] = 40.0
    population[250] = 80.0
    for index in range(300, 400):
        population[index] = 5.0
    metrics = series_metrics(steps, population)
    assert metrics["peak_population"] == 80.0
    assert metrics["peak_step"] == 250
    assert metrics["late_window_mean"] == pytest.approx(5.0)
    assert metrics["overshoot"] == pytest.approx(75.0)
    assert metrics["oscillation_amplitude"] == pytest.approx(30.0)


def test_mean_interval_widens_for_six_seeds():
    interval = mean_interval([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert interval.n == 6
    assert interval.mean == pytest.approx(3.5)
    assert interval.low < interval.mean < interval.high
    assert interval.low == pytest.approx(1.536, abs=0.01)
    assert interval.high == pytest.approx(5.464, abs=0.01)


def _row(founder: int, offspring: int, seed: int, late: float, overshoot: float, lifespan: float, generation: float) -> RunRow:
    return RunRow(
        founder=founder,
        offspring=offspring,
        seed=seed,
        peak_population=late + overshoot,
        peak_step=100,
        overshoot=overshoot,
        oscillation_amplitude=overshoot,
        final_population=late,
        late_window_mean=late,
        starvation_deaths=10,
        mean_lifespan_at_death=lifespan,
        max_generation=generation,
        total_births=20,
        n_steps=3000,
    )


def _grid(late_for_offspring: dict[int, float], lifespan_for_offspring: dict[int, float], generation_for_offspring: dict[int, float], overshoot_of_founder) -> list[RunRow]:
    return [
        _row(
            founder,
            offspring,
            seed,
            late_for_offspring[offspring],
            overshoot_of_founder(founder),
            lifespan_for_offspring[offspring],
            generation_for_offspring[offspring],
        )
        for founder in FOUNDERS
        for offspring in (5, 8)
        for seed in SEEDS
    ]


def test_flat_late_window_covers_zero_and_offspring_contrast_is_directional():
    rows = _grid(
        late_for_offspring={5: 30.0, 8: 50.0},
        lifespan_for_offspring={5: 40.0, 8: 55.0},
        generation_for_offspring={5: 2.0, 8: 4.0},
        overshoot_of_founder=lambda founder: float(founder),
    )
    for offspring in (5, 8):
        slope = slope_for_offspring(rows, offspring)
        assert slope.covers_zero
        assert slope.directional is None
        assert slope.slope == pytest.approx(0.0)

    late = contrasts_for_metric(rows, "late_window_mean")
    assert [contrast.founder for contrast in late] == list(FOUNDERS)
    for contrast in late:
        assert contrast.mean == pytest.approx(20.0)
        assert contrast.excludes_zero
        assert contrast.n_positive == 6
        assert contrast.directional == "positive"
    for metric in ("mean_lifespan_at_death", "max_generation"):
        assert set(CONTRAST_METRICS) >= {metric}
        assert all(contrast.directional == "positive" for contrast in contrasts_for_metric(rows, metric))


def _slope_grid(negative_seeds: set[int]) -> list[RunRow]:
    """Most seeds rise one-for-one with overshoot. A listed seed slopes slightly down."""
    rows = []
    for founder in FOUNDERS:
        overshoot = float(founder)
        for seed in SEEDS:
            late = -0.2 * overshoot if seed in negative_seeds else 2.0 * overshoot
            rows.append(_row(founder, 5, seed, late, overshoot, 40.0, 2.0))
    return rows


def test_slope_that_excludes_zero_needs_five_of_six_seeds():
    slope = slope_for_offspring(_slope_grid({256}), 5)
    assert not slope.covers_zero
    assert slope.slope > 0.0
    assert slope.seeds_agreeing == 5
    assert slope.directional == "positive"

    # A second disagreeing seed keeps the interval off zero, but 4 of 6 is not enough.
    weaker = slope_for_offspring(_slope_grid({256, 137}), 5)
    assert not weaker.covers_zero
    assert weaker.seeds_agreeing == 4
    assert weaker.directional is None


def test_write_outputs_reads_arm_results(tmp_path):
    population = [10.0, 30.0, 12.0, 12.0, 11.0, 11.0, 11.0, 11.0]
    for founder in (5, 12):
        for offspring in (5, 8):
            for seed in SEEDS:
                run_dir = tmp_path / f"founder_{founder}" / f"offspring_{offspring}" / f"seed_{seed}"
                plan = {
                    "founder_initial_resources": founder,
                    "offspring_initial_resources": offspring,
                    "seed": seed,
                }
                (run_dir / "baseline").mkdir(parents=True)
                (run_dir / "cell.json").write_text(json.dumps(plan), encoding="utf-8")
                arm = {
                    "arm": "baseline",
                    "summary": {
                        "final_population": 11,
                        "starvation_deaths": 4,
                        "mean_lifespan_at_death": 40 + offspring,
                        "max_generation": 2 if offspring == 5 else 3,
                        "total_births": 9,
                    },
                    "series": {"steps": list(range(len(population))), "population": population},
                }
                (run_dir / "baseline" / "arm_result.json").write_text(json.dumps(arm), encoding="utf-8")

    loaded = discover_runs(tmp_path)
    assert len(loaded) == 2 * 2 * 6
    figure = tmp_path / "devlog.png"
    payload = write_outputs(tmp_path, loaded, devlog_figure=figure)
    aggregate = tmp_path / "aggregate"
    assert (aggregate / "runs.csv").is_file()
    assert (aggregate / "summary.md").is_file()
    assert (aggregate / "population_trajectories.png").stat().st_size > 0
    assert figure.stat().st_size > 0
    assert payload["n_runs"] == 24
    assert "food budget" in (aggregate / "summary.md").read_text(encoding="utf-8")
