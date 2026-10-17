"""Registered hypothesis tests and descriptive summaries.

All confirmatory contrasts are *seed-paired*: every cell shares the seed's
founders, shocks and drought phase, so a contrast is computed per seed and
tested across seeds with a one-sided Wilcoxon signed-rank test. The six
confirmatory tests are Holm-corrected together at :data:`ALPHA`.

Metrics: ``bond_rmst`` (restricted mean survival time of bond units, ticks),
``welfare_d_lifespan`` (mean founder lifespan minus its solo twin's, ticks)
and ``trap_rate`` (share of member-steps spent wanting out but priced in).
Grid contrasts average over the three scope levels unless stated.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from farm.experiments.cooperation.config import (
    ALPHA,
    PRIMARY_EXIT_COST,
    PRIMARY_RECOGNITION,
    PRIMARY_SCOPE,
    grid_cell_name,
)
from farm.experiments.cooperation.experiment import MatrixOutputs
from farm.experiments.cooperation.metrics import MODE_COLUMNS

BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 20261016


@dataclass
class TestResult:
    key: str
    statement: str
    metric: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    positive_seeds: int
    p_value: float
    p_holm: float = float("nan")
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    confirmatory: list[TestResult]
    verdicts: dict[str, str]
    manipulation_checks: list[TestResult]
    grid: pd.DataFrame
    conditions: pd.DataFrame
    who_benefits: pd.DataFrame
    modes: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "confirmatory": [t.to_dict() for t in self.confirmatory],
            "verdicts": self.verdicts,
            "manipulation_checks": [t.to_dict() for t in self.manipulation_checks],
            "notes": self.notes,
        }


def _pivot(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    return runs.pivot_table(index="seed", columns="condition", values=metric, aggfunc="mean")


def _grid_mean(table: pd.DataFrame, exit_cost: float, recognition: float, scopes: Sequence[float] = PRIMARY_SCOPE):
    cols = [grid_cell_name(s, exit_cost, recognition) for s in scopes]
    return table[cols].mean(axis=1)


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if diff.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, diff.size, size=(BOOTSTRAP_RESAMPLES, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_test(key: str, statement: str, metric: str, diff: pd.Series, alternative: str = "greater") -> TestResult:
    d = diff.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi = _bootstrap_ci(d, rng)
    if d.size == 0 or np.allclose(d, 0.0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue)
    return TestResult(
        key=key,
        statement=statement,
        metric=metric,
        n=int(d.size),
        mean_diff=float(d.mean()) if d.size else float("nan"),
        ci_low=lo,
        ci_high=hi,
        positive_seeds=int((d > 0).sum()),
        p_value=p,
    )


def holm(results: list[TestResult], alpha: float = ALPHA) -> None:
    """Holm step-down correction, in place."""
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    running = 0.0
    stop = False
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i].p_value)
        running = max(running, adj)
        results[i].p_holm = running
        if not stop and results[i].p_value <= alpha / (m - rank):
            results[i].significant = True
        else:
            stop = True
            results[i].significant = False


HYPOTHESES: dict[str, str] = {
    "H1": "With full recognition, a higher exit cost (0.6 vs 0) lengthens bond life.",
    "H2": "Broad scope needs a high exit cost: the scope penalty on bond life at exit cost 0 shrinks at exit cost 0.6 (full recognition).",
    "H3": "With no exit cost, full recognition still lengthens bond life (by blocking outside offers).",
    "H4a": "Founder welfare is higher at a moderate exit cost (0.3) than at none (full recognition).",
    "H4b": "Founder welfare is higher at a moderate exit cost (0.3) than at a high one (0.6) (full recognition).",
    "H5": "Trapped time rises with exit cost (0.6 vs 0.3, full recognition).",
}


def confirmatory_tests(runs: pd.DataFrame) -> list[TestResult]:
    rmst = _pivot(runs, "bond_rmst")
    welfare = _pivot(runs, "welfare_d_lifespan")
    trap = _pivot(runs, "trap_rate")
    r_hi, r_lo = max(PRIMARY_RECOGNITION), min(PRIMARY_RECOGNITION)
    e_lo, e_mid, e_hi = PRIMARY_EXIT_COST
    s_lo, s_hi = min(PRIMARY_SCOPE), max(PRIMARY_SCOPE)

    def cell(table, s, e, r):
        return table[grid_cell_name(s, e, r)]

    tests = [
        paired_test("H1", HYPOTHESES["H1"], "bond_rmst", _grid_mean(rmst, e_hi, r_hi) - _grid_mean(rmst, e_lo, r_hi)),
        paired_test(
            "H2",
            HYPOTHESES["H2"],
            "bond_rmst",
            (cell(rmst, s_hi, e_hi, r_hi) - cell(rmst, s_lo, e_hi, r_hi))
            - (cell(rmst, s_hi, e_lo, r_hi) - cell(rmst, s_lo, e_lo, r_hi)),
        ),
        paired_test("H3", HYPOTHESES["H3"], "bond_rmst", _grid_mean(rmst, e_lo, r_hi) - _grid_mean(rmst, e_lo, r_lo)),
        paired_test(
            "H4a",
            HYPOTHESES["H4a"],
            "welfare_d_lifespan",
            _grid_mean(welfare, e_mid, r_hi) - _grid_mean(welfare, e_lo, r_hi),
        ),
        paired_test(
            "H4b",
            HYPOTHESES["H4b"],
            "welfare_d_lifespan",
            _grid_mean(welfare, e_mid, r_hi) - _grid_mean(welfare, e_hi, r_hi),
        ),
        paired_test("H5", HYPOTHESES["H5"], "trap_rate", _grid_mean(trap, e_hi, r_hi) - _grid_mean(trap, e_mid, r_hi)),
    ]
    holm(tests)
    return tests


def manipulation_checks(runs: pd.DataFrame) -> list[TestResult]:
    """Checks that the settings do what they claim (not part of the family)."""
    rmst = _pivot(runs, "bond_rmst")
    trap = _pivot(runs, "trap_rate")
    e_lo, _, e_hi = PRIMARY_EXIT_COST
    r_lo = min(PRIMARY_RECOGNITION)
    return [
        paired_test(
            "M1",
            "Unrecognised exit cost is inert: bond life at exit cost 0.6 vs 0 with no recognition (two-sided).",
            "bond_rmst",
            _grid_mean(rmst, e_hi, r_lo) - _grid_mean(rmst, e_lo, r_lo),
            alternative="two-sided",
        ),
        paired_test(
            "M2",
            "No exit payment, no trap: trap rate with exit cost 0 is zero at every recognition level (two-sided).",
            "trap_rate",
            pd.concat([_grid_mean(trap, e_lo, r) for r in PRIMARY_RECOGNITION], axis=1).mean(axis=1),
            alternative="two-sided",
        ),
    ]


def verdicts(tests: list[TestResult]) -> dict[str, str]:
    by_key = {t.key: t for t in tests}

    def word(ok: bool) -> str:
        return "supported" if ok else "not supported"

    out = {k: word(by_key[k].significant) for k in ("H1", "H2", "H3", "H5")}
    out["H4"] = word(by_key["H4a"].significant and by_key["H4b"].significant)
    return out


DESCRIPTIVE_METRICS: tuple[str, ...] = (
    "bond_rmst",
    "bond_survival_100",
    "coverage_mean",
    "welfare_d_lifespan",
    "welfare_better_share",
    "welfare_worse_share",
    "welfare_starved",
    "welfare_starved_solo",
    "trap_rate",
    "effort_mean",
    "population_mean",
    "switches",
    "offers_blocked",
    "surplus_share",
    "pooled_share",
)


def condition_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DESCRIPTIVE_METRICS if c in runs.columns]
    mode_cols = [f"bond_primary_{m}" for m in MODE_COLUMNS if f"bond_primary_{m}" in runs.columns]
    meta = runs.groupby("condition")[["family", "scope", "exit_cost", "recognition", "symmetry", "nesting"]].first()
    means = runs.groupby("condition")[cols + mode_cols].mean()
    sems = runs.groupby("condition")[["bond_rmst", "welfare_d_lifespan"]].sem().add_suffix("_sem")
    return meta.join(means).join(sems)


def grid_table(conditions: pd.DataFrame) -> pd.DataFrame:
    grid = conditions[conditions["family"] == "grid"].copy()
    return grid.sort_values(["recognition", "exit_cost", "scope"])


def who_benefits(founders: pd.DataFrame, solo_founders_skill: pd.Series | None = None) -> pd.DataFrame:
    """Mean lifespan gain by founder skill tercile and condition.

    Terciles are cut on the pooled founder skill distribution (founders are
    the same agents in every condition for a given seed).
    """
    df = founders.copy()
    edges = np.quantile(df["skill"], [1 / 3, 2 / 3])
    df["skill_tercile"] = np.select(
        [df["skill"] <= edges[0], df["skill"] <= edges[1]], ["low", "middle"], default="high"
    )
    table = df.groupby(["condition", "skill_tercile"]).agg(
        d_lifespan=("d_lifespan", "mean"),
        starved=("starved", "mean"),
        starved_solo=("starved_solo", "mean"),
        member_share=("member_share", "mean"),
    )
    return table.reset_index()


def mode_composition(units: pd.DataFrame) -> pd.DataFrame:
    bonds = units[(units["kind"] == "bond") & (units["primary_mode"] != "indeterminate")]
    comp = bonds.groupby("condition")["primary_mode"].value_counts(normalize=True).unstack(fill_value=0.0)
    return comp


def analyze(outputs: MatrixOutputs) -> AnalysisResult:
    runs = outputs.runs
    notes: list[str] = []
    grid_names = {
        grid_cell_name(s, e, r) for s in PRIMARY_SCOPE for e in PRIMARY_EXIT_COST for r in PRIMARY_RECOGNITION
    }
    have_grid = grid_names.issubset(set(runs["condition"]))
    if have_grid:
        tests = confirmatory_tests(runs)
        checks = manipulation_checks(runs)
        verdict = verdicts(tests)
    else:
        tests, checks, verdict = [], [], {}
        notes.append("primary grid incomplete; confirmatory tests skipped")
    conditions = condition_table(runs)
    return AnalysisResult(
        confirmatory=tests,
        verdicts=verdict,
        manipulation_checks=checks,
        grid=grid_table(conditions),
        conditions=conditions,
        who_benefits=who_benefits(outputs.founders),
        modes=mode_composition(outputs.units),
        notes=notes,
    )


def write_analysis(result: AnalysisResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    result.conditions.to_csv(out / "conditions_summary.csv")
    result.who_benefits.to_csv(out / "who_benefits.csv", index=False)
    result.modes.to_csv(out / "mode_composition.csv")
