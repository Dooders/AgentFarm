"""Tests for scripts/analyze_intrinsic_goals_pressure_sweep.py."""

from __future__ import annotations

import json
import os
import sys
import unittest
from tempfile import TemporaryDirectory

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.analyze_intrinsic_goals_pressure_sweep import (  # noqa: E402
    ARMS,
    SUMMARY_FILENAME,
    SWEEP_DIR_TEMPLATE,
    SummaryError,
    _diversity_sums,
    _load_summary,
    _normalized_diversity,
    analyze_summary,
    collect_entries,
    render_markdown,
)


def _make_summary(
    *,
    pressure: str,
    mean_pop_delta: float,
    gather_delta: float,
    start_div: float,
    end_div: float,
    num_replicates: int = 2,
) -> dict:
    """Build a minimal but structurally faithful three-arm summary payload."""

    def _paired(delta: float, dz: float, sig: bool) -> dict:
        return {
            "delta_mean": delta,
            "delta_std": 1.0,
            "delta_sem": 0.5,
            "ci95": [delta - 1.0, delta + 1.0],
            "t_stat": -3.0,
            "p_value": 0.01 if sig else 0.5,
            "cohen_dz": dz,
            "significant_p05": sig,
            "n": num_replicates,
        }

    # Per-gene diversity split across two genes so the summed value matches.
    def _div(total: float) -> dict:
        return {"reward_gather_bonus": total * 0.5, "reward_share_bonus": total * 0.5}

    arm_unique = {
        "goal_diversity_start": _div(start_div),
        "goal_diversity_end": _div(end_div),
    }
    arm_shared = {
        "goal_diversity_start": _div(0.0),
        "goal_diversity_end": _div(0.0),
    }
    arm_uniform = {
        "goal_diversity_start": _div(0.0),
        "goal_diversity_end": _div(0.0),
    }
    replicates = [
        {
            "index": i,
            "seed": 42 + i,
            "uniform": arm_uniform,
            "shared": arm_shared,
            "unique": arm_unique,
        }
        for i in range(num_replicates)
    ]

    def _delta_block(scale: float) -> dict:
        return {
            "mean_population": _paired(mean_pop_delta * scale, -5.0, True),
            "final_population": _paired((mean_pop_delta - 5) * scale, -4.0, True),
            "peak_population": _paired((mean_pop_delta - 10) * scale, -6.0, True),
            "total_births": _paired(-80.0 * scale, -3.0, True),
            "total_deaths": _paired(-40.0 * scale, -2.0, True),
            "action_share[gather]": _paired(gather_delta * scale, 6.0, True),
        }

    return {
        "config": {"selection_pressure": pressure, "arms": list(ARMS)},
        "replicates": replicates,
        "aggregate": {
            "num_replicates": num_replicates,
            "seeds": [42 + i for i in range(num_replicates)],
            "arms": list(ARMS),
            "per_arm": {
                "uniform": {"goal_diversity_end_sum": {"mean": 0.0}},
                "shared": {"goal_diversity_end_sum": {"mean": 0.0}},
                "unique": {"goal_diversity_end_sum": {"mean": end_div}},
            },
            "paired_deltas": {
                # Total effect matches the requested deltas exactly.
                "unique_minus_uniform": _delta_block(1.0),
                # Decomposition arms use scaled stand-ins for the fixture.
                "shared_minus_uniform": _delta_block(0.6),
                "unique_minus_shared": _delta_block(0.4),
            },
        },
    }


def _write_sweep(base: str, specs: dict) -> None:
    for pressure, kwargs in specs.items():
        run_dir = os.path.join(base, SWEEP_DIR_TEMPLATE.format(pressure=pressure))
        os.makedirs(run_dir, exist_ok=True)
        payload = _make_summary(pressure=pressure, **kwargs)
        with open(os.path.join(run_dir, SUMMARY_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)


class TestDiversitySums(unittest.TestCase):
    def test_averages_replicate_sums(self) -> None:
        summary = _make_summary(
            pressure="low",
            mean_pop_delta=-34.5,
            gather_delta=0.17,
            start_div=17.0,
            end_div=17.5,
            num_replicates=4,
        )
        sums = _diversity_sums(summary)
        self.assertAlmostEqual(sums["unique"]["start_sum"], 17.0)
        self.assertAlmostEqual(sums["unique"]["end_sum"], 17.5)
        self.assertAlmostEqual(sums["uniform"]["start_sum"], 0.0)


class TestAnalyzeSummary(unittest.TestCase):
    def test_extracts_paired_deltas_and_diversity(self) -> None:
        summary = _make_summary(
            pressure="high",
            mean_pop_delta=-10.0,
            gather_delta=0.05,
            start_div=17.0,
            end_div=3.0,
        )
        entry = analyze_summary("high", summary)
        self.assertEqual(entry["pressure"], "high")
        self.assertEqual(entry["num_replicates"], 2)
        total = entry["contrasts"]["unique_minus_uniform"]
        self.assertAlmostEqual(
            total["population_deltas"]["mean_population"]["delta_mean"], -10.0
        )
        self.assertAlmostEqual(total["gather_share_delta"]["delta_mean"], 0.05)
        # Decomposition contrasts are present.
        self.assertIn("shared_minus_uniform", entry["contrasts"])
        self.assertIn("unique_minus_shared", entry["contrasts"])
        self.assertAlmostEqual(entry["goal_diversity"]["unique"]["end_sum"], 3.0)
        self.assertAlmostEqual(
            entry["goal_diversity_end_sum_per_arm"]["unique"], 3.0
        )
        # Shared arm is represented in the diversity block.
        self.assertIn("shared", entry["goal_diversity"])


class TestNormalizedDiversity(unittest.TestCase):
    def test_normalizes_per_gene_std_by_span(self) -> None:
        # reward_gather_bonus / reward_share_bonus both span [0, 2].
        # _div splits the summed value evenly, so each gene's raw std is
        # end_div * 0.5; normalized by span 2 that is end_div * 0.25.
        summary = _make_summary(
            pressure="low",
            mean_pop_delta=-30.0,
            gather_delta=0.1,
            start_div=1.6,
            end_div=1.2,
            num_replicates=3,
        )
        norm = _normalized_diversity(summary)
        gather = norm["per_gene_unique"]["reward_gather_bonus"]
        self.assertAlmostEqual(gather["start"], 1.6 * 0.5 / 2.0)
        self.assertAlmostEqual(gather["end"], 1.2 * 0.5 / 2.0)
        # Shared arm is a monoculture: zero normalized diversity.
        self.assertAlmostEqual(norm["mean_across_genes"]["shared"]["end"], 0.0)
        for arm in ARMS:
            self.assertIn(arm, norm["mean_across_genes"])


class TestLoadSummary(unittest.TestCase):
    def test_rejects_summary_without_aggregate(self) -> None:
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"replicates": [], "aggregate": None}, fh)
            with self.assertRaises(SummaryError):
                _load_summary(path)


class TestCollectEntries(unittest.TestCase):
    def test_discovers_all_pressures(self) -> None:
        with TemporaryDirectory() as tmp:
            _write_sweep(
                tmp,
                {
                    "low": dict(mean_pop_delta=-34.0, gather_delta=0.17,
                                start_div=17.0, end_div=17.5),
                    "medium": dict(mean_pop_delta=-30.0, gather_delta=0.14,
                                   start_div=17.0, end_div=17.0),
                    "high": dict(mean_pop_delta=-15.0, gather_delta=0.05,
                                 start_div=17.0, end_div=3.0),
                },
            )
            pairs = [
                (p, os.path.join(tmp, SWEEP_DIR_TEMPLATE.format(pressure=p),
                                 SUMMARY_FILENAME))
                for p in ("low", "medium", "high")
            ]
            entries, skipped = collect_entries(pairs)
            self.assertEqual([e["pressure"] for e in entries], ["low", "medium", "high"])
            self.assertEqual(skipped, [])

    def test_tolerates_missing_pressure(self) -> None:
        with TemporaryDirectory() as tmp:
            _write_sweep(
                tmp,
                {
                    "low": dict(mean_pop_delta=-34.0, gather_delta=0.17,
                                start_div=17.0, end_div=17.5),
                },
            )
            pairs = [
                (p, os.path.join(tmp, SWEEP_DIR_TEMPLATE.format(pressure=p),
                                 SUMMARY_FILENAME))
                for p in ("low", "medium", "high")
            ]
            entries, skipped = collect_entries(pairs)
            self.assertEqual([e["pressure"] for e in entries], ["low"])
            self.assertEqual([s["pressure"] for s in skipped], ["medium", "high"])
            self.assertTrue(all(s["reason"] == "missing" for s in skipped))


class TestRenderMarkdown(unittest.TestCase):
    def test_table_has_rows_and_json_serializable(self) -> None:
        summary = _make_summary(
            pressure="low",
            mean_pop_delta=-34.5,
            gather_delta=0.169,
            start_div=17.0,
            end_div=17.5,
        )
        entry = analyze_summary("low", summary)
        md = render_markdown([entry])
        self.assertIn("mean_population", md)
        self.assertIn("gather-share", md)
        self.assertIn("| low |", md)
        # The starred significant delta should render.
        self.assertIn("*", md)
        # The decomposition contrast headers and normalized-diversity sections.
        self.assertIn("mean-shift off default", md)
        self.assertIn("heterogeneity", md)
        self.assertIn("span-normalized", md)

    def test_empty_entries(self) -> None:
        md = render_markdown([])
        self.assertIn("No summaries found", md)


if __name__ == "__main__":
    unittest.main()
