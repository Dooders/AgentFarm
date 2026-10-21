"""Tests for scripts/merge_intrinsic_cooperation_runs.py.

Shards are synthetic per-replicate summaries in the shape the runner writes;
the merge must reproduce what one multi-replicate invocation would produce.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

pytest.importorskip("scipy")

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.runners import intrinsic_cooperation_experiment as runner
from farm.runners.intrinsic_cooperation_experiment import (
    ARM_NAMES,
    HYPOTHESES,
    MANIPULATION_CHECKS,
    SETTING_GENE_NAMES,
    aggregate_summaries,
)
from scripts.merge_intrinsic_cooperation_runs import (
    HYPOTHESES_TABLE_FILE,
    MERGED_SUMMARY_FILE,
    hypotheses_table,
    load_shards,
    main,
    merge_shards,
)

pytestmark = pytest.mark.unit

_SCALARS = (
    "mean_population",
    "final_population",
    "peak_population",
    "total_births",
    "total_deaths",
    "mean_coverage",
    "final_coverage",
    "mean_units_active",
    "units_formed",
    "mean_unit_lifetime",
    "bond_share_late",
    "policy_bond_prob_late",
    "mean_lifespan_at_death",
    "max_generation",
    "change_expected_exit_payment",
)


def _arm_summary(offset: float) -> dict:
    return {
        **{name: 10.0 + offset for name in _SCALARS},
        "mean_action_share": {"move": 0.5, "bond": 0.1 + offset / 100},
        "gene_mean_end": {gene: 0.5 + offset for gene in SETTING_GENE_NAMES},
        "gene_mean_change": {gene: offset for gene in SETTING_GENE_NAMES},
    }


def _config(seed: int, **overrides) -> dict:
    cfg = {"num_steps": 50, "seed": seed, "num_replicates": 1, "arms": list(ARM_NAMES), "alpha": 0.05,
           "max_population": 60, "cooperation": {"seed": seed, "output_dir": f"/out/seed_{seed}", "bond_range": 2.0}}
    cfg.update(overrides)
    return cfg


def _shard(seed: int, path: str = "", **overrides) -> dict:
    # Cooperation arms score higher than placebo/baseline, with a seed-dependent wobble.
    wobble = (seed % 3) * 0.1
    replicate = {
        "index": 0,
        "seed": seed,
        "baseline": _arm_summary(0.0 + wobble),
        "placebo": _arm_summary(0.2 + wobble),
        "cooperation": _arm_summary(1.0 + wobble),
        "cooperation_no_motive": _arm_summary(0.6 + wobble),
    }
    return {"config": _config(seed, **overrides), "replicates": [replicate], "aggregate": None, "hypotheses": None,
            "_path": path or f"shard_{seed}.json"}


def test_merge_reindexes_sorts_and_recomputes_like_one_run():
    shards = [_shard(105), _shard(101), _shard(103)]
    merged = merge_shards(shards)

    assert merged["config"]["seeds"] == [101, 103, 105]
    assert merged["config"]["seed"] == 101
    assert merged["config"]["num_replicates"] == 3
    assert [r["index"] for r in merged["replicates"]] == [0, 1, 2]
    assert merged["sources"] == ["shard_105.json", "shard_101.json", "shard_103.json"]

    expected_aggregate = aggregate_summaries(merged["replicates"], ARM_NAMES)
    expected_tests = runner.test_hypotheses(merged["replicates"], ARM_NAMES, 0.05)
    assert merged["aggregate"] == expected_aggregate
    assert merged["hypotheses"] == expected_tests
    assert {h["key"] for h in merged["hypotheses"]} == set(HYPOTHESES) | set(MANIPULATION_CHECKS)
    assert all(h["p_holm"] is None for h in merged["hypotheses"] if h.get("manipulation_check"))


def test_single_shard_has_no_tests():
    merged = merge_shards([_shard(101)])
    assert merged["aggregate"] is None and merged["hypotheses"] is None
    assert "fewer than two replicates" in hypotheses_table(merged["hypotheses"])


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError, match="duplicate seeds.*101"):
        merge_shards([_shard(101, "a.json"), _shard(101, "b.json")])


def test_config_mismatch_is_rejected_unless_allowed(capsys):
    shards = [_shard(101), _shard(102, num_steps=99)]
    with pytest.raises(ValueError, match="run configuration differs"):
        merge_shards(shards)
    merged = merge_shards(shards, allow_config_mismatch=True)
    assert merged["config"]["num_replicates"] == 2
    assert "warning" in capsys.readouterr().err


def test_shard_specific_keys_do_not_count_as_mismatch():
    # seed, num_replicates and cooperation.{seed,output_dir} differ between shards by construction.
    merge_shards([_shard(101), _shard(102, num_replicates=5)])


def test_hypotheses_table_lists_every_test():
    merged = merge_shards([_shard(101), _shard(102), _shard(103)])
    table = hypotheses_table(merged["hypotheses"])
    for key in HYPOTHESES:
        assert f"| {key} |" in table
    for key in MANIPULATION_CHECKS:
        assert f"| {key} (check) |" in table


def test_main_writes_summary_and_table():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for seed in (101, 102):
            path = os.path.join(tmp, f"seed_{seed}", "intrinsic_cooperation_summary.json")
            os.makedirs(os.path.dirname(path))
            shard = _shard(seed)
            del shard["_path"]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(shard, handle)
            paths.append(path)
        out = os.path.join(tmp, "merged")

        assert main([os.path.join(tmp, "seed_*", "intrinsic_cooperation_summary.json"), "--output-dir", out]) == 0

        with open(os.path.join(out, MERGED_SUMMARY_FILE), encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["config"]["seeds"] == [101, 102]
        assert payload["sources"] == paths
        assert len(payload["hypotheses"]) == len(HYPOTHESES) + len(MANIPULATION_CHECKS)
        with open(os.path.join(out, HYPOTHESES_TABLE_FILE), encoding="utf-8") as handle:
            assert "| S2 |" in handle.read()


def test_load_shards_rejects_non_summaries():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"config": {}}, handle)
        with pytest.raises(ValueError, match="missing 'replicates'"):
            load_shards([path])
