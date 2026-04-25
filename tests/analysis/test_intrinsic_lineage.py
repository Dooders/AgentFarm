"""Tests for intrinsic lineage loader and summary helpers.

Covers:
- load_intrinsic_snapshots: JSONL parsing, missing file, directory dispatch,
  malformed lines
- flatten_snapshots_to_agent_records: deduplication, birth/death time
  inference, parent_id normalisation
- build_intrinsic_lineage_dag: end-to-end from JSONL file
- compute_surviving_lineage_count_over_time: per-step founder count
- compute_lineage_depth_over_time: per-step depth stats
- extract_chromosomes_from_snapshots: last-seen chromosome extraction
- plot_intrinsic_lineage_tree: smoke test with gene colouring
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from farm.analysis.phylogenetics.intrinsic_loader import (
    load_intrinsic_snapshots,
    flatten_snapshots_to_agent_records,
    build_intrinsic_lineage_dag,
    compute_surviving_lineage_count_over_time,
    compute_lineage_depth_over_time,
    extract_chromosomes_from_snapshots,
)
from farm.analysis.phylogenetics.compute import PhylogeneticTree


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------


def _make_snapshot(step: int, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"step": step, "agents": agents}


def _make_agent(
    agent_id: str,
    parent_ids: List[str],
    generation: int = 0,
    chromosome: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    return {
        "agent_id": agent_id,
        "agent_type": "system",
        "generation": generation,
        "parent_ids": parent_ids,
        "chromosome": chromosome or {"learning_rate": 0.01, "gamma": 0.99},
    }


def _write_snapshots(path: Path, snapshots: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for snap in snapshots:
            fh.write(json.dumps(snap) + "\n")


# ---------------------------------------------------------------------------
# load_intrinsic_snapshots
# ---------------------------------------------------------------------------


class TestLoadIntrinsicSnapshots:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_intrinsic_snapshots(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_directory_dispatch(self, tmp_path):
        snap_file = tmp_path / "intrinsic_gene_snapshots.jsonl"
        _write_snapshots(snap_file, [_make_snapshot(0, [_make_agent("a1", [])])])
        result = load_intrinsic_snapshots(tmp_path)  # pass directory
        assert len(result) == 1
        assert result[0]["step"] == 0

    def test_basic_load(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [])]),
            _make_snapshot(100, [_make_agent("a1", []), _make_agent("a2", ["a1"])]),
        ]
        _write_snapshots(snap_file, snaps)
        result = load_intrinsic_snapshots(snap_file)
        assert len(result) == 2
        assert result[0]["step"] == 0
        assert result[1]["step"] == 100

    def test_malformed_line_skipped(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        with snap_file.open("w") as fh:
            fh.write('{"step": 0, "agents": []}\n')
            fh.write("NOT JSON\n")
            fh.write('{"step": 100, "agents": []}\n')
        result = load_intrinsic_snapshots(snap_file)
        assert len(result) == 2

    def test_empty_file_returns_empty(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snap_file.write_text("")
        result = load_intrinsic_snapshots(snap_file)
        assert result == []


# ---------------------------------------------------------------------------
# flatten_snapshots_to_agent_records
# ---------------------------------------------------------------------------


class TestFlattenSnapshots:
    def test_empty_snapshots_returns_empty(self):
        assert flatten_snapshots_to_agent_records([]) == []

    def test_single_snapshot_all_alive(self):
        snaps = [_make_snapshot(0, [_make_agent("a1", []), _make_agent("a2", ["a1"])])]
        records = flatten_snapshots_to_agent_records(snaps)
        assert len(records) == 2
        ids = {r["agent_id"] for r in records}
        assert ids == {"a1", "a2"}
        # Both alive at final step → death_time == None
        for r in records:
            assert r["death_time"] is None

    def test_birth_time_is_first_seen_step(self):
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [])]),
            _make_snapshot(100, [_make_agent("a1", []), _make_agent("a2", ["a1"])]),
        ]
        records = flatten_snapshots_to_agent_records(snaps)
        by_id = {r["agent_id"]: r for r in records}
        assert by_id["a1"]["birth_time"] == 0
        assert by_id["a2"]["birth_time"] == 100

    def test_death_time_inferred_for_absent_agent(self):
        """Agent absent at final step should have approximate death_time."""
        snaps = [
            _make_snapshot(0, [_make_agent("a1", []), _make_agent("a2", ["a1"])]),
            _make_snapshot(100, [_make_agent("a1", [])]),  # a2 absent
        ]
        records = flatten_snapshots_to_agent_records(snaps)
        by_id = {r["agent_id"]: r for r in records}
        assert by_id["a1"]["death_time"] is None  # still alive at step 100
        # a2 last seen at step 0; next step is 100
        assert by_id["a2"]["death_time"] == 100

    def test_agent_alive_at_final_step_has_none_death(self):
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [])]),
            _make_snapshot(200, [_make_agent("a1", [])]),
        ]
        records = flatten_snapshots_to_agent_records(snaps)
        assert records[0]["death_time"] is None

    def test_deduplication(self):
        """Same agent_id across multiple steps results in one record."""
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [])]),
            _make_snapshot(100, [_make_agent("a1", [])]),
            _make_snapshot(200, [_make_agent("a1", [])]),
        ]
        records = flatten_snapshots_to_agent_records(snaps)
        assert len(records) == 1
        assert records[0]["agent_id"] == "a1"

    def test_chromosome_from_last_snapshot(self):
        """The chromosome in the record should be from the last-seen snapshot."""
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [], chromosome={"learning_rate": 0.01})]),
            _make_snapshot(100, [_make_agent("a1", [], chromosome={"learning_rate": 0.02})]),
        ]
        records = flatten_snapshots_to_agent_records(snaps)
        lr = records[0]["chromosome"]["learning_rate"]
        assert lr == pytest.approx(0.02)
        # Confirm the initial value (0.01) was overwritten, not retained
        assert lr != pytest.approx(0.01)

    def test_parent_ids_normalised(self):
        snaps = [_make_snapshot(0, [_make_agent("a1", ["p1", "p2"])])]
        records = flatten_snapshots_to_agent_records(snaps)
        assert records[0]["parent_ids"] == ["p1", "p2"]

    def test_none_parent_ids_filtered(self):
        agent = _make_agent("a1", [])
        agent["parent_ids"] = [None, "p1"]
        snaps = [_make_snapshot(0, [agent])]
        records = flatten_snapshots_to_agent_records(snaps)
        assert records[0]["parent_ids"] == ["p1"]


# ---------------------------------------------------------------------------
# build_intrinsic_lineage_dag
# ---------------------------------------------------------------------------


class TestBuildIntrinsicLineageDag:
    def test_missing_file_returns_empty_tree(self, tmp_path):
        tree = build_intrinsic_lineage_dag(tmp_path / "nonexistent.jsonl")
        assert isinstance(tree, PhylogeneticTree)
        assert tree.nodes == {}

    def test_basic_lineage(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [_make_agent("founder", [])]),
            _make_snapshot(100, [
                _make_agent("founder", []),
                _make_agent("child1", ["founder"], generation=1),
                _make_agent("child2", ["founder"], generation=1),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        assert "founder" in tree.nodes
        assert "child1" in tree.nodes
        assert "child2" in tree.nodes
        assert tree.nodes["founder"].is_root is True
        assert tree.nodes["child1"].depth == 1
        assert tree.nodes["child2"].depth == 1

    def test_dag_when_two_parents(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [
                _make_agent("p1", []),
                _make_agent("p2", []),
            ]),
            _make_snapshot(100, [
                _make_agent("p1", []),
                _make_agent("p2", []),
                _make_agent("child", ["p1", "p2"], generation=1),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        assert tree.is_dag is True
        assert tree.nodes["child"].parent_ids == ["p1", "p2"]

    def test_directory_path_dispatch(self, tmp_path):
        snap_file = tmp_path / "intrinsic_gene_snapshots.jsonl"
        snaps = [_make_snapshot(0, [_make_agent("a1", [])])]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(tmp_path)
        assert "a1" in tree.nodes

    def test_deterministic_roots(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [
                _make_agent("z_founder", []),
                _make_agent("a_founder", []),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        assert tree.roots == sorted(tree.roots)


# ---------------------------------------------------------------------------
# compute_surviving_lineage_count_over_time
# ---------------------------------------------------------------------------


class TestSurvivingLineageCountOverTime:
    def test_empty_inputs(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        assert compute_surviving_lineage_count_over_time(tree, []) == []

    def test_single_lineage_throughout(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [_make_agent("root", [])]),
            _make_snapshot(100, [
                _make_agent("root", []),
                _make_agent("child", ["root"], generation=1),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_surviving_lineage_count_over_time(tree, raw)
        # Both steps should show 1 lineage (root's lineage)
        assert len(result) == 2
        steps = [r[0] for r in result]
        counts = [r[1] for r in result]
        assert steps == [0, 100]
        assert all(c == 1 for c in counts)

    def test_two_independent_lineages(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [
                _make_agent("r1", []),
                _make_agent("r2", []),
            ]),
            _make_snapshot(100, [
                _make_agent("r1", []),
                _make_agent("c1", ["r1"], generation=1),
                _make_agent("r2", []),
                _make_agent("c2", ["r2"], generation=1),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_surviving_lineage_count_over_time(tree, raw)
        counts = [r[1] for r in result]
        assert all(c == 2 for c in counts)

    def test_lineage_extinction(self, tmp_path):
        """When a lineage dies out between snapshots, count drops."""
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [
                _make_agent("r1", []),
                _make_agent("r2", []),
            ]),
            _make_snapshot(100, [
                _make_agent("r1", []),  # r2 lineage extinct
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_surviving_lineage_count_over_time(tree, raw)
        assert result[0] == (0, 2)
        assert result[1] == (100, 1)

    def test_sorted_by_step(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        # Write in reverse order to verify sorting
        snaps_reversed = [
            _make_snapshot(200, [_make_agent("r1", [])]),
            _make_snapshot(100, [_make_agent("r1", [])]),
            _make_snapshot(0, [_make_agent("r1", [])]),
        ]
        with snap_file.open("w") as fh:
            for s in snaps_reversed:
                fh.write(json.dumps(s) + "\n")
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_surviving_lineage_count_over_time(tree, raw)
        steps = [r[0] for r in result]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# compute_lineage_depth_over_time
# ---------------------------------------------------------------------------


class TestLineageDepthOverTime:
    def test_empty_returns_empty(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        assert compute_lineage_depth_over_time(tree, []) == []

    def test_depth_grows_over_time(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [_make_agent("r", [])]),
            _make_snapshot(100, [
                _make_agent("r", []),
                _make_agent("c", ["r"], generation=1),
            ]),
            _make_snapshot(200, [
                _make_agent("c", ["r"]),
                _make_agent("gc", ["c"], generation=2),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_lineage_depth_over_time(tree, raw)
        # step 0: only root at depth 0
        # step 100: root (0) + child (1)
        # step 200: child (1) + grandchild (2)
        assert len(result) == 3
        by_step = {r[0]: r for r in result}
        assert by_step[0][1] == 0   # max_depth at step 0
        assert by_step[100][1] == 1  # max_depth at step 100
        assert by_step[200][1] == 2  # max_depth at step 200

    def test_result_sorted_by_step(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [_make_agent("r", [])]),
            _make_snapshot(50, [_make_agent("r", [])]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        result = compute_lineage_depth_over_time(tree, raw)
        steps = [r[0] for r in result]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# extract_chromosomes_from_snapshots
# ---------------------------------------------------------------------------


class TestExtractChromosomes:
    def test_empty_snapshots(self):
        assert extract_chromosomes_from_snapshots([]) == {}

    def test_basic_extraction(self):
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [], chromosome={"learning_rate": 0.01})]),
        ]
        result = extract_chromosomes_from_snapshots(snaps)
        assert "a1" in result
        assert result["a1"]["learning_rate"] == pytest.approx(0.01)

    def test_last_seen_value_wins(self):
        snaps = [
            _make_snapshot(0, [_make_agent("a1", [], chromosome={"learning_rate": 0.01})]),
            _make_snapshot(100, [_make_agent("a1", [], chromosome={"learning_rate": 0.02})]),
        ]
        result = extract_chromosomes_from_snapshots(snaps)
        assert result["a1"]["learning_rate"] == pytest.approx(0.02)

    def test_multiple_agents(self):
        snaps = [
            _make_snapshot(0, [
                _make_agent("a1", [], chromosome={"lr": 0.01}),
                _make_agent("a2", [], chromosome={"lr": 0.05}),
            ]),
        ]
        result = extract_chromosomes_from_snapshots(snaps)
        assert set(result.keys()) == {"a1", "a2"}


# ---------------------------------------------------------------------------
# plot_intrinsic_lineage_tree – smoke tests
# ---------------------------------------------------------------------------


class TestPlotIntrinsicLineageTree:
    def _make_tree_and_chroms(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snaps = [
            _make_snapshot(0, [
                _make_agent("r", [], chromosome={"learning_rate": 0.005}),
            ]),
            _make_snapshot(100, [
                _make_agent("r", [], chromosome={"learning_rate": 0.005}),
                _make_agent("c1", ["r"], generation=1, chromosome={"learning_rate": 0.01}),
                _make_agent("c2", ["r"], generation=1, chromosome={"learning_rate": 0.02}),
            ]),
            _make_snapshot(200, [
                _make_agent("r", [], chromosome={"learning_rate": 0.005}),
                _make_agent("c1", ["r"], generation=1, chromosome={"learning_rate": 0.01}),
                _make_agent("gc1", ["c1"], generation=2, chromosome={"learning_rate": 0.015}),
            ]),
        ]
        _write_snapshots(snap_file, snaps)
        tree = build_intrinsic_lineage_dag(snap_file)
        raw = load_intrinsic_snapshots(snap_file)
        chroms = extract_chromosomes_from_snapshots(raw)
        return tree, chroms

    def test_plot_lineage_colouring(self, tmp_path):
        from farm.analysis.phylogenetics.plot import plot_intrinsic_lineage_tree
        from farm.analysis.common.context import AnalysisContext

        tree, chroms = self._make_tree_and_chroms(tmp_path)
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_intrinsic_lineage_tree(tree, ctx)
        assert result is not None
        assert Path(result).exists()

    def test_plot_gene_colouring(self, tmp_path):
        from farm.analysis.phylogenetics.plot import plot_intrinsic_lineage_tree
        from farm.analysis.common.context import AnalysisContext

        tree, chroms = self._make_tree_and_chroms(tmp_path)
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_intrinsic_lineage_tree(
            tree, ctx, gene="learning_rate", chromosomes=chroms
        )
        assert result is not None
        assert Path(result).exists()

    def test_plot_empty_tree_returns_none(self, tmp_path):
        from farm.analysis.phylogenetics.plot import plot_intrinsic_lineage_tree
        from farm.analysis.common.context import AnalysisContext

        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_intrinsic_lineage_tree(tree, ctx)
        assert result is None

    def test_plot_gene_missing_falls_back_gracefully(self, tmp_path):
        """When gene not in chromosomes, falls back to lineage colouring."""
        from farm.analysis.phylogenetics.plot import plot_intrinsic_lineage_tree
        from farm.analysis.common.context import AnalysisContext

        tree, chroms = self._make_tree_and_chroms(tmp_path)
        ctx = AnalysisContext(output_path=tmp_path)
        result = plot_intrinsic_lineage_tree(
            tree, ctx, gene="nonexistent_gene", chromosomes=chroms
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Public API import smoke test
# ---------------------------------------------------------------------------


def test_public_api_importable():
    from farm.analysis.phylogenetics import (
        load_intrinsic_snapshots,
        flatten_snapshots_to_agent_records,
        build_intrinsic_lineage_dag,
        compute_surviving_lineage_count_over_time,
        compute_lineage_depth_over_time,
        extract_chromosomes_from_snapshots,
        plot_intrinsic_lineage_tree,
    )
    assert callable(load_intrinsic_snapshots)
    assert callable(flatten_snapshots_to_agent_records)
    assert callable(build_intrinsic_lineage_dag)
    assert callable(compute_surviving_lineage_count_over_time)
    assert callable(compute_lineage_depth_over_time)
    assert callable(extract_chromosomes_from_snapshots)
    assert callable(plot_intrinsic_lineage_tree)
