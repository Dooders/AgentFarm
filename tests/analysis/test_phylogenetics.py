"""Unit tests for the phylogenetics analysis module.

Covers:
- PhylogeneticNode and PhylogeneticTree data structures
- build_phylogenetic_tree: single founder, multi-founder, dual-parent,
  broken/missing parent IDs, orphan nodes
- build_phylogenetic_tree_from_records: evolution_lineage.json format
- JSON and Newick serialisation
- PhylogeneticTree.summary() statistics
- analyze_phylogenetics()
- process_phylogenetics_data() dispatcher
- plot_phylogenetic_tree() (smoke-test; output saved to tmp dir)
- PhylogeneticsModule registration and protocol compliance
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from farm.analysis.phylogenetics.compute import (
    PhylogeneticNode,
    PhylogeneticTree,
    PhylogeneticTreeSummary,
    build_phylogenetic_tree,
    build_phylogenetic_tree_from_records,
)
from farm.analysis.phylogenetics.analyze import analyze_phylogenetics
from farm.analysis.phylogenetics.data import process_phylogenetics_data
from farm.analysis.phylogenetics.module import PhylogeneticsModule, phylogenetics_module
from farm.analysis.common.context import AnalysisContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(
    agent_id: str,
    genome_id: str = "::1",
    generation: int = 0,
    birth_time: int = 0,
    death_time: int = None,
):
    """Create a lightweight agent-like namespace."""
    return SimpleNamespace(
        agent_id=agent_id,
        genome_id=genome_id,
        generation=generation,
        birth_time=birth_time,
        death_time=death_time,
    )


def _record(
    candidate_id: str,
    parent_ids: list,
    generation: int = 0,
):
    return {
        "candidate_id": candidate_id,
        "parent_ids": parent_ids,
        "generation": generation,
    }


# ---------------------------------------------------------------------------
# build_phylogenetic_tree – single founder
# ---------------------------------------------------------------------------


class TestSingleFounder:
    def test_single_agent_no_parents(self):
        agents = [_agent("a1", genome_id="::1")]
        tree = build_phylogenetic_tree(agents)
        assert "a1" in tree.nodes
        assert tree.nodes["a1"].is_root is True
        assert tree.nodes["a1"].parent_ids == []
        assert tree.roots == ["a1"]

    def test_single_founder_with_two_generations(self):
        agents = [
            _agent("a1", genome_id="::1", generation=0),
            _agent("a2", genome_id="a1:1", generation=1),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.nodes["a1"].is_root is True
        assert tree.nodes["a2"].is_root is False
        assert tree.nodes["a2"].parent_ids == ["a1"]
        assert "a2" in tree.nodes["a1"].children_ids
        assert tree.nodes["a2"].depth == 1

    def test_linear_chain_depths(self):
        agents = [
            _agent("a1", genome_id="::1", generation=0),
            _agent("a2", genome_id="a1:1", generation=1),
            _agent("a3", genome_id="a2:1", generation=2),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.nodes["a1"].depth == 0
        assert tree.nodes["a2"].depth == 1
        assert tree.nodes["a3"].depth == 2

    def test_max_depth_prunes_without_unreachable_warning(self, caplog):
        agents = [
            _agent("a1", genome_id="::1", generation=0),
            _agent("a2", genome_id="a1:1", generation=1),
            _agent("a3", genome_id="a2:1", generation=2),
        ]
        with caplog.at_level("WARNING"):
            tree = build_phylogenetic_tree(agents, max_depth=0)

        assert tree.nodes["a1"].depth == 0
        assert tree.nodes["a2"].depth == -1
        assert tree.nodes["a3"].depth == -1
        assert not any("unreachable node" in record.message for record in caplog.records)

    def test_branching(self):
        agents = [
            _agent("root", genome_id="::1"),
            _agent("child1", genome_id="root:1"),
            _agent("child2", genome_id="root:2"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert sorted(tree.nodes["root"].children_ids) == ["child1", "child2"]
        assert tree.nodes["child1"].depth == 1
        assert tree.nodes["child2"].depth == 1


# ---------------------------------------------------------------------------
# build_phylogenetic_tree – multi-founder
# ---------------------------------------------------------------------------


class TestMultiFounder:
    def test_two_founders(self):
        agents = [
            _agent("f1", genome_id="::1"),
            _agent("f2", genome_id="::2"),
            _agent("c1", genome_id="f1:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert set(tree.roots) == {"f1", "f2"}
        assert tree.nodes["f1"].is_root is True
        assert tree.nodes["f2"].is_root is True

    def test_roots_are_sorted(self):
        agents = [
            _agent("z_founder", genome_id="::1"),
            _agent("a_founder", genome_id="::2"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.roots == sorted(["z_founder", "a_founder"])

    def test_independent_lineages_no_cross_edges(self):
        agents = [
            _agent("f1", genome_id="::1"),
            _agent("f2", genome_id="::2"),
            _agent("c1", genome_id="f1:1"),
            _agent("c2", genome_id="f2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert "c2" not in tree.nodes["f1"].children_ids
        assert "c1" not in tree.nodes["f2"].children_ids


# ---------------------------------------------------------------------------
# build_phylogenetic_tree – dual-parent (DAG)
# ---------------------------------------------------------------------------


class TestDualParent:
    def test_is_dag_when_two_parents(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.is_dag is True
        assert tree.nodes["child"].parent_ids == ["p1", "p2"]

    def test_child_registered_in_both_parents_children(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert "child" in tree.nodes["p1"].children_ids
        assert "child" in tree.nodes["p2"].children_ids

    def test_dag_depth_is_one(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.nodes["child"].depth == 1

    def test_single_parent_is_not_dag(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("c1", genome_id="p1:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        assert tree.is_dag is False


# ---------------------------------------------------------------------------
# build_phylogenetic_tree – broken / missing parent IDs
# ---------------------------------------------------------------------------


class TestBrokenParentIds:
    def test_none_genome_id_treated_as_founder(self):
        agents = [_agent("a1", genome_id=None)]
        tree = build_phylogenetic_tree(agents)
        assert tree.nodes["a1"].is_root is True
        assert tree.nodes["a1"].parent_ids == []

    def test_missing_parent_creates_orphan(self):
        # "ghost" parent is not in the dataset
        agents = [_agent("child", genome_id="ghost:1")]
        tree = build_phylogenetic_tree(agents)
        node = tree.nodes["child"]
        assert node.is_root is True
        assert node.is_orphan is True
        assert node.parent_ids == ["ghost"]

    def test_partial_parents_present(self):
        # p1 is real, p2 is missing
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("child", genome_id="p1:missing:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        # child's parent p1 is in tree -> child is not orphan/root
        assert tree.nodes["child"].is_root is False
        assert tree.nodes["child"].is_orphan is False

    def test_empty_genome_id_string(self):
        agents = [_agent("a1", genome_id="")]
        tree = build_phylogenetic_tree(agents)
        # Empty genome_id should parse gracefully -> founder
        node = tree.nodes["a1"]
        assert node.is_root is True

    def test_empty_agents_list(self):
        tree = build_phylogenetic_tree([])
        assert tree.nodes == {}
        assert tree.roots == []

    def test_malformed_genome_id_is_handled_gracefully(self):
        agents = [_agent("a1", genome_id="not:valid:genome:id:1:2:3")]
        # Should not raise
        tree = build_phylogenetic_tree(agents)
        assert "a1" in tree.nodes


# ---------------------------------------------------------------------------
# build_phylogenetic_tree_from_records
# ---------------------------------------------------------------------------


class TestBuildFromRecords:
    def test_empty_records(self):
        tree = build_phylogenetic_tree_from_records([])
        assert tree.nodes == {}
        assert tree.roots == []

    def test_single_founder_record(self):
        records = [_record("g0_c0", parent_ids=[])]
        tree = build_phylogenetic_tree_from_records(records)
        assert "g0_c0" in tree.nodes
        assert tree.nodes["g0_c0"].is_root is True

    def test_lineage_chain(self):
        records = [
            _record("g0_c0", parent_ids=[], generation=0),
            _record("g1_c0", parent_ids=["g0_c0"], generation=1),
            _record("g2_c0", parent_ids=["g1_c0"], generation=2),
        ]
        tree = build_phylogenetic_tree_from_records(records)
        assert tree.nodes["g0_c0"].depth == 0
        assert tree.nodes["g1_c0"].depth == 1
        assert tree.nodes["g2_c0"].depth == 2

    def test_dual_parent_record(self):
        records = [
            _record("g0_c0", parent_ids=[], generation=0),
            _record("g0_c1", parent_ids=[], generation=0),
            _record("g1_c0", parent_ids=["g0_c0", "g0_c1"], generation=1),
        ]
        tree = build_phylogenetic_tree_from_records(records)
        assert tree.is_dag is True

    def test_missing_id_key_skipped(self):
        records = [
            {"generation": 0, "parent_ids": []},  # missing candidate_id
            _record("g0_c0", parent_ids=[]),
        ]
        tree = build_phylogenetic_tree_from_records(records)
        # Only the valid record should appear
        assert len(tree.nodes) == 1
        assert "g0_c0" in tree.nodes

    def test_custom_id_key(self):
        records = [{"agent_id": "a1", "parent_ids": [], "generation": 0}]
        tree = build_phylogenetic_tree_from_records(records, id_key="agent_id")
        assert "a1" in tree.nodes

    def test_fallback_to_agent_id_when_primary_key_absent(self):
        records = [{"agent_id": "a1", "parent_ids": [], "generation": 0}]
        tree = build_phylogenetic_tree_from_records(records)
        # Falls back to "agent_id" key
        assert "a1" in tree.nodes

    def test_invalid_numeric_fields_fallback_to_defaults(self):
        records = [
            {
                "candidate_id": "a1",
                "parent_ids": [],
                "generation": "unknown",
                "birth_time": "",
                "death_time": "not-a-number",
            }
        ]
        tree = build_phylogenetic_tree_from_records(records)
        node = tree.nodes["a1"]
        assert node.generation == -1
        assert node.birth_time == -1
        assert node.death_time is None


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


class TestJsonExport:
    def test_to_dict_structure(self):
        agents = [
            _agent("f1", genome_id="::1"),
            _agent("c1", genome_id="f1:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        d = tree.to_dict()
        assert "is_dag" in d
        assert "roots" in d
        assert "nodes" in d
        assert "f1" in d["nodes"]
        assert "c1" in d["nodes"]

    def test_to_json_roundtrip(self):
        agents = [
            _agent("f1", genome_id="::1"),
            _agent("c1", genome_id="f1:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        json_str = tree.to_json()
        data = json.loads(json_str)
        assert data["nodes"]["f1"]["is_root"] is True
        assert data["nodes"]["c1"]["parent_ids"] == ["f1"]

    def test_children_ids_sorted_in_json(self):
        agents = [
            _agent("root", genome_id="::1"),
            _agent("z_child", genome_id="root:2"),
            _agent("a_child", genome_id="root:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        d = tree.to_dict()
        assert d["nodes"]["root"]["children_ids"] == sorted(
            d["nodes"]["root"]["children_ids"]
        )

    def test_dag_flag_preserved_in_json(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        data = json.loads(tree.to_json())
        assert data["is_dag"] is True


# ---------------------------------------------------------------------------
# Newick export
# ---------------------------------------------------------------------------


class TestNewickExport:
    def test_empty_tree(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        assert tree.to_newick() == "();"

    def test_single_root_no_children(self):
        agents = [_agent("r1", genome_id="::1")]
        tree = build_phylogenetic_tree(agents)
        newick = tree.to_newick()
        assert newick.endswith(";")
        assert "r1" in newick

    def test_root_with_two_children(self):
        agents = [
            _agent("root", genome_id="::1"),
            _agent("c1", genome_id="root:1"),
            _agent("c2", genome_id="root:2"),
        ]
        tree = build_phylogenetic_tree(agents)
        newick = tree.to_newick()
        assert newick.endswith(";")
        assert "root" in newick
        assert "c1" in newick
        assert "c2" in newick

    def test_multi_root_newick_wraps(self):
        agents = [
            _agent("r1", genome_id="::1"),
            _agent("r2", genome_id="::2"),
        ]
        tree = build_phylogenetic_tree(agents)
        newick = tree.to_newick()
        assert newick.startswith("(")
        assert newick.endswith(";")

    def test_dag_uses_first_parent_only(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        newick = tree.to_newick()
        # child should appear under p1's subtree (first parent)
        assert "child" in newick


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_empty_tree_summary(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        s = tree.summary()
        assert s.num_nodes == 0
        assert s.max_depth == 0

    def test_basic_summary(self):
        agents = [
            _agent("r1", genome_id="::1"),
            _agent("c1", genome_id="r1:1"),
            _agent("c2", genome_id="r1:2"),
        ]
        tree = build_phylogenetic_tree(agents)
        s = tree.summary()
        assert s.num_nodes == 3
        assert s.num_founders == 1
        assert s.max_depth == 1
        assert s.mean_branching_factor == pytest.approx(2.0)

    def test_orphan_counted(self):
        agents = [_agent("child", genome_id="missing_parent:1")]
        tree = build_phylogenetic_tree(agents)
        s = tree.summary()
        assert s.num_orphans == 1

    def test_dag_flag_in_summary(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        s = tree.summary()
        assert s.is_dag is True

    def test_lineage_survival_with_timing(self):
        agents = [
            _agent("root", genome_id="::1", birth_time=0, death_time=None),
            _agent("child", genome_id="root:1", birth_time=5, death_time=None),
        ]
        tree = build_phylogenetic_tree(agents)
        # max_step derived from timing data
        assert tree.max_step is not None
        s = tree.summary()
        assert s.num_surviving_lineages >= 0

    def test_no_timing_lineage_stats_are_minus_one(self):
        agents = [
            _agent("r1", genome_id="::1", birth_time=-1, death_time=None),
        ]
        tree = build_phylogenetic_tree(agents)
        tree.max_step = None  # force no timing
        s = tree.summary()
        assert s.num_surviving_lineages == -1
        assert s.num_lineages_at_final_step == -1


# ---------------------------------------------------------------------------
# analyze_phylogenetics
# ---------------------------------------------------------------------------


class TestAnalyzePhylogenetics:
    def test_empty_tree_returns_zeros(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        result = analyze_phylogenetics(tree)
        assert result["num_nodes"] == 0
        assert result["roots"] == []

    def test_basic_tree_analysis(self):
        agents = [
            _agent("r1", genome_id="::1"),
            _agent("c1", genome_id="r1:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        result = analyze_phylogenetics(tree)
        assert result["num_nodes"] == 2
        assert result["num_founders"] == 1
        assert result["max_depth"] == 1
        assert "r1" in result["roots"]

    def test_dag_flag_propagated(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        result = analyze_phylogenetics(tree)
        assert result["is_dag"] is True


# ---------------------------------------------------------------------------
# process_phylogenetics_data
# ---------------------------------------------------------------------------


class TestProcessPhylogeneticsData:
    def test_list_of_records(self):
        records = [
            _record("g0_c0", parent_ids=[], generation=0),
            _record("g1_c0", parent_ids=["g0_c0"], generation=1),
        ]
        tree = process_phylogenetics_data(records)
        assert isinstance(tree, PhylogeneticTree)
        assert "g0_c0" in tree.nodes

    def test_list_of_agents_dispatches_to_agent_builder(self):
        agents = [
            _agent("a1", genome_id="::1", generation=0),
            _agent("a2", genome_id="a1:1", generation=1),
        ]
        tree = process_phylogenetics_data(agents)
        assert isinstance(tree, PhylogeneticTree)
        assert "a1" in tree.nodes
        assert tree.nodes["a2"].parent_ids == ["a1"]

    def test_passthrough_for_tree_instance(self):
        original = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        result = process_phylogenetics_data(original)
        assert result is original

    def test_unknown_type_returns_empty_tree(self):
        result = process_phylogenetics_data(object())
        assert isinstance(result, PhylogeneticTree)
        assert result.nodes == {}

    def test_db_session_dispatch(self):
        session = MagicMock()
        # Simulate query returning two agents
        a1 = SimpleNamespace(
            agent_id="a1",
            genome_id="::1",
            generation=0,
            birth_time=0,
            death_time=None,
        )
        a2 = SimpleNamespace(
            agent_id="a2",
            genome_id="a1:1",
            generation=1,
            birth_time=1,
            death_time=None,
        )
        session.query.return_value.all.return_value = [a1, a2]

        # Patch the AgentModel inside farm.database.models so the import inside
        # process_phylogenetics_data succeeds with the mock session's query result
        with patch("farm.database.models.AgentModel"):
            tree = process_phylogenetics_data(session)

        assert isinstance(tree, PhylogeneticTree)


# ---------------------------------------------------------------------------
# plot_phylogenetic_tree – smoke test
# ---------------------------------------------------------------------------


class TestPlotPhylogeneticTree:
    def test_plot_small_tree(self):
        agents = [
            _agent("root", genome_id="::1", birth_time=0, death_time=None),
            _agent("c1", genome_id="root:1", birth_time=1, death_time=None),
            _agent("c2", genome_id="root:2", birth_time=1, death_time=None),
            _agent("c3", genome_id="c1:1", birth_time=2, death_time=5),
        ]
        tree = build_phylogenetic_tree(agents)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = AnalysisContext(output_path=Path(tmpdir))
            from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree
            result = plot_phylogenetic_tree(tree, ctx)
            assert result is not None
            assert Path(result).exists()

    def test_plot_empty_tree_returns_none(self):
        tree = PhylogeneticTree(nodes={}, roots=[], is_dag=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = AnalysisContext(output_path=Path(tmpdir))
            from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree
            result = plot_phylogenetic_tree(tree, ctx)
            assert result is None

    def test_plot_dag(self):
        agents = [
            _agent("p1", genome_id="::1"),
            _agent("p2", genome_id="::2"),
            _agent("child", genome_id="p1:p2:1"),
        ]
        tree = build_phylogenetic_tree(agents)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = AnalysisContext(output_path=Path(tmpdir))
            from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree
            result = plot_phylogenetic_tree(tree, ctx)
            assert result is not None

    def test_plot_large_tree_pruned(self):
        """Verify pruning activates without error for large trees."""
        # Create 50 founders + 50 children each = 2500 nodes
        agents = [_agent(f"f{i}", genome_id="::1") for i in range(50)]
        agents += [
            _agent(f"c{i}_{j}", genome_id=f"f{i}:{j}")
            for i in range(50)
            for j in range(50)
        ]
        tree = build_phylogenetic_tree(agents)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = AnalysisContext(output_path=Path(tmpdir))
            from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree
            result = plot_phylogenetic_tree(tree, ctx, max_nodes=100)
            assert result is not None


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


class TestPhylogeneticsModule:
    def test_module_name(self):
        assert phylogenetics_module.name == "phylogenetics"

    def test_module_has_description(self):
        assert phylogenetics_module.description

    def test_module_supports_database(self):
        assert phylogenetics_module.supports_database() is True

    def test_module_functions_registered(self):
        module = PhylogeneticsModule()
        module.register_functions()
        function_names = module.get_function_names()
        assert "analyze_phylogenetics" in function_names
        assert "plot_phylogenetic_tree" in function_names

    def test_module_groups_registered(self):
        module = PhylogeneticsModule()
        module.register_functions()
        groups = module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups

    def test_module_has_data_processor(self):
        module = PhylogeneticsModule()
        processor = module.get_data_processor()
        assert processor is not None

    def test_singleton_is_registered(self):
        # phylogenetics_module should implement the AnalysisModule protocol
        required = ["name", "description", "get_data_processor",
                    "get_analysis_functions", "get_function_groups"]
        for attr in required:
            assert hasattr(phylogenetics_module, attr), f"Missing attribute: {attr}"
