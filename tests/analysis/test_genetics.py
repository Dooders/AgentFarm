"""
Unit tests for the genetics analysis module.

Covers:
- parse_parent_ids shared helper
- build_agent_genetics_dataframe with a tiny fixture DB session
- build_evolution_experiment_dataframe with mock EvolutionExperimentResult
- analyze_genetics for both DB-backed and evolution-backed DataFrames
- GeneticsModule registration and protocol compliance
- process_genetics_data dispatcher
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from farm.analysis.genetics.compute import (
    AGENT_GENETICS_COLUMNS,
    EVOLUTION_GENETICS_COLUMNS,
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
    parse_parent_ids,
)
from farm.analysis.genetics.analyze import analyze_genetics
from farm.analysis.genetics.data import process_genetics_data
from farm.analysis.genetics.module import GeneticsModule, genetics_module


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str,
    agent_type: str = "SystemAgent",
    generation: int = 0,
    birth_time: int = 0,
    death_time: int = None,
    genome_id: str = "::1",
    action_weights: dict = None,
):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.agent_type = agent_type
    agent.generation = generation
    agent.birth_time = birth_time
    agent.death_time = death_time
    agent.genome_id = genome_id
    agent.action_weights = action_weights or {}
    return agent


def _make_evaluation(
    candidate_id: str,
    generation: int,
    fitness: float,
    parent_ids: tuple = ("seed", "seed"),
    chromosome_values: dict = None,
):
    ev = SimpleNamespace(
        candidate_id=candidate_id,
        generation=generation,
        fitness=fitness,
        parent_ids=parent_ids,
        chromosome_values=chromosome_values or {"learning_rate": 0.001, "gamma": 0.99},
    )
    return ev


# ---------------------------------------------------------------------------
# parse_parent_ids
# ---------------------------------------------------------------------------


class TestParseParentIds:
    def test_initial_agent_no_parents(self):
        assert parse_parent_ids("::1") == []

    def test_single_parent(self):
        assert parse_parent_ids("agent_a:1") == ["agent_a"]

    def test_two_parents(self):
        assert parse_parent_ids("agent_a:agent_b:1") == ["agent_a", "agent_b"]

    def test_malformed_returns_empty(self):
        # Should not raise; returns [] gracefully
        result = parse_parent_ids("")
        assert isinstance(result, list)

    def test_no_counter_two_parents(self):
        result = parse_parent_ids("parent1:parent2")
        assert "parent1" in result
        assert "parent2" in result


# ---------------------------------------------------------------------------
# build_agent_genetics_dataframe
# ---------------------------------------------------------------------------


class TestBuildAgentGeneticsDataframe:
    def _make_session(self, agents):
        session = MagicMock()
        session.query.return_value.all.return_value = agents
        return session

    def test_empty_session_returns_empty_df(self):
        session = self._make_session([])
        df = build_agent_genetics_dataframe(session)
        assert df.empty
        assert list(df.columns) == AGENT_GENETICS_COLUMNS

    def test_single_genesis_agent(self):
        agents = [_make_agent("a1", genome_id="::1")]
        df = build_agent_genetics_dataframe(self._make_session(agents))
        assert len(df) == 1
        assert df.iloc[0]["agent_id"] == "a1"
        assert df.iloc[0]["parent_ids"] == []

    def test_offspring_agent_has_parent_id(self):
        agents = [_make_agent("child1", generation=1, birth_time=5, genome_id="parent1:1")]
        df = build_agent_genetics_dataframe(self._make_session(agents))
        assert df.iloc[0]["parent_ids"] == ["parent1"]

    def test_multiple_agents_all_columns_present(self):
        agents = [
            _make_agent("a1", generation=0, genome_id="::1"),
            _make_agent("a2", generation=0, genome_id="::2"),
            _make_agent("a3", generation=1, genome_id="a1:1"),
        ]
        df = build_agent_genetics_dataframe(self._make_session(agents))
        assert len(df) == 3
        assert set(df.columns) == set(AGENT_GENETICS_COLUMNS)

    def test_action_weights_stored_as_dict(self):
        weights = {"move": 0.5, "gather": 0.5}
        agents = [_make_agent("a1", action_weights=weights)]
        df = build_agent_genetics_dataframe(self._make_session(agents))
        assert df.iloc[0]["action_weights"] == weights

    def test_none_genome_id_gives_empty_parent_ids(self):
        agents = [_make_agent("a1", genome_id=None)]
        df = build_agent_genetics_dataframe(self._make_session(agents))
        assert df.iloc[0]["parent_ids"] == []


# ---------------------------------------------------------------------------
# build_evolution_experiment_dataframe
# ---------------------------------------------------------------------------


class TestBuildEvolutionExperimentDataframe:
    def _make_result(self, evaluations):
        result = SimpleNamespace(evaluations=evaluations, generation_summaries=[])
        return result

    def test_empty_result_returns_empty_df(self):
        df = build_evolution_experiment_dataframe(self._make_result([]))
        assert df.empty
        assert list(df.columns) == EVOLUTION_GENETICS_COLUMNS

    def test_single_evaluation(self):
        ev = _make_evaluation("g0_c0", generation=0, fitness=42.0)
        df = build_evolution_experiment_dataframe(self._make_result([ev]))
        assert len(df) == 1
        assert df.iloc[0]["candidate_id"] == "g0_c0"
        assert df.iloc[0]["fitness"] == pytest.approx(42.0)

    def test_multi_generation(self):
        evals = [
            _make_evaluation(f"g{g}_c{c}", generation=g, fitness=float(g + c))
            for g in range(3)
            for c in range(4)
        ]
        df = build_evolution_experiment_dataframe(self._make_result(evals))
        assert len(df) == 12
        assert df["generation"].nunique() == 3

    def test_parent_ids_stored_as_list(self):
        ev = _make_evaluation("g1_c0", generation=1, fitness=5.0, parent_ids=("g0_c1", "g0_c2"))
        df = build_evolution_experiment_dataframe(self._make_result([ev]))
        assert df.iloc[0]["parent_ids"] == ["g0_c1", "g0_c2"]

    def test_chromosome_values_stored_as_dict(self):
        chrom = {"learning_rate": 0.01, "gamma": 0.95, "epsilon_decay": 0.999}
        ev = _make_evaluation("g0_c0", generation=0, fitness=1.0, chromosome_values=chrom)
        df = build_evolution_experiment_dataframe(self._make_result([ev]))
        assert df.iloc[0]["chromosome_values"] == chrom


# ---------------------------------------------------------------------------
# analyze_genetics
# ---------------------------------------------------------------------------


class TestAnalyzeGenetics:
    def test_empty_dataframe(self):
        result = analyze_genetics(pd.DataFrame())
        assert result["total_agents"] == 0

    def test_db_frame_generation_stats(self):
        data = {
            "agent_id": ["a1", "a2", "a3"],
            "agent_type": ["system", "system", "system"],
            "generation": [0, 0, 1],
            "birth_time": [0, 0, 5],
            "death_time": [None, None, None],
            "genome_id": ["::1", "::2", "a1:1"],
            "parent_ids": [[], [], ["a1"]],
            "action_weights": [{}, {}, {}],
        }
        df = pd.DataFrame(data)
        result = analyze_genetics(df)
        assert result["total_agents"] == 3
        assert result["max_generation"] == 1
        assert result["pct_with_parents"] == pytest.approx(100 / 3, rel=1e-3)

    def test_evolution_frame_fitness_stats(self):
        data = {
            "candidate_id": ["c0", "c1", "c2"],
            "generation": [0, 0, 1],
            "fitness": [10.0, 20.0, 15.0],
            "parent_ids": [["seed", "seed"], ["seed", "seed"], ["c1", "c0"]],
            "chromosome_values": [
                {"learning_rate": 0.001},
                {"learning_rate": 0.01},
                {"learning_rate": 0.005},
            ],
        }
        df = pd.DataFrame(data)
        result = analyze_genetics(df)
        assert result["best_fitness"] == pytest.approx(20.0)
        assert result["mean_fitness"] == pytest.approx(15.0)
        assert "gene_statistics" in result
        assert "learning_rate" in result["gene_statistics"]


# ---------------------------------------------------------------------------
# process_genetics_data
# ---------------------------------------------------------------------------


class TestProcessGeneticsData:
    def test_passthrough_dataframe(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = process_genetics_data(df)
        assert result is df

    def test_raises_on_unsupported_type(self):
        with pytest.raises(TypeError):
            process_genetics_data("not supported")

    def test_dispatches_to_db_accessor_for_session(self):
        session = MagicMock()
        session.query.return_value.all.return_value = []
        result = process_genetics_data(session)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == AGENT_GENETICS_COLUMNS

    def test_dispatches_to_evolution_accessor_for_result(self):
        evo_result = SimpleNamespace(evaluations=[], generation_summaries=[])
        result = process_genetics_data(evo_result)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == EVOLUTION_GENETICS_COLUMNS


# ---------------------------------------------------------------------------
# GeneticsModule
# ---------------------------------------------------------------------------


class TestGeneticsModule:
    def test_singleton_name(self):
        assert genetics_module.name == "genetics"

    def test_singleton_is_genetics_module(self):
        assert isinstance(genetics_module, GeneticsModule)

    def test_supports_database(self):
        assert genetics_module.supports_database() is True

    def test_get_db_filename(self):
        assert genetics_module.get_db_filename() == "simulation.db"

    def test_get_data_processor_is_callable(self):
        proc = genetics_module.get_data_processor()
        assert callable(proc.process)

    def test_registered_functions_not_empty(self):
        info = genetics_module.get_info()
        assert len(info["functions"]) > 0

    def test_function_groups_contain_all_and_analysis(self):
        groups = genetics_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups

    def test_protocol_attributes_present(self):
        required = ["name", "description", "get_data_processor", "get_analysis_functions", "get_function_groups"]
        for attr in required:
            assert hasattr(genetics_module, attr), f"Missing attribute: {attr}"
