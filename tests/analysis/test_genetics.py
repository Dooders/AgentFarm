"""
Unit tests for the genetics analysis module.

Covers:
- parse_parent_ids shared helper
- build_agent_genetics_dataframe with a tiny fixture DB session
- build_evolution_experiment_dataframe with mock EvolutionExperimentResult
- analyze_genetics for both DB-backed and evolution-backed DataFrames
- GeneticsModule registration and protocol compliance
- process_genetics_data dispatcher
- Diversity metrics: ContinuousLocusDiversity, CategoricalLocusDiversity,
  PopulationDiversitySummary, and timeseries computation
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from farm.analysis.genetics.compute import (
    AGENT_GENETICS_COLUMNS,
    ALLELE_FREQUENCY_COLUMNS,
    ALLELE_MEAN,
    ALLELE_VARIANCE,
    EVOLUTION_GENETICS_COLUMNS,
    SELECTION_PRESSURE_COLUMNS,
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
    compute_allele_frequency_timeseries,
    compute_categorical_locus_diversity,
    compute_continuous_locus_diversity,
    compute_evolution_diversity_timeseries,
    compute_population_diversity,
    compute_selection_pressure_summary,
    compute_fitness_gene_correlations,
    compute_pairwise_epistasis,
    FITNESS_GENE_CORRELATION_COLUMNS,
    PAIRWISE_EPISTASIS_COLUMNS,
    simulate_wright_fisher,
    WRIGHT_FISHER_COLUMNS,
    compute_fst_pairwise,
    FST_COLUMNS,
    compute_migration_counts,
    MIGRATION_COLUMNS,
    compute_gene_flow_timeseries,
    GENE_FLOW_COLUMNS,
)
from farm.analysis.genetics.utils import parse_parent_ids
from farm.analysis.genetics.analyze import analyze_genetics
from farm.analysis.genetics.data import process_genetics_data
from farm.analysis.genetics.module import GeneticsModule, genetics_module
from farm.database.models import AgentModel, Base


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

    def test_evolution_frame_skips_malformed_chromosome_values(self):
        data = {
            "candidate_id": ["c0", "c1", "c2", "c3"],
            "generation": [0, 0, 1, 1],
            "fitness": [10.0, 20.0, 15.0, 12.0],
            "chromosome_values": [
                {"learning_rate": 0.001},
                {"learning_rate": "bad"},
                None,
                {"learning_rate": 0.005},
            ],
        }
        df = pd.DataFrame(data)
        result = analyze_genetics(df)
        assert "gene_statistics" in result
        assert result["gene_statistics"]["learning_rate"]["mean"] == pytest.approx(0.003)


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
            process_genetics_data(12345)

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

    def test_loads_from_experiment_path(self, tmp_path):
        db_file = tmp_path / "simulation.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(
                AgentModel(
                    agent_id="agent_path_1",
                    agent_type="system",
                    birth_time=0,
                    genome_id="::1",
                    generation=0,
                )
            )
            session.commit()
        engine.dispose()

        result = process_genetics_data(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == AGENT_GENETICS_COLUMNS
        assert len(result) == 1


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

    def test_run_analysis_with_experiment_path(self, tmp_path):
        db_file = tmp_path / "simulation.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            session.add_all(
                [
                    AgentModel(
                        agent_id="parent_a",
                        agent_type="system",
                        birth_time=0,
                        genome_id="::1",
                        generation=0,
                    ),
                    AgentModel(
                        agent_id="child_a",
                        agent_type="system",
                        birth_time=5,
                        genome_id="parent_a:1",
                        generation=1,
                    ),
                ]
            )
            session.commit()
        engine.dispose()

        output_dir = tmp_path / "analysis_output"
        out_path, df = genetics_module.run_analysis(tmp_path, output_dir, group="analysis")

        assert out_path == output_dir
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(AGENT_GENETICS_COLUMNS).issubset(df.columns)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


class TestContinuousLocusDiversity:
    """Tests for compute_continuous_locus_diversity."""

    # --- Degenerate cases ---

    def test_single_individual_zero_variance(self):
        result = compute_continuous_locus_diversity([0.1], "lr", bounds=(0.0, 1.0))
        assert result.std == pytest.approx(0.0)
        assert result.normalized_variance == pytest.approx(0.0)
        assert result.coefficient_of_variation == pytest.approx(0.0)
        assert result.range_occupancy == pytest.approx(0.0)

    def test_identical_population_zero_diversity(self):
        values = [0.1, 0.1, 0.1, 0.1]
        result = compute_continuous_locus_diversity(values, "lr", bounds=(0.0, 1.0))
        assert result.std == pytest.approx(0.0)
        assert result.normalized_variance == pytest.approx(0.0)
        assert result.range_occupancy == pytest.approx(0.0)

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="no values"):
            compute_continuous_locus_diversity([], "lr")

    def test_reversed_bounds_raises(self):
        with pytest.raises(ValueError, match="bounds must satisfy bounds\\[0\\] < bounds\\[1\\]"):
            compute_continuous_locus_diversity([0.1, 0.2], "lr", bounds=(1.0, 0.0))

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="bounds must satisfy bounds\\[0\\] < bounds\\[1\\]"):
            compute_continuous_locus_diversity([0.5, 0.5], "lr", bounds=(0.5, 0.5))

    def test_invalid_entropy_bins_raises(self):
        with pytest.raises(ValueError, match="entropy_bins must be >= 1"):
            compute_continuous_locus_diversity([0.1, 0.2], "lr", entropy_bins=0, compute_entropy=True)

    def test_negative_entropy_bins_raises(self):
        with pytest.raises(ValueError, match="entropy_bins must be >= 1"):
            compute_continuous_locus_diversity([0.1, 0.2], "lr", entropy_bins=-5, compute_entropy=True)

    # --- Hand-computed fixture: values=[0.1, 0.2, 0.3], bounds=(0.0, 1.0) ---
    # mean = 0.2
    # var  = ((0.1-0.2)² + (0.2-0.2)² + (0.3-0.2)²) / 3 = 0.02/3
    # std  = sqrt(0.02/3) ≈ 0.081650
    # normalized_variance = 0.02/3 / 1.0² ≈ 0.006667
    # CV   = sqrt(0.02/3) / 0.2 ≈ 0.40825
    # range_occupancy = (0.3 - 0.1) / 1.0 = 0.2

    def test_fixture_mean(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        assert r.mean == pytest.approx(0.2)

    def test_fixture_std(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        assert r.std == pytest.approx((0.02 / 3) ** 0.5, rel=1e-6)

    def test_fixture_normalized_variance(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        assert r.normalized_variance == pytest.approx(0.02 / 3, rel=1e-6)

    def test_fixture_coefficient_of_variation(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        expected_cv = (0.02 / 3) ** 0.5 / 0.2
        assert r.coefficient_of_variation == pytest.approx(expected_cv, rel=1e-6)

    def test_fixture_range_occupancy(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        assert r.range_occupancy == pytest.approx(0.2)

    def test_fixture_n_individuals(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr", bounds=(0.0, 1.0))
        assert r.n_individuals == 3

    # --- Without bounds: normalized_variance and range_occupancy are nan ---

    def test_no_bounds_normalized_variance_nan(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr")
        assert math.isnan(r.normalized_variance)

    def test_no_bounds_range_occupancy_nan(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr")
        assert math.isnan(r.range_occupancy)

    # --- CV nan when mean == 0 ---

    def test_zero_mean_cv_nan(self):
        r = compute_continuous_locus_diversity([0.0, 0.0], "lr")
        assert math.isnan(r.coefficient_of_variation)

    # --- Entropy ---

    def test_entropy_none_by_default(self):
        r = compute_continuous_locus_diversity([0.1, 0.2, 0.3], "lr")
        assert r.shannon_entropy is None

    def test_entropy_non_negative_when_requested(self):
        r = compute_continuous_locus_diversity(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            "lr",
            bounds=(0.0, 1.0),
            compute_entropy=True,
        )
        assert r.shannon_entropy is not None
        assert r.shannon_entropy >= 0.0

    def test_entropy_identical_pop_zero_or_low(self):
        """All values in one bin → entropy = 0 (single occupied bin)."""
        r = compute_continuous_locus_diversity(
            [0.5, 0.5, 0.5],
            "lr",
            bounds=(0.0, 1.0),
            entropy_bins=5,
            compute_entropy=True,
        )
        assert r.shannon_entropy is not None
        assert r.shannon_entropy == pytest.approx(0.0)

    def test_entropy_with_out_of_bounds_values_is_still_finite(self):
        r = compute_continuous_locus_diversity(
            [-0.5, 0.1, 0.9, 1.5],
            "lr",
            bounds=(0.0, 1.0),
            entropy_bins=5,
            compute_entropy=True,
        )
        assert r.shannon_entropy is not None
        assert r.shannon_entropy > 0.0
        assert r.shannon_entropy < float("inf")

    def test_range_occupancy_clamped_to_one_when_values_exceed_bounds(self):
        r = compute_continuous_locus_diversity(
            [-0.5, 1.5],
            "lr",
            bounds=(0.0, 1.0),
        )
        assert r.range_occupancy == pytest.approx(1.0)


class TestCategoricalLocusDiversity:
    """Tests for compute_categorical_locus_diversity."""

    # --- Degenerate cases ---

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no weight vectors"):
            compute_categorical_locus_diversity([], "action_weights")

    def test_all_empty_dicts_raises(self):
        """Weight vectors with no categories (all empty dicts) should raise."""
        with pytest.raises(ValueError, match="no categories"):
            compute_categorical_locus_diversity([{}, {}], "action_weights")

    def test_single_empty_dict_raises(self):
        """A single empty dict provides no categories."""
        with pytest.raises(ValueError, match="no categories"):
            compute_categorical_locus_diversity([{}], "action_weights")

    def test_single_individual_heterozygosity_is_pure(self):
        """Single individual: diversity of the distribution is what it is;
        if it uses one action exclusively, He == 0."""
        r = compute_categorical_locus_diversity([{"move": 1.0}], "aw")
        assert r.expected_heterozygosity == pytest.approx(0.0)
        assert r.shannon_entropy == pytest.approx(0.0)
        assert r.simpson_index == pytest.approx(1.0)

    def test_identical_population_single_action(self):
        """All agents use move=1.0 → monomorphic, diversity = 0."""
        vecs = [{"move": 1.0}, {"move": 1.0}, {"move": 1.0}]
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.expected_heterozygosity == pytest.approx(0.0)
        assert r.shannon_entropy == pytest.approx(0.0)
        assert r.simpson_index == pytest.approx(1.0)

    # --- Hand-computed fixture: 2 equal actions (uniform → max entropy) ---
    # weight_vectors = [{"move": 0.5, "gather": 0.5}] * N
    # p_move = 0.5, p_gather = 0.5
    # H     = -(0.5 ln 0.5 + 0.5 ln 0.5) = ln(2) ≈ 0.693147
    # D     = 0.5² + 0.5² = 0.5
    # He    = 1 - 0.5 = 0.5

    def test_fixture_uniform_two_actions_entropy(self):
        vecs = [{"move": 0.5, "gather": 0.5}] * 4
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.shannon_entropy == pytest.approx(math.log(2), rel=1e-6)

    def test_fixture_uniform_two_actions_heterozygosity(self):
        vecs = [{"move": 0.5, "gather": 0.5}] * 4
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.expected_heterozygosity == pytest.approx(0.5)

    def test_fixture_uniform_two_actions_simpson(self):
        vecs = [{"move": 0.5, "gather": 0.5}] * 4
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.simpson_index == pytest.approx(0.5)

    def test_fixture_uniform_two_actions_allele_freqs(self):
        vecs = [{"move": 0.5, "gather": 0.5}] * 3
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.allele_frequencies["move"] == pytest.approx(0.5)
        assert r.allele_frequencies["gather"] == pytest.approx(0.5)

    # --- Max entropy for k uniform categories ---

    def test_uniform_k4_max_entropy(self):
        """4 equally-weighted categories → H = ln(4)."""
        vecs = [{"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}] * 5
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.shannon_entropy == pytest.approx(math.log(4), rel=1e-6)
        assert r.expected_heterozygosity == pytest.approx(1 - 1 / 4)

    # --- n_individuals ---

    def test_n_individuals(self):
        vecs = [{"move": 1.0}] * 7
        r = compute_categorical_locus_diversity(vecs, "aw")
        assert r.n_individuals == 7

    # --- Mixed population reduces heterozygosity ---

    def test_mixed_population_nonzero_heterozygosity(self):
        vecs = [
            {"move": 0.8, "gather": 0.2},
            {"move": 0.2, "gather": 0.8},
        ]
        r = compute_categorical_locus_diversity(vecs, "aw")
        # mean: p_move = 0.5, p_gather = 0.5 → He = 0.5
        assert r.expected_heterozygosity == pytest.approx(0.5)


class TestComputePopulationDiversity:
    """Tests for compute_population_diversity."""

    def test_empty_dataframe_returns_zero_individuals(self):
        result = compute_population_diversity(pd.DataFrame())
        assert result.n_individuals == 0

    def test_continuous_loci_from_chromosome_values(self):
        data = {
            "chromosome_values": [
                {"lr": 0.1, "gamma": 0.9},
                {"lr": 0.2, "gamma": 0.95},
                {"lr": 0.3, "gamma": 0.99},
            ]
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df, gene_bounds={"lr": (0.0, 1.0), "gamma": (0.8, 1.0)})
        assert "lr" in result.continuous_loci
        assert "gamma" in result.continuous_loci
        assert result.continuous_loci["lr"].n_individuals == 3

    def test_categorical_loci_from_action_weights(self):
        data = {
            "action_weights": [
                {"move": 0.5, "gather": 0.5},
                {"move": 0.5, "gather": 0.5},
            ]
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df)
        assert "action_weights" in result.categorical_loci
        assert result.categorical_loci["action_weights"].expected_heterozygosity == pytest.approx(0.5)

    def test_identical_pop_continuous_zero_normalized_variance(self):
        data = {"chromosome_values": [{"lr": 0.1}] * 5}
        df = pd.DataFrame(data)
        result = compute_population_diversity(df, gene_bounds={"lr": (0.0, 1.0)})
        assert result.continuous_loci["lr"].normalized_variance == pytest.approx(0.0)

    def test_mean_heterozygosity_matches_single_locus(self):
        vecs = [{"move": 0.5, "gather": 0.5}] * 3
        df = pd.DataFrame({"action_weights": vecs})
        result = compute_population_diversity(df)
        assert result.mean_heterozygosity == pytest.approx(0.5)

    def test_skips_empty_action_weight_dicts(self):
        """Rows with empty action weight dicts are filtered out."""
        data = {
            "action_weights": [
                {},
                {"move": 1.0},
                {"move": 1.0},
            ]
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df)
        # Only 2 non-empty rows contributed to categorical locus
        assert result.categorical_loci["action_weights"].n_individuals == 2

    def test_mean_normalized_variance_nan_without_bounds(self):
        data = {"chromosome_values": [{"lr": 0.1}, {"lr": 0.2}]}
        df = pd.DataFrame(data)
        result = compute_population_diversity(df)  # no gene_bounds
        assert math.isnan(result.mean_normalized_variance)

    def test_both_continuous_and_categorical(self):
        data = {
            "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}],
            "action_weights": [{"move": 0.5, "gather": 0.5}, {"move": 0.5, "gather": 0.5}],
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df, gene_bounds={"lr": (0.0, 1.0)})
        assert "lr" in result.continuous_loci
        assert "action_weights" in result.categorical_loci

    def test_non_numeric_chromosome_values_are_skipped(self):
        """Rows with non-numeric or None gene values are silently skipped."""
        data = {
            "chromosome_values": [
                {"lr": 0.1},
                {"lr": None},
                {"lr": "bad"},
                {"lr": 0.3},
            ]
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df)
        # Only the two finite numeric rows are counted
        assert result.continuous_loci["lr"].n_individuals == 2
        assert result.continuous_loci["lr"].mean == pytest.approx(0.2)

    def test_nan_chromosome_values_are_skipped(self):
        """NaN gene values are filtered out and don't poison locus stats."""
        data = {
            "chromosome_values": [
                {"lr": 0.1},
                {"lr": float("nan")},
                {"lr": 0.3},
            ]
        }
        df = pd.DataFrame(data)
        result = compute_population_diversity(df)
        assert result.continuous_loci["lr"].n_individuals == 2
        assert result.continuous_loci["lr"].mean == pytest.approx(0.2)


class TestComputeEvolutionDiversityTimeseries:
    """Tests for compute_evolution_diversity_timeseries."""

    def _make_result(self, evals):
        return SimpleNamespace(evaluations=evals, generation_summaries=[])

    def _make_eval(self, candidate_id, generation, fitness, chromosome_values):
        return SimpleNamespace(
            candidate_id=candidate_id,
            generation=generation,
            fitness=fitness,
            parent_ids=("seed", "seed"),
            chromosome_values=chromosome_values,
        )

    def test_empty_result_returns_empty_list(self):
        result = self._make_result([])
        ts = compute_evolution_diversity_timeseries(result)
        assert ts == []

    def test_returns_one_entry_per_generation(self):
        evals = [
            self._make_eval(f"g{g}_c{c}", g, float(g + c), {"lr": 0.1 + 0.05 * c})
            for g in range(3)
            for c in range(4)
        ]
        result = self._make_result(evals)
        ts = compute_evolution_diversity_timeseries(result)
        assert len(ts) == 3
        assert [r["generation"] for r in ts] == [0, 1, 2]

    def test_entries_sorted_by_generation(self):
        evals = [
            self._make_eval("g2_c0", 2, 1.0, {"lr": 0.1}),
            self._make_eval("g0_c0", 0, 2.0, {"lr": 0.2}),
            self._make_eval("g1_c0", 1, 3.0, {"lr": 0.3}),
        ]
        result = self._make_result(evals)
        ts = compute_evolution_diversity_timeseries(result)
        assert [r["generation"] for r in ts] == [0, 1, 2]

    def test_diversity_values_are_population_diversity_summary(self):
        from farm.analysis.genetics.compute import PopulationDiversitySummary

        evals = [
            self._make_eval(f"g0_c{c}", 0, float(c), {"lr": 0.1 * (c + 1)})
            for c in range(3)
        ]
        result = self._make_result(evals)
        ts = compute_evolution_diversity_timeseries(result)
        assert isinstance(ts[0]["diversity"], PopulationDiversitySummary)

    def test_identical_generation_zero_normalized_variance(self):
        """All candidates in one generation with the same lr → variance == 0."""
        evals = [
            self._make_eval(f"g0_c{c}", 0, 1.0, {"lr": 0.1})
            for c in range(4)
        ]
        result = self._make_result(evals)
        ts = compute_evolution_diversity_timeseries(
            result, gene_bounds={"lr": (0.0, 1.0)}
        )
        lr_div = ts[0]["diversity"].continuous_loci["lr"]
        assert lr_div.normalized_variance == pytest.approx(0.0)

    def test_with_gene_bounds(self):
        evals = [
            self._make_eval(f"g0_c{c}", 0, 1.0, {"lr": 0.1 * (c + 1)})
            for c in range(3)
        ]
        result = self._make_result(evals)
        ts = compute_evolution_diversity_timeseries(
            result, gene_bounds={"lr": (0.0, 1.0)}
        )
        lr_div = ts[0]["diversity"].continuous_loci["lr"]
        assert not math.isnan(lr_div.normalized_variance)
        assert not math.isnan(lr_div.range_occupancy)

    def test_fallback_to_generation_summaries_when_evaluations_missing(self):
        generation_summaries = [
            SimpleNamespace(
                generation=1,
                gene_statistics={"lr": {"mean": 0.3, "std": 0.1}},
            ),
            SimpleNamespace(
                generation=0,
                gene_statistics={"lr": {"mean": 0.2, "std": 0.0}},
            ),
        ]
        result = SimpleNamespace(evaluations=[], generation_summaries=generation_summaries)

        ts = compute_evolution_diversity_timeseries(
            result,
            gene_bounds={"lr": (0.0, 1.0)},
        )

        assert [row["generation"] for row in ts] == [0, 1]
        gen1_lr = ts[1]["diversity"].continuous_loci["lr"]
        assert gen1_lr.mean == pytest.approx(0.3)
        assert gen1_lr.std == pytest.approx(0.1)
        assert gen1_lr.normalized_variance == pytest.approx(0.01)
        assert math.isnan(gen1_lr.range_occupancy)

    def test_fallback_generation_summary_skips_malformed_gene_stats(self):
        generation_summaries = [
            SimpleNamespace(
                generation=0,
                gene_statistics={
                    "lr": {"mean": "bad", "std": 0.1},
                    "gamma": {"mean": 0.95, "std": 0.01},
                },
            )
        ]
        result = SimpleNamespace(evaluations=[], generation_summaries=generation_summaries)

        ts = compute_evolution_diversity_timeseries(result)
        diversity = ts[0]["diversity"]
        assert "lr" not in diversity.continuous_loci
        assert diversity.continuous_loci["gamma"].mean == pytest.approx(0.95)

# ---------------------------------------------------------------------------
# compute_allele_frequency_timeseries
# ---------------------------------------------------------------------------


def _continuous_df(n_gens: int, n_individuals: int, gene: str, value_fn) -> pd.DataFrame:
    """Build a minimal chromosome_values DataFrame for testing.

    Parameters
    ----------
    n_gens:
        Number of generations.
    n_individuals:
        Number of individuals per generation.
    gene:
        Gene name.
    value_fn:
        Callable(gen, ind) → float producing the gene value for individual
        *ind* in generation *gen*.
    """
    rows = []
    for g in range(n_gens):
        for i in range(n_individuals):
            rows.append({"generation": g, "chromosome_values": {gene: value_fn(g, i)}})
    return pd.DataFrame(rows)


def _categorical_df(n_gens: int, n_individuals: int, action_fn) -> pd.DataFrame:
    """Build a minimal action_weights DataFrame for testing."""
    rows = []
    for g in range(n_gens):
        for i in range(n_individuals):
            rows.append({"generation": g, "action_weights": action_fn(g, i)})
    return pd.DataFrame(rows)


class TestComputeAlleleFrequencyTimeseries:
    """Tests for compute_allele_frequency_timeseries."""

    # --- Output shape and columns ---

    def test_empty_df_returns_empty_with_correct_columns(self):
        result = compute_allele_frequency_timeseries(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == ALLELE_FREQUENCY_COLUMNS

    def test_missing_generation_column_returns_empty(self):
        df = pd.DataFrame({"chromosome_values": [{"lr": 0.1}]})
        result = compute_allele_frequency_timeseries(df)
        assert result.empty

    def test_non_integer_generation_values_are_skipped(self):
        df = pd.DataFrame(
            {
                "generation": [0.1, 0.9, 1.1, 1.9],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}, {"lr": 0.4}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        assert result.empty

    def test_integer_like_float_generation_values_are_accepted(self):
        df = pd.DataFrame(
            {
                "generation": [0.0, 0.0, 1.0, 1.0],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.3}, {"lr": 0.5}, {"lr": 0.7}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        mean_rows = result[result["allele"] == ALLELE_MEAN]
        assert sorted(mean_rows["generation"].tolist()) == [0, 1]
        assert mean_rows[mean_rows["generation"] == 0]["frequency"].iloc[0] == pytest.approx(0.2)
        assert mean_rows[mean_rows["generation"] == 1]["frequency"].iloc[0] == pytest.approx(0.6)

    def test_no_recognised_locus_columns_returns_empty(self):
        df = pd.DataFrame({"generation": [0, 1], "fitness": [1.0, 2.0]})
        result = compute_allele_frequency_timeseries(df)
        assert result.empty
        assert list(result.columns) == ALLELE_FREQUENCY_COLUMNS

    def test_output_columns_are_correct(self):
        df = _continuous_df(2, 3, "lr", lambda g, i: 0.1)
        result = compute_allele_frequency_timeseries(df)
        assert list(result.columns) == ALLELE_FREQUENCY_COLUMNS

    # --- Continuous loci ---

    def test_continuous_produces_mean_and_variance_alleles(self):
        df = _continuous_df(2, 4, "lr", lambda g, i: 0.1 + 0.01 * i)
        result = compute_allele_frequency_timeseries(df)
        alleles = set(result["allele"].unique())
        assert ALLELE_MEAN in alleles
        assert ALLELE_VARIANCE in alleles

    def test_continuous_one_row_per_generation_per_allele(self):
        n_gens = 5
        df = _continuous_df(n_gens, 4, "lr", lambda g, i: 0.1 + 0.01 * g)
        result = compute_allele_frequency_timeseries(df)
        mean_rows = result[result["allele"] == ALLELE_MEAN]
        assert len(mean_rows) == n_gens

    def test_continuous_mean_frequency_matches_population_mean(self):
        # 3 individuals with values 0.1, 0.2, 0.3 → mean = 0.2
        df = pd.DataFrame(
            {
                "generation": [0, 0, 0],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        mean_row = result[(result["allele"] == ALLELE_MEAN) & (result["generation"] == 0)]
        assert len(mean_row) == 1
        assert mean_row.iloc[0]["frequency"] == pytest.approx(0.2)

    def test_continuous_variance_frequency_matches_population_variance(self):
        # variance of [0.1, 0.2, 0.3] with ddof=0 = 0.02/3 ≈ 0.00667
        df = pd.DataFrame(
            {
                "generation": [0, 0, 0],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        var_row = result[(result["allele"] == ALLELE_VARIANCE) & (result["generation"] == 0)]
        assert len(var_row) == 1
        assert var_row.iloc[0]["frequency"] == pytest.approx(0.02 / 3, rel=1e-5)

    def test_continuous_n_individuals_is_correct(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0, 0],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        assert result[result["generation"] == 0]["n_individuals"].unique().tolist() == [3]

    def test_continuous_nan_values_excluded(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0, 0],
                "chromosome_values": [{"lr": 0.1}, {"lr": float("nan")}, {"lr": 0.3}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        mean_row = result[result["allele"] == ALLELE_MEAN]
        assert mean_row.iloc[0]["n_individuals"] == 2
        assert mean_row.iloc[0]["frequency"] == pytest.approx(0.2)

    def test_continuous_multiple_genes(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0],
                "chromosome_values": [{"lr": 0.1, "gamma": 0.9}, {"lr": 0.2, "gamma": 0.95}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        loci = set(result["locus"].unique())
        assert "lr" in loci
        assert "gamma" in loci

    def test_continuous_locus_type_is_continuous(self):
        df = _continuous_df(1, 2, "lr", lambda g, i: 0.1)
        result = compute_allele_frequency_timeseries(df)
        assert (result["locus_type"] == "continuous").all()

    def test_continuous_sorted_by_generation(self):
        df = _continuous_df(4, 2, "lr", lambda g, i: 0.1 * g)
        result = compute_allele_frequency_timeseries(df)
        mean_rows = result[result["allele"] == ALLELE_MEAN]
        assert mean_rows["generation"].is_monotonic_increasing

    # --- Categorical loci ---

    def test_categorical_produces_action_alleles(self):
        df = _categorical_df(2, 4, lambda g, i: {"move": 0.6, "gather": 0.4})
        result = compute_allele_frequency_timeseries(df)
        alleles = set(result["allele"].unique())
        assert "move" in alleles
        assert "gather" in alleles

    def test_categorical_frequencies_sum_to_one_per_generation(self):
        df = _categorical_df(3, 5, lambda g, i: {"move": 0.6, "gather": 0.4})
        result = compute_allele_frequency_timeseries(df)
        for gen, grp in result.groupby("generation"):
            assert grp["frequency"].sum() == pytest.approx(1.0, abs=1e-10)

    def test_categorical_allele_frequencies_match_mean_weights(self):
        # 2 individuals: one move=1.0, one gather=1.0 → mean p_move = 0.5
        df = pd.DataFrame(
            {
                "generation": [0, 0],
                "action_weights": [{"move": 1.0}, {"gather": 1.0}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        move_freq = result[(result["allele"] == "move") & (result["generation"] == 0)]["frequency"].iloc[0]
        gather_freq = result[(result["allele"] == "gather") & (result["generation"] == 0)]["frequency"].iloc[0]
        assert move_freq == pytest.approx(0.5)
        assert gather_freq == pytest.approx(0.5)

    def test_categorical_locus_type_is_categorical(self):
        df = _categorical_df(1, 2, lambda g, i: {"move": 1.0})
        result = compute_allele_frequency_timeseries(df)
        assert (result["locus_type"] == "categorical").all()

    def test_categorical_locus_name_is_action_weights(self):
        df = _categorical_df(1, 2, lambda g, i: {"move": 1.0})
        result = compute_allele_frequency_timeseries(df)
        assert (result["locus"] == "action_weights").all()

    def test_categorical_empty_dicts_skipped(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0, 0],
                "action_weights": [{}, {"move": 1.0}, {"move": 1.0}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        assert result.iloc[0]["n_individuals"] == 2

    def test_categorical_non_numeric_weights_are_skipped(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0],
                "action_weights": [{"move": 1.0}, {"move": "bad", "gather": 0.5}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        move_row = result[result["allele"] == "move"].iloc[0]
        gather_row = result[result["allele"] == "gather"].iloc[0]
        assert move_row["frequency"] == pytest.approx(2 / 3)
        assert gather_row["frequency"] == pytest.approx(1 / 3)
        assert move_row["n_individuals"] == 2

    # --- Both column types present ---

    def test_both_locus_types_present(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}],
                "action_weights": [{"move": 1.0}, {"move": 1.0}],
            }
        )
        result = compute_allele_frequency_timeseries(df)
        assert "continuous" in result["locus_type"].values
        assert "categorical" in result["locus_type"].values


# ---------------------------------------------------------------------------
# compute_selection_pressure_summary
# ---------------------------------------------------------------------------


class TestComputeSelectionPressureSummary:
    """Tests for compute_selection_pressure_summary."""

    # --- Output shape and columns ---

    def test_empty_df_returns_empty_with_correct_columns(self):
        result = compute_selection_pressure_summary(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == SELECTION_PRESSURE_COLUMNS

    def test_missing_required_column_returns_empty(self):
        df = pd.DataFrame({"generation": [0], "locus": ["lr"], "allele": [ALLELE_MEAN]})
        result = compute_selection_pressure_summary(df)
        assert result.empty

    def test_negative_significance_threshold_raises(self):
        df = _continuous_df(3, 3, "lr", lambda g, i: 0.1 + 0.01 * g)
        freq_df = compute_allele_frequency_timeseries(df)
        with pytest.raises(ValueError, match="significance_threshold"):
            compute_selection_pressure_summary(freq_df, significance_threshold=-0.5)

    def test_non_finite_significance_threshold_raises(self):
        df = _continuous_df(3, 3, "lr", lambda g, i: 0.1 + 0.01 * g)
        freq_df = compute_allele_frequency_timeseries(df)
        with pytest.raises(ValueError, match="significance_threshold"):
            compute_selection_pressure_summary(freq_df, significance_threshold=float("nan"))

    def test_output_columns_are_correct(self):
        df = _continuous_df(4, 5, "lr", lambda g, i: 0.1 + 0.01 * g)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        assert list(result.columns) == SELECTION_PRESSURE_COLUMNS

    def test_one_row_per_locus_allele(self):
        # 1 continuous gene → 2 alleles (__mean__, __variance__)
        df = _continuous_df(5, 4, "lr", lambda g, i: 0.1 + 0.02 * g)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        assert len(result) == 2  # __mean__ and __variance__

    # --- Drift-only population (no directional signal expected) ---

    def test_drift_only_continuous_not_flagged(self):
        """Flat mean across generations should not be flagged as under selection."""
        # Same value every generation → delta = 0 always → z_score = 0
        df = _continuous_df(10, 20, "lr", lambda g, i: 0.1)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert not mean_row["is_under_selection"]
        assert mean_row["cumulative_shift"] == pytest.approx(0.0)

    def test_drift_only_categorical_not_flagged(self):
        """Constant allele frequencies → no selection signal."""
        df = _categorical_df(10, 20, lambda g, i: {"move": 0.5, "gather": 0.5})
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df, pop_size=20)
        # All deltas are zero → mean_delta = 0, and with a Wright–Fisher drift
        # baseline (pop_size provided), drift_std > 0 so z_score should be 0.0.
        for _, row in result.iterrows():
            assert not row["is_under_selection"]
            assert row["z_score"] == pytest.approx(0.0)

    # --- Directional selection (clear signal) ---

    def test_directional_selection_continuous_flagged(self):
        """Linearly increasing mean value should be flagged under selection."""
        # mean increases by 0.05 per generation over 10 generations
        df = _continuous_df(10, 50, "lr", lambda g, i: 0.1 + 0.05 * g + 0.001 * (i - 25))
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df, significance_threshold=2.0)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert mean_row["is_under_selection"]
        assert mean_row["regression_slope"] > 0
        assert mean_row["cumulative_shift"] > 0

    def test_directional_selection_categorical_flagged_with_pop_size(self):
        """Allele frequency sweeping from 0.1 to 0.9 should be flagged."""
        n_gens = 10
        rows = []
        for g in range(n_gens):
            target_freq = 0.1 + 0.08 * g  # 0.1 → 0.82 over 10 gens
            rows.append({"generation": g, "action_weights": {"move": target_freq, "gather": 1.0 - target_freq}})
        df = pd.DataFrame(rows)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df, pop_size=100, significance_threshold=2.0)
        move_row = result[result["allele"] == "move"].iloc[0]
        assert move_row["is_under_selection"]
        assert move_row["regression_slope"] > 0

    # --- Effect size and cumulative shift ---

    def test_cumulative_shift_matches_first_last_difference(self):
        df = pd.DataFrame(
            {
                "generation": [0, 0, 1, 1, 2, 2],
                "chromosome_values": [
                    {"lr": 0.1}, {"lr": 0.1},
                    {"lr": 0.2}, {"lr": 0.2},
                    {"lr": 0.4}, {"lr": 0.4},
                ],
            }
        )
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        # first gen mean = 0.1, last gen mean = 0.4 → shift = 0.3
        assert mean_row["cumulative_shift"] == pytest.approx(0.3)

    def test_effect_size_is_absolute_cumulative_shift(self):
        # Decreasing mean: shift = negative, effect_size = positive
        df = pd.DataFrame(
            {
                "generation": [0, 0, 1, 1],
                "chromosome_values": [
                    {"lr": 0.5}, {"lr": 0.5},
                    {"lr": 0.1}, {"lr": 0.1},
                ],
            }
        )
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert mean_row["cumulative_shift"] == pytest.approx(-0.4)
        assert mean_row["effect_size"] == pytest.approx(0.4)

    def test_regression_slope_positive_for_increasing_trend(self):
        df = pd.DataFrame(
            {
                "generation": [0, 1, 2, 3],
                "chromosome_values": [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}, {"lr": 0.4}],
            }
        )
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert mean_row["regression_slope"] > 0

    def test_regression_slope_negative_for_decreasing_trend(self):
        df = pd.DataFrame(
            {
                "generation": [0, 1, 2, 3],
                "chromosome_values": [{"lr": 0.4}, {"lr": 0.3}, {"lr": 0.2}, {"lr": 0.1}],
            }
        )
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert mean_row["regression_slope"] < 0

    # --- Single generation: nan for multi-generation statistics ---

    def test_single_generation_cumulative_shift_is_nan(self):
        df = _continuous_df(1, 4, "lr", lambda g, i: 0.1)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert math.isnan(mean_row["cumulative_shift"])

    def test_single_generation_z_score_is_nan(self):
        df = _continuous_df(1, 4, "lr", lambda g, i: 0.1)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        mean_row = result[result["allele"] == ALLELE_MEAN].iloc[0]
        assert math.isnan(mean_row["z_score"])

    def test_single_generation_is_under_selection_false(self):
        df = _continuous_df(1, 4, "lr", lambda g, i: 0.1)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        assert not result["is_under_selection"].any()

    # --- Boundary-collapse detection ---

    def test_continuous_collapse_detected_when_variance_reaches_zero(self):
        """All individuals identical in last generation → variance = 0 → collapse."""
        rows = []
        # Generations 0-3: spread; generation 4: collapsed (all same value)
        for g in range(4):
            for i in range(5):
                rows.append({"generation": g, "chromosome_values": {"lr": 0.1 + 0.01 * i}})
        for i in range(5):  # generation 4: all same
            rows.append({"generation": 4, "chromosome_values": {"lr": 0.5}})
        df = pd.DataFrame(rows)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        var_row = result[result["allele"] == ALLELE_VARIANCE].iloc[0]
        assert var_row["collapse_detected"]

    def test_continuous_no_collapse_when_variance_positive(self):
        """Non-zero final variance → no collapse."""
        df = _continuous_df(5, 5, "lr", lambda g, i: 0.1 + 0.01 * i)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        var_row = result[result["allele"] == ALLELE_VARIANCE].iloc[0]
        assert not var_row["collapse_detected"]

    def test_categorical_collapse_detected_when_allele_fixed(self):
        """Allele frequency sweeping to 1.0 in final generation → collapse."""
        rows = []
        for g in range(5):
            freq = min(0.2 * g + 0.2, 1.0)  # 0.2, 0.4, 0.6, 0.8, 1.0
            rows.append({"generation": g, "action_weights": {"move": freq, "gather": 1.0 - freq}})
        df = pd.DataFrame(rows)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        move_row = result[result["allele"] == "move"].iloc[0]
        assert move_row["collapse_detected"]

    def test_categorical_no_collapse_when_alleles_balanced(self):
        df = _categorical_df(5, 10, lambda g, i: {"move": 0.5, "gather": 0.5})
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        assert not result["collapse_detected"].any()

    # --- n_generations column ---

    def test_n_generations_matches_number_of_observed_generations(self):
        n_gens = 7
        df = _continuous_df(n_gens, 3, "lr", lambda g, i: 0.1)
        freq_df = compute_allele_frequency_timeseries(df)
        result = compute_selection_pressure_summary(freq_df)
        assert (result["n_generations"] == n_gens).all()

    # --- Configurable significance threshold ---

    def test_higher_threshold_reduces_number_flagged(self):
        # Build a strong but finite directional signal
        df = _continuous_df(10, 50, "lr", lambda g, i: 0.05 * g + 0.001 * (i - 25))
        freq_df = compute_allele_frequency_timeseries(df)
        result_low = compute_selection_pressure_summary(freq_df, significance_threshold=1.0)
        result_high = compute_selection_pressure_summary(freq_df, significance_threshold=10.0)
        n_low = int(result_low["is_under_selection"].sum())
        n_high = int(result_high["is_under_selection"].sum())
        assert n_low >= n_high  # higher threshold cannot flag more


# ---------------------------------------------------------------------------
# Helpers for fitness landscape tests
# ---------------------------------------------------------------------------


def _make_evolution_df(
    n_candidates: int,
    gene_specs: dict,
    fitness_fn,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic evolution DataFrame.

    Parameters
    ----------
    n_candidates:
        Total number of candidate rows.
    gene_specs:
        ``{gene_name: (min_val, max_val)}`` defining uniform sampling bounds.
    fitness_fn:
        Callable ``(gene_dict) -> float``.
    seed:
        NumPy random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_candidates):
        chrom = {
            gene: float(rng.uniform(lo, hi))
            for gene, (lo, hi) in gene_specs.items()
        }
        rows.append(
            {
                "candidate_id": f"c{i}",
                "generation": i % 5,
                "fitness": fitness_fn(chrom),
                "parent_ids": ["seed"],
                "chromosome_values": chrom,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestComputeFitnessGeneCorrelations
# ---------------------------------------------------------------------------


class TestComputeFitnessGeneCorrelations:
    """Tests for compute_fitness_gene_correlations."""

    def test_empty_df_returns_empty_with_correct_columns(self):
        result = compute_fitness_gene_correlations(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == FITNESS_GENE_CORRELATION_COLUMNS

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"fitness": [1.0, 2.0]})  # no chromosome_values
        result = compute_fitness_gene_correlations(df)
        assert result.empty

    def test_output_columns_are_correct(self):
        df = _make_evolution_df(50, {"lr": (0.0, 1.0)}, lambda g: g["lr"])
        result = compute_fitness_gene_correlations(df)
        assert list(result.columns) == FITNESS_GENE_CORRELATION_COLUMNS

    def test_one_row_per_gene(self):
        df = _make_evolution_df(
            50,
            {"lr": (0.0, 1.0), "gamma": (0.8, 1.0)},
            lambda g: g["lr"] + g["gamma"],
        )
        result = compute_fitness_gene_correlations(df)
        assert set(result["gene"]) == {"lr", "gamma"}

    def test_known_linear_effect_recovered(self):
        """fitness = 10 * lr + noise → high |pearson_r| for lr."""
        rng = np.random.default_rng(42)
        n = 200
        lr_vals = rng.uniform(0.0, 1.0, n)
        noise = rng.normal(0, 0.1, n)
        fitness = 10.0 * lr_vals + noise

        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(fitness[i]),
                "parent_ids": ["seed"],
                "chromosome_values": {"lr": float(lr_vals[i])},
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        result = compute_fitness_gene_correlations(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["gene"] == "lr"
        assert row["pearson_r"] > 0.9, f"Expected strong positive correlation, got {row['pearson_r']}"
        assert row["ols_slope"] > 8.0, f"Expected slope ~10, got {row['ols_slope']}"
        assert row["ols_p"] < 1e-10

    def test_noise_only_not_significant_after_bh(self):
        """Pure noise → no gene should survive BH correction."""
        rng = np.random.default_rng(7)
        n = 100
        lr_vals = rng.uniform(0.0, 1.0, n)
        gamma_vals = rng.uniform(0.8, 1.0, n)
        fitness = rng.normal(0, 1.0, n)  # completely independent

        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(fitness[i]),
                "parent_ids": ["seed"],
                "chromosome_values": {
                    "lr": float(lr_vals[i]),
                    "gamma": float(gamma_vals[i]),
                },
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        result = compute_fitness_gene_correlations(df)
        # With pure noise, BH should reject nothing at α=0.05
        assert not result["bh_rejected"].any(), (
            f"Expected no BH rejections for pure noise, got {result['bh_rejected'].sum()}"
        )

    def test_sorted_by_descending_effect_size(self):
        df = _make_evolution_df(
            100,
            {"strong": (0.0, 1.0), "weak": (0.0, 1.0)},
            lambda g: 10 * g["strong"] + 0.1 * g["weak"],
        )
        result = compute_fitness_gene_correlations(df)
        assert len(result) >= 2
        effect_sizes = result["effect_size"].tolist()
        assert effect_sizes == sorted(effect_sizes, reverse=True)

    def test_ci_bounds_present_and_ordered(self):
        df = _make_evolution_df(50, {"lr": (0.0, 1.0)}, lambda g: 5 * g["lr"])
        result = compute_fitness_gene_correlations(df)
        row = result.iloc[0]
        assert row["ci_lower"] < row["ols_slope"] < row["ci_upper"]

    def test_min_samples_filter(self):
        """Genes with fewer than min_samples rows should be skipped."""
        df = _make_evolution_df(5, {"lr": (0.0, 1.0)}, lambda g: g["lr"])
        result = compute_fitness_gene_correlations(df, min_samples=10)
        assert result.empty

    def test_constant_gene_excluded(self):
        """Zero-variance genes must not appear in results."""
        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(i),
                "parent_ids": ["seed"],
                "chromosome_values": {"lr": 0.5, "gamma": float(i) / 50},
            }
            for i in range(50)
        ]
        df = pd.DataFrame(rows)
        result = compute_fitness_gene_correlations(df)
        assert "lr" not in result["gene"].values

    def test_bh_rejected_column_is_bool(self):
        df = _make_evolution_df(50, {"lr": (0.0, 1.0)}, lambda g: g["lr"])
        result = compute_fitness_gene_correlations(df)
        assert result["bh_rejected"].dtype == bool or result["bh_rejected"].isin([True, False]).all()

    def test_missing_gene_values_use_complete_cases_per_gene(self):
        rows = [
            {
                "candidate_id": "c0",
                "generation": 0,
                "fitness": 1.0,
                "parent_ids": ["seed"],
                "chromosome_values": {"a": 1.0, "b": 1.0},
            },
            {
                "candidate_id": "c1",
                "generation": 0,
                "fitness": 2.0,
                "parent_ids": ["seed"],
                "chromosome_values": {"a": 2.0},
            },
            {
                "candidate_id": "c2",
                "generation": 0,
                "fitness": 3.0,
                "parent_ids": ["seed"],
                "chromosome_values": {"b": 3.0},
            },
            {
                "candidate_id": "c3",
                "generation": 0,
                "fitness": 4.0,
                "parent_ids": ["seed"],
                "chromosome_values": {"a": 4.0, "b": 4.0},
            },
            {
                "candidate_id": "c4",
                "generation": 0,
                "fitness": 5.0,
                "parent_ids": ["seed"],
                "chromosome_values": {"a": 5.0},
            },
        ]
        df = pd.DataFrame(rows)
        result = compute_fitness_gene_correlations(df, min_samples=3)

        assert set(result["gene"]) == {"a", "b"}
        n_samples_by_gene = dict(zip(result["gene"], result["n_samples"]))
        assert n_samples_by_gene["a"] == 4
        assert n_samples_by_gene["b"] == 3
        assert result["pearson_r"].notna().all()
        assert result["ols_p"].notna().all()

    def test_invalid_parameters_raise(self):
        df = _make_evolution_df(20, {"lr": (0.0, 1.0)}, lambda g: g["lr"])
        with pytest.raises(ValueError, match="alpha"):
            compute_fitness_gene_correlations(df, alpha=0.0)
        with pytest.raises(ValueError, match="min_samples"):
            compute_fitness_gene_correlations(df, min_samples=1)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_fitness_gene_correlations(df, confidence_level=1.0)


# ---------------------------------------------------------------------------
# TestComputePairwiseEpistasis
# ---------------------------------------------------------------------------


class TestComputePairwiseEpistasis:
    """Tests for compute_pairwise_epistasis."""

    def test_empty_df_returns_empty_with_correct_columns(self):
        result = compute_pairwise_epistasis(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == PAIRWISE_EPISTASIS_COLUMNS

    def test_single_gene_returns_empty(self):
        df = _make_evolution_df(50, {"lr": (0.0, 1.0)}, lambda g: g["lr"])
        result = compute_pairwise_epistasis(df)
        assert result.empty

    def test_output_columns_are_correct(self):
        df = _make_evolution_df(
            50,
            {"lr": (0.0, 1.0), "gamma": (0.8, 1.0)},
            lambda g: g["lr"] * g["gamma"],
        )
        result = compute_pairwise_epistasis(df, min_samples=20)
        assert list(result.columns) == PAIRWISE_EPISTASIS_COLUMNS

    def test_known_interaction_recovered(self):
        """fitness = 5 * lr * gamma → large |interaction_coef| for lr × gamma."""
        rng = np.random.default_rng(99)
        n = 300
        lr_vals = rng.uniform(0.0, 1.0, n)
        gamma_vals = rng.uniform(0.0, 1.0, n)
        noise = rng.normal(0, 0.05, n)
        fitness = 5.0 * lr_vals * gamma_vals + noise

        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(fitness[i]),
                "parent_ids": ["seed"],
                "chromosome_values": {
                    "lr": float(lr_vals[i]),
                    "gamma": float(gamma_vals[i]),
                },
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        result = compute_pairwise_epistasis(df, min_samples=20)
        assert len(result) == 1
        row = result.iloc[0]
        assert abs(row["interaction_coef"]) > 3.0, (
            f"Expected interaction ~5, got {row['interaction_coef']}"
        )
        assert row["interaction_p"] < 1e-5

    def test_pure_noise_no_significant_hits_after_bh(self):
        """Pure additive noise → no pair significant after BH correction."""
        rng = np.random.default_rng(13)
        n = 200
        lr_vals = rng.uniform(0.0, 1.0, n)
        gamma_vals = rng.uniform(0.8, 1.0, n)
        epsilon_vals = rng.uniform(0.9, 1.0, n)
        fitness = rng.normal(0, 1.0, n)  # independent of all genes

        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(fitness[i]),
                "parent_ids": ["seed"],
                "chromosome_values": {
                    "lr": float(lr_vals[i]),
                    "gamma": float(gamma_vals[i]),
                    "epsilon": float(epsilon_vals[i]),
                },
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        result = compute_pairwise_epistasis(df, min_samples=20)
        # Pure noise: BH should not reject any interaction
        assert not result["bh_rejected"].any(), (
            f"Expected no BH rejections for pure noise, got {result['bh_rejected'].sum()}"
        )

    def test_sorted_by_descending_absolute_interaction(self):
        rng = np.random.default_rng(55)
        n = 200
        a = rng.uniform(0, 1, n)
        b = rng.uniform(0, 1, n)
        c = rng.uniform(0, 1, n)
        # Strong a*b interaction, no a*c or b*c
        fitness = 10 * a * b + 0.5 * c + rng.normal(0, 0.1, n)

        rows = [
            {
                "candidate_id": f"c{i}",
                "generation": 0,
                "fitness": float(fitness[i]),
                "parent_ids": ["seed"],
                "chromosome_values": {
                    "a": float(a[i]),
                    "b": float(b[i]),
                    "c": float(c[i]),
                },
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        result = compute_pairwise_epistasis(df, min_samples=20)
        assert len(result) >= 1
        abs_coefs = result["interaction_coef"].abs().tolist()
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_min_samples_filter(self):
        df = _make_evolution_df(
            15,
            {"lr": (0.0, 1.0), "gamma": (0.8, 1.0)},
            lambda g: g["lr"] * g["gamma"],
        )
        result = compute_pairwise_epistasis(df, min_samples=20)
        assert result.empty

    def test_bh_column_is_bool(self):
        df = _make_evolution_df(
            60,
            {"lr": (0.0, 1.0), "gamma": (0.8, 1.0)},
            lambda g: g["lr"] * g["gamma"],
        )
        result = compute_pairwise_epistasis(df, min_samples=20)
        if not result.empty:
            assert result["bh_rejected"].isin([True, False]).all()

    def test_three_genes_produces_three_pairs(self):
        df = _make_evolution_df(
            60,
            {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)},
            lambda g: g["a"] + g["b"] + g["c"],
        )
        result = compute_pairwise_epistasis(df, min_samples=20)
        assert len(result) == 3

    def test_pairwise_uses_complete_cases_when_gene_values_missing(self):
        rng = np.random.default_rng(101)
        rows = []
        for i in range(30):
            a = float(rng.uniform(0.0, 1.0))
            b = float(rng.uniform(0.0, 1.0))
            fitness = 4.0 * a * b
            chrom = {"a": a, "b": b} if i % 2 == 0 else {"a": a}
            rows.append(
                {
                    "candidate_id": f"c{i}",
                    "generation": 0,
                    "fitness": fitness,
                    "parent_ids": ["seed"],
                    "chromosome_values": chrom,
                }
            )

        df = pd.DataFrame(rows)
        result = compute_pairwise_epistasis(df, min_samples=10)

        assert len(result) == 1
        assert int(result.iloc[0]["n_samples"]) == 15
        assert pd.notna(result.iloc[0]["interaction_p"])

    def test_invalid_parameters_raise(self):
        df = _make_evolution_df(
            60,
            {"lr": (0.0, 1.0), "gamma": (0.8, 1.0)},
            lambda g: g["lr"] * g["gamma"],
        )
        with pytest.raises(ValueError, match="alpha"):
            compute_pairwise_epistasis(df, alpha=1.5)
        with pytest.raises(ValueError, match="min_samples"):
            compute_pairwise_epistasis(df, min_samples=1)
        with pytest.raises(ValueError, match="min_gene_variance"):
            compute_pairwise_epistasis(df, min_gene_variance=-1.0)


# ---------------------------------------------------------------------------
# simulate_wright_fisher
# ---------------------------------------------------------------------------


def _make_two_allele_freqs(p: float = 0.5) -> dict:
    return {"A": p, "B": 1.0 - p}


class TestSimulateWrightFisher:
    """Tests for simulate_wright_fisher."""

    # --- Output shape and columns ---

    def test_output_columns_correct(self):
        result = simulate_wright_fisher({"A": 1.0}, n_effective=10, n_generations=5)
        assert list(result.columns) == WRIGHT_FISHER_COLUMNS

    def test_output_row_count(self):
        result = simulate_wright_fisher(
            _make_two_allele_freqs(), n_effective=10, n_generations=5
        )
        # (n_generations + 1) rows per allele
        assert len(result) == 2 * 6

    def test_generation_zero_matches_initial(self):
        init = {"A": 0.3, "B": 0.7}
        result = simulate_wright_fisher(init, n_effective=50, n_generations=3)
        gen0 = result[result["generation"] == 0].set_index("allele")["frequency"]
        assert abs(gen0["A"] - 0.3) < 1e-9
        assert abs(gen0["B"] - 0.7) < 1e-9

    def test_frequencies_sum_to_one_per_generation(self):
        result = simulate_wright_fisher(
            {"A": 0.4, "B": 0.4, "C": 0.2}, n_effective=100, n_generations=10, seed=0
        )
        for gen, grp in result.groupby("generation"):
            total = grp["frequency"].sum()
            assert abs(total - 1.0) < 1e-9, f"generation {gen}: sum={total}"

    def test_frequencies_in_unit_interval(self):
        result = simulate_wright_fisher(
            _make_two_allele_freqs(0.5), n_effective=20, n_generations=20, seed=1
        )
        assert (result["frequency"] >= 0.0).all()
        assert (result["frequency"] <= 1.0).all()

    def test_zero_generations_returns_only_initial(self):
        result = simulate_wright_fisher({"A": 0.6, "B": 0.4}, n_effective=50, n_generations=0)
        assert set(result["generation"].unique()) == {0}
        assert len(result) == 2

    def test_single_allele_stays_fixed(self):
        result = simulate_wright_fisher({"X": 1.0}, n_effective=100, n_generations=10, seed=42)
        assert (result["frequency"] == 1.0).all()

    # --- Determinism with seed ---

    def test_seeded_runs_are_identical(self):
        kwargs = dict(
            initial_frequencies=_make_two_allele_freqs(0.3),
            n_effective=50,
            n_generations=10,
        )
        r1 = simulate_wright_fisher(**kwargs, seed=7)
        r2 = simulate_wright_fisher(**kwargs, seed=7)
        assert r1["frequency"].tolist() == r2["frequency"].tolist()

    def test_different_seeds_differ(self):
        kwargs = dict(
            initial_frequencies=_make_two_allele_freqs(0.5),
            n_effective=20,
            n_generations=20,
        )
        r1 = simulate_wright_fisher(**kwargs, seed=0)
        r2 = simulate_wright_fisher(**kwargs, seed=99)
        # With high probability the trajectories will differ
        assert r1["frequency"].tolist() != r2["frequency"].tolist()

    # --- Statistical properties ---

    def test_mean_trajectory_preserved(self):
        """E[p(t)] = p(0) across many independent runs (neutral drift)."""
        rng = np.random.default_rng(42)
        p0 = 0.4
        n_reps = 500
        n_eff = 200
        n_gen = 20
        final_freqs = []
        for s in range(n_reps):
            r = simulate_wright_fisher(
                {"A": p0, "B": 1.0 - p0},
                n_effective=n_eff,
                n_generations=n_gen,
                seed=int(rng.integers(0, 10**6)),
            )
            last_gen = r[r["generation"] == n_gen]
            freq_a = float(last_gen.loc[last_gen["allele"] == "A", "frequency"].iloc[0])
            final_freqs.append(freq_a)
        mean_final = float(np.mean(final_freqs))
        # Mean should be close to p0; allow 3 SEM tolerance
        sem = float(np.std(final_freqs)) / (n_reps ** 0.5)
        assert abs(mean_final - p0) < 3 * sem, (
            f"mean final frequency {mean_final:.4f} deviates too much from p0={p0}"
        )

    def test_variance_scales_with_population_size(self):
        """Smaller N_e → greater drift variance."""
        p0 = 0.5
        n_gen = 30
        n_reps = 300
        results = {}
        for n_eff in (10, 500):
            final_freqs = []
            for s in range(n_reps):
                r = simulate_wright_fisher(
                    {"A": p0, "B": 1.0 - p0},
                    n_effective=n_eff,
                    n_generations=n_gen,
                    seed=s,
                )
                last = r[(r["generation"] == n_gen) & (r["allele"] == "A")]
                final_freqs.append(float(last["frequency"].iloc[0]))
            results[n_eff] = float(np.var(final_freqs))
        assert results[10] > results[500], (
            "smaller N_e should show higher variance: "
            f"var(N=10)={results[10]:.4f}, var(N=500)={results[500]:.4f}"
        )

    # --- Input validation ---

    def test_empty_frequencies_raises(self):
        with pytest.raises(ValueError, match="initial_frequencies"):
            simulate_wright_fisher({}, n_effective=10, n_generations=5)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            simulate_wright_fisher({"A": -0.1, "B": 1.1}, n_effective=10, n_generations=5)

    def test_frequencies_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            simulate_wright_fisher({"A": 0.3, "B": 0.3}, n_effective=10, n_generations=5)

    def test_zero_n_effective_raises(self):
        with pytest.raises(ValueError, match="n_effective"):
            simulate_wright_fisher({"A": 1.0}, n_effective=0, n_generations=5)

    def test_negative_n_generations_raises(self):
        with pytest.raises(ValueError, match="n_generations"):
            simulate_wright_fisher({"A": 1.0}, n_effective=10, n_generations=-1)


# ---------------------------------------------------------------------------
# compute_fst_pairwise
# ---------------------------------------------------------------------------


def _make_evolution_df_subpop(
    n_per_subpop: int,
    subpops: dict,  # {subpop_name: {gene: (lo, hi)}}
    seed: int = 0,
) -> pd.DataFrame:
    """Build an evolution-style DataFrame with agent_type subpopulations.

    Each subpop has distinct gene ranges to create differentiation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    cand_idx = 0
    for subpop, gene_ranges in subpops.items():
        for _ in range(n_per_subpop):
            chrom = {
                gene: float(rng.uniform(lo, hi))
                for gene, (lo, hi) in gene_ranges.items()
            }
            rows.append(
                {
                    "candidate_id": f"c{cand_idx}",
                    "generation": 0,
                    "fitness": 1.0,
                    "parent_ids": [],
                    "chromosome_values": chrom,
                    "agent_type": subpop,
                }
            )
            cand_idx += 1
    return pd.DataFrame(rows)


def _make_agent_df_subpop(
    n_per_subpop: int,
    subpops: dict,  # {subpop_name: {action: (lo, hi)}}
    seed: int = 0,
) -> pd.DataFrame:
    """Build an agent genetics DataFrame with agent_type subpopulations."""
    rng = np.random.default_rng(seed)
    rows = []
    agent_idx = 0
    for subpop, action_ranges in subpops.items():
        for _ in range(n_per_subpop):
            weights_raw = {a: float(rng.uniform(lo, hi)) for a, (lo, hi) in action_ranges.items()}
            total = sum(weights_raw.values())
            weights = {a: w / total for a, w in weights_raw.items()}
            rows.append(
                {
                    "agent_id": f"a{agent_idx}",
                    "agent_type": subpop,
                    "generation": 0,
                    "birth_time": 0,
                    "death_time": None,
                    "genome_id": f"::{agent_idx}",
                    "parent_ids": [],
                    "action_weights": weights,
                }
            )
            agent_idx += 1
    return pd.DataFrame(rows)


class TestComputeFstPairwise:
    """Tests for compute_fst_pairwise."""

    # --- Output shape and columns ---

    def test_output_columns_correct(self):
        df = _make_evolution_df_subpop(
            20, {"A": {"lr": (0.0, 0.1)}, "B": {"lr": (0.9, 1.0)}}
        )
        result = compute_fst_pairwise(df)
        assert list(result.columns) == FST_COLUMNS

    def test_empty_df_returns_empty(self):
        result = compute_fst_pairwise(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == FST_COLUMNS

    def test_missing_subpop_col_returns_empty(self):
        df = _make_evolution_df_subpop(10, {"A": {"lr": (0.0, 1.0)}})
        result = compute_fst_pairwise(df, subpop_col="nonexistent")
        assert result.empty

    def test_single_subpopulation_returns_empty(self):
        df = _make_evolution_df_subpop(20, {"A": {"lr": (0.0, 0.5)}})
        result = compute_fst_pairwise(df)
        assert result.empty

    def test_no_locus_columns_returns_empty(self):
        df = pd.DataFrame({"agent_type": ["A", "B", "A"], "other": [1, 2, 3]})
        result = compute_fst_pairwise(df)
        assert result.empty

    # --- F_ST values for continuous loci ---

    def test_isolated_subpops_high_fst_continuous(self):
        """Subpopulations with completely non-overlapping gene ranges → high F_ST."""
        df = _make_evolution_df_subpop(
            100,
            {"A": {"lr": (0.0, 0.01)}, "B": {"lr": (0.99, 1.0)}},
            seed=0,
        )
        result = compute_fst_pairwise(df)
        fst_vals = result[result["locus"] == "lr"]["fst"].values
        assert len(fst_vals) == 1
        assert fst_vals[0] > 0.9, f"expected high F_ST, got {fst_vals[0]:.4f}"

    def test_fully_mixed_subpops_low_fst_continuous(self):
        """Subpopulations drawn from the same distribution → F_ST ≈ 0."""
        rng = np.random.default_rng(42)
        vals = rng.uniform(0.0, 1.0, 200)
        rows = [
            {
                "chromosome_values": {"lr": float(v)},
                "agent_type": "A" if i < 100 else "B",
                "generation": 0,
            }
            for i, v in enumerate(vals)
        ]
        df = pd.DataFrame(rows)
        result = compute_fst_pairwise(df)
        fst_vals = result[result["locus"] == "lr"]["fst"].values
        assert len(fst_vals) == 1
        assert fst_vals[0] < 0.05, f"expected ~0 F_ST for mixed pops, got {fst_vals[0]:.4f}"

    # --- F_ST values for categorical loci ---

    def test_isolated_subpops_high_fst_categorical(self):
        """Subpopulations with completely different action preferences → high F_ST."""
        n = 50
        rows = []
        for i in range(n):
            rows.append(
                {"agent_type": "A", "action_weights": {"move": 1.0, "gather": 0.0}, "generation": 0}
            )
        for i in range(n):
            rows.append(
                {"agent_type": "B", "action_weights": {"move": 0.0, "gather": 1.0}, "generation": 0}
            )
        df = pd.DataFrame(rows)
        result = compute_fst_pairwise(df)
        fst_row = result[result["locus"] == "action_weights"]
        assert len(fst_row) == 1
        assert float(fst_row["fst"].iloc[0]) > 0.9

    def test_identical_subpops_zero_fst_categorical(self):
        """Subpopulations with identical action distributions → F_ST == 0."""
        n = 30
        weights = {"move": 0.5, "gather": 0.5}
        rows = (
            [{"agent_type": "A", "action_weights": weights, "generation": 0}] * n
            + [{"agent_type": "B", "action_weights": weights, "generation": 0}] * n
        )
        df = pd.DataFrame(rows)
        result = compute_fst_pairwise(df)
        fst_row = result[result["locus"] == "action_weights"]
        assert len(fst_row) == 1
        assert abs(float(fst_row["fst"].iloc[0])) < 1e-9

    # --- F_ST range ---

    def test_fst_in_unit_interval(self):
        df = _make_evolution_df_subpop(
            30,
            {
                "A": {"lr": (0.0, 0.5), "gamma": (0.8, 1.0)},
                "B": {"lr": (0.5, 1.0), "gamma": (0.0, 0.2)},
                "C": {"lr": (0.2, 0.7), "gamma": (0.3, 0.7)},
            },
            seed=5,
        )
        result = compute_fst_pairwise(df)
        assert (result["fst"] >= 0.0).all()
        assert (result["fst"] <= 1.0).all()

    def test_three_subpops_produce_three_pairs(self):
        df = _make_evolution_df_subpop(
            20,
            {
                "A": {"lr": (0.0, 0.1)},
                "B": {"lr": (0.45, 0.55)},
                "C": {"lr": (0.9, 1.0)},
            },
            seed=7,
        )
        result = compute_fst_pairwise(df)
        locus_result = result[result["locus"] == "lr"]
        assert len(locus_result) == 3

    def test_custom_subpop_col(self):
        df = _make_evolution_df_subpop(20, {"A": {"lr": (0.0, 0.1)}, "B": {"lr": (0.9, 1.0)}})
        df = df.rename(columns={"agent_type": "region"})
        result = compute_fst_pairwise(df, subpop_col="region")
        assert not result.empty
        assert (result["fst"] > 0.0).any()


# ---------------------------------------------------------------------------
# compute_migration_counts
# ---------------------------------------------------------------------------


def _make_lineage_df(
    n_agents: int = 20,
    n_generations: int = 3,
    migration_rate: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a DataFrame that mimics lineage-based migration.

    Agents in generation 0 are assigned to subpopulations A or B.
    Each subsequent generation, offspring mostly inherit their parent's
    subpopulation; a fraction ``migration_rate`` switches subpopulation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    agent_idx = 0

    # Generation 0: genesis agents, equally split
    gen0_ids = []
    for i in range(n_agents):
        aid = f"g0_a{i}"
        subpop = "A" if i < n_agents // 2 else "B"
        rows.append(
            {
                "agent_id": aid,
                "agent_type": subpop,
                "generation": 0,
                "parent_ids": [],
                "action_weights": {"move": 0.5, "gather": 0.5},
            }
        )
        gen0_ids.append((aid, subpop))

    prev_gen = gen0_ids
    for gen in range(1, n_generations + 1):
        new_gen = []
        for i in range(n_agents):
            parent_aid, parent_subpop = prev_gen[rng.integers(0, len(prev_gen))]
            # Occasionally migrate
            if rng.random() < migration_rate:
                child_subpop = "B" if parent_subpop == "A" else "A"
            else:
                child_subpop = parent_subpop
            aid = f"g{gen}_a{i}"
            rows.append(
                {
                    "agent_id": aid,
                    "agent_type": child_subpop,
                    "generation": gen,
                    "parent_ids": [parent_aid],
                    "action_weights": {"move": 0.5, "gather": 0.5},
                }
            )
            new_gen.append((aid, child_subpop))
        prev_gen = new_gen

    return pd.DataFrame(rows)


class TestComputeMigrationCounts:
    """Tests for compute_migration_counts."""

    def test_output_columns_correct(self):
        df = _make_lineage_df()
        result = compute_migration_counts(df)
        assert list(result.columns) == MIGRATION_COLUMNS

    def test_empty_df_returns_empty(self):
        result = compute_migration_counts(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == MIGRATION_COLUMNS

    def test_missing_subpop_col_returns_empty(self):
        df = _make_lineage_df()
        result = compute_migration_counts(df, subpop_col="nonexistent")
        assert result.empty

    def test_missing_parent_ids_col_returns_empty(self):
        df = pd.DataFrame({"agent_id": ["a1"], "agent_type": ["A"]})
        result = compute_migration_counts(df)
        assert result.empty

    def test_genesis_agents_excluded(self):
        """Genesis agents (empty parent_ids) must not appear in the result."""
        df = _make_lineage_df(n_agents=10, n_generations=2)
        result = compute_migration_counts(df)
        # Generation-0 agents have no parents; none should appear in result
        gen0_ids = set(df[df["generation"] == 0]["agent_id"])
        assert len(set(result["agent_id"]) & gen0_ids) == 0

    def test_zero_migration_rate_no_migrants(self):
        """With migration_rate=0, no agents should be flagged as migrants."""
        df = _make_lineage_df(n_agents=20, n_generations=3, migration_rate=0.0)
        result = compute_migration_counts(df)
        assert not result.empty
        assert result["is_migrant"].sum() == 0

    def test_full_migration_rate_all_migrants(self):
        """With migration_rate=1.0, all offspring switch subpopulation."""
        df = _make_lineage_df(n_agents=20, n_generations=2, migration_rate=1.0, seed=1)
        result = compute_migration_counts(df)
        assert not result.empty
        assert result["is_migrant"].all()

    def test_partial_migration_some_migrants(self):
        df = _make_lineage_df(n_agents=40, n_generations=5, migration_rate=0.3, seed=3)
        result = compute_migration_counts(df)
        n_migrants = int(result["is_migrant"].sum())
        n_non_migrants = int((~result["is_migrant"]).sum())
        assert n_migrants > 0 and n_non_migrants > 0

    def test_subpop_definition_column_reflects_subpop_col(self):
        df = _make_lineage_df()
        result = compute_migration_counts(df, subpop_col="agent_type")
        assert (result["subpop_definition"] == "agent_type").all()

    def test_candidate_id_column_supported(self):
        """Supports candidate_id as the individual identifier (evolution DataFrames)."""
        df = _make_lineage_df()
        df = df.rename(columns={"agent_id": "candidate_id"})
        result = compute_migration_counts(df)
        assert not result.empty
        assert "agent_id" in result.columns


# ---------------------------------------------------------------------------
# compute_gene_flow_timeseries
# ---------------------------------------------------------------------------


def _make_multigen_evolution_df(
    n_gens: int = 5,
    n_per_subpop_per_gen: int = 20,
    gene_ranges_a: Optional[dict] = None,
    gene_ranges_b: Optional[dict] = None,
    migration_rate: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Multi-generation evolution DataFrame with two subpopulations."""
    if gene_ranges_a is None:
        gene_ranges_a = {"lr": (0.0, 0.1), "gamma": (0.9, 1.0)}
    if gene_ranges_b is None:
        gene_ranges_b = {"lr": (0.9, 1.0), "gamma": (0.0, 0.1)}
    rng = np.random.default_rng(seed)
    rows = []
    cand_idx = 0
    prev_gen_info = {}  # cand_id -> agent_type

    for gen in range(n_gens):
        gen_info = {}
        for subpop, gene_ranges in [("A", gene_ranges_a), ("B", gene_ranges_b)]:
            for _ in range(n_per_subpop_per_gen):
                cid = f"c{cand_idx}"
                # Choose parent from previous generation
                if gen == 0 or not prev_gen_info:
                    parent_ids = []
                    effective_subpop = subpop
                else:
                    prev_ids = list(prev_gen_info.keys())
                    parent_id = prev_ids[rng.integers(0, len(prev_ids))]
                    parent_type = prev_gen_info[parent_id]
                    # Occasionally migrate
                    if rng.random() < migration_rate:
                        effective_subpop = "B" if parent_type == "A" else "A"
                    else:
                        effective_subpop = parent_type
                    parent_ids = [parent_id]
                # Gene values from this subpop's distribution
                effective_ranges = gene_ranges_a if effective_subpop == "A" else gene_ranges_b
                chrom = {g: float(rng.uniform(lo, hi)) for g, (lo, hi) in effective_ranges.items()}
                rows.append(
                    {
                        "agent_id": cid,
                        "agent_type": effective_subpop,
                        "generation": gen,
                        "parent_ids": parent_ids,
                        "chromosome_values": chrom,
                    }
                )
                gen_info[cid] = effective_subpop
                cand_idx += 1
        prev_gen_info = gen_info

    return pd.DataFrame(rows)


class TestComputeGeneFlowTimeseries:
    """Tests for compute_gene_flow_timeseries."""

    def test_output_columns_correct(self):
        df = _make_multigen_evolution_df(n_gens=3)
        result = compute_gene_flow_timeseries(df)
        assert list(result.columns) == GENE_FLOW_COLUMNS

    def test_empty_df_returns_empty(self):
        result = compute_gene_flow_timeseries(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == GENE_FLOW_COLUMNS

    def test_missing_generation_col_returns_empty(self):
        df = _make_multigen_evolution_df()
        df = df.drop(columns=["generation"])
        result = compute_gene_flow_timeseries(df)
        assert result.empty

    def test_missing_subpop_col_returns_empty(self):
        df = _make_multigen_evolution_df()
        result = compute_gene_flow_timeseries(df, subpop_col="nonexistent")
        assert result.empty

    def test_no_locus_columns_returns_empty(self):
        df = _make_multigen_evolution_df()
        df = df.drop(columns=["chromosome_values"])
        result = compute_gene_flow_timeseries(df)
        assert result.empty

    def test_rows_per_generation(self):
        n_gens = 4
        df = _make_multigen_evolution_df(n_gens=n_gens, n_per_subpop_per_gen=20)
        result = compute_gene_flow_timeseries(df)
        assert not result.empty
        # Exactly n_gens unique generations represented
        assert result["generation"].nunique() == n_gens

    def test_fst_in_unit_interval(self):
        df = _make_multigen_evolution_df(n_gens=3)
        result = compute_gene_flow_timeseries(df)
        assert (result["fst"] >= 0.0).all()
        assert (result["fst"] <= 1.0).all()

    def test_n_migrants_non_negative_integers(self):
        df = _make_multigen_evolution_df(n_gens=3, migration_rate=0.2)
        result = compute_gene_flow_timeseries(df)
        assert (result["n_migrants"] >= 0).all()
        assert result["n_migrants"].dtype in (int, np.int64, np.int32, object)

    def test_isolated_pops_high_fst_over_time(self):
        """Isolated subpopulations with very different gene ranges → high F_ST every gen."""
        df = _make_multigen_evolution_df(
            n_gens=5,
            n_per_subpop_per_gen=30,
            gene_ranges_a={"lr": (0.0, 0.01)},
            gene_ranges_b={"lr": (0.99, 1.0)},
            migration_rate=0.0,
            seed=11,
        )
        result = compute_gene_flow_timeseries(df)
        lr_rows = result[result["locus"] == "lr"]
        assert not lr_rows.empty
        assert (lr_rows["fst"] > 0.9).all(), lr_rows["fst"].tolist()

    def test_fully_mixed_pops_low_fst_over_time(self):
        """Subpopulations drawn from the same gene distribution → F_ST ≈ 0."""
        rng = np.random.default_rng(99)
        n = 30
        rows = []
        for gen in range(4):
            vals = rng.uniform(0.0, 1.0, n * 2)
            for i, v in enumerate(vals):
                rows.append(
                    {
                        "agent_type": "A" if i < n else "B",
                        "generation": gen,
                        "chromosome_values": {"lr": float(v)},
                        "parent_ids": [],
                    }
                )
        df = pd.DataFrame(rows)
        result = compute_gene_flow_timeseries(df)
        lr_rows = result[result["locus"] == "lr"]
        assert not lr_rows.empty
        assert (lr_rows["fst"] < 0.15).all(), lr_rows["fst"].tolist()

    def test_migration_events_counted_per_generation(self):
        """With migration, n_migrants > 0 in at least some generations."""
        df = _make_multigen_evolution_df(
            n_gens=4,
            n_per_subpop_per_gen=30,
            migration_rate=0.5,
            seed=77,
        )
        result = compute_gene_flow_timeseries(df)
        # At least some generations/pairs should show migrants
        assert result["n_migrants"].sum() > 0
