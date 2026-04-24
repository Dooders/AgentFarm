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
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from farm.analysis.genetics.compute import (
    AGENT_GENETICS_COLUMNS,
    EVOLUTION_GENETICS_COLUMNS,
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
    compute_categorical_locus_diversity,
    compute_continuous_locus_diversity,
    compute_evolution_diversity_timeseries,
    compute_population_diversity,
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
