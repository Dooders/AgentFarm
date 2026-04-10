"""Tests for multi-generation recombination and weight mutation.

Covers:
- MutationConfig validation (noise_std, noise_fraction).
- mutate_state_dict: noise applied, shapes preserved, non-tensor entries copied.
- mutate_state_dict: noise_fraction < 1.0 (sparse mask).
- mutate_state_dict: noise_std=0 is a no-op numerically.
- mutate_state_dict: seed reproducibility.
- GenerationConfig validation.
- GenerationSummary.to_dict round-trip.
- LineageRecord construction.
- run_multi_generation_search: 2-generation smoke test with tiny network.
- run_multi_generation_search: mutation enabled path.
- run_multi_generation_search: best_vs_original strategy.
- run_multi_generation_search: output files (summary JSON, lineage JSON).
- __init__.py exports for all new symbols.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.crossover import (
    MutationConfig,
    mutate_state_dict,
)
from farm.core.decision.training.crossover_search import (
    CrossoverRecipe,
    FineTuneRegime,
    GenerationConfig,
    GenerationSummary,
    LineageRecord,
    SearchConfig,
    _search_config_with_generation_seed_bump,
    run_multi_generation_search,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 4
OUTPUT_DIM = 2
HIDDEN_SIZE = 8
N_STATES = 40


def _make_net(seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=HIDDEN_SIZE)


def _make_states(n: int = N_STATES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _tiny_search_config(max_runs: int = 2) -> SearchConfig:
    """One recipe + one regime, capped at *max_runs* children."""
    recipes = [CrossoverRecipe("random", alpha=0.5, seed=0)]
    regimes = [FineTuneRegime("smoke", epochs=1, lr=1e-3, batch_size=8, seed=0)]
    return SearchConfig(
        crossover_recipes=recipes,
        finetune_regimes=regimes,
        max_runs=max_runs,
    )


def _tiny_gen_config(
    num_generations: int = 2,
    mutation_std: float = 0.0,
    strategy: str = "best",
) -> GenerationConfig:
    """Minimal GenerationConfig for fast tests."""
    mutation = MutationConfig(noise_std=mutation_std, seed=42) if mutation_std > 0 else None
    return GenerationConfig(
        num_generations=num_generations,
        search_config=_tiny_search_config(max_runs=2),
        mutation_config=mutation,
        selection_strategy=strategy,
        seed=0,
    )


# ===========================================================================
# _search_config_with_generation_seed_bump
# ===========================================================================


class TestSearchConfigGenerationSeedBump:
    def test_zero_bump_returns_same_object(self):
        sc = _tiny_search_config()
        assert _search_config_with_generation_seed_bump(sc, 0) is sc

    def test_bump_shifts_integer_seeds(self):
        sc = SearchConfig(
            crossover_recipes=[CrossoverRecipe("random", alpha=0.5, seed=10)],
            finetune_regimes=[FineTuneRegime("smoke", epochs=1, lr=1e-3, batch_size=8, seed=20)],
        )
        out = _search_config_with_generation_seed_bump(sc, 7)
        assert out.crossover_recipes[0].seed == 17
        assert out.finetune_regimes[0].seed == 27
        assert sc.crossover_recipes[0].seed == 10
        assert sc.finetune_regimes[0].seed == 20

    def test_none_seeds_unchanged(self):
        sc = SearchConfig(
            crossover_recipes=[CrossoverRecipe("layer")],
            finetune_regimes=[
                FineTuneRegime("smoke", epochs=1, lr=1e-3, batch_size=8, seed=None),
            ],
        )
        out = _search_config_with_generation_seed_bump(sc, 99)
        assert out.crossover_recipes[0].seed is None
        assert out.finetune_regimes[0].seed is None


# ===========================================================================
# MutationConfig
# ===========================================================================


class TestMutationConfig:
    def test_defaults(self):
        cfg = MutationConfig()
        assert cfg.noise_std == 0.01
        assert cfg.noise_fraction == 1.0
        assert cfg.seed is None

    def test_custom_values(self):
        cfg = MutationConfig(noise_std=0.05, noise_fraction=0.5, seed=42)
        assert cfg.noise_std == 0.05
        assert cfg.noise_fraction == 0.5
        assert cfg.seed == 42

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="noise_std"):
            MutationConfig(noise_std=-0.1)

    def test_zero_std_valid(self):
        cfg = MutationConfig(noise_std=0.0)
        assert cfg.noise_std == 0.0

    def test_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="noise_fraction"):
            MutationConfig(noise_fraction=0.0)

    def test_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="noise_fraction"):
            MutationConfig(noise_fraction=1.1)

    def test_fraction_one_valid(self):
        cfg = MutationConfig(noise_fraction=1.0)
        assert cfg.noise_fraction == 1.0


# ===========================================================================
# mutate_state_dict
# ===========================================================================


class TestMutateStateDict:
    def _simple_sd(self):
        """Small state dict with two float tensors."""
        return {
            "w": torch.tensor([1.0, 2.0, 3.0]),
            "b": torch.tensor([0.1, 0.2]),
        }

    def test_shapes_preserved(self):
        sd = self._simple_sd()
        cfg = MutationConfig(noise_std=0.01, seed=0)
        result = mutate_state_dict(sd, cfg)
        for k in sd:
            assert result[k].shape == sd[k].shape

    def test_noise_applied(self):
        """With nonzero std, values should differ from the original."""
        sd = self._simple_sd()
        cfg = MutationConfig(noise_std=1.0, seed=42)
        result = mutate_state_dict(sd, cfg)
        # At least one key should be different
        changed = any(not torch.equal(sd[k], result[k]) for k in sd)
        assert changed

    def test_zero_std_is_noop(self):
        """noise_std=0 should leave all tensors numerically unchanged."""
        sd = self._simple_sd()
        cfg = MutationConfig(noise_std=0.0, seed=0)
        result = mutate_state_dict(sd, cfg)
        for k in sd:
            assert torch.allclose(sd[k], result[k])

    def test_input_not_modified(self):
        """mutate_state_dict must not modify the source state dict in-place."""
        sd = self._simple_sd()
        original_w = sd["w"].clone()
        cfg = MutationConfig(noise_std=1.0, seed=7)
        mutate_state_dict(sd, cfg)
        assert torch.equal(sd["w"], original_w)

    def test_seed_reproducibility(self):
        sd = self._simple_sd()
        cfg1 = MutationConfig(noise_std=0.5, seed=99)
        cfg2 = MutationConfig(noise_std=0.5, seed=99)
        r1 = mutate_state_dict(sd, cfg1)
        r2 = mutate_state_dict(sd, cfg2)
        for k in sd:
            assert torch.equal(r1[k], r2[k])

    def test_different_seeds_differ(self):
        sd = self._simple_sd()
        r1 = mutate_state_dict(sd, MutationConfig(noise_std=1.0, seed=1))
        r2 = mutate_state_dict(sd, MutationConfig(noise_std=1.0, seed=2))
        any_diff = any(not torch.equal(r1[k], r2[k]) for k in sd)
        assert any_diff

    def test_non_tensor_passthrough(self):
        """Non-tensor entries (scalars, strings) must be deep-copied unchanged."""
        sd = {
            "w": torch.tensor([1.0, 2.0]),
            "version": 3,
            "label": "test",
        }
        cfg = MutationConfig(noise_std=0.1, seed=0)
        result = mutate_state_dict(sd, cfg)
        assert result["version"] == 3
        assert result["label"] == "test"

    def test_sparse_fraction(self):
        """noise_fraction < 1.0 should still produce output of the correct shape."""
        sd = {"w": torch.zeros(100)}
        cfg = MutationConfig(noise_std=1.0, noise_fraction=0.5, seed=0)
        result = mutate_state_dict(sd, cfg)
        assert result["w"].shape == (100,)
        # Some elements should be nonzero (noise applied); some should remain zero.
        n_nonzero = int((result["w"] != 0).sum())
        assert 0 < n_nonzero < 100

    def test_on_model_state_dict(self):
        """mutate_state_dict works end-to-end on a real BaseQNetwork state dict."""
        net = _make_net(seed=5)
        sd = net.state_dict()
        cfg = MutationConfig(noise_std=0.01, seed=3)
        mutated = mutate_state_dict(sd, cfg)
        # Can load back into an identical fresh network
        fresh = _make_net(seed=5)
        fresh.load_state_dict(mutated)  # should not raise

    def test_float32_output(self):
        """Output tensors must be float32 regardless of input dtype."""
        sd = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
        cfg = MutationConfig(noise_std=0.01, seed=0)
        result = mutate_state_dict(sd, cfg)
        assert result["w"].dtype == torch.float32


# ===========================================================================
# GenerationConfig
# ===========================================================================


class TestGenerationConfig:
    def test_basic_construction(self):
        cfg = _tiny_gen_config(num_generations=3)
        assert cfg.num_generations == 3
        assert cfg.selection_strategy == "best"

    def test_zero_generations_raises(self):
        with pytest.raises(ValueError, match="num_generations"):
            GenerationConfig(
                num_generations=0,
                search_config=_tiny_search_config(),
            )

    def test_negative_generations_raises(self):
        with pytest.raises(ValueError, match="num_generations"):
            GenerationConfig(
                num_generations=-1,
                search_config=_tiny_search_config(),
            )

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="selection_strategy"):
            GenerationConfig(
                num_generations=1,
                search_config=_tiny_search_config(),
                selection_strategy="tournament",
            )

    def test_with_mutation(self):
        cfg = _tiny_gen_config(mutation_std=0.01)
        assert cfg.mutation_config is not None
        assert cfg.mutation_config.noise_std == 0.01

    def test_no_mutation(self):
        cfg = _tiny_gen_config(mutation_std=0.0)
        assert cfg.mutation_config is None


# ===========================================================================
# GenerationSummary
# ===========================================================================


class TestGenerationSummary:
    def _make_summary(self, gen: int = 0) -> GenerationSummary:
        return GenerationSummary(
            generation=gen,
            n_children=3,
            best_primary_metric=0.85,
            mean_primary_metric=0.77,
            best_child_id="000_random_a0p50_s0_smoke",
            promoted_parent_a_id="000_random_a0p50_s0_smoke",
            promoted_parent_b_id="original_parent_b",
            run_dir="/tmp/gen_0",
        )

    def test_to_dict_keys(self):
        s = self._make_summary()
        d = s.to_dict()
        expected = {
            "generation", "n_children", "best_primary_metric",
            "mean_primary_metric", "best_child_id",
            "promoted_parent_a_id", "promoted_parent_b_id", "run_dir",
        }
        assert expected <= set(d.keys())

    def test_to_dict_values(self):
        s = self._make_summary(gen=2)
        d = s.to_dict()
        assert d["generation"] == 2
        assert d["n_children"] == 3
        assert abs(d["best_primary_metric"] - 0.85) < 1e-6

    def test_json_serialisable(self):
        s = self._make_summary()
        json.dumps(s.to_dict())  # should not raise


# ===========================================================================
# LineageRecord
# ===========================================================================


class TestLineageRecord:
    def test_construction(self):
        lr = LineageRecord(
            generation=1,
            child_id="001_layer_a0p50_sNone_smoke",
            parent_a_id="000_random_a0p50_s0_smoke",
            parent_b_id="original_parent_b",
            primary_metric=0.72,
            mutation_applied=True,
        )
        assert lr.generation == 1
        assert lr.mutation_applied is True


# ===========================================================================
# run_multi_generation_search
# ===========================================================================


class TestRunMultiGenerationSearch:
    def test_two_generation_smoke(self):
        """Acceptance criterion: produces >= 2 generations with logged metrics."""
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2)

        with tempfile.TemporaryDirectory() as run_dir:
            summaries, lineage = run_multi_generation_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        assert len(summaries) == 2
        for s in summaries:
            assert s.n_children >= 1
            assert 0.0 <= s.best_primary_metric <= 1.0
            assert 0.0 <= s.mean_primary_metric <= 1.0

    def test_lineage_populated(self):
        """Lineage registry has one record per child per generation."""
        parent_a = _make_net(seed=2)
        parent_b = _make_net(seed=3)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2)

        with tempfile.TemporaryDirectory() as run_dir:
            summaries, lineage = run_multi_generation_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        total_children = sum(s.n_children for s in summaries)
        assert len(lineage) == total_children
        # Generation 0 should show original parent IDs
        gen0_records = [lr for lr in lineage if lr.generation == 0]
        assert all(lr.parent_a_id == "original_parent_a" for lr in gen0_records)
        assert all(lr.parent_b_id == "original_parent_b" for lr in gen0_records)

    def test_output_files_created(self):
        """Summary and lineage JSON must be written to run_dir."""
        parent_a = _make_net(seed=4)
        parent_b = _make_net(seed=5)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2)

        with tempfile.TemporaryDirectory() as run_dir:
            run_multi_generation_search(parent_a, parent_b, states, cfg, run_dir)

            assert os.path.isfile(os.path.join(run_dir, "multi_gen_summary.json"))
            assert os.path.isfile(os.path.join(run_dir, "lineage.json"))
            # Per-generation sub-dirs
            assert os.path.isdir(os.path.join(run_dir, "gen_0"))
            assert os.path.isdir(os.path.join(run_dir, "gen_1"))

    def test_summary_json_content(self):
        """multi_gen_summary.json must be a list of serialisable dicts."""
        parent_a = _make_net(seed=6)
        parent_b = _make_net(seed=7)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2)

        with tempfile.TemporaryDirectory() as run_dir:
            run_multi_generation_search(parent_a, parent_b, states, cfg, run_dir)

            with open(os.path.join(run_dir, "multi_gen_summary.json")) as fh:
                summary_data = json.load(fh)

        assert isinstance(summary_data, list)
        assert len(summary_data) == 2
        assert "best_primary_metric" in summary_data[0]
        assert "generation" in summary_data[0]

    def test_lineage_json_content(self):
        """lineage.json must contain records for both generations."""
        parent_a = _make_net(seed=8)
        parent_b = _make_net(seed=9)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2)

        with tempfile.TemporaryDirectory() as run_dir:
            run_multi_generation_search(parent_a, parent_b, states, cfg, run_dir)

            with open(os.path.join(run_dir, "lineage.json")) as fh:
                lineage_data = json.load(fh)

        assert isinstance(lineage_data, list)
        generations_seen = {r["generation"] for r in lineage_data}
        assert 0 in generations_seen
        assert 1 in generations_seen
        # All required fields present
        for record in lineage_data:
            assert "child_id" in record
            assert "parent_a_id" in record
            assert "parent_b_id" in record
            assert "primary_metric" in record
            assert "mutation_applied" in record

    def test_with_mutation_enabled(self):
        """Pipeline runs without error when mutation is enabled."""
        parent_a = _make_net(seed=10)
        parent_b = _make_net(seed=11)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2, mutation_std=0.05)

        with tempfile.TemporaryDirectory() as run_dir:
            summaries, lineage = run_multi_generation_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        assert len(summaries) == 2
        # Generation 1+ records should have mutation_applied=True
        gen1_records = [lr for lr in lineage if lr.generation == 1]
        assert all(lr.mutation_applied for lr in gen1_records)
        # Generation 0 records should have mutation_applied=False
        gen0_records = [lr for lr in lineage if lr.generation == 0]
        assert all(not lr.mutation_applied for lr in gen0_records)

    def test_best_vs_original_strategy(self):
        """best_vs_original: generation-1 parent_b_id must be 'original_parent_b'."""
        parent_a = _make_net(seed=12)
        parent_b = _make_net(seed=13)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=2, strategy="best_vs_original")

        with tempfile.TemporaryDirectory() as run_dir:
            _, lineage = run_multi_generation_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        gen1_records = [lr for lr in lineage if lr.generation == 1]
        assert all(lr.parent_b_id == "original_parent_b" for lr in gen1_records)

    def test_parent_state_restored(self):
        """Parents' training mode and requires_grad must be restored after search."""
        parent_a = _make_net(seed=14)
        parent_b = _make_net(seed=15)
        # Put in train mode with gradients enabled
        parent_a.train()
        parent_b.train()
        for p in parent_a.parameters():
            p.requires_grad_(True)
        for p in parent_b.parameters():
            p.requires_grad_(True)

        states = _make_states()
        cfg = _tiny_gen_config(num_generations=1)

        with tempfile.TemporaryDirectory() as run_dir:
            run_multi_generation_search(parent_a, parent_b, states, cfg, run_dir)

        # run_crossover_search restores training state; multi-gen should too
        assert all(p.requires_grad for p in parent_a.parameters())

    def test_single_generation(self):
        """With num_generations=1, should behave like a single-shot search."""
        parent_a = _make_net(seed=20)
        parent_b = _make_net(seed=21)
        states = _make_states()
        cfg = _tiny_gen_config(num_generations=1)

        with tempfile.TemporaryDirectory() as run_dir:
            summaries, lineage = run_multi_generation_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        assert len(summaries) == 1
        assert summaries[0].promoted_parent_a_id is None
        assert summaries[0].promoted_parent_b_id is None


# ===========================================================================
# __init__.py exports
# ===========================================================================


class TestInitExports:
    def test_new_symbols_exported(self):
        from farm.core.decision import training as pkg

        assert hasattr(pkg, "MutationConfig")
        assert hasattr(pkg, "mutate_state_dict")
        assert hasattr(pkg, "GenerationConfig")
        assert hasattr(pkg, "GenerationSummary")
        assert hasattr(pkg, "LineageRecord")
        assert hasattr(pkg, "run_multi_generation_search")
