"""Smoke tests for farm/core/decision/training/crossover_search.py.

Covers:
- CrossoverRecipe validation (valid/invalid mode and alpha).
- FineTuneRegime construction.
- SearchConfig.default() and SearchConfig.minimal() produce expected pair counts.
- SearchConfig.max_runs truncation.
- SearchConfig.pairs() Cartesian product order.
- ManifestEntry.to_dict() round-trip serialisation.
- _make_child_id() produces deterministic strings.
- build_leaderboard() sorts by primary_metric, appends parent baselines.
- generate_recommendation() produces non-empty text for varied manifests.
- run_crossover_search() end-to-end with 2-3 configs on synthetic data
  (fast, uses tiny networks and few epochs so the test stays fast).
- __init__.py exports.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.crossover_search import (
    CrossoverRecipe,
    FineTuneRegime,
    ManifestEntry,
    SearchConfig,
    _make_child_id,
    build_leaderboard,
    generate_recommendation,
    run_crossover_search,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 4
OUTPUT_DIM = 2
HIDDEN_SIZE = 8
N_STATES = 60


def _make_net(seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=HIDDEN_SIZE)


def _make_states(n: int = N_STATES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _make_entry(
    child_id: str = "000_random_a0p50_s0_short",
    primary_metric: float = 0.75,
    agree_a: float = 0.80,
    agree_b: float = 0.75,
    degenerate: bool = False,
    regime: str = "short",
    mode: str = "random",
    alpha: float = 0.5,
    seed: int = 0,
) -> ManifestEntry:
    """Construct a minimal ManifestEntry for unit tests."""
    return ManifestEntry(
        child_id=child_id,
        crossover_mode=mode,
        crossover_alpha=alpha,
        crossover_seed=seed,
        finetune_regime=regime,
        finetune_epochs=5,
        finetune_lr=1e-3,
        finetune_quantization_applied="none",
        child_pt_path="/tmp/child.pt",
        eval_report_path="/tmp/eval.json",
        run_config_path="/tmp/run_config.json",
        primary_metric=primary_metric,
        child_vs_parent_a_agreement=agree_a,
        child_vs_parent_b_agreement=agree_b,
        oracle_agreement=max(agree_a, agree_b),
        kl_divergence_a=0.5,
        kl_divergence_b=0.6,
        mse_a=1.0,
        mse_b=1.2,
        cosine_a=0.85,
        cosine_b=0.80,
        degenerate=degenerate,
    )


# ---------------------------------------------------------------------------
# CrossoverRecipe
# ---------------------------------------------------------------------------


class TestCrossoverRecipe:
    def test_valid_modes(self):
        for mode in ("random", "layer", "weighted"):
            r = CrossoverRecipe(mode)
            assert r.mode == mode

    def test_default_alpha(self):
        r = CrossoverRecipe("random")
        assert r.alpha == 0.5

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            CrossoverRecipe("invalid_mode")

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha must be"):
            CrossoverRecipe("weighted", alpha=1.5)

    def test_alpha_boundary(self):
        CrossoverRecipe("weighted", alpha=0.0)
        CrossoverRecipe("weighted", alpha=1.0)

    def test_seed_optional(self):
        r = CrossoverRecipe("random", seed=None)
        assert r.seed is None
        r2 = CrossoverRecipe("random", seed=42)
        assert r2.seed == 42


# ---------------------------------------------------------------------------
# FineTuneRegime
# ---------------------------------------------------------------------------


class TestFineTuneRegime:
    def test_basic_construction(self):
        r = FineTuneRegime("short", epochs=5, lr=1e-3)
        assert r.name == "short"
        assert r.epochs == 5
        assert r.lr == 1e-3
        assert r.quantization_applied == "none"
        assert r.loss_fn == "kl"

    def test_qat_regime(self):
        r = FineTuneRegime(
            "qat_short",
            epochs=5,
            lr=1e-3,
            quantization_applied="ptq_dynamic",
        )
        assert r.quantization_applied == "ptq_dynamic"


# ---------------------------------------------------------------------------
# SearchConfig
# ---------------------------------------------------------------------------


class TestSearchConfig:
    def test_default_pair_count(self):
        cfg = SearchConfig.default()
        pairs = cfg.pairs()
        assert len(pairs) == 14  # 7 recipes × 2 regimes

    def test_minimal_pair_count(self):
        cfg = SearchConfig.minimal()
        pairs = cfg.pairs()
        assert len(pairs) == 9  # 3 recipes × 3 regimes

    def test_default_with_qat_pair_count(self):
        cfg = SearchConfig.default_with_qat()
        assert len(cfg.pairs()) == 21  # 7 recipes × 3 regimes (short, long, short_qat)

    def test_minimal_with_qat_pair_count(self):
        cfg = SearchConfig.minimal_with_qat()
        assert len(cfg.pairs()) == 9

    def test_minimal_with_qat_includes_short_qat_regime(self):
        cfg = SearchConfig.minimal_with_qat()
        regime_names = {reg.name for _, reg in cfg.pairs()}
        assert "short_qat" in regime_names
        assert cfg.pairs()[0][1].quantization_applied == "none"
        qat_pairs = [(a, b) for a, b in cfg.pairs() if b.name == "short_qat"]
        assert all(r.quantization_applied == "ptq_dynamic" for _, r in qat_pairs)

    def test_minimal_has_all_modes(self):
        cfg = SearchConfig.minimal()
        modes = {r.mode for r, _ in cfg.pairs()}
        assert modes == {"random", "layer", "weighted"}

    def test_max_runs_truncation(self):
        cfg = SearchConfig.default()
        cfg.max_runs = 3
        assert len(cfg.pairs()) == 3

    def test_max_runs_none_no_truncation(self):
        cfg = SearchConfig.default()
        cfg.max_runs = None
        assert len(cfg.pairs()) == 14

    def test_pairs_cartesian_order(self):
        """pairs() is recipe-major: all regimes for recipe 0 before recipe 1."""
        recipes = [CrossoverRecipe("layer"), CrossoverRecipe("weighted", alpha=0.5)]
        regimes = [
            FineTuneRegime("short", epochs=2, lr=1e-3),
            FineTuneRegime("long", epochs=5, lr=1e-4),
        ]
        cfg = SearchConfig(crossover_recipes=recipes, finetune_regimes=regimes)
        pairs = cfg.pairs()
        assert len(pairs) == 4
        # First two pairs: recipe[0] × both regimes
        assert pairs[0][0].mode == "layer"
        assert pairs[0][1].name == "short"
        assert pairs[1][0].mode == "layer"
        assert pairs[1][1].name == "long"
        # Third and fourth: recipe[1] × both regimes
        assert pairs[2][0].mode == "weighted"
        assert pairs[3][0].mode == "weighted"

    def test_empty_config(self):
        cfg = SearchConfig()
        assert cfg.pairs() == []


# ---------------------------------------------------------------------------
# ManifestEntry
# ---------------------------------------------------------------------------


class TestManifestEntry:
    def test_to_dict_keys(self):
        entry = _make_entry()
        d = entry.to_dict()
        required_keys = [
            "child_id",
            "crossover_mode",
            "crossover_alpha",
            "crossover_seed",
            "finetune_regime",
            "finetune_epochs",
            "finetune_lr",
            "finetune_quantization_applied",
            "child_pt_path",
            "eval_report_path",
            "run_config_path",
            "primary_metric",
            "child_vs_parent_a_agreement",
            "child_vs_parent_b_agreement",
            "oracle_agreement",
            "kl_divergence_a",
            "kl_divergence_b",
            "mse_a",
            "mse_b",
            "cosine_a",
            "cosine_b",
            "degenerate",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_json_roundtrip(self):
        entry = _make_entry()
        d = entry.to_dict()
        # Should serialise without errors
        serialised = json.dumps(d)
        restored = json.loads(serialised)
        assert restored["child_id"] == entry.child_id
        assert abs(restored["primary_metric"] - entry.primary_metric) < 1e-9


# ---------------------------------------------------------------------------
# _make_child_id
# ---------------------------------------------------------------------------


class TestMakeChildId:
    def test_deterministic(self):
        recipe = CrossoverRecipe("random", alpha=0.5, seed=0)
        regime = FineTuneRegime("short", epochs=5, lr=1e-3)
        id1 = _make_child_id(recipe, regime, run_idx=0)
        id2 = _make_child_id(recipe, regime, run_idx=0)
        assert id1 == id2

    def test_different_runs_differ(self):
        recipe = CrossoverRecipe("random", alpha=0.5, seed=0)
        regime = FineTuneRegime("short", epochs=5, lr=1e-3)
        id1 = _make_child_id(recipe, regime, run_idx=0)
        id2 = _make_child_id(recipe, regime, run_idx=1)
        assert id1 != id2

    def test_contains_mode_and_regime(self):
        recipe = CrossoverRecipe("weighted", alpha=0.5)
        regime = FineTuneRegime("long", epochs=20, lr=1e-4)
        cid = _make_child_id(recipe, regime, run_idx=5)
        assert "weighted" in cid
        assert "long" in cid

    def test_none_seed(self):
        recipe = CrossoverRecipe("layer", seed=None)
        regime = FineTuneRegime("short", epochs=5, lr=1e-3)
        cid = _make_child_id(recipe, regime, run_idx=0)
        assert "sNone" in cid


# ---------------------------------------------------------------------------
# build_leaderboard
# ---------------------------------------------------------------------------


class TestBuildLeaderboard:
    def _make_manifest(self) -> list[ManifestEntry]:
        return [
            _make_entry("c0", primary_metric=0.60, agree_a=0.70, agree_b=0.60),
            _make_entry("c1", primary_metric=0.80, agree_a=0.85, agree_b=0.80),
            _make_entry("c2", primary_metric=0.70, agree_a=0.75, agree_b=0.70),
        ]

    def test_sorted_descending(self):
        manifest = self._make_manifest()
        lb = build_leaderboard(manifest)
        primary_values = [row["primary_metric"] for row in lb if not row["is_parent_baseline"]]
        assert primary_values == sorted(primary_values, reverse=True)

    def test_rank_assigned(self):
        manifest = self._make_manifest()
        lb = build_leaderboard(manifest)
        child_rows = [row for row in lb if not row["is_parent_baseline"]]
        assert child_rows[0]["rank"] == 1
        assert child_rows[1]["rank"] == 2
        assert child_rows[2]["rank"] == 3

    def test_parent_baselines_appended(self):
        manifest = self._make_manifest()
        lb = build_leaderboard(manifest, parent_baseline_agreement=0.65)
        baseline_rows = [row for row in lb if row["is_parent_baseline"]]
        assert len(baseline_rows) == 2
        labels = {row["child_id"] for row in baseline_rows}
        assert labels == {"parent_a", "parent_b"}

    def test_no_parent_baseline_when_none(self):
        manifest = self._make_manifest()
        lb = build_leaderboard(manifest, parent_baseline_agreement=None)
        baseline_rows = [row for row in lb if row["is_parent_baseline"]]
        assert len(baseline_rows) == 0

    def test_parent_baseline_primary_metric(self):
        manifest = self._make_manifest()
        lb = build_leaderboard(manifest, parent_baseline_agreement=0.65)
        for row in lb:
            if row["is_parent_baseline"]:
                assert row["primary_metric"] == pytest.approx(0.65)

    def test_empty_manifest(self):
        lb = build_leaderboard([])
        assert lb == []

    def test_empty_manifest_with_baseline(self):
        lb = build_leaderboard([], parent_baseline_agreement=0.5)
        baseline_rows = [r for r in lb if r.get("is_parent_baseline")]
        assert len(baseline_rows) == 2


# ---------------------------------------------------------------------------
# generate_recommendation
# ---------------------------------------------------------------------------


class TestGenerateRecommendation:
    def _make_varied_manifest(self) -> list[ManifestEntry]:
        return [
            _make_entry("c0_random_short", 0.60, mode="random", regime="short"),
            _make_entry("c1_random_long", 0.72, mode="random", regime="long"),
            _make_entry("c2_layer_short", 0.65, mode="layer", regime="short"),
            _make_entry("c3_layer_long", 0.75, mode="layer", regime="long"),
            _make_entry("c4_weighted_short", 0.70, mode="weighted", regime="short"),
            _make_entry("c5_weighted_long", 0.80, mode="weighted", regime="long"),
        ]

    def test_non_empty(self):
        manifest = self._make_varied_manifest()
        rec = generate_recommendation(manifest)
        assert len(rec) > 0

    def test_contains_recommended_label(self):
        manifest = self._make_varied_manifest()
        rec = generate_recommendation(manifest)
        assert "recommended" in rec.lower()

    def test_mentions_best_child(self):
        manifest = self._make_varied_manifest()
        rec = generate_recommendation(manifest)
        # Best child is "c5_weighted_long" (primary=0.80)
        assert "c5_weighted_long" in rec

    def test_empty_manifest(self):
        rec = generate_recommendation([])
        assert "empty" in rec.lower()

    def test_includes_baseline(self):
        manifest = self._make_varied_manifest()
        rec = generate_recommendation(manifest, parent_baseline_agreement=0.55)
        assert "0.5500" in rec or "0.55" in rec

    def test_mentions_all_modes(self):
        manifest = self._make_varied_manifest()
        rec = generate_recommendation(manifest)
        for mode in ("random", "layer", "weighted"):
            assert mode in rec

    def test_recommended_regime_hyperparams_from_best_primary_in_regime(self):
        """Same regime label with different LR/epochs: use the best child's hyperparams."""
        low = _make_entry("c_low", 0.50, regime="tune", mode="random")
        low.finetune_lr = 1e-2
        low.finetune_epochs = 3
        high = _make_entry("c_high", 0.90, regime="tune", mode="layer")
        high.finetune_lr = 1e-4
        high.finetune_epochs = 30
        rec = generate_recommendation([low, high])
        assert "epochs=30" in rec
        assert "1e-04" in rec


# ---------------------------------------------------------------------------
# run_crossover_search — end-to-end smoke test
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestRunCrossoverSearch:
    """End-to-end tests for run_crossover_search().

    Uses tiny networks (hidden=8, 2 actions) and few epochs (2) so the test
    runs in a few seconds on CPU.  The goal is to verify runner wiring, not
    training quality.
    """

    def _tiny_config(self, max_runs: int = 2) -> SearchConfig:
        """Return a minimal SearchConfig for smoke testing (2 pairs)."""
        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=0),
            CrossoverRecipe("weighted", alpha=0.5),
        ]
        regimes = [
            FineTuneRegime("smoke", epochs=2, lr=1e-3, batch_size=8, val_fraction=0.1, seed=0)
        ]
        return SearchConfig(
            crossover_recipes=recipes,
            finetune_regimes=regimes,
            max_runs=max_runs,
        )

    def test_produces_expected_manifest_length(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
        assert len(manifest) == 2

    def test_restores_parent_training_mode_and_requires_grad(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        parent_a.train()
        parent_b.train()
        for p in parent_a.parameters():
            p.requires_grad_(True)
        for p in parent_b.parameters():
            p.requires_grad_(True)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=1), run_dir
            )

        assert parent_a.training
        assert parent_b.training
        assert all(p.requires_grad for p in parent_a.parameters())
        assert all(p.requires_grad for p in parent_b.parameters())

    def test_per_child_artifacts_exist(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
            for entry in manifest:
                assert os.path.isfile(entry.child_pt_path), (
                    f"child.pt missing: {entry.child_pt_path}"
                )
                assert os.path.isfile(entry.eval_report_path), (
                    f"eval_report.json missing: {entry.eval_report_path}"
                )
                assert os.path.isfile(entry.run_config_path), (
                    f"run_config.json missing: {entry.run_config_path}"
                )

    def test_manifest_json_written(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
            manifest_path = os.path.join(run_dir, "manifest.json")
            assert os.path.isfile(manifest_path)
            with open(manifest_path, encoding="utf-8") as fh:
                manifest_data = json.load(fh)
            assert len(manifest_data) == 2

    def test_leaderboard_files_written(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
            assert os.path.isfile(os.path.join(run_dir, "leaderboard.csv"))
            assert os.path.isfile(os.path.join(run_dir, "leaderboard.json"))
            assert os.path.isfile(os.path.join(run_dir, "recommendation.txt"))

    def test_leaderboard_sorted_by_primary_metric(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            _, leaderboard = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
            child_rows = [r for r in leaderboard if not r.get("is_parent_baseline")]
            metrics = [r["primary_metric"] for r in child_rows]
            assert metrics == sorted(metrics, reverse=True)

    def test_manifest_entry_fields_valid(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=2), run_dir
            )
            for entry in manifest:
                assert 0.0 <= entry.primary_metric <= 1.0
                assert 0.0 <= entry.child_vs_parent_a_agreement <= 1.0
                assert 0.0 <= entry.child_vs_parent_b_agreement <= 1.0
                assert entry.kl_divergence_a >= 0.0
                assert entry.kl_divergence_b >= 0.0
                assert entry.mse_a >= 0.0
                assert entry.mse_b >= 0.0
                assert isinstance(entry.degenerate, bool)

    def test_eval_report_json_schema(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=1), run_dir
            )
            with open(manifest[0].eval_report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            assert "schema_version" in report
            assert "comparisons" in report
            assert "child_vs_parent_a" in report["comparisons"]
            assert "child_vs_parent_b" in report["comparisons"]
            assert "summary" in report

    def test_run_config_json_schema(self):
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=1), run_dir
            )
            with open(manifest[0].run_config_path, encoding="utf-8") as fh:
                rc = json.load(fh)
            assert "crossover" in rc
            assert "finetune" in rc
            assert "torch_version" in rc
            assert rc["crossover"]["mode"] in ("random", "layer", "weighted")

    def test_checkpoint_loadable(self):
        """Best child checkpoint must be loadable back into a fresh BaseQNetwork."""
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, _ = run_crossover_search(
                parent_a, parent_b, states, self._tiny_config(max_runs=1), run_dir
            )
            ckpt_path = manifest[0].child_pt_path
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            fresh = _make_net()
            fresh.load_state_dict(sd)  # should not raise

    def test_reproducibility(self):
        """Same seeds + same states → identical primary_metric ranking."""
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states()
        cfg = self._tiny_config(max_runs=3)

        with tempfile.TemporaryDirectory() as run1, tempfile.TemporaryDirectory() as run2:
            manifest1, _ = run_crossover_search(parent_a, parent_b, states, cfg, run1)
            manifest2, _ = run_crossover_search(parent_a, parent_b, states, cfg, run2)

        # Build sorted rankings from both runs
        rank1 = [e.child_id.split("_", 1)[1] for e in sorted(
            manifest1, key=lambda e: e.primary_metric, reverse=True
        )]
        rank2 = [e.child_id.split("_", 1)[1] for e in sorted(
            manifest2, key=lambda e: e.primary_metric, reverse=True
        )]
        # Same crossover recipe + same seed → same child; ranking should match.
        assert rank1 == rank2

    def test_parallel_workers_matches_sequential_child_ids(self):
        """Process pool path should produce the same child_id ordering as sequential."""
        parent_a = _make_net(seed=0)
        parent_b = _make_net(seed=1)
        states = _make_states(n=60, seed=3)
        cfg = self._tiny_config(max_runs=2)
        with tempfile.TemporaryDirectory() as d_seq, tempfile.TemporaryDirectory() as d_par:
            m_seq, _ = run_crossover_search(
                parent_a, parent_b, states, cfg, d_seq, num_workers=1
            )
            m_par, _ = run_crossover_search(
                parent_a, parent_b, states, cfg, d_par, num_workers=2
            )
        assert [e.child_id for e in m_seq] == [e.child_id for e in m_par]

    def test_three_config_smoke(self):
        """Acceptance criterion: run at least 3 configs without manual checkpoint editing."""
        parent_a = _make_net(seed=7)
        parent_b = _make_net(seed=13)
        states = _make_states(n=80, seed=99)

        recipes = [
            CrossoverRecipe("random", alpha=0.5, seed=42),
            CrossoverRecipe("layer"),
            CrossoverRecipe("weighted", alpha=0.5),
        ]
        regimes = [FineTuneRegime("smoke", epochs=2, lr=1e-3, batch_size=8, seed=0)]
        cfg = SearchConfig(crossover_recipes=recipes, finetune_regimes=regimes)

        with tempfile.TemporaryDirectory() as run_dir:
            manifest, leaderboard = run_crossover_search(
                parent_a, parent_b, states, cfg, run_dir
            )

        assert len(manifest) == 3
        # Leaderboard must be sorted (primary metric descending)
        child_rows = [r for r in leaderboard if not r.get("is_parent_baseline")]
        for i in range(len(child_rows) - 1):
            assert child_rows[i]["primary_metric"] >= child_rows[i + 1]["primary_metric"]


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_search_symbols_exported(self):
        from farm.core.decision import training as pkg

        assert hasattr(pkg, "CrossoverRecipe")
        assert hasattr(pkg, "FineTuneRegime")
        assert hasattr(pkg, "SearchConfig")
        assert hasattr(pkg, "ManifestEntry")
        assert hasattr(pkg, "run_crossover_search")
        assert hasattr(pkg, "build_leaderboard")
        assert hasattr(pkg, "generate_recommendation")
        assert hasattr(pkg, "LEADERBOARD_COLUMNS")
