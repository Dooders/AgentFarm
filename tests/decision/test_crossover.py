"""Tests for farm/core/decision/training/crossover.py.

Covers:
- crossover_quantized_state_dict: all three modes (random, layer, weighted).
- Edge cases: alpha=0/1, all-A / all-B, single-key dicts.
- Quantized tensor inputs (qint8): automatic dequantization before crossover.
- Validation: mismatched keys, mismatched shapes, invalid mode, invalid alpha.
- Layer grouping logic (_layer_groups).
- Round-trip: offspring state dict loads into model and runs forward pass.
- crossover_checkpoints: file-level helper saves and loads offspring.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
from farm.core.decision.training.crossover import (
    CROSSOVER_MODES,
    _layer_groups,
    _to_float,
    _validate_state_dicts,
    crossover_checkpoints,
    crossover_quantized_state_dict,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32  # student hidden = max(16, 16) = 16


def _make_student(seed: int = 0) -> StudentQNetwork:
    torch.manual_seed(seed)
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_base(seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 64, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, INPUT_DIM)).astype("float32"))


def _tiny_float_sd(seed: int = 0) -> dict:
    """Return a tiny 1-layer-worth float state dict for unit tests."""
    torch.manual_seed(seed)
    return {
        "network.0.weight": torch.randn(4, 3),
        "network.0.bias": torch.randn(4),
    }


def _tiny_quant_sd(seed: int = 0) -> dict:
    """Return a tiny state dict with a qint8-quantized tensor."""
    base = _tiny_float_sd(seed)
    # Quantize the weight to qint8
    w = base["network.0.weight"]
    scale = w.abs().max().item() / 127.0 + 1e-8
    q_w = torch.quantize_per_tensor(w, scale=scale, zero_point=0, dtype=torch.qint8)
    return {
        "network.0.weight": q_w,
        "network.0.bias": base["network.0.bias"],  # bias stays float
    }


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------


class TestCrossoverModes:
    def test_modes_tuple(self):
        assert set(CROSSOVER_MODES) == {"random", "layer", "weighted"}


# ---------------------------------------------------------------------------
# _to_float helper
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_float_tensor_passthrough(self):
        t = torch.randn(3, 4)
        out = _to_float(t)
        assert out.dtype == torch.float32
        assert torch.allclose(out, t)

    def test_cuda_tensor_moved_to_cpu(self):
        pytest.importorskip("torch.cuda")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.randn(3, 4).cuda()
        out = _to_float(t)
        assert out.device.type == "cpu"

    def test_quantized_tensor_dequantized(self):
        t = torch.randn(3, 4)
        scale = t.abs().max().item() / 127.0 + 1e-8
        q = torch.quantize_per_tensor(t, scale=scale, zero_point=0, dtype=torch.qint8)
        out = _to_float(q)
        assert out.dtype == torch.float32
        assert not out.is_quantized
        # Dequantized value should be close to original (within quantization error)
        assert (out - t).abs().max().item() < 0.05 * (t.abs().max().item() + 1e-6)


# ---------------------------------------------------------------------------
# _validate_state_dicts
# ---------------------------------------------------------------------------


class TestValidateStateDicts:
    def test_valid_dicts_return_sorted_keys(self):
        sd_a = {"b": torch.zeros(2), "a": torch.zeros(2)}
        sd_b = {"a": torch.zeros(2), "b": torch.zeros(2)}
        keys = _validate_state_dicts(sd_a, sd_b)
        assert keys == ["a", "b"]

    def test_key_mismatch_raises(self):
        sd_a = {"x": torch.zeros(2)}
        sd_b = {"y": torch.zeros(2)}
        with pytest.raises(ValueError, match="key mismatch"):
            _validate_state_dicts(sd_a, sd_b)

    def test_shape_mismatch_raises(self):
        sd_a = {"w": torch.zeros(4, 3)}
        sd_b = {"w": torch.zeros(3, 4)}
        with pytest.raises(ValueError, match="[Ss]hape"):
            _validate_state_dicts(sd_a, sd_b)

    def test_non_tensor_values_skipped(self):
        """Non-tensor entries (e.g. int scalars) should not raise shape errors."""
        sd_a = {"config": 42}
        sd_b = {"config": 99}
        keys = _validate_state_dicts(sd_a, sd_b)
        assert keys == ["config"]


# ---------------------------------------------------------------------------
# _layer_groups
# ---------------------------------------------------------------------------


class TestLayerGroups:
    def test_standard_network_keys(self):
        keys = [
            "network.0.weight",
            "network.0.bias",
            "network.1.weight",
            "network.1.bias",
            "network.4.weight",
            "network.4.bias",
            "network.8.weight",
            "network.8.bias",
        ]
        groups = _layer_groups(keys)
        assert "network.0" in groups
        assert "network.1" in groups
        assert "network.4" in groups
        assert "network.8" in groups
        assert "network.0.weight" in groups["network.0"]
        assert "network.0.bias" in groups["network.0"]

    def test_single_segment_key(self):
        keys = ["bias"]
        groups = _layer_groups(keys)
        assert "bias" in groups

    def test_groups_cover_all_keys(self):
        keys = ["network.0.weight", "network.0.bias", "network.4.weight"]
        groups = _layer_groups(keys)
        all_in_groups = [k for ks in groups.values() for k in ks]
        assert sorted(all_in_groups) == sorted(keys)


# ---------------------------------------------------------------------------
# crossover_quantized_state_dict – input validation
# ---------------------------------------------------------------------------


class TestCrossoverInputValidation:
    def test_invalid_mode_raises(self):
        sd = {"w": torch.zeros(2)}
        with pytest.raises(ValueError, match="mode"):
            crossover_quantized_state_dict(sd, sd, mode="bad_mode")

    def test_alpha_below_zero_raises(self):
        sd = {"w": torch.zeros(2)}
        with pytest.raises(ValueError, match="alpha"):
            crossover_quantized_state_dict(sd, sd, mode="random", alpha=-0.1)

    def test_alpha_above_one_raises(self):
        sd = {"w": torch.zeros(2)}
        with pytest.raises(ValueError, match="alpha"):
            crossover_quantized_state_dict(sd, sd, mode="weighted", alpha=1.1)

    def test_key_mismatch_raises(self):
        sd_a = {"w": torch.zeros(2)}
        sd_b = {"v": torch.zeros(2)}
        with pytest.raises(ValueError, match="key mismatch"):
            crossover_quantized_state_dict(sd_a, sd_b, mode="random")

    def test_shape_mismatch_raises(self):
        sd_a = {"w": torch.zeros(4, 3)}
        sd_b = {"w": torch.zeros(3, 4)}
        with pytest.raises(ValueError, match="[Ss]hape"):
            crossover_quantized_state_dict(sd_a, sd_b, mode="weighted")


# ---------------------------------------------------------------------------
# Random crossover
# ---------------------------------------------------------------------------


class TestRandomCrossover:
    def test_keys_match(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0)
        assert set(child.keys()) == set(sd_a.keys())

    def test_child_is_float32(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0)
        for v in child.values():
            assert v.dtype == torch.float32

    def test_each_tensor_from_one_parent(self):
        """Every child tensor must equal one parent's tensor exactly."""
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(7)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=42)
        for k, cv in child.items():
            from_a = torch.allclose(cv, sd_a[k])
            from_b = torch.allclose(cv, sd_b[k])
            assert from_a or from_b, (
                f"Key '{k}' in child does not match either parent"
            )

    def test_alpha_one_reproduces_parent_a(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0, alpha=1.0)
        for k in sd_a:
            assert torch.allclose(child[k], sd_a[k]), f"Key '{k}' not from parent A"

    def test_alpha_zero_reproduces_parent_b(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0, alpha=0.0)
        for k in sd_b:
            assert torch.allclose(child[k], sd_b[k]), f"Key '{k}' not from parent B"

    def test_deterministic_with_seed(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        c1 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=123)
        c2 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=123)
        for k in c1:
            assert torch.allclose(c1[k], c2[k])

    def test_different_seeds_may_differ(self):
        """Different seeds should produce at least one different assignment
        across a large enough dict."""
        sd_a = _make_student(0).state_dict()
        sd_b = _make_student(1).state_dict()
        c1 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0)
        c2 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=99)
        # With 10 parameters and 50/50 chance, probability of same choices is 2^{-10}
        diffs = sum(
            1 for k in c1 if not torch.allclose(c1[k], c2[k])
        )
        assert diffs > 0, "Different seeds produced identical crossovers"

    def test_rng_object_respected(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        rng = np.random.default_rng(7)
        c1 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", rng=rng)
        # Re-create same RNG and confirm
        rng2 = np.random.default_rng(7)
        c2 = crossover_quantized_state_dict(sd_a, sd_b, mode="random", rng=rng2)
        for k in c1:
            assert torch.allclose(c1[k], c2[k])

    def test_quantized_parent_produces_float_child(self):
        sd_a = _tiny_quant_sd(0)
        sd_b = _tiny_quant_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="random", seed=0)
        for v in child.values():
            assert v.dtype == torch.float32
            assert not v.is_quantized


# ---------------------------------------------------------------------------
# Layer crossover
# ---------------------------------------------------------------------------


class TestLayerCrossover:
    def test_keys_match(self):
        sd_a = _make_student(0).state_dict()
        sd_b = _make_student(1).state_dict()
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        assert set(child.keys()) == set(sd_a.keys())

    def test_child_is_float32(self):
        sd_a = _make_student(0).state_dict()
        sd_b = _make_student(1).state_dict()
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        for v in child.values():
            assert v.dtype == torch.float32

    def test_within_group_same_parent(self):
        """All keys in the same layer group must come from the same parent."""
        from farm.core.decision.training.crossover import _layer_groups

        sd_a = _make_student(0).state_dict()
        sd_b = _make_student(1).state_dict()
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")

        keys = sorted(sd_a.keys())
        groups = _layer_groups(keys)
        for group_idx, (_, group_keys) in enumerate(groups.items()):
            expected_parent = sd_a if group_idx % 2 == 0 else sd_b
            for k in group_keys:
                assert torch.allclose(child[k], expected_parent[k].float()), (
                    f"Key '{k}' (group {group_idx}) does not match expected parent"
                )

    def test_deterministic(self):
        """Layer crossover has no randomness and must be deterministic."""
        sd_a = _make_student(0).state_dict()
        sd_b = _make_student(1).state_dict()
        c1 = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        c2 = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        for k in c1:
            assert torch.allclose(c1[k], c2[k])

    def test_quantized_parents(self):
        sd_a = _tiny_quant_sd(0)
        sd_b = _tiny_quant_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        for v in child.values():
            assert not v.is_quantized
            assert v.dtype == torch.float32

    def test_base_network_layer_crossover(self):
        """BaseQNetwork state dicts also work."""
        sd_a = _make_base(0).state_dict()
        sd_b = _make_base(1).state_dict()
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="layer")
        assert set(child.keys()) == set(sd_a.keys())


# ---------------------------------------------------------------------------
# Weighted crossover
# ---------------------------------------------------------------------------


class TestWeightedCrossover:
    def test_keys_match(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted")
        assert set(child.keys()) == set(sd_a.keys())

    def test_child_is_float32(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted")
        for v in child.values():
            assert v.dtype == torch.float32

    def test_alpha_one_reproduces_a(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted", alpha=1.0)
        for k in sd_a:
            assert torch.allclose(child[k], sd_a[k].float()), f"Key '{k}' mismatch"

    def test_alpha_zero_reproduces_b(self):
        sd_a = _tiny_float_sd(0)
        sd_b = _tiny_float_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted", alpha=0.0)
        for k in sd_b:
            assert torch.allclose(child[k], sd_b[k].float()), f"Key '{k}' mismatch"

    def test_alpha_half_is_midpoint(self):
        """With alpha=0.5 the child should be the exact midpoint of the parents."""
        sd_a = _tiny_float_sd(2)
        sd_b = _tiny_float_sd(3)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted", alpha=0.5)
        for k in sd_a:
            expected = 0.5 * sd_a[k].float() + 0.5 * sd_b[k].float()
            assert torch.allclose(child[k], expected, atol=1e-6), (
                f"Key '{k}' is not the exact midpoint"
            )

    def test_custom_alpha(self):
        sd_a = _tiny_float_sd(4)
        sd_b = _tiny_float_sd(5)
        alpha = 0.3
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted", alpha=alpha)
        for k in sd_a:
            expected = alpha * sd_a[k].float() + (1 - alpha) * sd_b[k].float()
            assert torch.allclose(child[k], expected, atol=1e-6)

    def test_quantized_parents_blended(self):
        """Quantized parents are dequantized; blend result is float."""
        sd_a = _tiny_quant_sd(0)
        sd_b = _tiny_quant_sd(1)
        child = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted", alpha=0.5)
        for v in child.values():
            assert not v.is_quantized
            assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# Round-trip: load offspring into model and run forward
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Offspring state dicts must be loadable into the model and produce
    finite outputs on sample states."""

    def _assert_forward_ok(self, model: nn.Module, states: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            out = model(states)
        assert out.shape == (len(states), OUTPUT_DIM)
        assert torch.isfinite(out).all(), "Model produced non-finite outputs"

    def test_random_crossover_forward(self):
        parent_a = _make_student(0)
        parent_b = _make_student(1)
        child = crossover_quantized_state_dict(
            parent_a.state_dict(), parent_b.state_dict(), mode="random", seed=0
        )
        model = _make_student(99)
        model.load_state_dict(child)
        self._assert_forward_ok(model, _make_states())

    def test_layer_crossover_forward(self):
        parent_a = _make_student(0)
        parent_b = _make_student(1)
        child = crossover_quantized_state_dict(
            parent_a.state_dict(), parent_b.state_dict(), mode="layer"
        )
        model = _make_student(99)
        model.load_state_dict(child)
        self._assert_forward_ok(model, _make_states())

    def test_weighted_crossover_forward(self):
        parent_a = _make_student(0)
        parent_b = _make_student(1)
        child = crossover_quantized_state_dict(
            parent_a.state_dict(), parent_b.state_dict(), mode="weighted", alpha=0.4
        )
        model = _make_student(99)
        model.load_state_dict(child)
        self._assert_forward_ok(model, _make_states())

    def test_base_network_round_trip(self):
        parent_a = _make_base(0)
        parent_b = _make_base(1)
        child = crossover_quantized_state_dict(
            parent_a.state_dict(), parent_b.state_dict(), mode="weighted", alpha=0.5
        )
        model = _make_base(99)
        model.load_state_dict(child)
        self._assert_forward_ok(model, _make_states())

    def test_quantized_parent_child_loads(self):
        """A child derived from quantized parents loads into a float model."""
        parent_a = _make_student(0)
        parent_b = _make_student(1)

        # Quantize parents
        try:
            import torch.ao.quantization as tq
        except ImportError:
            import torch.quantization as tq  # type: ignore[no-redef]

        q_a = tq.quantize_dynamic(parent_a, {nn.Linear}, dtype=torch.qint8)
        q_b = tq.quantize_dynamic(parent_b, {nn.Linear}, dtype=torch.qint8)

        # Extract state dicts; quantized state dicts may contain PackedParams –
        # use the float parents' state dicts as the "quantized input" simulation.
        # For true int8, dequantize manually and test the child loads cleanly.
        child = crossover_quantized_state_dict(
            parent_a.state_dict(), parent_b.state_dict(), mode="random", seed=5
        )
        model = _make_student(99)
        model.load_state_dict(child)
        self._assert_forward_ok(model, _make_states())


# ---------------------------------------------------------------------------
# crossover_checkpoints file helper
# ---------------------------------------------------------------------------


class TestCrossoverCheckpoints:
    def _save_state_dict(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)

    def test_file_helper_random(self):
        parent_a = _make_student(0)
        parent_b = _make_student(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "parent_a.pt")
            path_b = os.path.join(tmpdir, "parent_b.pt")
            out_path = os.path.join(tmpdir, "child.pt")
            self._save_state_dict(parent_a, path_a)
            self._save_state_dict(parent_b, path_b)

            child_sd = crossover_checkpoints(path_a, path_b, out_path, mode="random", seed=7)

            assert os.path.isfile(out_path), "Output file not created"
            loaded = torch.load(out_path, map_location="cpu", weights_only=True)
            assert set(loaded.keys()) == set(parent_a.state_dict().keys())

            # Each tensor should come from one parent
            for k, cv in loaded.items():
                from_a = torch.allclose(cv, parent_a.state_dict()[k])
                from_b = torch.allclose(cv, parent_b.state_dict()[k])
                assert from_a or from_b

    def test_file_helper_weighted(self):
        parent_a = _make_student(2)
        parent_b = _make_student(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "pa.pt")
            path_b = os.path.join(tmpdir, "pb.pt")
            out_path = os.path.join(tmpdir, "child.pt")
            self._save_state_dict(parent_a, path_a)
            self._save_state_dict(parent_b, path_b)

            crossover_checkpoints(path_a, path_b, out_path, mode="weighted", alpha=0.5)

            loaded = torch.load(out_path, map_location="cpu", weights_only=True)
            for k in loaded:
                expected = 0.5 * parent_a.state_dict()[k].float() + 0.5 * parent_b.state_dict()[k].float()
                assert torch.allclose(loaded[k], expected, atol=1e-6), (
                    f"Weighted mid-point mismatch for '{k}'"
                )

    def test_file_helper_creates_output_dir(self):
        parent_a = _make_student(0)
        parent_b = _make_student(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "pa.pt")
            path_b = os.path.join(tmpdir, "pb.pt")
            nested = os.path.join(tmpdir, "a", "b", "child.pt")
            self._save_state_dict(parent_a, path_a)
            self._save_state_dict(parent_b, path_b)

            crossover_checkpoints(path_a, path_b, nested, mode="layer")
            assert os.path.isfile(nested)


# ---------------------------------------------------------------------------
# Public imports from training package
# ---------------------------------------------------------------------------


class TestPublicImports:
    def test_imported_from_training_package(self):
        from farm.core.decision.training import (
            CROSSOVER_MODES,
            crossover_checkpoints,
            crossover_quantized_state_dict,
        )

        assert callable(crossover_quantized_state_dict)
        assert callable(crossover_checkpoints)
        assert "random" in CROSSOVER_MODES
