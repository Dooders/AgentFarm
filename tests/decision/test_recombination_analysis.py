"""Tests for recombination_analysis: DisagreementRecord, extract_disagreements,
worst_k_states, export_disagreements_csv, export_disagreements_json,
extract_activations, and __init__ exports.

All tests use tiny synthetic tensors / tiny BaseQNetwork models for speed.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.recombination_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    WORST_K_CRITERIA,
    DisagreementRecord,
    export_disagreements_csv,
    export_disagreements_json,
    extract_activations,
    extract_disagreements,
    worst_k_states,
    _validate_states_2d,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 6
OUTPUT_DIM = 3
HIDDEN_SIZE = 8  # tiny for speed
N_STATES = 20
SEED = 0


def _make_model(seed: int = SEED) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=HIDDEN_SIZE,
    )


def _make_states(n: int = N_STATES, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _make_records(
    n: int = N_STATES,
    *,
    include_logits: bool = False,
    k_values=None,
) -> list[DisagreementRecord]:
    pa = _make_model(0)
    pb = _make_model(1)
    child = _make_model(2)
    states = _make_states(n)
    return extract_disagreements(
        pa, pb, child, states,
        include_logits=include_logits,
        k_values=k_values or [1, 2, 3],
    )


# ---------------------------------------------------------------------------
# DisagreementRecord
# ---------------------------------------------------------------------------


class TestDisagreementRecord:
    def _make_record(self, **kwargs) -> DisagreementRecord:
        defaults = dict(
            state_index=5,
            child_action=2,
            parent_a_action=0,
            parent_b_action=2,
            agrees_with_parent_a=False,
            agrees_with_parent_b=True,
            agrees_with_any_parent=True,
            parent_a_in_child_top_k={1: False, 2: True, 3: True},
            parent_b_in_child_top_k={1: True, 2: True, 3: True},
            kl_child_vs_parent_a=0.52,
            kl_child_vs_parent_b=0.02,
            mse_child_vs_parent_a=1.23,
            mse_child_vs_parent_b=0.05,
            cosine_child_vs_parent_a=0.65,
            cosine_child_vs_parent_b=0.98,
        )
        defaults.update(kwargs)
        return DisagreementRecord(**defaults)

    def test_to_dict_has_required_keys(self):
        r = self._make_record()
        d = r.to_dict()
        for key in (
            "state_index",
            "child_action",
            "parent_a_action",
            "parent_b_action",
            "agrees_with_parent_a",
            "agrees_with_parent_b",
            "agrees_with_any_parent",
            "parent_a_in_child_top_k",
            "parent_b_in_child_top_k",
            "kl_child_vs_parent_a",
            "kl_child_vs_parent_b",
            "mse_child_vs_parent_a",
            "mse_child_vs_parent_b",
            "cosine_child_vs_parent_a",
            "cosine_child_vs_parent_b",
        ):
            assert key in d, f"Missing key: {key!r}"

    def test_to_dict_top_k_keys_are_strings(self):
        r = self._make_record()
        d = r.to_dict()
        for k in d["parent_a_in_child_top_k"]:
            assert isinstance(k, str), "top-k keys must be strings for JSON compatibility"
        for k in d["parent_b_in_child_top_k"]:
            assert isinstance(k, str)

    def test_to_dict_json_serialisable(self):
        r = self._make_record()
        json.dumps(r.to_dict())  # must not raise

    def test_to_dict_with_logits(self):
        r = self._make_record(
            child_logits=[0.1, 0.2, 0.8],
            parent_a_logits=[0.7, 0.2, 0.1],
            parent_b_logits=[0.1, 0.2, 0.8],
        )
        d = r.to_dict()
        assert d["child_logits"] == pytest.approx([0.1, 0.2, 0.8])
        assert d["parent_a_logits"] is not None

    def test_flat_csv_row_has_required_keys(self):
        r = self._make_record()
        row = r.flat_csv_row()
        for key in (
            "state_index",
            "child_action",
            "parent_a_action",
            "parent_b_action",
            "agrees_with_parent_a",
            "agrees_with_parent_b",
            "agrees_with_any_parent",
            "kl_child_vs_parent_a",
            "kl_child_vs_parent_b",
            "mse_child_vs_parent_a",
            "mse_child_vs_parent_b",
            "cosine_child_vs_parent_a",
            "cosine_child_vs_parent_b",
        ):
            assert key in row, f"Missing CSV key: {key!r}"

    def test_flat_csv_row_includes_top_k_columns(self):
        r = self._make_record()
        row = r.flat_csv_row()
        assert "parent_a_in_top_k_1" in row
        assert "parent_b_in_top_k_2" in row

    def test_flat_csv_row_respects_k_values_filter(self):
        r = self._make_record()
        row = r.flat_csv_row(k_values=[1])
        assert "parent_a_in_top_k_1" in row
        # k=2 should not be included when k_values=[1]
        assert "parent_a_in_top_k_2" not in row


# ---------------------------------------------------------------------------
# extract_disagreements
# ---------------------------------------------------------------------------


class TestExtractDisagreements:
    def test_returns_one_record_per_state(self):
        records = _make_records(N_STATES)
        assert len(records) == N_STATES

    def test_state_indices_are_consecutive(self):
        records = _make_records(N_STATES)
        assert [r.state_index for r in records] == list(range(N_STATES))

    def test_agreement_flags_consistent(self):
        """agrees_with_parent_a == (child_action == parent_a_action) per record."""
        records = _make_records(N_STATES)
        for r in records:
            assert r.agrees_with_parent_a == (r.child_action == r.parent_a_action)
            assert r.agrees_with_parent_b == (r.child_action == r.parent_b_action)
            assert r.agrees_with_any_parent == (
                r.agrees_with_parent_a or r.agrees_with_parent_b
            )

    def test_kl_non_negative(self):
        records = _make_records(N_STATES)
        for r in records:
            assert r.kl_child_vs_parent_a >= 0.0
            assert r.kl_child_vs_parent_b >= 0.0

    def test_mse_non_negative(self):
        records = _make_records(N_STATES)
        for r in records:
            assert r.mse_child_vs_parent_a >= 0.0
            assert r.mse_child_vs_parent_b >= 0.0

    def test_cosine_in_range(self):
        records = _make_records(N_STATES)
        for r in records:
            assert -1.0 <= r.cosine_child_vs_parent_a <= 1.0
            assert -1.0 <= r.cosine_child_vs_parent_b <= 1.0

    def test_top_k_keys_present(self):
        records = _make_records(N_STATES, k_values=[1, 2, 3])
        for r in records:
            assert set(r.parent_a_in_child_top_k.keys()) == {1, 2, 3}
            assert set(r.parent_b_in_child_top_k.keys()) == {1, 2, 3}

    def test_top_1_matches_agreement(self):
        """parent_a_in_child_top_k[1] must equal agrees_with_parent_a for every state."""
        records = _make_records(N_STATES, k_values=[1, 2])
        for r in records:
            assert r.parent_a_in_child_top_k[1] == r.agrees_with_parent_a
            assert r.parent_b_in_child_top_k[1] == r.agrees_with_parent_b

    def test_top_k_non_decreasing(self):
        """If parent's action is in child's top-k, it's also in child's top-(k+1)."""
        records = _make_records(N_STATES, k_values=[1, 2, 3])
        for r in records:
            for k1, k2 in [(1, 2), (2, 3)]:
                # top-k non-decreasing: if k1 includes it, k2 must too
                if r.parent_a_in_child_top_k[k1]:
                    assert r.parent_a_in_child_top_k[k2]
                if r.parent_b_in_child_top_k[k1]:
                    assert r.parent_b_in_child_top_k[k2]

    def test_logits_none_by_default(self):
        records = _make_records(N_STATES, include_logits=False)
        for r in records:
            assert r.child_logits is None
            assert r.parent_a_logits is None
            assert r.parent_b_logits is None

    def test_logits_present_when_requested(self):
        records = _make_records(N_STATES, include_logits=True)
        for r in records:
            assert r.child_logits is not None
            assert len(r.child_logits) == OUTPUT_DIM
            assert r.parent_a_logits is not None
            assert r.parent_b_logits is not None

    def test_identical_child_and_parent_a_full_agreement(self):
        pa = _make_model(seed=5)
        pb = _make_model(seed=6)
        child = _make_model(seed=5)  # identical to pa
        states = _make_states(30)
        records = extract_disagreements(pa, pb, child, states)
        assert all(r.agrees_with_parent_a for r in records)
        assert all(r.kl_child_vs_parent_a == pytest.approx(0.0, abs=1e-5) for r in records)
        assert all(r.mse_child_vs_parent_a == pytest.approx(0.0, abs=1e-5) for r in records)
        assert all(
            r.cosine_child_vs_parent_a == pytest.approx(1.0, abs=1e-5) for r in records
        )

    def test_batch_size_chunking_yields_same_result(self):
        pa = _make_model(0)
        pb = _make_model(1)
        child = _make_model(2)
        states = _make_states(25)
        full = extract_disagreements(pa, pb, child, states, batch_size=25)
        chunked = extract_disagreements(pa, pb, child, states, batch_size=7)
        assert len(full) == len(chunked)
        for r_f, r_c in zip(full, chunked):
            assert r_f.state_index == r_c.state_index
            assert r_f.child_action == r_c.child_action
            assert r_f.kl_child_vs_parent_a == pytest.approx(
                r_c.kl_child_vs_parent_a, rel=1e-5, abs=1e-7
            )

    def test_raises_on_1d_states(self):
        pa, pb, child = _make_model(0), _make_model(1), _make_model(2)
        with pytest.raises(ValueError, match="2-D"):
            extract_disagreements(pa, pb, child, np.ones(INPUT_DIM, dtype="float32"))

    def test_raises_on_empty_states(self):
        pa, pb, child = _make_model(0), _make_model(1), _make_model(2)
        with pytest.raises(ValueError, match="non-empty"):
            extract_disagreements(
                pa, pb, child, np.empty((0, INPUT_DIM), dtype="float32")
            )


# ---------------------------------------------------------------------------
# worst_k_states
# ---------------------------------------------------------------------------


class TestWorstKStates:
    def test_returns_at_most_k_records(self):
        records = _make_records(N_STATES)
        worst = worst_k_states(records, k=5)
        assert len(worst) == 5

    def test_returns_all_when_k_exceeds_n(self):
        records = _make_records(5)
        worst = worst_k_states(records, k=100)
        assert len(worst) == 5

    def test_sorted_descending_by_max_kl(self):
        records = _make_records(N_STATES)
        worst = worst_k_states(records, k=N_STATES, criterion="max_kl")
        scores = [max(r.kl_child_vs_parent_a, r.kl_child_vs_parent_b) for r in worst]
        assert scores == sorted(scores, reverse=True)

    def test_sorted_descending_by_kl_parent_a(self):
        records = _make_records(N_STATES)
        worst = worst_k_states(records, k=N_STATES, criterion="kl_parent_a")
        scores = [r.kl_child_vs_parent_a for r in worst]
        assert scores == sorted(scores, reverse=True)

    def test_sorted_descending_by_max_mse(self):
        records = _make_records(N_STATES)
        worst = worst_k_states(records, k=N_STATES, criterion="max_mse")
        scores = [max(r.mse_child_vs_parent_a, r.mse_child_vs_parent_b) for r in worst]
        assert scores == sorted(scores, reverse=True)

    def test_all_criteria_accepted(self):
        records = _make_records(N_STATES)
        for crit in WORST_K_CRITERIA:
            result = worst_k_states(records, k=3, criterion=crit)
            assert len(result) <= 3

    def test_raises_on_invalid_criterion(self):
        records = _make_records(N_STATES)
        with pytest.raises(ValueError, match="criterion"):
            worst_k_states(records, k=3, criterion="not_a_criterion")

    def test_raises_on_zero_k(self):
        records = _make_records(N_STATES)
        with pytest.raises(ValueError, match="positive"):
            worst_k_states(records, k=0)

    def test_raises_on_negative_k(self):
        records = _make_records(N_STATES)
        with pytest.raises(ValueError, match="positive"):
            worst_k_states(records, k=-1)


# ---------------------------------------------------------------------------
# export_disagreements_csv
# ---------------------------------------------------------------------------


class TestExportDisagreementsCsv:
    def test_creates_file(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            export_disagreements_csv(records, path)
            assert os.path.isfile(path)

    def test_has_correct_row_count(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            export_disagreements_csv(records, path)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == N_STATES

    def test_has_required_columns(self):
        records = _make_records(N_STATES, k_values=[1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            export_disagreements_csv(records, path, k_values=[1, 2])
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
            for col in (
                "state_index",
                "child_action",
                "parent_a_action",
                "parent_b_action",
                "agrees_with_parent_a",
                "agrees_with_parent_b",
                "agrees_with_any_parent",
                "kl_child_vs_parent_a",
                "kl_child_vs_parent_b",
                "mse_child_vs_parent_a",
                "mse_child_vs_parent_b",
                "cosine_child_vs_parent_a",
                "cosine_child_vs_parent_b",
                "parent_a_in_top_k_1",
                "parent_b_in_top_k_2",
            ):
                assert col in fieldnames, f"Missing column: {col!r}"

    def test_state_index_values_correct(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            export_disagreements_csv(records, path)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            indices = [int(row["state_index"]) for row in rows]
            assert indices == list(range(N_STATES))

    def test_empty_records_creates_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.csv")
            export_disagreements_csv([], path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) == 0


# ---------------------------------------------------------------------------
# export_disagreements_json
# ---------------------------------------------------------------------------


class TestExportDisagreementsJson:
    def test_creates_file(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            assert os.path.isfile(path)

    def test_valid_json(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert isinstance(doc, dict)

    def test_top_level_keys(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        for key in (
            "schema_version",
            "n_states",
            "n_disagree_with_parent_a",
            "n_disagree_with_parent_b",
            "n_disagree_with_both",
            "records",
            "metadata",
        ):
            assert key in doc, f"Missing JSON key: {key!r}"

    def test_schema_version(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert doc["schema_version"] == ANALYSIS_SCHEMA_VERSION

    def test_n_states_matches(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert doc["n_states"] == N_STATES
        assert len(doc["records"]) == N_STATES

    def test_disagree_count_consistent(self):
        records = _make_records(N_STATES)
        expected_a = sum(not r.agrees_with_parent_a for r in records)
        expected_b = sum(not r.agrees_with_parent_b for r in records)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert doc["n_disagree_with_parent_a"] == expected_a
        assert doc["n_disagree_with_parent_b"] == expected_b

    def test_extra_metadata_included(self):
        records = _make_records(N_STATES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path, extra_metadata={"foo": "bar"})
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert doc["metadata"]["foo"] == "bar"

    def test_logits_in_json_when_requested(self):
        records = _make_records(N_STATES, include_logits=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        first = doc["records"][0]
        assert first["child_logits"] is not None
        assert len(first["child_logits"]) == OUTPUT_DIM

    def test_logits_null_when_not_requested(self):
        records = _make_records(N_STATES, include_logits=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            export_disagreements_json(records, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        assert doc["records"][0]["child_logits"] is None


# ---------------------------------------------------------------------------
# extract_activations
# ---------------------------------------------------------------------------


class TestExtractActivations:
    def test_returns_numpy_array(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        assert isinstance(acts, np.ndarray)

    def test_shape_first_dim_matches_states(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        assert acts.shape[0] == N_STATES

    def test_dtype_is_float32(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        assert acts.dtype == np.float32

    def test_activation_dim_matches_hidden_size(self):
        """For layer_index=4 (first ReLU) in BaseQNetwork, width = hidden_size."""
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        assert acts.shape[1] == HIDDEN_SIZE

    def test_max_states_caps_output(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4, max_states=7)
        assert acts.shape[0] == 7

    def test_batch_size_does_not_affect_result(self):
        """Activation values should be identical regardless of batch_size."""
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts_full = extract_activations(model, states, layer_index=4, batch_size=N_STATES)
        acts_chunked = extract_activations(model, states, layer_index=4, batch_size=5)
        np.testing.assert_allclose(acts_full, acts_chunked, rtol=1e-5, atol=1e-7)

    def test_first_linear_layer_index(self):
        """layer_index=2 hooks the first nn.Linear in BaseQNetwork.network."""
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=2)
        # First Linear output before LayerNorm has shape (N, hidden_size)
        assert acts.shape == (N_STATES, HIDDEN_SIZE)

    def test_raises_on_out_of_range_layer_index(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        n_modules = len(list(model.modules()))
        with pytest.raises(ValueError, match="out of range"):
            extract_activations(model, states, layer_index=n_modules + 10)

    def test_raises_on_invalid_states(self):
        model = _make_model(0)
        with pytest.raises(ValueError, match="2-D"):
            extract_activations(model, np.ones(INPUT_DIM, dtype="float32"), layer_index=4)

    def test_activations_relu_non_negative(self):
        """Post-ReLU activations (layer_index=4) should be ≥ 0."""
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        assert np.all(acts >= -1e-7), "Post-ReLU activations must be non-negative"

    def test_second_hidden_relu_layer(self):
        """layer_index=8 is the second ReLU in BaseQNetwork.network."""
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=8)
        assert acts.shape == (N_STATES, HIDDEN_SIZE)
        assert np.all(acts >= -1e-7)

    def test_can_save_as_npy(self):
        model = _make_model(0)
        states = _make_states(N_STATES)
        acts = extract_activations(model, states, layer_index=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "acts.npy")
            np.save(path, acts)
            loaded = np.load(path)
        np.testing.assert_array_equal(acts, loaded)


# ---------------------------------------------------------------------------
# _validate_states_2d helper
# ---------------------------------------------------------------------------


class TestValidateStates2d:
    def test_valid_2d_array(self):
        arr = np.ones((10, INPUT_DIM), dtype="float32")
        result = _validate_states_2d(arr)
        assert result.shape == (10, INPUT_DIM)

    def test_converts_float64_to_float32(self):
        arr = np.ones((5, INPUT_DIM), dtype="float64")
        result = _validate_states_2d(arr)
        assert result.dtype == np.float32

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            _validate_states_2d(np.ones(INPUT_DIM, dtype="float32"))

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_states_2d(np.empty((0, INPUT_DIM), dtype="float32"))


# ---------------------------------------------------------------------------
# Module-level __init__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_imports_from_training_init(self):
        from farm.core.decision.training import (  # noqa: F401
            ANALYSIS_SCHEMA_VERSION,
            DisagreementRecord,
            WORST_K_CRITERIA,
            export_disagreements_csv,
            export_disagreements_json,
            extract_activations,
            extract_disagreements,
            worst_k_states,
        )
        assert isinstance(ANALYSIS_SCHEMA_VERSION, str)
        assert isinstance(WORST_K_CRITERIA, frozenset)
        assert issubclass(DisagreementRecord, object)

    def test_analysis_schema_version_is_string(self):
        assert isinstance(ANALYSIS_SCHEMA_VERSION, str)
        assert len(ANALYSIS_SCHEMA_VERSION) > 0

    def test_worst_k_criteria_non_empty(self):
        assert len(WORST_K_CRITERIA) > 0
