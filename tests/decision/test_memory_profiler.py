"""Tests for farm/core/decision/training/memory_profiler.py.

Covers:
- profile_peak_ram context manager: tracemalloc tracking, RSS snapshots, elapsed time.
- profile_model_stage: state_dict_bytes, checkpoint_bytes, peak_ram population.
- PipelineMemoryReport: to_dict schema and summary aggregation.
- profile_model_stage with a dynamically-quantized model.
- Edge cases: batch_size larger than states, zero warmup.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import StudentQNetwork
from farm.core.decision.training.memory_profiler import (
    PeakRAMSample,
    PipelineMemoryReport,
    StageMemoryProfile,
    profile_model_stage,
    profile_peak_ram,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32  # → student hidden = max(16, 16) = 16


def _make_student(seed: int = 0) -> StudentQNetwork:
    torch.manual_seed(seed)
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


# ---------------------------------------------------------------------------
# profile_peak_ram context manager
# ---------------------------------------------------------------------------


class TestProfilePeakRam:
    def test_yields_peak_ram_sample(self):
        with profile_peak_ram() as sample:
            _unused = [0] * 1000  # allocate something
        assert isinstance(sample, PeakRAMSample)

    def test_elapsed_seconds_positive(self):
        with profile_peak_ram() as sample:
            pass
        assert sample.elapsed_seconds >= 0.0

    def test_tracemalloc_peak_bytes_non_negative(self):
        with profile_peak_ram() as sample:
            _buf = bytearray(1024 * 1024)  # 1 MiB Python allocation
        assert sample.tracemalloc_peak_bytes is not None
        assert sample.tracemalloc_peak_bytes >= 0

    def test_tracemalloc_peak_bytes_captures_large_allocation(self):
        # Allocate ≥ 512 KiB inside the block; tracemalloc peak must be > 0.
        with profile_peak_ram() as sample:
            _buf = bytearray(512 * 1024)
        assert sample.tracemalloc_peak_bytes is not None
        assert sample.tracemalloc_peak_bytes > 0

    def test_rss_fields_populated_or_none(self):
        with profile_peak_ram() as sample:
            pass
        # Both are either both None (psutil absent) or both int
        if sample.rss_bytes_before is not None:
            assert isinstance(sample.rss_bytes_before, int)
            assert isinstance(sample.rss_bytes_after, int)

    def test_rss_delta_consistent(self):
        with profile_peak_ram() as sample:
            pass
        if sample.rss_bytes_before is not None and sample.rss_bytes_after is not None:
            assert sample.rss_delta_bytes == sample.rss_bytes_after - sample.rss_bytes_before

    def test_to_dict_keys(self):
        with profile_peak_ram() as sample:
            pass
        d = sample.to_dict()
        assert "tracemalloc_peak_bytes" in d
        assert "rss_bytes_before" in d
        assert "rss_bytes_after" in d
        assert "rss_delta_bytes" in d
        assert "process_peak_rss_ru" in d
        assert "cuda_peak_bytes" in d
        assert "elapsed_seconds" in d

    def test_cuda_peak_none_on_cpu(self):
        with profile_peak_ram(device=torch.device("cpu")) as sample:
            pass
        assert sample.cuda_peak_bytes is None

    def test_json_serialisable(self):
        with profile_peak_ram() as sample:
            pass
        # Should not raise
        json.dumps(sample.to_dict())


# ---------------------------------------------------------------------------
# profile_model_stage
# ---------------------------------------------------------------------------


class TestProfileModelStage:
    def test_returns_stage_profile(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        assert isinstance(profile, StageMemoryProfile)
        assert profile.stage == "student"

    def test_state_dict_bytes_positive(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        assert profile.state_dict_bytes > 0

    def test_state_dict_bytes_matches_manual_count(self):
        model = _make_student()
        states = _make_states()
        expected = sum(t.nelement() * t.element_size() for t in model.state_dict().values())
        profile = profile_model_stage("student", model, states)
        assert profile.state_dict_bytes == expected

    def test_checkpoint_bytes_none_when_no_path(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        assert profile.checkpoint_bytes is None

    def test_checkpoint_bytes_from_file(self):
        model = _make_student()
        states = _make_states()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "student.pt")
            torch.save(model.state_dict(), ckpt_path)
            profile = profile_model_stage("student", model, states, checkpoint_path=ckpt_path)
        assert profile.checkpoint_bytes is not None
        assert profile.checkpoint_bytes > 0

    def test_batch_size_clamped_to_state_count(self):
        model = _make_student()
        states = _make_states(n=10)
        profile = profile_model_stage("student", model, states, batch_size=100)
        assert profile.batch_size == 10

    def test_batch_size_respected(self):
        model = _make_student()
        states = _make_states(n=200)
        profile = profile_model_stage("student", model, states, batch_size=32)
        assert profile.batch_size == 32

    def test_n_forward_passes_stored(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states, n_forward_passes=7)
        assert profile.n_forward_passes == 7

    def test_device_cpu(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states, device=torch.device("cpu"))
        assert profile.device == "cpu"

    def test_peak_ram_populated(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        assert isinstance(profile.peak_ram, PeakRAMSample)
        assert profile.peak_ram.elapsed_seconds >= 0.0
        assert profile.peak_ram.tracemalloc_peak_bytes is not None

    def test_zero_warmup_passes(self):
        model = _make_student()
        states = _make_states()
        # Should not raise with n_warmup=0
        profile = profile_model_stage("student", model, states, n_warmup=0)
        assert profile.stage == "student"

    def test_notes_populated(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        assert isinstance(profile.notes, list)

    def test_to_dict_schema(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        d = profile.to_dict()
        assert "stage" in d
        assert "batch_size" in d
        assert "state_dict_bytes" in d
        assert "checkpoint_bytes" in d
        assert "peak_ram" in d
        assert "device" in d
        assert "notes" in d

    def test_json_serialisable(self):
        model = _make_student()
        states = _make_states()
        profile = profile_model_stage("student", model, states)
        json.dumps(profile.to_dict())

    def test_different_stages_same_architecture(self):
        """Parent and student profiles on the same architecture should both succeed."""
        model_a = _make_student(seed=0)
        model_b = _make_student(seed=1)
        states = _make_states()
        p_a = profile_model_stage("parent", model_a, states)
        p_b = profile_model_stage("student", model_b, states)
        assert p_a.stage == "parent"
        assert p_b.stage == "student"
        # Both should have positive state-dict bytes
        assert p_a.state_dict_bytes > 0
        assert p_b.state_dict_bytes > 0


# ---------------------------------------------------------------------------
# Quantized model stage
# ---------------------------------------------------------------------------


class TestProfileQuantizedStage:
    """Profile a dynamically-quantized model to ensure it works end-to-end."""

    @pytest.fixture
    def quantized_model(self):
        from farm.core.decision.training.quantize_ptq import (
            PostTrainingQuantizer,
            QuantizationConfig,
        )

        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, _ = quantizer.quantize(student)
        return q_model

    def test_quantized_stage_profile(self, quantized_model):
        states = _make_states()
        profile = profile_model_stage("quantized", quantized_model, states)
        assert profile.stage == "quantized"
        assert profile.state_dict_bytes > 0
        assert profile.peak_ram.tracemalloc_peak_bytes is not None

    def test_quantized_smaller_state_dict_than_float(self, quantized_model):
        float_model = _make_student()
        states = _make_states()
        fp = profile_model_stage("float", float_model, states)
        qp = profile_model_stage("quantized", quantized_model, states)
        # int8 weight packing → quantized state dict ≤ float state dict
        assert qp.state_dict_bytes <= fp.state_dict_bytes


# ---------------------------------------------------------------------------
# PipelineMemoryReport
# ---------------------------------------------------------------------------


class TestPipelineMemoryReport:
    def _make_profiles(self) -> list[StageMemoryProfile]:
        states = _make_states()
        profiles = []
        for i, stage in enumerate(["parent", "student"]):
            model = _make_student(seed=i)
            profiles.append(profile_model_stage(stage, model, states))
        return profiles

    def test_empty_report(self):
        report = PipelineMemoryReport(stages=[])
        d = report.to_dict()
        assert d["stages"] == []
        assert d["summary"] == {}

    def test_to_dict_has_stages_and_summary(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        d = report.to_dict()
        assert "stages" in d
        assert "summary" in d

    def test_summary_contains_all_stage_keys(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        summary = report.to_dict()["summary"]
        for p in profiles:
            assert p.stage in summary

    def test_summary_values(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        summary = report.to_dict()["summary"]
        for p in profiles:
            entry = summary[p.stage]
            assert "state_dict_bytes" in entry
            assert "checkpoint_bytes" in entry
            assert "tracemalloc_peak_bytes" in entry
            assert "rss_delta_bytes" in entry

    def test_json_serialisable(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        json.dumps(report.to_dict())

    def test_stages_list_length(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        d = report.to_dict()
        assert len(d["stages"]) == len(profiles)

    def test_summary_state_dict_bytes_matches_profile(self):
        profiles = self._make_profiles()
        report = PipelineMemoryReport(stages=profiles)
        d = report.to_dict()
        for p in profiles:
            assert d["summary"][p.stage]["state_dict_bytes"] == p.state_dict_bytes
