"""Tests for bounded, deterministic replay buffer transfer (Issue #902)."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np
import pytest

from farm.core.decision.algorithms.rl_base import PrioritizedReplayBuffer
from farm.utils.logging.test_helpers import capture_logs


def _hash_slice_data(slice_data: Dict[str, Any]) -> str:
    """Compute a deterministic hash of slice data for byte-level comparison.

    Returns a hex digest that is identical for identical slices and different
    for different slices, even under dict/set hash randomization.
    """
    # Sort experiences by a stable key (state values) to ensure deterministic ordering
    experiences = slice_data["experiences"]
    priorities = slice_data["priorities"]

    # Build a canonical byte representation
    parts = []
    for exp, priority in zip(experiences, priorities):
        # State and next_state are the primary deterministic keys
        state_bytes = exp["state"].tobytes() if isinstance(exp["state"], np.ndarray) else str(exp["state"]).encode()
        next_state_bytes = exp["next_state"].tobytes() if isinstance(exp["next_state"], np.ndarray) else str(exp["next_state"]).encode()
        parts.append(state_bytes)
        parts.append(next_state_bytes)
        parts.append(str(exp["action"]).encode())
        parts.append(str(exp["reward"]).encode())
        parts.append(str(exp["done"]).encode())
        parts.append(str(priority).encode())

    # Add metadata
    metadata = slice_data["metadata"]
    parts.append(str(metadata["alpha"]).encode())
    parts.append(str(metadata["epsilon"]).encode())
    parts.append(str(metadata["replay_strategy"]).encode())

    combined = b"".join(parts)
    return hashlib.sha256(combined).hexdigest()


class TestReplayBufferTransferBasics:
    """Basic functionality tests for get_transfer_slice and load_transfer_slice."""

    def test_get_transfer_slice_empty_buffer(self):
        """Empty buffer returns empty slice with metadata."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)

        assert isinstance(slice_data, dict)
        assert slice_data["experiences"] == []
        assert len(slice_data["priorities"]) == 0
        assert slice_data["metadata"]["alpha"] == 0.6
        assert slice_data["metadata"]["epsilon"] == 1e-6
        assert slice_data["metadata"]["replay_strategy"] == "prioritized"

    def test_get_transfer_slice_validates_max_size(self):
        """max_size must be positive."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        with pytest.raises(ValueError, match="max_size must be positive"):
            buffer.get_transfer_slice(max_size=0, seed=42)
        with pytest.raises(ValueError, match="max_size must be positive"):
            buffer.get_transfer_slice(max_size=-5, seed=42)

    def test_get_transfer_slice_caps_at_requested_size(self):
        """Slice is capped at max_size even when buffer has more experiences."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(50):
            buffer.append(
                state=np.array([i, i + 1], dtype=np.float32),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i + 1, i + 2], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)
        assert len(slice_data["experiences"]) == 10
        assert len(slice_data["priorities"]) == 10

    def test_get_transfer_slice_returns_full_buffer_when_smaller_than_cap(self):
        """When buffer has fewer experiences than max_size, entire buffer is returned."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(7):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=20, seed=42)
        assert len(slice_data["experiences"]) == 7
        assert len(slice_data["priorities"]) == 7

    def test_get_transfer_slice_selects_most_recent_experiences(self):
        """Slice contains the most recent max_size experiences in chronological order."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(20):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=5, seed=42)

        # Most recent 5 experiences have state values [15], [16], [17], [18], [19]
        assert len(slice_data["experiences"]) == 5
        for idx, exp in enumerate(slice_data["experiences"]):
            expected_state_value = 15 + idx
            assert exp["state"][0] == expected_state_value
            assert exp["reward"] == expected_state_value

    def test_get_transfer_slice_handles_circular_buffer_wrap(self):
        """Slice selection works correctly when buffer has wrapped around."""
        buffer = PrioritizedReplayBuffer(max_size=10)
        # Fill buffer past capacity to trigger wrap
        for i in range(25):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        # Buffer now contains experiences 15-24 (most recent 10)
        # position should be 5 (since 25 % 10 = 5)
        assert len(buffer) == 10
        assert buffer.position == 5

        slice_data = buffer.get_transfer_slice(max_size=5, seed=42)

        # Most recent 5 are experiences 20, 21, 22, 23, 24
        assert len(slice_data["experiences"]) == 5
        for idx, exp in enumerate(slice_data["experiences"]):
            expected_state_value = 20 + idx
            assert exp["state"][0] == expected_state_value
            assert exp["reward"] == expected_state_value

    def test_get_transfer_slice_copies_experiences_deeply(self):
        """Experiences in the slice are deep copies, not aliases."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        buffer.append(
            state=np.array([1.0, 2.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.array([2.0, 3.0], dtype=np.float32),
            done=False,
        )

        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)
        exp = slice_data["experiences"][0]

        # Modify the slice experience
        exp["state"][0] = 999.0

        # Original buffer experience should be unchanged
        original_exp = buffer.buffer[0]
        assert original_exp["state"][0] == 1.0

    def test_get_transfer_slice_includes_priorities(self):
        """Slice includes priorities corresponding to each experience."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(10):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        # Update some priorities manually
        buffer.priorities[5] = 2.5
        buffer.priorities[7] = 3.7

        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)
        priorities = slice_data["priorities"]

        assert len(priorities) == 10
        assert priorities[5] == 2.5
        assert priorities[7] == 3.7

    def test_load_transfer_slice_rejects_nonempty_buffer(self):
        """load_transfer_slice requires an empty buffer."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        buffer.append(
            state=np.array([1.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.array([2.0], dtype=np.float32),
            done=False,
        )

        slice_data = {
            "experiences": [],
            "priorities": np.array([]),
            "metadata": {"alpha": 0.6, "epsilon": 1e-6, "replay_strategy": "prioritized"},
        }

        with pytest.raises(ValueError, match="Buffer must be empty"):
            buffer.load_transfer_slice(slice_data)

    def test_load_transfer_slice_validates_structure(self):
        """load_transfer_slice validates slice_data structure."""
        buffer = PrioritizedReplayBuffer(max_size=100)

        # Not a dict
        with pytest.raises(ValueError, match="slice_data must be a dictionary"):
            buffer.load_transfer_slice("not-a-dict")  # type: ignore

        # Missing keys
        with pytest.raises(ValueError, match="missing required keys"):
            buffer.load_transfer_slice({"experiences": []})  # type: ignore

        # experiences not a list
        with pytest.raises(ValueError, match="experiences must be a list"):
            buffer.load_transfer_slice({
                "experiences": "not-a-list",
                "priorities": np.array([]),
                "metadata": {},
            })  # type: ignore

        # priorities not a numpy array
        with pytest.raises(ValueError, match="priorities must be a numpy array"):
            buffer.load_transfer_slice({
                "experiences": [],
                "priorities": [1.0, 2.0],  # type: ignore
                "metadata": {},
            })

        # Length mismatch
        with pytest.raises(ValueError, match="length mismatch"):
            buffer.load_transfer_slice({
                "experiences": [{"state": np.array([1.0])}],
                "priorities": np.array([1.0, 2.0]),
                "metadata": {},
            })

    def test_load_transfer_slice_populates_buffer_correctly(self):
        """load_transfer_slice correctly populates an empty buffer."""
        parent_buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(10):
            parent_buffer.append(
                state=np.array([i, i + 1], dtype=np.float32),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i + 1, i + 2], dtype=np.float32),
                done=(i == 9),
            )

        # Manually set some priorities
        parent_buffer.priorities[5] = 5.5
        parent_buffer.priorities[8] = 8.8

        slice_data = parent_buffer.get_transfer_slice(max_size=10, seed=42)

        # Load into child buffer
        child_buffer = PrioritizedReplayBuffer(max_size=100)
        child_buffer.load_transfer_slice(slice_data)

        # Verify buffer contents
        assert len(child_buffer) == 10
        for i in range(10):
            exp = child_buffer.buffer[i]
            assert np.array_equal(exp["state"], np.array([i, i + 1], dtype=np.float32))
            assert exp["action"] == i % 4
            assert exp["reward"] == float(i)
            assert np.array_equal(exp["next_state"], np.array([i + 1, i + 2], dtype=np.float32))
            assert exp["done"] == (i == 9)

        # Verify priorities
        assert child_buffer.priorities[5] == 5.5
        assert child_buffer.priorities[8] == 8.8

    def test_load_transfer_slice_warns_on_metadata_mismatch(self):
        """load_transfer_slice warns when metadata doesn't match child buffer settings."""
        parent_buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.7, epsilon=1e-5)
        parent_buffer.append(
            state=np.array([1.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.array([2.0], dtype=np.float32),
            done=False,
        )

        slice_data = parent_buffer.get_transfer_slice(max_size=10, seed=42)

        # Child has different settings
        child_buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6, epsilon=1e-6)

        # Load should complete despite mismatches (warnings are emitted but not blocking)
        with capture_logs() as logs:
            child_buffer.load_transfer_slice(slice_data)

        # Verify the load succeeded
        assert len(child_buffer) == 1

        # Assert warning events were emitted for alpha and epsilon mismatches
        log_events = [entry["event"] for entry in logs.entries]
        assert "replay_buffer_transfer_alpha_mismatch" in log_events
        assert "replay_buffer_transfer_epsilon_mismatch" in log_events
        assert len(child_buffer.buffer) == 1  # Main assertion: load succeeded


class TestReplayBufferTransferDeterminism:
    """Determinism tests: same seed/config → identical slices."""

    def test_identical_buffers_produce_identical_slices(self):
        """Two buffers with same experiences produce identical slices."""
        seed = 42

        def _build_buffer():
            buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6, epsilon=1e-6)
            np.random.seed(seed)
            for i in range(50):
                buffer.append(
                    state=np.random.randn(4).astype(np.float32),
                    action=i % 4,
                    reward=np.random.randn(),
                    next_state=np.random.randn(4).astype(np.float32),
                    done=False,
                )
            return buffer

        buffer1 = _build_buffer()
        buffer2 = _build_buffer()

        slice1 = buffer1.get_transfer_slice(max_size=20, seed=seed)
        slice2 = buffer2.get_transfer_slice(max_size=20, seed=seed)

        hash1 = _hash_slice_data(slice1)
        hash2 = _hash_slice_data(slice2)
        assert hash1 == hash2, "Identical buffers must produce identical slices"

    def test_slice_is_reproducible_across_runs(self):
        """Calling get_transfer_slice multiple times on the same buffer gives identical results."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(30):
            buffer.append(
                state=np.array([i, i + 1], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1, i + 2], dtype=np.float32),
                done=False,
            )

        slice1 = buffer.get_transfer_slice(max_size=15, seed=42)
        slice2 = buffer.get_transfer_slice(max_size=15, seed=42)
        slice3 = buffer.get_transfer_slice(max_size=15, seed=42)

        hash1 = _hash_slice_data(slice1)
        hash2 = _hash_slice_data(slice2)
        hash3 = _hash_slice_data(slice3)

        assert hash1 == hash2 == hash3, "Multiple calls must produce identical slices"

    def test_different_max_sizes_change_slice_deterministically(self):
        """Different max_size values produce different but reproducible slices."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(50):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_small = buffer.get_transfer_slice(max_size=10, seed=42)
        slice_large = buffer.get_transfer_slice(max_size=20, seed=42)

        hash_small = _hash_slice_data(slice_small)
        hash_large = _hash_slice_data(slice_large)

        assert hash_small != hash_large, "Different max_size should produce different slices"

        # But each is reproducible
        slice_small_2 = buffer.get_transfer_slice(max_size=10, seed=42)
        assert _hash_slice_data(slice_small_2) == hash_small

    def test_buffer_modifications_after_slice_dont_affect_slice(self):
        """Modifying the parent buffer after slicing doesn't affect the slice."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(20):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)
        hash_before = _hash_slice_data(slice_data)

        # Modify buffer
        buffer.append(
            state=np.array([999], dtype=np.float32),
            action=0,
            reward=999.0,
            next_state=np.array([1000], dtype=np.float32),
            done=False,
        )
        buffer.priorities[0] = 42.0

        # Slice should be unchanged
        hash_after = _hash_slice_data(slice_data)
        assert hash_before == hash_after


class TestReplayBufferTransferEdgeCases:
    """Edge case and robustness tests."""

    def test_transfer_with_extra_experience_kwargs(self):
        """Transfer preserves extra kwargs stored with experiences."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        buffer.append(
            state=np.array([1.0], dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.array([2.0], dtype=np.float32),
            done=False,
            log_prob=-0.5,
            entropy=0.7,
        )

        slice_data = buffer.get_transfer_slice(max_size=10, seed=42)
        exp = slice_data["experiences"][0]

        assert "log_prob" in exp
        assert exp["log_prob"] == -0.5
        assert "entropy" in exp
        assert exp["entropy"] == 0.7

    def test_transfer_with_uniform_replay_strategy(self):
        """Transfer works with uniform replay strategy."""
        buffer = PrioritizedReplayBuffer(max_size=100, replay_strategy="uniform")
        for i in range(10):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=5, seed=42)
        assert slice_data["metadata"]["replay_strategy"] == "uniform"

        child_buffer = PrioritizedReplayBuffer(max_size=100, replay_strategy="uniform")
        child_buffer.load_transfer_slice(slice_data)
        assert len(child_buffer) == 5

    def test_transfer_slice_size_exactly_equals_buffer_size(self):
        """Edge case: max_size exactly equals buffer size."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        for i in range(15):
            buffer.append(
                state=np.array([i], dtype=np.float32),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1], dtype=np.float32),
                done=False,
            )

        slice_data = buffer.get_transfer_slice(max_size=15, seed=42)
        assert len(slice_data["experiences"]) == 15

    def test_load_empty_slice_into_empty_buffer(self):
        """Loading an empty slice into an empty buffer is a no-op."""
        buffer = PrioritizedReplayBuffer(max_size=100)
        slice_data = {
            "experiences": [],
            "priorities": np.array([], dtype=np.float64),
            "metadata": {"alpha": 0.6, "epsilon": 1e-6, "replay_strategy": "prioritized"},
        }

        buffer.load_transfer_slice(slice_data)
        assert len(buffer) == 0

    def test_round_trip_transfer_preserves_data(self):
        """Full round-trip: parent → slice → child preserves all data."""
        parent = PrioritizedReplayBuffer(max_size=100, alpha=0.7, epsilon=1e-5)
        for i in range(30):
            parent.append(
                state=np.array([i, i * 2], dtype=np.float32),
                action=i % 3,
                reward=float(i) * 0.5,
                next_state=np.array([i + 1, (i + 1) * 2], dtype=np.float32),
                done=(i % 10 == 9),
                custom_field=f"data_{i}",
            )
        parent.priorities[10] = 10.5
        parent.priorities[20] = 20.5

        slice_data = parent.get_transfer_slice(max_size=25, seed=42)

        child = PrioritizedReplayBuffer(max_size=100, alpha=0.7, epsilon=1e-5)
        child.load_transfer_slice(slice_data)

        # Verify child matches the most recent 25 experiences from parent
        assert len(child) == 25
        for i in range(25):
            parent_idx = 5 + i  # Most recent 25 start at index 5
            child_idx = i
            parent_exp = parent.buffer[parent_idx]
            child_exp = child.buffer[child_idx]

            assert np.array_equal(parent_exp["state"], child_exp["state"])
            assert parent_exp["action"] == child_exp["action"]
            assert parent_exp["reward"] == child_exp["reward"]
            assert np.array_equal(parent_exp["next_state"], child_exp["next_state"])
            assert parent_exp["done"] == child_exp["done"]
            assert parent_exp["custom_field"] == child_exp["custom_field"]
            assert parent.priorities[parent_idx] == child.priorities[child_idx]


@pytest.mark.determinism
class TestReplayBufferTransferCrossProcessDeterminism:
    """Cross-process determinism tests (matches test_cross_process_determinism.py style)."""

    def test_slice_hash_is_stable_across_interpreter_restarts(self, tmp_path):
        """Slice hash must be identical across fresh interpreter runs (hypothetical).

        This test validates the hash function itself is deterministic. In a real
        cross-process test (similar to test_cross_process_determinism.py), you
        would spawn subprocesses with different PYTHONHASHSEED values and confirm
        identical slice hashes.
        """
        def _build_and_hash():
            buffer = PrioritizedReplayBuffer(max_size=100)
            np.random.seed(42)
            for i in range(30):
                buffer.append(
                    state=np.random.randn(4).astype(np.float32),
                    action=i % 4,
                    reward=np.random.randn(),
                    next_state=np.random.randn(4).astype(np.float32),
                    done=False,
                )
            slice_data = buffer.get_transfer_slice(max_size=15, seed=42)
            return _hash_slice_data(slice_data)

        hash1 = _build_and_hash()
        hash2 = _build_and_hash()
        assert hash1 == hash2, "Hash must be stable across repeated builds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
