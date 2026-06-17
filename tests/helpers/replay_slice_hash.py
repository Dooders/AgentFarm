"""Build a deterministic replay-buffer transfer slice and print its hash.

Used by ``tests/decision/test_replay_buffer_transfer.py`` to verify that slice
selection is byte-identical across fresh interpreters run with *different*
``PYTHONHASHSEED`` values (Issue #902 acceptance criterion). The script does not
pre-seed the interpreter; the buffer is filled from a local ``RandomState`` so
the only thing that varies between subprocesses is hash randomization.
"""

import hashlib
import sys

import numpy as np

from farm.core.decision.algorithms.rl_base import PrioritizedReplayBuffer


def build_slice_hash() -> str:
    """Construct a fixed buffer, take a transfer slice, and hash it canonically."""
    buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6, epsilon=1e-6)
    rng = np.random.RandomState(42)
    for i in range(50):
        buffer.append(
            state=rng.randn(4).astype(np.float32),
            action=i % 4,
            reward=float(rng.randn()),
            next_state=rng.randn(4).astype(np.float32),
            done=bool(i % 7 == 0),
        )

    slice_data = buffer.get_transfer_slice(max_size=20)

    hasher = hashlib.sha256()
    for exp, priority in zip(slice_data["experiences"], slice_data["priorities"]):
        hasher.update(np.ascontiguousarray(exp["state"]).tobytes())
        hasher.update(np.ascontiguousarray(exp["next_state"]).tobytes())
        hasher.update(str(exp["action"]).encode())
        hasher.update(repr(float(exp["reward"])).encode())
        hasher.update(str(bool(exp["done"])).encode())
        hasher.update(repr(float(priority)).encode())

    metadata = slice_data["metadata"]
    hasher.update(repr(metadata["alpha"]).encode())
    hasher.update(repr(metadata["epsilon"]).encode())
    hasher.update(str(metadata["replay_strategy"]).encode())
    return hasher.hexdigest()


if __name__ == "__main__":
    sys.stdout.write(build_slice_hash())
