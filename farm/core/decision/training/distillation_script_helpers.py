"""Shared helpers for distillation / quantization CLI scripts.

Keeps checkpoint and state loading consistent across ``scripts/``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork


def load_float_student_checkpoint(
    path: str,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
    *,
    not_found_template: str,
    bad_state_template: str,
) -> StudentQNetwork:
    """Load a float :class:`StudentQNetwork` from a state-dict ``.pt`` file."""
    model = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=parent_hidden,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(not_found_template.format(path=path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            bad_state_template.format(path=path, type_name=type(state).__name__)
        )
    model.load_state_dict(state)
    model.eval()
    return model


def load_base_qnetwork_checkpoint(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    *,
    not_found_template: str = "Parent checkpoint not found: {path!r}",
    bad_state_template: str = (
        "Checkpoint at {path!r} does not contain a state dict (got {type_name})."
    ),
    loaded_template: str = "  Loaded network from: {path}",
    random_weights_message: str = "  No checkpoint provided; using random weights.",
) -> BaseQNetwork:
    """Load a :class:`BaseQNetwork` from a state-dict ``.pt`` file, or random weights if *path* is empty."""
    net = BaseQNetwork(
        input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size
    )
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(not_found_template.format(path=path))
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                bad_state_template.format(
                    path=path, type_name=type(state).__name__
                )
            )
        net.load_state_dict(state)
        print(loaded_template.format(path=path))
    else:
        print(random_weights_message)
    return net


def load_distillation_states(
    states_file: str,
    n_states: int,
    input_dim: int,
    seed: Optional[int],
    *,
    file_not_found_template: str = "States file not found: {path!r}",
) -> np.ndarray:
    """Load ``.npy`` states or synthesise standard-normal calibration data."""
    if states_file:
        if not os.path.isfile(states_file):
            raise FileNotFoundError(file_not_found_template.format(path=states_file))
        states = np.load(states_file).astype("float32")
        if states.ndim != 2:
            raise ValueError(
                f"Loaded states must be a 2-D array with shape (N, input_dim); got {states.shape!r}"
            )
        if states.shape[1] != input_dim:
            raise ValueError(
                f"States input_dim mismatch: expected {input_dim}, got {states.shape[1]}"
            )
        print(f"  Loaded states from {states_file}: shape={states.shape}")
        return states
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n_states, input_dim)).astype("float32")
    print(f"  Using {n_states} synthetic random states (shape={states.shape})")
    return states
