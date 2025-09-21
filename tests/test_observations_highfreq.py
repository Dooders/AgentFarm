import torch
import pytest

from farm.core.observations import AgentObservation, ObservationConfig
from farm.core.channels import Channel


@pytest.fixture
def config_with_highfreq():
    # Mark RESOURCES and ALLIES_HP as high-frequency
    return ObservationConfig(R=3, device="cpu", dtype="float32", high_frequency_channels=["RESOURCES", "ALLIES_HP"])


def test_prebuilt_grid_copy_and_points_update(config_with_highfreq):
    obs = AgentObservation(config_with_highfreq)

    # Store full grid for RESOURCES (high-frequency)
    grid = torch.zeros(2 * config_with_highfreq.R + 1, 2 * config_with_highfreq.R + 1, dtype=config_with_highfreq.torch_dtype)
    grid[1:3, 1:3] = 0.7
    res_idx = 3  # Channel.RESOURCES
    obs._store_sparse_grid(res_idx, grid)

    # Store points for ALLIES_HP (high-frequency) via public handler-style method
    # Use direct sparse API to simulate handler behavior
    allies_idx = Channel.ALLIES_HP
    obs._store_sparse_points(allies_idx, [(2, 3, 0.5), (4, 3, 0.8)], accumulate=False)

    dense = obs.tensor()
    # Grid should copy exactly
    assert torch.allclose(dense[res_idx], grid)
    # Points should be reflected
    assert dense[allies_idx, 2, 3] == pytest.approx(0.5)
    assert dense[allies_idx, 4, 3] == pytest.approx(0.8)


def test_vectorized_sparse_population_and_metrics():
    # Use non-high-frequency channels to test sparse point operations
    config = ObservationConfig(R=3, device="cpu", dtype="float32")
    obs = AgentObservation(config)

    enemies_idx = Channel.ENEMIES_HP
    # Verify ENEMIES_HP is not high-frequency (not in config.high_frequency_channels)
    assert "ENEMIES_HP" not in config.high_frequency_channels
    # Populate a bunch of points in SparsePoints-based sparse storage path
    # This channel uses SparsePoints since it's not high-frequency
    points = []
    for y in range(0, 2 * config.R + 1):
        for x in range(0, 2 * config.R + 1):
            val = (y + x) / 100.0
            points.append((y, x, val))
    obs._store_sparse_points(enemies_idx, points, accumulate=False)

    dense = obs.tensor()
    # Validate a few positions
    assert dense[enemies_idx, 0, 0] == pytest.approx(0.0)
    assert dense[enemies_idx, 3, 4] == pytest.approx((3 + 4) / 100.0)
    assert dense[enemies_idx, 6, 6] == pytest.approx((6 + 6) / 100.0)

    # Metrics should reflect sparse point application at least once
    # This happens when tensor() is called and _build_dense_tensor processes SparsePoints data
    metrics = obs.get_metrics()
    assert metrics["sparse_apply_calls"] >= 1


def test_decay_and_clear_on_prebuilt_channel(config_with_highfreq):
    obs = AgentObservation(config_with_highfreq)

    # Use ALLY_SIGNAL as dynamic but not prebuilt; then check ALLIES_HP prebuilt clear
    allies_idx = Channel.ALLIES_HP
    # Seed values
    obs._store_sparse_points(allies_idx, [(2, 2, 1.0), (3, 3, 0.4)], accumulate=False)

    # Clear should zero prebuilt slice
    obs._clear_sparse_channel(allies_idx)
    dense = obs.tensor()
    assert torch.all(dense[allies_idx] == 0.0)

    # Now test decay on prebuilt grid channel RESOURCES by writing grid, then decaying via decay helper
    res_idx = Channel.RESOURCES
    S = 2 * config_with_highfreq.R + 1
    grid = torch.ones(S, S, dtype=config_with_highfreq.torch_dtype)
    obs._store_sparse_grid(res_idx, grid)
    # Apply decay path that hits prebuilt
    obs._decay_sparse_channel(res_idx, 0.5)
    dense = obs.tensor()
    assert torch.allclose(dense[res_idx], torch.full((S, S), 0.5, dtype=config_with_highfreq.torch_dtype))

