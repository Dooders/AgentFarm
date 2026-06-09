"""Tests for farm/utils/spatial.py (bilinear value distribution).

Verifies exact-cell placement, four-cell weight splitting, and the key
mass-conservation invariant via Hypothesis.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

import torch

from farm.utils.spatial import bilinear_distribute_value  # noqa: E402

WIDTH, HEIGHT = 10, 8


def make_grid():
    # Grid tensors are indexed (y, x) -> shape (H, W).
    return torch.zeros((HEIGHT, WIDTH), dtype=torch.float64)


class TestBilinearExamples:
    def test_integer_position_places_full_value_in_single_cell(self):
        grid = make_grid()
        bilinear_distribute_value((3.0, 5.0), 2.5, grid, (WIDTH, HEIGHT))
        assert grid[5, 3].item() == pytest.approx(2.5)
        assert grid.sum().item() == pytest.approx(2.5)

    def test_fractional_position_splits_across_four_cells(self):
        grid = make_grid()
        bilinear_distribute_value((2.25, 4.75), 1.0, grid, (WIDTH, HEIGHT))
        # Weights: (1-xf)(1-yf), (1-xf)yf, xf(1-yf), xf*yf with xf=0.25, yf=0.75
        assert grid[4, 2].item() == pytest.approx(0.75 * 0.25)
        assert grid[5, 2].item() == pytest.approx(0.75 * 0.75)
        assert grid[4, 3].item() == pytest.approx(0.25 * 0.25)
        assert grid[5, 3].item() == pytest.approx(0.25 * 0.75)

    def test_repeated_distributions_accumulate(self):
        grid = make_grid()
        bilinear_distribute_value((1.5, 1.5), 1.0, grid, (WIDTH, HEIGHT))
        bilinear_distribute_value((1.5, 1.5), 1.0, grid, (WIDTH, HEIGHT))
        assert grid.sum().item() == pytest.approx(2.0)
        assert grid[1, 1].item() == pytest.approx(0.5)

    def test_corner_position_keeps_value_in_grid(self):
        grid = make_grid()
        bilinear_distribute_value((WIDTH - 1.0, HEIGHT - 1.0), 1.0, grid, (WIDTH, HEIGHT))
        assert grid[HEIGHT - 1, WIDTH - 1].item() == pytest.approx(1.0)


class TestBilinearProperties:
    @given(
        x=st.floats(0, WIDTH - 1, allow_nan=False),
        y=st.floats(0, HEIGHT - 1, allow_nan=False),
        value=st.floats(-100, 100, allow_nan=False),
    )
    def test_total_mass_is_conserved(self, x, y, value):
        grid = make_grid()
        bilinear_distribute_value((x, y), value, grid, (WIDTH, HEIGHT))
        assert grid.sum().item() == pytest.approx(value, abs=1e-9)

    @given(
        x=st.floats(0, WIDTH - 1, allow_nan=False),
        y=st.floats(0, HEIGHT - 1, allow_nan=False),
    )
    def test_positive_value_yields_non_negative_cells(self, x, y):
        grid = make_grid()
        bilinear_distribute_value((x, y), 1.0, grid, (WIDTH, HEIGHT))
        assert (grid >= 0).all()

    @given(
        x=st.floats(0, WIDTH - 1, allow_nan=False),
        y=st.floats(0, HEIGHT - 1, allow_nan=False),
    )
    def test_mass_lands_in_cells_adjacent_to_position(self, x, y):
        grid = make_grid()
        bilinear_distribute_value((x, y), 1.0, grid, (WIDTH, HEIGHT))
        nonzero = grid.nonzero()
        for y_idx, x_idx in nonzero.tolist():
            assert abs(x_idx - x) <= 1.0
            assert abs(y_idx - y) <= 1.0
