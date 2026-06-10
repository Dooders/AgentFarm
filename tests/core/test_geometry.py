"""Tests for farm/core/geometry.py.

Combines example-based tests with Hypothesis property tests for the
continuous-to-discrete position conversion used by the spatial grid.
"""

from hypothesis import given
from hypothesis import strategies as st

from farm.core.geometry import discretize_position_continuous

GRID = (10, 8)  # (width, height)


class TestDiscretizeExamples:
    def test_floor_is_default(self):
        assert discretize_position_continuous((3.9, 2.9), GRID) == (3, 2)

    def test_round_method(self):
        assert discretize_position_continuous((3.6, 2.4), GRID, method="round") == (4, 2)

    def test_ceil_method(self):
        assert discretize_position_continuous((3.1, 2.1), GRID, method="ceil") == (4, 3)

    def test_unknown_method_falls_back_to_floor(self):
        assert discretize_position_continuous((3.9, 2.9), GRID, method="nonsense") == (3, 2)

    def test_negative_coordinates_clamp_to_zero(self):
        assert discretize_position_continuous((-5.0, -0.1), GRID) == (0, 0)

    def test_coordinates_beyond_grid_clamp_to_max_index(self):
        assert discretize_position_continuous((99.0, 99.0), GRID) == (GRID[0] - 1, GRID[1] - 1)

    def test_exact_grid_corner(self):
        assert discretize_position_continuous((0.0, 0.0), GRID) == (0, 0)


grid_sizes = st.tuples(st.integers(1, 200), st.integers(1, 200))
coordinates = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
methods = st.sampled_from(["floor", "round", "ceil"])


class TestDiscretizeProperties:
    @given(x=coordinates, y=coordinates, grid_size=grid_sizes, method=methods)
    def test_result_always_within_grid_bounds(self, x, y, grid_size, method):
        x_idx, y_idx = discretize_position_continuous((x, y), grid_size, method=method)
        assert 0 <= x_idx < grid_size[0]
        assert 0 <= y_idx < grid_size[1]

    @given(
        x=st.integers(0, 99),
        y=st.integers(0, 99),
        method=methods,
    )
    def test_integer_positions_map_to_themselves(self, x, y, method):
        assert discretize_position_continuous((float(x), float(y)), (100, 100), method=method) == (x, y)

    @given(x=coordinates, y=coordinates, grid_size=grid_sizes)
    def test_floor_never_exceeds_ceil(self, x, y, grid_size):
        floor_idx = discretize_position_continuous((x, y), grid_size, method="floor")
        ceil_idx = discretize_position_continuous((x, y), grid_size, method="ceil")
        assert floor_idx[0] <= ceil_idx[0]
        assert floor_idx[1] <= ceil_idx[1]
