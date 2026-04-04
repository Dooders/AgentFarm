"""Deprecated farm.core.spatial_index compatibility shim."""

import pytest


def test_spatial_index_shim_emits_deprecation_and_reexports():
    with pytest.warns(DeprecationWarning, match="farm.core.spatial_index is deprecated"):
        from farm.core import spatial_index as shim  # noqa: PLC0415

    assert shim.SpatialIndex is not None
    assert "SpatialIndex" in shim.__all__
