"""Tests for canonical package versioning."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import farm.analysis as farm_analysis
import farm.api as farm_api
from farm import __version__ as farm_version
from farm._version import __version__ as source_version


def test_farm_version_matches_source_module() -> None:
    assert farm_version == source_version


def test_installed_package_version_matches_source() -> None:
    try:
        installed = version("agentfarm")
    except PackageNotFoundError:
        pytest.skip("agentfarm is not installed; run `pip install -e .` to enable this check")
    assert installed == source_version


def test_subpackages_share_canonical_version() -> None:
    """Subpackages must report the unified version, not independent literals."""
    assert farm_api.__version__ == source_version
    assert farm_analysis.__version__ == source_version
