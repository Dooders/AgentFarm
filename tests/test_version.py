"""Tests for canonical package versioning."""

from importlib.metadata import version

from farm import __version__ as farm_version
from farm._version import __version__ as source_version


def test_farm_version_matches_source_module() -> None:
    assert farm_version == source_version


def test_installed_package_version_matches_source() -> None:
    assert version("agentfarm") == source_version
