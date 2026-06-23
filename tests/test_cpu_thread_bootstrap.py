"""Tests for farm/cpu_thread_bootstrap.py – pre-import CPU thread pinning."""

import os

import pytest

from farm.cpu_thread_bootstrap import (
    CPU_THREAD_ENV_VARS,
    pin_cpu_math_threads,
    resolve_cpu_threads_from_config,
    validate_cpu_threads,
)


@pytest.fixture(autouse=True)
def clean_thread_env(monkeypatch):
    """Run each test with the CPU thread env vars cleared."""
    for var in CPU_THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


class TestValidateCpuThreads:
    def test_none_passes_through(self):
        assert validate_cpu_threads(None) is None

    def test_valid_int(self):
        assert validate_cpu_threads(4) == 4

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            validate_cpu_threads(0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            validate_cpu_threads(-1)

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            validate_cpu_threads(True)

    def test_non_int_rejected(self):
        with pytest.raises(ValueError):
            validate_cpu_threads(2.0)


class TestPinCpuMathThreads:
    def test_sets_all_env_vars(self):
        applied = pin_cpu_math_threads(3)
        assert applied == 3
        for var in CPU_THREAD_ENV_VARS:
            assert os.environ[var] == "3"

    def test_none_leaves_env_untouched(self):
        assert pin_cpu_math_threads(None) is None
        for var in CPU_THREAD_ENV_VARS:
            assert var not in os.environ

    def test_respects_preexisting_env_by_default(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        applied = pin_cpu_math_threads(1)
        # Pre-existing var is respected; the others are newly set.
        assert os.environ["OMP_NUM_THREADS"] == "8"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert applied == 1

    def test_override_replaces_preexisting(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        pin_cpu_math_threads(2, override=True)
        assert os.environ["OMP_NUM_THREADS"] == "2"

    def test_returns_none_when_all_present_and_no_override(self, monkeypatch):
        for var in CPU_THREAD_ENV_VARS:
            monkeypatch.setenv(var, "8")
        assert pin_cpu_math_threads(1) is None

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            pin_cpu_math_threads(0)


class TestResolveCpuThreadsFromConfig:
    def _write(self, path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_reads_base_default(self, tmp_path):
        cfg = tmp_path / "cfg"
        self._write(str(cfg / "default.yaml"), "cpu_threads: 1\n")
        self._write(str(cfg / "environments" / "development.yaml"), "foo: bar\n")
        assert (
            resolve_cpu_threads_from_config(
                environment="development", config_dir=str(cfg)
            )
            == 1
        )

    def test_environment_overrides_base(self, tmp_path):
        cfg = tmp_path / "cfg"
        self._write(str(cfg / "default.yaml"), "cpu_threads: 1\n")
        self._write(
            str(cfg / "environments" / "production.yaml"), "cpu_threads: 8\n"
        )
        assert (
            resolve_cpu_threads_from_config(
                environment="production", config_dir=str(cfg)
            )
            == 8
        )

    def test_profile_has_highest_precedence(self, tmp_path):
        cfg = tmp_path / "cfg"
        self._write(str(cfg / "default.yaml"), "cpu_threads: 1\n")
        self._write(
            str(cfg / "environments" / "development.yaml"), "cpu_threads: 4\n"
        )
        self._write(str(cfg / "profiles" / "benchmark.yaml"), "cpu_threads: 16\n")
        assert (
            resolve_cpu_threads_from_config(
                environment="development",
                profile="benchmark",
                config_dir=str(cfg),
            )
            == 16
        )

    def test_explicit_null_returns_none(self, tmp_path):
        cfg = tmp_path / "cfg"
        self._write(str(cfg / "default.yaml"), "cpu_threads: null\n")
        assert (
            resolve_cpu_threads_from_config(
                environment="development", config_dir=str(cfg), default=1
            )
            is None
        )

    def test_nested_device_key(self, tmp_path):
        cfg = tmp_path / "cfg"
        self._write(str(cfg / "default.yaml"), "device:\n  cpu_threads: 5\n")
        assert (
            resolve_cpu_threads_from_config(
                environment="development", config_dir=str(cfg)
            )
            == 5
        )

    def test_missing_files_falls_back_to_default(self, tmp_path):
        cfg = tmp_path / "does_not_exist"
        assert (
            resolve_cpu_threads_from_config(config_dir=str(cfg), default=2) == 2
        )

    def test_real_repo_default_is_one(self):
        # The shipped default config pins a single CPU thread.
        assert resolve_cpu_threads_from_config(environment="development") == 1
