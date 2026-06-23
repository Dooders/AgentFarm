"""Tests for farm/core/device_utils.py – DeviceManager and convenience helpers."""

import os

import pytest
import torch

from farm.core.device_utils import (
    DeviceManager,
    create_device_from_config,
    get_device,
    get_device_manager,
    safe_tensor_to_device,
)
import farm.core.device_utils as _dutils


def _raise_value_error(*_args, **_kwargs):
    raise ValueError("bad cpu_threads")


@pytest.fixture(autouse=True)
def reset_global_device_manager():
    """Ensure each test starts with a fresh global device manager."""
    _dutils._global_device_manager = None
    yield
    _dutils._global_device_manager = None


class TestDeviceManagerInit:
    def test_defaults(self):
        dm = DeviceManager()
        assert dm.preference == "auto"
        assert dm.fallback is True
        assert dm.memory_fraction is None
        assert dm.validate_compatibility is True
        assert dm.cpu_threads == 1
        assert not dm._initialized

    def test_custom_params(self):
        dm = DeviceManager(
            preference="cpu",
            fallback=False,
            memory_fraction=0.5,
            validate_compatibility=False,
            cpu_threads=2,
        )
        assert dm.preference == "cpu"
        assert dm.fallback is False
        assert dm.memory_fraction == 0.5
        assert dm.validate_compatibility is False
        assert dm.cpu_threads == 2


class TestDeviceManagerGetDevice:
    def test_cpu_thread_policy_applied_for_cpu_device(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
        monkeypatch.setenv("OMP_NUM_THREADS", "999")
        monkeypatch.setenv("MKL_NUM_THREADS", "999")
        dm = DeviceManager(preference="cpu", cpu_threads=2)
        device = dm.get_device()
        assert device.type == "cpu"
        assert calls == [2]
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["MKL_NUM_THREADS"] == "2"

    def test_invalid_cpu_threads_raise_value_error(self):
        dm = DeviceManager(preference="cpu", cpu_threads=0)
        with pytest.raises(ValueError, match="cpu_threads must be >= 1"):
            dm.get_device()

    def test_bool_cpu_threads_raise_value_error(self):
        dm = DeviceManager(preference="cpu", cpu_threads=True)
        with pytest.raises(ValueError, match="cpu_threads must be an int >= 1"):
            dm.get_device()

    def test_failed_configuration_does_not_leave_manager_initialized(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_configure_cpu_threads", _raise_value_error)
        dm = DeviceManager(preference="cpu", cpu_threads=2)
        with pytest.raises(ValueError, match="bad cpu_threads"):
            dm.get_device()
        assert dm._initialized is False
        assert dm._device is None

    def test_cpu_preference_returns_cpu(self):
        dm = DeviceManager(preference="cpu")
        device = dm.get_device()
        assert device.type == "cpu"

    def test_auto_returns_device(self):
        dm = DeviceManager(preference="auto")
        device = dm.get_device()
        assert device.type in ("cpu", "cuda")

    def test_device_cached_after_first_call(self):
        dm = DeviceManager(preference="cpu")
        d1 = dm.get_device()
        d2 = dm.get_device()
        assert d1 == d2
        assert dm._initialized

    def test_unknown_preference_falls_back_to_auto(self):
        dm = DeviceManager(preference="tpu")
        device = dm.get_device()
        assert device.type in ("cpu", "cuda")

    def test_cuda_preference_without_cuda_falls_back_when_allowed(self):
        dm = DeviceManager(preference="cuda", fallback=True)
        device = dm.get_device()
        # On CPU-only CI this should fall back to cpu
        assert device.type in ("cpu", "cuda")

    def test_cuda_preference_without_cuda_raises_when_no_fallback(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; this test targets CPU-only environments")
        dm = DeviceManager(preference="cuda", fallback=False)
        with pytest.raises(RuntimeError):
            dm.get_device()

    def test_cuda_colon_n_preference_falls_back_when_unavailable(self):
        dm = DeviceManager(preference="cuda:99", fallback=True)
        device = dm.get_device()
        assert device.type in ("cpu", "cuda")


class TestDeviceManagerReset:
    def test_reset_clears_initialized_flag(self):
        dm = DeviceManager(preference="cpu")
        dm.get_device()
        assert dm._initialized
        dm.reset()
        assert not dm._initialized
        assert dm._device is None


class TestTensorCompatibility:
    def test_same_device_is_compatible(self):
        dm = DeviceManager(preference="cpu")
        cpu_device = torch.device("cpu")
        t = torch.tensor([1.0])
        assert dm.validate_tensor_compatibility(t, cpu_device) is True

    def test_validation_disabled_always_returns_true(self):
        dm = DeviceManager(preference="cpu", validate_compatibility=False)
        t = torch.tensor([1.0])
        assert dm.validate_tensor_compatibility(t, torch.device("cpu")) is True

    def test_safe_tensor_to_device_cpu(self):
        dm = DeviceManager(preference="cpu")
        t = torch.tensor([1.0, 2.0])
        result = dm.safe_tensor_to_device(t, torch.device("cpu"))
        assert result.device.type == "cpu"


class TestConvenienceFunctions:
    def test_get_device_returns_device(self):
        d = get_device(preference="cpu")
        assert isinstance(d, torch.device)
        assert d.type == "cpu"

    def test_get_device_manager_singleton(self):
        m1 = get_device_manager(preference="cpu")
        m2 = get_device_manager(preference="cpu")
        assert m1 is m2

    def test_get_device_manager_reconfigures_on_different_preference(self):
        m1 = get_device_manager(preference="cpu")
        m1.get_device()  # force initialization
        m2 = get_device_manager(preference="auto")
        # After reconfiguration the manager should be reset
        assert not m2._initialized or m2.preference == "auto"

    def test_get_device_manager_reconfigures_on_different_cpu_threads(self):
        m1 = get_device_manager(preference="cpu", cpu_threads=1)
        m2 = get_device_manager(preference="cpu", cpu_threads=2)
        assert m1 is m2
        assert m2.cpu_threads == 2
        assert not m2._initialized

    def test_safe_tensor_to_device_function(self):
        t = torch.tensor([3.0])
        result = safe_tensor_to_device(t, torch.device("cpu"))
        assert result.device.type == "cpu"


class TestGetOptimalDevice:
    def test_returns_device_without_size(self):
        dm = DeviceManager(preference="cpu")
        d = dm.get_optimal_device_for_model()
        assert isinstance(d, torch.device)

    def test_returns_device_with_size(self):
        dm = DeviceManager(preference="cpu")
        d = dm.get_optimal_device_for_model(model_size_mb=10.0)
        assert isinstance(d, torch.device)


class TestDeviceManagerCaching:
    def test_cuda_availability_cached(self, monkeypatch):
        calls = {"is_available": 0}

        def fake_is_available():
            calls["is_available"] += 1
            return False

        dm = DeviceManager()
        monkeypatch.setattr(torch.cuda, "is_available", fake_is_available)

        assert dm._is_cuda_available() is False
        assert dm._is_cuda_available() is False
        assert calls["is_available"] == 1


class TestCreateDeviceFromConfig:
    def test_create_device_from_config_applies_cpu_threads(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))

        class DeviceSettings:
            device_preference = "cpu"
            device_fallback = True
            device_memory_fraction = None
            device_validate_compatibility = True
            cpu_threads = 3

        class Config:
            device = DeviceSettings()

        device = create_device_from_config(Config())
        assert device.type == "cpu"
        manager = get_device_manager(preference="cpu", cpu_threads=3)
        assert manager.cpu_threads == 3
        assert calls == [3]
