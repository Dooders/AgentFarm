import os
import random
import socket
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Directories and files that unconditionally import torch and will fail to
# collect when torch is not installed.  Listed here so that
# ``pytest_ignore_collect`` can skip them gracefully.
_TORCH_DEPENDENT_DIRS = {"decision"}
_TORCH_DEPENDENT_FILES = {
    "test_channels.py",
    "test_default_behavior.py",
    "test_observations.py",
    "test_observation_pipeline.py",
    "test_state_models.py",
    "test_observations_highfreq.py",
    "test_deterministic.py",
}


def pytest_ignore_collect(collection_path, config):
    """Skip torch-dependent directories/files when torch is not available.

    This prevents collection-time ImportError for files that unconditionally
    import torch at module level.
    """
    if torch is not None:
        return None
    if collection_path.is_dir() and collection_path.name in _TORCH_DEPENDENT_DIRS:
        return True
    if collection_path.is_file() and collection_path.name in _TORCH_DEPENDENT_FILES:
        return True
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Auto-mark tests under certain directories and skip torch-dependent ones.

    - tests/integration/** -> integration
    - tests/decision/** -> ml (heuristic)
    - ml-marked tests are skipped automatically when torch is not installed.
    """
    torch_skip_marker = pytest.mark.skip(reason="torch not installed") if torch is None else None

    for item in items:
        path = str(item.fspath)
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
        if "/tests/decision/" in path:
            item.add_marker(pytest.mark.ml)
        if torch_skip_marker is not None and item.get_closest_marker("ml"):
            item.add_marker(torch_skip_marker)


@pytest.fixture(autouse=True)
def set_test_seed():
    """Ensure deterministic RNG for tests unless overridden.

    Use PYTEST_SEED env var to vary if needed.
    """
    seed = int(os.getenv("PYTEST_SEED", "1234"))
    random.seed(seed)
    np.random.seed(seed)
    yield


@pytest.fixture()
def fast_sleep(monkeypatch):
    """Speed up sleeps in tests by no-oping time.sleep."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    yield


@pytest.fixture()
def disable_network(monkeypatch):
    """Prevent accidental real network calls during tests."""

    def deny(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Network access is disabled during tests")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    yield


@pytest.fixture()
def tmp_db_path(tmp_path):
    """Provide a temporary SQLite DB file path within pytest's unique temp dir."""
    path = tmp_path / "test.db"
    yield str(path)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@pytest.fixture()
def db(tmp_db_path):
    """Provide a database instance implementing DatabaseProtocol.

    Returns a SimulationDatabase with a seeded simulation record.
    The returned instance satisfies DatabaseProtocol for type-safe testing.
    """
    from farm.core.interfaces import DatabaseProtocol
    from farm.database.database import SimulationDatabase

    # Create concrete implementation that satisfies DatabaseProtocol
    database: DatabaseProtocol = SimulationDatabase(str(tmp_db_path), simulation_id="test_sim")
    # Ensure a simulation record exists for FK constraints
    database.add_simulation_record(
        simulation_id="test_sim",
        start_time=time.time(),
        status="running",
        parameters={},
    )
    try:
        yield database
    finally:
        database.close()
        try:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)
        except OSError:
            pass


@pytest.fixture()
def env():
    """Provide a lightweight Environment suitable for fast tests."""
    from farm.config.config import SimulationConfig
    from farm.core.environment import Environment

    config = SimulationConfig(
        width=20,
        height=20,
        system_agents=0,
        independent_agents=0,
        control_agents=0,
        initial_resources=3,
        max_steps=10,
        seed=1234,
    )

    environment = Environment(
        width=config.width,
        height=config.height,
        resource_distribution={"amount": 3},
        config=config,
        db_path=":memory:",
    )
    try:
        yield environment
    finally:
        environment.cleanup()


@contextmanager
def env_vars(**overrides):
    """Context manager to temporarily set environment variables during a test."""
    old_values = {k: os.environ.get(k) for k in overrides.keys()}
    for key, value in overrides.items():
        if value is None and key in os.environ:
            os.environ.pop(key)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, prev in old_values.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


# Determinism Testing Helpers

def seed_all_rngs(seed: int) -> None:
    """
    Comprehensive RNG seeding function for deterministic behavior.
    
    Parameters
    ----------
    seed : int
        Seed value to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def capture_component_state(component: Any) -> Dict[str, Any]:
    """
    Extract serializable state from agent components.
    
    Parameters
    ----------
    component : Any
        Agent component to extract state from
        
    Returns
    -------
    Dict[str, Any]
        Serializable component state
    """
    component_state = {}
    component_name = component.__class__.__name__.lower()
    
    if "resource" in component_name:
        component_state.update({
            "level": getattr(component, "level", None),
            "starvation_counter": getattr(component, "starvation_counter", None),
        })
    elif "movement" in component_name:
        component_state.update({
            "target_position": getattr(component, "target_position", None),
            "last_position": getattr(component, "last_position", None),
        })
    elif "reproduction" in component_name:
        component_state.update({
            "offspring_created": getattr(component, "offspring_created", None),
            "reproduction_cooldown": getattr(component, "reproduction_cooldown", None),
        })
    
    return component_state


def compare_states_detailed(state1: Dict[str, Any], state2: Dict[str, Any], tolerance: float = 1e-9) -> bool:
    """
    Enhanced comparison with floating-point tolerance.
    
    Parameters
    ----------
    state1 : Dict[str, Any]
        First state to compare
    state2 : Dict[str, Any]
        Second state to compare
    tolerance : float
        Floating-point tolerance for numerical comparisons
        
    Returns
    -------
    bool
        True if states are equivalent within tolerance
    """
    def _compare_values(v1, v2):
        if isinstance(v1, (int, str, bool)) and isinstance(v2, (int, str, bool)):
            return v1 == v2
        elif isinstance(v1, (float, np.floating)) and isinstance(v2, (float, np.floating)):
            return abs(v1 - v2) < tolerance
        elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if len(v1) != len(v2):
                return False
            return all(_compare_values(a, b) for a, b in zip(v1, v2))
        elif isinstance(v1, dict) and isinstance(v2, dict):
            if set(v1.keys()) != set(v2.keys()):
                return False
            return all(_compare_values(v1[k], v2[k]) for k in v1.keys())
        else:
            return v1 == v2
    
    return _compare_values(state1, state2)


@pytest.fixture
def deterministic_seed():
    """Provide a deterministic seed for determinism tests."""
    return 42


@pytest.fixture
def multiple_test_seeds():
    """Provide multiple seeds for comprehensive determinism testing."""
    return [42, 123, 456, 789, 999]
