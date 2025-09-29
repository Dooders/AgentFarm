import os
import random
import socket
import time
from contextlib import contextmanager

import numpy as np
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Auto-mark tests under certain directories.

    - tests/integration/** -> integration
    - tests/decision/** -> ml (heuristic)
    """
    for item in items:
        path = str(item.fspath)
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
        if "/tests/decision/" in path:
            item.add_marker(pytest.mark.ml)


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
    """Provide a SimulationDatabase with a seeded simulation record."""
    from farm.database.database import SimulationDatabase

    database = SimulationDatabase(str(tmp_db_path), simulation_id="test_sim")
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
