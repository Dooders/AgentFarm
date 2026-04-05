"""Tests for farm/core/services/factory.py – AgentServiceFactory."""

from unittest.mock import MagicMock

import pytest

from farm.core.services.factory import AgentServiceFactory
from farm.core.services.implementations import (
    EnvironmentAgentLifecycleService,
    EnvironmentLoggingService,
    EnvironmentMetricsService,
    EnvironmentTimeService,
    EnvironmentValidationService,
)


def _mock_environment():
    """Return a minimal mock environment."""
    env = MagicMock()
    env.config = MagicMock()
    return env


class TestAgentServiceFactoryNoEnvironment:
    """create_services with no environment returns None for all services."""

    def test_all_none_when_no_environment(self):
        result = AgentServiceFactory.create_services()
        metrics, logging, validation, time, lifecycle, config = result
        assert metrics is None
        assert logging is None
        assert validation is None
        assert time is None
        assert lifecycle is None
        assert config is None

    def test_explicit_services_returned_unchanged(self):
        mock_metrics = MagicMock()
        mock_logging = MagicMock()
        result = AgentServiceFactory.create_services(
            metrics_service=mock_metrics,
            logging_service=mock_logging,
        )
        metrics, logging, validation, time, lifecycle, config = result
        assert metrics is mock_metrics
        assert logging is mock_logging
        assert validation is None

    def test_explicit_config_returned_unchanged(self):
        mock_config = MagicMock()
        result = AgentServiceFactory.create_services(config=mock_config)
        _, _, _, _, _, config = result
        assert config is mock_config


class TestAgentServiceFactoryWithEnvironment:
    """create_services derives services from the environment when provided."""

    def test_returns_environment_services(self):
        env = _mock_environment()
        result = AgentServiceFactory.create_services(environment=env)
        metrics, logging, validation, time, lifecycle, config = result

        assert isinstance(metrics, EnvironmentMetricsService)
        assert isinstance(logging, EnvironmentLoggingService)
        assert isinstance(validation, EnvironmentValidationService)
        assert isinstance(time, EnvironmentTimeService)
        assert isinstance(lifecycle, EnvironmentAgentLifecycleService)

    def test_config_derived_from_environment(self):
        env = _mock_environment()
        result = AgentServiceFactory.create_services(environment=env)
        _, _, _, _, _, config = result
        assert config is env.config

    def test_explicit_service_overrides_derived(self):
        env = _mock_environment()
        mock_metrics = MagicMock()
        result = AgentServiceFactory.create_services(
            environment=env,
            metrics_service=mock_metrics,
        )
        metrics, *_ = result
        assert metrics is mock_metrics

    def test_explicit_config_overrides_environment_config(self):
        env = _mock_environment()
        my_config = MagicMock()
        result = AgentServiceFactory.create_services(environment=env, config=my_config)
        _, _, _, _, _, config = result
        assert config is my_config

    def test_environment_without_config_attr_returns_none_config(self):
        env = MagicMock(spec=[])  # no 'config' attribute
        result = AgentServiceFactory.create_services(environment=env)
        _, _, _, _, _, config = result
        assert config is None
