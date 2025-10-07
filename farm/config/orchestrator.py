"""
Configuration Orchestrator - Main Entry Point for Agent Farm Configuration.

This module provides the primary interface for loading, validating, and caching
Agent Farm configurations. It implements the orchestrator pattern to coordinate
between caching, loading, and validation components while maintaining clean
separation of concerns.

Key Features:
- Unified configuration loading with caching and validation
- Automatic error recovery and repair
- Performance optimization through intelligent caching
- Clean separation of components without circular dependencies
- Comprehensive error handling and logging

Example Usage:
    # Basic usage with global orchestrator
    from farm.config import load_config
    config = load_config("production", profile="benchmark")

    # Advanced usage with custom orchestrator
    from farm.config import ConfigurationOrchestrator
    orchestrator = ConfigurationOrchestrator()
    config = orchestrator.load_config(
        environment="development",
        validate=True,
        auto_repair=True
    )

    # Check cache performance
    stats = orchestrator.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
"""

import logging
from typing import Any, Dict, Optional, Tuple

from .cache import ConfigCache, OptimizedConfigLoader
from .config import SimulationConfig
from .validation import SafeConfigLoader, ConfigurationRecovery

logger = logging.getLogger(__name__)


class ConfigurationOrchestrator:
    """
    Orchestrates configuration loading, validation, and caching.

    This class coordinates between cache, loader, and validator components
    to provide a clean API while breaking circular dependencies. It follows
    the orchestrator pattern to maintain separation of concerns.
    """

    def __init__(
        self,
        cache: Optional[ConfigCache] = None,
        loader: Optional[OptimizedConfigLoader] = None,
        validator: Optional[SafeConfigLoader] = None,
    ):
        """
        Initialize the configuration orchestrator.

        Args:
            cache: Configuration cache instance (creates default if None)
            loader: Configuration loader instance (creates default if None)
            validator: Configuration validator instance (creates default if None)
        """
        self.cache = cache or ConfigCache()
        self.loader = loader or OptimizedConfigLoader(cache=self.cache)
        self.validator = validator or SafeConfigLoader()

        logger.info("ConfigurationOrchestrator initialized")

    def load_config(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        validate: bool = True,
        use_cache: bool = True,
        strict_validation: bool = False,
        auto_repair: bool = False,
        config_dir: str = "farm/config",
    ) -> SimulationConfig:
        """
        Load configuration with full pipeline: cache → load → validate.

        This method orchestrates the complete configuration loading process,
        coordinating between cache, loader, and validator components.

        Args:
            environment: Environment name (e.g., 'development', 'production')
            profile: Optional profile name for additional overrides
            validate: Whether to validate the loaded configuration
            use_cache: Whether to use caching for performance
            strict_validation: Whether to treat warnings as errors
            auto_repair: Whether to attempt automatic repair of validation errors
            config_dir: Base directory containing configuration files

        Returns:
            Loaded and validated SimulationConfig instance

        Raises:
            ConfigurationError: If configuration cannot be loaded or validated
            ValidationError: If validation fails and auto_repair is disabled
            FileNotFoundError: If required configuration files are missing
        """
        logger.info(
            f"Loading configuration: environment={environment}, "
            f"profile={profile}, validate={validate}, use_cache={use_cache}"
        )

        try:
            # Step 1: Attempt to load from cache (if enabled)
            if use_cache:
                config_dict = self._load_from_cache(environment, profile, config_dir)
                if config_dict is not None:
                    logger.debug("Configuration loaded from cache")
                    config = SimulationConfig.from_dict(config_dict)
                    if validate:
                        config = self._validate_config(config, strict_validation, auto_repair)
                    return config

            # Step 2: Load configuration from files
            logger.debug("Loading configuration from files")
            config = self._load_from_files(environment, profile, config_dir)

            # Step 3: Cache the loaded configuration (if caching enabled)
            if use_cache:
                self._cache_config(config, environment, profile, config_dir)
                logger.debug("Configuration cached")

            # Step 4: Validate configuration (if validation enabled)
            if validate:
                config = self._validate_config(config, strict_validation, auto_repair)

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def load_config_with_status(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        validate: bool = True,
        use_cache: bool = True,
        strict_validation: bool = False,
        auto_repair: bool = False,
        config_dir: str = "farm/config",
    ) -> Tuple[SimulationConfig, Dict[str, Any]]:
        """
        Load configuration with detailed status information.

        Args:
            environment: Environment name
            profile: Optional profile name
            validate: Whether to validate the configuration
            use_cache: Whether to use caching
            strict_validation: Whether to treat warnings as errors
            auto_repair: Whether to attempt automatic repair
            config_dir: Configuration directory

        Returns:
            Tuple of (config, status_dict) where status_dict contains
            information about the loading process, validation results, etc.
        """
        status = {
            "environment": environment,
            "profile": profile,
            "cached": False,
            "validated": validate,
            "errors": [],
            "warnings": [],
            "repair_actions": [],
            "fallback_used": False,
        }

        try:
            config = self.load_config(
                environment=environment,
                profile=profile,
                validate=validate,
                use_cache=use_cache,
                strict_validation=strict_validation,
                auto_repair=auto_repair,
                config_dir=config_dir,
            )

            # If we get here without exception, loading was successful
            status["success"] = True

        except Exception as e:
            # If validation fails but auto_repair is enabled, we might still get a config
            if validate and auto_repair:
                try:
                    config, validation_status = self.validator.load_config_safely(
                        environment=environment,
                        profile=profile,
                        config_dir=config_dir,
                        strict_validation=strict_validation,
                        auto_repair=True,
                    )
                    status.update(validation_status)
                    status["success"] = True
                except Exception:
                    status["success"] = False
                    status["errors"].append(str(e))
                    raise
            else:
                status["success"] = False
                status["errors"].append(str(e))
                raise

        return config, status

    def invalidate_cache(
        self,
        environment: Optional[str] = None,
        profile: Optional[str] = None,
        config_dir: str = "farm/config",
    ) -> None:
        """
        Invalidate cached configurations.

        Args:
            environment: Specific environment to invalidate (None for all)
            profile: Specific profile to invalidate (None for all)
            config_dir: Configuration directory
        """
        if environment is None:
            # Clear entire cache
            self.cache.clear()
            logger.info("Cleared entire configuration cache")
        else:
            # Invalidate specific cache entry
            cache_key = self.loader._create_cache_key(environment, profile, config_dir)
            self.cache.invalidate(cache_key)
            logger.info(f"Invalidated cache for {environment}/{profile}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()

    def preload_common_configs(
        self,
        environments: Optional[list] = None,
        profiles: Optional[list] = None,
        config_dir: str = "farm/config",
    ) -> None:
        """
        Preload commonly used configurations into cache.

        Args:
            environments: List of environments to preload (default: common ones)
            profiles: List of profiles to preload (default: common ones)
            config_dir: Configuration directory
        """
        if environments is None:
            environments = ["development", "production", "testing"]

        if profiles is None:
            profiles = [None, "benchmark", "simulation"]

        logger.info("Preloading common configurations into cache")

        for env in environments:
            for profile in profiles:
                try:
                    self.load_config(
                        environment=env,
                        profile=profile,
                        validate=False,  # Skip validation during preload
                        use_cache=True,
                        config_dir=config_dir,
                    )
                    logger.debug(f"Preloaded config: {env}/{profile}")
                except Exception as e:
                    logger.warning(f"Failed to preload {env}/{profile}: {e}")

        logger.info("Configuration preload completed")

    def _load_from_cache(self, environment: str, profile: Optional[str], config_dir: str) -> Optional[Dict[str, Any]]:
        """Load configuration dict from cache if available."""
        cache_key = self.loader._create_cache_key(environment, profile, config_dir)
        filepaths = self.loader._get_config_filepaths(environment, profile, config_dir)
        return self.cache.get(cache_key, filepaths)

    def _load_from_files(self, environment: str, profile: Optional[str], config_dir: str) -> SimulationConfig:
        """Load configuration from files."""
        return self.loader.load_centralized_config(
            environment=environment,
            profile=profile,
            config_dir=config_dir,
            use_cache=False,  # Don't use cache here, we handle caching at orchestrator level
        )

    def _cache_config(
        self,
        config: SimulationConfig,
        environment: str,
        profile: Optional[str],
        config_dir: str,
    ) -> None:
        """Cache the loaded configuration."""
        cache_key = self.loader._create_cache_key(environment, profile, config_dir)
        filepaths = self.loader._get_config_filepaths(environment, profile, config_dir)
        self.cache.put(cache_key, config, filepaths)

    def _validate_config(
        self,
        config: SimulationConfig,
        strict_validation: bool = False,
        auto_repair: bool = False,
    ) -> SimulationConfig:
        """
        Validate configuration and apply repairs if needed.

        Args:
            config: Configuration to validate
            strict_validation: Whether to treat warnings as errors
            auto_repair: Whether to attempt automatic repair

        Returns:
            Validated (and potentially repaired) configuration
        """
        logger.debug("Validating configuration")

        # Convert to dict for validation
        config_dict = config.to_dict()

        # Validate the config dict
        validated_dict, status = self.validator.validate_config_dict(
            config_dict, strict_validation=strict_validation, auto_repair=auto_repair
        )

        # Check if validation was successful
        if not status.get("success", False):
            from .validation import ValidationError

            errors = status.get("errors", [])
            warnings = status.get("warnings", [])
            raise ValidationError(
                f"Configuration validation failed: {len(errors)} errors",
                details={"errors": errors, "warnings": warnings, "status": status},
            )

        # Convert back to SimulationConfig
        validated_config = SimulationConfig.from_dict(validated_dict)

        logger.debug("Configuration validation completed")
        return validated_config


# Global orchestrator instance for backward compatibility
_global_orchestrator = ConfigurationOrchestrator()


def get_global_orchestrator() -> ConfigurationOrchestrator:
    """Get the global configuration orchestrator instance."""
    return _global_orchestrator


def load_config(
    environment: str = "development",
    profile: Optional[str] = None,
    validate: bool = True,
    use_cache: bool = True,
    strict_validation: bool = False,
    auto_repair: bool = False,
    config_dir: str = "farm/config",
) -> SimulationConfig:
    """
    Load configuration using the global orchestrator.

    This function provides backward compatibility with existing code
    that expects a simple load_config function.

    Args:
        environment: Environment name
        profile: Optional profile name
        validate: Whether to validate configuration
        use_cache: Whether to use caching
        strict_validation: Whether to treat warnings as errors
        auto_repair: Whether to attempt automatic repair
        config_dir: Configuration directory

    Returns:
        Loaded configuration
    """
    return _global_orchestrator.load_config(
        environment=environment,
        profile=profile,
        validate=validate,
        use_cache=use_cache,
        strict_validation=strict_validation,
        auto_repair=auto_repair,
        config_dir=config_dir,
    )
