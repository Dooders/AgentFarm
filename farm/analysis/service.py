"""
Analysis service layer.

Provides high-level API for running analysis with validation, caching, and progress tracking.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from farm.analysis.exceptions import ConfigurationError, ModuleNotFoundError
from farm.analysis.registry import get_module, get_module_names, register_modules
from farm.core.services import IConfigService
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisRequest:
    """Request object for analysis execution.

    Attributes:
        module_name: Name of the analysis module to run
        experiment_path: Path to experiment data
        output_path: Path to save analysis results
        group: Function group to execute (default: "all")
        processor_kwargs: Arguments for data processor
        analysis_kwargs: Arguments for specific analysis functions
        enable_caching: Whether to use cached results if available
        force_refresh: Force recomputation even if cache exists
        progress_callback: Optional callback for progress updates
        metadata: Additional metadata to attach to results
    """

    module_name: str
    experiment_path: Path
    output_path: Path
    group: str = "all"
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    analysis_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enable_caching: bool = True
    force_refresh: bool = False
    progress_callback: Optional[Callable[[str, float], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.experiment_path, str):
            self.experiment_path = Path(self.experiment_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "module_name": self.module_name,
            "experiment_path": str(self.experiment_path),
            "output_path": str(self.output_path),
            "group": self.group,
            "processor_kwargs": self.processor_kwargs,
            "analysis_kwargs": self.analysis_kwargs,
            "metadata": self.metadata,
        }

    def get_cache_key(self) -> str:
        """Generate cache key for this request.

        Returns:
            Hash key based on request parameters
        """
        # Create dict with parameters that affect output
        key_dict = {
            "module": self.module_name,
            "experiment": str(self.experiment_path.absolute()),
            "group": self.group,
            "processor_kwargs": self.processor_kwargs,
            "analysis_kwargs": self.analysis_kwargs,
        }

        # Create deterministic hash
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


@dataclass
class AnalysisResult:
    """Result object from analysis execution.

    Attributes:
        success: Whether analysis completed successfully
        module_name: Name of module that ran
        output_path: Path where results were saved
        dataframe: Processed DataFrame (if available)
        execution_time: Time taken to run analysis (seconds)
        error: Error message if failed
        metadata: Additional metadata from execution
        cache_hit: Whether result was loaded from cache
        timestamp: When analysis was run
    """

    success: bool
    module_name: str
    output_path: Path
    dataframe: Optional[pd.DataFrame] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "module_name": self.module_name,
            "output_path": str(self.output_path),
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat(),
            "dataframe_shape": (
                self.dataframe.shape if self.dataframe is not None and hasattr(self.dataframe, 'shape')
                else (len(self.dataframe),) if isinstance(self.dataframe, dict) and self.dataframe is not None
                else None
            ),
        }

    def save_summary(self, path: Optional[Path] = None) -> Path:
        """Save result summary to JSON file.

        Args:
            path: Path to save to (default: output_path/analysis_summary.json)

        Returns:
            Path where summary was saved
        """
        if path is None:
            path = self.output_path / "analysis_summary.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path


class AnalysisCache:
    """Simple file-based cache for analysis results."""

    def __init__(self, cache_dir: Path):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, cache_key: str) -> Path:
        """Get path to cached result.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def has(self, cache_key: str) -> bool:
        """Check if cached result exists.

        Args:
            cache_key: Cache key

        Returns:
            True if cache exists
        """
        return self.get_cache_path(cache_key).exists()

    def get(self, cache_key: str) -> Optional[tuple[Path, pd.DataFrame]]:
        """Get cached result.

        Args:
            cache_key: Cache key

        Returns:
            Tuple of (output_path, dataframe) or None if not found
        """
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            data = pd.read_pickle(cache_path)
            logger.info(f"Loaded result from cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def put(
        self, cache_key: str, output_path: Path, dataframe: Optional[pd.DataFrame]
    ) -> None:
        """Store result in cache.

        Args:
            cache_key: Cache key
            output_path: Path where results are stored
            dataframe: Processed DataFrame
        """
        cache_path = self.get_cache_path(cache_key)

        try:
            pd.to_pickle((output_path, dataframe), cache_path)
            logger.info(f"Cached result: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache result {cache_key}: {e}")

    def clear(self) -> int:
        """Clear all cached results.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached results")
        return count


class AnalysisService:
    """Service for running analysis modules with advanced features.

    Provides:
    - Request validation
    - Result caching
    - Progress tracking
    - Error handling
    - Metadata management
    """

    def __init__(
        self,
        config_service: IConfigService,
        cache_dir: Optional[Path] = None,
        auto_register: bool = True,
    ):
        """Initialize analysis service.

        Args:
            config_service: Configuration service
            cache_dir: Directory for caching results (default: .analysis_cache)
            auto_register: Whether to auto-register modules on init
        """
        self.config_service = config_service

        # Set up caching
        if cache_dir is None:
            cache_dir = Path.cwd() / ".analysis_cache"
        self.cache = AnalysisCache(cache_dir)

        # Register modules if requested
        if auto_register:
            register_modules(config_service=config_service)

        logger.info("AnalysisService initialized")

    def validate_request(self, request: AnalysisRequest) -> None:
        """Validate analysis request.

        Args:
            request: Request to validate

        Raises:
            ConfigurationError: If request is invalid
            ModuleNotFoundError: If module doesn't exist
        """
        # Check module exists
        available_modules = get_module_names()
        if request.module_name not in available_modules:
            raise ModuleNotFoundError(request.module_name, available_modules)

        # Check experiment path exists
        if not request.experiment_path.exists():
            raise ConfigurationError(
                f"Experiment path does not exist: {request.experiment_path}"
            )

        # Check function group is valid
        module = get_module(request.module_name)
        available_groups = module.get_function_groups()
        if request.group not in available_groups:
            raise ConfigurationError(
                f"Invalid function group '{request.group}'. "
                f"Available groups: {', '.join(available_groups)}"
            )

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        """Run analysis with full service features.

        Args:
            request: Analysis request

        Returns:
            Analysis result with metadata
        """
        import time

        start_time = time.time()

        # Validate request
        try:
            self.validate_request(request)
        except Exception as e:
            return AnalysisResult(
                success=False,
                module_name=request.module_name,
                output_path=request.output_path,
                error=str(e),
                execution_time=time.time() - start_time,
            )

        # Check cache
        cache_key = request.get_cache_key()
        if request.enable_caching and not request.force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                output_path, dataframe = cached
                logger.info(f"Using cached result for {request.module_name}")
                # For cached results, execution time includes request validation and cache retrieval
                # This is typically much faster than running the actual analysis
                return AnalysisResult(
                    success=True,
                    module_name=request.module_name,
                    output_path=output_path,
                    dataframe=dataframe,
                    execution_time=time.time() - start_time,
                    cache_hit=True,
                    metadata=request.metadata,
                )

        # Run analysis
        try:
            module = get_module(request.module_name)

            logger.info(
                f"Running {request.module_name} analysis " f"(group: {request.group})"
            )

            output_path, dataframe = module.run_analysis(
                experiment_path=request.experiment_path,
                output_path=request.output_path,
                group=request.group,
                processor_kwargs=request.processor_kwargs,
                analysis_kwargs=request.analysis_kwargs,
                progress_callback=request.progress_callback,
            )

            # Cache result if enabled
            if request.enable_caching:
                self.cache.put(cache_key, output_path, dataframe)

            execution_time = time.time() - start_time

            result = AnalysisResult(
                success=True,
                module_name=request.module_name,
                output_path=output_path,
                dataframe=dataframe,
                execution_time=execution_time,
                metadata=request.metadata,
                cache_hit=False,
            )

            # Save summary
            result.save_summary()

            logger.info(
                f"Analysis completed in {execution_time:.2f}s "
                f"({len(dataframe) if dataframe is not None else 0} records)"
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                success=False,
                module_name=request.module_name,
                output_path=request.output_path,
                error=str(e),
                execution_time=time.time() - start_time,
                metadata=request.metadata,
            )

    def run_batch(
        self, requests: List[AnalysisRequest], fail_fast: bool = False
    ) -> List[AnalysisResult]:
        """Run multiple analysis requests in batch.

        Args:
            requests: List of analysis requests
            fail_fast: If True, stop on first failure

        Returns:
            List of analysis results
        """
        results = []

        for i, request in enumerate(requests):
            logger.info(f"Running batch analysis {i+1}/{len(requests)}")

            result = self.run(request)
            results.append(result)

            if fail_fast and not result.success:
                logger.error(f"Batch analysis failed at request {i+1}")
                break

        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch analysis complete: {successful}/{len(results)} successful")

        return results

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get information about a specific module.

        Args:
            module_name: Module name

        Returns:
            Module information dictionary

        Raises:
            ModuleNotFoundError: If module doesn't exist
        """
        module = get_module(module_name)
        return module.get_info()

    def list_modules(self) -> List[Dict[str, str]]:
        """List all available modules.

        Returns:
            List of module info dictionaries
        """
        modules = []
        for name in get_module_names():
            module = get_module(name)
            modules.append(
                {
                    "name": module.name,
                    "description": module.description,
                    "supports_database": module.supports_database(),
                }
            )
        return modules

    def clear_cache(self) -> int:
        """Clear analysis cache.

        Returns:
            Number of cache entries cleared
        """
        return self.cache.clear()
