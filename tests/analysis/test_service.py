"""
Tests for analysis service layer.
"""

import pytest
from pathlib import Path
import time

from farm.analysis.core import BaseAnalysisModule
from farm.analysis.service import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisCache,
    AnalysisService,
    AnalysisSuiteResult,
)
from farm.analysis.exceptions import ModuleNotFoundError, ConfigurationError
from farm.analysis.registry import registry


def _stub_module(name: str, mock_data_processor):
    """Minimal analysis module with the given registry name."""

    class StubModule(BaseAnalysisModule):
        def __init__(self):
            super().__init__(name=name, description=f"stub {name}")

        def register_functions(self):
            def stub_func(df, ctx, **kwargs):
                return {"module": name}

            stub_func.__name__ = "stub_func"
            self._functions = {"stub_func": stub_func}
            self._groups = {"all": [stub_func]}

        def get_data_processor(self):
            return mock_data_processor

    return StubModule()


class TestAnalysisRequest:
    """Tests for AnalysisRequest."""

    def test_request_initialization(self, tmp_path):
        """Test request initialization."""
        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=tmp_path / "experiment",
            output_path=tmp_path / "output"
        )

        assert request.module_name == "test_module"
        assert isinstance(request.experiment_path, Path)
        assert isinstance(request.output_path, Path)
        assert request.group == "all"

    def test_string_path_conversion(self, tmp_path):
        """Test automatic conversion of string paths."""
        request = AnalysisRequest(
            module_name="test",
            experiment_path=str(tmp_path / "exp"),
            output_path=str(tmp_path / "out")
        )

        assert isinstance(request.experiment_path, Path)
        assert isinstance(request.output_path, Path)

    def test_to_dict(self, tmp_path):
        """Test serialization to dictionary."""
        request = AnalysisRequest(
            module_name="test",
            experiment_path=tmp_path / "exp",
            output_path=tmp_path / "out",
            metadata={'key': 'value'}
        )

        data = request.to_dict()
        assert data['module_name'] == 'test'
        assert 'experiment_path' in data
        assert data['metadata'] == {'key': 'value'}

    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation."""
        request1 = AnalysisRequest(
            module_name="test",
            experiment_path=tmp_path / "exp",
            output_path=tmp_path / "out"
        )

        request2 = AnalysisRequest(
            module_name="test",
            experiment_path=tmp_path / "exp",
            output_path=tmp_path / "out"
        )

        # Same parameters should produce same cache key
        assert request1.get_cache_key() == request2.get_cache_key()

        # Different parameters should produce different key
        request3 = AnalysisRequest(
            module_name="different",
            experiment_path=tmp_path / "exp",
            output_path=tmp_path / "out"
        )
        assert request1.get_cache_key() != request3.get_cache_key()


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_result_initialization(self, temp_output_dir, sample_simulation_data):
        """Test result initialization."""
        result = AnalysisResult(
            success=True,
            module_name="test",
            output_path=temp_output_dir,
            dataframe=sample_simulation_data
        )

        assert result.success
        assert result.module_name == "test"
        assert result.dataframe is sample_simulation_data
        assert not result.cache_hit

    def test_failed_result(self, temp_output_dir):
        """Test failed result."""
        result = AnalysisResult(
            success=False,
            module_name="test",
            output_path=temp_output_dir,
            error="Test error message"
        )

        assert not result.success
        assert result.error == "Test error message"

    def test_to_dict(self, temp_output_dir, sample_simulation_data):
        """Test serialization to dictionary."""
        result = AnalysisResult(
            success=True,
            module_name="test",
            output_path=temp_output_dir,
            dataframe=sample_simulation_data,
            execution_time=1.5
        )

        data = result.to_dict()
        assert data['success']
        assert data['module_name'] == 'test'
        assert data['execution_time'] == 1.5
        assert data['dataframe_shape'] == sample_simulation_data.shape

    def test_save_summary(self, temp_output_dir):
        """Test saving result summary."""
        result = AnalysisResult(
            success=True,
            module_name="test",
            output_path=temp_output_dir
        )

        summary_path = result.save_summary()
        assert summary_path.exists()
        assert summary_path.suffix == '.json'


class TestAnalysisCache:
    """Tests for AnalysisCache."""

    def test_cache_initialization(self, tmp_path):
        """Test cache initialization."""
        cache_dir = tmp_path / "cache"
        cache = AnalysisCache(cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_cache_miss(self, tmp_path):
        """Test cache miss."""
        cache = AnalysisCache(tmp_path / "cache")

        assert not cache.has("nonexistent_key")
        assert cache.get("nonexistent_key") is None

    def test_cache_put_and_get(self, tmp_path, sample_simulation_data):
        """Test storing and retrieving from cache."""
        cache = AnalysisCache(tmp_path / "cache")

        key = "test_key"
        output_path = tmp_path / "output"

        # Store in cache
        cache.put(key, output_path, sample_simulation_data)

        # Should now exist
        assert cache.has(key)

        # Retrieve from cache
        cached_output, cached_df = cache.get(key)
        assert cached_output == output_path
        assert cached_df.equals(sample_simulation_data)

    def test_cache_clear(self, tmp_path, sample_simulation_data):
        """Test clearing cache."""
        cache = AnalysisCache(tmp_path / "cache")

        # Add some entries
        cache.put("key1", tmp_path, sample_simulation_data)
        cache.put("key2", tmp_path, sample_simulation_data)

        # Clear cache
        count = cache.clear()
        assert count == 2

        # Should be gone
        assert not cache.has("key1")
        assert not cache.has("key2")


class TestAnalysisService:
    """Tests for AnalysisService."""

    def test_service_initialization(self, config_service_mock):
        """Test service initialization."""
        service = AnalysisService(
            config_service=config_service_mock,
            auto_register=False
        )

        assert service.config_service is config_service_mock
        assert service.cache is not None

    def test_validate_request_valid(self, config_service_mock, minimal_module, tmp_path):
        """Test validation of valid request."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output"
        )

        # Should not raise
        service.validate_request(request)

    def test_validate_request_invalid_module(self, config_service_mock, tmp_path):
        """Test validation fails for invalid module."""
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="nonexistent_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output"
        )

        with pytest.raises(ModuleNotFoundError):
            service.validate_request(request)

    def test_validate_request_missing_path(self, config_service_mock, minimal_module, tmp_path):
        """Test validation fails for missing experiment path."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=tmp_path / "nonexistent",
            output_path=tmp_path / "output"
        )

        with pytest.raises(ConfigurationError):
            service.validate_request(request)

    def test_validate_request_invalid_group(self, config_service_mock, minimal_module, tmp_path):
        """Test validation fails for invalid function group."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            group="nonexistent_group"
        )

        with pytest.raises(ConfigurationError):
            service.validate_request(request)

    def test_run_analysis(self, config_service_mock, minimal_module, tmp_path):
        """Test running analysis."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            enable_caching=False
        )

        result = service.run(request)

        assert result.success
        assert result.module_name == "test_module"
        assert result.execution_time > 0
        assert not result.cache_hit

    def test_run_analysis_with_caching(self, config_service_mock, minimal_module, tmp_path):
        """Test analysis with caching."""
        registry.register(minimal_module)
        service = AnalysisService(
            config_service=config_service_mock,
            cache_dir=tmp_path / "cache",
            auto_register=False
        )

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            enable_caching=True
        )

        # First run - should not hit cache
        result1 = service.run(request)
        assert not result1.cache_hit

        # Second run - should hit cache
        result2 = service.run(request)
        assert result2.cache_hit

    def test_run_analysis_force_refresh(self, config_service_mock, minimal_module, tmp_path):
        """Test force refresh bypasses cache."""
        registry.register(minimal_module)
        service = AnalysisService(
            config_service=config_service_mock,
            cache_dir=tmp_path / "cache",
            auto_register=False
        )

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output",
            enable_caching=True,
            force_refresh=True
        )

        # First run
        result1 = service.run(request)

        # Second run with force_refresh should not hit cache
        result2 = service.run(request)
        assert not result2.cache_hit

    def test_run_batch(self, config_service_mock, minimal_module, tmp_path):
        """Test running batch of analyses."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        requests = [
            AnalysisRequest(
                module_name="test_module",
                experiment_path=exp_path,
                output_path=tmp_path / f"output_{i}",
                enable_caching=False
            )
            for i in range(3)
        ]

        results = service.run_batch(requests)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_get_module_info(self, config_service_mock, minimal_module):
        """Test getting module info."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        info = service.get_module_info("test_module")

        assert info['name'] == 'test_module'
        assert 'description' in info
        assert 'functions' in info

    def test_list_modules(self, config_service_mock, minimal_module):
        """Test listing all modules."""
        registry.register(minimal_module)
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        modules = service.list_modules()

        assert len(modules) > 0
        assert any(m['name'] == 'test_module' for m in modules)

    def test_clear_cache(self, config_service_mock, minimal_module, tmp_path):
        """Test clearing service cache."""
        registry.register(minimal_module)
        service = AnalysisService(
            config_service=config_service_mock,
            cache_dir=tmp_path / "cache",
            auto_register=False
        )

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        # Run analysis to populate cache
        request = AnalysisRequest(
            module_name="test_module",
            experiment_path=exp_path,
            output_path=tmp_path / "output"
        )
        service.run(request)

        # Clear cache
        count = service.clear_cache()
        assert count >= 1

    def test_run_suite_named_modules_list(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        registry.register(_stub_module("alpha", mock_data_processor))
        registry.register(_stub_module("beta", mock_data_processor))
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        out_base = tmp_path / "suite_out"

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=out_base,
            modules=["alpha", "beta"],
            enable_caching=False,
            write_unified_summary=False,
        )

        assert isinstance(suite, AnalysisSuiteResult)
        assert suite.suite_name is None
        assert suite.modules_requested == ["alpha", "beta"]
        assert len(suite.results) == 2
        assert all(r.success for r in suite.results)
        assert suite.summary["success_count"] == 2
        assert suite.summary["all_successful"] is True
        assert "alpha" in suite.summary["per_module"]
        assert "beta" in suite.summary["per_module"]
        assert (out_base / "alpha").is_dir()
        assert (out_base / "beta").is_dir()

    def test_run_suite_builtin_system_dynamics(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        for mod_name in ("population", "resources", "temporal"):
            registry.register(_stub_module(mod_name, mock_data_processor))
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        out_base = tmp_path / "sd_out"

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=out_base,
            suite="system_dynamics",
            enable_caching=False,
            write_unified_summary=True,
            report_formats="markdown",
        )

        assert suite.suite_name == "system_dynamics"
        assert suite.modules_requested == ["population", "resources", "temporal"]
        assert suite.summary["suite"] == "system_dynamics"
        summary_file = out_base / "system_dynamics_suite_summary.json"
        assert summary_file.is_file()
        md_file = out_base / "system_dynamics_suite_report.md"
        assert md_file.is_file()
        assert "population" in md_file.read_text(encoding="utf-8")

    def test_run_suite_agent_behavior(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        for mod_name in ("actions", "agents", "spatial", "learning"):
            registry.register(_stub_module(mod_name, mock_data_processor))
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        out_base = tmp_path / "ab_out"

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=out_base,
            suite="agent_behavior",
            enable_caching=False,
            write_unified_summary=False,
        )

        assert suite.suite_name == "agent_behavior"
        assert len(suite.results) == 4
        assert suite.summary["success_count"] == 4

    def test_run_suite_full_empty_registry_raises(
        self, config_service_mock, tmp_path
    ):
        service = AnalysisService(config_service=config_service_mock, auto_register=False)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        with pytest.raises(ConfigurationError, match="No modules to run"):
            service.run_suite(
                experiment_path=exp_path,
                output_path=tmp_path / "out",
                suite="full",
                enable_caching=False,
                write_unified_summary=False,
            )

    def test_run_suite_modules_executed_tracks_actual_results(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        """modules_executed should reflect only the modules that actually ran."""
        registry.register(_stub_module("alpha", mock_data_processor))
        registry.register(_stub_module("beta", mock_data_processor))
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=tmp_path / "suite_out",
            modules=["alpha", "beta"],
            enable_caching=False,
            write_unified_summary=False,
        )

        # In the success case both lists must match
        assert suite.modules_requested == ["alpha", "beta"]
        assert suite.modules_executed == ["alpha", "beta"]
        assert suite.summary["modules_requested"] == ["alpha", "beta"]
        assert suite.summary["modules_executed"] == ["alpha", "beta"]

    def test_run_suite_fail_fast_modules_executed_is_shorter(
        self, config_service_mock, tmp_path
    ):
        """When fail_fast stops early, modules_executed must be a prefix of modules_requested."""
        from farm.analysis.core import SimpleDataProcessor

        def ok_processor(data, **kwargs):
            import pandas as pd
            return pd.DataFrame({"test": [1]})

        def bad_processor(data, **kwargs):
            raise RuntimeError("intentional data-processing failure")

        ok_proc = SimpleDataProcessor(ok_processor)
        bad_proc = SimpleDataProcessor(bad_processor)

        registry.register(_stub_module("alpha", ok_proc))

        class FailingModule(BaseAnalysisModule):
            def __init__(self):
                super().__init__(name="broken", description="always fails at data load")

            def register_functions(self):
                def noop(df, ctx, **kwargs):
                    return {}

                noop.__name__ = "noop"
                self._functions = {"noop": noop}
                self._groups = {"all": [noop]}

            def get_data_processor(self):
                return bad_proc

        registry.register(FailingModule())
        registry.register(_stub_module("gamma", ok_proc))

        service = AnalysisService(config_service=config_service_mock, auto_register=False)
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=tmp_path / "suite_out",
            modules=["alpha", "broken", "gamma"],
            enable_caching=False,
            write_unified_summary=False,
            fail_fast=True,
        )

        # Full planned list is preserved on modules_requested
        assert suite.modules_requested == ["alpha", "broken", "gamma"]
        # Only the modules that actually ran appear in modules_executed
        assert "gamma" not in suite.modules_executed
        assert suite.summary["modules_requested"] == ["alpha", "broken", "gamma"]
        assert "gamma" not in suite.summary["modules_executed"]

    def test_run_suite_custom_slug_deterministic(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        """Custom suite slugs should be deterministic hashes, not 'custom_suite'."""
        registry.register(_stub_module("alpha", mock_data_processor))
        registry.register(_stub_module("beta", mock_data_processor))
        service = AnalysisService(config_service=config_service_mock, auto_register=False)

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        suite = service.run_suite(
            experiment_path=exp_path,
            output_path=tmp_path / "suite_out",
            modules=["alpha", "beta"],
            enable_caching=False,
            write_unified_summary=True,
        )

        # Slug must not be the generic "custom_suite" literal
        slug = suite._default_slug()
        assert slug != "custom_suite"
        assert slug.startswith("custom_")
        # Same module combination produces the same slug (deterministic)
        assert slug == suite._default_slug()
        # Summary file uses the hash slug
        summary_files = list((tmp_path / "suite_out").glob("*_suite_summary.json"))
        assert len(summary_files) == 1
        assert summary_files[0].name.startswith("custom_")
        assert "custom_suite_suite_summary" not in summary_files[0].name

    def test_run_suite_different_module_combinations_different_slugs(
        self, config_service_mock, mock_data_processor, tmp_path
    ):
        """Two different module combinations must produce different slugs."""

        def _make_result(mods):
            results = [
                AnalysisResult(
                    module_name=m,
                    success=True,
                    output_path=tmp_path / m,
                    execution_time=0.0,
                )
                for m in mods
            ]
            return AnalysisSuiteResult(
                suite_name=None,
                modules_requested=mods,
                experiment_path=tmp_path,
                output_base=tmp_path / "out",
                results=results,
            )

        suite_ab = _make_result(["alpha", "beta"])
        suite_ac = _make_result(["alpha", "gamma"])
        assert suite_ab._default_slug() != suite_ac._default_slug()
