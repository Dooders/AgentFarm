"""
Tests for analysis service layer.
"""

import pytest
from pathlib import Path
import time

from farm.analysis.service import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisCache,
    AnalysisService
)
from farm.analysis.exceptions import ModuleNotFoundError, ConfigurationError
from farm.analysis.registry import registry


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
        assert result.cache_hit == False
    
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
        assert data['success'] == True
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
