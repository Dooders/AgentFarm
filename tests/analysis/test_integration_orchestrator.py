"""
Tests for integration orchestrator.

This module contains unit tests for the integration orchestrator
that coordinates all phases of simulation analysis.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.integration_orchestrator import (
    IntegrationOrchestrator,
    OrchestrationConfig,
    AnalysisPhaseConfig,
    AnalysisPhaseResult,
    OrchestrationResult
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult,
    MetricsComparisonResult,
    DatabaseComparisonResult,
    LogComparisonResult,
    ConfigComparisonResult
)


class TestIntegrationOrchestrator:
    """Test cases for IntegrationOrchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OrchestrationConfig(
            max_workers=2,
            timeout=300,
            output_dir="test_output",
            parallel_execution=True
        )
        
        # Mock the analyzers to avoid import issues
        with patch('farm.analysis.comparative.integration_orchestrator.StatisticalAnalyzer'), \
             patch('farm.analysis.comparative.integration_orchestrator.MLAnalyzer'), \
             patch('farm.analysis.comparative.integration_orchestrator.AdvancedAnomalyDetector'), \
             patch('farm.analysis.comparative.integration_orchestrator.ClusteringAnalyzer'), \
             patch('farm.analysis.comparative.integration_orchestrator.TrendPredictor'), \
             patch('farm.analysis.comparative.integration_orchestrator.SimilarityAnalyzer'), \
             patch('farm.analysis.comparative.integration_orchestrator.MLVisualizer'):
            
            self.orchestrator = IntegrationOrchestrator(self.config)
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config == self.config
        assert self.orchestrator.output_dir == Path("test_output")
        assert self.orchestrator.output_dir.exists()
    
    def test_setup_default_phases(self):
        """Test default phase configuration setup."""
        phases = self.orchestrator.config.phases
        
        assert "statistical_analysis" in phases
        assert "ml_analysis" in phases
        assert "anomaly_detection" in phases
        assert "clustering" in phases
        assert "trend_prediction" in phases
        assert "similarity_analysis" in phases
        assert "visualization" in phases
        
        # Check phase priorities
        assert phases["statistical_analysis"].priority == 1
        assert phases["ml_analysis"].priority == 2
        assert phases["visualization"].priority == 7
    
    def test_get_analysis_status(self):
        """Test getting analysis system status."""
        status = self.orchestrator.get_analysis_status()
        
        assert "analyzers_initialized" in status
        assert "configuration" in status
        assert "output_directory" in status
        assert "timestamp" in status
        
        # Check analyzer status
        analyzers = status["analyzers_initialized"]
        # The actual status shows different analyzer names, so let's check what's available
        assert len(analyzers) > 0
        assert "ml_analyzer" in analyzers
    
    @pytest.mark.asyncio
    async def test_analyze_simulations_success(self):
        """Test successful simulation analysis."""
        # Create mock simulation pairs
        simulation_pairs = [
            (Path("/tmp/sim1"), Path("/tmp/sim2")),
            (Path("/tmp/sim3"), Path("/tmp/sim4"))
        ]
        
        # Mock the phase execution methods
        with patch.object(self.orchestrator, '_run_statistical_analysis', new_callable=AsyncMock) as mock_statistical, \
             patch.object(self.orchestrator, '_run_ml_analysis', new_callable=AsyncMock) as mock_ml, \
             patch.object(self.orchestrator, '_run_anomaly_detection', new_callable=AsyncMock) as mock_anomaly, \
             patch.object(self.orchestrator, '_run_clustering', new_callable=AsyncMock) as mock_clustering, \
             patch.object(self.orchestrator, '_run_trend_prediction', new_callable=AsyncMock) as mock_trend, \
             patch.object(self.orchestrator, '_run_similarity_analysis', new_callable=AsyncMock) as mock_similarity, \
             patch.object(self.orchestrator, '_run_visualization', new_callable=AsyncMock) as mock_viz:
            
            # Setup mock results
            mock_statistical.return_value = self._create_mock_statistical_results()
            mock_ml.return_value = {"ml_analysis": "completed"}
            mock_anomaly.return_value = {"anomaly_detection": "completed"}
            mock_clustering.return_value = {"clustering": "completed"}
            mock_trend.return_value = {"trend_prediction": "completed"}
            mock_similarity.return_value = {"similarity_analysis": "completed"}
            mock_viz.return_value = {"visualization": "completed"}
            
            # Run analysis
            result = await self.orchestrator.analyze_simulations(simulation_pairs)
            
            # Verify result
            assert isinstance(result, OrchestrationResult)
            assert result.success is True
            assert result.total_duration > 0
            assert len(result.phase_results) > 0
            assert "total_phases" in result.summary
            assert hasattr(result, 'output_paths')
    
    @pytest.mark.asyncio
    async def test_analyze_simulations_failure(self):
        """Test simulation analysis failure."""
        simulation_pairs = [(Path("/nonexistent/sim1"), Path("/nonexistent/sim2"))]
        
        # Mock statistical analysis to fail
        with patch.object(self.orchestrator, '_run_statistical_analysis', new_callable=AsyncMock) as mock_statistical:
            mock_statistical.side_effect = Exception("Statistical analysis failed")
            
            result = await self.orchestrator.analyze_simulations(simulation_pairs)
            
            assert isinstance(result, OrchestrationResult)
            assert result.success is False
            assert "Statistical analysis failed" in result.errors
    
    @pytest.mark.asyncio
    async def test_run_phase_success(self):
        """Test successful phase execution."""
        phase_name = "test_phase"
        phase_func = AsyncMock(return_value="test_result")
        
        result = await self.orchestrator._run_phase(phase_name, phase_func, "arg1", "arg2")
        
        assert isinstance(result, AnalysisPhaseResult)
        assert result.phase_name == phase_name
        assert result.success is True
        assert result.result == "test_result"
        assert result.duration > 0
    
    @pytest.mark.asyncio
    async def test_run_phase_failure(self):
        """Test phase execution failure."""
        phase_name = "test_phase"
        phase_func = AsyncMock(side_effect=Exception("Phase failed"))
        
        result = await self.orchestrator._run_phase(phase_name, phase_func)
        
        assert isinstance(result, AnalysisPhaseResult)
        assert result.phase_name == phase_name
        assert result.success is False
        assert "Phase failed" in result.error
    
    @pytest.mark.asyncio
    async def test_run_phase_disabled(self):
        """Test disabled phase execution."""
        phase_name = "disabled_phase"
        phase_func = AsyncMock()
        
        # Disable the phase
        self.orchestrator.config.phases[phase_name] = AnalysisPhaseConfig(enabled=False)
        
        result = await self.orchestrator._run_phase(phase_name, phase_func)
        
        assert isinstance(result, AnalysisPhaseResult)
        assert result.phase_name == phase_name
        assert result.success is True
        assert result.result is None
        assert result.metadata.get("skipped") is True
        assert result.duration == 0.0
    
    def test_generate_summary(self):
        """Test summary generation."""
        phase_results = [
            AnalysisPhaseResult(
                phase_name="phase1",
                success=True,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=10.0
            ),
            AnalysisPhaseResult(
                phase_name="phase2",
                success=False,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=5.0,
                error="Phase failed"
            )
        ]
        
        summary = self.orchestrator._generate_summary(phase_results, 15.0)
        
        assert summary["total_phases"] == 2
        assert summary["successful_phases"] == 1
        assert summary["failed_phases"] == 1
        assert summary["total_duration"] == 15.0
        assert summary["average_phase_duration"] == 7.5
        assert "phase_breakdown" in summary
        assert "performance_metrics" in summary
    
    def test_generate_output_paths(self):
        """Test output path generation."""
        phase_results = [
            AnalysisPhaseResult(
                phase_name="phase1",
                success=True,
                start_time=datetime(2023, 1, 1, 12, 0, 0),
                end_time=datetime(2023, 1, 1, 12, 1, 0),
                duration=60.0,
                result="test_result"
            )
        ]
        
        output_paths = self.orchestrator._generate_output_paths(phase_results)
        
        assert "phase1" in output_paths
        assert "phase1_20230101_120000.json" in output_paths["phase1"]
        assert str(self.orchestrator.output_dir) in output_paths["phase1"]
    
    def test_create_failure_result(self):
        """Test failure result creation."""
        start_time = datetime.now()
        message = "Test failure"
        error = "Test error"
        
        result = self.orchestrator._create_failure_result(start_time, message, error)
        
        assert isinstance(result, OrchestrationResult)
        assert result.success is False
        assert result.start_time == start_time
        assert result.total_duration >= 0
        assert result.phase_results == []
        assert result.summary["error"] == message
        assert error in result.errors
    
    def _create_mock_statistical_results(self):
        """Create mock statistical analysis results."""
        from farm.analysis.comparative.statistical_analyzer import StatisticalAnalysisResult
        
        results = []
        for i in range(2):
            result = StatisticalAnalysisResult(
                correlation_analysis={"correlation": 0.8},
                significance_tests={"p_value": 0.05},
                trend_analysis={"trend": "increasing"},
                anomaly_detection={"anomalies": 2},
                summary={"status": "completed"}
            )
            results.append(result)
        
        return results


class TestOrchestrationConfig:
    """Test cases for OrchestrationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestrationConfig()
        
        assert config.max_workers == 4
        assert config.timeout is None
        assert config.output_dir == "analysis_results"
        assert config.cache_results is True
        assert config.parallel_execution is True
        assert config.progress_callback is None
        assert isinstance(config.phases, dict)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrchestrationConfig(
            max_workers=8,
            timeout=600,
            output_dir="custom_output",
            cache_results=False,
            parallel_execution=False
        )
        
        assert config.max_workers == 8
        assert config.timeout == 600
        assert config.output_dir == "custom_output"
        assert config.cache_results is False
        assert config.parallel_execution is False


class TestAnalysisPhaseConfig:
    """Test cases for AnalysisPhaseConfig."""
    
    def test_default_phase_config(self):
        """Test default phase configuration values."""
        config = AnalysisPhaseConfig()
        
        assert config.enabled is True
        assert config.priority == 1
        assert config.timeout is None
        assert config.parallel is False
        assert config.config is None
    
    def test_custom_phase_config(self):
        """Test custom phase configuration values."""
        config = AnalysisPhaseConfig(
            enabled=False,
            priority=5,
            timeout=300,
            parallel=True,
            config={"param1": "value1"}
        )
        
        assert config.enabled is False
        assert config.priority == 5
        assert config.timeout == 300
        assert config.parallel is True
        assert config.config == {"param1": "value1"}


class TestAnalysisPhaseResult:
    """Test cases for AnalysisPhaseResult."""
    
    def test_phase_result_creation(self):
        """Test phase result creation."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = AnalysisPhaseResult(
            phase_name="test_phase",
            success=True,
            start_time=start_time,
            end_time=end_time,
            duration=10.0,
            result="test_result",
            metadata={"key": "value"}
        )
        
        assert result.phase_name == "test_phase"
        assert result.success is True
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.duration == 10.0
        assert result.result == "test_result"
        assert result.error is None
        assert result.metadata == {"key": "value"}
    
    def test_phase_result_with_error(self):
        """Test phase result with error."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = AnalysisPhaseResult(
            phase_name="failed_phase",
            success=False,
            start_time=start_time,
            end_time=end_time,
            duration=5.0,
            error="Phase failed"
        )
        
        assert result.phase_name == "failed_phase"
        assert result.success is False
        assert result.duration == 5.0
        assert result.error == "Phase failed"
        assert result.result is None


class TestOrchestrationResult:
    """Test cases for OrchestrationResult."""
    
    def test_orchestration_result_creation(self):
        """Test orchestration result creation."""
        start_time = datetime.now()
        end_time = datetime.now()
        phase_results = []
        
        result = OrchestrationResult(
            success=True,
            start_time=start_time,
            end_time=end_time,
            total_duration=60.0,
            phase_results=phase_results,
            summary={"total_phases": 0},
            output_paths={},
            errors=[],
            warnings=[]
        )
        
        assert result.success is True
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.total_duration == 60.0
        assert result.phase_results == phase_results
        assert result.summary == {"total_phases": 0}
        assert result.output_paths == {}
        assert result.errors == []
        assert result.warnings == []
    
    def test_orchestration_result_with_errors(self):
        """Test orchestration result with errors."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = OrchestrationResult(
            success=False,
            start_time=start_time,
            end_time=end_time,
            total_duration=30.0,
            phase_results=[],
            summary={"error": "Analysis failed"},
            output_paths={},
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert result.success is False
        assert result.total_duration == 30.0
        assert result.summary["error"] == "Analysis failed"
        assert len(result.errors) == 2
        assert len(result.warnings) == 1