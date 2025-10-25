"""
Tests for Automated Insights Generation.

This module contains comprehensive tests for the automated insight generation
system that identifies patterns, anomalies, and significant findings.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import shutil

from farm.analysis.comparative.automated_insights import (
    AutomatedInsightGenerator,
    InsightType,
    InsightSeverity,
    Insight,
    InsightGenerationConfig
)
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestAutomatedInsightGenerator:
    """Test cases for AutomatedInsightGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = InsightGenerationConfig(
            enable_performance_insights=True,
            enable_anomaly_detection=True,
            enable_trend_analysis=True,
            enable_correlation_analysis=True,
            enable_optimization_suggestions=True,
            min_confidence_threshold=0.5,
            max_insights_per_type=10,
            enable_ml_insights=True
        )
        self.generator = AutomatedInsightGenerator(config=self.config)
        
        # Mock analysis result
        self.mock_analysis = Mock(spec=OrchestrationResult)
        self.mock_analysis.success = True
        self.mock_analysis.total_duration = 120.5
        self.mock_analysis.phase_results = [
            Mock(phase_name="statistical_analysis", duration=60.0, success=True),
            Mock(phase_name="ml_analysis", duration=60.5, success=True)
        ]
        self.mock_analysis.errors = []
        self.mock_analysis.warnings = ["Minor warning"]
        self.mock_analysis.summary = {
            "total_phases": 2,
            "success_rate": 1.0,
            "performance_metrics": {
                "cpu_usage": 85.5,
                "memory_usage": 70.2,
                "disk_io": 45.8
            }
        }
        
        # Mock simulation data
        self.mock_simulation_data = {
            "performance_metrics": {
                "cpu_usage": [85.5, 87.2, 89.1, 86.8, 88.5],
                "memory_usage": [70.2, 72.1, 74.5, 71.8, 73.2],
                "disk_io": [45.8, 47.2, 48.9, 46.5, 47.8]
            },
            "simulation_results": {
                "total_agents": 1000,
                "simulation_time": 3600,
                "success_rate": 0.95
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.config == self.config
        assert self.generator.insight_history == []
        assert self.generator.performance_baseline is None
        assert self.generator.anomaly_threshold == 0.1
        assert self.generator.correlation_threshold == 0.7
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        generator = AutomatedInsightGenerator()
        assert generator.config is not None
        assert generator.config.enable_performance_insights is True
        assert generator.config.enable_anomaly_detection is True
    
    @pytest.mark.asyncio
    async def test_generate_insights(self):
        """Test generating insights from analysis result."""
        insights = await self.generator.generate_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
        
        # Check that different types of insights are generated
        insight_types = {insight.insight_type for insight in insights}
        assert len(insight_types) > 1  # Should have multiple types
    
    @pytest.mark.asyncio
    async def test_generate_insights_without_simulation_data(self):
        """Test generating insights without simulation data."""
        insights = await self.generator.generate_insights(self.mock_analysis)
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_performance_insights(self):
        """Test generating performance insights."""
        insights = await self.generator._generate_performance_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.PERFORMANCE for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_anomaly_insights(self):
        """Test generating anomaly insights."""
        insights = await self.generator._generate_anomaly_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.ANOMALY for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_trend_insights(self):
        """Test generating trend insights."""
        insights = await self.generator._generate_trend_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.TREND for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_correlation_insights(self):
        """Test generating correlation insights."""
        insights = await self.generator._generate_correlation_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.CORRELATION for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_optimization_insights(self):
        """Test generating optimization insights."""
        insights = await self.generator._generate_optimization_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.OPTIMIZATION for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    @pytest.mark.asyncio
    async def test_generate_ml_insights(self):
        """Test generating ML insights."""
        insights = await self.generator._generate_ml_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type == InsightType.ML_INSIGHT for insight in insights)
        assert all(insight.confidence > 0.0 for insight in insights)
    
    def test_detect_data_anomalies(self):
        """Test detecting data anomalies."""
        anomalies = self.generator._detect_data_anomalies(self.mock_simulation_data)
        
        assert isinstance(anomalies, list)
        assert all(isinstance(anomaly, dict) for anomaly in anomalies)
    
    def test_detect_data_anomalies_with_ml(self):
        """Test detecting data anomalies with ML."""
        with patch('farm.analysis.comparative.automated_insights.ISOLATION_FOREST_AVAILABLE', True):
            anomalies = self.generator._detect_data_anomalies(self.mock_simulation_data)
            
            assert isinstance(anomalies, list)
            assert all(isinstance(anomaly, dict) for anomaly in anomalies)
    
    def test_detect_data_anomalies_without_ml(self):
        """Test detecting data anomalies without ML."""
        with patch('farm.analysis.comparative.automated_insights.ISOLATION_FOREST_AVAILABLE', False):
            anomalies = self.generator._detect_data_anomalies(self.mock_simulation_data)
            
            assert isinstance(anomalies, list)
            assert all(isinstance(anomaly, dict) for anomaly in anomalies)
    
    def test_analyze_performance_trends(self):
        """Test analyzing performance trends."""
        trends = self.generator._analyze_performance_trends(self.mock_simulation_data)
        
        assert isinstance(trends, list)
        assert all(isinstance(trend, dict) for trend in trends)
    
    def test_analyze_performance_trends_with_ml(self):
        """Test analyzing performance trends with ML."""
        with patch('farm.analysis.comparative.automated_insights.LINEAR_REGRESSION_AVAILABLE', True):
            trends = self.generator._analyze_performance_trends(self.mock_simulation_data)
            
            assert isinstance(trends, list)
            assert all(isinstance(trend, dict) for trend in trends)
    
    def test_analyze_performance_trends_without_ml(self):
        """Test analyzing performance trends without ML."""
        with patch('farm.analysis.comparative.automated_insights.LINEAR_REGRESSION_AVAILABLE', False):
            trends = self.generator._analyze_performance_trends(self.mock_simulation_data)
            
            assert isinstance(trends, list)
            assert all(isinstance(trend, dict) for trend in trends)
    
    def test_find_correlations(self):
        """Test finding correlations."""
        correlations = self.generator._find_correlations(self.mock_simulation_data)
        
        assert isinstance(correlations, list)
        assert all(isinstance(corr, dict) for corr in correlations)
    
    def test_find_correlations_with_ml(self):
        """Test finding correlations with ML."""
        with patch('farm.analysis.comparative.automated_insights.PEARSONR_AVAILABLE', True):
            correlations = self.generator._find_correlations(self.mock_simulation_data)
            
            assert isinstance(correlations, list)
            assert all(isinstance(corr, dict) for corr in correlations)
    
    def test_find_correlations_without_ml(self):
        """Test finding correlations without ML."""
        with patch('farm.analysis.comparative.automated_insights.PEARSONR_AVAILABLE', False):
            correlations = self.generator._find_correlations(self.mock_simulation_data)
            
            assert isinstance(correlations, list)
            assert all(isinstance(corr, dict) for corr in correlations)
    
    def test_identify_optimization_opportunities(self):
        """Test identifying optimization opportunities."""
        opportunities = self.generator._identify_optimization_opportunities(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert isinstance(opportunities, list)
        assert all(isinstance(opp, dict) for opp in opportunities)
    
    def test_identify_optimization_opportunities_with_ml(self):
        """Test identifying optimization opportunities with ML."""
        with patch('farm.analysis.comparative.automated_insights.KMEANS_AVAILABLE', True):
            opportunities = self.generator._identify_optimization_opportunities(
                self.mock_analysis,
                self.mock_simulation_data
            )
            
            assert isinstance(opportunities, list)
            assert all(isinstance(opp, dict) for opp in opportunities)
    
    def test_identify_optimization_opportunities_without_ml(self):
        """Test identifying optimization opportunities without ML."""
        with patch('farm.analysis.comparative.automated_insights.KMEANS_AVAILABLE', False):
            opportunities = self.generator._identify_optimization_opportunities(
                self.mock_analysis,
                self.mock_simulation_data
            )
            
            assert isinstance(opportunities, list)
            assert all(isinstance(opp, dict) for opp in opportunities)
    
    def test_calculate_insight_confidence(self):
        """Test calculating insight confidence."""
        # Test high confidence
        confidence = self.generator._calculate_insight_confidence(
            "performance",
            {"cpu_usage": 95.0, "memory_usage": 90.0},
            {"cpu_usage": 80.0, "memory_usage": 75.0}
        )
        assert confidence > 0.8
        
        # Test low confidence
        confidence = self.generator._calculate_insight_confidence(
            "performance",
            {"cpu_usage": 50.0, "memory_usage": 45.0},
            {"cpu_usage": 80.0, "memory_usage": 75.0}
        )
        assert confidence < 0.5
    
    def test_rank_insights(self):
        """Test ranking insights by importance."""
        insights = [
            Insight(
                insight_type=InsightType.PERFORMANCE,
                title="High CPU Usage",
                description="CPU usage is above 90%",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data={"cpu_usage": 95.0},
                recommendations=["Optimize CPU usage"],
                created_at=datetime.now()
            ),
            Insight(
                insight_type=InsightType.ANOMALY,
                title="Minor Anomaly",
                description="Small data anomaly detected",
                severity=InsightSeverity.LOW,
                confidence=0.3,
                data={"anomaly_score": 0.2},
                recommendations=["Monitor data quality"],
                created_at=datetime.now()
            )
        ]
        
        ranked = self.generator._rank_insights(insights)
        
        assert len(ranked) == 2
        assert ranked[0].severity == InsightSeverity.HIGH  # High severity first
        assert ranked[1].severity == InsightSeverity.LOW
    
    def test_filter_insights_by_confidence(self):
        """Test filtering insights by confidence threshold."""
        insights = [
            Insight(
                insight_type=InsightType.PERFORMANCE,
                title="High Confidence",
                description="High confidence insight",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data={},
                recommendations=[],
                created_at=datetime.now()
            ),
            Insight(
                insight_type=InsightType.ANOMALY,
                title="Low Confidence",
                description="Low confidence insight",
                severity=InsightSeverity.LOW,
                confidence=0.3,
                data={},
                recommendations=[],
                created_at=datetime.now()
            )
        ]
        
        filtered = self.generator._filter_insights_by_confidence(insights, 0.5)
        
        assert len(filtered) == 1
        assert filtered[0].confidence > 0.5
    
    def test_limit_insights_per_type(self):
        """Test limiting insights per type."""
        insights = []
        for i in range(15):  # More than max_insights_per_type (10)
            insights.append(Insight(
                insight_type=InsightType.PERFORMANCE,
                title=f"Performance Insight {i}",
                description=f"Performance insight {i}",
                severity=InsightSeverity.MEDIUM,
                confidence=0.7,
                data={},
                recommendations=[],
                created_at=datetime.now()
            ))
        
        limited = self.generator._limit_insights_per_type(insights)
        
        assert len(limited) <= 10  # Should be limited to max_insights_per_type
    
    def test_update_performance_baseline(self):
        """Test updating performance baseline."""
        # Initial baseline should be None
        assert self.generator.performance_baseline is None
        
        # Update baseline
        self.generator._update_performance_baseline(self.mock_simulation_data)
        
        assert self.generator.performance_baseline is not None
        assert "cpu_usage" in self.generator.performance_baseline
        assert "memory_usage" in self.generator.performance_baseline
    
    def test_get_insight_history(self):
        """Test getting insight history."""
        # Add some insights to history
        self.generator.insight_history = [
            {"timestamp": "2023-01-01T00:00:00", "insight_count": 5},
            {"timestamp": "2023-01-02T00:00:00", "insight_count": 3}
        ]
        
        history = self.generator.get_insight_history()
        assert len(history) == 2
        assert history[0]["insight_count"] == 5
    
    def test_clear_insight_history(self):
        """Test clearing insight history."""
        # Add some insights to history
        self.generator.insight_history = [
            {"timestamp": "2023-01-01T00:00:00", "insight_count": 5}
        ]
        
        # Clear history
        self.generator.clear_insight_history()
        assert len(self.generator.insight_history) == 0
    
    def test_get_insight_statistics(self):
        """Test getting insight statistics."""
        # Add some insights to history
        self.generator.insight_history = [
            {"timestamp": "2023-01-01T00:00:00", "insight_count": 5, "types": ["performance", "anomaly"]},
            {"timestamp": "2023-01-02T00:00:00", "insight_count": 3, "types": ["trend", "correlation"]}
        ]
        
        stats = self.generator.get_insight_statistics()
        
        assert "total_insights" in stats
        assert "insights_per_day" in stats
        assert "most_common_type" in stats
        assert stats["total_insights"] == 8
    
    def test_insight_creation(self):
        """Test Insight creation."""
        insight = Insight(
            insight_type=InsightType.PERFORMANCE,
            title="Test Insight",
            description="Test description",
            severity=InsightSeverity.HIGH,
            confidence=0.8,
            data={"key": "value"},
            recommendations=["rec1", "rec2"],
            created_at=datetime.now()
        )
        
        assert insight.insight_type == InsightType.PERFORMANCE
        assert insight.title == "Test Insight"
        assert insight.description == "Test description"
        assert insight.severity == InsightSeverity.HIGH
        assert insight.confidence == 0.8
        assert insight.data == {"key": "value"}
        assert insight.recommendations == ["rec1", "rec2"]
    
    def test_insight_generation_config_creation(self):
        """Test InsightGenerationConfig creation."""
        config = InsightGenerationConfig(
            enable_performance_insights=True,
            enable_anomaly_detection=False,
            enable_trend_analysis=True,
            enable_correlation_analysis=False,
            enable_optimization_suggestions=True,
            min_confidence_threshold=0.6,
            max_insights_per_type=5,
            enable_ml_insights=False
        )
        
        assert config.enable_performance_insights is True
        assert config.enable_anomaly_detection is False
        assert config.enable_trend_analysis is True
        assert config.enable_correlation_analysis is False
        assert config.enable_optimization_suggestions is True
        assert config.min_confidence_threshold == 0.6
        assert config.max_insights_per_type == 5
        assert config.enable_ml_insights is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in insight generation."""
        # Test with invalid analysis result
        invalid_analysis = Mock(spec=OrchestrationResult)
        invalid_analysis.success = False
        invalid_analysis.errors = ["Test error"]
        invalid_analysis.phase_results = []
        invalid_analysis.summary = {}
        
        insights = await self.generator.generate_insights(invalid_analysis)
        
        # Should still generate some insights even with errors
        assert len(insights) >= 0
        assert all(isinstance(insight, Insight) for insight in insights)
    
    def test_insight_type_enum(self):
        """Test InsightType enum values."""
        assert InsightType.PERFORMANCE == "performance"
        assert InsightType.ANOMALY == "anomaly"
        assert InsightType.TREND == "trend"
        assert InsightType.CORRELATION == "correlation"
        assert InsightType.OPTIMIZATION == "optimization"
        assert InsightType.ML_INSIGHT == "ml_insight"
    
    def test_insight_severity_enum(self):
        """Test InsightSeverity enum values."""
        assert InsightSeverity.LOW == "low"
        assert InsightSeverity.MEDIUM == "medium"
        assert InsightSeverity.HIGH == "high"
        assert InsightSeverity.CRITICAL == "critical"