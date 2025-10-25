"""
Smart recommendation engine with context awareness.

This module provides intelligent recommendations based on analysis results,
user context, and historical patterns.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import statistics
import math

# Optional imports for advanced analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity

logger = get_logger(__name__)


class RecommendationType(Enum):
    """Types of recommendations."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    QUALITY_IMPROVEMENT = "quality_improvement"
    ERROR_RESOLUTION = "error_resolution"
    CONFIGURATION_ADJUSTMENT = "configuration_adjustment"
    WORKFLOW_ENHANCEMENT = "workflow_enhancement"
    RESOURCE_MANAGEMENT = "resource_management"
    ANALYSIS_DEEPENING = "analysis_deepening"
    MONITORING_SETUP = "monitoring_setup"
    DOCUMENTATION_UPDATE = "documentation_update"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserContext(Enum):
    """User context types."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Recommendation:
    """A smart recommendation with context and metadata."""
    
    id: str
    type: RecommendationType
    title: str
    description: str
    priority: RecommendationPriority
    confidence: float
    context: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    expected_impact: str = ""
    implementation_effort: str = ""
    time_estimate: str = ""
    resources_needed: List[str] = field(default_factory=list)
    related_insights: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationConfig:
    """Configuration for the recommendation engine."""
    
    # User context
    user_context: UserContext = UserContext.INTERMEDIATE
    include_advanced_recommendations: bool = True
    include_beginner_recommendations: bool = True
    
    # Recommendation filtering
    min_confidence: float = 0.6
    max_recommendations: int = 20
    prioritize_by_impact: bool = True
    
    # Context awareness
    consider_historical_patterns: bool = True
    consider_resource_constraints: bool = True
    consider_time_constraints: bool = True
    
    # Learning and adaptation
    enable_learning: bool = True
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.7


class SmartRecommendationEngine:
    """Smart recommendation engine with context awareness."""
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        """Initialize the recommendation engine."""
        self.config = config or RecommendationConfig()
        self.recommendations: List[Recommendation] = []
        self.user_history: List[Dict[str, Any]] = []
        self.learning_data: Dict[str, Any] = {}
        
        # Load historical data
        self._load_historical_data()
        
        logger.info("SmartRecommendationEngine initialized")
    
    async def generate_recommendations(self,
                                     analysis_result: OrchestrationResult,
                                     insights: List[Insight],
                                     user_context: Optional[Dict[str, Any]] = None) -> List[Recommendation]:
        """Generate smart recommendations based on analysis results and insights."""
        logger.info("Generating smart recommendations")
        
        self.recommendations = []
        context = user_context or {}
        
        # Generate recommendations based on analysis results
        await self._generate_performance_recommendations(analysis_result, insights, context)
        await self._generate_quality_recommendations(analysis_result, insights, context)
        await self._generate_error_recommendations(analysis_result, insights, context)
        await self._generate_configuration_recommendations(analysis_result, insights, context)
        await self._generate_workflow_recommendations(analysis_result, insights, context)
        await self._generate_resource_recommendations(analysis_result, insights, context)
        await self._generate_analysis_recommendations(analysis_result, insights, context)
        await self._generate_monitoring_recommendations(analysis_result, insights, context)
        await self._generate_documentation_recommendations(analysis_result, insights, context)
        await self._generate_automation_recommendations(analysis_result, insights, context)
        
        # Apply context-aware filtering and ranking
        self.recommendations = self._apply_context_filtering(self.recommendations, context)
        self.recommendations = self._rank_recommendations(self.recommendations, context)
        
        # Update learning data
        if self.config.enable_learning:
            self._update_learning_data(analysis_result, insights, context)
        
        logger.info(f"Generated {len(self.recommendations)} recommendations")
        return self.recommendations
    
    async def _generate_performance_recommendations(self,
                                                  analysis_result: OrchestrationResult,
                                                  insights: List[Insight],
                                                  context: Dict[str, Any]):
        """Generate performance optimization recommendations."""
        duration = analysis_result.total_duration
        phase_count = len(analysis_result.phase_results)
        
        # Long analysis duration
        if duration > 300:  # 5 minutes
            rec = Recommendation(
                id=f"perf_duration_{len(self.recommendations)}",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize Analysis Duration",
                description=f"Analysis took {duration:.2f} seconds. Consider performance optimizations to reduce execution time.",
                priority=RecommendationPriority.HIGH if duration > 600 else RecommendationPriority.MEDIUM,
                confidence=0.9,
                context=["long_duration", "performance"],
                expected_impact="Reduce analysis time by 30-50%",
                implementation_effort="Medium",
                time_estimate="2-4 hours",
                resources_needed=["Performance profiling tools", "Parallel processing framework"],
                tags=["performance", "optimization", "duration"]
            )
            self.recommendations.append(rec)
        
        # Multiple phases - parallel execution opportunity
        if phase_count > 2:
            rec = Recommendation(
                id=f"perf_parallel_{len(self.recommendations)}",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Implement Parallel Phase Execution",
                description=f"Analysis has {phase_count} phases. Independent phases can be executed in parallel for better performance.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["multiple_phases", "parallel_execution"],
                expected_impact="Reduce total analysis time by 40-60%",
                implementation_effort="High",
                time_estimate="1-2 days",
                resources_needed=["Concurrent execution framework", "Dependency analysis tools"],
                tags=["performance", "parallel", "phases"]
            )
            self.recommendations.append(rec)
        
        # Check for performance-related insights
        perf_insights = [i for i in insights if i.type == InsightType.PERFORMANCE_PATTERN]
        if perf_insights:
            rec = Recommendation(
                id=f"perf_insights_{len(self.recommendations)}",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Address Performance Patterns",
                description=f"Found {len(perf_insights)} performance-related insights. Review and address these patterns for better performance.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["performance_insights", "pattern_analysis"],
                expected_impact="Improve overall system performance",
                implementation_effort="Medium",
                time_estimate="4-8 hours",
                resources_needed=["Performance analysis tools", "Code profiling"],
                related_insights=[i.id for i in perf_insights],
                tags=["performance", "insights", "patterns"]
            )
            self.recommendations.append(rec)
    
    async def _generate_quality_recommendations(self,
                                              analysis_result: OrchestrationResult,
                                              insights: List[Insight],
                                              context: Dict[str, Any]):
        """Generate quality improvement recommendations."""
        errors = analysis_result.errors
        warnings = analysis_result.warnings
        
        # High error count
        if len(errors) > 5:
            rec = Recommendation(
                id=f"quality_errors_{len(self.recommendations)}",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Reduce Analysis Errors",
                description=f"Analysis generated {len(errors)} errors. Focus on error reduction to improve analysis quality.",
                priority=RecommendationPriority.CRITICAL,
                confidence=0.95,
                context=["high_error_count", "quality_issues"],
                expected_impact="Eliminate 80-90% of analysis errors",
                implementation_effort="High",
                time_estimate="1-3 days",
                resources_needed=["Error analysis tools", "Input validation framework"],
                tags=["quality", "errors", "reliability"]
            )
            self.recommendations.append(rec)
        
        # High warning count
        if len(warnings) > 10:
            rec = Recommendation(
                id=f"quality_warnings_{len(self.recommendations)}",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Address Analysis Warnings",
                description=f"Analysis generated {len(warnings)} warnings. Addressing warnings improves analysis reliability.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["high_warning_count", "quality_improvement"],
                expected_impact="Improve analysis reliability and consistency",
                implementation_effort="Medium",
                time_estimate="4-8 hours",
                resources_needed=["Warning analysis tools", "Code review process"],
                tags=["quality", "warnings", "reliability"]
            )
            self.recommendations.append(rec)
        
        # Quality-related insights
        quality_insights = [i for i in insights if i.type == InsightType.QUALITY_ASSESSMENT]
        if quality_insights:
            rec = Recommendation(
                id=f"quality_insights_{len(self.recommendations)}",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Implement Quality Improvements",
                description=f"Found {len(quality_insights)} quality-related insights. Implement suggested improvements for better analysis quality.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["quality_insights", "improvement"],
                expected_impact="Improve overall analysis quality score",
                implementation_effort="Medium",
                time_estimate="6-12 hours",
                resources_needed=["Quality metrics framework", "Testing tools"],
                related_insights=[i.id for i in quality_insights],
                tags=["quality", "insights", "improvement"]
            )
            self.recommendations.append(rec)
    
    async def _generate_error_recommendations(self,
                                            analysis_result: OrchestrationResult,
                                            insights: List[Insight],
                                            context: Dict[str, Any]):
        """Generate error resolution recommendations."""
        errors = analysis_result.errors
        
        if not errors:
            return
        
        # Categorize errors
        error_categories = self._categorize_errors(errors)
        
        for category, error_list in error_categories.items():
            if category == "configuration":
                rec = Recommendation(
                    id=f"error_config_{len(self.recommendations)}",
                    type=RecommendationType.ERROR_RESOLUTION,
                    title="Fix Configuration Errors",
                    description=f"Found {len(error_list)} configuration-related errors. Review and update analysis configuration.",
                    priority=RecommendationPriority.HIGH,
                    confidence=0.9,
                    context=["configuration_errors", "error_resolution"],
                    expected_impact="Resolve configuration issues and improve analysis success rate",
                    implementation_effort="Low",
                    time_estimate="1-2 hours",
                    resources_needed=["Configuration validation tools", "Documentation"],
                    tags=["errors", "configuration", "resolution"]
                )
                self.recommendations.append(rec)
            
            elif category == "data":
                rec = Recommendation(
                    id=f"error_data_{len(self.recommendations)}",
                    type=RecommendationType.ERROR_RESOLUTION,
                    title="Fix Data-Related Errors",
                    description=f"Found {len(error_list)} data-related errors. Check data quality and format.",
                    priority=RecommendationPriority.HIGH,
                    confidence=0.9,
                    context=["data_errors", "error_resolution"],
                    expected_impact="Resolve data issues and improve analysis reliability",
                    implementation_effort="Medium",
                    time_estimate="2-4 hours",
                    resources_needed=["Data validation tools", "Data cleaning utilities"],
                    tags=["errors", "data", "resolution"]
                )
                self.recommendations.append(rec)
            
            elif category == "system":
                rec = Recommendation(
                    id=f"error_system_{len(self.recommendations)}",
                    type=RecommendationType.ERROR_RESOLUTION,
                    title="Resolve System Errors",
                    description=f"Found {len(error_list)} system-related errors. Check system resources and dependencies.",
                    priority=RecommendationPriority.CRITICAL,
                    confidence=0.9,
                    context=["system_errors", "error_resolution"],
                    expected_impact="Resolve system issues and ensure stable analysis execution",
                    implementation_effort="High",
                    time_estimate="4-8 hours",
                    resources_needed=["System monitoring tools", "Dependency management"],
                    tags=["errors", "system", "resolution"]
                )
                self.recommendations.append(rec)
    
    def _categorize_errors(self, errors: List[str]) -> Dict[str, List[str]]:
        """Categorize errors by type."""
        categories = {
            "configuration": [],
            "data": [],
            "system": [],
            "other": []
        }
        
        for error in errors:
            error_lower = error.lower()
            if any(keyword in error_lower for keyword in ["config", "setting", "parameter", "option"]):
                categories["configuration"].append(error)
            elif any(keyword in error_lower for keyword in ["data", "file", "input", "format"]):
                categories["data"].append(error)
            elif any(keyword in error_lower for keyword in ["memory", "disk", "network", "resource", "timeout"]):
                categories["system"].append(error)
            else:
                categories["other"].append(error)
        
        return {k: v for k, v in categories.items() if v}
    
    async def _generate_configuration_recommendations(self,
                                                    analysis_result: OrchestrationResult,
                                                    insights: List[Insight],
                                                    context: Dict[str, Any]):
        """Generate configuration adjustment recommendations."""
        # Check for configuration-related insights
        config_insights = [i for i in insights if "config" in i.tags or "configuration" in i.tags]
        
        if config_insights:
            rec = Recommendation(
                id=f"config_insights_{len(self.recommendations)}",
                type=RecommendationType.CONFIGURATION_ADJUSTMENT,
                title="Optimize Analysis Configuration",
                description=f"Found {len(config_insights)} configuration-related insights. Review and optimize analysis settings.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["configuration_insights", "optimization"],
                expected_impact="Improve analysis efficiency and reliability",
                implementation_effort="Low",
                time_estimate="2-4 hours",
                resources_needed=["Configuration management tools", "Performance testing"],
                related_insights=[i.id for i in config_insights],
                tags=["configuration", "optimization", "settings"]
            )
            self.recommendations.append(rec)
        
        # Check for resource-related configuration issues
        if analysis_result.total_duration > 600:  # 10 minutes
            rec = Recommendation(
                id=f"config_resources_{len(self.recommendations)}",
                type=RecommendationType.CONFIGURATION_ADJUSTMENT,
                title="Adjust Resource Configuration",
                description="Long analysis duration suggests resource configuration may need adjustment for better performance.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.7,
                context=["long_duration", "resource_configuration"],
                expected_impact="Improve resource utilization and analysis speed",
                implementation_effort="Medium",
                time_estimate="3-6 hours",
                resources_needed=["Resource monitoring tools", "Configuration templates"],
                tags=["configuration", "resources", "performance"]
            )
            self.recommendations.append(rec)
    
    async def _generate_workflow_recommendations(self,
                                               analysis_result: OrchestrationResult,
                                               insights: List[Insight],
                                               context: Dict[str, Any]):
        """Generate workflow enhancement recommendations."""
        phase_count = len(analysis_result.phase_results)
        
        # Complex analysis workflow
        if phase_count > 5:
            rec = Recommendation(
                id=f"workflow_complex_{len(self.recommendations)}",
                type=RecommendationType.WORKFLOW_ENHANCEMENT,
                title="Simplify Analysis Workflow",
                description=f"Analysis has {phase_count} phases. Consider simplifying the workflow for better maintainability.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.7,
                context=["complex_workflow", "maintainability"],
                expected_impact="Improve workflow maintainability and reduce complexity",
                implementation_effort="High",
                time_estimate="1-2 days",
                resources_needed=["Workflow analysis tools", "Process documentation"],
                tags=["workflow", "simplification", "maintainability"]
            )
            self.recommendations.append(rec)
        
        # Sequential phase execution
        if phase_count > 2:
            rec = Recommendation(
                id=f"workflow_sequential_{len(self.recommendations)}",
                type=RecommendationType.WORKFLOW_ENHANCEMENT,
                title="Implement Parallel Workflow",
                description=f"Analysis phases are executed sequentially. Consider implementing parallel execution for independent phases.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["sequential_execution", "parallel_workflow"],
                expected_impact="Reduce total analysis time through parallel execution",
                implementation_effort="High",
                time_estimate="2-3 days",
                resources_needed=["Parallel execution framework", "Dependency analysis"],
                tags=["workflow", "parallel", "performance"]
            )
            self.recommendations.append(rec)
    
    async def _generate_resource_recommendations(self,
                                               analysis_result: OrchestrationResult,
                                               insights: List[Insight],
                                               context: Dict[str, Any]):
        """Generate resource management recommendations."""
        duration = analysis_result.total_duration
        
        # High resource usage
        if duration > 1200:  # 20 minutes
            rec = Recommendation(
                id=f"resource_high_{len(self.recommendations)}",
                type=RecommendationType.RESOURCE_MANAGEMENT,
                title="Optimize Resource Usage",
                description=f"Analysis took {duration:.2f} seconds. Consider optimizing resource usage for better efficiency.",
                priority=RecommendationPriority.HIGH,
                confidence=0.8,
                context=["high_resource_usage", "optimization"],
                expected_impact="Reduce resource consumption and improve efficiency",
                implementation_effort="Medium",
                time_estimate="4-8 hours",
                resources_needed=["Resource monitoring tools", "Optimization frameworks"],
                tags=["resources", "optimization", "efficiency"]
            )
            self.recommendations.append(rec)
        
        # Memory-related warnings
        memory_warnings = [w for w in analysis_result.warnings if "memory" in w.lower()]
        if memory_warnings:
            rec = Recommendation(
                id=f"resource_memory_{len(self.recommendations)}",
                type=RecommendationType.RESOURCE_MANAGEMENT,
                title="Optimize Memory Usage",
                description=f"Found {len(memory_warnings)} memory-related warnings. Implement memory optimization strategies.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["memory_warnings", "resource_optimization"],
                expected_impact="Reduce memory usage and prevent out-of-memory errors",
                implementation_effort="Medium",
                time_estimate="3-6 hours",
                resources_needed=["Memory profiling tools", "Memory optimization libraries"],
                tags=["resources", "memory", "optimization"]
            )
            self.recommendations.append(rec)
    
    async def _generate_analysis_recommendations(self,
                                               analysis_result: OrchestrationResult,
                                               insights: List[Insight],
                                               context: Dict[str, Any]):
        """Generate analysis deepening recommendations."""
        # Check for clustering insights
        clustering_insights = [i for i in insights if i.type == InsightType.CLUSTERING_RESULT]
        if clustering_insights:
            rec = Recommendation(
                id=f"analysis_clustering_{len(self.recommendations)}",
                type=RecommendationType.ANALYSIS_DEEPENING,
                title="Deepen Clustering Analysis",
                description=f"Found {len(clustering_insights)} clustering insights. Consider deeper analysis of identified clusters.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["clustering_insights", "deep_analysis"],
                expected_impact="Gain deeper insights into data patterns and relationships",
                implementation_effort="Medium",
                time_estimate="4-8 hours",
                resources_needed=["Advanced clustering algorithms", "Visualization tools"],
                related_insights=[i.id for i in clustering_insights],
                tags=["analysis", "clustering", "deep_insights"]
            )
            self.recommendations.append(rec)
        
        # Check for correlation insights
        correlation_insights = [i for i in insights if i.type == InsightType.CORRELATION_FINDING]
        if correlation_insights:
            rec = Recommendation(
                id=f"analysis_correlation_{len(self.recommendations)}",
                type=RecommendationType.ANALYSIS_DEEPENING,
                title="Investigate Correlations",
                description=f"Found {len(correlation_insights)} correlation insights. Investigate these relationships for deeper understanding.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["correlation_insights", "relationship_analysis"],
                expected_impact="Understand causal relationships and dependencies",
                implementation_effort="Medium",
                time_estimate="3-6 hours",
                resources_needed=["Statistical analysis tools", "Correlation analysis libraries"],
                related_insights=[i.id for i in correlation_insights],
                tags=["analysis", "correlation", "relationships"]
            )
            self.recommendations.append(rec)
    
    async def _generate_monitoring_recommendations(self,
                                                 analysis_result: OrchestrationResult,
                                                 insights: List[Insight],
                                                 context: Dict[str, Any]):
        """Generate monitoring setup recommendations."""
        # High error rate
        if len(analysis_result.errors) > 3:
            rec = Recommendation(
                id=f"monitoring_errors_{len(self.recommendations)}",
                type=RecommendationType.MONITORING_SETUP,
                title="Implement Error Monitoring",
                description=f"Analysis generated {len(analysis_result.errors)} errors. Set up monitoring to track and alert on errors.",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                context=["high_error_rate", "monitoring"],
                expected_impact="Proactively detect and resolve issues",
                implementation_effort="Medium",
                time_estimate="4-8 hours",
                resources_needed=["Monitoring framework", "Alerting system"],
                tags=["monitoring", "errors", "alerting"]
            )
            self.recommendations.append(rec)
        
        # Performance monitoring
        if analysis_result.total_duration > 300:  # 5 minutes
            rec = Recommendation(
                id=f"monitoring_performance_{len(self.recommendations)}",
                type=RecommendationType.MONITORING_SETUP,
                title="Set Up Performance Monitoring",
                description="Long analysis duration suggests need for performance monitoring to track and optimize execution.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["long_duration", "performance_monitoring"],
                expected_impact="Track performance metrics and identify bottlenecks",
                implementation_effort="Medium",
                time_estimate="3-6 hours",
                resources_needed=["Performance monitoring tools", "Metrics collection"],
                tags=["monitoring", "performance", "metrics"]
            )
            self.recommendations.append(rec)
    
    async def _generate_documentation_recommendations(self,
                                                    analysis_result: OrchestrationResult,
                                                    insights: List[Insight],
                                                    context: Dict[str, Any]):
        """Generate documentation update recommendations."""
        # Complex analysis with many phases
        if len(analysis_result.phase_results) > 3:
            rec = Recommendation(
                id=f"doc_workflow_{len(self.recommendations)}",
                type=RecommendationType.DOCUMENTATION_UPDATE,
                title="Document Analysis Workflow",
                description=f"Analysis has {len(analysis_result.phase_results)} phases. Document the workflow for better understanding and maintenance.",
                priority=RecommendationPriority.LOW,
                confidence=0.7,
                context=["complex_workflow", "documentation"],
                expected_impact="Improve workflow understanding and maintainability",
                implementation_effort="Low",
                time_estimate="2-4 hours",
                resources_needed=["Documentation tools", "Workflow diagrams"],
                tags=["documentation", "workflow", "maintainability"]
            )
            self.recommendations.append(rec)
        
        # Many insights generated
        if len(insights) > 10:
            rec = Recommendation(
                id=f"doc_insights_{len(self.recommendations)}",
                type=RecommendationType.DOCUMENTATION_UPDATE,
                title="Document Analysis Insights",
                description=f"Generated {len(insights)} insights. Document these findings for future reference and knowledge sharing.",
                priority=RecommendationPriority.LOW,
                confidence=0.8,
                context=["many_insights", "knowledge_management"],
                expected_impact="Preserve knowledge and enable knowledge sharing",
                implementation_effort="Low",
                time_estimate="1-3 hours",
                resources_needed=["Documentation tools", "Knowledge management system"],
                tags=["documentation", "insights", "knowledge"]
            )
            self.recommendations.append(rec)
    
    async def _generate_automation_recommendations(self,
                                                 analysis_result: OrchestrationResult,
                                                 insights: List[Insight],
                                                 context: Dict[str, Any]):
        """Generate automation opportunity recommendations."""
        # Repetitive analysis patterns
        if len(analysis_result.phase_results) > 2:
            rec = Recommendation(
                id=f"automation_phases_{len(self.recommendations)}",
                type=RecommendationType.AUTOMATION_OPPORTUNITY,
                title="Automate Analysis Phases",
                description=f"Analysis has {len(analysis_result.phase_results)} phases. Consider automating repetitive phases for efficiency.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.7,
                context=["multiple_phases", "automation"],
                expected_impact="Reduce manual effort and improve consistency",
                implementation_effort="High",
                time_estimate="1-2 weeks",
                resources_needed=["Automation framework", "Workflow orchestration tools"],
                tags=["automation", "phases", "efficiency"]
            )
            self.recommendations.append(rec)
        
        # Error handling automation
        if analysis_result.errors:
            rec = Recommendation(
                id=f"automation_errors_{len(self.recommendations)}",
                type=RecommendationType.AUTOMATION_OPPORTUNITY,
                title="Automate Error Handling",
                description=f"Analysis generated {len(analysis_result.errors)} errors. Implement automated error handling and recovery.",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                context=["error_handling", "automation"],
                expected_impact="Improve reliability and reduce manual intervention",
                implementation_effort="Medium",
                time_estimate="3-5 days",
                resources_needed=["Error handling framework", "Recovery mechanisms"],
                tags=["automation", "errors", "reliability"]
            )
            self.recommendations.append(rec)
    
    def _apply_context_filtering(self, recommendations: List[Recommendation], context: Dict[str, Any]) -> List[Recommendation]:
        """Apply context-aware filtering to recommendations."""
        filtered = []
        
        for rec in recommendations:
            # Filter by confidence
            if rec.confidence < self.config.min_confidence:
                continue
            
            # Filter by user context
            if not self._is_recommendation_appropriate_for_context(rec, context):
                continue
            
            # Filter by resource constraints
            if self.config.consider_resource_constraints and not self._is_recommendation_feasible(rec, context):
                continue
            
            # Filter by time constraints
            if self.config.consider_time_constraints and not self._is_recommendation_timely(rec, context):
                continue
            
            filtered.append(rec)
        
        return filtered
    
    def _is_recommendation_appropriate_for_context(self, rec: Recommendation, context: Dict[str, Any]) -> bool:
        """Check if recommendation is appropriate for user context."""
        user_level = context.get("user_level", self.config.user_context.value)
        
        # Filter by user level
        if user_level == "beginner" and rec.implementation_effort == "High":
            return False
        
        if user_level == "expert" and rec.implementation_effort == "Low":
            return False
        
        return True
    
    def _is_recommendation_feasible(self, rec: Recommendation, context: Dict[str, Any]) -> bool:
        """Check if recommendation is feasible given resource constraints."""
        available_resources = context.get("available_resources", [])
        
        if not available_resources:
            return True
        
        for resource in rec.resources_needed:
            if resource not in available_resources:
                return False
        
        return True
    
    def _is_recommendation_timely(self, rec: Recommendation, context: Dict[str, Any]) -> bool:
        """Check if recommendation is timely given time constraints."""
        time_available = context.get("time_available", "unlimited")
        
        if time_available == "unlimited":
            return True
        
        if time_available == "limited" and rec.time_estimate in ["1-2 days", "2-3 days", "1-2 weeks"]:
            return False
        
        return True
    
    def _rank_recommendations(self, recommendations: List[Recommendation], context: Dict[str, Any]) -> List[Recommendation]:
        """Rank recommendations by priority and impact."""
        # Sort by priority and confidence
        priority_order = {
            RecommendationPriority.CRITICAL: 4,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1
        }
        
        recommendations.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.confidence),
            reverse=True
        )
        
        # Limit to max recommendations
        return recommendations[:self.config.max_recommendations]
    
    def _update_learning_data(self, analysis_result: OrchestrationResult, insights: List[Insight], context: Dict[str, Any]):
        """Update learning data based on current analysis."""
        # Record analysis patterns
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "duration": analysis_result.total_duration,
            "phase_count": len(analysis_result.phase_results),
            "error_count": len(analysis_result.errors),
            "warning_count": len(analysis_result.warnings),
            "insight_count": len(insights),
            "user_context": context
        }
        
        self.user_history.append(pattern)
        
        # Update learning data
        if "analysis_patterns" not in self.learning_data:
            self.learning_data["analysis_patterns"] = []
        
        self.learning_data["analysis_patterns"].append(pattern)
        
        # Keep only recent history
        if len(self.user_history) > 100:
            self.user_history = self.user_history[-50:]
    
    def _load_historical_data(self):
        """Load historical data for learning."""
        # This would load from a file or database in practice
        self.user_history = []
        self.learning_data = {}
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get a summary of generated recommendations."""
        if not self.recommendations:
            return {"total_recommendations": 0}
        
        # Count by type and priority
        type_counts = {}
        priority_counts = {}
        
        for rec in self.recommendations:
            type_counts[rec.type.value] = type_counts.get(rec.type.value, 0) + 1
            priority_counts[rec.priority.value] = priority_counts.get(rec.priority.value, 0) + 1
        
        return {
            "total_recommendations": len(self.recommendations),
            "by_type": type_counts,
            "by_priority": priority_counts,
            "avg_confidence": statistics.mean([r.confidence for r in self.recommendations]),
            "high_priority_count": priority_counts.get("high", 0) + priority_counts.get("critical", 0)
        }
    
    def export_recommendations(self, format: str = "json", file_path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Export recommendations to various formats."""
        if not self.recommendations:
            return "No recommendations to export"
        
        if format == "json":
            data = [rec.__dict__ for rec in self.recommendations]
            json_str = json.dumps(data, indent=2, default=str)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_str)
                return Path(file_path)
            else:
                return json_str
        
        elif format == "markdown":
            md_content = "# Analysis Recommendations\n\n"
            md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"Total recommendations: {len(self.recommendations)}\n\n"
            
            # Group by priority
            for priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH, RecommendationPriority.MEDIUM, RecommendationPriority.LOW]:
                priority_recs = [r for r in self.recommendations if r.priority == priority]
                if priority_recs:
                    md_content += f"## {priority.value.title()} Priority ({len(priority_recs)} recommendations)\n\n"
                    for rec in priority_recs:
                        md_content += f"### {rec.title}\n"
                        md_content += f"**Type**: {rec.type.value}\n"
                        md_content += f"**Confidence**: {rec.confidence:.2f}\n"
                        md_content += f"**Description**: {rec.description}\n\n"
                        if rec.expected_impact:
                            md_content += f"**Expected Impact**: {rec.expected_impact}\n\n"
                        if rec.implementation_effort:
                            md_content += f"**Implementation Effort**: {rec.implementation_effort}\n\n"
                        if rec.time_estimate:
                            md_content += f"**Time Estimate**: {rec.time_estimate}\n\n"
                        if rec.resources_needed:
                            md_content += "**Resources Needed**:\n"
                            for resource in rec.resources_needed:
                                md_content += f"- {resource}\n"
                            md_content += "\n"
                        md_content += "---\n\n"
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(md_content)
                return Path(file_path)
            else:
                return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")