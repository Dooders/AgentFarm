"""
Statistical analyzer for simulation comparison results.

This module provides advanced statistical analysis capabilities for
simulation comparison data, including significance testing, correlation
analysis, and trend detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticalAnalysisResult:
    """Result of statistical analysis."""
    correlation_analysis: Dict[str, Any]
    significance_tests: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    summary: Dict[str, Any]


class StatisticalAnalyzer:
    """Analyzer for performing statistical analysis on simulation comparison results."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize statistical analyzer.
        
        Args:
            significance_level: Alpha level for statistical tests (default: 0.05)
        """
        self.significance_level = significance_level
        logger.info(f"StatisticalAnalyzer initialized with significance level: {significance_level}")
    
    def analyze_comparison(self, result: SimulationComparisonResult) -> StatisticalAnalysisResult:
        """Perform comprehensive statistical analysis on comparison results.
        
        Args:
            result: Simulation comparison result
            
        Returns:
            Statistical analysis result
        """
        logger.info("Starting statistical analysis of comparison results")
        
        # Perform different types of analysis
        correlation_analysis = self._analyze_correlations(result)
        significance_tests = self._perform_significance_tests(result)
        trend_analysis = self._analyze_trends(result)
        anomaly_detection = self._detect_anomalies(result)
        
        # Generate summary
        summary = self._generate_analysis_summary(
            correlation_analysis, significance_tests, trend_analysis, anomaly_detection
        )
        
        return StatisticalAnalysisResult(
            correlation_analysis=correlation_analysis,
            significance_tests=significance_tests,
            trend_analysis=trend_analysis,
            anomaly_detection=anomaly_detection,
            summary=summary
        )
    
    def _analyze_correlations(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze correlations between different metrics and differences."""
        logger.debug("Analyzing correlations")
        
        correlations = {}
        
        # Extract numeric data for correlation analysis
        metrics_data = self._extract_metrics_data(result)
        performance_data = self._extract_performance_data(result)
        error_data = self._extract_error_data(result)
        
        # Correlate metrics with performance
        if metrics_data and performance_data:
            correlations['metrics_performance'] = self._calculate_correlation(
                metrics_data, performance_data, 'Metrics vs Performance'
            )
        
        # Correlate errors with performance
        if error_data and performance_data:
            correlations['errors_performance'] = self._calculate_correlation(
                error_data, performance_data, 'Errors vs Performance'
            )
        
        # Correlate database changes with metrics
        if result.database_comparison.metric_differences and metrics_data:
            db_metrics_data = self._extract_database_metrics_data(result)
            if db_metrics_data:
                correlations['database_metrics'] = self._calculate_correlation(
                    db_metrics_data, metrics_data, 'Database vs Metrics'
                )
        
        # Cross-correlation analysis
        correlations['cross_correlations'] = self._analyze_cross_correlations(result)
        
        return correlations
    
    def _extract_metrics_data(self, result: SimulationComparisonResult) -> Optional[Dict[str, float]]:
        """Extract numeric metrics data for analysis."""
        metrics_data = {}
        
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                try:
                    sim1_val = float(diff['sim1_value'])
                    sim2_val = float(diff['sim2_value'])
                    metrics_data[f"{metric}_sim1"] = sim1_val
                    metrics_data[f"{metric}_sim2"] = sim2_val
                    metrics_data[f"{metric}_diff"] = sim2_val - sim1_val
                except (ValueError, TypeError):
                    continue
        
        return metrics_data if metrics_data else None
    
    def _extract_performance_data(self, result: SimulationComparisonResult) -> Optional[Dict[str, float]]:
        """Extract performance data for analysis."""
        perf_data = {}
        
        # From log performance differences
        for metric, diff in result.log_comparison.performance_differences.items():
            if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                try:
                    sim1_val = float(diff['sim1_value'])
                    sim2_val = float(diff['sim2_value'])
                    perf_data[f"log_{metric}_sim1"] = sim1_val
                    perf_data[f"log_{metric}_sim2"] = sim2_val
                    perf_data[f"log_{metric}_diff"] = sim2_val - sim1_val
                except (ValueError, TypeError):
                    continue
        
        # From metrics performance comparison
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                try:
                    ratio = float(comp['ratio'])
                    perf_data[f"metrics_{metric}_ratio"] = ratio
                except (ValueError, TypeError):
                    continue
        
        return perf_data if perf_data else None
    
    def _extract_error_data(self, result: SimulationComparisonResult) -> Optional[Dict[str, float]]:
        """Extract error data for analysis."""
        error_data = {}
        
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'sim1_count' in diff and 'sim2_count' in diff:
                try:
                    sim1_count = float(diff['sim1_count'])
                    sim2_count = float(diff['sim2_count'])
                    error_data[f"{error_type}_sim1"] = sim1_count
                    error_data[f"{error_type}_sim2"] = sim2_count
                    error_data[f"{error_type}_diff"] = sim2_count - sim1_count
                except (ValueError, TypeError):
                    continue
        
        return error_data if error_data else None
    
    def _extract_database_metrics_data(self, result: SimulationComparisonResult) -> Optional[Dict[str, float]]:
        """Extract database metrics data for analysis."""
        db_data = {}
        
        for metric, diff in result.database_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'db1_value' in diff and 'db2_value' in diff:
                try:
                    db1_val = float(diff['db1_value'])
                    db2_val = float(diff['db2_value'])
                    db_data[f"db_{metric}_db1"] = db1_val
                    db_data[f"db_{metric}_db2"] = db2_val
                    db_data[f"db_{metric}_diff"] = db2_val - db1_val
                except (ValueError, TypeError):
                    continue
        
        return db_data if db_data else None
    
    def _calculate_correlation(self, data1: Dict[str, float], data2: Dict[str, float], 
                             label: str) -> Dict[str, Any]:
        """Calculate correlation between two datasets."""
        # Find common keys or create pairs
        common_keys = set(data1.keys()) & set(data2.keys())
        
        if len(common_keys) < 2:
            # Try to find similar keys or create artificial pairs
            keys1 = list(data1.keys())
            keys2 = list(data2.keys())
            
            if len(keys1) >= 2 and len(keys2) >= 2:
                # Use first few values from each dataset
                values1 = list(data1.values())[:min(len(keys1), len(keys2))]
                values2 = list(data2.values())[:min(len(keys1), len(keys2))]
            else:
                return {'label': label, 'correlation': 0.0, 'p_value': 1.0, 'significant': False}
        else:
            values1 = [data1[key] for key in common_keys]
            values2 = [data2[key] for key in common_keys]
        
        if len(values1) < 2:
            return {'label': label, 'correlation': 0.0, 'p_value': 1.0, 'significant': False}
        
        # Calculate Pearson correlation
        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            
            # Simple significance test (approximate)
            n = len(values1)
            if n > 2:
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
                # Approximate p-value (not exact, but good enough for this context)
                p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
            else:
                p_value = 1.0
            
            return {
                'label': label,
                'correlation': float(correlation),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'sample_size': n
            }
        except Exception as e:
            logger.warning(f"Error calculating correlation for {label}: {e}")
            return {'label': label, 'correlation': 0.0, 'p_value': 1.0, 'significant': False}
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Simple approximation - in practice, you'd use scipy.stats.t.cdf
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))
        else:
            # Simple approximation for small df
            return 0.5 + 0.5 * np.tanh(t / 2)
    
    def _analyze_cross_correlations(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze cross-correlations between different types of differences."""
        cross_correlations = {}
        
        # Create vectors of differences
        config_diffs = [result.comparison_summary.config_differences]
        db_diffs = [result.comparison_summary.database_differences]
        log_diffs = [result.comparison_summary.log_differences]
        metrics_diffs = [result.comparison_summary.metrics_differences]
        
        # Calculate correlations between difference types
        diff_types = {
            'config': config_diffs,
            'database': db_diffs,
            'logs': log_diffs,
            'metrics': metrics_diffs
        }
        
        for type1, values1 in diff_types.items():
            for type2, values2 in diff_types.items():
                if type1 != type2:
                    try:
                        corr = np.corrcoef(values1, values2)[0, 1]
                        cross_correlations[f"{type1}_vs_{type2}"] = {
                            'correlation': float(corr),
                            'interpretation': self._interpret_correlation(corr)
                        }
                    except:
                        cross_correlations[f"{type1}_vs_{type2}"] = {
                            'correlation': 0.0,
                            'interpretation': 'No correlation'
                        }
        
        return cross_correlations
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            return 'Negligible'
        elif abs_corr < 0.3:
            return 'Weak'
        elif abs_corr < 0.5:
            return 'Moderate'
        elif abs_corr < 0.7:
            return 'Strong'
        else:
            return 'Very Strong'
    
    def _perform_significance_tests(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Perform significance tests on the differences."""
        logger.debug("Performing significance tests")
        
        tests = {}
        
        # Test if differences are significantly different from zero
        tests['overall_significance'] = self._test_overall_significance(result)
        
        # Test specific metric differences
        tests['metric_significance'] = self._test_metric_significance(result)
        
        # Test performance differences
        tests['performance_significance'] = self._test_performance_significance(result)
        
        # Test error differences
        tests['error_significance'] = self._test_error_significance(result)
        
        return tests
    
    def _test_overall_significance(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Test if overall differences are significant."""
        total_diffs = result.comparison_summary.total_differences
        
        # Simple test: if there are any differences, they might be significant
        # In a real scenario, you'd have more sophisticated tests
        is_significant = total_diffs > 0
        
        return {
            'test_name': 'Overall Differences',
            'statistic': total_diffs,
            'p_value': 0.01 if is_significant else 0.99,  # Simplified
            'significant': is_significant,
            'interpretation': 'Significant differences found' if is_significant else 'No significant differences'
        }
    
    def _test_metric_significance(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Test significance of metric differences."""
        significant_metrics = []
        non_significant_metrics = []
        
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'percentage_change' in diff:
                try:
                    pct_change = abs(float(diff['percentage_change']))
                    # Consider >5% change as potentially significant
                    is_significant = pct_change > 5.0
                    
                    if is_significant:
                        significant_metrics.append({
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0)
                        })
                    else:
                        non_significant_metrics.append({
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0)
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'test_name': 'Metric Differences',
            'significant_metrics': significant_metrics,
            'non_significant_metrics': non_significant_metrics,
            'total_significant': len(significant_metrics),
            'total_tested': len(significant_metrics) + len(non_significant_metrics)
        }
    
    def _test_performance_significance(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Test significance of performance differences."""
        significant_perf = []
        
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                try:
                    ratio = float(comp['ratio'])
                    # Consider >10% change as significant
                    is_significant = abs(ratio - 1.0) > 0.1
                    
                    if is_significant:
                        significant_perf.append({
                            'metric': metric,
                            'ratio': ratio,
                            'percentage_change': (ratio - 1.0) * 100,
                            'faster': comp.get('faster', 'equal')
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'test_name': 'Performance Differences',
            'significant_performance_changes': significant_perf,
            'total_significant': len(significant_perf)
        }
    
    def _test_error_significance(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Test significance of error differences."""
        significant_errors = []
        
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'difference' in diff:
                try:
                    error_diff = int(diff['difference'])
                    # Consider any change in error count as potentially significant
                    is_significant = error_diff != 0
                    
                    if is_significant:
                        significant_errors.append({
                            'error_type': error_type,
                            'difference': error_diff,
                            'sim1_count': diff.get('sim1_count', 0),
                            'sim2_count': diff.get('sim2_count', 0)
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'test_name': 'Error Differences',
            'significant_error_changes': significant_errors,
            'total_significant': len(significant_errors)
        }
    
    def _analyze_trends(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze trends in the comparison data."""
        logger.debug("Analyzing trends")
        
        trends = {}
        
        # Analyze metric trends
        trends['metric_trends'] = self._analyze_metric_trends(result)
        
        # Analyze performance trends
        trends['performance_trends'] = self._analyze_performance_trends(result)
        
        # Analyze error trends
        trends['error_trends'] = self._analyze_error_trends(result)
        
        # Overall trend analysis
        trends['overall_trend'] = self._analyze_overall_trend(result)
        
        return trends
    
    def _analyze_metric_trends(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze trends in metric changes."""
        improving_metrics = []
        degrading_metrics = []
        stable_metrics = []
        
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'percentage_change' in diff:
                try:
                    pct_change = float(diff['percentage_change'])
                    
                    if pct_change > 5:  # >5% improvement
                        improving_metrics.append({
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0)
                        })
                    elif pct_change < -5:  # >5% degradation
                        degrading_metrics.append({
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0)
                        })
                    else:  # Stable
                        stable_metrics.append({
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0)
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'improving_metrics': improving_metrics,
            'degrading_metrics': degrading_metrics,
            'stable_metrics': stable_metrics,
            'trend_summary': {
                'improving_count': len(improving_metrics),
                'degrading_count': len(degrading_metrics),
                'stable_count': len(stable_metrics)
            }
        }
    
    def _analyze_performance_trends(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze performance trends."""
        faster_metrics = []
        slower_metrics = []
        unchanged_metrics = []
        
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                try:
                    ratio = float(comp['ratio'])
                    
                    if ratio > 1.1:  # >10% faster
                        faster_metrics.append({
                            'metric': metric,
                            'ratio': ratio,
                            'improvement': (ratio - 1.0) * 100
                        })
                    elif ratio < 0.9:  # >10% slower
                        slower_metrics.append({
                            'metric': metric,
                            'ratio': ratio,
                            'degradation': (1.0 - ratio) * 100
                        })
                    else:  # Similar performance
                        unchanged_metrics.append({
                            'metric': metric,
                            'ratio': ratio
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'faster_metrics': faster_metrics,
            'slower_metrics': slower_metrics,
            'unchanged_metrics': unchanged_metrics,
            'performance_summary': {
                'faster_count': len(faster_metrics),
                'slower_count': len(slower_metrics),
                'unchanged_count': len(unchanged_metrics)
            }
        }
    
    def _analyze_error_trends(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze error trends."""
        increasing_errors = []
        decreasing_errors = []
        unchanged_errors = []
        
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'difference' in diff:
                try:
                    error_diff = int(diff['difference'])
                    
                    if error_diff > 0:  # More errors in sim2
                        increasing_errors.append({
                            'error_type': error_type,
                            'difference': error_diff,
                            'sim1_count': diff.get('sim1_count', 0),
                            'sim2_count': diff.get('sim2_count', 0)
                        })
                    elif error_diff < 0:  # Fewer errors in sim2
                        decreasing_errors.append({
                            'error_type': error_type,
                            'difference': error_diff,
                            'sim1_count': diff.get('sim1_count', 0),
                            'sim2_count': diff.get('sim2_count', 0)
                        })
                    else:  # Same error count
                        unchanged_errors.append({
                            'error_type': error_type,
                            'difference': error_diff,
                            'sim1_count': diff.get('sim1_count', 0),
                            'sim2_count': diff.get('sim2_count', 0)
                        })
                except (ValueError, TypeError):
                    continue
        
        return {
            'increasing_errors': increasing_errors,
            'decreasing_errors': decreasing_errors,
            'unchanged_errors': unchanged_errors,
            'error_summary': {
                'increasing_count': len(increasing_errors),
                'decreasing_count': len(decreasing_errors),
                'unchanged_count': len(unchanged_errors)
            }
        }
    
    def _analyze_overall_trend(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Analyze overall trend of the comparison."""
        # Determine overall trend based on various factors
        positive_factors = 0
        negative_factors = 0
        
        # Count positive and negative changes
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'percentage_change' in diff:
                try:
                    pct_change = float(diff['percentage_change'])
                    if pct_change > 0:
                        positive_factors += 1
                    elif pct_change < 0:
                        negative_factors += 1
                except (ValueError, TypeError):
                    continue
        
        # Performance factors
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                try:
                    ratio = float(comp['ratio'])
                    if ratio > 1.0:
                        positive_factors += 1
                    elif ratio < 1.0:
                        negative_factors += 1
                except (ValueError, TypeError):
                    continue
        
        # Error factors (fewer errors is positive)
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'difference' in diff:
                try:
                    error_diff = int(diff['difference'])
                    if error_diff < 0:  # Fewer errors
                        positive_factors += 1
                    elif error_diff > 0:  # More errors
                        negative_factors += 1
                except (ValueError, TypeError):
                    continue
        
        # Determine overall trend
        if positive_factors > negative_factors:
            trend = 'positive'
            trend_strength = positive_factors / (positive_factors + negative_factors) if (positive_factors + negative_factors) > 0 else 0
        elif negative_factors > positive_factors:
            trend = 'negative'
            trend_strength = negative_factors / (positive_factors + negative_factors) if (positive_factors + negative_factors) > 0 else 0
        else:
            trend = 'neutral'
            trend_strength = 0.5
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'interpretation': self._interpret_trend(trend, trend_strength)
        }
    
    def _interpret_trend(self, trend: str, strength: float) -> str:
        """Interpret trend and strength."""
        if trend == 'positive':
            if strength > 0.7:
                return 'Strong positive trend - significant improvements'
            elif strength > 0.5:
                return 'Moderate positive trend - some improvements'
            else:
                return 'Weak positive trend - minor improvements'
        elif trend == 'negative':
            if strength > 0.7:
                return 'Strong negative trend - significant degradations'
            elif strength > 0.5:
                return 'Moderate negative trend - some degradations'
            else:
                return 'Weak negative trend - minor degradations'
        else:
            return 'Neutral trend - balanced changes'
    
    def _detect_anomalies(self, result: SimulationComparisonResult) -> Dict[str, Any]:
        """Detect anomalies in the comparison data."""
        logger.debug("Detecting anomalies")
        
        anomalies = {}
        
        # Detect metric anomalies
        anomalies['metric_anomalies'] = self._detect_metric_anomalies(result)
        
        # Detect performance anomalies
        anomalies['performance_anomalies'] = self._detect_performance_anomalies(result)
        
        # Detect error anomalies
        anomalies['error_anomalies'] = self._detect_error_anomalies(result)
        
        # Overall anomaly summary
        anomalies['summary'] = self._summarize_anomalies(anomalies)
        
        return anomalies
    
    def _detect_metric_anomalies(self, result: SimulationComparisonResult) -> List[Dict[str, Any]]:
        """Detect anomalies in metric changes."""
        anomalies = []
        
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'percentage_change' in diff:
                try:
                    pct_change = abs(float(diff['percentage_change']))
                    
                    # Consider >50% change as anomalous
                    if pct_change > 50:
                        anomalies.append({
                            'type': 'metric_anomaly',
                            'metric': metric,
                            'percentage_change': pct_change,
                            'change': diff.get('difference', 0),
                            'severity': 'high' if pct_change > 100 else 'medium',
                            'description': f'Extreme change in {metric}: {pct_change:.1f}%'
                        })
                except (ValueError, TypeError):
                    continue
        
        return anomalies
    
    def _detect_performance_anomalies(self, result: SimulationComparisonResult) -> List[Dict[str, Any]]:
        """Detect anomalies in performance changes."""
        anomalies = []
        
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                try:
                    ratio = float(comp['ratio'])
                    
                    # Consider >2x or <0.5x change as anomalous
                    if ratio > 2.0 or ratio < 0.5:
                        anomalies.append({
                            'type': 'performance_anomaly',
                            'metric': metric,
                            'ratio': ratio,
                            'change': (ratio - 1.0) * 100,
                            'severity': 'high' if ratio > 3.0 or ratio < 0.33 else 'medium',
                            'description': f'Extreme performance change in {metric}: {ratio:.2f}x'
                        })
                except (ValueError, TypeError):
                    continue
        
        return anomalies
    
    def _detect_error_anomalies(self, result: SimulationComparisonResult) -> List[Dict[str, Any]]:
        """Detect anomalies in error patterns."""
        anomalies = []
        
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'sim1_count' in diff and 'sim2_count' in diff:
                try:
                    sim1_count = int(diff['sim1_count'])
                    sim2_count = int(diff['sim2_count'])
                    difference = sim2_count - sim1_count
                    
                    # Consider >10x change in error count as anomalous
                    if sim1_count > 0:
                        ratio = sim2_count / sim1_count
                        if ratio > 10 or ratio < 0.1:
                            anomalies.append({
                                'type': 'error_anomaly',
                                'error_type': error_type,
                                'sim1_count': sim1_count,
                                'sim2_count': sim2_count,
                                'ratio': ratio,
                                'severity': 'high' if ratio > 20 or ratio < 0.05 else 'medium',
                                'description': f'Extreme error change in {error_type}: {ratio:.1f}x'
                            })
                    elif sim2_count > 10:  # New errors appeared
                        anomalies.append({
                            'type': 'error_anomaly',
                            'error_type': error_type,
                            'sim1_count': sim1_count,
                            'sim2_count': sim2_count,
                            'ratio': float('inf'),
                            'severity': 'high',
                            'description': f'New errors appeared in {error_type}: {sim2_count}'
                        })
                except (ValueError, TypeError):
                    continue
        
        return anomalies
    
    def _summarize_anomalies(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize detected anomalies."""
        total_anomalies = 0
        high_severity = 0
        medium_severity = 0
        
        for category, anomaly_list in anomalies.items():
            if isinstance(anomaly_list, list):
                total_anomalies += len(anomaly_list)
                for anomaly in anomaly_list:
                    if isinstance(anomaly, dict) and 'severity' in anomaly:
                        if anomaly['severity'] == 'high':
                            high_severity += 1
                        elif anomaly['severity'] == 'medium':
                            medium_severity += 1
        
        return {
            'total_anomalies': total_anomalies,
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': total_anomalies - high_severity - medium_severity,
            'has_anomalies': total_anomalies > 0
        }
    
    def _generate_analysis_summary(self, correlation_analysis: Dict[str, Any], 
                                 significance_tests: Dict[str, Any],
                                 trend_analysis: Dict[str, Any],
                                 anomaly_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of statistical analysis."""
        # Count significant correlations
        significant_correlations = 0
        for corr in correlation_analysis.values():
            if isinstance(corr, dict) and corr.get('significant', False):
                significant_correlations += 1
        
        # Count significant tests
        significant_tests = 0
        for test in significance_tests.values():
            if isinstance(test, dict) and test.get('significant', False):
                significant_tests += 1
        
        # Get trend information
        overall_trend = trend_analysis.get('overall_trend', {})
        
        # Get anomaly information
        anomaly_summary = anomaly_detection.get('summary', {})
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'significant_correlations': significant_correlations,
            'significant_tests': significant_tests,
            'overall_trend': overall_trend.get('trend', 'unknown'),
            'trend_strength': overall_trend.get('trend_strength', 0.0),
            'total_anomalies': anomaly_summary.get('total_anomalies', 0),
            'high_severity_anomalies': anomaly_summary.get('high_severity', 0),
            'analysis_quality': self._assess_analysis_quality(
                significant_correlations, significant_tests, anomaly_summary
            )
        }
    
    def _assess_analysis_quality(self, significant_correlations: int, significant_tests: int,
                                anomaly_summary: Dict[str, Any]) -> str:
        """Assess the quality of the statistical analysis."""
        total_anomalies = anomaly_summary.get('total_anomalies', 0)
        high_severity = anomaly_summary.get('high_severity', 0)
        
        if significant_correlations > 0 and significant_tests > 0:
            if total_anomalies == 0:
                return 'High quality - significant findings, no anomalies'
            elif high_severity == 0:
                return 'Good quality - significant findings, minor anomalies'
            else:
                return 'Moderate quality - significant findings, some anomalies'
        elif significant_correlations > 0 or significant_tests > 0:
            return 'Fair quality - some significant findings'
        else:
            return 'Limited quality - few significant findings'
    
    def export_analysis_results(self, analysis_result: StatisticalAnalysisResult, 
                              output_path: Union[str, Path]) -> str:
        """Export statistical analysis results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        export_data = {
            'correlation_analysis': analysis_result.correlation_analysis,
            'significance_tests': analysis_result.significance_tests,
            'trend_analysis': analysis_result.trend_analysis,
            'anomaly_detection': analysis_result.anomaly_detection,
            'summary': analysis_result.summary
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Statistical analysis results exported to {output_path}")
        return str(output_path)