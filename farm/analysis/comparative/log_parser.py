"""
Log file parser for extracting performance and error metrics.

This module provides functionality to parse simulation log files
and extract relevant metrics for comparison.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogMetrics:
    """Container for parsed log metrics."""
    
    performance_metrics: Dict[str, Any]
    error_metrics: Dict[str, Any]
    summary_stats: Dict[str, Any]
    log_file_info: Dict[str, Any]


class LogParser:
    """Parses log files to extract performance and error metrics."""
    
    def __init__(self):
        """Initialize log parser with common patterns."""
        self.performance_patterns = self._init_performance_patterns()
        self.error_patterns = self._init_error_patterns()
        self.timing_patterns = self._init_timing_patterns()
    
    def _init_performance_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for performance metrics."""
        return {
            'execution_time': re.compile(r'execution time[:\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', re.IGNORECASE),
            'memory_usage': re.compile(r'memory[:\s]+(\d+(?:\.\d+)?)\s*(MB|GB|KB)', re.IGNORECASE),
            'throughput': re.compile(r'throughput[:\s]+(\d+(?:\.\d+)?)\s*(ops/sec|operations/second)', re.IGNORECASE),
            'iterations': re.compile(r'iteration[:\s]+(\d+)', re.IGNORECASE),
            'agents': re.compile(r'agents[:\s]+(\d+)', re.IGNORECASE),
            'steps': re.compile(r'step[:\s]+(\d+)', re.IGNORECASE),
            'simulation_time': re.compile(r'simulation time[:\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', re.IGNORECASE)
        }
    
    def _init_error_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for error detection."""
        return {
            'error': re.compile(r'error|exception|failed|failure', re.IGNORECASE),
            'warning': re.compile(r'warning|warn', re.IGNORECASE),
            'critical': re.compile(r'critical|fatal|panic', re.IGNORECASE),
            'timeout': re.compile(r'timeout|timed out', re.IGNORECASE),
            'memory_error': re.compile(r'memory error|out of memory|oom', re.IGNORECASE),
            'database_error': re.compile(r'database error|sql error|connection error', re.IGNORECASE)
        }
    
    def _init_timing_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for timing information."""
        return {
            'timestamp': re.compile(r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)'),
            'duration': re.compile(r'duration[:\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', re.IGNORECASE),
            'elapsed': re.compile(r'elapsed[:\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', re.IGNORECASE),
            'runtime': re.compile(r'runtime[:\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', re.IGNORECASE)
        }
    
    def parse_performance_metrics(self, log_paths: List[Path]) -> Dict[str, Any]:
        """Extract performance metrics from log files.
        
        Args:
            log_paths: List of log file paths to parse
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Parsing performance metrics from {len(log_paths)} log files")
        
        all_metrics = defaultdict(list)
        file_metrics = {}
        
        for log_path in log_paths:
            try:
                file_metric = self._parse_single_file_performance(log_path)
                file_metrics[log_path.name] = file_metric
                
                # Aggregate metrics
                for metric_type, values in file_metric.items():
                    if isinstance(values, list):
                        all_metrics[metric_type].extend(values)
                    else:
                        all_metrics[metric_type].append(values)
                        
            except Exception as e:
                logger.warning(f"Failed to parse performance metrics from {log_path}: {e}")
                file_metrics[log_path.name] = {}
        
        # Calculate summary statistics
        summary = self._calculate_performance_summary(all_metrics)
        
        return {
            'file_metrics': file_metrics,
            'aggregated_metrics': dict(all_metrics),
            'summary': summary
        }
    
    def parse_error_metrics(self, log_paths: List[Path]) -> Dict[str, Any]:
        """Extract error and warning metrics from log files.
        
        Args:
            log_paths: List of log file paths to parse
            
        Returns:
            Dictionary containing error metrics
        """
        logger.info(f"Parsing error metrics from {len(log_paths)} log files")
        
        error_counts = Counter()
        warning_counts = Counter()
        error_details = defaultdict(list)
        file_errors = {}
        
        for log_path in log_paths:
            try:
                file_error_data = self._parse_single_file_errors(log_path)
                file_errors[log_path.name] = file_error_data
                
                # Aggregate error counts
                for error_type, count in file_error_data.get('error_counts', {}).items():
                    error_counts[error_type] += count
                
                for warning_type, count in file_error_data.get('warning_counts', {}).items():
                    warning_counts[warning_type] += count
                
                # Collect error details
                for error_detail in file_error_data.get('error_details', []):
                    error_details[error_detail['type']].append(error_detail)
                    
            except Exception as e:
                logger.warning(f"Failed to parse error metrics from {log_path}: {e}")
                file_errors[log_path.name] = {}
        
        return {
            'file_errors': file_errors,
            'total_error_counts': dict(error_counts),
            'total_warning_counts': dict(warning_counts),
            'error_details': dict(error_details),
            'summary': {
                'total_errors': sum(error_counts.values()),
                'total_warnings': sum(warning_counts.values()),
                'unique_error_types': len(error_counts),
                'unique_warning_types': len(warning_counts)
            }
        }
    
    def parse_log_file(self, log_path: Path) -> LogMetrics:
        """Parse a single log file for all metrics.
        
        Args:
            log_path: Path to log file
            
        Returns:
            LogMetrics object containing all parsed data
        """
        logger.info(f"Parsing log file: {log_path}")
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
            return LogMetrics({}, {}, {}, {})
        
        # Parse different types of metrics
        performance_metrics = self._parse_single_file_performance(log_path)
        error_metrics = self._parse_single_file_errors(log_path)
        summary_stats = self._calculate_file_summary(lines, log_path)
        
        log_file_info = {
            'file_name': log_path.name,
            'file_size': log_path.stat().st_size,
            'total_lines': len(lines),
            'parsed_at': datetime.now().isoformat()
        }
        
        return LogMetrics(
            performance_metrics=performance_metrics,
            error_metrics=error_metrics,
            summary_stats=summary_stats,
            log_file_info=log_file_info
        )
    
    def _parse_single_file_performance(self, log_path: Path) -> Dict[str, Any]:
        """Parse performance metrics from a single log file."""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {log_path}: {e}")
            return {}
        
        metrics = {}
        
        # Extract performance metrics using patterns
        for metric_name, pattern in self.performance_patterns.items():
            matches = pattern.findall(content)
            if matches:
                values = []
                for match in matches:
                    if isinstance(match, tuple):
                        value, unit = match
                        # Convert to standard units
                        value = float(value)
                        if unit.lower() in ['ms']:
                            value = value / 1000  # Convert ms to seconds
                        values.append(value)
                    else:
                        values.append(float(match))
                
                metrics[metric_name] = values
                if len(values) == 1:
                    metrics[metric_name] = values[0]
        
        # Extract timing information
        timing_metrics = self._extract_timing_metrics(content)
        metrics.update(timing_metrics)
        
        return metrics
    
    def _parse_single_file_errors(self, log_path: Path) -> Dict[str, Any]:
        """Parse error metrics from a single log file."""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {log_path}: {e}")
            return {}
        
        error_counts = Counter()
        warning_counts = Counter()
        error_details = []
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for errors
            for error_type, pattern in self.error_patterns.items():
                if pattern.search(line):
                    if error_type == 'warning':
                        warning_counts[error_type] += 1
                    else:
                        error_counts[error_type] += 1
                    
                    # Extract error details
                    error_details.append({
                        'type': error_type,
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'timestamp': self._extract_timestamp_from_line(line)
                    })
        
        return {
            'error_counts': dict(error_counts),
            'warning_counts': dict(warning_counts),
            'error_details': error_details
        }
    
    def _extract_timing_metrics(self, content: str) -> Dict[str, Any]:
        """Extract timing-related metrics from log content."""
        timing_metrics = {}
        
        for timing_type, pattern in self.timing_patterns.items():
            matches = pattern.findall(content)
            if matches:
                if timing_type == 'timestamp':
                    timing_metrics['timestamps'] = matches
                else:
                    values = []
                    for match in matches:
                        if isinstance(match, tuple):
                            value, unit = match
                            value = float(value)
                            if unit.lower() in ['ms']:
                                value = value / 1000
                            values.append(value)
                        else:
                            values.append(float(match))
                    
                    timing_metrics[timing_type] = values
                    if len(values) == 1:
                        timing_metrics[timing_type] = values[0]
        
        return timing_metrics
    
    def _extract_timestamp_from_line(self, line: str) -> Optional[str]:
        """Extract timestamp from a log line."""
        timestamp_match = self.timing_patterns['timestamp'].search(line)
        return timestamp_match.group(1) if timestamp_match else None
    
    def _calculate_performance_summary(self, all_metrics: Dict[str, List]) -> Dict[str, Any]:
        """Calculate summary statistics for performance metrics."""
        summary = {}
        
        for metric_name, values in all_metrics.items():
            if not values:
                continue
            
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if not numeric_values:
                continue
            
            summary[metric_name] = {
                'count': len(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': sum(numeric_values) / len(numeric_values),
                'total': sum(numeric_values)
            }
            
            # Calculate standard deviation
            if len(numeric_values) > 1:
                mean = summary[metric_name]['mean']
                variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                summary[metric_name]['std_dev'] = variance ** 0.5
        
        return summary
    
    def _calculate_file_summary(self, lines: List[str], log_path: Path) -> Dict[str, Any]:
        """Calculate basic summary statistics for a log file."""
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        
        # Count different log levels (basic patterns)
        info_count = len([line for line in lines if 'info' in line.lower()])
        debug_count = len([line for line in lines if 'debug' in line.lower()])
        error_count = len([line for line in lines if 'error' in line.lower()])
        warning_count = len([line for line in lines if 'warning' in line.lower()])
        
        return {
            'total_lines': total_lines,
            'non_empty_lines': non_empty_lines,
            'empty_lines': total_lines - non_empty_lines,
            'info_messages': info_count,
            'debug_messages': debug_count,
            'error_messages': error_count,
            'warning_messages': warning_count,
            'file_size_bytes': log_path.stat().st_size
        }