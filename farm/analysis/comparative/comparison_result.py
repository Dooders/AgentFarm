"""
Comparison result data structures.

This module defines data structures for storing and organizing
the results of simulation comparisons.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonSummary:
    """Summary of comparison results."""
    
    total_differences: int = 0
    config_differences: int = 0
    database_differences: int = 0
    log_differences: int = 0
    metrics_differences: int = 0
    severity: str = "low"  # low, medium, high
    comparison_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate total differences after initialization."""
        self.total_differences = (
            self.config_differences + 
            self.database_differences + 
            self.log_differences + 
            self.metrics_differences
        )
        
        # Determine severity
        if self.total_differences > 100:
            self.severity = "high"
        elif self.total_differences > 10:
            self.severity = "medium"


@dataclass
class ConfigComparisonResult:
    """Result of configuration comparison."""
    
    differences: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    significant_changes: List[str] = field(default_factory=list)
    formatted_output: str = ""


@dataclass
class DatabaseComparisonResult:
    """Result of database comparison."""
    
    schema_differences: Dict[str, Any] = field(default_factory=dict)
    data_differences: Dict[str, Any] = field(default_factory=dict)
    metric_differences: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogComparisonResult:
    """Result of log file comparison."""
    
    performance_differences: Dict[str, Any] = field(default_factory=dict)
    error_differences: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsComparisonResult:
    """Result of metrics comparison."""
    
    metric_differences: Dict[str, Any] = field(default_factory=dict)
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationComparisonResult:
    """Complete result of simulation comparison."""
    
    simulation1_path: Path
    simulation2_path: Path
    comparison_summary: ComparisonSummary
    config_comparison: ConfigComparisonResult
    database_comparison: DatabaseComparisonResult
    log_comparison: LogComparisonResult
    metrics_comparison: MetricsComparisonResult
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Update summary after initialization."""
        # Only update if the summary is empty (all values are 0)
        if (self.comparison_summary.config_differences == 0 and 
            self.comparison_summary.database_differences == 0 and 
            self.comparison_summary.log_differences == 0 and 
            self.comparison_summary.metrics_differences == 0):
            self._update_summary()
    
    def _update_summary(self):
        """Update the comparison summary with current results."""
        # Count differences from each component
        self.comparison_summary.config_differences = self._count_config_differences()
        self.comparison_summary.database_differences = self._count_database_differences()
        self.comparison_summary.log_differences = self._count_log_differences()
        self.comparison_summary.metrics_differences = self._count_metrics_differences()
        
        # Update total and severity
        self.comparison_summary.total_differences = (
            self.comparison_summary.config_differences +
            self.comparison_summary.database_differences +
            self.comparison_summary.log_differences +
            self.comparison_summary.metrics_differences
        )
        
        # Determine severity
        if self.comparison_summary.total_differences > 100:
            self.comparison_summary.severity = "high"
        elif self.comparison_summary.total_differences > 10:
            self.comparison_summary.severity = "medium"
        else:
            self.comparison_summary.severity = "low"
    
    def _count_config_differences(self) -> int:
        """Count configuration differences."""
        if not self.config_comparison.differences:
            return 0
        
        # Count items in differences dict
        total = 0
        for key, value in self.config_comparison.differences.items():
            if isinstance(value, list):
                total += len(value)
            elif isinstance(value, dict):
                total += len(value)
            else:
                total += 1
        
        return total
    
    def _count_database_differences(self) -> int:
        """Count database differences."""
        total = 0
        
        # Count schema differences
        if self.database_comparison.schema_differences:
            total += len(self.database_comparison.schema_differences)
        
        # Count data differences
        if self.database_comparison.data_differences:
            total += len(self.database_comparison.data_differences)
        
        # Count metric differences
        if self.database_comparison.metric_differences:
            total += len(self.database_comparison.metric_differences)
        
        return total
    
    def _count_log_differences(self) -> int:
        """Count log differences."""
        total = 0
        
        # Count performance differences
        if self.log_comparison.performance_differences:
            total += len(self.log_comparison.performance_differences)
        
        # Count error differences
        if self.log_comparison.error_differences:
            total += len(self.log_comparison.error_differences)
        
        return total
    
    def _count_metrics_differences(self) -> int:
        """Count metrics differences."""
        if not self.metrics_comparison.metric_differences:
            return 0
        
        return len(self.metrics_comparison.metric_differences)
    
    def get_summary_text(self) -> str:
        """Get human-readable summary of comparison."""
        lines = []
        lines.append("Simulation Comparison Summary")
        lines.append("=" * 50)
        lines.append(f"Simulation 1: {self.simulation1_path}")
        lines.append(f"Simulation 2: {self.simulation2_path}")
        lines.append(f"Comparison Time: {self.comparison_summary.comparison_time}")
        lines.append("")
        
        lines.append("Differences Found:")
        lines.append(f"  Configuration: {self.comparison_summary.config_differences}")
        lines.append(f"  Database: {self.comparison_summary.database_differences}")
        lines.append(f"  Logs: {self.comparison_summary.log_differences}")
        lines.append(f"  Metrics: {self.comparison_summary.metrics_differences}")
        lines.append(f"  Total: {self.comparison_summary.total_differences}")
        lines.append("")
        
        lines.append(f"Severity: {self.comparison_summary.severity.upper()}")
        
        return "\n".join(lines)
    
    def get_detailed_report(self) -> str:
        """Get detailed report of all differences."""
        lines = []
        lines.append(self.get_summary_text())
        lines.append("")
        
        # Configuration differences
        if self.config_comparison.differences:
            lines.append("Configuration Differences:")
            lines.append("-" * 30)
            lines.append(self.config_comparison.formatted_output)
            lines.append("")
        
        # Database differences
        if self.database_comparison.schema_differences or self.database_comparison.data_differences:
            lines.append("Database Differences:")
            lines.append("-" * 30)
            
            if self.database_comparison.schema_differences:
                lines.append("Schema Changes:")
                for table, diff in self.database_comparison.schema_differences.items():
                    if isinstance(diff, list):
                        lines.append(f"  {table}: {len(diff)} changes")
                    else:
                        lines.append(f"  {table}: {diff}")
                lines.append("")
            
            if self.database_comparison.data_differences:
                lines.append("Data Changes:")
                for table, diff in self.database_comparison.data_differences.items():
                    if isinstance(diff, dict) and 'difference' in diff:
                        lines.append(f"  {table}: {diff['difference']} row difference")
                lines.append("")
        
        # Log differences
        if self.log_comparison.performance_differences or self.log_comparison.error_differences:
            lines.append("Log Differences:")
            lines.append("-" * 30)
            
            if self.log_comparison.performance_differences:
                lines.append("Performance Differences:")
                for metric, diff in self.log_comparison.performance_differences.items():
                    lines.append(f"  {metric}: {diff}")
                lines.append("")
            
            if self.log_comparison.error_differences:
                lines.append("Error Differences:")
                for error_type, count in self.log_comparison.error_differences.items():
                    lines.append(f"  {error_type}: {count}")
                lines.append("")
        
        # Metrics differences
        if self.metrics_comparison.metric_differences:
            lines.append("Metrics Differences:")
            lines.append("-" * 30)
            for metric, diff in self.metrics_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'db1_value' in diff and 'db2_value' in diff:
                    lines.append(f"  {metric}: {diff['db1_value']} -> {diff['db2_value']}")
                else:
                    lines.append(f"  {metric}: {diff}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'simulation1_path': str(self.simulation1_path),
            'simulation2_path': str(self.simulation2_path),
            'comparison_summary': {
                'total_differences': self.comparison_summary.total_differences,
                'config_differences': self.comparison_summary.config_differences,
                'database_differences': self.comparison_summary.database_differences,
                'log_differences': self.comparison_summary.log_differences,
                'metrics_differences': self.comparison_summary.metrics_differences,
                'severity': self.comparison_summary.severity,
                'comparison_time': self.comparison_summary.comparison_time.isoformat()
            },
            'config_comparison': {
                'differences': self.config_comparison.differences,
                'summary': self.config_comparison.summary,
                'significant_changes': self.config_comparison.significant_changes
            },
            'database_comparison': {
                'schema_differences': self.database_comparison.schema_differences,
                'data_differences': self.database_comparison.data_differences,
                'metric_differences': self.database_comparison.metric_differences,
                'summary': self.database_comparison.summary
            },
            'log_comparison': {
                'performance_differences': self.log_comparison.performance_differences,
                'error_differences': self.log_comparison.error_differences,
                'summary': self.log_comparison.summary
            },
            'metrics_comparison': {
                'metric_differences': self.metrics_comparison.metric_differences,
                'performance_comparison': self.metrics_comparison.performance_comparison,
                'summary': self.metrics_comparison.summary
            },
            'metadata': self.metadata
        }
    
    def save_to_file(self, output_path: Union[str, Path], format: str = "json") -> None:
        """Save comparison result to file.
        
        Args:
            output_path: Path to save the result
            format: Output format ('json', 'yaml', 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            import json
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        
        elif format.lower() == "yaml":
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        elif format.lower() == "txt":
            with open(output_path, 'w') as f:
                f.write(self.get_detailed_report())
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Comparison result saved to {output_path}")
    
    def is_significantly_different(self, threshold: float = 0.1) -> bool:
        """Check if simulations are significantly different.
        
        Args:
            threshold: Threshold for considering differences significant (0.0 to 1.0)
        
        Returns:
            True if simulations are significantly different
        """
        if self.comparison_summary.severity == "high":
            return True
        
        if self.comparison_summary.severity == "medium" and threshold < 0.5:
            return True
        
        if self.comparison_summary.total_differences > 0 and threshold < 0.1:
            return True
        
        return False