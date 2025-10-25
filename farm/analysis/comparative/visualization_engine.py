"""
Visualization engine for simulation comparison results.

This module provides comprehensive visualization capabilities for
simulation comparison data, including charts, graphs, and interactive plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class VisualizationEngine:
    """Engine for creating visualizations from simulation comparison results."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None, style: str = "default"):
        """Initialize visualization engine.
        
        Args:
            output_dir: Directory to save visualization files
            style: Matplotlib style to use ('default', 'seaborn', 'ggplot', etc.)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Configure matplotlib for better quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
        logger.info(f"VisualizationEngine initialized with output directory: {self.output_dir}")
    
    def create_comparison_dashboard(self, result: SimulationComparisonResult) -> Dict[str, str]:
        """Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            result: Simulation comparison result
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Creating comparison dashboard")
        
        dashboard_files = {}
        
        # Create individual visualizations
        dashboard_files['summary_chart'] = self.create_summary_chart(result)
        dashboard_files['metrics_comparison'] = self.create_metrics_comparison_chart(result)
        dashboard_files['database_analysis'] = self.create_database_analysis_chart(result)
        dashboard_files['performance_comparison'] = self.create_performance_comparison_chart(result)
        dashboard_files['error_analysis'] = self.create_error_analysis_chart(result)
        dashboard_files['heatmap'] = self.create_differences_heatmap(result)
        
        # Create combined dashboard
        dashboard_files['dashboard'] = self.create_combined_dashboard(result, dashboard_files)
        
        return dashboard_files
    
    def create_summary_chart(self, result: SimulationComparisonResult) -> str:
        """Create a summary chart showing overall differences."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of differences by category
        categories = ['Configuration', 'Database', 'Logs', 'Metrics']
        values = [
            result.comparison_summary.config_differences,
            result.comparison_summary.database_differences,
            result.comparison_summary.log_differences,
            result.comparison_summary.metrics_differences
        ]
        
        # Only show categories with differences
        non_zero = [(cat, val) for cat, val in zip(categories, values) if val > 0]
        if non_zero:
            cats, vals = zip(*non_zero)
            ax1.pie(vals, labels=cats, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Differences by Category')
        else:
            ax1.text(0.5, 0.5, 'No Differences Found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Differences by Category')
        
        # Bar chart of total differences
        severity_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        color = severity_colors.get(result.comparison_summary.severity, 'gray')
        
        ax2.bar(['Total Differences'], [result.comparison_summary.total_differences], color=color)
        ax2.set_title(f'Total Differences (Severity: {result.comparison_summary.severity.upper()})')
        ax2.set_ylabel('Number of Differences')
        
        # Add value labels on bars
        ax2.text(0, result.comparison_summary.total_differences + 0.1, 
                str(result.comparison_summary.total_differences), 
                ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"summary_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_metrics_comparison_chart(self, result: SimulationComparisonResult) -> str:
        """Create a chart comparing metrics between simulations."""
        if not result.metrics_comparison.metric_differences:
            # Create empty chart if no metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Metrics Differences Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics Comparison')
        else:
            # Prepare data for plotting
            metrics_data = []
            for metric, diff in result.metrics_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                    metrics_data.append({
                        'metric': metric,
                        'sim1_value': diff['sim1_value'],
                        'sim2_value': diff['sim2_value'],
                        'difference': diff.get('difference', 0),
                        'percentage_change': diff.get('percentage_change', 0)
                    })
            
            if not metrics_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No Numeric Metrics to Compare', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Metrics Comparison')
            else:
                df = pd.DataFrame(metrics_data)
                
                # Create subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Bar chart comparing values
                x = np.arange(len(df))
                width = 0.35
                
                ax1.bar(x - width/2, df['sim1_value'], width, label='Simulation 1', alpha=0.8)
                ax1.bar(x + width/2, df['sim2_value'], width, label='Simulation 2', alpha=0.8)
                
                ax1.set_xlabel('Metrics')
                ax1.set_ylabel('Values')
                ax1.set_title('Metrics Comparison: Simulation 1 vs Simulation 2')
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['metric'], rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Percentage change chart
                colors = ['red' if x < 0 else 'green' for x in df['percentage_change']]
                ax2.bar(x, df['percentage_change'], color=colors, alpha=0.7)
                ax2.set_xlabel('Metrics')
                ax2.set_ylabel('Percentage Change (%)')
                ax2.set_title('Percentage Change from Simulation 1 to Simulation 2')
                ax2.set_xticks(x)
                ax2.set_xticklabels(df['metric'], rotation=45, ha='right')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_database_analysis_chart(self, result: SimulationComparisonResult) -> str:
        """Create a chart analyzing database differences."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Schema differences
        if result.database_comparison.schema_differences:
            schema_data = []
            for table, changes in result.database_comparison.schema_differences.items():
                if isinstance(changes, dict):
                    schema_data.append({
                        'table': table,
                        'changes': len(changes)
                    })
                elif isinstance(changes, list):
                    schema_data.append({
                        'table': table,
                        'changes': len(changes)
                    })
                else:
                    schema_data.append({
                        'table': table,
                        'changes': 1
                    })
            
            if schema_data:
                df_schema = pd.DataFrame(schema_data)
                ax1.bar(df_schema['table'], df_schema['changes'])
                ax1.set_title('Schema Changes by Table')
                ax1.set_xlabel('Table')
                ax1.set_ylabel('Number of Changes')
                ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No Schema Changes', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Schema Changes by Table')
        
        # Data differences
        if result.database_comparison.data_differences:
            data_info = []
            for table, info in result.database_comparison.data_differences.items():
                if isinstance(info, dict) and 'difference' in info:
                    data_info.append({
                        'table': table,
                        'row_difference': info['difference']
                    })
            
            if data_info:
                df_data = pd.DataFrame(data_info)
                colors = ['red' if x < 0 else 'green' for x in df_data['row_difference']]
                ax2.bar(df_data['table'], df_data['row_difference'], color=colors, alpha=0.7)
                ax2.set_title('Row Count Differences by Table')
                ax2.set_xlabel('Table')
                ax2.set_ylabel('Row Difference')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Data Differences', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Row Count Differences by Table')
        
        # Metric differences
        if result.database_comparison.metric_differences:
            metric_data = []
            for metric, diff in result.database_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'db1_value' in diff and 'db2_value' in diff:
                    metric_data.append({
                        'metric': metric,
                        'db1_value': diff['db1_value'],
                        'db2_value': diff['db2_value']
                    })
            
            if metric_data:
                df_metrics = pd.DataFrame(metric_data)
                x = np.arange(len(df_metrics))
                width = 0.35
                
                ax3.bar(x - width/2, df_metrics['db1_value'], width, label='DB 1', alpha=0.8)
                ax3.bar(x + width/2, df_metrics['db2_value'], width, label='DB 2', alpha=0.8)
                
                ax3.set_title('Database Metrics Comparison')
                ax3.set_xlabel('Metrics')
                ax3.set_ylabel('Values')
                ax3.set_xticks(x)
                ax3.set_xticklabels(df_metrics['metric'], rotation=45, ha='right')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Metric Differences', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Database Metrics Comparison')
        
        # Summary statistics
        summary_stats = {
            'Schema Changes': len(result.database_comparison.schema_differences),
            'Data Changes': len(result.database_comparison.data_differences),
            'Metric Changes': len(result.database_comparison.metric_differences)
        }
        
        ax4.bar(summary_stats.keys(), summary_stats.values())
        ax4.set_title('Database Changes Summary')
        ax4.set_ylabel('Number of Changes')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"database_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_performance_comparison_chart(self, result: SimulationComparisonResult) -> str:
        """Create a chart comparing performance metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance differences from logs
        if result.log_comparison.performance_differences:
            perf_data = []
            for metric, diff in result.log_comparison.performance_differences.items():
                if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                    perf_data.append({
                        'metric': metric,
                        'sim1_value': diff['sim1_value'],
                        'sim2_value': diff['sim2_value'],
                        'difference': diff.get('difference', 0)
                    })
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                x = np.arange(len(df_perf))
                width = 0.35
                
                ax1.bar(x - width/2, df_perf['sim1_value'], width, label='Simulation 1', alpha=0.8)
                ax1.bar(x + width/2, df_perf['sim2_value'], width, label='Simulation 2', alpha=0.8)
                
                ax1.set_title('Performance Metrics Comparison')
                ax1.set_xlabel('Metrics')
                ax1.set_ylabel('Values')
                ax1.set_xticks(x)
                ax1.set_xticklabels(df_perf['metric'], rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No Performance Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Performance Metrics Comparison')
        else:
            ax1.text(0.5, 0.5, 'No Performance Differences', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Performance Metrics Comparison')
        
        # Performance comparison from metrics
        if result.metrics_comparison.performance_comparison:
            perf_comp_data = []
            for metric, comp in result.metrics_comparison.performance_comparison.items():
                if isinstance(comp, dict) and 'ratio' in comp:
                    perf_comp_data.append({
                        'metric': metric,
                        'ratio': comp['ratio'],
                        'faster': comp.get('faster', 'equal')
                    })
            
            if perf_comp_data:
                df_comp = pd.DataFrame(perf_comp_data)
                colors = ['green' if x > 1 else 'red' if x < 1 else 'gray' for x in df_comp['ratio']]
                
                ax2.bar(df_comp['metric'], df_comp['ratio'], color=colors, alpha=0.7)
                ax2.set_title('Performance Ratio (Sim2/Sim1)')
                ax2.set_xlabel('Metrics')
                ax2.set_ylabel('Ratio')
                ax2.axhline(y=1, color='black', linestyle='-', alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Performance Comparison Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Performance Ratio (Sim2/Sim1)')
        else:
            ax2.text(0.5, 0.5, 'No Performance Comparison', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance Ratio (Sim2/Sim1)')
        
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_error_analysis_chart(self, result: SimulationComparisonResult) -> str:
        """Create a chart analyzing error patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error differences from logs
        if result.log_comparison.error_differences:
            error_data = []
            for error_type, diff in result.log_comparison.error_differences.items():
                if isinstance(diff, dict) and 'sim1_count' in diff and 'sim2_count' in diff:
                    error_data.append({
                        'error_type': error_type,
                        'sim1_count': diff['sim1_count'],
                        'sim2_count': diff['sim2_count'],
                        'difference': diff.get('difference', 0)
                    })
            
            if error_data:
                df_errors = pd.DataFrame(error_data)
                x = np.arange(len(df_errors))
                width = 0.35
                
                ax1.bar(x - width/2, df_errors['sim1_count'], width, label='Simulation 1', alpha=0.8)
                ax1.bar(x + width/2, df_errors['sim2_count'], width, label='Simulation 2', alpha=0.8)
                
                ax1.set_title('Error Counts Comparison')
                ax1.set_xlabel('Error Types')
                ax1.set_ylabel('Count')
                ax1.set_xticks(x)
                ax1.set_xticklabels(df_errors['error_type'], rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No Error Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Error Counts Comparison')
        else:
            ax1.text(0.5, 0.5, 'No Error Differences', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Error Counts Comparison')
        
        # Error trend analysis
        if result.log_comparison.error_differences:
            error_trend_data = []
            for error_type, diff in result.log_comparison.error_differences.items():
                if isinstance(diff, dict) and 'difference' in diff:
                    error_trend_data.append({
                        'error_type': error_type,
                        'difference': diff['difference']
                    })
            
            if error_trend_data:
                df_trend = pd.DataFrame(error_trend_data)
                colors = ['red' if x > 0 else 'green' if x < 0 else 'gray' for x in df_trend['difference']]
                
                ax2.bar(df_trend['error_type'], df_trend['difference'], color=colors, alpha=0.7)
                ax2.set_title('Error Count Changes (Sim2 - Sim1)')
                ax2.set_xlabel('Error Types')
                ax2.set_ylabel('Count Difference')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Error Trend Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Error Count Changes (Sim2 - Sim1)')
        else:
            ax2.text(0.5, 0.5, 'No Error Trends', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Error Count Changes (Sim2 - Sim1)')
        
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_differences_heatmap(self, result: SimulationComparisonResult) -> str:
        """Create a heatmap showing differences across all categories."""
        # Prepare data for heatmap
        categories = ['Configuration', 'Database', 'Logs', 'Metrics']
        subcategories = ['Schema', 'Data', 'Performance', 'Errors']
        
        # Create a matrix of differences
        diff_matrix = np.zeros((len(categories), len(subcategories)))
        
        # Fill in the matrix with actual differences
        diff_matrix[0, 0] = result.comparison_summary.config_differences  # Config
        diff_matrix[1, 0] = len(result.database_comparison.schema_differences)  # DB Schema
        diff_matrix[1, 1] = len(result.database_comparison.data_differences)  # DB Data
        diff_matrix[2, 2] = len(result.log_comparison.performance_differences)  # Log Performance
        diff_matrix[2, 3] = len(result.log_comparison.error_differences)  # Log Errors
        diff_matrix[3, 0] = result.comparison_summary.metrics_differences  # Metrics
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(diff_matrix, cmap='Reds', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(subcategories)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(subcategories)
        ax.set_yticklabels(categories)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Differences')
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(subcategories)):
                text = ax.text(j, i, int(diff_matrix[i, j]),
                             ha="center", va="center", color="white" if diff_matrix[i, j] > diff_matrix.max()/2 else "black")
        
        ax.set_title('Differences Heatmap Across All Categories')
        plt.tight_layout()
        
        # Save the chart
        filename = self.output_dir / f"differences_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_combined_dashboard(self, result: SimulationComparisonResult, chart_files: Dict[str, str]) -> str:
        """Create a combined dashboard with all visualizations."""
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Add title
        fig.suptitle(f'Simulation Comparison Dashboard\n{result.simulation1_path.name} vs {result.simulation2_path.name}', 
                    fontsize=16, fontweight='bold')
        
        # Create a summary text box
        summary_text = f"""
        Comparison Summary:
        • Total Differences: {result.comparison_summary.total_differences}
        • Severity: {result.comparison_summary.severity.upper()}
        • Configuration: {result.comparison_summary.config_differences} differences
        • Database: {result.comparison_summary.database_differences} differences
        • Logs: {result.comparison_summary.log_differences} differences
        • Metrics: {result.comparison_summary.metrics_differences} differences
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Add text box
        plt.figtext(0.02, 0.95, summary_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add note about individual charts
        note_text = "Individual detailed charts have been saved as separate files."
        plt.figtext(0.02, 0.02, note_text, fontsize=9, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Save the dashboard
        filename = self.output_dir / f"comparison_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def create_interactive_plot(self, result: SimulationComparisonResult) -> str:
        """Create an interactive plot using plotly (if available)."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Summary', 'Metrics Comparison', 'Database Analysis', 'Performance'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Summary pie chart
            categories = ['Configuration', 'Database', 'Logs', 'Metrics']
            values = [
                result.comparison_summary.config_differences,
                result.comparison_summary.database_differences,
                result.comparison_summary.log_differences,
                result.comparison_summary.metrics_differences
            ]
            
            non_zero = [(cat, val) for cat, val in zip(categories, values) if val > 0]
            if non_zero:
                cats, vals = zip(*non_zero)
                fig.add_trace(go.Pie(labels=cats, values=vals, name="Differences"), row=1, col=1)
            
            # Metrics comparison
            if result.metrics_comparison.metric_differences:
                metrics_data = []
                for metric, diff in result.metrics_comparison.metric_differences.items():
                    if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                        metrics_data.append({
                            'metric': metric,
                            'sim1_value': diff['sim1_value'],
                            'sim2_value': diff['sim2_value']
                        })
                
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
                    fig.add_trace(go.Bar(x=df['metric'], y=df['sim1_value'], name='Simulation 1'), row=1, col=2)
                    fig.add_trace(go.Bar(x=df['metric'], y=df['sim2_value'], name='Simulation 2'), row=1, col=2)
            
            # Database analysis
            db_changes = {
                'Schema': len(result.database_comparison.schema_differences),
                'Data': len(result.database_comparison.data_differences),
                'Metrics': len(result.database_comparison.metric_differences)
            }
            
            fig.add_trace(go.Bar(x=list(db_changes.keys()), y=list(db_changes.values()), name='DB Changes'), row=2, col=1)
            
            # Performance scatter plot
            if result.metrics_comparison.performance_comparison:
                perf_data = []
                for metric, comp in result.metrics_comparison.performance_comparison.items():
                    if isinstance(comp, dict) and 'ratio' in comp:
                        perf_data.append({
                            'metric': metric,
                            'ratio': comp['ratio']
                        })
                
                if perf_data:
                    df_perf = pd.DataFrame(perf_data)
                    fig.add_trace(go.Scatter(x=df_perf['metric'], y=df_perf['ratio'], 
                                           mode='markers+lines', name='Performance Ratio'), row=2, col=2)
            
            # Update layout
            fig.update_layout(height=800, showlegend=True, 
                            title_text=f"Interactive Comparison Dashboard: {result.simulation1_path.name} vs {result.simulation2_path.name}")
            
            # Save as HTML
            filename = self.output_dir / f"interactive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            pyo.plot(fig, filename=str(filename), auto_open=False)
            
            return str(filename)
            
        except ImportError:
            logger.warning("Plotly not available, skipping interactive plot")
            return ""
    
    def export_data_for_external_tools(self, result: SimulationComparisonResult) -> Dict[str, str]:
        """Export data in formats suitable for external visualization tools."""
        export_files = {}
        
        # Export metrics data as CSV
        if result.metrics_comparison.metric_differences:
            metrics_data = []
            for metric, diff in result.metrics_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                    metrics_data.append({
                        'metric': metric,
                        'sim1_value': diff['sim1_value'],
                        'sim2_value': diff['sim2_value'],
                        'difference': diff.get('difference', 0),
                        'percentage_change': diff.get('percentage_change', 0)
                    })
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                csv_file = self.output_dir / f"metrics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_file, index=False)
                export_files['metrics_csv'] = str(csv_file)
        
        # Export summary data as JSON
        summary_data = {
            'simulation1_path': str(result.simulation1_path),
            'simulation2_path': str(result.simulation2_path),
            'comparison_time': result.comparison_summary.comparison_time.isoformat(),
            'total_differences': result.comparison_summary.total_differences,
            'severity': result.comparison_summary.severity,
            'config_differences': result.comparison_summary.config_differences,
            'database_differences': result.comparison_summary.database_differences,
            'log_differences': result.comparison_summary.log_differences,
            'metrics_differences': result.comparison_summary.metrics_differences
        }
        
        json_file = self.output_dir / f"summary_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        export_files['summary_json'] = str(json_file)
        
        return export_files