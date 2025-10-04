"""
Tests for common analysis metrics and utility functions.

Tests the utility functions in farm.analysis.common.metrics.
"""

import pytest
import pandas as pd
import numpy as np

from farm.analysis.common.metrics import (
    get_valid_numeric_columns,
    split_and_compare_groups,
    analyze_correlations,
    group_and_analyze,
    find_top_correlations,
)


class TestGetValidNumericColumns:
    """Test get_valid_numeric_columns function."""
    
    def test_all_numeric_columns(self):
        """Test filtering with all numeric columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.5, 6.5],
            'col3': [7, 8, 9]
        })
        
        columns = ['col1', 'col2', 'col3']
        result = get_valid_numeric_columns(df, columns)
        
        assert result == columns
    
    def test_mixed_column_types(self):
        """Test filtering with mixed column types."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'string': ['a', 'b', 'c'],
            'numeric2': [4.5, 5.5, 6.5]
        })
        
        columns = ['numeric1', 'string', 'numeric2']
        result = get_valid_numeric_columns(df, columns)
        
        assert result == ['numeric1', 'numeric2']
        assert 'string' not in result
    
    def test_nonexistent_columns(self):
        """Test with columns that don't exist in DataFrame."""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        columns = ['col1', 'nonexistent']
        result = get_valid_numeric_columns(df, columns)
        
        assert result == ['col1']
        assert 'nonexistent' not in result
    
    def test_empty_column_list(self):
        """Test with empty column list."""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        result = get_valid_numeric_columns(df, [])
        assert result == []


class TestSplitAndCompareGroups:
    """Test split_and_compare_groups function."""
    
    def test_median_split(self):
        """Test splitting groups by median."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'metric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = split_and_compare_groups(
            df,
            split_column='score',
            metrics=['metric'],
            split_method='median'
        )
        
        assert 'comparison_results' in result
        assert 'metric' in result['comparison_results']
        assert 'high_group_mean' in result['comparison_results']['metric']
        assert 'low_group_mean' in result['comparison_results']['metric']
        assert 'difference' in result['comparison_results']['metric']
    
    def test_mean_split(self):
        """Test splitting groups by mean."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5],
            'metric': [10, 20, 30, 40, 50]
        })
        
        result = split_and_compare_groups(
            df,
            split_column='score',
            metrics=['metric'],
            split_method='mean'
        )
        
        assert 'comparison_results' in result
    
    def test_custom_split_value(self):
        """Test with custom split value."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'metric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = split_and_compare_groups(
            df,
            split_column='score',
            split_value=5.0,
            metrics=['metric']
        )
        
        assert 'comparison_results' in result
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = split_and_compare_groups(
            df,
            split_column='score',
            metrics=['metric']
        )
        
        assert result == {}
    
    def test_missing_split_column(self):
        """Test with missing split column."""
        df = pd.DataFrame({
            'metric': [10, 20, 30]
        })
        
        result = split_and_compare_groups(
            df,
            split_column='nonexistent',
            metrics=['metric']
        )
        
        assert result == {}
    
    def test_auto_detect_metrics(self):
        """Test automatic detection of metric columns."""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5],
            'metric1': [10, 20, 30, 40, 50],
            'metric2': [5, 10, 15, 20, 25],
            'category': ['A', 'B', 'C', 'D', 'E']
        })
        
        result = split_and_compare_groups(
            df,
            split_column='score'
        )
        
        assert 'comparison_results' in result
        # Should include metric1 and metric2, but not category or score
        comparison = result['comparison_results']
        assert 'metric1' in comparison
        assert 'metric2' in comparison
        assert 'category' not in comparison


class TestAnalyzeCorrelations:
    """Test analyze_correlations function."""
    
    def test_basic_correlation(self):
        """Test basic correlation analysis."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'metric1': [2, 4, 6, 8, 10],  # Perfect positive correlation
            'metric2': [5, 4, 3, 2, 1]    # Perfect negative correlation
        })
        
        result = analyze_correlations(df, 'target', ['metric1', 'metric2'])
        
        assert 'metric1' in result
        assert 'metric2' in result
        assert result['metric1'] > 0.9  # Strong positive correlation
        assert result['metric2'] < -0.9  # Strong negative correlation
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = analyze_correlations(df, 'target')
        assert result == {}
    
    def test_missing_target_column(self):
        """Test with missing target column."""
        df = pd.DataFrame({
            'metric': [1, 2, 3]
        })
        
        result = analyze_correlations(df, 'nonexistent')
        assert result == {}
    
    def test_min_data_points(self):
        """Test minimum data points threshold."""
        df = pd.DataFrame({
            'target': [1, 2],
            'metric': [2, 4]
        })
        
        # Require at least 5 data points
        result = analyze_correlations(
            df,
            'target',
            ['metric'],
            min_data_points=5
        )
        
        assert result == {}
    
    def test_filter_condition(self):
        """Test with filter condition."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'metric': [2, 4, 6, 8, 10],
            'category': ['A', 'A', 'B', 'B', 'B']
        })
        
        # Filter to only category B
        filter_func = lambda df: df[df['category'] == 'B']
        
        result = analyze_correlations(
            df,
            'target',
            ['metric'],
            filter_condition=filter_func
        )
        
        assert 'metric' in result
    
    def test_auto_detect_metrics(self):
        """Test automatic detection of metric columns."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'metric1': [2, 4, 6, 8, 10],
            'metric2': [1, 3, 5, 7, 9],
            'category': ['A', 'B', 'C', 'D', 'E']
        })
        
        result = analyze_correlations(df, 'target')
        
        # Should include metric1 and metric2, but not category or target itself
        assert 'metric1' in result
        assert 'metric2' in result
        assert 'category' not in result
        assert 'target' not in result


class TestGroupAndAnalyze:
    """Test group_and_analyze function."""
    
    def test_basic_grouping(self):
        """Test basic group analysis."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        def analysis_func(group_df):
            return {
                'count': len(group_df),
                'mean_value': group_df['value'].mean()
            }
        
        result = group_and_analyze(
            df,
            'category',
            ['A', 'B', 'C'],
            analysis_func
        )
        
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert result['A']['count'] == 2
        assert result['B']['count'] == 2
        assert result['C']['count'] == 2
    
    def test_min_group_size(self):
        """Test minimum group size filtering."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'C', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        def analysis_func(group_df):
            return {'count': len(group_df)}
        
        result = group_and_analyze(
            df,
            'category',
            ['A', 'B', 'C'],
            analysis_func,
            min_group_size=3
        )
        
        # Only C has 3 or more members
        assert 'C' in result
        assert 'A' not in result  # Only has 2 members
        assert 'B' not in result  # Only has 1 member
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        def analysis_func(group_df):
            return {}
        
        result = group_and_analyze(
            df,
            'category',
            ['A', 'B'],
            analysis_func
        )
        
        assert result == {}
    
    def test_missing_group_column(self):
        """Test with missing group column."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        })
        
        def analysis_func(group_df):
            return {}
        
        result = group_and_analyze(
            df,
            'nonexistent',
            ['A', 'B'],
            analysis_func
        )
        
        assert result == {}


class TestFindTopCorrelations:
    """Test find_top_correlations function."""
    
    def test_find_top_correlations(self):
        """Test finding top correlations."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'pos1': [1, 2, 3, 4, 5],      # Perfect positive
            'pos2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Strong positive
            'neg1': [5, 4, 3, 2, 1],      # Perfect negative
            'neg2': [4.9, 3.9, 2.9, 1.9, 0.9],  # Strong negative
            'weak': [1, 1, 1, 1, 2]       # Weak correlation
        })
        
        result = find_top_correlations(
            df,
            'target',
            top_n=2,
            min_correlation=0.5
        )
        
        assert 'top_positive' in result
        assert 'top_negative' in result
        assert len(result['top_positive']) <= 2
        assert len(result['top_negative']) <= 2
    
    def test_min_correlation_threshold(self):
        """Test minimum correlation threshold."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'strong': [1, 2, 3, 4, 5],    # Strong correlation (1.0)
            'weak': [1, 1, 1, 1, 2]       # Weak correlation
        })
        
        result = find_top_correlations(
            df,
            'target',
            min_correlation=0.9
        )
        
        # Only strong correlation should be included
        top_pos = result['top_positive']
        assert 'strong' in top_pos
        assert 'weak' not in top_pos
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = find_top_correlations(df, 'target')
        
        assert result == {'top_positive': {}, 'top_negative': {}}
    
    def test_no_correlations_above_threshold(self):
        """Test when no correlations meet threshold."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'uncorrelated': [3, 1, 4, 2, 5]
        })
        
        result = find_top_correlations(
            df,
            'target',
            min_correlation=0.9
        )
        
        # No strong correlations
        assert result['top_positive'] == {}
        assert result['top_negative'] == {}
