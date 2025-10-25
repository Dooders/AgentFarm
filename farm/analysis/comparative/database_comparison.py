"""
Database comparison utilities for SQLite simulation databases.

This module provides functionality to compare two SQLite databases
containing simulation data and extract meaningful differences.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseComparisonResult:
    """Result of database comparison."""
    
    schema_differences: Dict[str, Any]
    data_differences: Dict[str, Any]
    metric_differences: Dict[str, Any]
    summary: Dict[str, Any]


class DatabaseComparison:
    """Handles comparison of SQLite simulation databases."""
    
    def __init__(self, db1_path: Union[str, Path], db2_path: Union[str, Path]):
        """Initialize database comparison.
        
        Args:
            db1_path: Path to first database
            db2_path: Path to second database
        """
        self.db1_path = Path(db1_path)
        self.db2_path = Path(db2_path)
        
        # Validate database files exist
        if not self.db1_path.exists():
            raise FileNotFoundError(f"Database 1 not found: {self.db1_path}")
        if not self.db2_path.exists():
            raise FileNotFoundError(f"Database 2 not found: {self.db2_path}")
    
    def compare_databases(self) -> DatabaseComparisonResult:
        """Compare two SQLite databases comprehensively.
        
        Returns:
            DatabaseComparisonResult containing all comparison data
        """
        logger.info(f"Comparing databases: {self.db1_path} vs {self.db2_path}")
        
        # Compare schemas
        schema_differences = self._compare_schemas()
        
        # Compare data
        data_differences = self._compare_data()
        
        # Compare metrics
        metric_differences = self._compare_metrics()
        
        # Generate summary
        summary = self._generate_summary(schema_differences, data_differences, metric_differences)
        
        return DatabaseComparisonResult(
            schema_differences=schema_differences,
            data_differences=data_differences,
            metric_differences=metric_differences,
            summary=summary
        )
    
    def _compare_schemas(self) -> Dict[str, Any]:
        """Compare database schemas."""
        logger.debug("Comparing database schemas")
        
        try:
            with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
                schema1 = self._get_database_schema(conn1)
                schema2 = self._get_database_schema(conn2)
                
                differences = {
                    'tables_added': [],
                    'tables_removed': [],
                    'tables_modified': [],
                    'columns_added': {},
                    'columns_removed': {},
                    'columns_modified': {}
                }
                
                # Compare tables
                tables1 = set(schema1.keys())
                tables2 = set(schema2.keys())
                
                differences['tables_added'] = list(tables2 - tables1)
                differences['tables_removed'] = list(tables1 - tables2)
                
                # Compare common tables
                common_tables = tables1 & tables2
                for table in common_tables:
                    table_diff = self._compare_table_schema(schema1[table], schema2[table])
                    if table_diff:
                        differences['tables_modified'].append({
                            'table': table,
                            'differences': table_diff
                        })
                
                return differences
                
        except Exception as e:
            logger.error(f"Error comparing schemas: {e}")
            return {'error': str(e)}
    
    def _get_database_schema(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get complete database schema."""
        schema = {}
        
        # Get table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'cid': row[0],
                    'name': row[1],
                    'type': row[2],
                    'notnull': bool(row[3]),
                    'dflt_value': row[4],
                    'pk': bool(row[5])
                })
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = [row[1] for row in cursor.fetchall()]
            
            schema[table] = {
                'columns': columns,
                'indexes': indexes
            }
        
        return schema
    
    def _compare_table_schema(self, schema1: Dict, schema2: Dict) -> Dict[str, Any]:
        """Compare schemas of two tables."""
        differences = {
            'columns_added': [],
            'columns_removed': [],
            'columns_modified': []
        }
        
        cols1 = {col['name']: col for col in schema1['columns']}
        cols2 = {col['name']: col for col in schema2['columns']}
        
        # Find added/removed columns
        names1 = set(cols1.keys())
        names2 = set(cols2.keys())
        
        differences['columns_added'] = list(names2 - names1)
        differences['columns_removed'] = list(names1 - names2)
        
        # Compare common columns
        common_names = names1 & names2
        for name in common_names:
            col1 = cols1[name]
            col2 = cols2[name]
            
            if col1 != col2:
                differences['columns_modified'].append({
                    'column': name,
                    'old': col1,
                    'new': col2
                })
        
        return differences if any(differences.values()) else {}
    
    def _compare_data(self) -> Dict[str, Any]:
        """Compare data between databases."""
        logger.debug("Comparing database data")
        
        try:
            with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
                # Get common tables
                tables1 = self._get_table_names(conn1)
                tables2 = self._get_table_names(conn2)
                common_tables = set(tables1) & set(tables2)
                
                differences = {
                    'table_row_counts': {},
                    'data_differences': {},
                    'summary_stats': {}
                }
                
                for table in common_tables:
                    # Compare row counts
                    count1 = self._get_table_row_count(conn1, table)
                    count2 = self._get_table_row_count(conn2, table)
                    
                    differences['table_row_counts'][table] = {
                        'db1_count': count1,
                        'db2_count': count2,
                        'difference': count2 - count1
                    }
                    
                    # Compare sample data if tables are small enough
                    if count1 <= 1000 and count2 <= 1000:
                        data_diff = self._compare_table_data(conn1, conn2, table)
                        if data_diff:
                            differences['data_differences'][table] = data_diff
                    
                    # Compare summary statistics for numeric columns
                    stats_diff = self._compare_table_statistics(conn1, conn2, table)
                    if stats_diff:
                        differences['summary_stats'][table] = stats_diff
                
                return differences
                
        except Exception as e:
            logger.error(f"Error comparing data: {e}")
            return {'error': str(e)}
    
    def _get_table_names(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of table names."""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def _get_table_row_count(self, conn: sqlite3.Connection, table: str) -> int:
        """Get row count for a table."""
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]
    
    def _compare_table_data(self, conn1: sqlite3.Connection, conn2: sqlite3.Connection, table: str) -> Dict[str, Any]:
        """Compare actual data in two tables."""
        try:
            # Load data from both tables
            df1 = pd.read_sql_query(f"SELECT * FROM {table}", conn1)
            df2 = pd.read_sql_query(f"SELECT * FROM {table}", conn2)
            
            differences = {
                'row_count_diff': len(df2) - len(df1),
                'column_differences': [],
                'sample_differences': []
            }
            
            # Compare columns
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            
            if cols1 != cols2:
                differences['column_differences'] = {
                    'added_columns': list(cols2 - cols1),
                    'removed_columns': list(cols1 - cols2)
                }
            
            # Compare common columns
            common_cols = cols1 & cols2
            if common_cols and len(df1) > 0 and len(df2) > 0:
                # Find rows that exist in one but not the other
                if len(df1) == len(df2):
                    # Compare row by row
                    for i, (row1, row2) in enumerate(zip(df1.itertuples(), df2.itertuples())):
                        if row1 != row2:
                            differences['sample_differences'].append({
                                'row_index': i,
                                'db1_row': row1._asdict(),
                                'db2_row': row2._asdict()
                            })
                            if len(differences['sample_differences']) >= 10:  # Limit output
                                break
            
            return differences if any(differences.values()) else {}
            
        except Exception as e:
            logger.warning(f"Could not compare data for table {table}: {e}")
            return {'error': str(e)}
    
    def _compare_table_statistics(self, conn1: sqlite3.Connection, conn2: sqlite3.Connection, table: str) -> Dict[str, Any]:
        """Compare summary statistics for numeric columns."""
        try:
            # Get numeric columns
            cursor1 = conn1.cursor()
            cursor1.execute(f"PRAGMA table_info({table})")
            columns = cursor1.fetchall()
            
            numeric_cols = []
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()
                if any(t in col_type for t in ['INT', 'REAL', 'NUMERIC', 'FLOAT', 'DOUBLE']):
                    numeric_cols.append(col_name)
            
            if not numeric_cols:
                return {}
            
            stats_diff = {}
            for col in numeric_cols:
                try:
                    # Get statistics for column in both databases
                    stats1 = self._get_column_statistics(conn1, table, col)
                    stats2 = self._get_column_statistics(conn2, table, col)
                    
                    if stats1 and stats2:
                        col_diff = self._compare_column_statistics(stats1, stats2)
                        if col_diff:
                            stats_diff[col] = col_diff
                            
                except Exception as e:
                    logger.debug(f"Could not compare statistics for {table}.{col}: {e}")
                    continue
            
            return stats_diff
            
        except Exception as e:
            logger.warning(f"Could not compare statistics for table {table}: {e}")
            return {}
    
    def _get_column_statistics(self, conn: sqlite3.Connection, table: str, column: str) -> Optional[Dict[str, float]]:
        """Get statistics for a numeric column."""
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as count,
                    MIN({column}) as min_val,
                    MAX({column}) as max_val,
                    AVG({column}) as mean_val,
                    SUM({column}) as sum_val
                FROM {table} 
                WHERE {column} IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0] > 0:  # count > 0
                return {
                    'count': result[0],
                    'min': result[1],
                    'max': result[2],
                    'mean': result[3],
                    'sum': result[4]
                }
            return None
            
        except Exception as e:
            logger.debug(f"Could not get statistics for {table}.{column}: {e}")
            return None
    
    def _compare_column_statistics(self, stats1: Dict[str, float], stats2: Dict[str, float]) -> Dict[str, Any]:
        """Compare statistics for two columns."""
        differences = {}
        
        for stat_name in ['count', 'min', 'max', 'mean', 'sum']:
            if stat_name in stats1 and stat_name in stats2:
                val1 = stats1[stat_name]
                val2 = stats2[stat_name]
                
                if val1 != val2:
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                    
                    differences[stat_name] = {
                        'db1_value': val1,
                        'db2_value': val2,
                        'absolute_difference': diff,
                        'percentage_change': pct_change
                    }
        
        return differences
    
    def _compare_metrics(self) -> Dict[str, Any]:
        """Compare simulation metrics between databases."""
        logger.debug("Comparing simulation metrics")
        
        try:
            with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
                metrics1 = self._extract_simulation_metrics(conn1)
                metrics2 = self._extract_simulation_metrics(conn2)
                
                differences = {
                    'metric_differences': {},
                    'performance_comparison': {},
                    'summary': {}
                }
                
                # Compare common metrics
                common_metrics = set(metrics1.keys()) & set(metrics2.keys())
                
                for metric in common_metrics:
                    val1 = metrics1[metric]
                    val2 = metrics2[metric]
                    
                    if val1 != val2:
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            diff = val2 - val1
                            pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                            
                            differences['metric_differences'][metric] = {
                                'db1_value': val1,
                                'db2_value': val2,
                                'absolute_difference': diff,
                                'percentage_change': pct_change
                            }
                        else:
                            differences['metric_differences'][metric] = {
                                'db1_value': val1,
                                'db2_value': val2,
                                'type': 'non_numeric'
                            }
                
                # Identify added/removed metrics
                differences['added_metrics'] = list(set(metrics2.keys()) - set(metrics1.keys()))
                differences['removed_metrics'] = list(set(metrics1.keys()) - set(metrics2.keys()))
                
                return differences
                
        except Exception as e:
            logger.error(f"Error comparing metrics: {e}")
            return {'error': str(e)}
    
    def _extract_simulation_metrics(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Extract simulation metrics from database."""
        metrics = {}
        
        try:
            # Look for common simulation metric tables
            cursor = conn.cursor()
            
            # Check for simulation metadata
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%simulation%'")
            sim_tables = [row[0] for row in cursor.fetchall()]
            
            for table in sim_tables:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    columns = [desc[0] for desc in cursor.description]
                    row = cursor.fetchone()
                    
                    if row:
                        for col, val in zip(columns, row):
                            if isinstance(val, (int, float)):
                                metrics[f"{table}.{col}"] = val
                except Exception as e:
                    logger.debug(f"Could not extract metrics from {table}: {e}")
                    continue
            
            # Look for step metrics
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%step%'")
            step_tables = [row[0] for row in cursor.fetchall()]
            
            for table in step_tables:
                try:
                    # Get summary statistics for numeric columns
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2].upper()
                        
                        if any(t in col_type for t in ['INT', 'REAL', 'NUMERIC', 'FLOAT', 'DOUBLE']):
                            try:
                                cursor.execute(f"""
                                    SELECT 
                                        COUNT(*) as count,
                                        AVG({col_name}) as avg_val,
                                        MAX({col_name}) as max_val,
                                        MIN({col_name}) as min_val
                                    FROM {table}
                                    WHERE {col_name} IS NOT NULL
                                """)
                                
                                result = cursor.fetchone()
                                if result and result[0] > 0:
                                    metrics[f"{table}.{col_name}_count"] = result[0]
                                    metrics[f"{table}.{col_name}_avg"] = result[1]
                                    metrics[f"{table}.{col_name}_max"] = result[2]
                                    metrics[f"{table}.{col_name}_min"] = result[3]
                                    
                            except Exception as e:
                                logger.debug(f"Could not extract metric for {table}.{col_name}: {e}")
                                continue
                                
                except Exception as e:
                    logger.debug(f"Could not process step table {table}: {e}")
                    continue
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error extracting simulation metrics: {e}")
            return {}
    
    def _generate_summary(self, schema_diff: Dict, data_diff: Dict, metrics_diff: Dict) -> Dict[str, Any]:
        """Generate summary of all differences."""
        summary = {
            'total_differences': 0,
            'schema_changes': 0,
            'data_changes': 0,
            'metric_changes': 0,
            'severity': 'low'
        }
        
        # Count schema changes
        if 'tables_added' in schema_diff:
            summary['schema_changes'] += len(schema_diff['tables_added'])
        if 'tables_removed' in schema_diff:
            summary['schema_changes'] += len(schema_diff['tables_removed'])
        if 'tables_modified' in schema_diff:
            summary['schema_changes'] += len(schema_diff['tables_modified'])
        
        # Count data changes
        if 'table_row_counts' in data_diff:
            row_diffs = [abs(diff['difference']) for diff in data_diff['table_row_counts'].values()]
            summary['data_changes'] = sum(row_diffs)
        
        # Count metric changes
        if 'metric_differences' in metrics_diff:
            summary['metric_changes'] = len(metrics_diff['metric_differences'])
        
        summary['total_differences'] = summary['schema_changes'] + summary['data_changes'] + summary['metric_changes']
        
        # Determine severity
        if summary['total_differences'] > 100:
            summary['severity'] = 'high'
        elif summary['total_differences'] > 10:
            summary['severity'] = 'medium'
        
        return summary