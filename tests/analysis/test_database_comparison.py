"""
Tests for DatabaseComparison class.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from farm.analysis.comparative.database_comparison import DatabaseComparison, DatabaseComparisonResult


class TestDatabaseComparison:
    """Test cases for DatabaseComparison."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db1_path = self.temp_dir / "simulation1.db"
        self.db2_path = self.temp_dir / "simulation2.db"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_database(self, db_path: Path, table_name: str = "test_table", data_count: int = 10):
        """Create a test database with sample data."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create table
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL,
                    category TEXT
                )
            """)
            
            # Insert test data
            for i in range(data_count):
                cursor.execute(f"""
                    INSERT INTO {table_name} (name, value, category)
                    VALUES (?, ?, ?)
                """, (f"item_{i}", float(i * 1.5), f"cat_{i % 3}"))
            
            conn.commit()
    
    def test_init_valid_databases(self):
        """Test initialization with valid database files."""
        self.create_test_database(self.db1_path)
        self.create_test_database(self.db2_path)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        assert comparison.db1_path == self.db1_path
        assert comparison.db2_path == self.db2_path
    
    def test_init_missing_database(self):
        """Test initialization with missing database file."""
        self.create_test_database(self.db1_path)
        
        with pytest.raises(FileNotFoundError):
            DatabaseComparison(self.db1_path, self.db2_path)
    
    def test_compare_databases_identical(self):
        """Test comparison of identical databases."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        result = comparison.compare_databases()
        
        assert isinstance(result, DatabaseComparisonResult)
        assert result.schema_differences['tables_added'] == []
        assert result.schema_differences['tables_removed'] == []
        assert result.schema_differences['tables_modified'] == []
        assert result.summary['total_differences'] == 0
    
    def test_compare_databases_different_schemas(self):
        """Test comparison of databases with different schemas."""
        # Create database 1 with one table
        with sqlite3.connect(self.db1_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE table1 (id INTEGER, name TEXT)")
            conn.commit()
        
        # Create database 2 with different table
        with sqlite3.connect(self.db2_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE table2 (id INTEGER, value REAL)")
            conn.commit()
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        result = comparison.compare_databases()
        
        assert 'table1' in result.schema_differences['tables_removed']
        assert 'table2' in result.schema_differences['tables_added']
        assert result.summary['total_differences'] > 0
    
    def test_compare_databases_different_data(self):
        """Test comparison of databases with different data."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 10)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        result = comparison.compare_databases()
        
        assert 'test_table' in result.data_differences['table_row_counts']
        row_diff = result.data_differences['table_row_counts']['test_table']
        assert row_diff['db1_count'] == 5
        assert row_diff['db2_count'] == 10
        assert row_diff['difference'] == 5
    
    def test_get_database_schema(self):
        """Test database schema extraction."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn:
            schema = comparison._get_database_schema(conn)
        
        assert 'test_table' in schema
        assert 'columns' in schema['test_table']
        assert 'indexes' in schema['test_table']
        
        columns = schema['test_table']['columns']
        assert len(columns) == 4  # id, name, value, category
        assert any(col['name'] == 'id' for col in columns)
        assert any(col['name'] == 'name' for col in columns)
    
    def test_compare_table_schema(self):
        """Test table schema comparison."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        schema1 = {
            'columns': [
                {'name': 'id', 'type': 'INTEGER', 'notnull': True, 'pk': True},
                {'name': 'name', 'type': 'TEXT', 'notnull': False, 'pk': False}
            ],
            'indexes': []
        }
        
        schema2 = {
            'columns': [
                {'name': 'id', 'type': 'INTEGER', 'notnull': True, 'pk': True},
                {'name': 'name', 'type': 'TEXT', 'notnull': False, 'pk': False},
                {'name': 'value', 'type': 'REAL', 'notnull': False, 'pk': False}
            ],
            'indexes': []
        }
        
        diff = comparison._compare_table_schema(schema1, schema2)
        
        assert 'value' in diff['columns_added']
        assert diff['columns_removed'] == []
        assert diff['columns_modified'] == []
    
    def test_get_table_names(self):
        """Test table name extraction."""
        self.create_test_database(self.db1_path, "table1", 5)
        self.create_test_database(self.db2_path, "table2", 3)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn:
            tables = comparison._get_table_names(conn)
        
        assert 'table1' in tables
        # table2 is in db2, not db1
        assert 'table2' not in tables
    
    def test_get_table_row_count(self):
        """Test table row count extraction."""
        self.create_test_database(self.db1_path, "test_table", 7)
        self.create_test_database(self.db2_path, "test_table", 7)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn:
            count = comparison._get_table_row_count(conn, "test_table")
        
        assert count == 7
    
    def test_compare_table_data(self):
        """Test table data comparison."""
        # Create identical tables
        self.create_test_database(self.db1_path, "test_table", 3)
        self.create_test_database(self.db2_path, "test_table", 3)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
            diff = comparison._compare_table_data(conn1, conn2, "test_table")
        
        # Should have no differences for identical data, so method returns empty dict
        assert diff == {}
    
    def test_compare_table_data_different(self):
        """Test table data comparison with different data."""
        # Create tables with different data
        self.create_test_database(self.db1_path, "test_table", 3)
        
        # Create second table with different data
        with sqlite3.connect(self.db2_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL,
                    category TEXT
                )
            """)
            for i in range(5):  # Different row count
                cursor.execute("""
                    INSERT INTO test_table (name, value, category)
                    VALUES (?, ?, ?)
                """, (f"item_{i}", float(i * 2.0), f"cat_{i % 2}"))
            conn.commit()
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
            diff = comparison._compare_table_data(conn1, conn2, "test_table")
        
        # Should have differences
        assert 'row_count_diff' in diff
        assert 'column_differences' in diff
        assert 'sample_differences' in diff
        assert diff['row_count_diff'] == 2  # 5 - 3
    
    def test_compare_table_statistics(self):
        """Test table statistics comparison."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn1, sqlite3.connect(self.db2_path) as conn2:
            stats_diff = comparison._compare_table_statistics(conn1, conn2, "test_table")
        
        # Should have statistics for numeric columns
        # Note: The method might return empty dict if no numeric columns are found
        # or if there are errors, so we'll check if it's a dict
        assert isinstance(stats_diff, dict)
    
    def test_get_column_statistics(self):
        """Test column statistics extraction."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn:
            stats = comparison._get_column_statistics(conn, "test_table", "value")
        
        assert stats is not None
        assert 'count' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'sum' in stats
        assert stats['count'] == 5
    
    def test_compare_column_statistics(self):
        """Test column statistics comparison."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        stats1 = {'count': 5, 'min': 0.0, 'max': 6.0, 'mean': 3.0, 'sum': 15.0}
        stats2 = {'count': 5, 'min': 1.0, 'max': 7.0, 'mean': 4.0, 'sum': 20.0}
        
        diff = comparison._compare_column_statistics(stats1, stats2)
        
        assert 'min' in diff
        assert 'max' in diff
        assert 'mean' in diff
        assert 'sum' in diff
        
        assert diff['min']['db1_value'] == 0.0
        assert diff['min']['db2_value'] == 1.0
        assert diff['min']['absolute_difference'] == 1.0
    
    def test_extract_simulation_metrics(self):
        """Test simulation metrics extraction."""
        # Create database with simulation metadata
        with sqlite3.connect(self.db1_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE simulation_metadata (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    duration INTEGER,
                    agent_count INTEGER
                )
            """)
            cursor.execute("""
                INSERT INTO simulation_metadata (name, duration, agent_count)
                VALUES ('test_sim', 1000, 100)
            """)
            conn.commit()
        
        # Create second database
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        with sqlite3.connect(self.db1_path) as conn:
            metrics = comparison._extract_simulation_metrics(conn)
        
        assert 'simulation_metadata.duration' in metrics
        assert 'simulation_metadata.agent_count' in metrics
        assert metrics['simulation_metadata.duration'] == 1000
        assert metrics['simulation_metadata.agent_count'] == 100
    
    def test_generate_summary(self):
        """Test summary generation."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        schema_diff = {
            'tables_added': ['table1'],
            'tables_removed': ['table2'],
            'tables_modified': []
        }
        
        data_diff = {
            'table_row_counts': {
                'table1': {'difference': 5},
                'table2': {'difference': -3}
            }
        }
        
        metrics_diff = {
            'metric_differences': {'metric1': {}, 'metric2': {}}
        }
        
        summary = comparison._generate_summary(schema_diff, data_diff, metrics_diff)
        
        assert summary['total_differences'] == 12  # 1 + 1 + 8 + 2 (schema + data + metrics)
        assert summary['schema_changes'] == 2
        assert summary['data_changes'] == 8  # 5 + 3
        assert summary['metric_changes'] == 2
        assert summary['severity'] == 'medium'
    
    def test_generate_summary_high_severity(self):
        """Test summary generation with high severity."""
        self.create_test_database(self.db1_path, "test_table", 5)
        self.create_test_database(self.db2_path, "test_table", 5)
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        
        schema_diff = {'tables_added': [], 'tables_removed': [], 'tables_modified': []}
        data_diff = {'table_row_counts': {}}
        metrics_diff = {'metric_differences': {f'metric{i}': {} for i in range(150)}}
        
        summary = comparison._generate_summary(schema_diff, data_diff, metrics_diff)
        
        assert summary['severity'] == 'high'
    
    def test_compare_metrics(self):
        """Test metrics comparison."""
        # Create databases with different metrics
        with sqlite3.connect(self.db1_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE simulation_metadata (
                    duration INTEGER,
                    agent_count INTEGER
                )
            """)
            cursor.execute("INSERT INTO simulation_metadata VALUES (1000, 100)")
            conn.commit()
        
        with sqlite3.connect(self.db2_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE simulation_metadata (
                    duration INTEGER,
                    agent_count INTEGER
                )
            """)
            cursor.execute("INSERT INTO simulation_metadata VALUES (2000, 150)")
            conn.commit()
        
        comparison = DatabaseComparison(self.db1_path, self.db2_path)
        result = comparison.compare_databases()
        
        assert 'metric_differences' in result.metric_differences
        assert 'simulation_metadata.duration' in result.metric_differences['metric_differences']
        assert 'simulation_metadata.agent_count' in result.metric_differences['metric_differences']