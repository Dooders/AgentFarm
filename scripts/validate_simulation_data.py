#!/usr/bin/env python3
"""
Validation script to ensure all tables and columns have data after a simulation runs.

This script connects to a simulation database and validates:
1. All expected tables exist
2. All tables have at least one row of data
3. All columns in each table have non-null values for at least one row
4. Reports any missing or empty tables/columns

Usage:
    python scripts/validate_simulation_data.py --db-path simulations/simulation.db
    python scripts/validate_simulation_data.py --db-path simulations/simulation.db --strict
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add workspace to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Import Base to get all registered tables
# This automatically loads all models via the models module
from farm.database.models import Base
from farm.database.research_models import Base as ResearchBase

try:
    from farm.utils.logging_config import configure_logging, get_logger
    # Configure logging
    configure_logging(environment="production", log_level="INFO")
    logger = get_logger(__name__)
    USE_LOGGING = True
except Exception:
    # Fallback to basic logging if farm logging not available
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    USE_LOGGING = False


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.missing_tables: List[str] = []
        self.empty_tables: List[str] = []
        self.columns_with_all_nulls: Dict[str, List[str]] = {}
        self.tables_with_data: List[Tuple[str, int]] = []
        self.warnings: List[str] = []

    def is_valid(self) -> bool:
        """Check if validation passed (no critical issues)."""
        return (
            len(self.missing_tables) == 0
            and len(self.empty_tables) == 0
            and len(self.columns_with_all_nulls) == 0
        )

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


def get_expected_tables() -> Set[str]:
    """Get the set of expected table names from SQLAlchemy models."""
    # Get all tables from Base metadata
    return {table.name for table in Base.metadata.tables.values()}


def validate_database(db_path: str, strict: bool = False) -> ValidationResult:
    """
    Validate that all tables and columns have data in the simulation database.

    Parameters
    ----------
    db_path : str
        Path to the simulation database file
    strict : bool
        If True, treat warnings as errors (optional tables must have data)

    Returns
    -------
    ValidationResult
        Object containing validation results
    """
    result = ValidationResult()

    # Check if database file exists
    if not Path(db_path).exists():
        logger.error("database_not_found", db_path=db_path)
        result.missing_tables = ["DATABASE_FILE_NOT_FOUND"]
        return result

    # Create database connection
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    inspector = inspect(engine)

    # Get expected tables from models
    expected_tables = get_expected_tables()
    logger.info(
        "validation_starting", db_path=db_path, expected_tables=len(expected_tables)
    )

    # Optional tables that may not have data in all simulations
    optional_tables = {
        "experiments",
        "research",
        "experiment_stats",
        "iteration_stats",
        "simulation_config",
    }

    # Get actual tables in database
    actual_tables = set(inspector.get_table_names())

    # Check for missing tables
    missing = expected_tables - actual_tables
    if missing:
        result.missing_tables = sorted(missing)
        logger.error("missing_tables", tables=result.missing_tables)

    # Validate each table
    for table_name in sorted(expected_tables & actual_tables):
        try:
            # Get row count
            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
            row_count = session.execute(count_query).scalar()

            if row_count == 0:
                if table_name in optional_tables and not strict:
                    result.warnings.append(
                        f"Optional table '{table_name}' is empty (acceptable)"
                    )
                    logger.warning("empty_optional_table", table=table_name)
                else:
                    result.empty_tables.append(table_name)
                    logger.error("empty_table", table=table_name)
                continue

            result.tables_with_data.append((table_name, row_count))
            logger.info("table_validated", table=table_name, rows=row_count)

            # Get columns for this table
            columns = [col["name"] for col in inspector.get_columns(table_name)]

            # Check each column for all-NULL values
            columns_with_nulls = []
            for column_name in columns:
                # Check if all values in this column are NULL
                null_check_query = text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL"
                )
                non_null_count = session.execute(null_check_query).scalar()

                if non_null_count == 0:
                    columns_with_nulls.append(column_name)
                    logger.warning(
                        "column_all_nulls",
                        table=table_name,
                        column=column_name,
                        total_rows=row_count,
                    )

            if columns_with_nulls:
                result.columns_with_all_nulls[table_name] = columns_with_nulls

        except Exception as e:
            logger.error(
                "validation_error",
                table=table_name,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            result.warnings.append(
                f"Error validating table '{table_name}': {str(e)}"
            )

    session.close()
    return result


def print_validation_report(result: ValidationResult, verbose: bool = False):
    """
    Print a formatted validation report.

    Parameters
    ----------
    result : ValidationResult
        Validation results to print
    verbose : bool
        If True, print detailed information about all tables
    """
    print("\n" + "=" * 80)
    print("SIMULATION DATA VALIDATION REPORT")
    print("=" * 80)

    # Missing tables
    if result.missing_tables:
        print("\n❌ MISSING TABLES:")
        for table in result.missing_tables:
            print(f"  - {table}")

    # Empty tables
    if result.empty_tables:
        print("\n❌ EMPTY TABLES (NO DATA):")
        for table in result.empty_tables:
            print(f"  - {table}")

    # Columns with all nulls
    if result.columns_with_all_nulls:
        print("\n⚠️  COLUMNS WITH ALL NULL VALUES:")
        for table, columns in result.columns_with_all_nulls.items():
            print(f"  Table: {table}")
            for column in columns:
                print(f"    - {column}")

    # Warnings
    if result.warnings:
        print("\n⚠️  WARNINGS:")
        for warning in result.warnings:
            print(f"  - {warning}")

    # Tables with data (only in verbose mode)
    if verbose and result.tables_with_data:
        print("\n✅ TABLES WITH DATA:")
        for table, count in sorted(result.tables_with_data, key=lambda x: x[0]):
            print(f"  - {table}: {count:,} rows")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Total tables expected: {len(get_expected_tables())}")
    print(f"  Tables with data: {len(result.tables_with_data)}")
    print(f"  Missing tables: {len(result.missing_tables)}")
    print(f"  Empty tables: {len(result.empty_tables)}")
    print(
        f"  Tables with null columns: {len(result.columns_with_all_nulls)}"
    )
    print(f"  Warnings: {len(result.warnings)}")

    if result.is_valid():
        print("\n✅ VALIDATION PASSED - All required tables and columns have data")
    else:
        print("\n❌ VALIDATION FAILED - Some tables or columns are missing or empty")

    print("=" * 80 + "\n")


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate simulation database has data in all tables and columns"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the simulation database file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: treat warnings as errors (optional tables must have data)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output: show all tables with data",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if validation fails",
    )

    args = parser.parse_args()

    # Run validation
    result = validate_database(args.db_path, strict=args.strict)

    # Print report
    print_validation_report(result, verbose=args.verbose)

    # Exit with appropriate code if requested
    if args.exit_code:
        if not result.is_valid():
            sys.exit(1)
        if args.strict and result.has_warnings():
            sys.exit(1)

    return result


if __name__ == "__main__":
    main()
