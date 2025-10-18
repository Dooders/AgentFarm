#!/usr/bin/env python3
"""
Standalone CLI tool for validating simulation databases.

This script provides a command-line interface for validating simulation databases
without running a full simulation. It can be used to check existing databases
for data integrity and consistency issues.

Usage:
    python scripts/validate_simulation_db.py path/to/simulation.db
    python scripts/validate_simulation_db.py --verbose path/to/simulation.db
    python scripts/validate_simulation_db.py --json path/to/simulation.db
    python scripts/validate_simulation_db.py --checks integrity path/to/simulation.db

Exit codes:
    0: All validations passed
    1: Warnings found (non-critical issues)
    2: Errors found (critical issues)
    3: Validation failed (could not connect to database)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from farm.database.validation import validate_simulation_database, ValidationSeverity
from farm.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for the validation CLI tool."""
    parser = argparse.ArgumentParser(
        description="Validate simulation database for integrity and consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s simulation.db                    # Basic validation
  %(prog)s --verbose simulation.db          # Show detailed results
  %(prog)s --json simulation.db             # Output JSON format
  %(prog)s --checks integrity simulation.db # Only run integrity checks
  %(prog)s --checks statistical simulation.db # Only run statistical checks
        """
    )
    
    parser.add_argument(
        "database_path",
        type=str,
        help="Path to the SQLite simulation database file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed validation results"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--checks", "-c",
        choices=["integrity", "statistical", "all"],
        default="all",
        help="Which validation categories to run (default: all)"
    )
    
    parser.add_argument(
        "--simulation-id",
        type=str,
        help="Specific simulation ID to validate (if not provided, validates all data)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level (default: WARNING)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(
        environment="development",
        log_level=args.log_level,
        disable_console=False
    )
    
    # Validate database path
    db_path = Path(args.database_path)
    if not db_path.exists():
        print(f"Error: Database file does not exist: {db_path}", file=sys.stderr)
        return 3
    
    if not db_path.is_file():
        print(f"Error: Path is not a file: {db_path}", file=sys.stderr)
        return 3
    
    # Determine which validators to run
    include_integrity = args.checks in ["integrity", "all"]
    include_statistical = args.checks in ["statistical", "all"]
    
    try:
        # Run validation
        logger.info(f"Starting validation of database: {db_path}")
        report = validate_simulation_database(
            database_path=str(db_path),
            simulation_id=args.simulation_id,
            include_integrity=include_integrity,
            include_statistical=include_statistical
        )
        
        # Output results
        if args.json:
            output_json_report(report)
        else:
            output_text_report(report, verbose=args.verbose)
        
        # Determine exit code
        if report.has_errors():
            return 2
        elif report.has_warnings():
            return 1
        else:
            return 0
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.json:
            error_report = {
                "error": str(e),
                "database_path": str(db_path),
                "success": False
            }
            print(json.dumps(error_report, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 3


def output_json_report(report) -> None:
    """Output validation report in JSON format."""
    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2))


def output_text_report(report, verbose: bool = False) -> None:
    """Output validation report in human-readable text format."""
    print(report.get_summary())
    
    if verbose and report.failed_checks > 0:
        print("\nDetailed Results:")
        print("=" * 50)
        
        for result in report.results:
            if not result.passed:
                print(f"\n[{result.severity.value.upper()}] {result.check_name}")
                print(f"  Message: {result.message}")
                if result.violation_count > 1:
                    print(f"  Violations: {result.violation_count}")
                if result.details:
                    print(f"  Details: {json.dumps(result.details, indent=4)}")
    
    # Print summary statistics
    print(f"\nValidation completed in {report.end_time - report.start_time:.2f} seconds")
    
    if report.is_clean():
        print("✅ All validations passed!")
    elif report.has_errors():
        print(f"❌ Found {report.error_count} errors and {report.warning_count} warnings")
    elif report.has_warnings():
        print(f"⚠️  Found {report.warning_count} warnings (no errors)")


if __name__ == "__main__":
    sys.exit(main())
