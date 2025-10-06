#!/usr/bin/env python3
"""
Example script: Batch validate multiple simulation databases

This script demonstrates how to validate multiple simulation databases
and generate a summary report.

Usage:
    python scripts/example_validate_batch.py simulations/
    python scripts/example_validate_batch.py --pattern "*.db" --output report.txt
"""

import argparse
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_simulation_data import validate_database, ValidationResult


def find_databases(directory: str, pattern: str = "*.db") -> list:
    """Find all database files in a directory."""
    path = Path(directory)
    if not path.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    databases = list(path.rglob(pattern))
    return sorted(databases)


def validate_batch(databases: list, strict: bool = False, output_file: str = None):
    """Validate multiple databases and generate a summary report."""
    results = {}
    
    print(f"\nValidating {len(databases)} database(s)...\n")
    print("=" * 80)
    
    for db_path in databases:
        print(f"\nValidating: {db_path}")
        print("-" * 80)
        
        result = validate_database(str(db_path), strict=strict)
        results[str(db_path)] = result
        
        # Print brief status
        if result.is_valid():
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            if result.missing_tables:
                print(f"   Missing tables: {len(result.missing_tables)}")
            if result.empty_tables:
                print(f"   Empty tables: {len(result.empty_tables)}")
            if result.columns_with_all_nulls:
                print(f"   Tables with null columns: {len(result.columns_with_all_nulls)}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r.is_valid())
    failed = len(results) - passed
    
    summary_lines = [
        "\n" + "=" * 80,
        "BATCH VALIDATION SUMMARY",
        "=" * 80,
        f"\nTotal databases: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        f"\nSuccess rate: {(passed/len(results)*100):.1f}%",
    ]
    
    if failed > 0:
        summary_lines.append("\nFailed databases:")
        for db_path, result in results.items():
            if not result.is_valid():
                summary_lines.append(f"  - {db_path}")
                if result.missing_tables:
                    summary_lines.append(f"    Missing: {', '.join(result.missing_tables)}")
                if result.empty_tables:
                    summary_lines.append(f"    Empty: {', '.join(result.empty_tables)}")
    
    summary_lines.append("\n" + "=" * 80)
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Write to output file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(summary_text)
        print(f"\nReport saved to: {output_file}")
    
    return passed == len(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch validate multiple simulation databases"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing simulation databases"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.db",
        help="File pattern to match (default: *.db)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: treat warnings as errors"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for summary report"
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if any validation fails"
    )
    
    args = parser.parse_args()
    
    # Find databases
    databases = find_databases(args.directory, args.pattern)
    
    if not databases:
        print(f"No databases found in {args.directory} matching pattern '{args.pattern}'")
        sys.exit(1)
    
    # Validate batch
    all_passed = validate_batch(
        databases,
        strict=args.strict,
        output_file=args.output
    )
    
    # Exit with appropriate code
    if args.exit_code and not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
