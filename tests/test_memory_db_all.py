#!/usr/bin/env python3
"""
Run all tests for the in-memory database implementation.

This script discovers and runs all unit and integration tests for the
in-memory database implementation.
"""

import argparse
import os
import sys
import unittest
import time


def run_tests(test_type="all", verbose=False):
    """
    Run the specified tests.
    
    Parameters
    ----------
    test_type : str
        Type of tests to run: "unit", "integration", or "all"
    verbose : bool
        Whether to show verbose output
    
    Returns
    -------
    bool
        True if all tests passed, False otherwise
    """
    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Determine which tests to run
    if test_type in ["unit", "all"]:
        print("Discovering unit tests...")
        unit_tests_path = os.path.join(project_root, "farm", "database")
        unit_tests = loader.discover(unit_tests_path, pattern="test_memory_db.py")
        suite.addTests(unit_tests)
        print(f"Found {unit_tests.countTestCases()} unit tests")
    
    if test_type in ["integration", "all"]:
        print("Discovering integration tests...")
        integration_tests_path = os.path.join(project_root, "farm", "core")
        integration_tests = loader.discover(integration_tests_path, pattern="test_memory_db_integration.py")
        suite.addTests(integration_tests)
        print(f"Found {integration_tests.countTestCases()} integration tests")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print(f"\nRunning {suite.countTestCases()} tests...")
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests in {end_time - start_time:.2f} seconds")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run in-memory database tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip performance tests)",
    )
    
    args = parser.parse_args()
    
    # Set environment variable for quick tests if requested
    if args.quick:
        os.environ["QUICK_TEST"] = "1"
        print("Running in quick mode (skipping performance tests)")
    
    # Run tests
    success = run_tests(args.type, args.verbose)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 