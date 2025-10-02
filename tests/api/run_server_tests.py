#!/usr/bin/env python3
"""
Test runner for FastAPI server tests.

This script runs all server-related tests including:
- Basic endpoint tests
- WebSocket functionality tests
- Background task tests
- Error handling tests
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


def run_server_tests():
    """Run all server-related tests."""
    print("Running FastAPI server tests...")
    print("=" * 50)

    # Test files to run
    test_files = [
        "tests/api/test_server.py",
        "tests/api/test_websocket.py",
        "tests/api/test_background_tasks.py",
    ]

    # Pytest arguments
    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ]

    # Add test files
    pytest_args.extend(test_files)

    # Run tests
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n" + "=" * 50)
        print("[SUCCESS] All server tests passed!")
    else:
        print("\n" + "=" * 50)
        print("[ERROR] Some server tests failed!")

    return exit_code


def run_specific_test_file(test_file: str):
    """Run a specific test file."""
    print(f"Running {test_file}...")
    print("=" * 50)

    pytest_args = [
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
        test_file,
    ]

    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print(f"\n[SUCCESS] {test_file} passed!")
    else:
        print(f"\n[ERROR] {test_file} failed!")

    return exit_code


def run_coverage_tests():
    """Run tests with coverage reporting."""
    print("Running server tests with coverage...")
    print("=" * 50)

    test_files = [
        "tests/api/test_server.py",
        "tests/api/test_websocket.py",
        "tests/api/test_background_tasks.py",
    ]

    pytest_args = [
        "-v",
        "--tb=short",
        "--cov=farm.api.server",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--disable-warnings",
    ]

    pytest_args.extend(test_files)

    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n" + "=" * 50)
        print("[SUCCESS] Coverage tests completed!")
        print("[INFO] HTML coverage report generated in htmlcov/")
    else:
        print("\n" + "=" * 50)
        print("[ERROR] Coverage tests failed!")

    return exit_code


def main():
    """Main entry point for the test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "server":
            return run_server_tests()
        elif command == "websocket":
            return run_specific_test_file("tests/api/test_websocket.py")
        elif command == "background":
            return run_specific_test_file("tests/api/test_background_tasks.py")
        elif command == "coverage":
            return run_coverage_tests()
        elif command == "help":
            print("Usage: python run_server_tests.py [command]")
            print("\nCommands:")
            print("  server     - Run all server tests (default)")
            print("  websocket  - Run WebSocket tests only")
            print("  background - Run background task tests only")
            print("  coverage   - Run tests with coverage reporting")
            print("  help       - Show this help message")
            return 0
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
            return 1
    else:
        # Default: run all server tests
        return run_server_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
