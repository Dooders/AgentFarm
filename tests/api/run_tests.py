#!/usr/bin/env python3
"""Test runner for API module tests."""

import sys
import subprocess
from pathlib import Path

def run_api_tests():
    """Run all API module tests."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Run pytest on the API tests
    cmd = [
        sys.executable, "-m", "pytest",
        str(project_root / "tests" / "api"),
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    print("Running API module tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        return_code = 0
    except subprocess.CalledProcessError as e:
        result = e
        return_code = e.returncode

    if return_code == 0:
        print("\n All API tests passed!")
    else:
        print("\n Some API tests failed!")

    return return_code

if __name__ == "__main__":
    exit_code = run_api_tests()
    sys.exit(exit_code)
