#!/usr/bin/env python3
"""
Test runner for simulation analysis tests.

This script runs the unit tests for the analysis module and provides
a summary of the results.
"""

import sys
import unittest
from pathlib import Path

# Add the workspace root to the Python path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

def run_analysis_tests():
    """Run the analysis module tests."""
    print("=" * 60)
    print("Running Simulation Analysis Tests")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = workspace_root / 'tests'
    
    # Run all test files
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    print("\nFound test files:")
    for test_file in start_dir.glob('test_*.py'):
        print(f"  ✓ {test_file.name}")
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 
        'sklearn', 'sqlalchemy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies available!")
    return True

def main():
    """Main test runner function."""
    print("Simulation Analysis Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nCannot run tests due to missing dependencies.")
        sys.exit(1)
    
    print()
    
    # Run tests
    success = run_analysis_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()