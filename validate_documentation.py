#!/usr/bin/env python3
"""
Documentation Validation Script

This script validates that all documentation is consistent, complete,
and that examples work correctly.
"""

import sys
import os
from pathlib import Path
import json
import re
from typing import List, Dict, Set

# Add the analysis module to path
sys.path.append(str(Path(__file__).parent))

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()

def check_file_content(file_path: str, required_sections: List[str]) -> Dict[str, bool]:
    """Check if a file contains required sections."""
    if not check_file_exists(file_path):
        return {section: False for section in required_sections}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {}
        for section in required_sections:
            results[section] = section.lower() in content.lower()
        
        return results
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {section: False for section in required_sections}

def validate_api_documentation():
    """Validate API documentation completeness."""
    print("Validating API Documentation...")
    
    api_file = "API_DOCUMENTATION.md"
    required_sections = [
        "SimulationAnalyzer",
        "analyze_population_dynamics",
        "analyze_resource_distribution", 
        "analyze_agent_interactions",
        "analyze_temporal_patterns",
        "analyze_advanced_time_series_models",
        "analyze_with_advanced_ml",
        "run_complete_analysis",
        "ReproducibilityManager",
        "AnalysisValidator"
    ]
    
    results = check_file_content(api_file, required_sections)
    
    print(f"  API Documentation ({api_file}):")
    for section, found in results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(results.values())

def validate_user_guide():
    """Validate user guide completeness."""
    print("\nValidating User Guide...")
    
    guide_file = "USER_GUIDE.md"
    required_sections = [
        "Getting Started",
        "Basic Usage",
        "Time Series Analysis",
        "Advanced Modeling",
        "Statistical Methods",
        "Visualization",
        "Reproducibility",
        "Troubleshooting",
        "Best Practices",
        "Examples"
    ]
    
    results = check_file_content(guide_file, required_sections)
    
    print(f"  User Guide ({guide_file}):")
    for section, found in results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(results.values())

def validate_time_series_guide():
    """Validate time series analysis guide."""
    print("\nValidating Time Series Analysis Guide...")
    
    guide_file = "TIME_SERIES_ANALYSIS_GUIDE.md"
    required_sections = [
        "Basic Time Series Analysis",
        "Advanced Time Series Modeling",
        "ARIMA Modeling",
        "Vector Autoregression",
        "Exponential Smoothing",
        "Model Comparison",
        "Statistical Methods Reference",
        "Best Practices",
        "Troubleshooting"
    ]
    
    results = check_file_content(guide_file, required_sections)
    
    print(f"  Time Series Guide ({guide_file}):")
    for section, found in results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(results.values())

def validate_test_documentation():
    """Validate test documentation."""
    print("\nValidating Test Documentation...")
    
    test_doc_file = "TEST_DOCUMENTATION.md"
    required_sections = [
        "test_simulation_analysis.py",
        "test_phase2_improvements.py", 
        "test_advanced_time_series.py",
        "Test Coverage",
        "Mock Data Generation",
        "Error Handling Coverage",
        "Performance Testing",
        "Reproducibility Testing"
    ]
    
    results = check_file_content(test_doc_file, required_sections)
    
    print(f"  Test Documentation ({test_doc_file}):")
    for section, found in results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(results.values())

def validate_test_coverage_report():
    """Validate test coverage report."""
    print("\nValidating Test Coverage Report...")
    
    coverage_file = "TEST_COVERAGE_REPORT.md"
    required_sections = [
        "Test Suite Structure",
        "Test Coverage Summary",
        "Statistical Methods Coverage",
        "Error Handling Coverage",
        "Performance Testing",
        "Reproducibility Testing",
        "Integration Testing"
    ]
    
    results = check_file_content(coverage_file, required_sections)
    
    print(f"  Test Coverage Report ({coverage_file}):")
    for section, found in results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(results.values())

def validate_phase_documentation():
    """Validate phase improvement documentation."""
    print("\nValidating Phase Documentation...")
    
    phase1_file = "PHASE_1_IMPROVEMENTS.md"
    phase2_file = "PHASE_2_IMPROVEMENTS.md"
    
    phase1_sections = [
        "Critical Fixes",
        "Statistical Methods",
        "Error Handling",
        "Unit Tests"
    ]
    
    phase2_sections = [
        "Advanced Time Series Analysis",
        "Advanced Machine Learning",
        "Effect Size Calculations",
        "Power Analysis",
        "Reproducibility Framework"
    ]
    
    phase1_results = check_file_content(phase1_file, phase1_sections)
    phase2_results = check_file_content(phase2_file, phase2_sections)
    
    print(f"  Phase 1 Documentation ({phase1_file}):")
    for section, found in phase1_results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    print(f"  Phase 2 Documentation ({phase2_file}):")
    for section, found in phase2_results.items():
        status = "✓" if found else "✗"
        print(f"    {status} {section}")
    
    return all(phase1_results.values()) and all(phase2_results.values())

def validate_test_files():
    """Validate that test files exist and have expected content."""
    print("\nValidating Test Files...")
    
    test_files = [
        "tests/test_simulation_analysis.py",
        "tests/test_phase2_improvements.py",
        "tests/test_advanced_time_series.py"
    ]
    
    all_exist = True
    for test_file in test_files:
        exists = check_file_exists(test_file)
        status = "✓" if exists else "✗"
        print(f"  {status} {test_file}")
        if not exists:
            all_exist = False
    
    return all_exist

def validate_example_files():
    """Validate that example files exist."""
    print("\nValidating Example Files...")
    
    example_files = [
        "examples/time_series_analysis_example.py",
        "examples/analysis_example.py"
    ]
    
    all_exist = True
    for example_file in example_files:
        exists = check_file_exists(example_file)
        status = "✓" if exists else "✗"
        print(f"  {status} {example_file}")
        if not exists:
            all_exist = False
    
    return all_exist

def validate_readme_files():
    """Validate README files."""
    print("\nValidating README Files...")
    
    readme_files = [
        "README.md",
        "README_TIME_SERIES.md",
        "TIME_SERIES_SUMMARY.md"
    ]
    
    all_exist = True
    for readme_file in readme_files:
        exists = check_file_exists(readme_file)
        status = "✓" if exists else "✗"
        print(f"  {status} {readme_file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_code_examples():
    """Check that code examples in documentation are syntactically valid."""
    print("\nValidating Code Examples...")
    
    # Files that contain code examples
    doc_files = [
        "API_DOCUMENTATION.md",
        "USER_GUIDE.md", 
        "TIME_SERIES_ANALYSIS_GUIDE.md",
        "README_TIME_SERIES.md"
    ]
    
    python_code_blocks = []
    
    for doc_file in doc_files:
        if not check_file_exists(doc_file):
            continue
            
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find Python code blocks
            code_pattern = r'```python\n(.*?)\n```'
            matches = re.findall(code_pattern, content, re.DOTALL)
            
            for i, code in enumerate(matches):
                python_code_blocks.append({
                    'file': doc_file,
                    'block': i + 1,
                    'code': code
                })
        
        except Exception as e:
            print(f"  Error reading {doc_file}: {e}")
    
    print(f"  Found {len(python_code_blocks)} Python code blocks in documentation")
    
    # Basic syntax validation (check for common issues)
    valid_blocks = 0
    for block in python_code_blocks:
        code = block['code']
        
        # Check for basic syntax issues
        issues = []
        
        # Check for unmatched quotes
        if code.count('"') % 2 != 0:
            issues.append("Unmatched double quotes")
        if code.count("'") % 2 != 0:
            issues.append("Unmatched single quotes")
        
        # Check for unmatched parentheses
        if code.count('(') != code.count(')'):
            issues.append("Unmatched parentheses")
        if code.count('[') != code.count(']'):
            issues.append("Unmatched square brackets")
        if code.count('{') != code.count('}'):
            issues.append("Unmatched curly braces")
        
        if not issues:
            valid_blocks += 1
        else:
            print(f"    Issues in {block['file']} block {block['block']}: {', '.join(issues)}")
    
    print(f"  Valid code blocks: {valid_blocks}/{len(python_code_blocks)}")
    
    return valid_blocks == len(python_code_blocks)

def check_documentation_consistency():
    """Check for consistency across documentation files."""
    print("\nValidating Documentation Consistency...")
    
    # Check that method names are consistent
    method_names = [
        "analyze_population_dynamics",
        "analyze_resource_distribution",
        "analyze_agent_interactions", 
        "analyze_temporal_patterns",
        "analyze_advanced_time_series_models",
        "analyze_with_advanced_ml",
        "run_complete_analysis"
    ]
    
    doc_files = [
        "API_DOCUMENTATION.md",
        "USER_GUIDE.md",
        "TIME_SERIES_ANALYSIS_GUIDE.md"
    ]
    
    consistency_issues = []
    
    for method in method_names:
        for doc_file in doc_files:
            if not check_file_exists(doc_file):
                continue
                
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if method not in content:
                    consistency_issues.append(f"{method} not found in {doc_file}")
            
            except Exception as e:
                consistency_issues.append(f"Error reading {doc_file}: {e}")
    
    if consistency_issues:
        print("  Consistency Issues:")
        for issue in consistency_issues:
            print(f"    • {issue}")
        return False
    else:
        print("  All method names are consistent across documentation")
        return True

def generate_validation_report():
    """Generate a comprehensive validation report."""
    print("DOCUMENTATION VALIDATION REPORT")
    print("=" * 60)
    
    validation_results = {
        "API Documentation": validate_api_documentation(),
        "User Guide": validate_user_guide(),
        "Time Series Guide": validate_time_series_guide(),
        "Test Documentation": validate_test_documentation(),
        "Test Coverage Report": validate_test_coverage_report(),
        "Phase Documentation": validate_phase_documentation(),
        "Test Files": validate_test_files(),
        "Example Files": validate_example_files(),
        "README Files": validate_readme_files(),
        "Code Examples": check_code_examples(),
        "Documentation Consistency": check_documentation_consistency()
    }
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(validation_results)
    
    for category, result in validation_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status:4} {category}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} validation categories passed")
    
    if passed == total:
        print("🎉 All documentation validation checks passed!")
        return 0
    else:
        print(f"❌ {total - passed} validation categories failed")
        return 1

def main():
    """Main validation function."""
    return generate_validation_report()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)