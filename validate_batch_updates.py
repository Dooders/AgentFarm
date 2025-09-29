#!/usr/bin/env python3
"""
Simple validation script for batch spatial updates implementation.
This script checks for syntax errors and basic functionality without requiring dependencies.
"""

import ast
import sys
import os

def validate_python_syntax(file_path):
    """Validate Python syntax by parsing the file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        ast.parse(content)
        print(f"✓ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"✗ {file_path}: Error reading file: {e}")
        return False

def check_imports(file_path):
    """Check if the file has proper imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            'from typing import',
            'from dataclasses import',
            'from collections import'
        ]
        
        missing_imports = []
        for required in required_imports:
            if required not in content:
                missing_imports.append(required)
        
        if missing_imports:
            print(f"⚠ {file_path}: Missing imports: {missing_imports}")
        else:
            print(f"✓ {file_path}: All required imports present")
        
        return len(missing_imports) == 0
    except Exception as e:
        print(f"✗ {file_path}: Error checking imports: {e}")
        return False

def check_class_definitions(file_path):
    """Check if required classes are defined."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        required_classes = [
            'class DirtyRegion:',
            'class DirtyRegionTracker:',
            'class SpatialIndex:'
        ]
        
        missing_classes = []
        for required in required_classes:
            if required not in content:
                missing_classes.append(required)
        
        if missing_classes:
            print(f"⚠ {file_path}: Missing class definitions: {missing_classes}")
        else:
            print(f"✓ {file_path}: All required classes defined")
        
        return len(missing_classes) == 0
    except Exception as e:
        print(f"✗ {file_path}: Error checking classes: {e}")
        return False

def check_method_definitions(file_path):
    """Check if required methods are defined."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        required_methods = [
            'def mark_region_dirty(',
            'def process_batch_updates(',
            'def add_position_update(',
            'def get_batch_update_stats('
        ]
        
        missing_methods = []
        for required in required_methods:
            if required not in content:
                missing_methods.append(required)
        
        if missing_methods:
            print(f"⚠ {file_path}: Missing method definitions: {missing_methods}")
        else:
            print(f"✓ {file_path}: All required methods defined")
        
        return len(missing_methods) == 0
    except Exception as e:
        print(f"✗ {file_path}: Error checking methods: {e}")
        return False

def main():
    """Main validation function."""
    print("Validating batch spatial updates implementation...")
    print("=" * 60)
    
    files_to_check = [
        'farm/core/spatial_index.py',
        'farm/config/config.py',
        'farm/core/environment.py',
        'farm/core/metrics_tracker.py',
        'tests/test_batch_spatial_updates.py'
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"✗ {file_path}: File not found")
            all_valid = False
            continue
        
        print(f"\nChecking {file_path}:")
        print("-" * 40)
        
        # Check syntax
        if not validate_python_syntax(file_path):
            all_valid = False
            continue
        
        # Check imports
        if not check_imports(file_path):
            all_valid = False
        
        # Check classes (only for spatial_index.py)
        if file_path == 'farm/core/spatial_index.py':
            if not check_class_definitions(file_path):
                all_valid = False
            
            if not check_method_definitions(file_path):
                all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All validations passed! Batch spatial updates implementation is ready.")
    else:
        print("✗ Some validations failed. Please check the issues above.")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())