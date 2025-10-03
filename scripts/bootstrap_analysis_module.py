#!/usr/bin/env python3
"""
Bootstrap script for creating new analysis modules.

This script creates the directory structure and template files
for a new analysis module following the established patterns.

Usage:
    python scripts/bootstrap_analysis_module.py <module_name>
    
Example:
    python scripts/bootstrap_analysis_module.py population
"""

import sys
from pathlib import Path
from typing import List


TEMPLATE_INIT = '''"""
{description}

Provides comprehensive analysis of {subject} including:
{features}
"""

from farm.analysis.{module_name}.module import {module_name}_module, {class_name}
from farm.analysis.{module_name}.compute import (
    # TODO: Add compute functions
)
from farm.analysis.{module_name}.analyze import (
    # TODO: Add analysis functions
)
from farm.analysis.{module_name}.plot import (
    # TODO: Add plot functions
)

__all__ = [
    "{module_name}_module",
    "{class_name}",
    # TODO: Add exports
]
'''


TEMPLATE_DATA = '''"""
{module_title} data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase


def process_{module_name}_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process {module_name} data from experiment.
    
    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options
        
    Returns:
        DataFrame with {module_name} metrics over time
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {{experiment_path}}")
    
    # TODO: Implement data loading
    # 1. Connect to database
    # 2. Load relevant data
    # 3. Transform to DataFrame
    # 4. Return processed data
    
    raise NotImplementedError("Implement data processing")
'''


TEMPLATE_COMPUTE = '''"""
{module_title} statistical computations.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_{module_name}_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive {module_name} statistics.
    
    Args:
        df: {module_title} data with required columns
        
    Returns:
        Dictionary of computed statistics
    """
    # TODO: Implement statistics computation
    # 1. Extract relevant columns
    # 2. Calculate statistics
    # 3. Compute derived metrics
    # 4. Return results dictionary
    
    raise NotImplementedError("Implement statistics computation")
'''


TEMPLATE_ANALYZE = '''"""
{module_title} analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.{module_name}.compute import (
    compute_{module_name}_statistics,
)


def analyze_{module_name}_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze {module_name} patterns and save results.
    
    Args:
        df: {module_title} data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing {module_name} patterns...")
    
    # TODO: Implement analysis
    # 1. Compute statistics
    # 2. Analyze patterns
    # 3. Save results to files
    # 4. Report progress
    
    raise NotImplementedError("Implement analysis")
'''


TEMPLATE_PLOT = '''"""
{module_title} visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_{module_name}_overview(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot {module_name} overview visualization.
    
    Args:
        df: {module_title} data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating {module_name} overview plot...")
    
    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # TODO: Implement visualization
    # 1. Create plot
    # 2. Add data
    # 3. Format axes
    # 4. Save figure
    
    raise NotImplementedError("Implement visualization")
'''


TEMPLATE_MODULE = '''"""
{module_title} analysis module implementation.
"""

from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

from farm.analysis.{module_name}.data import process_{module_name}_data
from farm.analysis.{module_name}.analyze import (
    analyze_{module_name}_patterns,
)
from farm.analysis.{module_name}.plot import (
    plot_{module_name}_overview,
)


class {class_name}(BaseAnalysisModule):
    """Module for analyzing {module_name} in simulations."""
    
    def __init__(self):
        super().__init__(
            name="{module_name}",
            description="{description}"
        )
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['step'],  # TODO: Update required columns
                column_types={{'step': int}}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)
    
    def register_functions(self) -> None:
        """Register all {module_name} analysis functions."""
        
        # Analysis functions
        self._functions = {{
            "analyze_patterns": make_analysis_function(analyze_{module_name}_patterns),
            "plot_overview": make_analysis_function(plot_{module_name}_overview),
            # TODO: Add more functions
        }}
        
        # Function groups
        self._groups = {{
            "all": list(self._functions.values()),
            "analysis": [
                self._functions["analyze_patterns"],
            ],
            "plots": [
                self._functions["plot_overview"],
            ],
            "basic": [
                self._functions["analyze_patterns"],
                self._functions["plot_overview"],
            ],
        }}
    
    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor for {module_name} analysis."""
        return SimpleDataProcessor(process_{module_name}_data)


# Create singleton instance
{module_name}_module = {class_name}()
'''


TEMPLATE_TEST = '''"""
Tests for {module_name} analysis module.
"""

import pytest
import pandas as pd
from pathlib import Path

from farm.analysis.{module_name} import (
    {module_name}_module,
    compute_{module_name}_statistics,
    analyze_{module_name}_patterns,
)
from farm.analysis.common.context import AnalysisContext


def test_{module_name}_module_registration():
    """Test module is properly registered."""
    assert {module_name}_module.name == "{module_name}"
    assert len({module_name}_module.get_function_names()) > 0


def test_compute_{module_name}_statistics():
    """Test {module_name} statistics computation."""
    # TODO: Create sample data
    df = pd.DataFrame({{
        'step': range(100),
        # Add more columns
    }})
    
    # TODO: Test statistics computation
    # stats = compute_{module_name}_statistics(df)
    # assert 'key_metric' in stats
    
    pytest.skip("TODO: Implement test")


def test_analyze_{module_name}_patterns(tmp_path):
    """Test {module_name} pattern analysis."""
    # TODO: Create sample data
    df = pd.DataFrame({{
        'step': range(100),
    }})
    
    # TODO: Test analysis
    # ctx = AnalysisContext(output_path=tmp_path)
    # analyze_{module_name}_patterns(df, ctx)
    # assert (tmp_path / "results.json").exists()
    
    pytest.skip("TODO: Implement test")


def test_{module_name}_module_integration(tmp_path):
    """Test full module execution."""
    pytest.skip("TODO: Implement integration test")
    
    # TODO: Add integration test
    # from farm.analysis.service import AnalysisService, AnalysisRequest
    # from farm.core.services import EnvConfigService
    # 
    # service = AnalysisService(EnvConfigService())
    # request = AnalysisRequest(
    #     module_name="{module_name}",
    #     experiment_path=sample_experiment_path,
    #     output_path=tmp_path,
    #     group="basic"
    # )
    # result = service.run(request)
    # assert result.success
'''


def create_module(module_name: str, base_path: Path = None) -> None:
    """Create a new analysis module from templates.
    
    Args:
        module_name: Name of the module (e.g., 'population')
        base_path: Base path for farm package (default: auto-detect)
    """
    if base_path is None:
        # Auto-detect farm package location
        script_dir = Path(__file__).parent
        base_path = script_dir.parent / "farm" / "analysis"
    
    # Validate module name
    if not module_name.isidentifier():
        raise ValueError(f"Invalid module name: {module_name}")
    
    # Create paths
    module_path = base_path / module_name
    test_path = base_path.parent.parent / "tests" / "analysis"
    
    # Check if module already exists
    if module_path.exists():
        raise FileExistsError(f"Module already exists: {module_path}")
    
    # Create module directory
    module_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare template variables
    class_name = ''.join(word.capitalize() for word in module_name.split('_')) + 'Module'
    module_title = ' '.join(word.capitalize() for word in module_name.split('_'))
    
    template_vars = {
        'module_name': module_name,
        'class_name': class_name,
        'module_title': module_title,
        'description': f"Analysis of {module_name} patterns in simulations",
        'subject': module_name,
        'features': f"- {module_title} metrics\n- Pattern analysis\n- Visualizations",
    }
    
    # Create files
    files = {
        '__init__.py': TEMPLATE_INIT,
        'data.py': TEMPLATE_DATA,
        'compute.py': TEMPLATE_COMPUTE,
        'analyze.py': TEMPLATE_ANALYZE,
        'plot.py': TEMPLATE_PLOT,
        'module.py': TEMPLATE_MODULE,
    }
    
    print(f"Creating module: {module_name}")
    print(f"Location: {module_path}")
    print()
    
    for filename, template in files.items():
        file_path = module_path / filename
        content = template.format(**template_vars)
        file_path.write_text(content)
        print(f"  Created: {filename}")
    
    # Create test file
    test_file = test_path / f"test_{module_name}.py"
    test_content = TEMPLATE_TEST.format(**template_vars)
    test_file.write_text(test_content)
    print(f"  Created: test_{module_name}.py")
    
    print()
    print("✓ Module structure created successfully!")
    print()
    print("Next steps:")
    print(f"  1. Edit {module_path}/data.py - Implement data processing")
    print(f"  2. Edit {module_path}/compute.py - Implement computations")
    print(f"  3. Edit {module_path}/analyze.py - Implement analysis")
    print(f"  4. Edit {module_path}/plot.py - Implement visualizations")
    print(f"  5. Update {module_path}/__init__.py - Export functions")
    print(f"  6. Write tests in {test_file}")
    print(f"  7. Register module in farm/analysis/__init__.py")
    print()
    print("Run tests:")
    print(f"  pytest {test_file}")
    print()
    print("Use the module:")
    print(f"  from farm.analysis.{module_name} import {module_name}_module")
    print(f"  # or via service:")
    print(f"  service.run(AnalysisRequest(module_name='{module_name}', ...))")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python bootstrap_analysis_module.py <module_name>")
        print()
        print("Example:")
        print("  python bootstrap_analysis_module.py population")
        print("  python bootstrap_analysis_module.py resources")
        print("  python bootstrap_analysis_module.py actions")
        sys.exit(1)
    
    module_name = sys.argv[1]
    
    try:
        create_module(module_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
