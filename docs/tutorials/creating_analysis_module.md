# Creating a New Analysis Module for AgentFarm

This tutorial guides you through the process of creating a new analysis module for the AgentFarm project. The analysis system uses a modular approach to organize different types of analysis, making it easy to add new analysis capabilities.

## Prerequisites

Before creating a new analysis module, you should be familiar with:
- Python programming
- Basic understanding of the AgentFarm simulation system
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization

## Step 1: Create a New Module Directory

First, create a new directory for your module inside the `farm/analysis` folder:

```bash
mkdir farm/analysis/your_module_name
```

## Step 2: Use the Template Module

AgentFarm provides a template module that you can use as a starting point. The template is located in `farm/analysis/template/module.py`.

You can either:
1. Copy the template files to your new module directory:
   ```bash
   cp farm/analysis/template/* farm/analysis/your_module_name/
   ```

2. Or create the files from scratch following the structure below.

## Step 3: Create the Module Files

Inside your new module directory, create these essential files:

1. `__init__.py` - Makes the directory a Python package
2. `processor.py` - Contains data processing logic
3. `visualizer.py` - Contains visualization functions
4. `module.py` - Defines the module class

## Step 4: Implement the Module Class

Create a `module.py` file that extends the base module. You can use the template module as a reference:

```python
from farm.analysis.base_module import AnalysisModule
from farm.analysis.your_module_name.processor import process_data
from farm.analysis.your_module_name.visualizer import (
    basic_visualization,
    advanced_visualization,
    # Add other visualization functions
)

class YourModuleAnalysis(AnalysisModule):
    """
    Analysis module for your specific analysis type.
    
    This module analyzes [describe what your module analyzes].
    """
    
    def __init__(self):
        super().__init__(
            name="your_module_name",
            description="Description of your analysis module",
            processor=process_data,
            functions={
                "basic": [basic_visualization],
                "advanced": [advanced_visualization],
                # Add other function groups as needed
                "all": [basic_visualization, advanced_visualization]
            }
        )

# Create a singleton instance
your_module_name_module = YourModuleAnalysis()
```

## Step 5: Implement the Data Processor

Create a `processor.py` file with your data processing logic:

```python
import logging
import os
import pandas as pd
import glob
import sqlite3
# Import other necessary libraries

def process_data(experiment_path, output_path=None, **kwargs):
    """
    Process simulation data for your specific analysis.
    
    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing simulation data
    output_path : str, optional
        Path to save processed data
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    pandas.DataFrame
        Processed data ready for analysis
    """
    logging.info(f"Processing data from {experiment_path}")
    
    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
    logging.info(f"Found {len(sim_folders)} simulation folders")
    
    # Initialize data collection
    results = []
    
    # Process each simulation
    for folder in sim_folders:
        try:
            # Extract simulation ID
            sim_id = os.path.basename(folder)
            
            # Process data from this simulation
            # Example: Read from database, CSV files, etc.
            db_path = os.path.join(folder, "simulation.db")
            if os.path.exists(db_path):
                # Connect to database
                conn = sqlite3.connect(db_path)
                
                # Extract relevant data
                # Example query:
                query = "SELECT * FROM your_table"
                sim_data = pd.read_sql_query(query, conn)
                
                # Process the data
                # ... your processing logic here ...
                
                # Add to results
                sim_result = {
                    'simulation_id': sim_id,
                    # Add other metrics
                }
                results.append(sim_result)
                
                conn.close()
            
        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")
    
    # Create DataFrame from results
    if not results:
        logging.warning("No data processed")
        return None
        
    df = pd.DataFrame(results)
    
    # Save processed data if output_path is provided
    if output_path:
        csv_path = os.path.join(output_path, "processed_data.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved processed data to {csv_path}")
    
    return df
```

## Step 6: Implement Visualization Functions

Create a `visualizer.py` file with your visualization functions:

```python
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import other visualization libraries as needed

def basic_visualization(df, output_path, **kwargs):
    """
    Create basic visualizations for your analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Processed data from the processor
    output_path : str
        Directory to save visualization outputs
    **kwargs : dict
        Additional keyword arguments
    """
    if df is None or df.empty:
        logging.warning("No data available for visualization")
        return
    
    logging.info("Creating basic visualizations")
    
    # Example: Create a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['your_metric'], kde=True)
    plt.title('Distribution of Your Metric')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Save the figure
    fig_path = os.path.join(output_path, "metric_distribution.png")
    plt.savefig(fig_path)
    plt.close()
    
    logging.info(f"Saved basic visualization to {fig_path}")

def advanced_visualization(df, output_path, **kwargs):
    """
    Create more advanced visualizations for your analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Processed data from the processor
    output_path : str
        Directory to save visualization outputs
    **kwargs : dict
        Additional keyword arguments
    """
    if df is None or df.empty:
        logging.warning("No data available for visualization")
        return
    
    logging.info("Creating advanced visualizations")
    
    # Example: Create a correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Metrics')
    
    # Save the figure
    fig_path = os.path.join(output_path, "correlation_heatmap.png")
    plt.savefig(fig_path)
    plt.close()
    
    logging.info(f"Saved advanced visualization to {fig_path}")
```

## Step 7: Register Your Module

Update the registry to include your new module. Edit the `farm/analysis/registry.py` file:

```python
def register_modules():
    """
    Register all available analysis modules.
    This function should be called once at application startup.
    """
    # Import modules here to avoid circular imports
    from farm.analysis.dominance.module import dominance_module
    from farm.analysis.your_module_name.module import your_module_name_module  # Add this line
    
    # Register modules
    registry.register_module(dominance_module)
    registry.register_module(your_module_name_module)  # Add this line
    
    # Add more modules as they become available
```

## Step 8: Create a Script to Run Your Analysis

Create a script to run your analysis, similar to the dominance_analysis.py:

```python
import logging

# Import analysis configuration
from analysis_config import run_analysis

def main():
    """
    Run your specific analysis using the module system.
    """
    # Run the analysis using the generic function with the module system
    output_path, df = run_analysis(
        analysis_type="your_module_name",
        function_group="all",  # Use all analysis functions
        # Alternatively, you can specify a specific group:
        # function_group="basic",  # Only basic analysis
        # function_group="advanced",  # Only advanced analysis
    )

    if df is not None and not df.empty:
        logging.info(
            f"Analysis complete. Processed {len(df)} simulations. Output saved to {output_path}"
        )

if __name__ == "__main__":
    main()
```

## Step 9: Test Your Module

Run your analysis script to test your new module:

```bash
python scripts/your_module_analysis.py
```

## Additional Tips

1. **Documentation**: Add a README.md file to your module directory explaining what the module does and how to use it.

2. **Look at Existing Modules**: Examine existing modules like the dominance module for examples of how to structure your code.

3. **Customization**: You can add more function groups or specialized visualizations based on your specific analysis needs.

4. **Error Handling**: Make sure to include proper error handling in your processor and visualization functions.

5. **Configuration**: You can add module-specific configuration options in your module class.

## Common Patterns

### Accessing Simulation Databases

Most analysis modules need to access the simulation databases. Here's a common pattern:

```python
import sqlite3
import pandas as pd

def query_simulation_db(db_path, query):
    """Query a simulation database and return results as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# In your processor function:
db_path = os.path.join(sim_folder, "simulation.db")
if os.path.exists(db_path):
    agents_df = query_simulation_db(db_path, "SELECT * FROM agents")
    actions_df = query_simulation_db(db_path, "SELECT * FROM actions")
    # Process the data...
```

### Grouping Visualizations

You can organize your visualizations into logical groups:

```python
# In your module.py:
functions={
    "basic": [plot_simple_metrics],
    "time_series": [plot_metrics_over_time, plot_cumulative_metrics],
    "comparative": [plot_comparison_charts, plot_correlation_matrix],
    "all": [plot_simple_metrics, plot_metrics_over_time, plot_cumulative_metrics, 
            plot_comparison_charts, plot_correlation_matrix]
}
```

This allows users to run specific groups of visualizations based on their needs.

## Conclusion

By following this tutorial, you've created a new analysis module that integrates with the existing analysis framework in AgentFarm. Your module can now be used to analyze simulation data and generate visualizations. 