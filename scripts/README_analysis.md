# Generic Analysis Framework

This framework provides a standardized way to create and run different types of analyses on simulation data. It abstracts common patterns like setting up directories, loading data, and running analysis functions.

## How to Use

### 1. Create a New Analysis Module

To create a new analysis module:

1. Create a new directory structure under `farm/analysis/your_analysis_type/`
2. Implement the following components:
   - Data processor function
   - Analysis/visualization functions
   - (Optional) Database loader function if using a database

### 2. Create an Analysis Script

Create a script in the `scripts/` directory (e.g., `your_analysis_type_analysis.py`) using the template below:

```python
import logging
import os

# Import analysis configuration
from analysis_config import run_analysis, save_analysis_data

# Import your analysis module's functions
from farm.analysis.your_analysis_type.analyze import process_your_data
from farm.analysis.your_analysis_type.query_your_db import load_data_from_db
from farm.analysis.your_analysis_type.plot import (
    plot_function1,
    plot_function2,
    # Add other plotting functions here
)


def main():
    # Define all analysis functions to run
    analysis_functions = [
        plot_function1,
        plot_function2,
        # Add other analysis functions here
    ]
    
    # Define any special keyword arguments for specific functions
    analysis_kwargs = {
        "plot_function1": {"param1": value1},
        "plot_function2": {"param2": value2},
    }
    
    # Run the analysis using the generic function
    output_path, df = run_analysis(
        analysis_type="your_analysis_type",
        data_processor=process_your_data,
        analysis_functions=analysis_functions,
        db_filename="your_analysis.db",  # Set to None if not using a database
        load_data_function=load_data_from_db,  # Set to None if not using a database
        processor_kwargs={
            # Add any keyword arguments for your data processor
            "param3": value3,
        },
        analysis_kwargs=analysis_kwargs
    )
    
    # Add any additional post-processing specific to your analysis
    if df is not None and not df.empty:
        # Example: Save the processed data to CSV
        save_analysis_data(df, output_path, "your_analysis_results")


if __name__ == "__main__":
    main()
```

### 3. Implement Required Functions

#### Data Processor Function

The data processor function should:
- Accept an experiment path as the first argument
- Process raw simulation data
- Return a DataFrame or save to a database and return None

Example:

```python
def process_your_data(experiment_path, save_to_db=False, db_path=None, **kwargs):
    # Process data from experiment_path
    df = pd.DataFrame(...)
    
    if save_to_db and db_path:
        # Save to database
        engine = sqlalchemy.create_engine(db_path)
        df.to_sql("your_table", engine, if_exists="replace", index=False)
        return None
    
    return df
```

#### Analysis/Visualization Functions

Each analysis function should:
- Accept a DataFrame as the first argument
- Accept an output path as the second argument
- Accept additional keyword arguments as needed
- Save results (plots, tables, etc.) to the output path

Example:

```python
def plot_function1(df, output_path, param1=default_value):
    # Create visualization
    plt.figure()
    # ... plotting code ...
    
    # Save to output path
    output_file = os.path.join(output_path, "plot_function1.png")
    plt.savefig(output_file)
    plt.close()
```

#### Database Loader Function (Optional)

If your data processor saves to a database, implement a loader function:

```python
def load_data_from_db(db_uri):
    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM your_table", engine)
    return df
```

## Parameters for `run_analysis`

- `analysis_type`: String identifier for the analysis type (used for directory naming)
- `data_processor`: Function to process raw data
- `analysis_functions`: List of analysis/visualization functions to run
- `db_filename`: Name of the database file (optional)
- `load_data_function`: Function to load data from database (optional)
- `processor_kwargs`: Additional keyword arguments for the data processor
- `analysis_kwargs`: Dictionary mapping function names to their keyword arguments

## Example

See `scripts/dominance_analysis.py` for a complete example of using the framework. 