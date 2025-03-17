# Relative Advantage Analysis Database

This directory contains SQLAlchemy models and utilities for storing and analyzing relative advantage data in a structured database format.

## Database Structure

The database is organized into the following tables:

1. **simulations** - Base simulation information
   - Primary key for all other tables
   - Contains iteration number

2. **resource_acquisition** - Resource acquisition advantages
   - Raw resource metrics for each agent type across simulation phases
   - Advantage metrics between agent types
   - Advantage trajectories over time

3. **reproduction_advantage** - Reproduction advantages
   - Success rates, offspring counts, efficiency metrics
   - Timing advantages for first reproduction
   - Comparative advantages between agent types

4. **survival_advantage** - Survival advantages
   - Survival rates and average lifespans
   - Death rates and comparative survival metrics
   - Advantage metrics between agent types

5. **population_growth** - Population growth advantages
   - Growth rates across simulation phases
   - Final population counts
   - Population advantage metrics

6. **combat_advantage** - Combat advantages
   - Win rates and damage statistics
   - Combat advantage metrics between agent types

7. **initial_positioning** - Initial positioning advantages
   - Resource distance metrics
   - Resources in range metrics
   - Positioning advantage metrics

8. **composite_advantage** - Composite advantages
   - Overall advantage scores
   - Component contributions to advantage
   - Weighted advantage metrics

9. **advantage_dominance_correlation** - Advantage-dominance correlations
   - Correlations between advantages and dominance outcomes
   - Dominant agent type identification
   - Predictive metrics for dominance

## Files

- **sqlalchemy_models.py** - SQLAlchemy models defining the database structure
- **import_csv_to_db.py** - Script to import CSV data into the database
- **query_relative_advantage_db.py** - Utility script with example queries

## Usage

### Setting up the Database

1. Import CSV data into the database:

```bash
python import_csv_to_db.py --csv-path data/relative_advantage_results.csv
```

This will create a `relative_advantage.db` SQLite database in the current directory and import data from the specified CSV path.

### Querying the Database

The `query_relative_advantage_db.py` script provides several query functions:

```bash
# Run all queries
python query_relative_advantage_db.py

# Run specific query types
python query_relative_advantage_db.py --query-type resource
python query_relative_advantage_db.py --query-type reproduction
python query_relative_advantage_db.py --query-type survival
python query_relative_advantage_db.py --query-type population
python query_relative_advantage_db.py --query-type composite
python query_relative_advantage_db.py --query-type correlation

# Run a custom SQL query and export to CSV
python query_relative_advantage_db.py --custom-query "SELECT s.iteration, ra.system_vs_independent_late_phase_advantage FROM simulations s JOIN resource_acquisition ra ON s.id = ra.simulation_id WHERE ra.system_vs_independent_late_phase_advantage > 0.5" --output-file high_resource_advantage.csv
```

### Example Queries

Here are some example SQL queries you can use with the `--custom-query` option:

#### Find simulations with high system vs independent advantage

```sql
SELECT s.iteration, ca.system_vs_independent_score, ca.system_vs_independent_resource_component, ca.system_vs_independent_reproduction_component
FROM simulations s
JOIN composite_advantage ca ON s.id = ca.simulation_id
WHERE ca.system_vs_independent_score > 0.6
ORDER BY ca.system_vs_independent_score DESC
```

#### Compare reproduction success rates across agent types

```sql
SELECT s.iteration, 
       ra.system_success_rate, 
       ra.independent_success_rate, 
       ra.control_success_rate
FROM simulations s
JOIN reproduction_advantage ra ON s.id = ra.simulation_id
ORDER BY ra.system_success_rate DESC
```

#### Analyze advantage-dominance correlations

```sql
SELECT s.iteration, 
       adc.dominant_type, 
       adc.resource_late_phase_correlation,
       adc.reproduction_success_rate_correlation,
       adc.composite_advantage_correlation
FROM simulations s
JOIN advantage_dominance_correlation adc ON s.id = adc.simulation_id
ORDER BY adc.composite_advantage_correlation DESC
```

## Extending the Database

To add new metrics or tables:

1. Update the SQLAlchemy models in `sqlalchemy_models.py`
2. Update the import script in `import_csv_to_db.py`
3. Add new query functions to `query_relative_advantage_db.py`

## Benefits of Using a Database

- **Structured Data**: Organizes the relative advantage metrics into logical tables
- **Efficient Queries**: Fast access to specific metrics without loading the entire dataset
- **Relationships**: Maintains relationships between different aspects of the data
- **Scalability**: Can handle large datasets efficiently
- **SQL Interface**: Enables complex queries and aggregations 