# Dominance Analysis Database

This directory contains SQLAlchemy models and utilities for storing and analyzing dominance data in a structured database format.

## Database Structure

The database is organized into the following tables:

1. **simulations** - Base simulation information
   - Primary key for all other tables
   - Contains iteration number

2. **dominance_metrics** - Core dominance metrics
   - Dominance types (population, survival, comprehensive)
   - Dominance scores for each agent type
   - Area under curve metrics
   - Growth trends and final ratios

3. **agent_populations** - Agent population and survival statistics
   - Final population counts
   - Survival statistics for each agent type
   - Initial counts

4. **reproduction_stats** - Reproduction statistics
   - Reproduction metrics for each agent type
   - Reproduction advantage metrics

5. **dominance_switching** - Dominance switching metrics
   - Switching counts and rates
   - Transition probabilities
   - Phase-specific switch counts

6. **resource_distribution** - Resource distribution metrics
   - Resource distances and availability for each agent type
   - Resource correlation metrics

7. **high_low_switching_comparison** - High vs low switching comparisons
   - Reproduction metrics comparing high and low switching simulations

8. **correlation_analysis** - Correlation analysis metrics
   - Reproduction correlations
   - Timing correlations
   - Efficiency correlations
   - Dominance correlations

## Files

- **sqlalchemy_models.py** - SQLAlchemy models defining the database structure
- **import_csv_to_db.py** - Script to import CSV data into the database
- **query_dominance_db.py** - Utility script with example queries

## Usage

### Setting up the Database

1. Import CSV data into the database:

```bash
python import_csv_to_db.py
```

This will create a `dominance.db` SQLite database in the current directory and import data from the default CSV path.

### Querying the Database

The `query_dominance_db.py` script provides several query functions:

```bash
# Run all queries
python query_dominance_db.py

# Run specific query types
python query_dominance_db.py --query-type dominance
python query_dominance_db.py --query-type population
python query_dominance_db.py --query-type reproduction
python query_dominance_db.py --query-type switching
python query_dominance_db.py --query-type resources
python query_dominance_db.py --query-type high-low
python query_dominance_db.py --query-type correlation

# Run a custom SQL query and export to CSV
python query_dominance_db.py --custom-query "SELECT * FROM simulations JOIN dominance_metrics ON simulations.id = dominance_metrics.simulation_id WHERE system_dominance_score > 0.5" --output-file high_system_dominance.csv
```

### Example Queries

Here are some example SQL queries you can use with the `--custom-query` option:

#### Find simulations with high system dominance

```sql
SELECT s.iteration, dm.system_dominance_score, dm.independent_dominance_score, dm.control_dominance_score
FROM simulations s
JOIN dominance_metrics dm ON s.id = dm.simulation_id
WHERE dm.system_dominance_score > 0.6
ORDER BY dm.system_dominance_score DESC
```

#### Compare reproduction success rates across agent types

```sql
SELECT s.iteration, 
       rs.system_reproduction_success_rate, 
       rs.independent_reproduction_success_rate, 
       rs.control_reproduction_success_rate
FROM simulations s
JOIN reproduction_stats rs ON s.id = rs.simulation_id
ORDER BY rs.system_reproduction_success_rate DESC
```

#### Analyze dominance switching patterns

```sql
SELECT s.iteration, 
       ds.total_switches, 
       ds.switches_per_step,
       ds.system_to_independent,
       ds.independent_to_system
FROM simulations s
JOIN dominance_switching ds ON s.id = ds.simulation_id
ORDER BY ds.total_switches DESC
```

## Extending the Database

To add new metrics or tables:

1. Update the SQLAlchemy models in `sqlalchemy_models.py`
2. Update the import script in `import_csv_to_db.py`
3. Add new query functions to `query_dominance_db.py`

## Benefits of Using a Database

- **Structured Data**: Organizes the 380+ columns into logical tables
- **Efficient Queries**: Fast access to specific metrics without loading the entire dataset
- **Relationships**: Maintains relationships between different aspects of the data
- **Scalability**: Can handle large datasets efficiently
- **SQL Interface**: Enables complex queries and aggregations 