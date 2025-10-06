# Simulation Data Validation Guide

## Quick Start

After running a simulation, validate the data to ensure completeness:

```bash
python scripts/validate_simulation_data.py --db-path simulations/simulation.db
```

## Overview

The simulation data validation system ensures that all expected tables and columns contain data after a simulation run. This is critical for:

- **Data Integrity**: Verify all simulation features are logging correctly
- **Quality Assurance**: Catch missing or incomplete data early
- **CI/CD Integration**: Automated testing of simulation outputs
- **Research Validation**: Ensure experiment data is complete for analysis

## Files

| File | Purpose |
|------|---------|
| `validate_simulation_data.py` | Main validation script |
| `example_run_and_validate.sh` | Example: Run simulation + validate |
| `example_validate_batch.py` | Example: Validate multiple databases |
| `README_validate_simulation_data.md` | Detailed documentation |
| `VALIDATION_GUIDE.md` | This quick reference guide |

## Basic Usage

### Single Database Validation

```bash
# Basic validation
python scripts/validate_simulation_data.py --db-path simulations/simulation.db

# Verbose output (shows all tables)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --verbose

# Strict mode (optional tables must have data)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --strict

# Exit with error code on failure (useful for scripts)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --exit-code
```

### Batch Validation

Validate multiple simulation databases at once:

```bash
# Validate all databases in a directory
python scripts/example_validate_batch.py simulations/

# Validate with custom pattern
python scripts/example_validate_batch.py experiments/ --pattern "exp_*.db"

# Save report to file
python scripts/example_validate_batch.py simulations/ --output validation_report.txt

# Strict mode with exit code
python scripts/example_validate_batch.py simulations/ --strict --exit-code
```

### Integrated Workflow

Run simulation and validate in one command:

```bash
# Run and validate
bash scripts/example_run_and_validate.sh --steps 1000

# With strict validation
bash scripts/example_run_and_validate.sh --steps 500 --strict
```

## Expected Tables

### Core Simulation Tables

These tables should always have data after a successful simulation:

| Table | Description | Expected Data |
|-------|-------------|---------------|
| `simulations` | Simulation metadata | 1 row |
| `simulation_steps` | Per-step metrics | num_steps rows |
| `agents` | Agent records | ≥ initial_population |
| `agent_states` | Agent states per step | Many rows |
| `agent_actions` | Actions taken | Many rows |
| `resource_states` | Resource tracking | Many rows |
| `interactions` | Generic interactions | Many rows (if enabled) |
| `reproduction_events` | Reproduction attempts | Some rows (if occurred) |
| `health_incidents` | Health changes | Some rows (if occurred) |

### Optional Tables

These tables may be empty depending on simulation configuration:

| Table | Description | When Empty |
|-------|-------------|------------|
| `learning_experiences` | Learning events | No learning enabled |
| `social_interactions` | Social behaviors | Feature not used |
| `experiments` | Experiment grouping | Standalone simulation |
| `simulation_config` | Configuration data | Not saved |
| `research` | Research metadata | Research mode disabled |
| `experiment_stats` | Experiment statistics | Research mode disabled |
| `iteration_stats` | Iteration statistics | Research mode disabled |

## Understanding Results

### ✅ Success

```
✅ VALIDATION PASSED - All required tables and columns have data
```

All expected tables exist and have data. Simulation ran successfully!

### ❌ Failures

#### Missing Tables

```
❌ MISSING TABLES:
  - research
  - experiment_stats
```

**Cause**: Tables not created in database
**Solution**: 
- Check if feature is enabled
- Verify database schema is up to date
- May be expected for optional tables

#### Empty Tables

```
❌ EMPTY TABLES (NO DATA):
  - learning_experiences
  - social_interactions
```

**Cause**: No data was written to these tables
**Solution**:
- Verify feature is enabled in config
- Check if conditions for data logging were met
- May be expected if feature wasn't used

### ⚠️ Warnings

#### Columns with NULL Values

```
⚠️  COLUMNS WITH ALL NULL VALUES:
  Table: agent_actions
    - action_target_id
    - state_before_id
```

**Cause**: All values in these columns are NULL
**Analysis**:
- May be expected for optional fields
- `action_target_id` is NULL for actions without targets
- Foreign keys may be NULL if feature is optional

#### Optional Empty Tables

```
⚠️  WARNINGS:
  - Optional table 'experiments' is empty (acceptable)
```

**Cause**: Optional feature not used
**Status**: Acceptable, no action needed

## Common Validation Issues

### Issue: All Tables Empty

**Symptoms**:
```
Empty tables: 13
```

**Causes**:
- Simulation didn't run or failed
- Database not persisted (in-memory mode with `--no-persist`)
- Wrong database path

**Solutions**:
1. Check simulation completed successfully
2. Verify database path is correct
3. Ensure `persist_db_on_completion` is enabled for in-memory DB

### Issue: Missing Critical Tables

**Symptoms**:
```
❌ MISSING TABLES:
  - agents
  - agent_states
```

**Causes**:
- Database corruption
- Schema mismatch
- Incomplete simulation

**Solutions**:
1. Re-run simulation
2. Check database file integrity
3. Verify SQLAlchemy models are correct

### Issue: Partial Data

**Symptoms**:
```
Tables with data: 8
Expected: 16
```

**Causes**:
- Optional features disabled
- Simulation stopped early
- Feature-specific data not logged

**Solutions**:
1. Enable required features in config
2. Run simulation to completion
3. Use `--strict` mode to identify issues

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Validate Simulation Data

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run simulation
        run: python run_simulation.py --steps 100
      
      - name: Validate data
        run: |
          python scripts/validate_simulation_data.py \
            --db-path simulations/simulation.db \
            --verbose \
            --exit-code
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Run Simulation') {
            steps {
                sh 'python run_simulation.py --steps 1000'
            }
        }
        stage('Validate Data') {
            steps {
                sh '''
                    python scripts/validate_simulation_data.py \
                        --db-path simulations/simulation.db \
                        --exit-code
                '''
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'simulations/*.db'
        }
    }
}
```

## Advanced Usage

### Custom Validation in Python

```python
import sys
sys.path.insert(0, '/path/to/workspace')

from scripts.validate_simulation_data import validate_database

# Validate database
result = validate_database('simulations/simulation.db', strict=False)

# Check results
if result.is_valid():
    print(f"✅ Validation passed")
    print(f"Tables with data: {len(result.tables_with_data)}")
else:
    print(f"❌ Validation failed")
    print(f"Missing: {result.missing_tables}")
    print(f"Empty: {result.empty_tables}")
    print(f"Null columns: {result.columns_with_all_nulls}")

# Access detailed results
for table, row_count in result.tables_with_data:
    print(f"{table}: {row_count:,} rows")
```

### Validation in Test Suite

```python
import pytest
from scripts.validate_simulation_data import validate_database

def test_simulation_data_complete():
    """Test that simulation produced complete data."""
    result = validate_database('test_simulation.db')
    
    assert len(result.missing_tables) == 0, \
        f"Missing tables: {result.missing_tables}"
    
    assert len(result.empty_tables) == 0, \
        f"Empty tables: {result.empty_tables}"
    
    assert result.is_valid(), "Validation failed"

def test_minimum_row_counts():
    """Test that tables have minimum expected rows."""
    result = validate_database('test_simulation.db')
    
    tables_dict = dict(result.tables_with_data)
    
    assert tables_dict.get('agents', 0) >= 10, \
        "Expected at least 10 agents"
    
    assert tables_dict.get('simulation_steps', 0) >= 100, \
        "Expected at least 100 simulation steps"
```

## Performance Considerations

### Large Databases

For databases with millions of rows:

1. **Validation time**: Proportional to database size
2. **Memory usage**: Minimal (streaming queries)
3. **Lock contention**: Uses read-only queries

### Optimization Tips

- Run validation after simulation completes
- Use `--exit-code` in automated pipelines (faster, less output)
- Skip verbose mode for CI/CD (faster)
- Validate in parallel for batch operations

## Troubleshooting

### Problem: Script Not Found

```bash
# Ensure you're in the workspace root
cd /path/to/workspace
python scripts/validate_simulation_data.py --db-path <path>
```

### Problem: Import Errors

```bash
# Add workspace to Python path
export PYTHONPATH=/path/to/workspace:$PYTHONPATH

# Or install package in development mode
pip install -e .
```

### Problem: Database Locked

```
sqlite3.OperationalError: database is locked
```

**Solutions**:
- Wait for simulation to complete
- Close other database connections
- Ensure in-memory DB is persisted

## Best Practices

1. **Always validate after simulation**: Catch data issues early
2. **Use verbose mode for debugging**: See all tables and row counts
3. **Enable exit codes in automation**: Fail fast on issues
4. **Archive validation reports**: Track data quality over time
5. **Review null columns**: Understand expected vs unexpected NULLs
6. **Test with strict mode**: Ensure optional features work when enabled

## Getting Help

For detailed information:
- See `README_validate_simulation_data.md` for complete documentation
- Check simulation logs for errors
- Review database schema in `farm/database/models.py`
- Examine validation source code in `validate_simulation_data.py`

## Summary

The validation system provides comprehensive data quality checks for simulation outputs:

- ✅ Automated validation of all tables and columns
- ✅ Detailed reporting of issues
- ✅ CI/CD integration support
- ✅ Batch validation capabilities
- ✅ Flexible modes (normal, strict, verbose)
- ✅ Exit code support for automation

Ensure data integrity and catch issues early by validating after every simulation run!
