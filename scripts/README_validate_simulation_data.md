# Simulation Data Validation Script

## Overview

The `validate_simulation_data.py` script validates that all tables and columns in a simulation database contain data after a simulation run. This helps ensure data integrity and completeness of simulation results.

## Purpose

This script performs comprehensive validation by:
1. **Checking table existence**: Verifies all expected tables exist in the database
2. **Validating data presence**: Ensures all tables have at least one row of data
3. **Checking column completeness**: Identifies columns where all values are NULL
4. **Reporting issues**: Provides detailed reports on any missing or incomplete data

## Usage

### Basic Usage

```bash
python scripts/validate_simulation_data.py --db-path <path-to-simulation-database>
```

### Example Commands

```bash
# Validate a simulation database
python scripts/validate_simulation_data.py --db-path simulations/simulation.db

# Verbose mode (shows all tables with data)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --verbose

# Strict mode (treats optional empty tables as errors)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --strict

# Exit with non-zero code on validation failure (useful for CI/CD)
python scripts/validate_simulation_data.py --db-path simulations/simulation.db --exit-code
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--db-path PATH` | **Required**. Path to the simulation database file |
| `--strict` | Strict mode: treats optional empty tables as errors |
| `--verbose, -v` | Verbose output: shows all tables with row counts |
| `--exit-code` | Exit with non-zero code if validation fails |
| `--help, -h` | Show help message and exit |

## Validation Checks

### 1. Table Existence
The script checks for the presence of all expected tables defined in the SQLAlchemy models:

**Core Tables:**
- `agents` - Agent records
- `agent_states` - Agent state snapshots per step
- `agent_actions` - Actions taken by agents
- `resource_states` - Resource state tracking
- `simulation_steps` - Per-step simulation metrics
- `interactions` - Generic interaction records
- `learning_experiences` - Learning event records
- `health_incidents` - Health change events
- `reproduction_events` - Reproduction attempts and outcomes
- `social_interactions` - Social behavior records
- `simulations` - Simulation metadata
- `simulation_config` - Simulation configuration data

**Optional Tables:**
- `experiments` - Experiment grouping (may be empty)
- `research` - Research project metadata
- `experiment_stats` - Experiment-level statistics
- `iteration_stats` - Iteration-level statistics

### 2. Data Presence
For each table, the script verifies that at least one row of data exists. Empty tables indicate:
- Features that weren't used during the simulation
- Potential configuration issues
- Missing data logging functionality

### 3. Column Completeness
The script identifies columns where all values are NULL, which may indicate:
- Optional fields that weren't populated
- Potential bugs in data logging
- Features that need to be enabled in configuration

## Output Format

### Validation Report Structure

```
================================================================================
SIMULATION DATA VALIDATION REPORT
================================================================================

❌ MISSING TABLES:
  - table_name_1
  - table_name_2

❌ EMPTY TABLES (NO DATA):
  - table_name_3
  - table_name_4

⚠️  COLUMNS WITH ALL NULL VALUES:
  Table: table_name_5
    - column_1
    - column_2

⚠️  WARNINGS:
  - Optional table 'experiments' is empty (acceptable)

✅ TABLES WITH DATA: (verbose mode only)
  - agent_actions: 46,205 rows
  - agent_states: 46,425 rows
  - agents: 217 rows

--------------------------------------------------------------------------------
SUMMARY:
  Total tables expected: 16
  Tables with data: 10
  Missing tables: 2
  Empty tables: 2
  Tables with null columns: 3
  Warnings: 1

✅ VALIDATION PASSED
or
❌ VALIDATION FAILED
================================================================================
```

### Status Indicators

- ✅ **Success** - Validation passed, all required data is present
- ❌ **Error** - Critical issues found (missing/empty tables)
- ⚠️ **Warning** - Non-critical issues (optional tables, null columns)

## Exit Codes

When using `--exit-code` flag:
- `0` - Validation passed successfully
- `1` - Validation failed (missing tables, empty required tables)

In strict mode (`--strict --exit-code`):
- `0` - Validation passed with no warnings
- `1` - Validation failed or warnings present

## Integration Examples

### CI/CD Pipeline

```bash
#!/bin/bash
# Run simulation
python run_simulation.py --steps 1000

# Validate results
python scripts/validate_simulation_data.py \
  --db-path simulations/simulation.db \
  --exit-code

if [ $? -eq 0 ]; then
  echo "Simulation data validation passed"
else
  echo "Simulation data validation failed"
  exit 1
fi
```

### Post-Simulation Check

```python
import subprocess
import sys

# Run validation after simulation
result = subprocess.run(
    [
        "python", 
        "scripts/validate_simulation_data.py",
        "--db-path", "simulations/simulation.db",
        "--verbose",
        "--exit-code"
    ],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ Simulation data is valid")
else:
    print("❌ Simulation data validation failed")
    print(result.stdout)
    sys.exit(1)
```

### Batch Validation

```bash
# Validate multiple simulation databases
for db in simulations/*.db; do
  echo "Validating $db..."
  python scripts/validate_simulation_data.py --db-path "$db"
done
```

## Common Issues and Solutions

### Missing Tables
**Issue**: Expected tables are not present in the database
**Solutions**:
- Ensure the simulation completed successfully
- Check database migration status
- Verify SQLAlchemy models are properly defined

### Empty Tables
**Issue**: Tables exist but contain no data
**Solutions**:
- Check if the feature is enabled in simulation config
- Verify data logging is functioning correctly
- Review simulation logs for errors
- Consider if the feature is expected to have data (e.g., `learning_experiences` may be empty if learning is disabled)

### Columns with All NULLs
**Issue**: Columns exist but all values are NULL
**Solutions**:
- Review if these are optional/nullable fields
- Check if the feature using this column is enabled
- Verify data is being written correctly in the code
- Some NULL columns are expected (e.g., `action_target_id` when actions don't have targets)

## Logging

The script provides structured logging output with:
- Validation start/completion events
- Per-table validation results
- Column-level null checks
- Error and warning messages

Log output uses JSON format for easy parsing and integration with log aggregation systems.

## Dependencies

Required Python packages:
- `sqlalchemy>=1.4.0` - Database ORM
- `structlog>=24.1.0` - Structured logging
- `pydantic>=2.0.0` - Data validation
- `deepdiff>=5.8.0` - Deep comparison
- `pyyaml>=6.0` - YAML parsing

These are included in the project's `requirements.txt`.

## Development Notes

### Extending the Script

To add validation for custom tables:
1. Define the table model in `farm/database/models.py`
2. The script will automatically detect it through SQLAlchemy's `Base.metadata`

To add custom validation rules:
1. Modify the `validate_database()` function
2. Add new validation checks in the table iteration loop
3. Update the `ValidationResult` class to track new issues

### Testing

```bash
# Test with sample database
python scripts/validate_simulation_data.py --db-path docs/sample/simulation.db --verbose

# Test strict mode
python scripts/validate_simulation_data.py --db-path docs/sample/simulation.db --strict

# Test exit codes
python scripts/validate_simulation_data.py --db-path docs/sample/simulation.db --exit-code
echo "Exit code: $?"
```

## Troubleshooting

### Script Won't Run
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH=/path/to/workspace:$PYTHONPATH
```

### Import Errors
The script automatically adds the workspace to Python path. If you encounter import errors:
```bash
# Run from workspace root
cd /path/to/workspace
python scripts/validate_simulation_data.py --db-path <path>
```

### Database Locked
If the database is locked:
- Ensure the simulation has completed
- Close any other connections to the database
- Wait for in-memory database persistence to complete

## Future Enhancements

Potential improvements:
- [ ] Add data quality checks (e.g., value ranges, consistency)
- [ ] Generate validation reports in JSON/HTML format
- [ ] Compare data completeness across multiple simulations
- [ ] Add performance metrics (query times)
- [ ] Create a validation dashboard
- [ ] Add automated fixing suggestions

## Support

For issues or questions:
1. Check simulation logs for errors
2. Review this documentation
3. Examine the validation report output
4. Contact the development team

## License

This script is part of the AgentFarm simulation framework and follows the same license terms.
