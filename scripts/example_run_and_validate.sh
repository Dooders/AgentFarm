#!/bin/bash
# Example script: Run simulation and validate data
#
# This script demonstrates how to run a simulation and automatically
# validate that all tables and columns have data.
#
# Usage:
#   bash scripts/example_run_and_validate.sh [--steps NUM_STEPS] [--strict]

set -e  # Exit on error

# Default values
STEPS=100
STRICT_MODE=""
OUTPUT_DIR="simulations"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --strict)
      STRICT_MODE="--strict"
      shift
      ;;
    --help)
      echo "Usage: $0 [--steps NUM_STEPS] [--strict]"
      echo ""
      echo "Options:"
      echo "  --steps NUM_STEPS   Number of simulation steps (default: 100)"
      echo "  --strict            Enable strict validation mode"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "=================================================="
echo "SIMULATION AND VALIDATION WORKFLOW"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Strict mode: ${STRICT_MODE:-disabled}"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Run the simulation
echo "Step 1: Running simulation..."
echo "--------------------------------------------------"
python3 run_simulation.py --steps "$STEPS" 2>&1 | tee simulation_run.log

if [ $? -ne 0 ]; then
  echo "❌ Simulation failed!"
  exit 1
fi

echo ""
echo "✅ Simulation completed successfully"
echo ""

# Step 2: Validate the data
echo "Step 2: Validating simulation data..."
echo "--------------------------------------------------"
DB_PATH="$OUTPUT_DIR/simulation.db"

if [ ! -f "$DB_PATH" ]; then
  echo "❌ Database file not found: $DB_PATH"
  exit 1
fi

# Run validation with exit code check
python3 scripts/validate_simulation_data.py \
  --db-path "$DB_PATH" \
  --verbose \
  --exit-code \
  $STRICT_MODE

VALIDATION_EXIT_CODE=$?

echo ""
if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
  echo "=================================================="
  echo "✅ WORKFLOW COMPLETED SUCCESSFULLY"
  echo "=================================================="
  echo ""
  echo "Simulation database: $DB_PATH"
  echo "All tables and columns validated successfully."
  exit 0
else
  echo "=================================================="
  echo "❌ WORKFLOW FAILED"
  echo "=================================================="
  echo ""
  echo "Simulation database: $DB_PATH"
  echo "Data validation found issues. Please review the report above."
  exit 1
fi
