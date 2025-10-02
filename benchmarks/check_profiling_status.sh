#!/bin/bash
# Check the status of running profiling jobs

echo "=========================================="
echo "Profiling Status Check"
echo "=========================================="
echo ""

# Check for running profiling processes
echo "## Running Processes"
PROCS=$(ps aux | grep -E 'python.*run_simulation|python.*profiling' | grep -v grep)
if [ -z "$PROCS" ]; then
    echo "✓ No active profiling processes"
else
    echo "$PROCS"
fi
echo ""

# Check profiling results directory
echo "## Generated Files"
if [ -d "profiling_results/phase1" ]; then
    echo ""
    echo "Phase 1 Results:"
    ls -lh profiling_results/phase1/ 2>/dev/null | tail -n +2 | awk '{printf "  %s %8s  %s\n", $6" "$7" "$8, $5, $9}'
    
    FILE_COUNT=$(ls -1 profiling_results/phase1/*.prof 2>/dev/null | wc -l)
    echo ""
    echo "  Profile files (.prof): $FILE_COUNT"
    
    LOG_COUNT=$(ls -1 profiling_results/phase1/*.log 2>/dev/null | wc -l)
    echo "  Log files (.log): $LOG_COUNT"
    
    SVG_COUNT=$(ls -1 profiling_results/phase1/*.svg 2>/dev/null | wc -l)
    echo "  Flame graphs (.svg): $SVG_COUNT"
else
    echo "✗ No results directory yet"
fi
echo ""

# Check for summary report
echo "## Reports"
if [ -f "profiling_results/phase1/PHASE1_REPORT.md" ]; then
    echo "✓ Phase 1 report generated"
    echo "  View: cat profiling_results/phase1/PHASE1_REPORT.md"
else
    echo "⧗ Phase 1 report pending"
fi

if [ -f "profiling_results/phase1/phase1_summary.json" ]; then
    echo "✓ Phase 1 summary JSON generated"
    TOTAL_RUNS=$(grep -o '"total_runs": [0-9]*' profiling_results/phase1/phase1_summary.json | cut -d: -f2 | tr -d ' ')
    SUCCESS=$(grep -o '"successful_runs": [0-9]*' profiling_results/phase1/phase1_summary.json | cut -d: -f2 | tr -d ' ')
    echo "  Runs: $SUCCESS/$TOTAL_RUNS successful"
else
    echo "⧗ Phase 1 summary pending"
fi
echo ""

# Show recent log activity
echo "## Recent Activity"
if [ -f "profiling_results/phase1/cprofile_baseline_log.txt" ]; then
    LAST_LINE=$(tail -1 profiling_results/phase1/cprofile_baseline_log.txt 2>/dev/null)
    if [ ! -z "$LAST_LINE" ]; then
        echo "Last log entry:"
        echo "  $LAST_LINE"
    fi
fi
echo ""

# Check simulations directory
echo "## Simulation Output"
if [ -f "simulations/profile_stats.prof" ]; then
    SIZE=$(ls -lh simulations/profile_stats.prof | awk '{print $5}')
    echo "✓ Profile data available (${SIZE})"
    echo "  Analyze: python3 benchmarks/analyze_cprofile.py simulations/profile_stats.prof"
    echo "  Visualize: snakeviz simulations/profile_stats.prof"
else
    echo "⧗ Waiting for profile data"
fi
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "To view results when ready:"
echo "  cat profiling_results/phase1/PHASE1_REPORT.md"
echo ""
echo "To analyze profiles:"
echo "  python3 benchmarks/analyze_cprofile.py profiling_results/phase1/cprofile_baseline_small.prof"
echo ""
echo "To visualize interactively:"
echo "  snakeviz profiling_results/phase1/cprofile_baseline_small.prof"
echo ""
echo "To re-run profiling:"
echo "  python3 benchmarks/run_phase1_profiling.py --quick"
echo ""
