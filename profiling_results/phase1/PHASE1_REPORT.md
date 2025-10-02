# Phase 1 Profiling Report

**Generated:** 2025-10-02T03:43:49.910891

## Summary

- **Total Runs:** 1
- **Successful:** 0
- **Failed:** 1
- **Total Time:** 10.4s

## Profiling Runs

### ✗ baseline_small (cprofile)

- **Steps:** 500
- **Agents:** 50
- **Duration:** 10.45s
- **Log:** `profiling_results/phase1/cprofile_baseline_small.log`

## Next Steps

1. Review cProfile output files to identify top time-consuming functions
2. Open py-spy flame graphs in browser to visualize call stacks
3. Import speedscope files at https://www.speedscope.app/ for interactive analysis
4. Document top 10 bottlenecks in profiling plan
5. Proceed to Phase 2: Component-Level Profiling
