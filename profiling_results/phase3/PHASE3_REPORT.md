# Phase 3 Line-Level Profiling Report

**Generated:** 2025-10-02T04:06:28.496196

## Summary

- **Total Profiles:** 4
- **Successful:** 3
- **Failed:** 1
- **Total Time:** 46.8s

## Function Profiles

### ✓ observe (line_profile)

- **Duration:** 26.85s
- **Output:** `profiling_results/phase3/line_profile_observe.txt`

**Profile Data (top 30 lines):**
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    21                                           @profile
    22                                           def profiled_get_observation(self, agent_id):
    23       100     235862.3   2358.6    100.0      return original_get_observation(self, agent_id)

2025-10-02 04:06:53 [debug    ] Database engine disposed

=== STDERR ===
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.

```

### ✓ agent_act (line_profile)

- **Duration:** 17.14s
- **Output:** `profiling_results/phase3/line_profile_agent_act.txt`

**Profile Data (top 30 lines):**
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    20                                           @profile
    21                                           def profiled_act(self):
    22       131    1215728.8   9280.4    100.0      return original_act(self)

2025-10-02 04:07:12 [debug    ] Database engine disposed

=== STDERR ===
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/home/ubuntu/.local/lib/python3.13/site-packages/tianshou/policy/modelfree/dqn.py:158: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /pytorch/torch/csrc/autograd/python_variable_indexing.cpp:306.)
  obs = batch[input]

```

### ✓ spatial_update (line_profile)

- **Duration:** 0.81s
- **Output:** `profiling_results/phase3/line_profile_spatial_update.txt`

**Profile Data (top 30 lines):**
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    18                                           @profile
    19                                           def profiled_update(self):
    20        10       4598.4    459.8    100.0      return original_update(self)


=== STDERR ===

```

### ✗ database_log (line_profile)

- **Duration:** 1.97s
- **Output:** `profiling_results/phase3/line_profile_database_log.txt`

## How to Interpret Results

### Line Profiler Output

- **Line #**: Line number in source code
- **Hits**: Number of times line was executed
- **Time**: Total time spent on that line (microseconds)
- **Per Hit**: Average time per execution
- **% Time**: Percentage of total function time

**Focus on:**
- Lines with high % Time (major contributors)
- Lines with high Hits and significant time (optimization opportunity)
- Unexpected slow lines (algorithmic issues)

### Memory Profiler Output

- **Mem usage**: Memory used at that point
- **Increment**: Memory added/removed by that line

**Focus on:**
- Large increments (memory allocations)
- Lines with repeated allocations
- Memory leaks (increasing without decreasing)

## Next Steps

1. Review line-by-line profiles for each function
2. Identify specific lines consuming most time
3. Analyze why those lines are slow:
   - Algorithmic complexity?
   - Unnecessary computations?
   - Inefficient data structures?
   - Too many allocations?
4. Plan specific optimizations for hot lines
5. Implement and validate improvements with benchmarks
