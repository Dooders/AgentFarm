# Centralized Storage Architecture

## Overview

This document provides visual diagrams to help understand the centralized storage architecture.

## Before: Separate Database Files

```
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Runner                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────┬──────────────┬───────
                              │             │              │
                              ▼             ▼              ▼
                    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                    │ sim_001.db   │  │ sim_002.db   │  │ sim_003.db   │
                    ├──────────────┤  ├──────────────┤  ├──────────────┤
                    │ agents       │  │ agents       │  │ agents       │
                    │ agent_states │  │ agent_states │  │ agent_states │
                    │ actions      │  │ actions      │  │ actions      │
                    │ resources    │  │ resources    │  │ resources    │
                    │ ...          │  │ ...          │  │ ...          │
                    └──────────────┘  └──────────────┘  └──────────────┘

Problems:
- Each database has schema overhead
- Hard to compare across simulations
- Need to open multiple files
- Difficult to track temporal patterns
```

## After: Centralized Database

```
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Runner                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ All simulations write to
                              │ the same database file
                              ▼
                    ┌──────────────────────────────┐
                    │   experiment_001.db          │
                    ├──────────────────────────────┤
                    │ experiments (metadata)       │
                    │ simulations (metadata)       │
                    ├──────────────────────────────┤
                    │ agents                       │
                    │   ├─ simulation_id: sim_001  │
                    │   ├─ simulation_id: sim_002  │
                    │   └─ simulation_id: sim_003  │
                    ├──────────────────────────────┤
                    │ agent_states                 │
                    │   ├─ simulation_id: sim_001  │
                    │   ├─ simulation_id: sim_002  │
                    │   └─ simulation_id: sim_003  │
                    ├──────────────────────────────┤
                    │ actions, resources, etc.     │
                    │   (all tagged with sim_id)   │
                    └──────────────────────────────┘

Benefits:
✓ One database file
✓ Easy cross-simulation queries
✓ Less storage overhead
✓ Better organization
```

## Architecture Layers

```
┌────────────────────────────────────────────────────────────────────┐
│                        Application Layer                           │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
│  │   Your Simulation Code   │  │   Query/Analysis Tools       │  │
│  │   - Environment          │  │   - query_experiment.py      │  │
│  │   - Agents               │  │   - ExperimentQueryTool      │  │
│  │   - run_simulation()     │  │   - pandas/matplotlib        │  │
│  └──────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Integration Layer                             │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
│  │   DatabaseFactory        │  │   ExperimentManager          │  │
│  │   - Smart creation       │  │   - High-level API           │  │
│  │   - Env var support      │  │   - Batch operations         │  │
│  │   - Same interface       │  │   - Status tracking          │  │
│  └──────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Database Layer                                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   ExperimentDatabase                         │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                │ │
│  │  │ SimulationContext│  │ SimulationContext│  ...           │ │
│  │  │ (sim_001)        │  │ (sim_002)        │                │ │
│  │  │  - logger        │  │  - logger        │                │ │
│  │  │  - query         │  │  - query         │                │ │
│  │  └──────────────────┘  └──────────────────┘                │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         SimulationDatabase (base class)                      │ │
│  │         - DataLogger, DataRetriever                          │ │
│  │         - Session management                                 │ │
│  │         - Transaction handling                               │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                 │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   SQLAlchemy ORM                             │ │
│  │  - Models: AgentModel, AgentStateModel, etc.                │ │
│  │  - Relationships and indexes                                │ │
│  │  - Query interface                                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   SQLite Database                            │ │
│  │  - WAL mode for concurrency                                 │ │
│  │  - Optimized PRAGMAs                                        │ │
│  │  - Indexes on simulation_id                                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Logging Simulation Data

```
Your Simulation Code
      │
      │ db.logger.log_step(step_number, agent_states, ...)
      │
      ▼
SimulationContext.logger (ExperimentDataLogger)
      │
      │ Tags all data with simulation_id
      │
      ▼
DataLogger (buffered writes)
      │
      │ Batch insert operations
      │
      ▼
SQLAlchemy Session
      │
      │ ORM operations
      │
      ▼
SQLite Database
      │
      │ Stored on disk with simulation_id tags
      │
      ▼
experiments/my_experiment.db
  ├─ agent_states
  │    ├─ (simulation_id: sim_001, step: 0, agent_id: a1, ...)
  │    ├─ (simulation_id: sim_001, step: 1, agent_id: a1, ...)
  │    ├─ (simulation_id: sim_002, step: 0, agent_id: a1, ...)
  │    └─ ...
  └─ ...
```

## Data Flow: Querying Simulation Data

```
Analysis/Query Code
      │
      │ tool.get_simulation_summary()
      │
      ▼
ExperimentQueryTool
      │
      │ Builds SQLAlchemy queries
      │
      ▼
SQLAlchemy Session
      │
      │ SELECT ... WHERE simulation_id = 'sim_001'
      │ GROUP BY simulation_id
      │
      ▼
SQLite Database
      │
      │ Uses indexes on simulation_id
      │ Returns filtered results
      │
      ▼
pandas DataFrame
      │
      │ Ready for analysis/visualization
      │
      ▼
Results (charts, statistics, exports)
```

## Component Interaction: Running an Experiment

```
Step 1: Setup
─────────────
ExperimentManager.__init__()
      │
      ▼
ExperimentDatabase(db_path, experiment_id)
      │
      ├─ Create/connect to database
      ├─ Create experiments table record
      └─ Initialize session management


Step 2: For Each Simulation
────────────────────────────
manager.create_simulation(sim_id, params)
      │
      ▼
ExperimentDatabase.create_simulation_context()
      │
      ├─ Create simulations table record
      └─ Return SimulationContext(sim_id)
            │
            ▼
      SimulationContext
            ├─ logger = ExperimentDataLogger(sim_id)
            ├─ parent_db = ExperimentDatabase
            └─ Same interface as SimulationDatabase


Step 3: Run Simulation
──────────────────────
Your simulation code
      │
      ├─ sim_context.logger.log_step(...)
      ├─ sim_context.logger.log_agent(...)
      ├─ sim_context.logger.log_action(...)
      └─ ... (all data automatically tagged with sim_id)


Step 4: Complete Simulation
────────────────────────────
manager.complete_simulation(sim_id, summary)
      │
      ├─ sim_context.flush_all_buffers()
      └─ UPDATE simulations SET status='completed', ...


Step 5: Cleanup
───────────────
manager.close() or __exit__()
      │
      ├─ Flush all contexts
      ├─ UPDATE experiments SET status='completed', ...
      └─ experiment_db.close()
```

## Database Schema: Key Tables

```
┌─────────────────────────────────────────────────────────────────┐
│                     experiments                                 │
├─────────────────────────────────────────────────────────────────┤
│ experiment_id (PK) │ name │ status │ creation_date │ ...        │
├─────────────────────────────────────────────────────────────────┤
│ exp_001            │ "..."│ running│ 2025-01-02    │ ...        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1:many
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     simulations                                 │
├─────────────────────────────────────────────────────────────────┤
│ simulation_id (PK) │ experiment_id (FK) │ status │ params │ ...│
├─────────────────────────────────────────────────────────────────┤
│ sim_001            │ exp_001            │ running│ {...}  │ ...│
│ sim_002            │ exp_001            │ running│ {...}  │ ...│
│ sim_003            │ exp_001            │ done   │ {...}  │ ...│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1:many
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────┐
│ agent_states │    │   agents     │      │   actions    │
├──────────────┤    ├──────────────┤      ├──────────────┤
│ sim_id (FK)  │    │ sim_id (FK)  │      │ sim_id (FK)  │
│ agent_id     │    │ agent_id(PK) │      │ step         │
│ step         │    │ birth_time   │      │ agent_id     │
│ health       │    │ agent_type   │      │ action_type  │
│ resources    │    │ ...          │      │ ...          │
│ ...          │    └──────────────┘      └──────────────┘
└──────────────┘

All data tables include simulation_id to separate data from different runs.
Indexed on (simulation_id, primary_key) for fast queries.
```

## Usage Patterns

### Pattern 1: Environment Variables
```
┌──────────────────┐
│ Set env vars:    │
│ USE_EXPERIMENT_DB│
│ EXPERIMENT_ID    │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Run existing     │
│ simulation code  │
│ (no changes!)    │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ DatabaseFactory  │
│ detects env vars │
│ creates          │
│ ExperimentDB     │
└──────────────────┘
```

### Pattern 2: DatabaseFactory
```
┌──────────────────┐
│ Import           │
│ DatabaseFactory  │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ db = Factory     │
│   .create(...)   │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Returns          │
│ SimulationContext│
│ with same        │
│ interface        │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Use db.logger    │
│ as normal        │
└──────────────────┘
```

### Pattern 3: ExperimentManager
```
┌──────────────────────┐
│ with                 │
│ ExperimentManager()  │
│   as mgr:            │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ for each sim:        │
│   ctx = mgr.create() │
│   run_sim(ctx)       │
│   mgr.complete()     │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Auto cleanup on exit │
│ Status tracking      │
│ Error handling       │
└──────────────────────┘
```

## Comparison: Query Performance

### Before (Separate Databases)
```
for each database file:
    ┌─────────────────────┐
    │ Open connection     │  ← Overhead
    │ Load schema         │  ← Overhead
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Query data          │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Close connection    │
    └─────────────────────┘
    
Total: N × (overhead + query time)
```

### After (Centralized Database)
```
┌─────────────────────┐
│ Open connection     │  ← Once
│ Load schema         │  ← Once
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Query all sims      │  ← Single query
│ WHERE sim_id IN(...) │     with index
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Close connection    │
└─────────────────────┘

Total: overhead + one query time
Speedup: Significant for N > 10
```

## Summary

The architecture provides:

1. **Backward Compatible**: Same interface as before
2. **Flexible**: Multiple usage patterns (env vars, factory, manager)
3. **Scalable**: Efficient queries across many simulations
4. **Organized**: Single database file per experiment
5. **Well-Tested**: Builds on existing ExperimentDatabase class

The key innovation is the `simulation_id` tag on all data, which allows:
- Multiple simulations in one database
- Fast filtering by simulation
- Easy cross-simulation comparisons
- Better temporal tracking
