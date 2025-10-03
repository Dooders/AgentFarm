# Analysis Code State Report
**Date:** 2025-10-03  
**Purpose:** Documentation of all analysis code locations before consolidation into `farm.analysis`

## Executive Summary

Analysis code is currently distributed across **6 major locations** with significant duplication and overlap. This report catalogs all analysis modules, scripts, and utilities to support the planned consolidation into a unified `farm.analysis` module.

### Key Metrics
- **Total Analysis Modules:** 70+ files
- **Main Locations:** 6 directories
- **Lines of Code:** ~15,000+ (estimated)
- **Analysis Types:** Population, Resources, Actions, Agents, Reproduction, Combat, Learning, Social, Dominance, Advantage, Genesis

---

## 1. Primary Analysis Module: `farm/analysis/`

**Status:** ✅ Modern, protocol-based architecture (v2.0.0)  
**Structure:** Well-organized with submodules  
**Purpose:** Primary analysis framework for simulation data

### Architecture
- **Core Framework:**
  - `protocols.py` - Type-safe protocol definitions
  - `core.py` - Base implementation classes
  - `validation.py` - Data validators
  - `exceptions.py` - Custom exception types
  - `registry.py` - Module discovery and registration
  - `service.py` - High-level service API
  - `base_module.py` - Legacy base class (deprecated)
  - `base.py` - Shared utilities

- **Common Utilities:**
  - `common/context.py` - Analysis execution context
  - `common/metrics.py` - Shared metrics calculations
  
- **Data Processing:**
  - `data/loaders.py` - Data loading utilities
  - `data/processors.py` - Data transformation pipelines

### Analysis Submodules

#### 1.1 Dominance Analysis (`dominance/`)
**Purpose:** Analyze agent dominance patterns and competitive dynamics

**Files:**
- `module.py` - Module implementation
- `analyze.py` - Core analysis functions
- `compute.py` - Statistical computations
- `plot.py` - Visualization functions
- `data.py` - Data processing
- `features.py` - Feature engineering
- `models.py` - Data models
- `ml.py` - Machine learning models
- `pipeline.py` - Analysis pipeline
- `db_io.py` - Database I/O
- `validation.py` - Dominance-specific validation
- `sqlalchemy_models.py` - Database models
- `query_dominance_db.py` - Database queries
- `DB.md` - Database documentation

**Key Functions:**
- Dominance hierarchy calculation
- Competition metrics
- Agent performance comparison
- Temporal dominance patterns

#### 1.2 Genesis Analysis (`genesis/`)
**Purpose:** Analyze initial conditions and early simulation dynamics

**Files:**
- `analyze.py` - Analysis functions
- `compute.py` - Genesis metrics computation
- `plot.py` - Visualization functions
- `README.md` - Module documentation

**Key Functions:**
- Initial positioning analysis
- Early population dynamics
- Founding lineage tracking
- Genesis event detection

#### 1.3 Advantage Analysis (`advantage/`)
**Purpose:** Analyze relative advantages between agent types

**Files:**
- `analyze.py` - Advantage analysis
- `compute.py` - Advantage calculations
- `plot.py` - Visualization
- `sqlalchemy_models.py` - Database models
- `import_csv_to_db.py` - Data import utilities
- `query_relative_advantage_db.py` - Database queries
- `DB.md` - Database documentation

**Key Functions:**
- Relative advantage computation
- Agent type comparison
- Advantage evolution over time
- Competitive edge analysis

#### 1.4 Social Behavior Analysis (`social_behavior/`)
**Purpose:** Analyze social interactions and cooperation

**Files:**
- `analyze.py` - Social analysis functions
- `compute.py` - Social metrics computation

**Key Functions:**
- Cooperation patterns
- Resource sharing analysis
- Social network metrics
- Group behavior analysis

### Standalone Analysis Modules

- `action_type_distribution.py` - Action frequency and distribution analysis
- `agent_analysis.py` - Agent-level metrics
- `comparative_analysis.py` - Cross-simulation comparison
- `health_resource_dynamics.py` - Health and resource correlation
- `learning_experience.py` - Learning and experience metrics
- `reproduction_diagnosis.py` - Reproduction failure diagnosis
- `reward_efficiency.py` - Reward efficiency analysis
- `null_module.py` - Null/placeholder module

### Supporting Files
- `template/module.py` - Template for new analysis modules
- `ARCHITECTURE.md` - System architecture documentation
- `README.md` - User guide
- `QUICK_REFERENCE.md` - Quick reference guide

---

## 2. Database Analyzers: `farm/database/analyzers/`

**Status:** ✅ Well-structured, database-focused  
**Purpose:** Direct database querying and analysis

### Analyzer Modules

1. **`simulation_analyzer.py`**
   - Survival rate calculation
   - Resource distribution analysis
   - Competitive interaction patterns
   - Resource efficiency metrics
   - Report generation

2. **`population_analyzer.py`**
   - Population dynamics over time
   - Birth/death rates
   - Population composition
   - Carrying capacity analysis

3. **`resource_analyzer.py`**
   - Resource accumulation patterns
   - Resource distribution
   - Efficiency metrics
   - Resource flow analysis

4. **`learning_analyzer.py`**
   - Learning curve analysis
   - Experience accumulation
   - Skill development tracking
   - Learning efficiency

5. **`action_stats_analyzer.py`**
   - Action frequency analysis
   - Action success rates
   - Action type distribution
   - Temporal action patterns

6. **`agent_analyzer.py`**
   - Individual agent metrics
   - Agent lifecycle analysis
   - Performance tracking
   - Behavioral patterns

7. **`temporal_pattern_analyzer.py`**
   - Time-series pattern detection
   - Seasonal patterns
   - Trend analysis
   - Periodicity detection

8. **`sequence_pattern_analyzer.py`**
   - Action sequence analysis
   - Pattern mining
   - Sequence clustering
   - Behavioral patterns

9. **`resource_impact_analyzer.py`**
   - Resource impact on outcomes
   - Resource-performance correlation
   - Resource threshold analysis

10. **`spatial_analysis.py`**
    - Spatial distribution analysis
    - Clustering patterns
    - Territory analysis
    - Movement patterns

11. **`causal_analyzer.py`**
    - Causal relationship detection
    - Intervention analysis
    - Causal graph construction

12. **`behavior_clustering_analyzer.py`**
    - Agent behavior clustering
    - Behavioral phenotypes
    - Cluster evolution

13. **`lifespan_analysis.py`**
    - Lifespan distribution
    - Mortality patterns
    - Survival curves

14. **`movement_analysis.py`**
    - Movement patterns
    - Migration analysis
    - Spatial behavior

15. **`location_analysis.py`**
    - Location preferences
    - Spatial distribution
    - Territory analysis

16. **`decision_pattern_analyzer.py`**
    - Decision tree analysis
    - Choice patterns
    - Decision quality metrics

17. **`analysis_utils.py`**
    - Shared utility functions
    - Statistical helpers
    - Data processing utilities

---

## 3. Chart Generation: `farm/charts/`

**Status:** ✅ Mature visualization system  
**Purpose:** Generate charts and visual analysis

### Chart Modules

1. **`chart_analyzer.py`** - Main analysis orchestrator
   - Analyzes all charts
   - LLM-powered insights (optional)
   - Comprehensive report generation

2. **`chart_simulation.py`** - Simulation-level charts
   - Population dynamics
   - Births and deaths
   - Resource efficiency
   - Agent health and age
   - Combat metrics
   - Resource sharing
   - Evolutionary metrics
   - Resource distribution entropy
   - Rewards
   - Average resources
   - Generational analysis

3. **`chart_actions.py`** - Action-level charts
   - Action type distribution
   - Action target distribution
   - Rewards by action type
   - Resource changes
   - Position changes
   - Action frequency over time
   - Rewards over time

4. **`chart_agents.py`** - Agent-level charts
   - Agent types over time
   - Lifespan distribution
   - Lineage size
   - Reproduction success rate

5. **`chart_resources.py`** - Resource-specific charts
   - Resource levels over time
   - Resource distribution
   - Resource consumption patterns

6. **`chart_states.py`** - Agent state charts
   - State transitions
   - State distribution
   - State evolution

7. **`chart_experience.py`** - Learning/experience charts
   - Experience accumulation
   - Learning curves
   - Skill development

8. **`chart_utils.py`** - Utility functions
   - Plot saving
   - Styling
   - Common formatting

9. **`llm_client.py`** - LLM integration
   - Chart analysis via LLM
   - Natural language insights

---

## 4. Research Analysis: `farm/research/analysis/`

**Status:** ✅ Research-focused  
**Purpose:** Research experiment analysis and plotting

### Modules

1. **`analysis.py`** - Core research analysis functions
   - `analyze_final_agent_counts()` - Final population analysis
   - `detect_early_terminations()` - Termination detection
   - `find_experiments()` - Experiment discovery
   - `process_action_distributions()` - Action distribution processing
   - `process_experiment()` - Full experiment processing
   - `process_experiment_by_agent_type()` - Type-specific analysis
   - `process_experiment_resource_consumption()` - Resource analysis
   - `process_experiment_resource_levels()` - Resource level tracking
   - `process_experiment_rewards_by_generation()` - Generational rewards
   - `validate_population_data()` - Population data validation
   - `validate_resource_level_data()` - Resource validation

2. **`database.py`** - Database interaction
   - `find_simulation_databases()` - Database discovery
   - `get_action_distribution_data()` - Action data retrieval
   - `get_columns_data()` - Column-specific queries
   - `get_columns_data_by_agent_type()` - Type-specific queries
   - `get_data()` - Generic data retrieval
   - `get_resource_consumption_data()` - Resource consumption
   - `get_resource_level_data()` - Resource levels
   - `get_rewards_by_generation()` - Generational rewards

3. **`dataframes.py`** - DataFrame utilities
   - `create_population_df()` - Population DataFrame creation

4. **`plotting.py`** - Research plotting functions
   - `plot_action_distributions()` - Action distribution plots
   - `plot_early_termination_analysis()` - Termination analysis plots
   - `plot_final_agent_counts()` - Final count visualization
   - `plot_marker_point()` - Marker plotting
   - `plot_mean_and_ci()` - Mean with confidence intervals
   - `plot_median_line()` - Median line plotting
   - `plot_population_trends_across_simulations()` - Population trends
   - `plot_population_trends_by_agent_type()` - Type-specific trends
   - `plot_reference_line()` - Reference line plotting
   - `plot_resource_consumption_trends()` - Resource consumption
   - `plot_resource_level_trends()` - Resource levels
   - `plot_rewards_by_generation()` - Generational rewards
   - `setup_plot_aesthetics()` - Plot styling

5. **`main.py`** - Main orchestration
   - Research analysis pipeline execution

6. **`util.py`** - Utility functions
   - `calculate_statistics()` - Statistical calculations

---

## 5. Analysis Scripts: `scripts/`

**Status:** ⚠️ Mixed quality, some deprecated  
**Purpose:** Standalone analysis scripts and utilities

### Core Analysis Scripts

1. **`scripts/analysis/core_analysis.py`** - Unified experiment analyzer
   - `UnifiedExperimentAnalyzer` class
   - Population dynamics analysis
   - Resource dynamics analysis
   - Action pattern analysis
   - Reproduction analysis
   - Comprehensive report generation
   - Visualization reports

2. **`scripts/analysis/social_analysis.py`** - Social behavior analysis

### Specialized Analysis Scripts

3. **`advantage_analysis.py`** - Advantage analysis orchestration
   - Uses `farm.analysis.advantage` module
   - Generates comprehensive advantage reports

4. **`genesis_analysis.py`** - Genesis event analysis
   - Initial positioning analysis
   - Founding lineage analysis

5. **`reproduction_analysis.py`** - Reproduction pattern analysis
   - Success/failure analysis
   - Resource requirements
   - Temporal patterns

6. **`generational_fitness_analysis.py`** - Generational fitness tracking
   - Fitness evolution
   - Selection pressure analysis

7. **`analyze_time_series.py`** - Time series analysis utilities
   - Trend detection
   - Pattern analysis

### Supporting Utilities

8. **`analysis_config.py`** - Analysis configuration
   - Path configuration
   - Logging setup
   - Common settings

9. **`data_extraction.py`** - Data extraction utilities
   - `extract_time_series()` - Time series extraction
   - `validate_dataframe()` - DataFrame validation

10. **`database_utils.py`** - Database utilities
    - `create_database_session()` - Session management
    - `get_simulation_database_path()` - Path resolution
    - `get_simulation_folders()` - Folder discovery
    - `validate_simulation_folder()` - Validation

11. **`visualization_utils.py`** - Visualization helpers
    - `create_time_series_plot()` - Time series plotting
    - `save_figure()` - Figure saving

12. **`visualization/mosaic_viz.py`** - Mosaic visualizations

### Other Scripts

13. **`animate_simulation.py`** - Animation generation
14. **`controlled_initial_conditions.py`** - Initial condition setup
15. **`create_initial_positioning_visualization.py`** - Positioning viz
16. **`predict.py`** - Prediction utilities
17. **`significant_events.py`** - Event detection

---

## 6. Top-Level Analysis: `/workspace/analysis/`

**Status:** ⚠️ Outdated, likely superseded  
**Purpose:** Legacy analysis scripts

### Files

1. **`simulation_analysis.py`**
   - Legacy simulation analyzer
   - Basic analysis capabilities
   - Should be migrated or deprecated

2. **`simulation_comparison.py`**
   - Cross-simulation comparison
   - Comparative metrics

3. **`reproducibility.py`**
   - Reproducibility testing
   - Seed validation

**Recommendation:** Migrate useful functionality to `farm.analysis`, deprecate rest.

---

## 7. API Analysis Controller: `farm/api/analysis_controller.py`

**Status:** ✅ Modern API interface  
**Purpose:** HTTP API for analysis execution

### Features
- Start/stop/pause analysis
- Progress monitoring
- State management
- Result retrieval
- Batch analysis
- Callback registration

**Integration:** Works with `farm.analysis.service.AnalysisService`

---

## 8. Core Analysis: `farm/core/analysis.py`

**Status:** ✅ Core functionality  
**Purpose:** Simulation-level analysis integration

### Functions
- `analyze_simulation()` - Main analysis entry point
- Integration with simulation lifecycle
- Real-time analysis hooks

---

## 9. Utility Analysis: `farm/utils/`

**Status:** ✅ Supporting utilities  
**Purpose:** Analysis support and validation

### Files

1. **`run_analysis.py`**
   - CLI for running analysis
   - Analysis orchestration

2. **`chart_analyzer_validator.py`**
   - Chart analyzer validation
   - Quality checks

---

## Analysis Categories and Capabilities

### By Analysis Type

#### 1. Population Analysis
**Locations:**
- `farm/analysis/` (base framework)
- `farm/database/analyzers/population_analyzer.py`
- `farm/charts/chart_simulation.py` (visualization)
- `farm/research/analysis/analysis.py`
- `scripts/analysis/core_analysis.py`

**Capabilities:**
- Population dynamics over time
- Birth/death rates
- Agent type composition
- Survival rates
- Carrying capacity
- Population stability

#### 2. Resource Analysis
**Locations:**
- `farm/analysis/health_resource_dynamics.py`
- `farm/database/analyzers/resource_analyzer.py`
- `farm/database/analyzers/resource_impact_analyzer.py`
- `farm/charts/chart_resources.py`
- `farm/research/analysis/analysis.py`

**Capabilities:**
- Resource accumulation
- Resource distribution
- Efficiency metrics
- Resource flow
- Impact on outcomes
- Consumption patterns

#### 3. Action Analysis
**Locations:**
- `farm/analysis/action_type_distribution.py`
- `farm/database/analyzers/action_stats_analyzer.py`
- `farm/database/analyzers/sequence_pattern_analyzer.py`
- `farm/database/analyzers/decision_pattern_analyzer.py`
- `farm/charts/chart_actions.py`

**Capabilities:**
- Action frequency
- Action success rates
- Action sequences
- Decision patterns
- Reward analysis
- Temporal patterns

#### 4. Agent Analysis
**Locations:**
- `farm/analysis/agent_analysis.py`
- `farm/database/analyzers/agent_analyzer.py`
- `farm/database/analyzers/lifespan_analysis.py`
- `farm/database/analyzers/behavior_clustering_analyzer.py`
- `farm/charts/chart_agents.py`

**Capabilities:**
- Individual agent metrics
- Lifespan analysis
- Behavioral clustering
- Performance tracking
- Agent lifecycle

#### 5. Learning Analysis
**Locations:**
- `farm/analysis/learning_experience.py`
- `farm/database/analyzers/learning_analyzer.py`
- `farm/charts/chart_experience.py`

**Capabilities:**
- Learning curves
- Experience accumulation
- Skill development
- Learning efficiency

#### 6. Social Analysis
**Locations:**
- `farm/analysis/social_behavior/`
- `scripts/analysis/social_analysis.py`

**Capabilities:**
- Cooperation patterns
- Resource sharing
- Social networks
- Group behavior

#### 7. Dominance Analysis
**Locations:**
- `farm/analysis/dominance/` (comprehensive module)

**Capabilities:**
- Dominance hierarchies
- Competition metrics
- Agent performance comparison
- Temporal dominance patterns
- Machine learning models

#### 8. Advantage Analysis
**Locations:**
- `farm/analysis/advantage/` (comprehensive module)
- `scripts/advantage_analysis.py`

**Capabilities:**
- Relative advantage computation
- Agent type comparison
- Advantage evolution
- Competitive edge analysis

#### 9. Genesis Analysis
**Locations:**
- `farm/analysis/genesis/`
- `scripts/genesis_analysis.py`

**Capabilities:**
- Initial positioning
- Early dynamics
- Founding lineage tracking
- Genesis event detection

#### 10. Reproduction Analysis
**Locations:**
- `farm/analysis/reproduction_diagnosis.py`
- `scripts/reproduction_analysis.py`

**Capabilities:**
- Success/failure analysis
- Resource requirements
- Temporal patterns
- Failure diagnosis

#### 11. Spatial Analysis
**Locations:**
- `farm/database/analyzers/spatial_analysis.py`
- `farm/database/analyzers/movement_analysis.py`
- `farm/database/analyzers/location_analysis.py`

**Capabilities:**
- Spatial distribution
- Clustering patterns
- Territory analysis
- Movement patterns
- Migration analysis

#### 12. Temporal Analysis
**Locations:**
- `farm/database/analyzers/temporal_pattern_analyzer.py`
- `scripts/analyze_time_series.py`

**Capabilities:**
- Time-series patterns
- Seasonal patterns
- Trend analysis
- Periodicity detection

#### 13. Comparative Analysis
**Locations:**
- `farm/analysis/comparative_analysis.py`
- `analysis/simulation_comparison.py`
- `farm/research/analysis/`

**Capabilities:**
- Cross-simulation comparison
- Experiment comparison
- Parameter sensitivity
- A/B testing

---

## Code Quality Assessment

### ✅ High Quality (Ready for use)
- `farm/analysis/` - Modern, well-architected
- `farm/database/analyzers/` - Clean, focused
- `farm/charts/` - Comprehensive visualization
- `farm/research/analysis/` - Well-structured
- `farm/api/analysis_controller.py` - Good API design

### ⚠️ Needs Review
- `scripts/` - Mixed quality, some duplication
- `analysis/` - Outdated, should be migrated

### 🔴 Deprecated/Redundant
- `farm/analysis/base_module.py` - Old system
- Some scripts that duplicate `farm.analysis` functionality

---

## Duplication and Overlap

### Major Duplications

1. **Population Analysis**
   - `farm/database/analyzers/population_analyzer.py`
   - `scripts/analysis/core_analysis.py` 
   - `farm/research/analysis/analysis.py`
   - **Recommendation:** Consolidate into `farm.analysis.population` module

2. **Resource Analysis**
   - `farm/database/analyzers/resource_analyzer.py`
   - `farm/database/analyzers/resource_impact_analyzer.py`
   - `scripts/analysis/core_analysis.py`
   - **Recommendation:** Consolidate into `farm.analysis.resources` module

3. **Action Analysis**
   - `farm/analysis/action_type_distribution.py`
   - `farm/database/analyzers/action_stats_analyzer.py`
   - `scripts/analysis/core_analysis.py`
   - **Recommendation:** Consolidate into `farm.analysis.actions` module

4. **Visualization**
   - `farm/charts/` (comprehensive)
   - `farm/research/analysis/plotting.py` (research-specific)
   - `scripts/visualization_utils.py` (utilities)
   - **Recommendation:** Keep separate but share common utilities

---

## Dependencies and Integration

### Internal Dependencies
```
farm.analysis.service
├── farm.analysis.core (base classes)
├── farm.analysis.protocols (interfaces)
├── farm.analysis.registry (module discovery)
├── farm.analysis.validation (data validation)
└── farm.analysis.{module}/ (specific analyses)

farm.database.analyzers.*
├── farm.database.database (SimulationDatabase)
├── farm.database.models (SQLAlchemy models)
└── farm.database.repositories (data access)

farm.charts.*
├── farm.database.database (data source)
└── farm.charts.llm_client (optional LLM insights)

farm.api.analysis_controller
└── farm.analysis.service (orchestration)
```

### External Dependencies
- pandas - Data manipulation
- numpy - Numerical computations
- matplotlib - Visualization
- sqlalchemy - Database access
- scipy - Statistical functions (some modules)
- scikit-learn - Machine learning (dominance module)

---

## Recommendations for Consolidation

### Phase 1: Core Infrastructure (Complete ✅)
- `farm.analysis` framework is complete
- Protocol-based architecture in place
- Service layer operational

### Phase 2: Migrate Database Analyzers
**Priority:** HIGH  
**Effort:** MEDIUM

1. Create new analysis modules in `farm.analysis/`:
   - `population/` - from `farm.database.analyzers.population_analyzer`
   - `resources/` - from `resource_analyzer` + `resource_impact_analyzer`
   - `actions/` - from `action_stats_analyzer` + `sequence_pattern_analyzer`
   - `agents/` - from `agent_analyzer` + `lifespan_analysis`
   - `learning/` - from `learning_analyzer`
   - `spatial/` - from `spatial_analysis` + `movement_analysis`
   - `temporal/` - from `temporal_pattern_analyzer`

2. Keep database-specific helpers in `farm.database.analyzers/analysis_utils.py`

### Phase 3: Consolidate Scripts
**Priority:** MEDIUM  
**Effort:** HIGH

1. Migrate useful functionality from `scripts/` to appropriate `farm.analysis` modules
2. Keep orchestration scripts (`advantage_analysis.py`, etc.) that use modules
3. Deprecate duplicate implementations
4. Move utilities to `farm.analysis.common.utils`

### Phase 4: Deprecate Legacy Code
**Priority:** LOW  
**Effort:** LOW

1. Mark `/workspace/analysis/` as deprecated
2. Add deprecation warnings
3. Document migration path
4. Remove in future release

### Phase 5: Unify Visualization
**Priority:** MEDIUM  
**Effort:** MEDIUM

1. Extract common plotting utilities from `farm.charts` and `farm.research.analysis.plotting`
2. Create `farm.analysis.visualization` module
3. Keep chart generation in `farm.charts` (simulation-focused)
4. Keep research plotting in `farm.research.analysis.plotting` (research-focused)

---

## Testing Coverage

### Well-Tested ✅
- `farm.analysis` - Comprehensive test suite in `tests/analysis/`
- `farm.database.analyzers` - Some coverage in `tests/`

### Needs Tests ⚠️
- `scripts/` - Limited testing
- `farm.charts` - Minimal testing
- `farm.research.analysis` - Needs more coverage

### Test Recommendations
1. Add integration tests for analyzer migration
2. Create regression tests for existing functionality
3. Add performance benchmarks for analysis operations

---

## Documentation Status

### Well-Documented ✅
- `farm/analysis/README.md` - Comprehensive user guide
- `farm/analysis/ARCHITECTURE.md` - Architecture overview
- `farm/analysis/QUICK_REFERENCE.md` - Quick reference
- Submodule documentation (DB.md files)

### Needs Documentation ⚠️
- `farm/database/analyzers/` - Need individual module docs
- `scripts/` - Limited documentation
- API documentation for chart generators

---

## Conclusion

The analysis code is currently well-structured in the `farm.analysis` module but has significant duplication in:
1. **Database analyzers** (`farm/database/analyzers/`) - Should be migrated to `farm.analysis`
2. **Analysis scripts** (`scripts/`) - Consolidate or deprecate
3. **Legacy code** (`/workspace/analysis/`) - Mark for deprecation

### Consolidation Benefits
- ✅ Single source of truth for analysis
- ✅ Reduced code duplication
- ✅ Better maintainability
- ✅ Improved testing
- ✅ Consistent API
- ✅ Easier discovery of capabilities

### Next Steps
1. Review and approve consolidation plan
2. Create new analysis modules following existing patterns
3. Migrate database analyzers
4. Update dependent code
5. Deprecate old implementations
6. Update documentation

---

## Appendix: File Counts

### By Location
- `farm/analysis/`: 55 files (49 Python, 6 Markdown)
- `farm/database/analyzers/`: 18 files (all Python)
- `farm/charts/`: 10 files (all Python)
- `farm/research/analysis/`: 6 files (all Python)
- `scripts/`: 25+ analysis-related files
- `/workspace/analysis/`: 3 files

**Total:** ~117 analysis-related files

### By Type
- Analysis modules: ~30
- Visualization modules: ~15
- Utility modules: ~20
- Scripts: ~25
- Documentation: ~15
- Tests: ~12

---

**Report Generated:** 2025-10-03  
**Report Version:** 1.0  
**For:** Analysis code consolidation project
