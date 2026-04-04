# API Reference

High-level map of major entry points in AgentFarm. For authoritative signatures, use the source modules cited below and [usage examples](usage_examples.md).

## Core API Index

A quick map of core modules and primary entry points:

### farm/core/environment.py
- Environment (PettingZoo AEC): orchestrates the simulation
  - action_space(agent?): Discrete action space (dynamic mapping)
  - observation_space(agent?): Box observation space (channels, (C, S, S))
  - reset(seed?, options?): initializes resources, PettingZoo state
  - step(action?): AEC-compliant step; updates once per cycle
  - get_nearby_agents/resources, get_nearest_resource: spatial queries
  - update(): cycle-level updates (regeneration, metrics, time)

### farm/core/state.py
- BaseState: abstract interface for to_tensor/to_dict
- AgentState, EnvironmentState, SimulationState: normalized states
- ModelState: captures ML module state and recent metrics

### farm/core/action.py
- ActionType enum; central action registry (`action_registry`)
- Helper utilities: distance, validation, name↔index mapping
- Built-in actions: move, gather, share, defend, attack, reproduce, pass

### farm/core/agent/
- **AgentCore**: coordinator over components + `IAgentBehavior`
- **AgentFactory**: builds agents for the simulation
- **Components / behaviors**: movement, combat, perception, learning, etc. (see package `__init__.py`)

### farm/core/observations.py and channels.py
- ObservationConfig: radius, decay, dtype/device
- AgentObservation: hybrid sparse/dense observation buffer
- ChannelRegistry and handlers: SELF_HP, RESOURCES, VISIBILITY, etc.

### farm/core/spatial (module)
- SpatialIndex: Orchestrates KD-tree, Quadtree, and Spatial Hash indices; batch updates with dirty regions
- Quadtree, QuadtreeNode: Hierarchical spatial partitioning API
- SpatialHashGrid: Uniform grid bucketing for neighbor/range queries
- Backward compatibility: `farm.core.spatial_index` re-exports these symbols

### farm/core/resource_manager.py
- ResourceManager: initialize/regenerate/consume, stats, add/remove

### farm/core/metrics_tracker.py
- MetricsTracker: step + cumulative metrics; DB logging support
- StepMetrics, CumulativeMetrics

### farm/core/services
- interfaces.py: DI interfaces (ISpatialQueryService, ILoggingService, ...)
- implementations.py: environment-backed services and adapters
- factory.py: derive services from an Environment instance

### Utilities
- device_utils.py: device selection/validation helpers

## Core Module (`farm.core`)

### Environment Class

The main simulation environment that manages the world state, agents, and simulation loop.

#### Constructor

```python
Environment(width: int, height: int, resource_distribution: str = "uniform",
           obs_config: ObservationConfig = None, **kwargs)
```

**Parameters:**
- `width` (int): Grid width in cells
- `height` (int): Grid height in cells
- `resource_distribution` (str): Resource distribution pattern ("uniform", "clustered", "scattered")
- `obs_config` (ObservationConfig): Observation system configuration
- `**kwargs`: Additional configuration parameters

#### Key Methods

##### Simulation Control
```python
step(actions: Dict[str, Any]) -> Dict[str, Any]
```
Execute one simulation step with the given agent actions.

**Parameters:**
- `actions` (Dict[str, Any]): Dictionary mapping agent IDs to their actions

**Returns:**
- Dictionary containing step results and observations

```python
reset() -> None
```
Reset the environment to initial state.

```python
close() -> None
```
Clean up environment resources.

##### Agent Management
```python
add_agent(agent: BaseAgent) -> None
```
Add an agent to the environment.

```python
remove_agent(agent_id: str) -> None
```
Remove an agent from the environment.

```python
get_agent(agent_id: str) -> BaseAgent
```
Get agent by ID.

##### Spatial Queries
```python
get_nearby_agents(position: Tuple[int, int], radius: int) -> List[str]
```
Get IDs of agents within radius of position.

```python
get_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float
```
Calculate Euclidean distance between positions.

##### Resource Management
```python
get_resource_count(position: Tuple[int, int]) -> int
```
Get resource count at position.

```python
set_resource_count(position: Tuple[int, int], count: int) -> None
```
Set resource count at position.

```python
regenerate_resources() -> None
```
Regenerate resources according to environment rules.

##### Observation Interface
```python
get_observation(agent_id: str) -> torch.Tensor
```
Get observation tensor for agent.

```python
get_world_view() -> torch.Tensor
```
Get full world state tensor.

##### Metrics and Analysis
```python
get_metrics() -> Dict[str, Any]
```
Get current simulation metrics.

```python
get_agent_statistics() -> Dict[str, Any]
```
Get statistics about agent population.

```python
get_resource_statistics() -> Dict[str, Any]
```
Get statistics about resource distribution.

#### Properties

- `width` (int): Environment width
- `height` (int): Environment height
- `agents` (Dict[str, BaseAgent]): Active agents
- `resources` (ResourceManager): Resource system
- `current_step` (int): Current simulation step
- `terminated_agents` (Set[str]): Terminated agent IDs

### ObservationConfig Class

Configuration class for the observation system.

#### Constructor

```python
ObservationConfig(
    R: int = 6,
    fov_radius: int = 5,
    device: str = "cpu",
    dtype: str = "float32",
    gamma_trail: float = 0.90,
    gamma_dmg: float = 0.85,
    gamma_sig: float = 0.92,
    gamma_known: float = 0.98,
    initialization: str = "zeros",
    random_min: float = 0.0,
    random_max: float = 1.0,
    storage_mode: StorageMode = StorageMode.HYBRID,
    enable_metrics: bool = True,
    high_frequency_channels: List[str] = []
)
```

**Parameters:**
- `R` (int): Observation radius in cells
- `fov_radius` (int): Field-of-view radius
- `device` (str): Torch device
- `dtype` (str): Torch dtype string (e.g., "float32")
- `gamma_trail/gamma_dmg/gamma_sig/gamma_known` (float): Decay rates for dynamic channels
- `initialization` (str): Tensor init method ("zeros" or "random")
- `random_min/random_max` (float): Random initialization range
- `storage_mode` (StorageMode): HYBRID or DENSE
- `enable_metrics` (bool): Enable metrics collection
- `high_frequency_channels` (List[str]): Channels to prebuild densely for faster dense construction

#### Properties

- `R` (int): Observation radius
- `fov_radius` (int): Field-of-view radius
- `high_frequency_channels` (List[str]): Prebuilt channel names for frequent access

### AgentObservation Class

Manages an agent's local observation of the world.

#### Constructor

```python
AgentObservation(config: ObservationConfig)
```

#### Methods

```python
update_observation(agent_position: Tuple[int, int],
                  environment: Environment) -> None
```
Update observation from current world state.

```python
get_tensor() -> torch.Tensor
```
Get observation as tensor.

```python
get_channel(channel_name: str) -> torch.Tensor
```
Get specific channel data.

```python
clear_instant_channels() -> None
```
Clear instant channels for new data.

```python
apply_decay() -> None
```
Apply temporal decay to dynamic channels.

```python
get_metrics() -> Dict
```

Returns metrics related to dense construction and sparse storage, including:
- `dense_bytes`
- `sparse_points`
- `sparse_logical_bytes`
- `memory_reduction_percent`
- `cache_hits`, `cache_misses`, `cache_hit_rate`
- `dense_rebuilds`, `dense_rebuild_time_s_total`
- `grid_population_ops`, `vectorized_point_assign_ops`, `prebuilt_channel_copies`, `prebuilt_channels_active`

## Agents Module (`farm.core.agent`)

### BaseAgent Class

Unified agent class that handles all agent behaviors including movement, resource gathering, sharing, and combat.

#### Constructor

```python
BaseAgent(agent_id: str, position: Tuple[int, int],
         resource_level: int, environment: Environment, **kwargs)
```

#### Abstract Methods

```python
decide_action() -> Dict[str, Any]
```
Decide next action based on current state.

#### Concrete Methods

```python
move(target_position: Tuple[int, int]) -> bool
```
Move to target position.

```python
gather() -> int
```
Gather resources from current position.

```python
attack(target_agent: BaseAgent) -> bool
```
Attack target agent.

```python
share(target_agent: BaseAgent, amount: int) -> bool
```
Share resources with target agent.

```python
reproduce() -> BaseAgent
```
Create offspring agent.

#### Properties

- `agent_id` (str): Unique agent identifier
- `position` (Tuple[int, int]): Current position
- `resource_level` (int): Current resource level
- `age` (int): Agent age in steps
- `is_terminated` (bool): Whether agent is terminated
- `observation` (AgentObservation): Agent's observation system

### BaseAgent Class

Unified agent class that handles all agent behaviors including movement, resource gathering, sharing, and combat.

#### Constructor

```python
BaseAgent(agent_id: str, position: Tuple[int, int],
          resource_level: int, environment: Environment,
          action_set: list[Action] = [], parent_ids: list[str] = [],
          generation: int = 0, use_memory: bool = False,
          memory_config: Optional[dict] = None)
```

#### Additional Methods

```python
learn(experience: Tuple) -> None
```
Learn from experience tuple.

```python
get_action_probabilities(state: torch.Tensor) -> torch.Tensor
```
Get action probabilities for current state.

```python
update_target_network() -> None
```
Update target network for stable learning.

### IndependentAgent Class

Self-interested agent that prioritizes individual survival.

#### Constructor

```python
IndependentAgent(agent_id: str, position: Tuple[int, int],
                resource_level: int, environment: Environment, **kwargs)
```

### ControlAgent Class

Controlled agent for experimental conditions.

#### Constructor

```python
ControlAgent(agent_id: str, position: Tuple[int, int],
            resource_level: int, environment: Environment, **kwargs)
```

## Actions Module (`farm.actions`)

### Action Functions

#### Movement Action

```python
def move_action(agent: BaseAgent, target_position: Tuple[int, int],
               environment: Environment) -> Dict[str, Any]:
```
Execute movement action.

#### Gathering Action

```python
def gather_action(agent: BaseAgent, environment: Environment) -> Dict[str, Any]:
```
Execute resource gathering action.

#### Combat Action

```python
def attack_action(agent: BaseAgent, target_agent: BaseAgent,
                 environment: Environment) -> Dict[str, Any]:
```
Execute attack action.

#### Sharing Action

```python
def share_action(agent: BaseAgent, target_agent: BaseAgent,
                amount: int, environment: Environment) -> Dict[str, Any]:
```
Execute resource sharing action.

#### Reproduction Action

```python
def reproduce_action(agent: BaseAgent, environment: Environment) -> Dict[str, Any]:
```
Execute reproduction action.

### Action Configuration Classes

#### MovementConfig

```python
@dataclass
class MovementConfig:
    cost_factor: float = 1.0
    diagonal_movement: bool = True
    obstacle_penalty: float = 5.0
    pathfinding_algorithm: str = "a_star"
```

#### GatheringConfig

```python
@dataclass
class GatheringConfig:
    base_efficiency: float = 1.0
    specialization_bonus: float = 1.5
    max_carry_capacity: int = 50
```

## Channels Module (`farm.core.channels`)

### ChannelHandler Class

Abstract base class for observation channel processors.

#### Constructor

```python
ChannelHandler(name: str, behavior: ChannelBehavior,
              gamma: float = 1.0)
```

#### Abstract Methods

```python
process(observation: torch.Tensor, channel_idx: int,
        config: ObservationConfig, agent_world_pos: Tuple[int, int],
        **kwargs) -> None
```
Process and update channel observation.

#### Concrete Methods

```python
clear(observation: torch.Tensor, channel_idx: int) -> None
```
Clear channel if it's INSTANT behavior.

```python
decay(observation: torch.Tensor, channel_idx: int,
      config: ObservationConfig) -> None
```
Apply temporal decay to dynamic channels.

### ChannelRegistry Class

Global registry for managing observation channels.

#### Constructor

```python
ChannelRegistry()
```

#### Methods

```python
register(handler: ChannelHandler, index: Optional[int] = None) -> int
```
Register a channel handler.

```python
get_index(name: str) -> int
```
Get channel index by name.

```python
get_name(index: int) -> str
```
Get channel name by index.

```python
get_handler(name: str) -> ChannelHandler
```
Get channel handler by name.

```python
apply_decay(observation: torch.Tensor, config: ObservationConfig) -> None
```
Apply decay to all dynamic channels.

```python
clear_instant(observation: torch.Tensor) -> None
```
Clear all instant channels.

#### Properties

- `num_channels` (int): Total number of registered channels
- `max_index` (int): Highest channel index

### ChannelBehavior Enum

```python
class ChannelBehavior(Enum):
    INSTANT = "instant"    # Immediate information, refreshed each step
    DYNAMIC = "dynamic"    # Persistent information with temporal decay
```

### Built-in Channel Handlers

#### SelfHPHandler

```python
SelfHPHandler()
```
Handles agent's own health information.

#### AlliesHPHandler

```python
AlliesHPHandler()
```
Handles visible allies' health information.

#### EnemiesHPHandler

```python
EnemiesHPHandler()
```
Handles visible enemies' health information.

#### ResourceHandler

```python
ResourceHandler()
```
Handles resource location and quantity information.

#### VisibilityHandler

```python
VisibilityHandler()
```
Handles field-of-view and visibility masks.

#### KnownEmptyHandler

```python
KnownEmptyHandler(gamma: float = 0.98)
```
Handles memory of explored empty areas.

#### TransientEventHandler

```python
TransientEventHandler(gamma: float = 0.95)
```
Handles temporary events like damage or signals.

#### TrailHandler

```python
TrailHandler(gamma: float = 0.95)
```
Handles agent movement trails.

#### GoalHandler

```python
GoalHandler()
```
Handles current goals and waypoints.

## Database Module (`farm.database`)

### DatabaseConnection Class

Manages database connections and operations.

#### Constructor

```python
DatabaseConnection(db_path: str = "simulation.db")
```

#### Methods

```python
connect() -> None
```
Establish database connection.

```python
disconnect() -> None
```
Close database connection.

```python
execute_query(query: str, params: Tuple = None) -> List[Dict]
```
Execute SQL query.

```python
insert_agent_state(agent_data: Dict) -> int
```
Insert agent state data.

```python
insert_simulation_step(step_data: Dict) -> int
```
Insert simulation step data.

```python
get_agent_trajectory(agent_id: str) -> List[Dict]
```
Get agent's trajectory over time.

### Repository Classes

#### AgentRepository

```python
class AgentRepository:
    def __init__(self, db_connection: DatabaseConnection)

    def save(self, agent: BaseAgent) -> None
    def get_by_id(self, agent_id: str) -> Dict
    def get_all_active(self) -> List[Dict]
    def get_trajectory(self, agent_id: str) -> List[Dict]
    def get_statistics(self) -> Dict
```

#### ResourceRepository

```python
class ResourceRepository:
    def __init__(self, db_connection: DatabaseConnection)

    def save_state(self, resource_map: np.ndarray) -> None
    def get_state_at_step(self, step: int) -> np.ndarray
    def get_statistics(self) -> Dict
    def get_distribution_history(self) -> List[Dict]
```

#### ActionRepository

```python
class ActionRepository:
    def __init__(self, db_connection: DatabaseConnection)

    def save_action(self, agent_id: str, action: Dict, result: Dict) -> None
    def get_agent_actions(self, agent_id: str) -> List[Dict]
    def get_action_frequencies(self) -> Dict
    def get_success_rates(self) -> Dict
```

## Analysis (`farm.analysis` and related)

The analysis stack is **modular** (registry + per-domain packages), not a small set of classes named `ComparativeAnalysis` / `AgentAnalysis` / `ExperimentAnalysis`.

- **Module index**: [docs/analysis/modules/README.md](analysis/modules/README.md)
- **Orchestrated runs**: `AnalysisService`, `AnalysisRequest`, `AnalysisResult` in `farm.analysis.service`
- **Registry**: `farm.analysis.registry` (`get_module`, `get_module_names`, …)
- **Cross-run comparison (API)**: `compare_simulations` in `farm.analysis.comparative_analysis`
- **Lightweight SQL/pandas summaries**: `SimulationAnalyzer` in `farm.core.analysis`
- **Figures / charts**: `farm.charts.chart_analyzer.ChartAnalyzer`

## Configuration (`farm.config`)

- **Primary type**: `SimulationConfig` and nested dataclasses defined in `farm/config/config.py`. The `farm.config` package re-exports the main public types (see `farm/config/__init__.py`).
- **Typical load paths**:
  - `SimulationConfig.from_centralized_config(environment=..., profile=...)`
  - `load_config(...)` in `farm.config` (wrapper around `ConfigurationOrchestrator`, see `farm/config/orchestrator.py`)
- **Also useful**: `ConfigurationOrchestrator`, `get_global_orchestrator`, validation types in `farm.config.validation`, templates in `farm.config.template`.

There is **no** `ConfigBuilder` in this repository. There are **no** standalone `save_config` / `merge_configs` / `validate_config` functions matching the old stubs that used to appear here.

Human-facing guides: [Configuration guide](config/configuration_guide.md), [configuration API notes](config/configuration_api.md).

## Runners (`farm.runners`)

### `ExperimentRunner` — `farm.runners.experiment_runner`

```python
ExperimentRunner(
    base_config: SimulationConfig,
    experiment_name: str,
    db_path: Optional[Path] = None,
    chart_analyzer: Optional[ChartAnalyzerProtocol] = None,
)
```

```python
run_iterations(
    num_iterations: int,
    config_variations: Optional[List[Dict]] = None,
    num_steps: int = 1000,
    path: Optional[Path] = None,
    run_analysis: bool = True,
) -> None
```

**Important:** when `config_variations` is used, each dict’s keys are applied with `setattr` on the **top-level** `SimulationConfig` only (see `_create_iteration_config`). Nested fields (`population`, `resources`, …) are not traversed automatically.

**Single simulation runs** use `farm.core.simulation.run_simulation` (returns `Environment`).

There is **no** `SimulationRunner` class in `farm.runners`.

## Utilities (`farm.utils`)

### Structured logging (primary)

Exported from `farm.utils` (see `farm/utils/__init__.py`):

```python
configure_logging(...)
get_logger(__name__)
bind_context(...) / unbind_context(...) / clear_context(...)
log_performance, log_errors, log_context, log_step, log_simulation, log_experiment
AgentLogger, DatabaseLogger, LogSampler, PerformanceMonitor
```

Use **`configure_logging`** (structlog-based), not a legacy `setup_logging` helper.

### Other helpers

- **`bilinear_distribute_value`** — `farm.utils.spatial` (re-exported from `farm.utils`).
- **Short IDs** — `ShortUUID` lives in `farm.utils.short_id` (not re-exported from `farm.utils`); identity helpers in `farm.utils.identity`.

## Exceptions (selected)

There is **no** single `AgentFarmError` hierarchy. Notable types:

- **Config loading / validation**: `ConfigurationError`, `ValidationError`, `ConfigurationValidator` in `farm.config.validation` (see that module for the full set).
- **Analysis pipeline**: `AnalysisError` and subclasses (`DataValidationError`, `ModuleNotFoundError`, `ConfigurationError`, …) in `farm.analysis.exceptions` — note `ModuleNotFoundError` here is **not** the Python builtin.

Search the codebase for `class .*Error` when debugging a specific subsystem.

## Constants and enums (selected)

### Action types

Executable actions use **`ActionType`** (`IntEnum`) and the **`Action`** dataclass in `farm.core.action` (values and ordering differ from older docs; read `farm/core/action.py`).

### Observation channels

Channels are identified by **string names** and `ChannelHandler` implementations in `farm.core.channels`, not by a single `Channel` `IntEnum` in the public API.

### Resource placement

`resource_distribution` passed to `Environment` is typically a **string** (e.g. `"uniform"`) or a callable; there is no guaranteed `ResourceDistribution` enum in core.

## Type hints

Types vary by submodule. Common patterns: agent ids as `str`, grid positions as `(int, int)` or floats depending on API, observations as `torch.Tensor` where documented in `farm.core.observations`.

---

This page is a **high-level index** and entry-point reference. Prefer reading the cited modules and [usage examples](usage_examples.md) for authoritative signatures.
