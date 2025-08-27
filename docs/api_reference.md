# API Reference

This comprehensive API reference documents all public classes, functions, and modules in AgentFarm. The API is organized by module and provides detailed information about parameters, return values, and usage examples.

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
ObservationConfig(R: int = 6, fov_radius: int = 5,
                 decay_factors: Dict[str, float] = None)
```

**Parameters:**
- `R` (int): Observation radius in cells
- `fov_radius` (int): Field-of-view radius
- `decay_factors` (Dict[str, float]): Channel-specific decay rates

#### Properties

- `R` (int): Observation radius
- `fov_radius` (int): Field-of-view radius
- `decay_factors` (Dict[str, float]): Decay factors for dynamic channels

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

## Agents Module (`farm.agents`)

### BaseAgent Class

Abstract base class for all agent types.

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

### SystemAgent Class

Standard agent with balanced behaviors and learning capabilities.

#### Constructor

```python
SystemAgent(agent_id: str, position: Tuple[int, int],
           resource_level: int, environment: Environment,
           learning_rate: float = 0.001, memory_size: int = 5000, **kwargs)
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

## Analysis Module (`farm.analysis`)

### ComparativeAnalysis Class

Performs comparative analysis between simulation runs.

#### Constructor

```python
ComparativeAnalysis(base_path: str = "results")
```

#### Methods

```python
compare_runs(run_paths: List[str]) -> Dict[str, Any]
```
Compare multiple simulation runs.

```python
generate_report(results: Dict, output_path: str) -> None
```
Generate analysis report.

```python
create_visualization(results: Dict, metric: str) -> plt.Figure
```
Create visualization for specific metric.

### AgentAnalysis Class

Analyzes individual agent behaviors.

#### Constructor

```python
AgentAnalysis(agent_id: str, simulation_path: str)
```

#### Methods

```python
analyze_behavior() -> Dict[str, Any]
```
Analyze agent behavior patterns.

```python
plot_trajectory() -> plt.Figure
```
Plot agent's movement trajectory.

```python
calculate_efficiency() -> float
```
Calculate agent's resource efficiency.

### ExperimentAnalysis Class

Analyzes experimental results.

#### Constructor

```python
ExperimentAnalysis(experiment_path: str)
```

#### Methods

```python
load_results() -> Dict[str, Any]
```
Load experiment results.

```python
perform_statistical_tests() -> Dict[str, Any]
```
Perform statistical analysis.

```python
generate_summary_report() -> str
```
Generate experiment summary.

## Configuration Module (`farm.core.config`)

### ConfigBuilder Class

Builder pattern for creating configurations programmatically.

#### Constructor

```python
ConfigBuilder()
```

#### Methods

```python
set_environment(width: int, height: int, **kwargs) -> ConfigBuilder
```
Set environment parameters.

```python
set_agents(count: int, resources: int, **kwargs) -> ConfigBuilder
```
Set agent parameters.

```python
set_learning(rate: float, memory: int, **kwargs) -> ConfigBuilder
```
Set learning parameters.

```python
enable_channels(channels: List[str]) -> ConfigBuilder
```
Enable specific observation channels.

```python
build() -> Dict[str, Any]
```
Build final configuration dictionary.

### Configuration Loading Functions

```python
def load_config(config_path: str) -> Dict[str, Any]
```
Load configuration from YAML file.

```python
def save_config(config: Dict[str, Any], config_path: str) -> None
```
Save configuration to YAML file.

```python
def merge_configs(base_config: Dict, override_config: Dict) -> Dict[str, Any]
```
Merge two configurations.

```python
def validate_config(config: Dict[str, Any]) -> List[str]
```
Validate configuration and return error messages.

## Runners Module (`farm.runners`)

### SimulationRunner Class

Runs individual simulations.

#### Constructor

```python
SimulationRunner(config: Dict[str, Any])
```

#### Methods

```python
run() -> Dict[str, Any]
```
Run simulation.

```python
run_with_visualization() -> Dict[str, Any]
```
Run simulation with real-time visualization.

```python
save_results(output_path: str) -> None
```
Save simulation results.

### ExperimentRunner Class

Runs parameter experiments.

#### Constructor

```python
ExperimentRunner()
```

#### Methods

```python
run_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]
```
Run parameter experiment.

```python
run_parameter_sweep(parameters: Dict[str, List],
                   base_config: Dict[str, Any],
                   replications: int = 3) -> Dict[str, Any]
```
Run parameter sweep.

```python
generate_comparison_report(results: Dict[str, Any],
                          output_path: str) -> None
```
Generate experiment comparison report.

## Utilities Module (`farm.utils`)

### ShortUUID Class

Generates short unique identifiers.

#### Constructor

```python
ShortUUID()
```

#### Methods

```python
generate() -> str
```
Generate short UUID.

### Timing Utilities

```python
def time_function(func: Callable) -> Callable
```
Decorator to time function execution.

```python
class Timer:
    def __enter__(self)
    def __exit__(self, *args)
    def elapsed() -> float
```
Context manager for timing code blocks.

### Logging Utilities

```python
def setup_logging(level: str = "INFO", log_file: str = None) -> None
```
Set up logging configuration.

```python
def get_logger(name: str) -> logging.Logger
```
Get configured logger.

## Exceptions

### Core Exceptions

```python
class AgentFarmError(Exception)
```
Base exception for AgentFarm errors.

```python
class ConfigurationError(AgentFarmError)
```
Configuration-related errors.

```python
class SimulationError(AgentFarmError)
```
Simulation execution errors.

```python
class DatabaseError(AgentFarmError)
```
Database operation errors.

### Channel Exceptions

```python
class ChannelError(AgentFarmError)
```
Channel-related errors.

```python
class ChannelNotFoundError(ChannelError)
```
Channel not found error.

```python
class ChannelRegistrationError(ChannelError)
```
Channel registration error.

### Agent Exceptions

```python
class AgentError(AgentFarmError)
```
Agent-related errors.

```python
class AgentNotFoundError(AgentError)
```
Agent not found error.

```python
class InvalidActionError(AgentError)
```
Invalid action error.

## Constants and Enums

### Action Enum

```python
class Action(IntEnum):
    MOVE = 0
    GATHER = 1
    ATTACK = 2
    SHARE = 3
    REPRODUCE = 4
    NO_OP = 5
```

### Channel Enum

```python
class Channel(IntEnum):
    SELF_HP = 0
    ALLIES_HP = 1
    ENEMIES_HP = 2
    RESOURCES = 3
    OBSTACLES = 4
    TERRAIN_COST = 5
    VISIBILITY = 6
    KNOWN_EMPTY = 7
    DAMAGE_HEAT = 8
    TRAILS = 9
    ALLY_SIGNAL = 10
    GOAL = 11
```

### ResourceDistribution Enum

```python
class ResourceDistribution(Enum):
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    SCATTERED = "scattered"
```

## Type Hints

### Common Types

```python
AgentID = str
Position = Tuple[int, int]
ActionDict = Dict[str, Any]
Observation = torch.Tensor
Config = Dict[str, Any]
Metrics = Dict[str, Union[int, float, List]]
```

### Generic Types

```python
from typing import TypeVar
AgentType = TypeVar('AgentType', bound=BaseAgent)
ChannelType = TypeVar('ChannelType', bound=ChannelHandler)
RepositoryType = TypeVar('RepositoryType', bound=BaseRepository)
```

This API reference provides comprehensive documentation for all public interfaces in AgentFarm. For more detailed examples and tutorials, see the usage examples documentation.
