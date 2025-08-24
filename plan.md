# Refactoring Plan: Making Environment Compatible with PettingZoo

Based on my analysis of the codebase, I've developed a comprehensive plan to refactor your `Environment` class to be fully compatible with PettingZoo's multi-agent environment interface. PettingZoo is the standard for multi-agent reinforcement learning and integrates seamlessly with Gymnasium.

## Key Insights from Codebase Analysis

- **Current Structure**: 
  - `Environment` manages a grid-based world with agents, resources, time steps, and database logging
  - Agents are autonomous with their own DQN modules (MoveModule, AttackModule, etc.) that decide actions internally
  - Multi-agent simulation where agents reproduce, die, and each has independent decision-making
  - Existing `BaseEnvironment` in `farm/environments/base_environment.py` already inherits from `gym.Env`

- **Perfect Fit for PettingZoo**:
  - Each agent has its own model/policy (decentralized learning)
  - Dynamic population (agents can be born/die during simulation)
  - Simultaneous actions by multiple agents
  - Individual observations and rewards per agent

- **Reproduction Architecture Decision**:
  - **Agent-Level Reproduction**: Agents handle their own reproduction through `reproduce()` and `create_offspring()` methods
  - **Environment Integration**: Agents call `environment.add_agent()` to register offspring and `environment.remove_agent()` on death
  - **PettingZoo Adaptation**: Environment's `add_agent()` and `remove_agent()` methods handle both existing logic and PettingZoo tracking
  - **Benefits**: Maintains agent autonomy, preserves existing sophisticated reproduction logic, enables different strategies per agent type

- **Challenges**:
  - Agents currently decide actions internally via `act()` method
  - Need to externalize action execution while preserving agent models
  - Database logging tightly coupled but should remain optional
  - State/observation standardization for PettingZoo interface

- **Opportunities**:
  - No backwards compatibility required - can aggressively refactor
  - Existing KD-trees and spatial queries can be reused for observations
  - Agent DQN modules can become policy networks for RL training
  - Config system can define PettingZoo spaces and parameters

## High-Level Refactoring Goals

- Transform `Environment` to inherit from `pettingzoo.ParallelEnv`
- Preserve agent models but make them compatible with external action provision
- Define standard PettingZoo spaces: per-agent observation and action spaces
- Maintain core features (resources, DB logging) as configurable options
- Update simulation runners to use PettingZoo interface
- Enable both autonomous mode (agents use internal models) and RL mode (external policies)

## Detailed Refactoring Plan

### Phase 1: Preparation and Dependencies

1. **Install PettingZoo**:
   ```bash
   pip install pettingzoo
   ```

2. **Define PettingZoo Spaces**:
   - **Observation Space**: Per-agent dict combining agent state and environment context
     ```python
     observation_space = spaces.Dict({
         "position": spaces.Box(low=0, high=1, shape=(2,)),  # Normalized x,y
         "health": spaces.Box(low=0, high=1),                # Health ratio
         "resources": spaces.Box(low=0, high=1),             # Resource ratio
         "nearby_agents": spaces.Box(low=0, high=1, shape=(num_features,)),
         "nearby_resources": spaces.Box(low=0, high=1, shape=(num_features,)),
         "age": spaces.Box(low=0, high=1),                   # Normalized age
         "is_defending": spaces.Discrete(2),                 # Boolean
     })
     ```
   - **Action Space**: MultiDiscrete for combined actions
     ```python
     action_space = spaces.MultiDiscrete([
         4,  # Move: 0=up, 1=down, 2=left, 3=right
         2,  # Attack: 0=no, 1=yes
         2,  # Gather: 0=no, 1=yes
         2,  # Share: 0=no, 1=yes
         2,  # Reproduce: 0=no, 1=yes
     ])
     ```

3. **Enhance State Classes**:
   - Add `to_pettingzoo_observation()` method to `AgentState` in `farm/core/state.py`
   - Create helper methods for nearby agent/resource features using existing KD-trees

### Phase 2: Core Environment Refactor

**Target**: Transform `farm/core/environment.py` to inherit from `pettingzoo.ParallelEnv`

1. **Update Class Inheritance**:
   ```python
   from pettingzoo import ParallelEnv
   
   class Environment(ParallelEnv):
       def __init__(self, width, height, resource_distribution, **kwargs):
           super().__init__()
           # ... existing initialization ...
           
           # PettingZoo required attributes
           self.agents = []  # List of active agent IDs
           self.possible_agents = []  # All possible agent IDs
           self.action_spaces = {}  # Per-agent action spaces
           self.observation_spaces = {}  # Per-agent observation spaces
   ```

2. **Implement PettingZoo Required Methods**:

   **`reset()`**:
   ```python
   def reset(self, seed=None, options=None):
       # Reset environment state
       self.time = 0
       self.agents = []
       self.resources = []
       
       # Initialize resources and agents (move from __init__)
       self.initialize_resources(self.resource_distribution)
       self._initialize_agents()
       
       # Update PettingZoo agent lists
       self.agents = [agent.agent_id for agent in self.agents if agent.alive]
       self.possible_agents = self.agents.copy()
       
       # Return observations and infos for all agents
       observations = {}
       infos = {}
       for agent_id in self.agents:
           agent = self.get_agent(agent_id)
           observations[agent_id] = agent.get_pettingzoo_observation()
           infos[agent_id] = {"agent_type": agent.__class__.__name__}
       
       return observations, infos
   ```

       **`step(actions)`**:
     ```python
     def step(self, actions):
         # actions = {"agent_1": [2, 1, 0, 0, 0], "agent_2": [0, 0, 1, 0, 0], ...}
         
         # Execute actions for all agents
         for agent_id, action in actions.items():
             if agent_id in self.agents:
                 agent = self.get_agent(agent_id)
                 agent.act(external_action=action)  # Use modified act() method
         
         # Update environment (resources, metrics, etc.)
         self._update_resources()
         self._update_metrics()
         
         # Note: Births and deaths are handled automatically through 
         # the updated add_agent() and remove_agent() methods
         
         # Prepare return values
         observations = {}
         rewards = {}
         terminations = {}
         truncations = {}
         infos = {}
         
         for agent_id in self.agents:
             agent = self.get_agent(agent_id)
             observations[agent_id] = agent.get_pettingzoo_observation()
             rewards[agent_id] = agent.calculate_step_reward()
             terminations[agent_id] = not agent.alive
             truncations[agent_id] = self.time >= self.max_steps
             infos[agent_id] = {
                 "agent_type": agent.__class__.__name__,
                 "age": self.time - agent.birth_time,
                 "generation": agent.generation,
             }
         
         return observations, rewards, terminations, truncations, infos
     ```

   **`render()`**:
   ```python
   def render(self, mode="human"):
       if mode == "human":
           # Use existing visualization logic
           return self._render_human()
       elif mode == "rgb_array":
           # Return numpy array for automated processing
           return self._render_rgb_array()
   ```

3. **Update Existing Agent Management Methods**:
   ```python
   def add_agent(self, agent):
       """Add an agent to the environment with PettingZoo tracking.
       
       This method handles both the existing database logging and 
       PettingZoo agent tracking for dynamic population management.
       """
       # Existing database logging logic
       agent_data = [
           {
               "simulation_id": self.simulation_id,
               "agent_id": agent.agent_id,
               "birth_time": self.time,
               "agent_type": agent.__class__.__name__,
               "position": agent.position,
               "initial_resources": agent.resource_level,
               "starting_health": agent.starting_health,
               "starvation_threshold": agent.starvation_threshold,
               "genome_id": getattr(agent, "genome_id", None),
               "generation": getattr(agent, "generation", 0),
               "action_weights": agent.get_action_weights(),
           }
       ]

       # Add to environment (existing logic)
       self.agents.append(agent)
       if self.time == 0:
           self.initial_agent_count += 1

       # Batch log to database using SQLAlchemy (existing logic)
       if self.db is not None:
           self.db.logger.log_agents_batch(agent_data)
       
       # NEW: PettingZoo tracking
       if hasattr(self, 'agents') and isinstance(self.agents, list):
           # Add to PettingZoo agent list
           self.agents.append(agent.agent_id)
           self.possible_agents.append(agent.agent_id)
           # Initialize spaces for new agent
           self.action_spaces[agent.agent_id] = self._get_action_space()
           self.observation_spaces[agent.agent_id] = self._get_observation_space()

   def remove_agent(self, agent):
       """Remove an agent from the environment with PettingZoo tracking.
       
       This method handles both the existing death recording and 
       PettingZoo agent tracking removal.
       """
       # Existing death recording logic
       self.record_death()
       
       # Remove from environment list (existing logic)
       self.agents.remove(agent)
       
       # NEW: PettingZoo tracking removal
       if hasattr(self, 'agents') and isinstance(self.agents, list):
           # Remove from PettingZoo agent list
           if agent.agent_id in self.agents:
               self.agents.remove(agent.agent_id)
   ```

### Phase 3: Agent Refactoring

**Target**: Make agents compatible with external action provision while preserving their models

1. **Update BaseAgent** (`farm/agents/base_agent.py`):

   **Modify Existing `act()` Method**:
   ```python
   def act(self, external_action=None):
       """Execute action - either external (PettingZoo) or internal (autonomous).
       
       Args:
           external_action: Optional action array from PettingZoo [move_dir, attack, gather, share, reproduce]
                          If None, uses internal decision-making (autonomous mode)
       """
       if external_action is not None:
           # PettingZoo mode: execute external action
           move_dir, attack, gather, share, reproduce = external_action
           
           # Execute actions in priority order using existing action modules
           if move_dir < 4:  # Valid move action
               self.move_module.execute_move(move_dir)
           if attack:
               self.attack_module.execute_attack()
           if gather:
               self.gather_module.execute_gather()
           if share:
               self.share_module.execute_share()
           if reproduce:
               self.reproduce_module.execute_reproduce()
       else:
           # Autonomous mode: use internal decision-making (existing logic)
           action = self.decide_action()
           action.execute(self)
       
       # Always update models and store experience (existing logic)
       self.train_all_modules()
   ```

   **Add Observation Method**:
   ```python
   def get_pettingzoo_observation(self):
       """Get observation compatible with PettingZoo interface."""
       # Get base agent state
       base_state = self.get_state()
       
       # Get nearby features using existing KD-trees
       nearby_agents = self.environment.get_nearby_agents(self.position, 30)
       nearby_resources = self.environment.get_nearby_resources(self.position, 30)
       
       return {
           "position": [self.position[0] / self.environment.width, 
                       self.position[1] / self.environment.height],
           "health": self.current_health / self.starting_health,
           "resources": self.resource_level / self.config.max_resource_amount,
           "nearby_agents": self._encode_nearby_agents(nearby_agents),
           "nearby_resources": self._encode_nearby_resources(nearby_resources),
           "age": min((self.environment.time - self.birth_time) / 1000, 1.0),
           "is_defending": int(self.is_defending),
       }
   ```

2. **Refactor Action Modules** (`farm/actions/*`):

   **Add Execution Methods**:
   ```python
   # In MoveModule
   def execute_move(self, direction):
       """Execute specific move direction."""
       # Use existing movement logic but with fixed direction
       # Remove epsilon-greedy selection
   
   # In AttackModule  
   def execute_attack(self, direction):
       """Execute attack in specific direction."""
       # Use existing attack logic but with fixed direction
   ```

### Phase 4: Configuration and Integration

1. **Update SimulationConfig** (`farm/core/config.py`):
   ```python
   # Add PettingZoo configuration
   use_pettingzoo: bool = True
   max_steps: int = 1000
   enable_autonomous_mode: bool = False  # Use internal agent models
   enable_rl_mode: bool = True  # Use external policies
   
   # Observation parameters
   observation_radius: float = 30.0
   max_nearby_agents: int = 10
   max_nearby_resources: int = 10
   
   # Action space parameters
   enable_reproduction: bool = True
   enable_sharing: bool = True
   ```

2. **Update Simulation Runners** (`farm/core/simulation.py`):
   ```python
   def run_simulation(num_steps, config, **kwargs):
       # Create PettingZoo environment
       env = Environment(
           width=config.width,
           height=config.height,
           resource_distribution={"type": "random", "amount": config.initial_resources},
           config=config,
           **kwargs
       )
       
       # Initialize environment
       observations, infos = env.reset()
       
       # Main simulation loop
       for step in range(num_steps):
           if config.enable_rl_mode:
               # RL mode: external policies provide actions
               actions = {}
               for agent_id in env.agents:
                   # Get agent's policy (could be trained model)
                   agent = env.get_agent(agent_id)
                   action = agent.get_policy_action(observations[agent_id])
                   actions[agent_id] = action
           else:
               # Autonomous mode: agents decide internally
               actions = {aid: None for aid in env.agents}
           
           # Step environment
           observations, rewards, terminations, truncations, infos = env.step(actions)
           
           # Check if simulation should end
           if len(env.agents) == 0:
               break
       
       return env
   ```

### Phase 5: Training Integration

1. **Create Training Wrapper**:
   ```python
   class AgentFarmWrapper:
       """Wrapper for training individual agent policies."""
       
       def __init__(self, env, agent_id):
           self.env = env
           self.agent_id = agent_id
           
       def step(self, action):
           # Create actions dict with only this agent's action
           actions = {self.agent_id: action}
           
           # Step environment
           obs, rewards, terminations, truncations, infos = self.env.step(actions)
           
           # Return only this agent's data
           return (obs[self.agent_id], rewards[self.agent_id], 
                   terminations[self.agent_id], truncations[self.agent_id], 
                   infos[self.agent_id])
   ```

2. **Training Loop Example**:
   ```python
   # Train individual agent policies
   env = Environment(...)
   observations, infos = env.reset()
   
   # Create policy for each agent
   policies = {}
   for agent_id in env.agents:
       policies[agent_id] = StableBaselines3("PPO", "MlpPolicy", env)
   
   # Training loop
   for episode in range(num_episodes):
       observations, infos = env.reset()
       
       for step in range(max_steps):
           actions = {}
           for agent_id in env.agents:
               action, _ = policies[agent_id].predict(observations[agent_id])
               actions[agent_id] = action
           
           observations, rewards, terminations, truncations, infos = env.step(actions)
           
           # Update policies
           for agent_id in env.agents:
               policies[agent_id].store_transition(
                   observations[agent_id], actions[agent_id], rewards[agent_id]
               )
   ```

### Phase 6: Testing and Validation

1. **Unit Tests**:
   - Test PettingZoo interface compliance
   - Verify observation/action space consistency
   - Test dynamic population handling (births/deaths)

2. **Integration Tests**:
   - Test with Stable-Baselines3 training
   - Verify agent models still work in autonomous mode
   - Test database logging integration

3. **Performance Tests**:
   - Benchmark step time with PettingZoo interface
   - Compare with original simulation performance

## Benefits of This Approach

1. **Natural Multi-Agent Fit**: PettingZoo is designed exactly for your use case
2. **Preserves Agent Models**: Each agent keeps its DQN modules and can learn independently
3. **Flexible Training**: Can train individual agents or use external policies
4. **Standard Interface**: Compatible with all PettingZoo-based RL libraries
5. **Dynamic Population**: Handles agent births/deaths naturally through agent-controlled reproduction
6. **Backward Compatibility**: Can still run autonomous simulations
7. **Agent Autonomy**: Maintains sophisticated agent-level reproduction logic with DQN-based decision making
8. **Scalable Architecture**: Different agent types can have different reproduction strategies without environment changes

## Estimated Timeline

- **Phase 1-2**: 2-3 days (core PettingZoo integration)
- **Phase 3**: 1-2 days (agent refactoring)
- **Phase 4-5**: 1-2 days (configuration and training)
- **Phase 6**: 1 day (testing)

**Total**: 5-8 days for complete PettingZoo integration

This approach leverages PettingZoo's strengths while preserving your sophisticated agent architecture and enabling modern multi-agent reinforcement learning workflows.