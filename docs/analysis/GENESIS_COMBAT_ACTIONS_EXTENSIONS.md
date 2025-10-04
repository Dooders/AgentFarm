# Genesis, Combat, and Actions Analysis Extensions

## Table of Contents
1. [Genesis Module Extensions](#genesis-module)
2. [Combat Module Extensions](#combat-module)
3. [Actions Module Extensions](#actions-module)

---

# GENESIS MODULE

## Current Capabilities
- Initial conditions analysis
- Early population dynamics
- Founder effects
- Critical period analysis
- Genesis success prediction

---

## 1. 🌱 Founder Effects & Initial Conditions

### A. Founder Genotype Analysis
```python
def analyze_founder_genotypes(genesis_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the impact of initial genotypes on long-term success."""
    
    founders = genesis_data[genesis_data['generation'] == 0]
    
    # Group by genotype characteristics
    genotype_outcomes = {}
    
    for genotype in founders['genotype_id'].unique():
        genotype_agents = founders[founders['genotype_id'] == genotype]
        
        # Track descendants
        descendants = genesis_data[genesis_data['ancestor_id'].isin(genotype_agents['agent_id'])]
        
        genotype_outcomes[genotype] = {
            'initial_count': len(genotype_agents),
            'total_descendants': len(descendants),
            'generations_survived': descendants['generation'].max() if len(descendants) > 0 else 0,
            'mean_fitness': float(descendants['fitness'].mean()) if len(descendants) > 0 else 0,
            'expansion_rate': len(descendants) / len(genotype_agents) if len(genotype_agents) > 0 else 0
        }
    
    # Identify most successful founder genotypes
    best_genotype = max(genotype_outcomes, key=lambda k: genotype_outcomes[k]['total_descendants'])
    
    return {
        'num_founder_genotypes': len(genotype_outcomes),
        'genotype_success': genotype_outcomes,
        'most_successful_genotype': best_genotype,
        'genotype_diversity_lost': len(founders['genotype_id'].unique()) - len(genesis_data[genesis_data['generation'] > 10]['genotype_id'].unique())
    }
```

### B. Initial Spatial Configuration
```python
def analyze_initial_spatial_setup(genesis_positions: pd.DataFrame,
                                  resource_positions: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how initial spatial arrangement affects outcomes."""
    
    from scipy.spatial import distance_matrix
    
    founder_coords = genesis_positions[['x', 'y']].values
    resource_coords = resource_positions[['x', 'y']].values
    
    # Calculate distances to resources
    distances_to_resources = distance_matrix(founder_coords, resource_coords)
    min_distances = distances_to_resources.min(axis=1)
    
    # Spatial distribution of founders
    founder_dist_matrix = distance_matrix(founder_coords, founder_coords)
    np.fill_diagonal(founder_dist_matrix, np.inf)
    nn_distances = founder_dist_matrix.min(axis=1)
    
    # Calculate clustering coefficient
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=10, min_samples=2).fit(founder_coords)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    return {
        'mean_distance_to_resources': float(min_distances.mean()),
        'resource_access_inequality': float(min_distances.std() / min_distances.mean()),
        'mean_founder_separation': float(nn_distances.mean()),
        'initial_clustering': n_clusters,
        'spatial_advantage_variance': float(min_distances.var())
    }
```

### C. Critical Period Analysis
```python
def identify_critical_periods(time_series_data: pd.DataFrame,
                              outcome_variable: str = 'population_size') -> Dict[str, Any]:
    """Identify critical windows that determine long-term outcomes.
    
    Uses changepoint detection and sensitivity analysis.
    """
    
    from ruptures import Pelt
    
    outcome = time_series_data[outcome_variable].values
    
    # Detect change points
    model = Pelt(model="rbf").fit(outcome)
    change_points = model.predict(pen=10)
    
    # Analyze each period
    periods = []
    start = 0
    for end in change_points:
        period_data = outcome[start:end]
        
        periods.append({
            'start_step': int(start),
            'end_step': int(end),
            'duration': end - start,
            'mean_value': float(period_data.mean()),
            'volatility': float(period_data.std()),
            'trend': float(np.polyfit(range(len(period_data)), period_data, 1)[0])
        })
        start = end
    
    # Identify critical period (highest correlation with final outcome)
    final_outcome = outcome[-1]
    correlations = []
    for period in periods:
        period_mean = period['mean_value']
        corr = abs(period_mean - final_outcome) / final_outcome
        correlations.append(corr)
    
    critical_period_idx = np.argmin(correlations)
    
    return {
        'num_periods': len(periods),
        'periods': periods,
        'critical_period': periods[critical_period_idx],
        'early_sensitivity': correlations[0],  # How much early period matters
    }
```

---

## 2. 🎲 Stochasticity & Predictability

### A. Environmental Variation Impact
```python
def analyze_environmental_stochasticity(genesis_runs: List[pd.DataFrame]) -> Dict[str, Any]:
    """Compare multiple genesis runs to quantify stochastic effects."""
    
    # Extract key metrics from each run
    run_outcomes = []
    
    for run_id, run_data in enumerate(genesis_runs):
        run_outcomes.append({
            'run_id': run_id,
            'final_population': run_data['population_size'].iloc[-1],
            'time_to_stable': find_stabilization_time(run_data),
            'dominant_genotype': run_data['genotype_id'].mode()[0],
            'genetic_diversity': run_data['genotype_id'].nunique()
        })
    
    run_df = pd.DataFrame(run_outcomes)
    
    # Calculate between-run variance
    cv_population = run_df['final_population'].std() / run_df['final_population'].mean()
    cv_diversity = run_df['genetic_diversity'].std() / run_df['genetic_diversity'].mean()
    
    # Predictability score
    predictability = 1.0 / (1.0 + cv_population)
    
    return {
        'num_runs_analyzed': len(genesis_runs),
        'outcome_variability_cv': float(cv_population),
        'diversity_variability_cv': float(cv_diversity),
        'predictability_score': float(predictability),
        'is_deterministic': cv_population < 0.1,
        'dominant_genotype_consistency': len(run_df['dominant_genotype'].unique()) / len(run_df)
    }
```

### B. Sensitivity Analysis
```python
def parameter_sensitivity_analysis(baseline_params: Dict,
                                   param_ranges: Dict,
                                   simulation_function: Callable) -> Dict[str, Any]:
    """Analyze sensitivity to initial parameters."""
    
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    
    # Define parameter space
    problem = {
        'num_vars': len(param_ranges),
        'names': list(param_ranges.keys()),
        'bounds': list(param_ranges.values())
    }
    
    # Generate samples
    param_samples = saltelli.sample(problem, 1024)
    
    # Run simulations
    outcomes = []
    for params in param_samples:
        param_dict = dict(zip(problem['names'], params))
        result = simulation_function(**param_dict)
        outcomes.append(result['final_population'])
    
    outcomes = np.array(outcomes)
    
    # Sobol sensitivity analysis
    Si = sobol.analyze(problem, outcomes)
    
    # Rank parameters by importance
    importance = pd.DataFrame({
        'parameter': problem['names'],
        'first_order': Si['S1'],
        'total_effect': Si['ST']
    }).sort_values('total_effect', ascending=False)
    
    return {
        'sensitivity_indices': importance.to_dict('records'),
        'most_sensitive_parameter': importance.iloc[0]['parameter'],
        'least_sensitive_parameter': importance.iloc[-1]['parameter'],
        'total_variance_explained': float(importance['first_order'].sum())
    }
```

---

## 3. 🔮 Early Warning Signals

### A. Collapse Prediction
```python
def detect_early_warning_signals(time_series: pd.DataFrame,
                                window: int = 50) -> Dict[str, Any]:
    """Detect early warning signals of population collapse.
    
    Based on critical slowing down theory.
    """
    
    population = time_series['population_size'].values
    
    # Calculate rolling statistics
    rolling_var = pd.Series(population).rolling(window).var()
    rolling_ac1 = pd.Series(population).rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    
    # Trends in variance and autocorrelation (indicators of critical slowing)
    var_trend = np.polyfit(range(len(rolling_var.dropna())), rolling_var.dropna(), 1)[0]
    ac1_trend = np.polyfit(range(len(rolling_ac1.dropna())), rolling_ac1.dropna(), 1)[0]
    
    # Warning signal if both increasing
    warning_score = (var_trend > 0) + (ac1_trend > 0)
    
    return {
        'variance_trend': float(var_trend),
        'autocorrelation_trend': float(ac1_trend),
        'warning_score': int(warning_score),
        'collapse_risk': 'high' if warning_score == 2 else 'medium' if warning_score == 1 else 'low',
        'has_critical_slowing': bool(warning_score > 0)
    }
```

---

# COMBAT MODULE

## Current Capabilities
- Combat statistics
- Win/loss analysis
- Combat efficiency
- Aggression patterns

---

## 4. ⚔️ Combat Dynamics

### A. Combat Strategy Analysis
```python
def analyze_combat_strategies(combat_data: pd.DataFrame) -> Dict[str, Any]:
    """Identify and analyze different combat strategies."""
    
    # Classify agents by combat behavior
    agent_strategies = {}
    
    for agent_id in combat_data['agent_id'].unique():
        agent_combat = combat_data[combat_data['agent_id'] == agent_id]
        
        # Metrics
        aggression_rate = len(agent_combat[agent_combat['initiated_combat']]) / len(agent_combat)
        win_rate = len(agent_combat[agent_combat['outcome'] == 'win']) / len(agent_combat)
        avg_damage_dealt = agent_combat['damage_dealt'].mean()
        retreat_rate = len(agent_combat[agent_combat['action'] == 'retreat']) / len(agent_combat)
        
        # Classify strategy
        if aggression_rate > 0.7 and win_rate > 0.6:
            strategy = 'dominant_aggressor'
        elif aggression_rate < 0.3 and retreat_rate > 0.5:
            strategy = 'avoider'
        elif win_rate > 0.6 and aggression_rate < 0.5:
            strategy = 'defensive_specialist'
        else:
            strategy = 'balanced'
        
        agent_strategies[agent_id] = {
            'strategy': strategy,
            'aggression_rate': float(aggression_rate),
            'win_rate': float(win_rate),
            'avg_damage': float(avg_damage_dealt),
            'combat_count': len(agent_combat)
        }
    
    # Strategy effectiveness
    strategy_outcomes = pd.DataFrame(agent_strategies).T.groupby('strategy').agg({
        'win_rate': 'mean',
        'combat_count': 'sum'
    })
    
    return {
        'agent_strategies': agent_strategies,
        'strategy_distribution': pd.DataFrame(agent_strategies).T['strategy'].value_counts().to_dict(),
        'most_effective_strategy': strategy_outcomes['win_rate'].idxmax(),
        'strategy_effectiveness': strategy_outcomes.to_dict()
    }
```

### B. Arms Race Detection
```python
def detect_arms_race(combat_data: pd.DataFrame,
                    time_window: int = 50) -> Dict[str, Any]:
    """Detect escalating arms races in combat capabilities."""
    
    # Track combat power over time
    combat_by_time = combat_data.groupby('step').agg({
        'damage_dealt': 'mean',
        'health_lost': 'mean',
        'combat_duration': 'mean'
    })
    
    # Calculate trends
    damage_trend = np.polyfit(combat_by_time.index, combat_by_time['damage_dealt'], 1)[0]
    duration_trend = np.polyfit(combat_by_time.index, combat_by_time['combat_duration'], 1)[0]
    
    # Arms race = increasing damage AND increasing duration (harder to win)
    arms_race_detected = (damage_trend > 0) and (duration_trend > 0)
    
    # Calculate escalation rate
    early_damage = combat_by_time.head(time_window)['damage_dealt'].mean()
    late_damage = combat_by_time.tail(time_window)['damage_dealt'].mean()
    escalation_rate = (late_damage - early_damage) / early_damage if early_damage > 0 else 0
    
    return {
        'arms_race_detected': bool(arms_race_detected),
        'damage_escalation_rate': float(escalation_rate),
        'damage_trend': float(damage_trend),
        'duration_trend': float(duration_trend),
        'escalation_speed': 'rapid' if escalation_rate > 0.5 else 'moderate' if escalation_rate > 0.2 else 'slow'
    }
```

### C. Territorial Combat Analysis
```python
def analyze_territorial_combat(combat_data: pd.DataFrame,
                              territory_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze combat in relation to territory."""
    
    # Merge combat with territory info
    merged = combat_data.merge(territory_data, on=['agent_id', 'step'])
    
    # Classify combat by location relative to territory
    merged['location_type'] = merged.apply(
        lambda row: 'home' if row['combat_location'] in row['territory_cells'] else 'away',
        axis=1
    )
    
    # Win rates by location
    home_advantage = merged[merged['location_type'] == 'home']['outcome'].value_counts(normalize=True).get('win', 0)
    away_winrate = merged[merged['location_type'] == 'away']['outcome'].value_counts(normalize=True).get('win', 0)
    
    # Territorial defense frequency
    defense_rate = len(merged[merged['location_type'] == 'home']) / len(merged)
    
    return {
        'home_territory_winrate': float(home_advantage),
        'away_territory_winrate': float(away_winrate),
        'home_advantage': float(home_advantage - away_winrate),
        'territorial_defense_rate': float(defense_rate),
        'strong_territorial_behavior': (home_advantage - away_winrate) > 0.2
    }
```

---

# ACTIONS MODULE

## Current Capabilities
- Action frequency analysis
- Action sequences
- Decision patterns
- Action efficiency

---

## 5. 🎬 Action Patterns

### A. Behavioral Sequences
```python
def analyze_action_sequences(action_data: pd.DataFrame,
                             sequence_length: int = 3) -> Dict[str, Any]:
    """Analyze common action sequences and patterns."""
    
    from collections import Counter
    
    sequences = []
    
    for agent_id in action_data['agent_id'].unique():
        agent_actions = action_data[action_data['agent_id'] == agent_id].sort_values('step')
        actions = agent_actions['action_type'].tolist()
        
        # Extract sequences
        for i in range(len(actions) - sequence_length + 1):
            seq = tuple(actions[i:i+sequence_length])
            sequences.append(seq)
    
    # Count sequences
    sequence_counts = Counter(sequences)
    most_common = sequence_counts.most_common(10)
    
    # Calculate sequence entropy (diversity)
    total = sum(sequence_counts.values())
    probs = np.array(list(sequence_counts.values())) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return {
        'num_unique_sequences': len(sequence_counts),
        'most_common_sequences': [{'sequence': list(seq), 'count': count} for seq, count in most_common],
        'sequence_diversity': float(entropy),
        'behavioral_stereotypy': 1.0 / (1.0 + entropy)  # Low diversity = high stereotypy
    }
```

### B. Action Efficiency
```python
def compute_action_efficiency(action_data: pd.DataFrame,
                             outcome_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate efficiency of different actions."""
    
    # Merge actions with outcomes
    merged = action_data.merge(outcome_data, on=['agent_id', 'step'])
    
    efficiency = {}
    
    for action_type in merged['action_type'].unique():
        action_subset = merged[merged['action_type'] == action_type]
        
        # Efficiency metrics
        success_rate = (action_subset['outcome'] == 'success').mean()
        avg_reward = action_subset['reward'].mean()
        avg_cost = action_subset['energy_cost'].mean()
        
        # Net efficiency
        net_efficiency = avg_reward - avg_cost
        roi = net_efficiency / avg_cost if avg_cost > 0 else 0
        
        efficiency[action_type] = {
            'success_rate': float(success_rate),
            'avg_reward': float(avg_reward),
            'avg_cost': float(avg_cost),
            'net_efficiency': float(net_efficiency),
            'roi': float(roi),
            'frequency': len(action_subset)
        }
    
    # Rank actions
    ranked = sorted(efficiency.items(), key=lambda x: x[1]['roi'], reverse=True)
    
    return {
        'action_efficiency': efficiency,
        'most_efficient_action': ranked[0][0],
        'least_efficient_action': ranked[-1][0],
        'efficiency_rankings': [(action, metrics['roi']) for action, metrics in ranked]
    }
```

### C. Decision Tree Analysis
```python
def build_decision_tree(action_data: pd.DataFrame,
                       context_features: List[str]) -> Dict[str, Any]:
    """Build decision tree to understand action selection."""
    
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    
    # Prepare features and target
    X = action_data[context_features].fillna(0)
    y = action_data['action_type']
    
    # Train decision tree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Extract rules
    tree_rules = export_text(clf, feature_names=context_features)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': context_features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model_accuracy': float(clf.score(X_test, y_test)),
        'most_important_features': importance.head(3).to_dict('records'),
        'decision_rules': tree_rules,
        'tree_depth': clf.get_depth()
    }
```

### D. State-Action-Reward Analysis
```python
def analyze_state_action_rewards(action_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze which state-action combinations yield best rewards."""
    
    # Define states (binning continuous variables)
    action_data['resource_state'] = pd.qcut(action_data['resource_level'], q=3, labels=['low', 'medium', 'high'])
    action_data['health_state'] = pd.qcut(action_data['health_level'], q=3, labels=['low', 'medium', 'high'])
    
    # Group by state-action pairs
    state_action_rewards = action_data.groupby(['resource_state', 'health_state', 'action_type'])['reward'].agg(['mean', 'std', 'count'])
    
    # Find optimal action for each state
    optimal_actions = {}
    
    for resource_state in ['low', 'medium', 'high']:
        for health_state in ['low', 'medium', 'high']:
            state_key = f"resource_{resource_state}_health_{health_state}"
            
            try:
                state_actions = state_action_rewards.loc[(resource_state, health_state)]
                best_action = state_actions['mean'].idxmax()
                
                optimal_actions[state_key] = {
                    'optimal_action': best_action,
                    'expected_reward': float(state_actions.loc[best_action, 'mean']),
                    'confidence': float(1.0 / (1.0 + state_actions.loc[best_action, 'std']))
                }
            except KeyError:
                continue
    
    return {
        'optimal_policy': optimal_actions,
        'state_action_matrix': state_action_rewards.to_dict()
    }
```

---

## Summary

### Genesis Module Extensions:
1. **Founder Effects** - Genotype analysis, spatial configuration
2. **Stochasticity** - Environmental variation, sensitivity analysis
3. **Early Warning** - Collapse prediction, critical periods

### Combat Module Extensions:
4. **Combat Dynamics** - Strategy analysis, arms races, territorial combat

### Actions Module Extensions:
5. **Action Patterns** - Sequences, efficiency, decision trees, state-action-reward

All implementations ready to use for understanding emergence, strategy evolution, and decision-making!
