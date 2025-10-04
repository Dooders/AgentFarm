# Agent, Learning, and Resource Analysis Extensions

## Table of Contents
1. [Agents Module Extensions](#agents-module)
2. [Learning Module Extensions](#learning-module)
3. [Resources Module Extensions](#resources-module)

---

# AGENTS MODULE

## Current Capabilities
- Lifespan analysis
- Behavior clustering
- Performance metrics
- Learning curves per agent
- Agent statistics

---

## 1. 🧬 Evolutionary Analysis

### A. Lineage Tracking
```python
def trace_agent_lineages(agents_df: pd.DataFrame) -> Dict[str, Any]:
    """Track evolutionary lineages and family trees."""
    
    # Build family tree
    lineages = {}
    for _, agent in agents_df.iterrows():
        if 'parent_id' in agent:
            parent = agent['parent_id']
            if parent not in lineages:
                lineages[parent] = []
            lineages[parent].append(agent['agent_id'])
    
    # Analyze lineage success
    lineage_stats = {}
    for founder in [a for a in agents_df['agent_id'] if a not in sum(lineages.values(), [])]:
        descendants = get_all_descendants(founder, lineages)
        
        lineage_stats[founder] = {
            'num_descendants': len(descendants),
            'generations': calculate_max_depth(founder, lineages),
            'mean_lifespan': agents_df[agents_df['agent_id'].isin(descendants)]['lifespan'].mean(),
            'lineage_fitness': len(descendants) / agents_df['generation'].max()
        }
    
    return {
        'num_lineages': len(lineage_stats),
        'most_successful_lineage': max(lineage_stats, key=lambda k: lineage_stats[k]['num_descendants']),
        'lineage_details': lineage_stats,
        'extinction_rate': sum(1 for l in lineage_stats.values() if l['num_descendants'] == 0) / len(lineage_stats)
    }
```

### B. Trait Evolution
```python
def analyze_trait_evolution(agents_df: pd.DataFrame, 
                           traits: List[str]) -> Dict[str, Any]:
    """Analyze how traits evolve over generations."""
    
    from scipy import stats
    
    results = {}
    
    for trait in traits:
        if trait not in agents_df.columns:
            continue
        
        # Trait values by generation
        by_generation = agents_df.groupby('generation')[trait].agg(['mean', 'std', 'min', 'max'])
        
        # Calculate heritability (parent-offspring correlation)
        if 'parent_id' in agents_df.columns:
            parent_child_pairs = agents_df.merge(
                agents_df[['agent_id', trait]], 
                left_on='parent_id', 
                right_on='agent_id',
                suffixes=('_child', '_parent')
            )
            heritability = parent_child_pairs[[f'{trait}_parent', f'{trait}_child']].corr().iloc[0, 1]
        else:
            heritability = None
        
        # Selection gradient
        correlation_fitness = stats.spearmanr(agents_df[trait], agents_df['fitness_score'])
        
        results[trait] = {
            'initial_mean': float(by_generation['mean'].iloc[0]),
            'final_mean': float(by_generation['mean'].iloc[-1]),
            'total_change': float(by_generation['mean'].iloc[-1] - by_generation['mean'].iloc[0]),
            'heritability': float(heritability) if heritability else None,
            'selection_gradient': float(correlation_fitness.correlation),
            'is_under_selection': abs(correlation_fitness.correlation) > 0.3
        }
    
    return results
```

---

## 2. 🎭 Personality & Behavior

### A. Behavioral Syndromes
```python
def identify_behavioral_syndromes(behavior_data: pd.DataFrame) -> Dict[str, Any]:
    """Identify behavioral syndromes (correlations across contexts).
    
    Behavioral syndrome = consistent individual differences in behavior
    across contexts/time.
    """
    
    from sklearn.decomposition import PCA
    
    # Pivot to get agent x behavior matrix
    agent_behaviors = behavior_data.pivot_table(
        index='agent_id',
        columns='behavior_type',
        values='frequency',
        aggfunc='mean'
    ).fillna(0)
    
    # PCA to find behavioral axes
    pca = PCA(n_components=min(3, agent_behaviors.shape[1]))
    behavior_pcs = pca.fit_transform(agent_behaviors)
    
    # Cluster agents by behavioral profile
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    syndromes = kmeans.fit_predict(behavior_pcs)
    
    # Characterize each syndrome
    syndrome_profiles = {}
    for syndrome_id in range(4):
        agents_in_syndrome = agent_behaviors.index[syndromes == syndrome_id]
        profile = agent_behaviors.loc[agents_in_syndrome].mean()
        
        syndrome_profiles[syndrome_id] = {
            'size': int((syndromes == syndrome_id).sum()),
            'dominant_behaviors': profile.nlargest(3).to_dict(),
            'behavioral_consistency': float(agent_behaviors.loc[agents_in_syndrome].std(axis=1).mean())
        }
    
    return {
        'num_syndromes': len(syndrome_profiles),
        'syndrome_profiles': syndrome_profiles,
        'variance_explained_by_pc1': float(pca.explained_variance_ratio_[0]),
        'behavioral_dimensionality': sum(pca.explained_variance_ratio_ > 0.1)
    }
```

### B. Context-Dependent Behavior
```python
def analyze_context_dependent_behavior(behavior_data: pd.DataFrame,
                                       context_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how behavior changes with context."""
    
    # Merge behavior with context (e.g., resource availability, danger)
    merged = behavior_data.merge(context_data, on=['step', 'agent_id'])
    
    results = {}
    
    for behavior in merged['behavior_type'].unique():
        behavior_subset = merged[merged['behavior_type'] == behavior]
        
        # Analyze by context bins
        context_var = 'resource_availability'  # example
        bins = pd.qcut(behavior_subset[context_var], q=4, labels=['low', 'med_low', 'med_high', 'high'])
        
        by_context = behavior_subset.groupby(bins)['frequency'].mean()
        
        # Calculate plasticity (variance across contexts)
        plasticity = by_context.std() / by_context.mean() if by_context.mean() > 0 else 0
        
        results[behavior] = {
            'mean_by_context': by_context.to_dict(),
            'plasticity_score': float(plasticity),
            'context_sensitivity': 'high' if plasticity > 0.5 else 'low'
        }
    
    return results
```

---

## 3. 🏆 Performance & Success

### A. Success Factor Analysis
```python
def identify_success_factors(agents_df: pd.DataFrame,
                            success_metric: str = 'lifespan') -> Dict[str, Any]:
    """Identify which traits/behaviors predict success."""
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # Features: all numeric columns except success metric
    feature_cols = agents_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove(success_metric)
    
    X = agents_df[feature_cols].fillna(agents_df[feature_cols].mean())
    y = agents_df[success_metric]
    
    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Permutation importance for robustness
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    return {
        'top_success_factors': importances.head(5).to_dict('records'),
        'model_r2_score': float(rf.score(X, y)),
        'key_traits': importances.head(3)['feature'].tolist()
    }
```

### B. Risk-Taking Analysis
```python
def analyze_risk_taking_strategies(agents_df: pd.DataFrame,
                                   action_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze risk-taking behavior and its outcomes."""
    
    # Calculate risk score per agent
    risk_scores = {}
    
    for agent_id in agents_df['agent_id']:
        agent_actions = action_data[action_data['agent_id'] == agent_id]
        
        # Risk indicators
        risky_actions = agent_actions[agent_actions['action_type'].isin(['attack', 'explore_far'])]
        risk_frequency = len(risky_actions) / len(agent_actions) if len(agent_actions) > 0 else 0
        
        # Outcome
        agent_info = agents_df[agents_df['agent_id'] == agent_id].iloc[0]
        
        risk_scores[agent_id] = {
            'risk_frequency': risk_frequency,
            'lifespan': agent_info['lifespan'],
            'final_resources': agent_info.get('final_resources', 0)
        }
    
    risk_df = pd.DataFrame(risk_scores).T
    
    # Correlation between risk and outcomes
    risk_lifespan_corr = risk_df['risk_frequency'].corr(risk_df['lifespan'])
    risk_resources_corr = risk_df['risk_frequency'].corr(risk_df['final_resources'])
    
    # Identify optimal risk level
    risk_bins = pd.qcut(risk_df['risk_frequency'], q=4, labels=['low', 'medium', 'high', 'very_high'])
    outcomes_by_risk = risk_df.groupby(risk_bins).mean()
    
    return {
        'risk_lifespan_correlation': float(risk_lifespan_corr),
        'risk_resources_correlation': float(risk_resources_corr),
        'optimal_risk_level': outcomes_by_risk['lifespan'].idxmax(),
        'outcomes_by_risk': outcomes_by_risk.to_dict()
    }
```

---

# LEARNING MODULE

## Current Capabilities
- Learning performance analysis
- Agent learning curves
- Module performance comparison
- Learning progress tracking

---

## 4. 🎓 Learning Dynamics

### A. Learning Rate Analysis
```python
def analyze_learning_rates(learning_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how quickly agents learn."""
    
    learning_rates = {}
    
    for agent_id in learning_data['agent_id'].unique():
        agent_data = learning_data[learning_data['agent_id'] == agent_id].sort_values('step')
        rewards = agent_data['reward'].values
        
        if len(rewards) < 10:
            continue
        
        # Fit exponential learning curve: r(t) = r_max * (1 - exp(-k*t))
        from scipy.optimize import curve_fit
        
        def exp_learning(t, r_max, k):
            return r_max * (1 - np.exp(-k * t))
        
        try:
            params, _ = curve_fit(exp_learning, range(len(rewards)), rewards, p0=[rewards.max(), 0.1])
            r_max, k = params
            
            learning_rates[agent_id] = {
                'learning_rate_k': float(k),
                'asymptotic_performance': float(r_max),
                'time_to_90_percent': float(np.log(10) / k) if k > 0 else None
            }
        except:
            continue
    
    return {
        'agent_learning_rates': learning_rates,
        'mean_learning_rate': np.mean([v['learning_rate_k'] for v in learning_rates.values()]),
        'fastest_learner': max(learning_rates, key=lambda k: learning_rates[k]['learning_rate_k'])
    }
```

### B. Exploration vs Exploitation
```python
def analyze_exploration_exploitation(action_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze exploration-exploitation trade-off over time."""
    
    # Calculate action entropy as exploration measure
    exploration_over_time = []
    
    for step in sorted(action_data['step'].unique()):
        step_actions = action_data[action_data['step'] == step]
        action_counts = step_actions['action_type'].value_counts()
        
        # Shannon entropy
        probs = action_counts / action_counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        exploration_over_time.append({
            'step': step,
            'entropy': entropy,
            'unique_actions': len(action_counts)
        })
    
    entropy_df = pd.DataFrame(exploration_over_time)
    
    # Detect shift from exploration to exploitation
    early_entropy = entropy_df.head(len(entropy_df)//3)['entropy'].mean()
    late_entropy = entropy_df.tail(len(entropy_df)//3)['entropy'].mean()
    
    return {
        'initial_exploration': float(early_entropy),
        'final_exploration': float(late_entropy),
        'entropy_decrease': float(early_entropy - late_entropy),
        'exploration_phase_length': int((entropy_df['entropy'] > early_entropy * 0.8).sum()),
        'has_clear_transition': (early_entropy - late_entropy) > 0.5
    }
```

### C. Transfer Learning
```python
def analyze_transfer_learning(learning_data: pd.DataFrame,
                              task_changes: List[int]) -> Dict[str, Any]:
    """Analyze ability to transfer learning across tasks."""
    
    transfer_effects = []
    
    for i in range(len(task_changes) - 1):
        start_step = task_changes[i]
        end_step = task_changes[i + 1]
        
        task_data = learning_data[
            (learning_data['step'] >= start_step) & 
            (learning_data['step'] < end_step)
        ]
        
        # Performance at task start vs baseline
        initial_performance = task_data.head(10)['reward'].mean()
        baseline_performance = learning_data.head(10)['reward'].mean()
        
        # Positive transfer if initial > baseline
        transfer_effect = initial_performance - baseline_performance
        
        transfer_effects.append({
            'task_number': i + 1,
            'initial_performance': float(initial_performance),
            'transfer_effect': float(transfer_effect),
            'has_positive_transfer': transfer_effect > 0
        })
    
    return {
        'transfer_effects': transfer_effects,
        'mean_transfer': np.mean([t['transfer_effect'] for t in transfer_effects]),
        'positive_transfer_rate': sum(t['has_positive_transfer'] for t in transfer_effects) / len(transfer_effects)
    }
```

---

## 5. 🧠 Meta-Learning

### A. Learning-to-Learn
```python
def analyze_meta_learning(learning_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze if agents get better at learning over time."""
    
    # Split into episodes/tasks
    episode_length = 100
    episodes = []
    
    for start in range(0, len(learning_data), episode_length):
        episode_data = learning_data.iloc[start:start+episode_length]
        
        if len(episode_data) < 20:
            continue
        
        # Calculate learning speed (reward improvement rate)
        rewards = episode_data['reward'].values
        learning_speed = (rewards[-10:].mean() - rewards[:10].mean()) / len(rewards)
        
        episodes.append({
            'episode': len(episodes),
            'learning_speed': learning_speed,
            'final_performance': rewards[-10:].mean()
        })
    
    episodes_df = pd.DataFrame(episodes)
    
    # Check if learning speed increases over episodes
    if len(episodes_df) > 2:
        meta_learning_trend = np.polyfit(episodes_df['episode'], episodes_df['learning_speed'], 1)[0]
    else:
        meta_learning_trend = 0
    
    return {
        'episodes_analyzed': len(episodes),
        'meta_learning_trend': float(meta_learning_trend),
        'has_meta_learning': meta_learning_trend > 0,
        'early_vs_late_learning_speed': {
            'early': float(episodes_df.head(3)['learning_speed'].mean()),
            'late': float(episodes_df.tail(3)['learning_speed'].mean())
        }
    }
```

---

# RESOURCES MODULE

## Current Capabilities
- Resource distribution patterns
- Consumption analysis
- Efficiency metrics
- Hotspot identification

---

## 6. 🌾 Resource Dynamics

### A. Regeneration Analysis
```python
def analyze_resource_regeneration(resource_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze resource regeneration patterns."""
    
    # Track resource levels over time at locations
    regeneration_rates = []
    
    for location_id in resource_data['location_id'].unique():
        location_data = resource_data[resource_data['location_id'] == location_id].sort_values('step')
        
        # Calculate regeneration (increase after depletion)
        resource_levels = location_data['amount'].values
        
        # Find depletion-regeneration cycles
        depleted = resource_levels < resource_levels.max() * 0.2
        regenerated = resource_levels > resource_levels.max() * 0.8
        
        # Calculate average regeneration time
        depletion_points = np.where(depleted)[0]
        regeneration_points = np.where(regenerated)[0]
        
        if len(depletion_points) > 0 and len(regeneration_points) > 0:
            # Find recovery time
            for dep_point in depletion_points:
                future_regen = regeneration_points[regeneration_points > dep_point]
                if len(future_regen) > 0:
                    recovery_time = future_regen[0] - dep_point
                    regeneration_rates.append(recovery_time)
    
    return {
        'mean_regeneration_time': float(np.mean(regeneration_rates)) if regeneration_rates else None,
        'regeneration_variability': float(np.std(regeneration_rates)) if regeneration_rates else None,
        'num_cycles_observed': len(regeneration_rates)
    }
```

### B. Tragedy of the Commons
```python
def detect_tragedy_of_commons(resource_data: pd.DataFrame,
                              population_data: pd.DataFrame) -> Dict[str, Any]:
    """Detect overexploitation of shared resources."""
    
    # Merge resources with population
    merged = resource_data.merge(population_data, on='step')
    
    # Calculate per-capita resource availability
    merged['per_capita_resources'] = merged['total_resources'] / merged['total_agents']
    
    # Detect tragedy: declining resources despite stable/growing population
    population_trend = np.polyfit(merged['step'], merged['total_agents'], 1)[0]
    resource_trend = np.polyfit(merged['step'], merged['per_capita_resources'], 1)[0]
    
    tragedy_occurring = (population_trend >= 0) and (resource_trend < 0)
    
    # Calculate carrying capacity vs actual population
    max_sustainable = merged['per_capita_resources'].quantile(0.9) * merged['total_agents'].mean()
    current_population = merged['total_agents'].iloc[-1]
    overshoot = (current_population - max_sustainable) / max_sustainable
    
    return {
        'tragedy_detected': bool(tragedy_occurring),
        'population_trend': float(population_trend),
        'resource_depletion_rate': float(resource_trend),
        'population_overshoot': float(overshoot),
        'sustainability_index': float(1.0 / (1.0 + abs(overshoot))) if overshoot else 1.0
    }
```

### C. Resource Patchiness
```python
def analyze_resource_patchiness(resource_positions: pd.DataFrame) -> Dict[str, Any]:
    """Analyze spatial distribution of resources (patchy vs uniform)."""
    
    from scipy.stats import moranI
    from sklearn.metrics import pairwise_distances
    
    coords = resource_positions[['x', 'y']].values
    amounts = resource_positions['amount'].values
    
    # Calculate patchiness using Moran's I
    # (already implemented in spatial extensions, simplified here)
    
    # Alternative: coefficient of variation
    mean_nn_distance = calculate_mean_nearest_neighbor_distance(coords)
    cv_resources = amounts.std() / amounts.mean() if amounts.mean() > 0 else 0
    
    # Lloyd's index of patchiness
    variance_density = amounts.var()
    mean_density = amounts.mean()
    lloyds_index = variance_density / mean_density if mean_density > 0 else 0
    
    return {
        'lloyds_patchiness_index': float(lloyds_index),
        'cv_resource_amounts': float(cv_resources),
        'distribution_type': 'patchy' if lloyds_index > 1 else 'uniform' if lloyds_index < 0.5 else 'random',
        'mean_patch_separation': float(mean_nn_distance)
    }
```

---

## Summary

### Agents Module Extensions:
1. **Evolutionary Analysis** - Lineages, trait evolution
2. **Personality & Behavior** - Behavioral syndromes, context-dependence
3. **Performance Analysis** - Success factors, risk-taking

### Learning Module Extensions:
4. **Learning Dynamics** - Learning rates, exploration-exploitation, transfer
5. **Meta-Learning** - Learning-to-learn analysis

### Resources Module Extensions:
6. **Resource Dynamics** - Regeneration, tragedy of commons, patchiness

All ready to implement with clear research applications!
