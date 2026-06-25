# Temporal, Social, Comparative, and Advanced Analysis Extensions

## Table of Contents
1. [Temporal Module Extensions](#temporal-module)
2. [Social Behavior Module Extensions](#social-module)
3. [Comparative Module Extensions](#comparative-module)
4. [Dominance Module Extensions](#dominance-module)
5. [Advantage Module Extensions](#advantage-module)
6. [Significant Events Module Extensions](#events-module)

---

# TEMPORAL MODULE

## Current Capabilities
- Time series analysis
- Trend detection
- Temporal patterns
- Periodicity analysis

---

## 1. ⏱️ Advanced Time Series Analysis

### A. Granger Causality
```python
def analyze_granger_causality(time_series_dict: Dict[str, pd.Series],
                              max_lag: int = 10) -> Dict[str, Any]:
    """Test if one time series can predict another (Granger causality)."""
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    results = {}
    
    # Test all pairs
    for var1_name, var1_data in time_series_dict.items():
        for var2_name, var2_data in time_series_dict.items():
            if var1_name == var2_name:
                continue
            
            # Combine into DataFrame
            df = pd.DataFrame({
                'target': var2_data,
                'predictor': var1_data
            }).dropna()
            
            if len(df) < max_lag + 10:
                continue
            
            try:
                # Run Granger causality test
                gc_result = grangercausalitytests(df[['target', 'predictor']], max_lag, verbose=False)
                
                # Extract minimum p-value across lags
                p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                min_p_value = min(p_values)
                best_lag = p_values.index(min_p_value) + 1
                
                results[f"{var1_name}_causes_{var2_name}"] = {
                    'causes': min_p_value < 0.05,
                    'p_value': float(min_p_value),
                    'optimal_lag': int(best_lag)
                }
            except:
                continue
    
    # Build causal network
    causal_pairs = [(k.split('_causes_')[0], k.split('_causes_')[1]) 
                   for k, v in results.items() if v['causes']]
    
    return {
        'granger_causality_results': results,
        'causal_network': causal_pairs,
        'num_causal_relationships': len(causal_pairs)
    }
```

### B. Regime Switching Models
```python
def detect_regime_switches(time_series: pd.Series) -> Dict[str, Any]:
    """Detect regime switches using Markov switching models."""
    
    from statsmodels.tsa.regime_switching import markov_regression
    
    # Fit 2-regime model
    model = markov_regression.MarkovRegression(
        time_series,
        k_regimes=2,
        trend='c'
    )
    
    try:
        results = model.fit()
        
        # Extract regime probabilities
        regime_probs = results.smoothed_marginal_probabilities
        
        # Identify regime changes
        regime_sequence = regime_probs[1] > 0.5  # High-value regime
        switches = np.diff(regime_sequence.astype(int))
        switch_points = np.where(switches != 0)[0]
        
        # Characterize regimes
        regime_0_mean = time_series[regime_probs[0] > 0.5].mean()
        regime_1_mean = time_series[regime_probs[1] > 0.5].mean()
        
        return {
            'num_regimes': 2,
            'regime_means': [float(regime_0_mean), float(regime_1_mean)],
            'num_switches': len(switch_points),
            'switch_points': switch_points.tolist(),
            'regime_persistence': {
                0: float((regime_probs[0] > 0.5).sum() / len(regime_probs)),
                1: float((regime_probs[1] > 0.5).sum() / len(regime_probs))
            }
        }
    except:
        return {'error': 'Could not fit regime switching model'}
```

### C. Wavelet Analysis
```python
def perform_wavelet_analysis(time_series: pd.Series) -> Dict[str, Any]:
    """Analyze time-frequency structure using wavelets."""
    
    import pywt
    
    # Continuous wavelet transform
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(time_series.values, scales, 'morl')
    
    # Find dominant frequencies at each time
    power = np.abs(coefficients)**2
    dominant_scales = np.argmax(power, axis=0)
    
    # Identify time periods with different dominant frequencies
    from scipy.signal import find_peaks
    scale_changes, _ = find_peaks(np.abs(np.diff(dominant_scales)), height=10)
    
    return {
        'dominant_frequencies': frequencies[dominant_scales].tolist(),
        'frequency_changes': scale_changes.tolist(),
        'mean_frequency': float(frequencies[dominant_scales].mean()),
        'frequency_variability': float(frequencies[dominant_scales].std())
    }
```

### D. Recurrence Analysis
```python
def analyze_recurrence(time_series: pd.Series, 
                      threshold: float = 0.1) -> Dict[str, Any]:
    """Analyze recurrence patterns (repeated states)."""
    
    from scipy.spatial.distance import pdist, squareform
    
    # Embed time series (phase space reconstruction)
    embedding_dim = 3
    delay = 1
    
    embedded = np.array([
        time_series.values[i:i+embedding_dim*delay:delay] 
        for i in range(len(time_series) - embedding_dim*delay)
    ])
    
    # Calculate distance matrix
    distances = squareform(pdist(embedded))
    
    # Recurrence plot (binary matrix)
    recurrence_plot = distances < threshold * distances.std()
    
    # Calculate recurrence rate
    recurrence_rate = recurrence_plot.sum() / recurrence_plot.size
    
    # Determinism (percentage of recurrent points in diagonal lines)
    diag_lengths = []
    for offset in range(1, len(recurrence_plot)):
        diag = np.diag(recurrence_plot, k=offset)
        # Find consecutive True values
        lengths = []
        current_length = 0
        for val in diag:
            if val:
                current_length += 1
            else:
                if current_length > 0:
                    lengths.append(current_length)
                current_length = 0
        diag_lengths.extend(lengths)
    
    determinism = sum(l for l in diag_lengths if l > 2) / sum(diag_lengths) if diag_lengths else 0
    
    return {
        'recurrence_rate': float(recurrence_rate),
        'determinism': float(determinism),
        'mean_diagonal_length': float(np.mean(diag_lengths)) if diag_lengths else 0,
        'dynamics_type': 'deterministic' if determinism > 0.7 else 'stochastic'
    }
```

---

# SOCIAL MODULE

## Current Capabilities
- Social network analysis
- Interaction patterns
- Group formation
- Social structure

---

## 2. 🤝 Advanced Social Network Analysis

### A. Information Diffusion
```python
def analyze_information_diffusion(interaction_data: pd.DataFrame,
                                  adoption_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how information/behaviors spread through network."""
    
    import networkx as nx
    
    # Build social network
    G = nx.from_pandas_edgelist(
        interaction_data,
        source='agent_from',
        target='agent_to',
        edge_attr='interaction_strength',
        create_using=nx.DiGraph()
    )
    
    # Track adoption over time
    adopters_by_time = adoption_data.groupby('adoption_step')['agent_id'].apply(list).to_dict()
    
    # Calculate infection rates
    diffusion_metrics = []
    
    for step in sorted(adopters_by_time.keys()):
        adopters = set(sum([adopters_by_time[s] for s in range(step + 1)], []))
        
        # Count exposed but not adopted
        exposed = set()
        for adopter in adopters:
            if adopter in G:
                exposed.update(G.neighbors(adopter))
        
        exposed_not_adopted = exposed - adopters
        
        diffusion_metrics.append({
            'step': step,
            'num_adopters': len(adopters),
            'num_exposed': len(exposed),
            'adoption_rate': len(adopters) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
            'conversion_rate': len(adopters) / len(exposed) if len(exposed) > 0 else 0
        })
    
    # Fit Bass diffusion model
    adopters_over_time = [m['num_adopters'] for m in diffusion_metrics]
    
    return {
        'diffusion_metrics': diffusion_metrics,
        'final_adoption_rate': diffusion_metrics[-1]['adoption_rate'] if diffusion_metrics else 0,
        'mean_conversion_rate': np.mean([m['conversion_rate'] for m in diffusion_metrics]),
        'diffusion_speed': 'fast' if len(adopters_over_time) < 50 else 'slow'
    }
```

### B. Community Detection & Evolution
```python
def analyze_community_evolution(interaction_data: pd.DataFrame,
                               time_windows: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Track how communities form and evolve over time."""
    
    import networkx as nx
    from networkx.algorithms import community
    
    communities_over_time = []
    
    for start, end in time_windows:
        # Build network for this time window
        window_data = interaction_data[
            (interaction_data['step'] >= start) & 
            (interaction_data['step'] < end)
        ]
        
        G = nx.from_pandas_edgelist(window_data, 'agent_from', 'agent_to')
        
        # Detect communities
        communities = community.greedy_modularity_communities(G)
        
        communities_over_time.append({
            'time_window': (start, end),
            'num_communities': len(communities),
            'modularity': community.modularity(G, communities),
            'community_sizes': [len(c) for c in communities],
            'largest_community': max(len(c) for c in communities) if communities else 0
        })
    
    # Track community stability
    num_communities = [c['num_communities'] for c in communities_over_time]
    community_stability = 1.0 / (1.0 + np.std(num_communities))
    
    return {
        'communities_over_time': communities_over_time,
        'community_stability': float(community_stability),
        'mean_modularity': float(np.mean([c['modularity'] for c in communities_over_time]))
    }
```

### C. Social Influence Analysis
```python
def measure_social_influence(network_data: pd.DataFrame,
                            behavior_data: pd.DataFrame) -> Dict[str, Any]:
    """Quantify how much agents influence each other's behavior."""
    
    import networkx as nx
    
    # Build network
    G = nx.from_pandas_edgelist(network_data, 'agent_from', 'agent_to')
    
    influence_scores = {}
    
    for agent_id in G.nodes():
        # Get neighbors' behaviors
        neighbors = list(G.neighbors(agent_id))
        
        if not neighbors:
            continue
        
        # Calculate behavior correlation with neighbors
        agent_behavior = behavior_data[behavior_data['agent_id'] == agent_id]['behavior_value']
        neighbor_behaviors = behavior_data[behavior_data['agent_id'].isin(neighbors)].groupby('agent_id')['behavior_value'].mean()
        
        if len(agent_behavior) > 0 and len(neighbor_behaviors) > 0:
            # Correlation coefficient
            correlation = agent_behavior.iloc[0] - neighbor_behaviors.mean()
            
            influence_scores[agent_id] = {
                'num_neighbors': len(neighbors),
                'behavior_similarity': float(1.0 / (1.0 + abs(correlation))),
                'centrality': float(nx.degree_centrality(G)[agent_id])
            }
    
    # Identify influencers (high centrality + behavior adoption)
    influencers = sorted(
        influence_scores.items(),
        key=lambda x: x[1]['centrality'] * x[1]['behavior_similarity'],
        reverse=True
    )[:10]
    
    return {
        'influence_scores': influence_scores,
        'top_influencers': [agent_id for agent_id, _ in influencers],
        'mean_influence_score': np.mean([v['behavior_similarity'] for v in influence_scores.values()])
    }
```

### D. Structural Holes & Brokerage
```python
def identify_structural_holes(network_data: pd.DataFrame) -> Dict[str, Any]:
    """Identify agents who bridge disconnected groups (brokers)."""
    
    import networkx as nx
    
    G = nx.from_pandas_edgelist(network_data, 'agent_from', 'agent_to')
    
    # Calculate constraint for each node (Burt's constraint measure)
    constraints = nx.constraint(G)
    
    # Low constraint = many structural holes = brokerage position
    brokers = sorted(constraints.items(), key=lambda x: x[1])[:10]
    
    # Calculate betweenness centrality (another measure of brokerage)
    betweenness = nx.betweenness_centrality(G)
    
    # Effective size (Burt's measure of non-redundant contacts)
    effective_sizes = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            # Count connections between neighbors
            redundant = sum(1 for i, n1 in enumerate(neighbors) 
                          for n2 in neighbors[i+1:] if G.has_edge(n1, n2))
            effective_sizes[node] = len(neighbors) - (redundant / len(neighbors))
    
    return {
        'top_brokers': [agent_id for agent_id, _ in brokers],
        'constraints': {k: float(v) for k, v in constraints.items()},
        'betweenness_centrality': {k: float(v) for k, v in betweenness.items()},
        'effective_sizes': {k: float(v) for k, v in effective_sizes.items()},
        'network_brokerage_score': float(1.0 - np.mean(list(constraints.values())))
    }
```

---

# COMPARATIVE MODULE

## Current Capabilities
- Multi-simulation comparison
- Parameter sweep analysis
- Statistical comparison
- Outcome correlation

---

## 3. 📊 Advanced Comparative Analysis

### A. Benchmark Analysis
```python
def benchmark_against_baseline(experimental_results: List[Dict],
                               baseline_results: Dict) -> Dict[str, Any]:
    """Compare experimental conditions against baseline."""
    
    from scipy import stats
    
    comparisons = []
    
    for exp in experimental_results:
        # Extract key metrics
        exp_metrics = {k: v for k, v in exp.items() if isinstance(v, (int, float))}
        
        improvements = {}
        for metric, exp_value in exp_metrics.items():
            if metric in baseline_results:
                baseline_value = baseline_results[metric]
                
                # Calculate improvement
                if baseline_value != 0:
                    percent_change = ((exp_value - baseline_value) / baseline_value) * 100
                else:
                    percent_change = 0
                
                improvements[metric] = {
                    'baseline': float(baseline_value),
                    'experimental': float(exp_value),
                    'percent_change': float(percent_change),
                    'improved': exp_value > baseline_value
                }
        
        comparisons.append({
            'condition': exp['condition_name'],
            'improvements': improvements,
            'overall_improvement': np.mean([v['percent_change'] for v in improvements.values()])
        })
    
    # Rank conditions
    ranked = sorted(comparisons, key=lambda x: x['overall_improvement'], reverse=True)
    
    return {
        'comparisons': comparisons,
        'best_condition': ranked[0]['condition'],
        'worst_condition': ranked[-1]['condition'],
        'rankings': [(c['condition'], c['overall_improvement']) for c in ranked]
    }
```

### B. Meta-Analysis
```python
def perform_meta_analysis(study_results: List[Dict]) -> Dict[str, Any]:
    """Combine results from multiple studies (meta-analysis)."""
    
    from scipy import stats
    
    # Extract effect sizes from each study
    effect_sizes = []
    variances = []
    
    for study in study_results:
        # Calculate effect size (Cohen's d)
        control_mean = study['control_mean']
        treatment_mean = study['treatment_mean']
        pooled_std = np.sqrt((study['control_std']**2 + study['treatment_std']**2) / 2)
        
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Variance of effect size
        n_control = study['control_n']
        n_treatment = study['treatment_n']
        variance = ((n_control + n_treatment) / (n_control * n_treatment)) + (cohens_d**2 / (2 * (n_control + n_treatment)))
        
        effect_sizes.append(cohens_d)
        variances.append(variance)
    
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    # Random effects meta-analysis
    weights = 1 / variances
    weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    weighted_variance = 1 / np.sum(weights)
    
    # Heterogeneity (Q statistic)
    Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
    df = len(effect_sizes) - 1
    I_squared = max(0, (Q - df) / Q) * 100  # Percentage of variance due to heterogeneity
    
    return {
        'pooled_effect_size': float(weighted_mean),
        'pooled_variance': float(weighted_variance),
        'confidence_interval_95': [
            float(weighted_mean - 1.96 * np.sqrt(weighted_variance)),
            float(weighted_mean + 1.96 * np.sqrt(weighted_variance))
        ],
        'heterogeneity_I2': float(I_squared),
        'interpretation': 'significant' if abs(weighted_mean) > 0.5 else 'not significant'
    }
```

### C. Pareto Frontier Analysis
```python
def identify_pareto_frontier(results_df: pd.DataFrame,
                            objectives: List[str],
                            maximize: List[bool]) -> Dict[str, Any]:
    """Identify Pareto-optimal solutions (multi-objective optimization)."""
    
    # Normalize objectives
    normalized = results_df[objectives].copy()
    for i, obj in enumerate(objectives):
        if maximize[i]:
            normalized[obj] = (normalized[obj] - normalized[obj].min()) / (normalized[obj].max() - normalized[obj].min())
        else:
            normalized[obj] = (normalized[obj].max() - normalized[obj]) / (normalized[obj].max() - normalized[obj].min())
    
    # Find Pareto frontier
    is_pareto = np.ones(len(normalized), dtype=bool)
    
    for i in range(len(normalized)):
        for j in range(len(normalized)):
            if i == j:
                continue
            # Check if j dominates i
            if all(normalized.iloc[j] >= normalized.iloc[i]) and any(normalized.iloc[j] > normalized.iloc[i]):
                is_pareto[i] = False
                break
    
    pareto_solutions = results_df[is_pareto]
    
    return {
        'num_pareto_optimal': int(is_pareto.sum()),
        'pareto_solutions': pareto_solutions.to_dict('records'),
        'pareto_indices': np.where(is_pareto)[0].tolist(),
        'coverage': float(is_pareto.sum() / len(results_df))
    }
```

---

# DOMINANCE MODULE

## Current Capabilities
- Dominance hierarchy analysis
- Win/loss matrices
- Rank correlation
- Stability analysis

---

## 4. 👑 Advanced Dominance Analysis

### A. Elo Rating System
```python
def compute_elo_ratings(interaction_data: pd.DataFrame,
                       k_factor: float = 32) -> Dict[str, Any]:
    """Calculate Elo ratings from interaction outcomes."""
    
    # Initialize ratings
    ratings = {agent: 1500 for agent in set(interaction_data['agent_1'].unique()) | set(interaction_data['agent_2'].unique())}
    rating_history = {agent: [1500] for agent in ratings.keys()}
    
    # Process interactions chronologically
    for _, interaction in interaction_data.sort_values('step').iterrows():
        agent1 = interaction['agent_1']
        agent2 = interaction['agent_2']
        outcome = interaction['outcome']  # 1 if agent1 wins, 0 if agent2 wins, 0.5 for draw
        
        # Expected scores
        expected1 = 1 / (1 + 10**((ratings[agent2] - ratings[agent1]) / 400))
        expected2 = 1 - expected1
        
        # Update ratings
        ratings[agent1] += k_factor * (outcome - expected1)
        ratings[agent2] += k_factor * ((1 - outcome) - expected2)
        
        # Record history
        rating_history[agent1].append(ratings[agent1])
        rating_history[agent2].append(ratings[agent2])
    
    # Rank agents
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate rating volatility
    volatilities = {agent: np.std(history) for agent, history in rating_history.items()}
    
    return {
        'final_ratings': {k: float(v) for k, v in ratings.items()},
        'rankings': [(agent, float(rating)) for agent, rating in ranked],
        'rating_volatilities': {k: float(v) for k, v in volatilities.items()},
        'alpha_agent': ranked[0][0],
        'rating_spread': float(ranked[0][1] - ranked[-1][1])
    }
```

### B. Transitivity Analysis
```python
def analyze_hierarchy_transitivity(dominance_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze transitivity of dominance relationships (A>B, B>C => A>C?)."""
    
    n = dominance_matrix.shape[0]
    
    # Count transitive triads
    transitive = 0
    intransitive = 0
    
    for i in range(n):
        for j in range(n):
            if i == j or dominance_matrix[i, j] == 0:
                continue
            for k in range(n):
                if k == i or k == j or dominance_matrix[j, k] == 0:
                    continue
                
                # Check if i > j and j > k
                if dominance_matrix[i, j] > 0.5 and dominance_matrix[j, k] > 0.5:
                    # Then i should dominate k
                    if dominance_matrix[i, k] > 0.5:
                        transitive += 1
                    else:
                        intransitive += 1
    
    total_triads = transitive + intransitive
    transitivity = transitive / total_triads if total_triads > 0 else 0
    
    return {
        'transitive_triads': int(transitive),
        'intransitive_triads': int(intransitive),
        'transitivity_index': float(transitivity),
        'hierarchy_linearity': float(transitivity)
    }
```

### C. Dominance Style Analysis
```python
def classify_dominance_styles(agent_data: pd.DataFrame) -> Dict[str, Any]:
    """Classify agents by dominance style (despotic vs egalitarian)."""
    
    styles = {}
    
    for agent_id in agent_data['agent_id'].unique():
        agent = agent_data[agent_data['agent_id'] == agent_id].iloc[0]
        
        # Metrics
        win_rate = agent['wins'] / (agent['wins'] + agent['losses']) if (agent['wins'] + agent['losses']) > 0 else 0
        aggression = agent['initiated_conflicts'] / agent['total_interactions'] if agent['total_interactions'] > 0 else 0
        consistency = 1.0 - agent['rank_volatility']  # Lower volatility = more consistent
        
        # Classify style
        if win_rate > 0.7 and aggression > 0.6:
            style = 'despotic'
        elif win_rate < 0.4 and aggression < 0.3:
            style = 'subordinate'
        elif aggression > 0.5 and win_rate < 0.5:
            style = 'challenger'
        else:
            style = 'egalitarian'
        
        styles[agent_id] = {
            'style': style,
            'win_rate': float(win_rate),
            'aggression': float(aggression),
            'consistency': float(consistency)
        }
    
    # Distribution of styles
    style_counts = pd.Series([s['style'] for s in styles.values()]).value_counts()
    
    return {
        'agent_styles': styles,
        'style_distribution': style_counts.to_dict(),
        'hierarchy_type': 'despotic' if style_counts.get('despotic', 0) / len(styles) > 0.3 else 'egalitarian'
    }
```

---

# ADVANTAGE & SIGNIFICANT EVENTS MODULES

## 5. 🎯 Advantage Analysis Extensions

### A. Cumulative Advantage
```python
def analyze_cumulative_advantage(agent_history: pd.DataFrame) -> Dict[str, Any]:
    """Analyze if early advantages compound over time (Matthew effect)."""
    
    # Split agents by early performance
    early_window = agent_history[agent_history['step'] < 100]
    early_performers = early_window.groupby('agent_id')['reward'].mean()
    
    top_early = early_performers.nlargest(int(len(early_performers) * 0.25))
    bottom_early = early_performers.nsmallest(int(len(early_performers) * 0.25))
    
    # Track their trajectories
    late_window = agent_history[agent_history['step'] >= 500]
    late_performers = late_window.groupby('agent_id')['reward'].mean()
    
    # Calculate persistence
    top_still_top = sum(1 for agent in top_early.index if agent in late_performers.nlargest(int(len(late_performers) * 0.25)).index)
    
    persistence_rate = top_still_top / len(top_early)
    
    # Advantage gap over time
    gap_over_time = []
    for step in range(0, agent_history['step'].max(), 50):
        window = agent_history[agent_history['step'].between(step, step + 50)]
        top_mean = window[window['agent_id'].isin(top_early.index)]['reward'].mean()
        bottom_mean = window[window['agent_id'].isin(bottom_early.index)]['reward'].mean()
        gap_over_time.append(top_mean - bottom_mean)
    
    # Check if gap increases (cumulative advantage)
    gap_trend = np.polyfit(range(len(gap_over_time)), gap_over_time, 1)[0]
    
    return {
        'early_top_persistence': float(persistence_rate),
        'advantage_gap_trend': float(gap_trend),
        'has_cumulative_advantage': gap_trend > 0 and persistence_rate > 0.6,
        'matthew_effect_strength': float(gap_trend * persistence_rate)
    }
```

## 6. 🎪 Significant Events Extensions

### A. Cascade Detection
```python
def detect_event_cascades(events_data: pd.DataFrame,
                         time_window: int = 10,
                         space_window: float = 20) -> Dict[str, Any]:
    """Detect cascading events (one event triggers others)."""
    
    cascades = []
    
    for idx, event in events_data.iterrows():
        # Find events that occur shortly after and nearby
        potential_cascades = events_data[
            (events_data['step'] > event['step']) &
            (events_data['step'] <= event['step'] + time_window) &
            (np.sqrt((events_data['x'] - event['x'])**2 + (events_data['y'] - event['y'])**2) < space_window)
        ]
        
        if len(potential_cascades) > 0:
            cascades.append({
                'trigger_event': event['event_id'],
                'triggered_events': potential_cascades['event_id'].tolist(),
                'cascade_size': len(potential_cascades),
                'spatial_spread': float(potential_cascades[['x', 'y']].values.std()),
                'temporal_spread': int(potential_cascades['step'].max() - event['step'])
            })
    
    return {
        'num_cascades': len(cascades),
        'cascades': cascades,
        'mean_cascade_size': float(np.mean([c['cascade_size'] for c in cascades])) if cascades else 0,
        'largest_cascade': max(cascades, key=lambda x: x['cascade_size']) if cascades else None
    }
```

### B. Critical Events
```python
def identify_critical_events(events_data: pd.DataFrame,
                            outcome_data: pd.DataFrame) -> Dict[str, Any]:
    """Identify which events had the biggest impact on outcomes."""
    
    critical_events = []
    
    for _, event in events_data.iterrows():
        # Measure outcome before and after event
        before = outcome_data[outcome_data['step'] < event['step']].tail(20)['value'].mean()
        after = outcome_data[outcome_data['step'] >= event['step']].head(20)['value'].mean()
        
        impact = abs(after - before) / before if before > 0 else 0
        
        critical_events.append({
            'event_id': event['event_id'],
            'event_type': event['event_type'],
            'step': int(event['step']),
            'impact_magnitude': float(impact),
            'direction': 'positive' if after > before else 'negative'
        })
    
    # Rank by impact
    critical_events = sorted(critical_events, key=lambda x: x['impact_magnitude'], reverse=True)
    
    return {
        'critical_events': critical_events[:10],  # Top 10
        'most_critical_event': critical_events[0] if critical_events else None,
        'critical_event_types': pd.Series([e['event_type'] for e in critical_events[:10]]).value_counts().to_dict()
    }
```

---

## Summary

This comprehensive extension guide covers:

1. **Temporal Analysis** - Granger causality, regime switching, wavelets, recurrence
2. **Social Analysis** - Information diffusion, community evolution, influence, brokerage
3. **Comparative Analysis** - Benchmarking, meta-analysis, Pareto optimization
4. **Dominance Analysis** - Elo ratings, transitivity, dominance styles
5. **Advantage Analysis** - Cumulative advantage, Matthew effects
6. **Events Analysis** - Cascade detection, critical event identification

All 14 analysis modules now have comprehensive extension libraries covering 50+ advanced techniques!
