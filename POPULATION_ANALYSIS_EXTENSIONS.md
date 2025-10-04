# Advanced Population Analysis - Extension Ideas

## Overview
Beyond the current implementations, here are sophisticated population analyses you could add to gain deeper insights into your simulations.

---

## 1. 🔄 Temporal Pattern Analysis

### A. Cyclic Behavior Detection
Identify periodic patterns in population dynamics.

```python
def detect_population_cycles(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect cyclic patterns using FFT and autocorrelation."""
    
    from scipy import signal, fft
    
    population = df['total_agents'].values
    
    # Autocorrelation
    autocorr = np.correlate(population, population, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Find peaks (potential cycle periods)
    peaks, properties = signal.find_peaks(autocorr, prominence=0.3)
    
    # FFT for frequency analysis
    fft_vals = fft.fft(population)
    frequencies = fft.fftfreq(len(population), d=1)
    power = np.abs(fft_vals)**2
    
    # Find dominant frequencies
    positive_freq_idx = frequencies > 0
    dominant_idx = np.argmax(power[positive_freq_idx])
    dominant_period = 1 / frequencies[positive_freq_idx][dominant_idx]
    
    return {
        'has_cycles': len(peaks) > 0,
        'cycle_periods': peaks.tolist(),
        'dominant_period': float(dominant_period),
        'autocorrelation_at_lag_10': float(autocorr[10]),
        'spectral_entropy': float(spectral_entropy(power))
    }
```

**Use Cases:**
- Predator-prey dynamics
- Resource regeneration cycles
- Seasonal patterns
- Boom-bust cycles

### B. Regime Change Detection
Identify when population dynamics fundamentally shift.

```python
def detect_regime_changes(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect regime changes using change point detection."""
    
    from ruptures import Pelt
    
    population = df['total_agents'].values
    
    # Change point detection
    model = Pelt(model="rbf").fit(population)
    change_points = model.predict(pen=10)
    
    # Analyze each regime
    regimes = []
    start = 0
    for end in change_points:
        segment = population[start:end]
        regimes.append({
            'start_step': int(start),
            'end_step': int(end),
            'mean': float(np.mean(segment)),
            'volatility': float(np.std(segment)),
            'trend': float(np.polyfit(range(len(segment)), segment, 1)[0])
        })
        start = end
    
    return {
        'num_regimes': len(regimes),
        'regimes': regimes,
        'regime_transitions': change_points[:-1]
    }
```

**Use Cases:**
- Environmental shifts
- Strategy changes
- Phase transitions
- Critical events

---

## 2. 📊 Survival & Mortality Analysis

### A. Lifespan Distribution Analysis
Analyze individual agent lifespans (if you have birth/death tracking).

```python
def analyze_lifespan_distribution(agent_lifespans: np.ndarray) -> Dict[str, Any]:
    """Comprehensive lifespan statistics and distribution fitting."""
    
    from scipy import stats
    
    # Basic statistics
    results = {
        'mean_lifespan': float(np.mean(agent_lifespans)),
        'median_lifespan': float(np.median(agent_lifespans)),
        'max_lifespan': float(np.max(agent_lifespans)),
        'lifespan_std': float(np.std(agent_lifespans))
    }
    
    # Fit distributions
    distributions = ['expon', 'gamma', 'weibull_min', 'lognorm']
    best_fit = None
    best_aic = np.inf
    
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(agent_lifespans)
        aic = 2 * len(params) - 2 * np.sum(dist.logpdf(agent_lifespans, *params))
        
        if aic < best_aic:
            best_aic = aic
            best_fit = {'distribution': dist_name, 'params': params, 'aic': aic}
    
    results['best_fit_distribution'] = best_fit
    
    # Survival curve (Kaplan-Meier style)
    sorted_lifespans = np.sort(agent_lifespans)
    survival_prob = 1 - np.arange(1, len(sorted_lifespans) + 1) / len(sorted_lifespans)
    
    results['survival_curve'] = {
        'time': sorted_lifespans.tolist(),
        'probability': survival_prob.tolist()
    }
    
    return results
```

### B. Hazard Rate Analysis
Calculate instantaneous death rates.

```python
def compute_hazard_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute instantaneous mortality risk over time."""
    
    if 'deaths' not in df.columns:
        return {}
    
    # Hazard rate = deaths / population at risk
    hazard_rate = df['deaths'] / df['total_agents']
    
    # Smoothed hazard rate
    smoothed = hazard_rate.rolling(window=10, min_periods=1).mean()
    
    # Cumulative hazard
    cumulative_hazard = hazard_rate.cumsum()
    
    return {
        'mean_hazard_rate': float(hazard_rate.mean()),
        'hazard_rate_trend': float(np.polyfit(df['step'], hazard_rate, 1)[0]),
        'max_hazard_period': int(hazard_rate.idxmax()),
        'cumulative_hazard': cumulative_hazard.tolist()
    }
```

---

## 3. 🎯 Carrying Capacity & Equilibrium Analysis

### A. Estimate Carrying Capacity
Find the maximum sustainable population.

```python
def estimate_carrying_capacity(df: pd.DataFrame) -> Dict[str, Any]:
    """Estimate carrying capacity using logistic growth model."""
    
    from scipy.optimize import curve_fit
    
    def logistic_model(t, r, K, N0):
        """Logistic growth: N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))"""
        return K / (1 + ((K - N0) / N0) * np.exp(-r * t))
    
    steps = df['step'].values
    population = df['total_agents'].values
    
    try:
        # Fit logistic curve
        params, _ = curve_fit(
            logistic_model, 
            steps, 
            population,
            p0=[0.1, np.max(population), population[0]],
            maxfev=10000
        )
        
        r, K, N0 = params
        
        # Calculate how close we are to carrying capacity
        final_pop = population[-1]
        percent_of_capacity = (final_pop / K) * 100
        
        return {
            'estimated_capacity': float(K),
            'growth_rate': float(r),
            'initial_population': float(N0),
            'current_percent_of_capacity': float(percent_of_capacity),
            'time_to_capacity': estimate_time_to_capacity(r, K, final_pop)
        }
    except:
        return {'error': 'Could not fit logistic model'}
```

### B. Equilibrium Analysis
Find stable population levels.

```python
def find_equilibrium_points(df: pd.DataFrame, tolerance: float = 0.01) -> Dict[str, Any]:
    """Identify equilibrium points where population stabilizes."""
    
    population = df['total_agents'].values
    steps = df['step'].values
    
    # Calculate derivative (rate of change)
    dpdt = np.gradient(population)
    
    # Find points where derivative is near zero
    equilibria = []
    in_equilibrium = False
    start_idx = 0
    
    for i, rate in enumerate(dpdt):
        if abs(rate) < tolerance:
            if not in_equilibrium:
                in_equilibrium = True
                start_idx = i
        else:
            if in_equilibrium:
                # End of equilibrium period
                equilibria.append({
                    'start_step': int(steps[start_idx]),
                    'end_step': int(steps[i-1]),
                    'duration': i - start_idx,
                    'population_level': float(np.mean(population[start_idx:i])),
                    'stability': float(np.std(population[start_idx:i]))
                })
                in_equilibrium = False
    
    return {
        'num_equilibria': len(equilibria),
        'equilibria': equilibria,
        'has_stable_equilibrium': any(e['duration'] > 50 for e in equilibria)
    }
```

---

## 4. 🧬 Age Structure & Cohort Analysis

### A. Population Pyramid
Analyze age distribution (if you track generations).

```python
def analyze_age_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze population age/generation structure."""
    
    # Assuming you have generation or birth_time data
    # This would require additional data from your simulation
    
    age_groups = {
        'young': (0, 10),      # Steps 0-10 old
        'adult': (11, 50),     # Steps 11-50 old
        'old': (51, np.inf)    # Steps 51+ old
    }
    
    # Calculate proportions
    proportions = {}
    for group, (min_age, max_age) in age_groups.items():
        # This is pseudocode - you'd query your database
        count = get_agent_count_by_age(min_age, max_age)
        proportions[group] = count / df['total_agents'].iloc[-1]
    
    # Dependency ratio
    dependency_ratio = (proportions['young'] + proportions['old']) / proportions['adult']
    
    return {
        'age_distribution': proportions,
        'dependency_ratio': float(dependency_ratio),
        'median_age': calculate_median_age(),
        'aging_rate': calculate_aging_trend()
    }
```

### B. Cohort Analysis
Track specific generations over time.

```python
def analyze_cohorts(agents_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze survival and performance of different cohorts/generations."""
    
    cohorts = agents_df.groupby('generation')
    
    results = []
    for gen, group in cohorts:
        results.append({
            'generation': int(gen),
            'initial_size': len(group),
            'mean_lifespan': float(group['lifespan'].mean()),
            'survival_to_maturity': float((group['lifespan'] > 20).sum() / len(group)),
            'mean_resources': float(group['final_resources'].mean()),
            'success_rate': float((group['success_metric'] > threshold).sum() / len(group))
        })
    
    return {
        'cohorts': results,
        'best_generation': max(results, key=lambda x: x['mean_lifespan'])['generation'],
        'generational_improvement': calculate_trend([c['mean_lifespan'] for c in results])
    }
```

---

## 5. 🌍 Spatial Population Analysis

### A. Spatial Distribution Analysis
Analyze clustering and dispersion.

```python
def analyze_spatial_distribution(agents_positions: np.ndarray) -> Dict[str, Any]:
    """Analyze spatial patterns in agent distribution."""
    
    from scipy.spatial import distance_matrix
    from sklearn.cluster import DBSCAN
    
    # Calculate nearest neighbor distances
    dist_matrix = distance_matrix(agents_positions, agents_positions)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor_dist = dist_matrix.min(axis=1)
    
    # Clustering analysis
    clustering = DBSCAN(eps=10, min_samples=5).fit(agents_positions)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    # Spatial autocorrelation (Moran's I)
    moran_i = calculate_morans_i(agents_positions)
    
    # Dispersion metrics
    centroid = agents_positions.mean(axis=0)
    distances_from_center = np.linalg.norm(agents_positions - centroid, axis=1)
    
    return {
        'mean_nearest_neighbor_distance': float(nearest_neighbor_dist.mean()),
        'clustering_coefficient': float(n_clusters / len(agents_positions)),
        'num_spatial_clusters': int(n_clusters),
        'spatial_autocorrelation': float(moran_i),
        'mean_distance_from_center': float(distances_from_center.mean()),
        'spatial_dispersion': float(distances_from_center.std()),
        'is_clustered': moran_i > 0.3
    }
```

### B. Territory Analysis
Analyze space usage and territory.

```python
def analyze_territories(agents_df: pd.DataFrame, grid_size: int = 100) -> Dict[str, Any]:
    """Analyze territorial behavior and space usage."""
    
    # Create spatial bins
    x_bins = np.linspace(0, grid_size, 20)
    y_bins = np.linspace(0, grid_size, 20)
    
    # Count agents per cell
    H, _, _ = np.histogram2d(
        agents_df['position_x'], 
        agents_df['position_y'],
        bins=[x_bins, y_bins]
    )
    
    # Occupied cells
    occupied_cells = (H > 0).sum()
    total_cells = H.size
    
    # Population density variance (territory exclusivity)
    density_variance = np.var(H)
    
    return {
        'space_utilization': float(occupied_cells / total_cells),
        'mean_density': float(H[H > 0].mean()),
        'max_density': float(H.max()),
        'density_variance': float(density_variance),
        'territorial_exclusivity': float(density_variance / H.mean())
    }
```

---

## 6. 🔗 Competition & Interaction Analysis

### A. Competitive Exclusion
Analyze dominance between types.

```python
def analyze_competitive_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze competition between agent types."""
    
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    available = [t for t in agent_types if t in df.columns]
    
    results = {}
    
    for i, type1 in enumerate(available):
        for type2 in available[i+1:]:
            # Competitive Lotka-Volterra analysis
            ratio = df[type1] / (df[type2] + 1e-6)
            
            results[f'{type1}_vs_{type2}'] = {
                'final_ratio': float(ratio.iloc[-1]),
                'mean_ratio': float(ratio.mean()),
                'ratio_trend': float(np.polyfit(df['step'], ratio, 1)[0]),
                'dominance_shifts': int((ratio.diff().abs() > 0.5).sum()),
                'competitive_outcome': determine_outcome(ratio)
            }
    
    return results

def determine_outcome(ratio_series):
    """Determine competitive outcome."""
    final_ratio = ratio_series.iloc[-1]
    trend = np.polyfit(range(len(ratio_series)), ratio_series, 1)[0]
    
    if final_ratio > 2 and trend > 0:
        return 'Type 1 dominance'
    elif final_ratio < 0.5 and trend < 0:
        return 'Type 2 dominance'
    elif abs(trend) < 0.01:
        return 'Coexistence'
    else:
        return 'Unstable competition'
```

### B. Resource Competition Index
Measure resource competition intensity.

```python
def compute_resource_competition(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze resource competition intensity."""
    
    if 'avg_resources' not in df.columns:
        return {}
    
    population = df['total_agents']
    resources = df['avg_resources']
    
    # Per-capita resource availability
    per_capita = resources / population
    
    # Competition index (inverse of per-capita resources)
    competition_index = 1 / (per_capita + 1e-6)
    
    # Critical competition threshold
    critical_threshold = np.percentile(per_capita, 25)
    
    return {
        'mean_competition_index': float(competition_index.mean()),
        'max_competition_step': int(competition_index.idxmax()),
        'resource_limitation_frequency': float((per_capita < critical_threshold).mean()),
        'competition_intensity_trend': float(np.polyfit(df['step'], competition_index, 1)[0])
    }
```

---

## 7. 📈 Predictive Modeling

### A. Population Forecasting
Predict future population levels.

```python
def forecast_population(df: pd.DataFrame, steps_ahead: int = 50) -> Dict[str, Any]:
    """Forecast future population using multiple methods."""
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.linear_model import LinearRegression
    
    population = df['total_agents'].values
    
    # Method 1: Linear extrapolation
    X = df['step'].values.reshape(-1, 1)
    linear_model = LinearRegression().fit(X, population)
    future_steps = np.arange(len(population), len(population) + steps_ahead).reshape(-1, 1)
    linear_forecast = linear_model.predict(future_steps)
    
    # Method 2: Exponential smoothing
    exp_model = ExponentialSmoothing(population, trend='add', seasonal=None)
    exp_fit = exp_model.fit()
    exp_forecast = exp_fit.forecast(steps_ahead)
    
    # Method 3: ARIMA (if you have statsmodels)
    # arima_model = ARIMA(population, order=(1,1,1))
    # arima_fit = arima_model.fit()
    # arima_forecast = arima_fit.forecast(steps_ahead)
    
    return {
        'forecast_horizon': steps_ahead,
        'linear_forecast': linear_forecast.tolist(),
        'exponential_forecast': exp_forecast.tolist(),
        'confidence_intervals': calculate_forecast_ci(population, steps_ahead),
        'trend_direction': 'increasing' if linear_forecast[-1] > population[-1] else 'decreasing'
    }
```

### B. Risk Analysis
Predict extinction probability.

```python
def estimate_extinction_risk(df: pd.DataFrame) -> Dict[str, Any]:
    """Estimate probability of population collapse."""
    
    population = df['total_agents'].values
    
    # Calculate volatility
    returns = np.diff(np.log(population + 1))
    volatility = np.std(returns)
    
    # Minimum viable population threshold
    mvp_threshold = np.percentile(population, 10)
    
    # Times population dropped below threshold
    below_threshold = (population < mvp_threshold).sum()
    
    # Trend toward zero
    trend = np.polyfit(df['step'], population, 1)[0]
    
    # Simple risk score (0-1)
    risk_score = (
        0.4 * (volatility / np.mean(population)) +
        0.3 * (below_threshold / len(population)) +
        0.3 * max(0, -trend / np.mean(population))
    )
    
    return {
        'extinction_risk_score': float(min(risk_score, 1.0)),
        'volatility': float(volatility),
        'minimum_viable_population': float(mvp_threshold),
        'times_below_mvp': int(below_threshold),
        'trend_toward_extinction': trend < 0,
        'risk_level': categorize_risk(risk_score)
    }
```

---

## 8. 🔬 Statistical Validation

### A. Randomness Testing
Test if population changes are random or deterministic.

```python
def test_randomness(df: pd.DataFrame) -> Dict[str, Any]:
    """Test whether population dynamics show random vs deterministic patterns."""
    
    from scipy import stats
    
    population = df['total_agents'].values
    changes = np.diff(population)
    
    # Runs test for randomness
    median = np.median(changes)
    runs = np.sum(np.diff((changes > median).astype(int)) != 0) + 1
    n1 = np.sum(changes > median)
    n2 = np.sum(changes <= median)
    
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    z_score = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(changes, lags=[10], return_df=True)
    
    return {
        'runs_test_z_score': float(z_score),
        'is_random': abs(z_score) < 1.96,  # 95% confidence
        'autocorrelation_test_p_value': float(lb_test['lb_pvalue'].values[0]),
        'has_significant_autocorrelation': float(lb_test['lb_pvalue'].values[0]) < 0.05
    }
```

---

## 9. 🎲 Stochastic Analysis

### A. Noise Decomposition
Separate signal from noise.

```python
def decompose_population_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Decompose population into trend, seasonality, and noise."""
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    population = df['total_agents']
    
    # Perform decomposition
    decomposition = seasonal_decompose(
        population,
        model='additive',
        period=min(50, len(population) // 2),
        extrapolate_trend='freq'
    )
    
    # Calculate signal-to-noise ratio
    signal_power = np.var(decomposition.trend.dropna())
    noise_power = np.var(decomposition.resid.dropna())
    snr = 10 * np.log10(signal_power / noise_power)
    
    return {
        'signal_to_noise_ratio': float(snr),
        'trend_strength': float(np.var(decomposition.trend.dropna()) / np.var(population)),
        'noise_level': float(np.std(decomposition.resid.dropna())),
        'has_seasonal_component': float(np.var(decomposition.seasonal.dropna())) > 0.01
    }
```

---

## 10. 🏆 Comparative Analysis

### A. Multi-Simulation Comparison
Compare populations across different experiments.

```python
def compare_simulations(simulations: List[pd.DataFrame]) -> Dict[str, Any]:
    """Compare population dynamics across multiple simulations."""
    
    # Align time series
    max_length = max(len(sim) for sim in simulations)
    aligned = []
    for sim in simulations:
        if len(sim) < max_length:
            # Pad with last value
            padded = np.pad(sim['total_agents'].values, 
                          (0, max_length - len(sim)), 
                          mode='edge')
        else:
            padded = sim['total_agents'].values[:max_length]
        aligned.append(padded)
    
    aligned = np.array(aligned)
    
    # Statistics across simulations
    mean_trajectory = np.mean(aligned, axis=0)
    std_trajectory = np.std(aligned, axis=0)
    
    # Variability between simulations
    between_sim_var = np.var([sim.mean() for sim in aligned])
    within_sim_var = np.mean([np.var(sim) for sim in aligned])
    
    return {
        'num_simulations': len(simulations),
        'mean_final_population': float(mean_trajectory[-1]),
        'std_final_population': float(std_trajectory[-1]),
        'coefficient_of_variation': float(std_trajectory[-1] / mean_trajectory[-1]),
        'between_simulation_variance': float(between_sim_var),
        'within_simulation_variance': float(within_sim_var),
        'reproducibility_score': 1 - (between_sim_var / (between_sim_var + within_sim_var))
    }
```

---

## Implementation Priority

### High Priority (Implement First):
1. ✅ **Carrying Capacity Estimation** - Fundamental metric
2. ✅ **Equilibrium Analysis** - Understanding stability
3. ✅ **Competition Analysis** - Understanding dynamics between types
4. ✅ **Risk Analysis** - Practical for monitoring

### Medium Priority:
5. **Cycle Detection** - If you see oscillations
6. **Spatial Analysis** - If location matters
7. **Forecasting** - For predictions
8. **Survival Analysis** - If tracking individuals

### Lower Priority (Advanced):
9. **Regime Detection** - For complex dynamics
10. **Stochastic Analysis** - For theoretical understanding
11. **Age Structure** - If relevant to your model
12. **Statistical Tests** - For validation

---

## Quick Win: Add to Your Module

Here's a complete function you can add immediately:

```python
def analyze_population_sustainability(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Comprehensive sustainability analysis combining multiple metrics."""
    
    results = {
        'carrying_capacity': estimate_carrying_capacity(df),
        'equilibrium': find_equilibrium_points(df),
        'extinction_risk': estimate_extinction_risk(df),
        'resource_competition': compute_resource_competition(df)
    }
    
    # Save results
    output_file = ctx.get_output_file("population_sustainability.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    ctx.logger.info(f"Saved sustainability analysis to {output_file}")
```

---

## Summary

The analyses above cover:
- **Temporal**: Cycles, regimes, trends
- **Survival**: Lifespans, hazards, cohorts  
- **Capacity**: Carrying capacity, equilibrium
- **Structure**: Age, spatial, territory
- **Competition**: Inter-type dynamics, resources
- **Prediction**: Forecasting, risk assessment
- **Validation**: Statistical tests, noise analysis
- **Comparison**: Multi-simulation studies

Pick the ones most relevant to your research questions!
