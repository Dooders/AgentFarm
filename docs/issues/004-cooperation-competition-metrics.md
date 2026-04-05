# [Feature] Add cooperation and competition dynamics metrics

**Labels:** `enhancement`, `analysis`

## Summary

The documentation promises `measure_cooperation_levels()` and `measure_competition_intensity()` but neither exists. The closest real implementation is `analyze_competitive_interactions()` which only counts attack actions. There are no cooperation metrics at all.

## Proposed Implementation

### Cooperation Rate

```python
# cooperation_rate per step = share_actions / total_interactions
```

### Competition Intensity

```python
# competition_intensity per step = attack_events / total_interactions_per_step
```

### Additional Metrics

- **Alliance formation rate**: How frequently system agents act near each other
- **Resource sharing index**: Proportion of resource transfers vs hoarding events
- **Combat escalation**: Rate of change of `competition_intensity` over time

## Suggested Location

Either:
- New `farm/analysis/social_dynamics/` module extending `BaseAnalysisModule`, **or**
- Added to the existing `farm/analysis/social_behavior/` module

## Acceptance Criteria

- [ ] `cooperation_rate` metric computed per-step from action data
- [ ] `competition_intensity` metric computed per-step
- [ ] Time-series trend analysis for both metrics
- [ ] Tests covering both metrics
- [ ] Accessible via `AnalysisService`
