"""
Logging Performance Patterns and Best Practices

This file demonstrates efficient logging patterns to minimize performance
overhead while maintaining observability.
"""

from farm.utils.logging_config import get_logger
from farm.utils.logging_utils import LogSampler
import time

logger = get_logger(__name__)


# =============================================================================
# PATTERN 1: Conditional Logging (Only if Level is Enabled)
# =============================================================================

def pattern_conditional_logging():
    """Only compute expensive values if logging level is enabled"""
    
    # BAD: Always computes expensive value even if DEBUG is disabled
    def bad_approach(agents):
        logger.debug(f"Agent stats: {compute_expensive_stats(agents)}")
    
    # GOOD: Only computes if DEBUG level is enabled
    def good_approach(agents):
        if logger.isEnabledFor(logging.DEBUG):
            stats = compute_expensive_stats(agents)
            logger.debug("agent_stats", **stats)
    
    # BETTER: Use lazy evaluation with lambda (structlog evaluates lazily)
    def better_approach(agents):
        logger.debug(
            "agent_stats",
            stats=lambda: compute_expensive_stats(agents)  # Only called if logged
        )


# =============================================================================
# PATTERN 2: Periodic Logging (Every N Events)
# =============================================================================

def pattern_periodic_logging():
    """Log every N occurrences to reduce log volume"""
    
    class PeriodicLogger:
        def __init__(self, interval=100):
            self.interval = interval
            self.counter = 0
        
        def maybe_log(self, **kwargs):
            self.counter += 1
            if self.counter % self.interval == 0:
                logger.info("periodic_event", count=self.counter, **kwargs)
    
    # Usage in simulation loop
    def simulation_loop(environment):
        action_logger = PeriodicLogger(interval=100)
        
        for step in range(1000):
            for agent in environment.agents:
                result = agent.act()
                # Only logs every 100 actions
                action_logger.maybe_log(
                    agent_id=agent.agent_id,
                    action=result.action_name,
                )


# =============================================================================
# PATTERN 3: Sampled Logging (Log N% of Events)
# =============================================================================

def pattern_sampled_logging():
    """Use LogSampler for probabilistic sampling"""
    
    from farm.utils.logging_utils import LogSampler
    
    # Log 10% of high-frequency events
    sampler = LogSampler(sample_rate=0.1)
    
    def process_agent_action(agent, action):
        if sampler.should_log():
            logger.debug(
                "agent_action_sampled",
                agent_id=agent.agent_id,
                action=action.name,
                sample_rate=0.1,
            )


# =============================================================================
# PATTERN 4: Time-Based Throttling
# =============================================================================

def pattern_time_throttle():
    """Limit logging to once per time interval"""
    
    class TimeThrottledLogger:
        def __init__(self, min_interval_seconds=1.0):
            self.min_interval = min_interval_seconds
            self.last_log_time = {}
        
        def log_if_ready(self, event_key, **kwargs):
            current_time = time.time()
            last_time = self.last_log_time.get(event_key, 0)
            
            if current_time - last_time >= self.min_interval:
                logger.info(event_key, **kwargs)
                self.last_log_time[event_key] = current_time
                return True
            return False
    
    # Usage
    throttled_logger = TimeThrottledLogger(min_interval_seconds=5.0)
    
    def update_loop():
        while True:
            # Only logs once every 5 seconds, no matter how many iterations
            throttled_logger.log_if_ready(
                "loop_status",
                iterations=iteration_count,
                active_agents=len(agents),
            )


# =============================================================================
# PATTERN 5: Aggregate Logging (Batch Statistics)
# =============================================================================

def pattern_aggregate_logging():
    """Collect statistics and log in batches"""
    
    class AggregateLogger:
        def __init__(self, flush_interval=100):
            self.flush_interval = flush_interval
            self.stats = {
                'count': 0,
                'success': 0,
                'failure': 0,
                'total_duration': 0,
            }
        
        def record_event(self, success, duration):
            self.stats['count'] += 1
            if success:
                self.stats['success'] += 1
            else:
                self.stats['failure'] += 1
            self.stats['total_duration'] += duration
            
            # Flush periodically
            if self.stats['count'] % self.flush_interval == 0:
                self.flush()
        
        def flush(self):
            if self.stats['count'] > 0:
                avg_duration = self.stats['total_duration'] / self.stats['count']
                success_rate = self.stats['success'] / self.stats['count']
                
                logger.info(
                    "aggregated_events",
                    total_events=self.stats['count'],
                    success_count=self.stats['success'],
                    failure_count=self.stats['failure'],
                    success_rate=round(success_rate, 3),
                    avg_duration_ms=round(avg_duration * 1000, 2),
                )
                
                # Reset stats
                self.stats = {
                    'count': 0,
                    'success': 0,
                    'failure': 0,
                    'total_duration': 0,
                }
    
    # Usage
    action_logger = AggregateLogger(flush_interval=100)
    
    for agent in agents:
        start = time.time()
        success = agent.perform_action()
        duration = time.time() - start
        action_logger.record_event(success, duration)


# =============================================================================
# PATTERN 6: Milestone-Based Logging
# =============================================================================

def pattern_milestone_logging():
    """Log only at significant thresholds"""
    
    class MilestoneLogger:
        def __init__(self, milestones):
            self.milestones = set(milestones)
            self.logged = set()
        
        def check_and_log(self, current_value, **kwargs):
            # Find milestones we've crossed but not logged yet
            for milestone in self.milestones:
                if current_value >= milestone and milestone not in self.logged:
                    logger.info(
                        "milestone_reached",
                        milestone=milestone,
                        current_value=current_value,
                        **kwargs
                    )
                    self.logged.add(milestone)
    
    # Usage for population
    population_logger = MilestoneLogger([10, 50, 100, 500, 1000])
    
    def on_population_change(new_population):
        population_logger.check_and_log(
            new_population,
            step=current_step,
            resources=total_resources,
        )


# =============================================================================
# PATTERN 7: Context-Aware Logging Levels
# =============================================================================

def pattern_context_aware_levels():
    """Adjust logging based on context/conditions"""
    
    def log_agent_action(agent, action, result):
        # Use different levels based on significance
        if action.name in ['reproduce', 'attack']:
            # High-impact actions always at INFO
            logger.info(
                "significant_action",
                agent_id=agent.agent_id,
                action=action.name,
                success=result.success,
            )
        elif not result.success:
            # Failed actions at WARNING
            logger.warning(
                "action_failed",
                agent_id=agent.agent_id,
                action=action.name,
                reason=result.reason,
            )
        else:
            # Normal successful actions at DEBUG
            logger.debug(
                "action_completed",
                agent_id=agent.agent_id,
                action=action.name,
            )


# =============================================================================
# PATTERN 8: Structured Metrics Logging
# =============================================================================

def pattern_structured_metrics():
    """Log metrics in structured format for analysis"""
    
    class MetricsLogger:
        def __init__(self, log_interval=100):
            self.log_interval = log_interval
            self.metrics = {}
        
        def update_metric(self, name, value):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
        
        def log_summary(self, step):
            if step % self.log_interval == 0:
                summary = {}
                for name, values in self.metrics.items():
                    if values:
                        summary[name] = {
                            'mean': round(sum(values) / len(values), 2),
                            'min': round(min(values), 2),
                            'max': round(max(values), 2),
                            'count': len(values),
                        }
                
                logger.info(
                    "metrics_summary",
                    step=step,
                    interval=self.log_interval,
                    metrics=summary,
                )
                
                # Reset
                self.metrics = {}
    
    # Usage
    metrics = MetricsLogger(log_interval=100)
    
    for step in range(1000):
        for agent in agents:
            metrics.update_metric('health', agent.health)
            metrics.update_metric('resources', agent.resources)
        
        metrics.log_summary(step)


# =============================================================================
# PATTERN 9: Lazy String Formatting
# =============================================================================

def pattern_lazy_formatting():
    """Avoid string formatting unless log is emitted"""
    
    # BAD: Always formats string even if not logged
    def bad_approach(agents):
        formatted_list = ", ".join(f"{a.id}:{a.health}" for a in agents)
        logger.debug(f"Agent list: {formatted_list}")
    
    # GOOD: Only formats if DEBUG is enabled
    def good_approach(agents):
        if logger.isEnabledFor(logging.DEBUG):
            formatted_list = ", ".join(f"{a.id}:{a.health}" for a in agents)
            logger.debug("agent_list", formatted=formatted_list)
    
    # BETTER: Use structlog's lazy evaluation
    def better_approach(agents):
        logger.debug(
            "agent_list",
            agents=[{"id": a.id, "health": a.health} for a in agents],
        )


# =============================================================================
# PATTERN 10: Hierarchical Logging (Progressive Detail)
# =============================================================================

def pattern_hierarchical_logging():
    """Provide different detail levels at different log levels"""
    
    def log_simulation_state(environment):
        # Always log high-level summary at INFO
        logger.info(
            "simulation_state",
            step=environment.time,
            agents=len(environment.agents),
            resources=environment.cached_total_resources,
        )
        
        # Add more detail at DEBUG
        if logger.isEnabledFor(logging.DEBUG):
            agent_types = {}
            for agent in environment._agent_objects.values():
                agent_type = agent.__class__.__name__
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            logger.debug(
                "simulation_state_detailed",
                step=environment.time,
                agent_types=agent_types,
                active_resources=len([r for r in environment.resources if r.amount > 0]),
                avg_agent_health=sum(a.health for a in environment._agent_objects.values()) / len(environment._agent_objects) if environment._agent_objects else 0,
            )


# =============================================================================
# COMPLETE EXAMPLE: Efficient Logging in Simulation Loop
# =============================================================================

class EfficientSimulationLogger:
    """Combines multiple patterns for optimal logging"""
    
    def __init__(self):
        # Periodic loggers for different frequencies
        self.step_logger = PeriodicLogger(interval=100)
        self.action_sampler = LogSampler(sample_rate=0.01)  # 1% of actions
        
        # Milestone tracking
        self.population_milestones = MilestoneLogger([10, 50, 100, 500, 1000])
        
        # Aggregate statistics
        self.action_stats = AggregateLogger(flush_interval=1000)
        
        # Time throttling for warnings
        self.warning_throttle = TimeThrottledLogger(min_interval_seconds=10.0)
        
        # Metrics accumulation
        self.metrics = MetricsLogger(log_interval=100)
    
    def log_step_start(self, environment):
        """Log at step granularity"""
        self.step_logger.maybe_log(
            step=environment.time,
            agents=len(environment.agents),
            resources=environment.cached_total_resources,
        )
    
    def log_agent_action(self, agent, action, result, duration):
        """Log individual actions with sampling"""
        # Always aggregate
        self.action_stats.record_event(result.success, duration)
        
        # Sample for detailed logs
        if self.action_sampler.should_log():
            logger.debug(
                "action_sampled",
                agent_id=agent.agent_id,
                action=action.name,
                success=result.success,
                duration_ms=round(duration * 1000, 2),
            )
        
        # Always log significant actions
        if action.name in ['reproduce', 'attack']:
            logger.info(
                "significant_action",
                agent_id=agent.agent_id,
                action=action.name,
                success=result.success,
            )
    
    def log_population_change(self, new_population, step):
        """Log population milestones"""
        self.population_milestones.check_and_log(
            new_population,
            step=step,
        )
    
    def log_resource_warning(self, resource_ratio):
        """Log resource warnings with throttling"""
        if resource_ratio < 0.1:
            self.warning_throttle.log_if_ready(
                "resources_low",
                ratio=round(resource_ratio, 3),
            )
    
    def log_metrics(self, environment):
        """Update and log metrics"""
        for agent in environment._agent_objects.values():
            self.metrics.update_metric('health', agent.health)
            self.metrics.update_metric('resources', agent.resources)
        
        self.metrics.log_summary(environment.time)
    
    def flush_all(self):
        """Force flush all pending logs"""
        self.action_stats.flush()
        self.metrics.log_summary(current_step)


# =============================================================================
# PERFORMANCE MEASUREMENT
# =============================================================================

def measure_logging_overhead():
    """Measure performance impact of logging"""
    import time
    
    iterations = 100000
    
    # Baseline: No logging
    start = time.time()
    for i in range(iterations):
        x = i * 2
    baseline_time = time.time() - start
    
    # With DEBUG logging (disabled)
    start = time.time()
    for i in range(iterations):
        x = i * 2
        logger.debug("test", value=x)
    disabled_time = time.time() - start
    
    # With INFO logging (enabled)
    start = time.time()
    for i in range(iterations):
        x = i * 2
        logger.info("test", value=x)
    enabled_time = time.time() - start
    
    print(f"Baseline: {baseline_time:.3f}s")
    print(f"Disabled logging: {disabled_time:.3f}s ({(disabled_time/baseline_time - 1)*100:.1f}% overhead)")
    print(f"Enabled logging: {enabled_time:.3f}s ({(enabled_time/baseline_time - 1)*100:.1f}% overhead)")


# =============================================================================
# BEST PRACTICES SUMMARY
# =============================================================================

"""
LOGGING BEST PRACTICES:

1. **Use appropriate log levels:**
   - DEBUG: Detailed diagnostic (< 1% of events)
   - INFO: Important milestones (key events)
   - WARNING: Abnormal conditions
   - ERROR: Error conditions

2. **Sample high-frequency events:**
   - Use LogSampler for 1-10% sampling
   - Use periodic logging (every N events)
   - Aggregate and log statistics

3. **Lazy evaluation:**
   - Check isEnabledFor() before expensive computations
   - Use lambda for deferred evaluation
   - Avoid string formatting unless logged

4. **Structured data:**
   - Use kwargs for structured fields
   - Avoid long strings, prefer JSON-serializable data
   - Include context (step, agent_id, simulation_id)

5. **Performance:**
   - Target < 1% overhead from logging
   - Batch database writes
   - Use time-based throttling for warnings

6. **Maintainability:**
   - Use consistent event names
   - Include units in metric names
   - Document expected log volume
   - Test with different log levels

7. **Analysis-friendly:**
   - Use structured format (JSON)
   - Include timestamps
   - Add correlation IDs (simulation_id)
   - Make logs grep-able and parseable
"""
