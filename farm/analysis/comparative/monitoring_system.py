"""
Monitoring and alerting system for simulation analysis.

This module provides comprehensive monitoring, alerting, and observability
capabilities for the analysis system.
"""

import time
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import psutil
import asyncio
from collections import deque, defaultdict

# Optional imports for monitoring
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from farm.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule definition."""
    
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    cooldown: int = 300  # seconds
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricData:
    """Metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int
    thread_count: int


class MonitoringSystem:
    """Comprehensive monitoring and alerting system."""
    
    def __init__(self, 
                 prometheus_port: int = 9090,
                 redis_url: Optional[str] = None,
                 metrics_retention_days: int = 30,
                 alert_cooldown: int = 300):
        """Initialize the monitoring system."""
        self.prometheus_port = prometheus_port
        self.redis_url = redis_url
        self.metrics_retention_days = metrics_retention_days
        self.alert_cooldown = alert_cooldown
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.system_metrics_history: deque = deque(maxlen=1000)
        
        # Alerting
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None
        
        # Prometheus metrics
        self.prometheus_metrics = {}
        self.registry = None
        
        # Redis connection
        self.redis_client = None
        
        # Initialize components
        self._initialize_prometheus()
        self._initialize_redis()
        self._setup_default_alert_rules()
        
        logger.info("MonitoringSystem initialized")
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics will be disabled")
            return
        
        try:
            self.registry = CollectorRegistry()
            
            # Analysis metrics
            self.prometheus_metrics = {
                'analysis_requests_total': Counter(
                    'analysis_requests_total',
                    'Total number of analysis requests',
                    ['status', 'phase'],
                    registry=self.registry
                ),
                'analysis_duration_seconds': Histogram(
                    'analysis_duration_seconds',
                    'Analysis duration in seconds',
                    ['phase'],
                    registry=self.registry
                ),
                'analysis_errors_total': Counter(
                    'analysis_errors_total',
                    'Total number of analysis errors',
                    ['error_type', 'phase'],
                    registry=self.registry
                ),
                'active_analyses': Gauge(
                    'active_analyses',
                    'Number of currently active analyses',
                    registry=self.registry
                ),
                'system_cpu_percent': Gauge(
                    'system_cpu_percent',
                    'System CPU usage percentage',
                    registry=self.registry
                ),
                'system_memory_percent': Gauge(
                    'system_memory_percent',
                    'System memory usage percentage',
                    registry=self.registry
                ),
                'system_disk_percent': Gauge(
                    'system_disk_percent',
                    'System disk usage percentage',
                    registry=self.registry
                ),
                'alert_count': Gauge(
                    'alert_count',
                    'Number of active alerts',
                    ['severity'],
                    registry=self.registry
                )
            }
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection for distributed monitoring."""
        if not REDIS_AVAILABLE or not self.redis_url:
            logger.warning("Redis not available - distributed monitoring disabled")
            return
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        
        # High CPU usage
        self.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            description="CPU usage is above 80%",
            condition=lambda metrics: metrics.get('cpu_percent', 0) > 80,
            severity=AlertSeverity.WARNING,
            cooldown=300,
            tags=["system", "cpu"]
        ))
        
        # High memory usage
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            description="Memory usage is above 85%",
            condition=lambda metrics: metrics.get('memory_percent', 0) > 85,
            severity=AlertSeverity.WARNING,
            cooldown=300,
            tags=["system", "memory"]
        ))
        
        # Critical memory usage
        self.add_alert_rule(AlertRule(
            name="critical_memory_usage",
            description="Memory usage is above 95%",
            condition=lambda metrics: metrics.get('memory_percent', 0) > 95,
            severity=AlertSeverity.CRITICAL,
            cooldown=60,
            tags=["system", "memory", "critical"]
        ))
        
        # High disk usage
        self.add_alert_rule(AlertRule(
            name="high_disk_usage",
            description="Disk usage is above 90%",
            condition=lambda metrics: metrics.get('disk_usage_percent', 0) > 90,
            severity=AlertSeverity.WARNING,
            cooldown=600,
            tags=["system", "disk"]
        ))
        
        # Analysis failure rate
        self.add_alert_rule(AlertRule(
            name="high_analysis_failure_rate",
            description="Analysis failure rate is above 10%",
            condition=lambda metrics: metrics.get('analysis_failure_rate', 0) > 0.1,
            severity=AlertSeverity.ERROR,
            cooldown=300,
            tags=["analysis", "failure"]
        ))
        
        # Long running analysis
        self.add_alert_rule(AlertRule(
            name="long_running_analysis",
            description="Analysis has been running for more than 1 hour",
            condition=lambda metrics: metrics.get('max_analysis_duration', 0) > 3600,
            severity=AlertSeverity.WARNING,
            cooldown=600,
            tags=["analysis", "performance"]
        ))
        
        logger.info("Default alert rules configured")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        
        # Start system metrics collection
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        # Start alert processing
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
        
        # Start Prometheus server
        if PROMETHEUS_AVAILABLE and self.registry:
            try:
                start_http_server(self.prometheus_port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        logger.info("Monitoring system stopped")
    
    def _monitor_system(self):
        """Monitor system resources."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Store in Redis if available
                if self.redis_client:
                    self._store_metrics_redis(metrics)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(30)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # Load average
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # Process count
        process_count = len(psutil.pids())
        
        # Thread count
        thread_count = threading.active_count()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            memory_available_mb=memory.available / (1024**2),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_sent_mb=network.bytes_sent / (1024**2),
            network_recv_mb=network.bytes_recv / (1024**2),
            load_average=list(load_avg),
            process_count=process_count,
            thread_count=thread_count
        )
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics."""
        if not self.prometheus_metrics:
            return
        
        try:
            self.prometheus_metrics['system_cpu_percent'].set(metrics.cpu_percent)
            self.prometheus_metrics['system_memory_percent'].set(metrics.memory_percent)
            self.prometheus_metrics['system_disk_percent'].set(metrics.disk_usage_percent)
            
            # Update alert counts
            for severity in AlertSeverity:
                count = sum(1 for alert in self.active_alerts.values() 
                           if alert.severity == severity and alert.status == AlertStatus.ACTIVE)
                self.prometheus_metrics['alert_count'].labels(severity=severity.value).set(count)
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _store_metrics_redis(self, metrics: SystemMetrics):
        """Store metrics in Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"metrics:system:{int(metrics.timestamp.timestamp())}"
            data = {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'timestamp': metrics.timestamp.isoformat()
            }
            
            self.redis_client.setex(key, 86400 * self.metrics_retention_days, json.dumps(data))
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    def _process_alerts(self):
        """Process alert rules and generate alerts."""
        while self.monitoring_active:
            try:
                # Get latest system metrics
                if self.system_metrics_history:
                    latest_metrics = self.system_metrics_history[-1]
                    metrics_dict = {
                        'cpu_percent': latest_metrics.cpu_percent,
                        'memory_percent': latest_metrics.memory_percent,
                        'disk_usage_percent': latest_metrics.disk_usage_percent,
                        'process_count': latest_metrics.process_count,
                        'thread_count': latest_metrics.thread_count
                    }
                    
                    # Check alert rules
                    for rule_name, rule in self.alert_rules.items():
                        if not rule.enabled:
                            continue
                        
                        # Check cooldown
                        last_alert = self.last_alert_time.get(rule_name)
                        if last_alert and (datetime.now() - last_alert).total_seconds() < rule.cooldown:
                            continue
                        
                        # Check condition
                        try:
                            if rule.condition(metrics_dict):
                                self._trigger_alert(rule, metrics_dict)
                        except Exception as e:
                            logger.error(f"Error evaluating alert rule {rule_name}: {e}")
                
                time.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                time.sleep(60)
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        # Check if alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if (alert.rule_name == rule.name and 
                alert.status == AlertStatus.ACTIVE and
                (datetime.now() - alert.created_at).total_seconds() < rule.cooldown):
                existing_alert = alert
                break
        
        if existing_alert:
            return  # Alert already active
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description} (Current: {metrics})",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'metrics': metrics,
                'rule_metadata': rule.metadata
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[rule.name] = datetime.now()
        
        # Log alert
        logger.warning(f"ALERT TRIGGERED: {rule.name} - {alert.message}")
        
        # Send notification (placeholder)
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification (placeholder for actual implementation)."""
        # This would integrate with actual notification systems like:
        # - Email (SMTP)
        # - Slack webhooks
        # - PagerDuty
        # - Custom webhooks
        
        notification_data = {
            'alert_id': alert.id,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value,
            'message': alert.message,
            'timestamp': alert.created_at.isoformat(),
            'metadata': alert.metadata
        }
        
        # Store in Redis for distributed systems
        if self.redis_client:
            try:
                key = f"alerts:active:{alert.id}"
                self.redis_client.setex(key, 86400, json.dumps(notification_data))
            except Exception as e:
                logger.error(f"Error storing alert in Redis: {e}")
        
        logger.info(f"Alert notification sent: {alert.id}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Alert rule removed: {rule_name}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            alert.updated_at = datetime.now()
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()
            
            logger.info(f"Alert resolved: {alert_id}")
    
    def record_analysis_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record an analysis metric."""
        metric_data = MetricData(
            name=metric_name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        self.metrics_history.append(metric_data)
        
        # Update Prometheus metrics
        if metric_name in self.prometheus_metrics:
            if hasattr(self.prometheus_metrics[metric_name], 'labels'):
                self.prometheus_metrics[metric_name].labels(**labels).inc(value)
            else:
                self.prometheus_metrics[metric_name].inc(value)
    
    def record_analysis_duration(self, phase: str, duration: float):
        """Record analysis phase duration."""
        self.record_analysis_metric('analysis_duration_seconds', duration, {'phase': phase})
        
        if self.prometheus_metrics.get('analysis_duration_seconds'):
            self.prometheus_metrics['analysis_duration_seconds'].labels(phase=phase).observe(duration)
    
    def record_analysis_request(self, status: str, phase: str):
        """Record analysis request."""
        self.record_analysis_metric('analysis_requests_total', 1, {'status': status, 'phase': phase})
        
        if self.prometheus_metrics.get('analysis_requests_total'):
            self.prometheus_metrics['analysis_requests_total'].labels(status=status, phase=phase).inc()
    
    def record_analysis_error(self, error_type: str, phase: str):
        """Record analysis error."""
        self.record_analysis_metric('analysis_errors_total', 1, {'error_type': error_type, 'phase': phase})
        
        if self.prometheus_metrics.get('analysis_errors_total'):
            self.prometheus_metrics['analysis_errors_total'].labels(error_type=error_type, phase=phase).inc()
    
    def set_active_analyses_count(self, count: int):
        """Set the number of active analyses."""
        if self.prometheus_metrics.get('active_analyses'):
            self.prometheus_metrics['active_analyses'].set(count)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.system_metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.system_metrics_history[-1]
        
        return {
            "status": "healthy",
            "timestamp": latest_metrics.timestamp.isoformat(),
            "cpu_percent": latest_metrics.cpu_percent,
            "memory_percent": latest_metrics.memory_percent,
            "disk_usage_percent": latest_metrics.disk_usage_percent,
            "process_count": latest_metrics.process_count,
            "thread_count": latest_metrics.thread_count,
            "active_alerts": len([a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]),
            "monitoring_active": self.monitoring_active
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]
        
        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {
                severity.value: len([a for a in active_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "recent_alerts": [
                {
                    "id": alert.id,
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in sorted(active_alerts, key=lambda x: x.created_at, reverse=True)[:10]
            ]
        }
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        recent_system_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_system_metrics:
            return {"status": "no_data"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics)
        avg_memory = sum(m.memory_percent for m in recent_system_metrics) / len(recent_system_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_system_metrics) / len(recent_system_metrics)
        
        return {
            "period_hours": hours,
            "data_points": len(recent_system_metrics),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_usage_percent": round(avg_disk, 2)
            },
            "current": {
                "cpu_percent": recent_system_metrics[-1].cpu_percent,
                "memory_percent": recent_system_metrics[-1].memory_percent,
                "disk_usage_percent": recent_system_metrics[-1].disk_usage_percent
            },
            "analysis_metrics": len(recent_metrics)
        }
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export metrics in specified format."""
        if format == "json":
            return {
                "system_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "disk_usage_percent": m.disk_usage_percent,
                        "process_count": m.process_count,
                        "thread_count": m.thread_count
                    }
                    for m in self.system_metrics_history
                ],
                "analysis_metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    }
                    for m in self.metrics_history
                ],
                "alerts": [
                    {
                        "id": alert.id,
                        "rule_name": alert.rule_name,
                        "severity": alert.severity.value,
                        "status": alert.status.value,
                        "message": alert.message,
                        "created_at": alert.created_at.isoformat(),
                        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                    }
                    for alert in self.alert_history
                ]
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage
if __name__ == "__main__":
    # Create monitoring system
    monitoring = MonitoringSystem()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    try:
        # Simulate some activity
        monitoring.record_analysis_request("success", "basic_analysis")
        monitoring.record_analysis_duration("basic_analysis", 5.2)
        monitoring.set_active_analyses_count(3)
        
        # Let it run for a bit
        time.sleep(60)
        
        # Get status
        status = monitoring.get_system_status()
        print("System Status:", json.dumps(status, indent=2))
        
        # Get alert summary
        alerts = monitoring.get_alert_summary()
        print("Alert Summary:", json.dumps(alerts, indent=2))
        
    finally:
        # Stop monitoring
        monitoring.stop_monitoring()