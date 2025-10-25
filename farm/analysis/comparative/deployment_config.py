"""
Deployment and configuration management for simulation analysis system.

This module provides deployment configurations, environment management,
and production-ready setup for the analysis system.
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import shutil
import subprocess
import sys

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for different environments."""
    
    name: str
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    cache_enabled: bool = True
    cache_size_gb: float = 1.0
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8080
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30  # seconds
    auto_restart: bool = True
    max_restarts: int = 3
    restart_delay: int = 5  # seconds


@dataclass
class DockerConfig:
    """Docker configuration for containerized deployment."""
    
    base_image: str = "python:3.9-slim"
    maintainer: str = "Analysis System"
    working_dir: str = "/app"
    expose_ports: List[int] = field(default_factory=lambda: [8000, 8080, 9090])
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=lambda: ["/app/data", "/app/logs", "/app/cache"])
    healthcheck_cmd: str = "curl -f http://localhost:8000/health || exit 1"
    healthcheck_interval: int = 30
    healthcheck_timeout: int = 10
    healthcheck_retries: int = 3


@dataclass
class KubernetesConfig:
    """Kubernetes configuration for cluster deployment."""
    
    namespace: str = "simulation-analysis"
    replicas: int = 2
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    storage_class: str = "standard"
    storage_size: str = "10Gi"
    service_type: str = "ClusterIP"
    ingress_enabled: bool = True
    ingress_host: str = "analysis.local"
    tls_secret: Optional[str] = None
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    log_aggregation: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_retention_days: int = 30
    alerting_enabled: bool = True
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)


class DeploymentManager:
    """Manages deployment configurations and environments."""
    
    def __init__(self, config_dir: Union[str, Path] = "deployment_configs"):
        """Initialize the deployment manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default environments
        self.environments = {
            "development": EnvironmentConfig(
                name="development",
                debug=True,
                log_level="DEBUG",
                max_workers=2,
                memory_limit_gb=4.0,
                cache_size_gb=0.5,
                api_port=8000,
                dashboard_port=8080,
                monitoring_enabled=False
            ),
            "staging": EnvironmentConfig(
                name="staging",
                debug=False,
                log_level="INFO",
                max_workers=4,
                memory_limit_gb=8.0,
                cache_size_gb=2.0,
                api_port=8000,
                dashboard_port=8080,
                monitoring_enabled=True
            ),
            "production": EnvironmentConfig(
                name="production",
                debug=False,
                log_level="WARNING",
                max_workers=8,
                memory_limit_gb=16.0,
                cache_size_gb=4.0,
                api_port=8000,
                dashboard_port=8080,
                ssl_enabled=True,
                monitoring_enabled=True,
                rate_limiting=True,
                auto_restart=True
            )
        }
        
        logger.info("DeploymentManager initialized")
    
    def create_environment_config(self, env_name: str, config: EnvironmentConfig) -> Path:
        """Create environment configuration file."""
        config_file = self.config_dir / f"{env_name}.yaml"
        
        config_dict = {
            "environment": env_name,
            "config": {
                "debug": config.debug,
                "log_level": config.log_level,
                "max_workers": config.max_workers,
                "memory_limit_gb": config.memory_limit_gb,
                "cache_enabled": config.cache_enabled,
                "cache_size_gb": config.cache_size_gb,
                "database_url": config.database_url,
                "redis_url": config.redis_url,
                "api_host": config.api_host,
                "api_port": config.api_port,
                "dashboard_port": config.dashboard_port,
                "ssl_enabled": config.ssl_enabled,
                "ssl_cert_path": config.ssl_cert_path,
                "ssl_key_path": config.ssl_key_path,
                "cors_origins": config.cors_origins,
                "rate_limiting": config.rate_limiting,
                "rate_limit_requests": config.rate_limit_requests,
                "rate_limit_window": config.rate_limit_window,
                "monitoring_enabled": config.monitoring_enabled,
                "metrics_port": config.metrics_port,
                "health_check_interval": config.health_check_interval,
                "auto_restart": config.auto_restart,
                "max_restarts": config.max_restarts,
                "restart_delay": config.restart_delay
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Environment config created: {config_file}")
        return config_file
    
    def create_docker_config(self, docker_config: DockerConfig) -> Path:
        """Create Docker configuration files."""
        
        # Dockerfile
        dockerfile_content = f"""
FROM {docker_config.base_image}

LABEL maintainer="{docker_config.maintainer}"

WORKDIR {docker_config.working_dir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache

# Set environment variables
"""
        
        for key, value in docker_config.environment_vars.items():
            dockerfile_content += f"ENV {key}={value}\n"
        
        dockerfile_content += f"""
# Expose ports
"""
        for port in docker_config.expose_ports:
            dockerfile_content += f"EXPOSE {port}\n"
        
        dockerfile_content += f"""
# Health check
HEALTHCHECK --interval={docker_config.healthcheck_interval}s \\
    --timeout={docker_config.healthcheck_timeout}s \\
    --start-period=5s \\
    --retries={docker_config.healthcheck_retries} \\
    CMD {docker_config.healthcheck_cmd}

# Default command
CMD ["python", "-m", "farm.analysis.comparative.api_endpoints"]
"""
        
        dockerfile_path = self.config_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # docker-compose.yml
        compose_content = f"""
version: '3.8'

services:
  analysis-api:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./cache:/app/cache
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
"""
        
        compose_path = self.config_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        # .dockerignore
        dockerignore_content = """
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

.DS_Store
.vscode
.idea
*.swp
*.swo
*~

# Project specific
tests/
docs/
*.md
.gitignore
README.md
"""
        
        dockerignore_path = self.config_dir / ".dockerignore"
        with open(dockerignore_path, 'w') as f:
            f.write(dockerignore_content)
        
        logger.info("Docker configuration created")
        return dockerfile_path
    
    def create_kubernetes_config(self, k8s_config: KubernetesConfig) -> Path:
        """Create Kubernetes configuration files."""
        k8s_dir = self.config_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        # Namespace
        namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {k8s_config.namespace}
  labels:
    name: {k8s_config.namespace}
"""
        
        with open(k8s_dir / "namespace.yaml", 'w') as f:
            f.write(namespace_yaml)
        
        # Deployment
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simulation-analysis
  namespace: {k8s_config.namespace}
  labels:
    app: simulation-analysis
spec:
  replicas: {k8s_config.replicas}
  selector:
    matchLabels:
      app: simulation-analysis
  template:
    metadata:
      labels:
        app: simulation-analysis
    spec:
      containers:
      - name: analysis-api
        image: simulation-analysis:latest
        ports:
        - containerPort: 8000
        - containerPort: 8080
        - containerPort: 9090
        resources:
          requests:
            cpu: {k8s_config.cpu_request}
            memory: {k8s_config.memory_request}
          limits:
            cpu: {k8s_config.cpu_limit}
            memory: {k8s_config.memory_limit}
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        - name: cache-volume
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: analysis-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: analysis-logs-pvc
      - name: cache-volume
        persistentVolumeClaim:
          claimName: analysis-cache-pvc
"""
        
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_yaml)
        
        # Service
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: simulation-analysis-service
  namespace: {k8s_config.namespace}
spec:
  selector:
    app: simulation-analysis
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: dashboard
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: {k8s_config.service_type}
"""
        
        with open(k8s_dir / "service.yaml", 'w') as f:
            f.write(service_yaml)
        
        # PersistentVolumeClaim
        pvc_yaml = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: analysis-data-pvc
  namespace: {k8s_config.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {k8s_config.storage_size}
  storageClassName: {k8s_config.storage_class}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: analysis-logs-pvc
  namespace: {k8s_config.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: {k8s_config.storage_class}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: analysis-cache-pvc
  namespace: {k8s_config.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  storageClassName: {k8s_config.storage_class}
"""
        
        with open(k8s_dir / "pvc.yaml", 'w') as f:
            f.write(pvc_yaml)
        
        # Ingress (if enabled)
        if k8s_config.ingress_enabled:
            ingress_yaml = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: simulation-analysis-ingress
  namespace: {k8s_config.namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  rules:
  - host: {k8s_config.ingress_host}
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: simulation-analysis-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: simulation-analysis-service
            port:
              number: 8080
"""
            
            if k8s_config.tls_secret:
                ingress_yaml += f"""
  tls:
  - hosts:
    - {k8s_config.ingress_host}
    secretName: {k8s_config.tls_secret}
"""
            
            with open(k8s_dir / "ingress.yaml", 'w') as f:
                f.write(ingress_yaml)
        
        logger.info("Kubernetes configuration created")
        return k8s_dir
    
    def create_monitoring_config(self, monitoring_config: MonitoringConfig) -> Path:
        """Create monitoring configuration files."""
        monitoring_dir = self.config_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        if monitoring_config.prometheus_enabled:
            prometheus_yml = f"""
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'simulation-analysis'
    static_configs:
      - targets: ['analysis-api:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
"""
            
            with open(monitoring_dir / "prometheus.yml", 'w') as f:
                f.write(prometheus_yml)
            
            # Alert rules
            alert_rules_yml = f"""
groups:
- name: simulation-analysis
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{{status=~"5.."}}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: (process_resident_memory_bytes / 1024 / 1024 / 1024) > {monitoring_config.metrics_retention_days}
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }} GB"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "Service {{ $labels.instance }} is down"
"""
            
            with open(monitoring_dir / "alert_rules.yml", 'w') as f:
                f.write(alert_rules_yml)
        
        # Grafana configuration
        if monitoring_config.grafana_enabled:
            grafana_dir = monitoring_dir / "grafana"
            grafana_dir.mkdir(exist_ok=True)
            
            # Datasource configuration
            datasources_dir = grafana_dir / "datasources"
            datasources_dir.mkdir(exist_ok=True)
            
            datasource_yml = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""
            
            with open(datasources_dir / "prometheus.yml", 'w') as f:
                f.write(datasource_yml)
            
            # Dashboard configuration
            dashboards_dir = grafana_dir / "dashboards"
            dashboards_dir.mkdir(exist_ok=True)
            
            dashboard_yml = """
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
"""
            
            with open(dashboards_dir / "dashboard.yml", 'w') as f:
                f.write(dashboard_yml)
        
        logger.info("Monitoring configuration created")
        return monitoring_dir
    
    def create_startup_scripts(self) -> Path:
        """Create startup and management scripts."""
        scripts_dir = self.config_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Start script
        start_script = """#!/bin/bash
# Simulation Analysis System Startup Script

set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-development}
CONFIG_FILE="deployment_configs/${ENVIRONMENT}.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | xargs)

# Start the API server
echo "Starting Simulation Analysis API Server..."
python -m farm.analysis.comparative.api_endpoints --host $API_HOST --port $API_PORT &

# Start the dashboard
echo "Starting Simulation Analysis Dashboard..."
python -m farm.analysis.comparative.web_dashboard --host $API_HOST --port $DASHBOARD_PORT &

# Wait for processes
wait
"""
        
        with open(scripts_dir / "start.sh", 'w') as f:
            f.write(start_script)
        
        # Make executable
        os.chmod(scripts_dir / "start.sh", 0o755)
        
        # Stop script
        stop_script = """#!/bin/bash
# Simulation Analysis System Stop Script

echo "Stopping Simulation Analysis System..."

# Kill API server
pkill -f "farm.analysis.comparative.api_endpoints" || true

# Kill dashboard
pkill -f "farm.analysis.comparative.web_dashboard" || true

echo "System stopped"
"""
        
        with open(scripts_dir / "stop.sh", 'w') as f:
            f.write(stop_script)
        
        os.chmod(scripts_dir / "stop.sh", 0o755)
        
        # Health check script
        health_script = """#!/bin/bash
# Health check script

API_URL=${API_URL:-http://localhost:8000}
DASHBOARD_URL=${DASHBOARD_URL:-http://localhost:8080}

# Check API health
echo "Checking API health..."
if curl -f -s "$API_URL/health" > /dev/null; then
    echo "API is healthy"
else
    echo "API is unhealthy"
    exit 1
fi

# Check dashboard
echo "Checking dashboard..."
if curl -f -s "$DASHBOARD_URL/" > /dev/null; then
    echo "Dashboard is healthy"
else
    echo "Dashboard is unhealthy"
    exit 1
fi

echo "All services are healthy"
"""
        
        with open(scripts_dir / "health_check.sh", 'w') as f:
            f.write(health_script)
        
        os.chmod(scripts_dir / "health_check.sh", 0o755)
        
        logger.info("Startup scripts created")
        return scripts_dir
    
    def generate_requirements(self) -> Path:
        """Generate requirements.txt for deployment."""
        requirements_content = """
# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Analysis dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Optional ML dependencies
xgboost>=1.7.0
lightgbm>=4.0.0

# Database dependencies
sqlalchemy>=2.0.0
alembic>=1.12.0

# Caching
redis>=5.0.0

# Monitoring
prometheus-client>=0.17.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
jinja2>=3.1.0
aiofiles>=23.0.0
aiohttp>=3.8.0

# Development dependencies (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
"""
        
        requirements_path = self.config_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        logger.info("Requirements file generated")
        return requirements_path
    
    def create_all_configs(self) -> Dict[str, Path]:
        """Create all deployment configurations."""
        configs = {}
        
        # Create environment configs
        for env_name, env_config in self.environments.items():
            configs[f"{env_name}_env"] = self.create_environment_config(env_name, env_config)
        
        # Create Docker config
        docker_config = DockerConfig()
        configs["docker"] = self.create_docker_config(docker_config)
        
        # Create Kubernetes config
        k8s_config = KubernetesConfig()
        configs["kubernetes"] = self.create_kubernetes_config(k8s_config)
        
        # Create monitoring config
        monitoring_config = MonitoringConfig()
        configs["monitoring"] = self.create_monitoring_config(monitoring_config)
        
        # Create startup scripts
        configs["scripts"] = self.create_startup_scripts()
        
        # Generate requirements
        configs["requirements"] = self.generate_requirements()
        
        logger.info("All deployment configurations created")
        return configs
    
    def deploy_environment(self, environment: str, method: str = "local") -> bool:
        """Deploy to specified environment using specified method."""
        try:
            if method == "local":
                return self._deploy_local(environment)
            elif method == "docker":
                return self._deploy_docker(environment)
            elif method == "kubernetes":
                return self._deploy_kubernetes(environment)
            else:
                logger.error(f"Unknown deployment method: {method}")
                return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _deploy_local(self, environment: str) -> bool:
        """Deploy locally."""
        logger.info(f"Deploying locally for environment: {environment}")
        
        # Load environment config
        config_file = self.config_dir / f"{environment}.yaml"
        if not config_file.exists():
            logger.error(f"Environment config not found: {config_file}")
            return False
        
        # Start services using startup script
        start_script = self.config_dir / "scripts" / "start.sh"
        if start_script.exists():
            result = subprocess.run([str(start_script)], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Local deployment successful")
                return True
            else:
                logger.error(f"Local deployment failed: {result.stderr}")
                return False
        else:
            logger.error("Startup script not found")
            return False
    
    def _deploy_docker(self, environment: str) -> bool:
        """Deploy using Docker."""
        logger.info(f"Deploying with Docker for environment: {environment}")
        
        # Build Docker image
        dockerfile = self.config_dir / "Dockerfile"
        if not dockerfile.exists():
            logger.error("Dockerfile not found")
            return False
        
        build_result = subprocess.run([
            "docker", "build", "-t", "simulation-analysis", "-f", str(dockerfile), "."
        ], capture_output=True, text=True)
        
        if build_result.returncode != 0:
            logger.error(f"Docker build failed: {build_result.stderr}")
            return False
        
        # Start with docker-compose
        compose_file = self.config_dir / "docker-compose.yml"
        if compose_file.exists():
            result = subprocess.run([
                "docker-compose", "-f", str(compose_file), "up", "-d"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Docker deployment successful")
                return True
            else:
                logger.error(f"Docker deployment failed: {result.stderr}")
                return False
        else:
            logger.error("docker-compose.yml not found")
            return False
    
    def _deploy_kubernetes(self, environment: str) -> bool:
        """Deploy to Kubernetes."""
        logger.info(f"Deploying to Kubernetes for environment: {environment}")
        
        k8s_dir = self.config_dir / "kubernetes"
        if not k8s_dir.exists():
            logger.error("Kubernetes config directory not found")
            return False
        
        # Apply Kubernetes manifests
        manifest_files = [
            "namespace.yaml",
            "pvc.yaml",
            "deployment.yaml",
            "service.yaml",
            "ingress.yaml"
        ]
        
        for manifest in manifest_files:
            manifest_path = k8s_dir / manifest
            if manifest_path.exists():
                result = subprocess.run([
                    "kubectl", "apply", "-f", str(manifest_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to apply {manifest}: {result.stderr}")
                    return False
        
        logger.info("Kubernetes deployment successful")
        return True


# Example usage
if __name__ == "__main__":
    # Create deployment manager
    deployment_manager = DeploymentManager()
    
    # Create all configurations
    configs = deployment_manager.create_all_configs()
    
    print("Deployment configurations created:")
    for name, path in configs.items():
        print(f"  {name}: {path}")
    
    # Deploy to development environment
    success = deployment_manager.deploy_environment("development", "local")
    print(f"Development deployment: {'Success' if success else 'Failed'}")