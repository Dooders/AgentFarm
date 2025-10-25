"""
Web-based dashboard for interactive simulation analysis.

This module provides a web dashboard for visualizing and interacting with
simulation analysis results.
"""

import json
import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import base64
from io import BytesIO

# Optional imports for web framework and visualization
try:
    from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import IntegrationOrchestrator
from farm.analysis.comparative.performance_optimizer import PerformanceOptimizer
from farm.analysis.comparative.reporting_system import ReportingSystem
from farm.analysis.comparative.api_endpoints import AnalysisAPIServer

logger = get_logger(__name__)


class WebDashboard:
    """Web-based dashboard for simulation analysis."""
    
    def __init__(self, 
                 api_server: Optional[AnalysisAPIServer] = None,
                 static_dir: Union[str, Path] = "dashboard_static",
                 templates_dir: Union[str, Path] = "dashboard_templates"):
        """Initialize the web dashboard."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for web dashboard functionality")
        
        self.api_server = api_server or AnalysisAPIServer()
        self.static_dir = Path(static_dir)
        self.templates_dir = Path(templates_dir)
        
        # Create directories
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Simulation Analysis Dashboard",
            description="Interactive web dashboard for simulation analysis",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup static files and templates
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Setup routes
        self._setup_routes()
        
        # Create default templates and static files
        self._create_default_assets()
        
        logger.info("WebDashboard initialized")
    
    def _setup_routes(self):
        """Setup dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "Simulation Analysis Dashboard",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.get("/analysis", response_class=HTMLResponse)
        async def analysis_page(request: Request):
            """Analysis management page."""
            return self.templates.TemplateResponse("analysis.html", {
                "request": request,
                "title": "Analysis Management"
            })
        
        @self.app.get("/reports", response_class=HTMLResponse)
        async def reports_page(request: Request):
            """Reports page."""
            return self.templates.TemplateResponse("reports.html", {
                "request": request,
                "title": "Analysis Reports"
            })
        
        @self.app.get("/visualizations", response_class=HTMLResponse)
        async def visualizations_page(request: Request):
            """Visualizations page."""
            return self.templates.TemplateResponse("visualizations.html", {
                "request": request,
                "title": "Data Visualizations"
            })
        
        @self.app.get("/system", response_class=HTMLResponse)
        async def system_page(request: Request):
            """System status page."""
            return self.templates.TemplateResponse("system.html", {
                "request": request,
                "title": "System Status"
            })
        
        @self.app.post("/api/analyze")
        async def start_analysis_api(
            request: Request,
            simulation_pairs: str = Form(...),
            analysis_config: str = Form("{}"),
            orchestration_config: str = Form("{}")
        ):
            """Start analysis via API."""
            try:
                # Parse form data
                pairs_data = json.loads(simulation_pairs)
                analysis_cfg = json.loads(analysis_config)
                orchestration_cfg = json.loads(orchestration_config)
                
                # Start analysis
                result = await self.api_server.start_analysis(
                    simulation_pairs=pairs_data,
                    analysis_config=analysis_cfg,
                    orchestration_config=orchestration_cfg
                )
                
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error starting analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/status/{analysis_id}")
        async def get_analysis_status_api(analysis_id: str):
            """Get analysis status via API."""
            try:
                status = await self.api_server.get_status(analysis_id)
                return JSONResponse(content=status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/results/{analysis_id}")
        async def get_analysis_results_api(analysis_id: str):
            """Get analysis results via API."""
            try:
                results = await self.api_server.get_results(analysis_id)
                return JSONResponse(content=results)
            except Exception as e:
                logger.error(f"Error getting results: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/report")
        async def generate_report_api(
            analysis_id: str = Form(...),
            format: str = Form("html"),
            include_charts: bool = Form(True),
            include_raw_data: bool = Form(False)
        ):
            """Generate report via API."""
            try:
                result = await self.api_server.generate_report(
                    analysis_id=analysis_id,
                    format=format,
                    include_charts=include_charts,
                    include_raw_data=include_raw_data
                )
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/visualizations/{analysis_id}")
        async def get_visualizations_api(analysis_id: str):
            """Get visualizations for analysis."""
            try:
                # Get analysis results
                results = await self.api_server.get_results(analysis_id)
                
                if "error" in results:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                # Generate visualizations
                visualizations = self._generate_visualizations(results)
                
                return JSONResponse(content=visualizations)
                
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/system/status")
        async def get_system_status_api():
            """Get system status via API."""
            try:
                status = await self.api_server.get_system_status()
                return JSONResponse(content=status)
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/analyses")
        async def list_analyses_api():
            """List all analyses via API."""
            try:
                analyses = await self.api_server.list_analyses()
                return JSONResponse(content=analyses)
            except Exception as e:
                logger.error(f"Error listing analyses: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations for analysis results."""
        visualizations = {}
        
        try:
            orchestration_result = analysis_results.get("orchestration_result")
            if not orchestration_result:
                return {"error": "No orchestration result available"}
            
            # Phase duration chart
            if "phase_results" in orchestration_result:
                visualizations["phase_duration"] = self._create_phase_duration_chart(
                    orchestration_result["phase_results"]
                )
            
            # Performance metrics
            if "performance_summary" in analysis_results:
                visualizations["performance"] = self._create_performance_charts(
                    analysis_results["performance_summary"]
                )
            
            # System resources
            if "system_resources" in analysis_results:
                visualizations["resources"] = self._create_resource_charts(
                    analysis_results["system_resources"]
                )
            
            # ML analysis results (if available)
            ml_results = self._extract_ml_results(orchestration_result)
            if ml_results:
                visualizations["ml_analysis"] = self._create_ml_visualizations(ml_results)
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _create_phase_duration_chart(self, phase_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create phase duration chart."""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available"}
        
        phases = [p.get("phase_name", "Unknown") for p in phase_results]
        durations = [p.get("duration", 0) for p in phase_results]
        successes = [p.get("success", False) for p in phase_results]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=phases,
                y=durations,
                marker_color=['green' if s else 'red' for s in successes],
                text=[f"{d:.2f}s" for d in durations],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Phase Duration Analysis",
            xaxis_title="Phase",
            yaxis_title="Duration (seconds)",
            showlegend=False
        )
        
        return {
            "type": "plotly",
            "data": json.dumps(fig, cls=PlotlyJSONEncoder)
        }
    
    def _create_performance_charts(self, performance_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance charts."""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available"}
        
        charts = {}
        
        # Operations duration chart
        if "operations" in performance_summary:
            operations = performance_summary["operations"]
            if operations:
                op_names = [op["name"] for op in operations]
                op_durations = [op["duration"] for op in operations]
                
                fig = go.Figure(data=[
                    go.Bar(x=op_names, y=op_durations)
                ])
                
                fig.update_layout(
                    title="Operation Duration Analysis",
                    xaxis_title="Operation",
                    yaxis_title="Duration (seconds)"
                )
                
                charts["operations"] = {
                    "type": "plotly",
                    "data": json.dumps(fig, cls=PlotlyJSONEncoder)
                }
        
        return charts
    
    def _create_resource_charts(self, system_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource usage charts."""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available"}
        
        charts = {}
        
        # Memory usage pie chart
        memory = system_resources.get("memory", {})
        if memory:
            labels = ["Used", "Available"]
            values = [memory.get("used_gb", 0), memory.get("available_gb", 0)]
            
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values)
            ])
            
            fig.update_layout(title="Memory Usage")
            
            charts["memory"] = {
                "type": "plotly",
                "data": json.dumps(fig, cls=PlotlyJSONEncoder)
            }
        
        # CPU usage gauge
        cpu_percent = system_resources.get("cpu_percent", 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Usage (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ))
        
        charts["cpu"] = {
            "type": "plotly",
            "data": json.dumps(fig, cls=PlotlyJSONEncoder)
        }
        
        return charts
    
    def _extract_ml_results(self, orchestration_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract ML analysis results from orchestration result."""
        # This would extract ML-specific results from the orchestration result
        # Implementation depends on the structure of the orchestration result
        return None
    
    def _create_ml_visualizations(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ML-specific visualizations."""
        # This would create visualizations for ML analysis results
        # Implementation depends on the ML results structure
        return {}
    
    def _create_default_assets(self):
        """Create default HTML templates and static assets."""
        
        # Create main dashboard template
        dashboard_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/plotly.js@2.14.0/dist/plotly.min.js"></script>
            <style>
                .dashboard-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }
                .metric-card { background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
                .chart-container { background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
                .status-badge { font-size: 0.8rem; padding: 0.25rem 0.5rem; }
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <div class="container">
                    <h1>{{ title }}</h1>
                    <p class="lead">Interactive simulation analysis and visualization platform</p>
                </div>
            </div>
            
            <div class="container mt-4">
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Active Analyses</h5>
                            <h2 id="active-analyses">-</h2>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Completed Analyses</h5>
                            <h2 id="completed-analyses">-</h2>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>System Status</h5>
                            <span id="system-status" class="badge status-badge">-</span>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Last Updated</h5>
                            <small id="last-updated">-</small>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="chart-container">
                            <h4>Recent Analysis Activity</h4>
                            <div id="activity-chart"></div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>System Resources</h4>
                            <div id="resources-chart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>Analysis Performance</h4>
                            <div id="performance-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Dashboard JavaScript functionality
                async function updateDashboard() {
                    try {
                        const response = await fetch('/api/system/status');
                        const data = await response.json();
                        
                        document.getElementById('active-analyses').textContent = data.active_analyses || 0;
                        document.getElementById('completed-analyses').textContent = data.completed_analyses || 0;
                        document.getElementById('system-status').textContent = data.system_status?.status || 'Unknown';
                        document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                        
                        // Update charts if data is available
                        if (data.performance_summary) {
                            updatePerformanceChart(data.performance_summary);
                        }
                        
                        if (data.system_resources) {
                            updateResourcesChart(data.system_resources);
                        }
                        
                    } catch (error) {
                        console.error('Error updating dashboard:', error);
                    }
                }
                
                function updatePerformanceChart(performanceData) {
                    // Implementation for performance chart
                    console.log('Updating performance chart:', performanceData);
                }
                
                function updateResourcesChart(resourceData) {
                    // Implementation for resources chart
                    console.log('Updating resources chart:', resourceData);
                }
                
                // Update dashboard every 5 seconds
                setInterval(updateDashboard, 5000);
                
                // Initial load
                updateDashboard();
            </script>
        </body>
        </html>
        """
        
        # Save dashboard template
        dashboard_file = self.templates_dir / "dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        # Create other templates (simplified versions)
        templates = {
            "analysis.html": """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>{{ title }}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <h1>{{ title }}</h1>
                    <p>Analysis management interface coming soon...</p>
                </div>
            </body>
            </html>
            """,
            "reports.html": """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>{{ title }}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <h1>{{ title }}</h1>
                    <p>Reports interface coming soon...</p>
                </div>
            </body>
            </html>
            """,
            "visualizations.html": """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>{{ title }}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/plotly.js@2.14.0/dist/plotly.min.js"></script>
            </head>
            <body>
                <div class="container mt-4">
                    <h1>{{ title }}</h1>
                    <p>Visualizations interface coming soon...</p>
                </div>
            </body>
            </html>
            """,
            "system.html": """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>{{ title }}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <h1>{{ title }}</h1>
                    <p>System status interface coming soon...</p>
                </div>
            </body>
            </html>
            """
        }
        
        for filename, content in templates.items():
            template_file = self.templates_dir / filename
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info("Default dashboard assets created")
    
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """Run the web dashboard."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create and run the dashboard
    dashboard = WebDashboard()
    dashboard.run_dashboard(host="0.0.0.0", port=8080)