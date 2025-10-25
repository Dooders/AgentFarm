"""
REST API endpoints for simulation analysis services.

This module provides REST API endpoints for accessing analysis functionality
through HTTP requests.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import traceback

# Optional imports for web framework
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import IntegrationOrchestrator, OrchestrationConfig
from farm.analysis.comparative.performance_optimizer import PerformanceOptimizer, PerformanceConfig
from farm.analysis.comparative.reporting_system import ReportingSystem, ReportConfig

logger = get_logger(__name__)

# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    
    class AnalysisRequest(BaseModel):
        """Request model for analysis operations."""
        simulation_pairs: List[List[str]] = Field(..., description="List of simulation path pairs to analyze")
        analysis_config: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration")
        orchestration_config: Optional[Dict[str, Any]] = Field(None, description="Orchestration configuration")
        performance_config: Optional[Dict[str, Any]] = Field(None, description="Performance configuration")
        report_config: Optional[Dict[str, Any]] = Field(None, description="Report configuration")
    
    class AnalysisResponse(BaseModel):
        """Response model for analysis operations."""
        success: bool
        analysis_id: str
        status: str
        message: str
        results: Optional[Dict[str, Any]] = None
        errors: List[str] = []
        warnings: List[str] = []
        created_at: datetime
        completed_at: Optional[datetime] = None
        duration: Optional[float] = None
    
    class StatusResponse(BaseModel):
        """Response model for status checks."""
        analysis_id: str
        status: str
        progress: float
        current_phase: Optional[str] = None
        completed_phases: List[str] = []
        errors: List[str] = []
        warnings: List[str] = []
        created_at: datetime
        updated_at: datetime
    
    class ReportRequest(BaseModel):
        """Request model for report generation."""
        analysis_id: str
        format: str = Field("html", description="Report format (html, json, markdown)")
        include_charts: bool = Field(True, description="Include charts in report")
        include_raw_data: bool = Field(False, description="Include raw data in report")
    
    class ReportResponse(BaseModel):
        """Response model for report generation."""
        success: bool
        report_id: str
        report_url: str
        format: str
        file_size_mb: float
        created_at: datetime


class AnalysisAPIServer:
    """REST API server for simulation analysis services."""
    
    def __init__(self, 
                 orchestration_config: Optional[OrchestrationConfig] = None,
                 performance_config: Optional[PerformanceConfig] = None,
                 report_config: Optional[ReportConfig] = None):
        """Initialize the API server."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API server functionality")
        
        self.app = FastAPI(
            title="Simulation Analysis API",
            description="REST API for simulation comparison analysis",
            version="1.0.0"
        )
        
        # Initialize components
        self.orchestrator = IntegrationOrchestrator(orchestration_config)
        self.performance_optimizer = PerformanceOptimizer(performance_config)
        self.reporting_system = ReportingSystem(report_config)
        
        # Analysis tracking
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # Security
        self.security = HTTPBearer(auto_error=False)
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info("AnalysisAPIServer initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Simulation Analysis API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "health": "/health",
                    "analyze": "/analyze",
                    "status": "/status/{analysis_id}",
                    "results": "/results/{analysis_id}",
                    "report": "/report",
                    "reports": "/reports",
                    "system": "/system/status"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                system_status = self.orchestrator.get_analysis_status()
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "components": system_status
                }
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"status": "unhealthy", "error": str(e)}
                )
        
        @self.app.post("/analyze", response_model=AnalysisResponse)
        async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
            """Start a new analysis."""
            try:
                analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # Validate simulation pairs
                if not request.simulation_pairs:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="At least one simulation pair is required"
                    )
                
                # Check if simulation paths exist
                for pair in request.simulation_pairs:
                    if len(pair) != 2:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Each simulation pair must contain exactly 2 paths"
                        )
                    
                    for path in pair:
                        if not Path(path).exists():
                            raise HTTPException(
                                status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Simulation path not found: {path}"
                            )
                
                # Initialize analysis tracking
                self.active_analyses[analysis_id] = {
                    "status": "started",
                    "progress": 0.0,
                    "current_phase": None,
                    "completed_phases": [],
                    "errors": [],
                    "warnings": [],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "request": request.dict()
                }
                
                # Start analysis in background
                background_tasks.add_task(
                    self._run_analysis_background,
                    analysis_id,
                    request
                )
                
                return AnalysisResponse(
                    success=True,
                    analysis_id=analysis_id,
                    status="started",
                    message="Analysis started successfully",
                    created_at=datetime.now()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting analysis: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start analysis: {str(e)}"
                )
        
        @self.app.get("/status/{analysis_id}", response_model=StatusResponse)
        async def get_analysis_status(analysis_id: str):
            """Get analysis status."""
            if analysis_id not in self.active_analyses:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Analysis not found"
                )
            
            analysis = self.active_analyses[analysis_id]
            
            return StatusResponse(
                analysis_id=analysis_id,
                status=analysis["status"],
                progress=analysis["progress"],
                current_phase=analysis.get("current_phase"),
                completed_phases=analysis["completed_phases"],
                errors=analysis["errors"],
                warnings=analysis["warnings"],
                created_at=analysis["created_at"],
                updated_at=analysis["updated_at"]
            )
        
        @self.app.get("/results/{analysis_id}")
        async def get_analysis_results(analysis_id: str):
            """Get analysis results."""
            if analysis_id not in self.analysis_results:
                if analysis_id in self.active_analyses:
                    analysis = self.active_analyses[analysis_id]
                    if analysis["status"] != "completed":
                        raise HTTPException(
                            status_code=status.HTTP_202_ACCEPTED,
                            detail="Analysis not yet completed"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Analysis not found"
                    )
            
            return self.analysis_results[analysis_id]
        
        @self.app.post("/report", response_model=ReportResponse)
        async def generate_report(request: ReportRequest):
            """Generate analysis report."""
            try:
                if request.analysis_id not in self.analysis_results:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Analysis results not found"
                    )
                
                # Get analysis results
                analysis_data = self.analysis_results[request.analysis_id]
                orchestration_result = analysis_data.get("orchestration_result")
                
                if not orchestration_result:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No orchestration result available for report generation"
                    )
                
                # Configure report generation
                report_config = ReportConfig(
                    format=request.format,
                    include_charts=request.include_charts,
                    include_raw_data=request.include_raw_data
                )
                
                reporting_system = ReportingSystem(report_config)
                
                # Generate report
                report = reporting_system.generate_comprehensive_report(
                    orchestration_result,
                    analysis_data,
                    {}
                )
                
                # Generate report ID and URL
                report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                report_url = f"/reports/{report_id}.{request.format}"
                
                # Calculate file size
                report_file = reporting_system.output_dir / f"analysis_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.{request.format}"
                file_size_mb = report_file.stat().st_size / (1024 * 1024) if report_file.exists() else 0.0
                
                return ReportResponse(
                    success=True,
                    report_id=report_id,
                    report_url=report_url,
                    format=request.format,
                    file_size_mb=file_size_mb,
                    created_at=datetime.now()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate report: {str(e)}"
                )
        
        @self.app.get("/reports")
        async def list_reports():
            """List available reports."""
            try:
                report_summary = self.reporting_system.get_report_summary()
                return report_summary
            except Exception as e:
                logger.error(f"Error listing reports: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list reports: {str(e)}"
                )
        
        @self.app.get("/reports/{report_id}")
        async def download_report(report_id: str):
            """Download a specific report."""
            try:
                # Find report file
                report_files = list(self.reporting_system.output_dir.glob(f"*{report_id}*"))
                if not report_files:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Report not found"
                    )
                
                report_file = report_files[0]
                
                return FileResponse(
                    path=str(report_file),
                    filename=report_file.name,
                    media_type='application/octet-stream'
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error downloading report: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to download report: {str(e)}"
                )
        
        @self.app.get("/system/status")
        async def get_system_status():
            """Get system status and performance metrics."""
            try:
                system_status = self.orchestrator.get_analysis_status()
                performance_summary = self.performance_optimizer.get_performance_summary()
                system_resources = self.performance_optimizer.get_system_resources()
                
                return {
                    "system_status": system_status,
                    "performance_summary": performance_summary,
                    "system_resources": system_resources,
                    "active_analyses": len(self.active_analyses),
                    "completed_analyses": len(self.analysis_results),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get system status: {str(e)}"
                )
        
        @self.app.delete("/analyses/{analysis_id}")
        async def cancel_analysis(analysis_id: str):
            """Cancel an active analysis."""
            if analysis_id not in self.active_analyses:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Analysis not found"
                )
            
            analysis = self.active_analyses[analysis_id]
            if analysis["status"] == "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot cancel completed analysis"
                )
            
            # Mark as cancelled
            analysis["status"] = "cancelled"
            analysis["updated_at"] = datetime.now()
            
            return {"message": "Analysis cancelled successfully"}
        
        @self.app.get("/analyses")
        async def list_analyses():
            """List all analyses."""
            analyses = []
            for analysis_id, analysis in self.active_analyses.items():
                analyses.append({
                    "analysis_id": analysis_id,
                    "status": analysis["status"],
                    "progress": analysis["progress"],
                    "created_at": analysis["created_at"],
                    "updated_at": analysis["updated_at"]
                })
            
            return {
                "total_analyses": len(analyses),
                "analyses": analyses
            }
    
    async def _run_analysis_background(self, analysis_id: str, request: AnalysisRequest):
        """Run analysis in background task."""
        try:
            # Update status
            self.active_analyses[analysis_id]["status"] = "running"
            self.active_analyses[analysis_id]["updated_at"] = datetime.now()
            
            # Convert simulation pairs to tuples
            simulation_pairs = [tuple(pair) for pair in request.simulation_pairs]
            
            # Start performance monitoring
            self.performance_optimizer.start_resource_monitoring()
            
            # Run analysis
            with self.performance_optimizer.profile_operation("full_analysis"):
                result = await self.orchestrator.analyze_simulations(
                    simulation_pairs,
                    request.analysis_config
                )
            
            # Stop performance monitoring
            self.performance_optimizer.stop_resource_monitoring()
            
            # Store results
            self.analysis_results[analysis_id] = {
                "orchestration_result": result,
                "performance_summary": self.performance_optimizer.get_performance_summary(),
                "system_resources": self.performance_optimizer.get_system_resources(),
                "completed_at": datetime.now()
            }
            
            # Update status
            self.active_analyses[analysis_id]["status"] = "completed"
            self.active_analyses[analysis_id]["progress"] = 100.0
            self.active_analyses[analysis_id]["updated_at"] = datetime.now()
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            
            # Update status with error
            self.active_analyses[analysis_id]["status"] = "failed"
            self.active_analyses[analysis_id]["errors"].append(str(e))
            self.active_analyses[analysis_id]["updated_at"] = datetime.now()
            
            # Store error results
            self.analysis_results[analysis_id] = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now()
            }
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Standalone API client for testing
class AnalysisAPIClient:
    """Client for interacting with the analysis API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def start_analysis(self, simulation_pairs: List[List[str]], **kwargs) -> Dict[str, Any]:
        """Start a new analysis."""
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        request_data = {
            "simulation_pairs": simulation_pairs,
            **kwargs
        }
        
        async with self.session.post(f"{self.base_url}/analyze", json=request_data) as response:
            return await response.json()
    
    async def get_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status."""
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        async with self.session.get(f"{self.base_url}/status/{analysis_id}") as response:
            return await response.json()
    
    async def get_results(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results."""
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        async with self.session.get(f"{self.base_url}/results/{analysis_id}") as response:
            return await response.json()
    
    async def generate_report(self, analysis_id: str, format: str = "html", **kwargs) -> Dict[str, Any]:
        """Generate analysis report."""
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        request_data = {
            "analysis_id": analysis_id,
            "format": format,
            **kwargs
        }
        
        async with self.session.post(f"{self.base_url}/report", json=request_data) as response:
            return await response.json()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        async with self.session.get(f"{self.base_url}/system/status") as response:
            return await response.json()


# Example usage and testing
if __name__ == "__main__":
    # Example of running the API server
    server = AnalysisAPIServer()
    server.run_server(host="0.0.0.0", port=8000)