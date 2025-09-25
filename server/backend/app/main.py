import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import models
from .db import engine
from .routers import simulation, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Store CORS settings for logging
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.182:3000",
    "*",  # Temporarily allow all origins for testing
]

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(simulation.router)
app.include_router(config.router)


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Simulation API"}


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")
    logger.info("CORS configuration:")
    logger.info(f"Allowed origins: {CORS_ORIGINS}")
