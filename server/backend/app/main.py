from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import simulation
from .db import engine
from . import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simulation.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Simulation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
