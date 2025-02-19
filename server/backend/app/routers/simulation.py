from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models
from ..db import get_db
from datetime import datetime
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationRequest(BaseModel):
    name: str

router = APIRouter(
    prefix="/simulation",
    tags=["simulation"]
)

@router.post("/run")
def run_simulation(request: SimulationRequest, db: Session = Depends(get_db)):
    logger.info(f"Received simulation request with name: {request.name}")
    
    # Add your simulation logic here
    result = 0.0  # Replace with actual simulation result
    
    db_result = models.SimulationResult(
        name=request.name,
        result=result,
        timestamp=datetime.now()
    )
    
    try:
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        logger.info(f"Simulation completed successfully: {db_result}")
        return db_result
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")