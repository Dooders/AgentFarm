from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ..db import get_db
from apps.tree.family_tree import generate_interactive_tree
import shutil

router = APIRouter(
    prefix="/tree",
    tags=["tree"]
)

# Cache settings
CACHE_DURATION = timedelta(minutes=5)
last_generation_time = None
cached_html = None

# Add this function to copy static files
def copy_tree_static_files():
    """Copy tree static files to the server's static directory."""
    source_dir = Path("apps/tree/static")
    dest_dir = Path("static/tree")
    
    if source_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        for file in source_dir.glob("*"):
            if file.is_file():
                shutil.copy2(str(file), str(dest_dir / file.name))

@router.get("/", response_class=HTMLResponse)
async def get_tree(
    max_generations: Optional[int] = None,
    color_by: str = "offspring",
    force_refresh: bool = False,
    db: Session = Depends(get_db)
):
    """Generate and return the family tree visualization."""
    global last_generation_time, cached_html

    current_time = datetime.now()
    needs_refresh = (
        force_refresh
        or last_generation_time is None
        or current_time - last_generation_time > CACHE_DURATION
    )

    if needs_refresh:
        try:
            # Update output path to use the static directory
            output_path = Path("static/tree/family_tree")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            generate_interactive_tree(
                db,
                output_path=str(output_path),
                max_generations=max_generations,
                color_by=color_by,
            )
            html_path = output_path.with_suffix(".html")
            with open(html_path, "r") as f:
                cached_html = f.read()
            last_generation_time = current_time
        except Exception as e:
            return Response(
                content=f"Error generating tree: {str(e)}",
                media_type="text/plain",
                status_code=500
            )
    
    return Response(
        content=cached_html,
        media_type="text/html",
        headers={"Cache-Control": "public, max-age=300"},
    )

@router.get("/refresh")
async def force_refresh():
    """Force a refresh of the visualization."""
    global last_generation_time
    last_generation_time = None
    return {"status": "Cache cleared"}

# Call this function when the router is initialized
copy_tree_static_files() 