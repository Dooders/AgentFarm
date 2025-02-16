from fastapi import FastAPI, WebSocket, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional
from pathlib import Path
import time
from datetime import datetime, timedelta

from family_tree import generate_interactive_tree, get_database_session

app = FastAPI()

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cache settings
CACHE_DURATION = timedelta(minutes=5)
last_generation_time = None
cached_html = None

@app.get("/", response_class=HTMLResponse)
async def get_tree(
    max_generations: Optional[int] = None, 
    color_by: str = "offspring",
    force_refresh: bool = False
):
    """Generate and return the family tree visualization."""
    global last_generation_time, cached_html

    current_time = datetime.now()
    needs_refresh = (
        force_refresh or 
        last_generation_time is None or 
        current_time - last_generation_time > CACHE_DURATION
    )

    if needs_refresh:
        session = get_database_session()
        try:
            start_time = time.time()
            generate_interactive_tree(
                session,
                output_path="static/family_tree",
                max_generations=max_generations,
                color_by=color_by
            )
            with open("static/family_tree.html", "r") as f:
                cached_html = f.read()
            last_generation_time = current_time
            print(f"Tree generation took {time.time() - start_time:.2f} seconds")
        finally:
            session.close()
    else:
        print("Serving cached visualization")

    return Response(
        content=cached_html,
        media_type="text/html",
        headers={
            "Cache-Control": "public, max-age=300"
        }
    )

@app.get("/refresh")
async def force_refresh():
    """Force a refresh of the visualization."""
    global last_generation_time
    last_generation_time = None
    return {"status": "Cache cleared"} 