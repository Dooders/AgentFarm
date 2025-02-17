import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from apps.tree.family_tree import generate_interactive_tree, get_database_session

app = FastAPI()

# Create static directory relative to this file
static_dir = Path(os.path.dirname(__file__)) / "static"
static_dir.mkdir(exist_ok=True)

# Serve static files from the static directory in the tree app
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Cache settings
CACHE_DURATION = timedelta(minutes=5)
last_generation_time = None
cached_html = None


@app.get("/", response_class=HTMLResponse)
async def get_tree(
    max_generations: Optional[int] = None,
    color_by: str = "offspring",
    force_refresh: bool = False,
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
        session = get_database_session()
        try:
            start_time = time.time()
            # Update output path to use the static directory
            output_path = static_dir / "family_tree"
            generate_interactive_tree(
                session,
                output_path=str(output_path),
                max_generations=max_generations,
                color_by=color_by,
            )
            html_path = output_path.with_suffix(".html")
            with open(html_path, "r") as f:
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
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/refresh")
async def force_refresh():
    """Force a refresh of the visualization."""
    global last_generation_time
    last_generation_time = None
    return {"status": "Cache cleared"}
