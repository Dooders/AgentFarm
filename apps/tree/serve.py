import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Create static directory if it doesn't exist
static_dir = Path(DIRECTORY) / "static"
static_dir.mkdir(exist_ok=True)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving tree visualization at http://localhost:{PORT}")
    httpd.serve_forever() 